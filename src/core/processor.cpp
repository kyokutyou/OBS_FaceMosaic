#include "core/processor.hpp"

#include <condition_variable>
#include <deque>
#include <exception>
#include <atomic>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace obs_face_mosaic::core {
namespace {

FrameKey frame_key(const Frame& frame) {
  return FrameKey{frame.generation, frame.id, frame.pts_ns};
}

bool valid_frame_layout(const Frame& frame) {
  if (frame.width <= 0 || frame.height <= 0) {
    return false;
  }
  const auto width = static_cast<std::size_t>(frame.width);
  const auto height = static_cast<std::size_t>(frame.height);
  constexpr std::size_t channels = 4;
  if (width > std::numeric_limits<std::size_t>::max() / height / channels) {
    return false;
  }
  return frame.rgba.size() == width * height * channels;
}

}  // namespace

struct CoreProcessor::Impl {
  struct WorkItem {
    Frame frame;
    ProcessorClock::time_point submitted_at;
    ProcessorClock::time_point enqueued_at;
    std::uint64_t epoch{0};
  };

  struct QueuedResult {
    ProcessResult result;
    ProcessorClock::time_point submitted_at;
    ProcessorClock::time_point completed_at;
  };

  Impl(DetectFunction detect_function, ProcessorConfig processor_config)
      : detect(std::move(detect_function)), config(std::move(processor_config)),
        processing_deadline_ms(config.processing_deadline.count()) {
    if (!detect) {
      throw std::invalid_argument("a face detector callback is required");
    }
    if (config.max_pending_frames == 0 || config.max_completed_results == 0 ||
        config.processing_deadline <= std::chrono::milliseconds::zero() ||
        config.last_processed_hold < std::chrono::milliseconds::zero() ||
        config.recovery_successes == 0) {
      throw std::invalid_argument("invalid processor limits or timing configuration");
    }
  }

  void start() { worker = std::thread([this] { run(); }); }

  void reset_recovery_progress_locked() {
    recovery_count = 0;
    recovery_last_success_id.reset();
  }

  void stop() {
    std::lock_guard<std::mutex> stop_guard(stop_mutex);
    {
      std::lock_guard<std::mutex> guard(mutex);
      if (!stop_requested) {
        stop_requested = true;
        ++epoch;
        pending.clear();
        completed.clear();
        latest_processed.reset();
        reset_recovery_progress_locked();
        state_value = ProcessorState::stopping;
      }
    }
    pending_changed.notify_all();
    if (worker.joinable()) {
      worker.join();
    }
  }

  void run() {
    for (;;) {
      WorkItem item;
      {
        std::unique_lock<std::mutex> lock(mutex);
        pending_changed.wait(lock,
                             [this] { return stop_requested || !pending.empty(); });
        if (stop_requested) {
          return;
        }
        item = std::move(pending.front());
        pending.pop_front();
        in_flight = true;
        in_flight_epoch = item.epoch;
        in_flight_has_result = false;
      }
      if (config.performance_metrics) {
        config.performance_metrics->record(
            diagnostics::Stage::core_queue_wait,
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                ProcessorClock::now() - item.enqueued_at));
      }
      try {
        process(item);
      } catch (...) {
        fail_with_black(item, ProcessReason::processing_failure, false);
      }
      {
        std::lock_guard<std::mutex> guard(mutex);
        if (in_flight && in_flight_epoch == item.epoch) {
          in_flight = false;
          in_flight_has_result = false;
        }
      }
    }
  }

  std::size_t outstanding_locked() const {
    const bool count_in_flight =
        in_flight && in_flight_epoch == epoch && !in_flight_has_result;
    return completed.size() + pending.size() + (count_in_flight ? 1U : 0U);
  }

  bool is_current(const WorkItem& item) {
    std::lock_guard<std::mutex> guard(mutex);
    if (stop_requested || item.epoch != epoch || !generation_set ||
        item.frame.generation != generation || !ready || manual_mask) {
      ++metrics_value.stale_generation_drops;
      return false;
    }
    return true;
  }

  bool deadline_expired(const WorkItem& item,
                        ProcessorClock::time_point now) const {
    const auto deadline = std::chrono::milliseconds(
        processing_deadline_ms.load(std::memory_order_relaxed));
    return now < item.submitted_at ||
           now - item.submitted_at > deadline;
  }

  void process(WorkItem& item) {
    if (deadline_expired(item, ProcessorClock::now())) {
      fail_with_drop(item, ProcessReason::deadline_expired, true);
      return;
    }
    if (!is_current(item)) {
      return;
    }

    std::vector<FaceBox> faces;
    try {
      faces = detect(item.frame);
    } catch (...) {
      fail_with_black(item, ProcessReason::detector_failure, true);
      return;
    }

    if (deadline_expired(item, ProcessorClock::now())) {
      fail_with_drop(item, ProcessReason::deadline_expired, true);
      return;
    }

    if (!is_current(item)) {
      return;
    }

    try {
      diagnostics::ScopedMeasurement mosaic_measurement(
          config.performance_metrics.get(), diagnostics::Stage::mosaic);
      apply_face_mosaic(item.frame, faces, config.mosaic);
    } catch (...) {
      fail_with_black(item, ProcessReason::processing_failure, false);
      return;
    }

    const auto completed_at = ProcessorClock::now();
    if (deadline_expired(item, completed_at)) {
      fail_with_drop(item, ProcessReason::deadline_expired, true);
      return;
    }

    std::shared_ptr<const Frame> processed;
    try {
      processed = std::make_shared<const Frame>(std::move(item.frame));
    } catch (...) {
      fail_with_black(item, ProcessReason::processing_failure, false);
      return;
    }
    bool recovery_frame = false;
    bool publish_processed = false;
    {
      std::lock_guard<std::mutex> guard(mutex);
      if (stop_requested || item.epoch != epoch || !generation_set ||
          processed->generation != generation || !ready || manual_mask) {
        ++metrics_value.stale_generation_drops;
        return;
      }

      ++metrics_value.completed;
      if (state_value == ProcessorState::awaiting_input ||
          state_value == ProcessorState::obscured) {
        state_value = ProcessorState::recovering;
        reset_recovery_progress_locked();
      }

      if (state_value == ProcessorState::recovering) {
        // Recovery requires successful processing of three contiguous input
        // IDs. Queue eviction, rejected input, or any other ID hole starts a
        // new streak so a skipped frame cannot count toward recovery.
        if (recovery_last_success_id &&
            processed->id > *recovery_last_success_id &&
            processed->id - *recovery_last_success_id == 1) {
          ++recovery_count;
        } else {
          recovery_count = 1;
        }
        recovery_last_success_id = processed->id;
        if (recovery_count >= config.recovery_successes) {
          state_value = ProcessorState::running;
          publish_processed = true;
        } else {
          recovery_frame = true;
        }
      } else if (state_value == ProcessorState::running) {
        publish_processed = true;
      } else {
        // Readiness and manual masking change the epoch, so this branch is only
        // a defensive guard against any future state transition.
        ++metrics_value.stale_generation_drops;
        return;
      }
    }

    if (publish_processed) {
      enqueue_result(ProcessResult{OutputKind::processed, frame_key(*processed),
                                   std::move(processed), ProcessReason::none},
                     item.submitted_at, completed_at, item.epoch);
    } else if (recovery_frame) {
      enqueue_black_for(item, ProcessReason::recovering, completed_at);
    }
  }

  void fail_with_drop(const WorkItem& item, ProcessReason reason,
                      bool deadline) {
    const auto completed_at = ProcessorClock::now();
    if (deadline && config.performance_metrics) {
      config.performance_metrics->record(
          diagnostics::Stage::core_processing_deadline,
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              completed_at - item.submitted_at),
          true);
    }
    {
      std::lock_guard<std::mutex> guard(mutex);
      if (stop_requested || item.epoch != epoch || !generation_set ||
          item.frame.generation != generation) {
        ++metrics_value.stale_generation_drops;
        return;
      }
      if (deadline) {
        ++metrics_value.deadline_drops;
      }
      state_value = ProcessorState::obscured;
      reset_recovery_progress_locked();
      latest_processed.reset();
    }
    enqueue_result(ProcessResult{OutputKind::drop, frame_key(item.frame), nullptr,
                                 reason},
                   item.submitted_at, completed_at, item.epoch);
  }

  void obscure_processing_failure_locked() {
    state_value = ProcessorState::obscured;
    reset_recovery_progress_locked();
    latest_processed.reset();
    completed.clear();
    pending.clear();
    // Publish before allocating a black frame or waiting for another input.
    if (config.on_processing_failure) {
      try { config.on_processing_failure(); } catch (...) {}
    }
  }

  void fail_with_black(const WorkItem& item, ProcessReason reason,
                       bool detector_failure) {
    {
      std::lock_guard<std::mutex> guard(mutex);
      if (stop_requested || item.epoch != epoch || !generation_set ||
          item.frame.generation != generation) {
        ++metrics_value.stale_generation_drops;
        return;
      }
      if (detector_failure) ++metrics_value.detector_failures;
      obscure_processing_failure_locked();
    }
    std::shared_ptr<const Frame> black;
    try {
      black = std::make_shared<const Frame>(make_black_frame(
          item.frame.width, item.frame.height, item.frame.generation,
          item.frame.id, item.frame.pts_ns));
    } catch (...) {
      // A missing black allocation is represented as drop. The adapter must
      // keep its opaque draw fallback active rather than reusing input pixels.
      reason = ProcessReason::processing_failure;
    }

    const auto completed_at = ProcessorClock::now();
    {
      std::lock_guard<std::mutex> guard(mutex);
      if (stop_requested || item.epoch != epoch || !generation_set ||
          item.frame.generation != generation) {
        ++metrics_value.stale_generation_drops;
        return;
      }
    }
    enqueue_result(ProcessResult{black ? OutputKind::black : OutputKind::drop,
                                 frame_key(item.frame), std::move(black), reason},
                   item.submitted_at, completed_at, item.epoch);
  }

  void enqueue_black_for(const WorkItem& item, ProcessReason reason,
                         ProcessorClock::time_point completed_at) {
    std::shared_ptr<const Frame> black;
    try {
      black = std::make_shared<const Frame>(make_black_frame(
          item.frame.width, item.frame.height, item.frame.generation,
          item.frame.id, item.frame.pts_ns));
    } catch (...) {
      reason = ProcessReason::processing_failure;
    }

    {
      std::lock_guard<std::mutex> guard(mutex);
      if (stop_requested || item.epoch != epoch || !generation_set ||
          item.frame.generation != generation || manual_mask || !ready) {
        ++metrics_value.stale_generation_drops;
        return;
      }
      if (!black) {
        obscure_processing_failure_locked();
      }
    }
    enqueue_result(ProcessResult{black ? OutputKind::black : OutputKind::drop,
                                 frame_key(item.frame), std::move(black), reason},
                   item.submitted_at, completed_at, item.epoch);
  }

  void enqueue_result(ProcessResult result,
                      ProcessorClock::time_point submitted_at,
                      ProcessorClock::time_point completed_at,
                      std::uint64_t result_epoch) {
    const bool processing_succeeded =
        result.reason == ProcessReason::none ||
        result.reason == ProcessReason::recovering;
    bool enqueued = false;
    {
      std::lock_guard<std::mutex> guard(mutex);
      if (stop_requested || result_epoch != epoch ||
          result.key.generation != generation) {
        ++metrics_value.stale_generation_drops;
        return;
      }
      if (completed.size() >= config.max_completed_results) {
        completed.pop_front();
        ++metrics_value.output_evictions;
      }
      try {
        completed.push_back(
            QueuedResult{std::move(result), submitted_at, completed_at});
        enqueued = true;
        if (in_flight && in_flight_epoch == result_epoch) {
          in_flight_has_result = true;
        }
      } catch (...) {
        obscure_processing_failure_locked();
        ++metrics_value.output_evictions;
      }
    }
    if (enqueued && processing_succeeded && config.performance_metrics) {
      config.performance_metrics->record(
          diagnostics::Stage::core_processing_success,
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              completed_at - submitted_at));
    }
  }

  DetectFunction detect;
  ProcessorConfig config;
  mutable std::mutex mutex;
  std::mutex stop_mutex;
  std::condition_variable pending_changed;
  std::thread worker;
  std::deque<WorkItem> pending;
  std::deque<QueuedResult> completed;
  std::shared_ptr<const Frame> latest_processed;
  ProcessorClock::time_point latest_completed_at{};
  ProcessorMetrics metrics_value{};
  ProcessorState state_value{ProcessorState::initializing};
  std::uint64_t generation{0};
  std::uint64_t epoch{1};
  std::optional<std::uint64_t> last_submitted_id;
  std::size_t recovery_count{0};
  std::optional<std::uint64_t> recovery_last_success_id;
  std::uint64_t in_flight_epoch{0};
  bool generation_set{false};
  bool ready{false};
  bool manual_mask{false};
  bool in_flight{false};
  bool in_flight_has_result{false};
  bool stop_requested{false};
  // The worker checks deadlines outside mutex, while poll checks under it.
  // Keep the live value atomic so an L-only update never races either path.
  std::atomic<std::chrono::milliseconds::rep> processing_deadline_ms;
};

CoreProcessor::CoreProcessor(DetectFunction detect, ProcessorConfig config)
    : impl_(std::make_unique<Impl>(std::move(detect), std::move(config))) {
  impl_->start();
}

CoreProcessor::~CoreProcessor() { stop(); }

bool CoreProcessor::reset_generation(std::uint64_t generation) {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  if (impl_->stop_requested ||
      (impl_->generation_set && generation <= impl_->generation)) {
    return false;
  }
  impl_->generation = generation;
  impl_->generation_set = true;
  ++impl_->epoch;
  impl_->pending.clear();
  impl_->completed.clear();
  impl_->latest_processed.reset();
  impl_->last_submitted_id.reset();
  impl_->reset_recovery_progress_locked();
  impl_->state_value = impl_->manual_mask
                           ? ProcessorState::manual_mask
                           : (impl_->ready ? ProcessorState::awaiting_input
                                           : ProcessorState::initializing);
  return true;
}

bool CoreProcessor::reset_generation(
    std::uint64_t generation,
    std::chrono::milliseconds processing_deadline) {
  if (processing_deadline <= std::chrono::milliseconds::zero()) {
    return false;
  }
  std::lock_guard<std::mutex> guard(impl_->mutex);
  if (impl_->stop_requested ||
      (impl_->generation_set && generation <= impl_->generation)) {
    return false;
  }
  impl_->processing_deadline_ms.store(processing_deadline.count(),
                                      std::memory_order_relaxed);
  impl_->generation = generation;
  impl_->generation_set = true;
  ++impl_->epoch;
  impl_->pending.clear();
  impl_->completed.clear();
  impl_->latest_processed.reset();
  impl_->last_submitted_id.reset();
  impl_->reset_recovery_progress_locked();
  impl_->state_value = impl_->manual_mask
                           ? ProcessorState::manual_mask
                           : (impl_->ready ? ProcessorState::awaiting_input
                                           : ProcessorState::initializing);
  return true;
}

void CoreProcessor::set_processing_deadline(
    std::chrono::milliseconds processing_deadline) {
  if (processing_deadline <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument("processing deadline must be positive");
  }
  impl_->processing_deadline_ms.store(processing_deadline.count(),
                                      std::memory_order_relaxed);
}

void CoreProcessor::set_ready(bool ready) {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  if (impl_->stop_requested || impl_->ready == ready) {
    return;
  }
  impl_->ready = ready;
  ++impl_->epoch;
  impl_->pending.clear();
  impl_->completed.clear();
  impl_->latest_processed.reset();
  impl_->reset_recovery_progress_locked();
  impl_->state_value = impl_->manual_mask
                           ? ProcessorState::manual_mask
                           : (ready ? (impl_->generation_set
                                           ? ProcessorState::awaiting_input
                                           : ProcessorState::initializing)
                                    : ProcessorState::initializing);
}

void CoreProcessor::set_manual_mask(bool enabled) {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  if (impl_->stop_requested || impl_->manual_mask == enabled) {
    return;
  }
  impl_->manual_mask = enabled;
  ++impl_->epoch;
  impl_->pending.clear();
  impl_->completed.clear();
  impl_->latest_processed.reset();
  impl_->reset_recovery_progress_locked();
  impl_->state_value = enabled
                           ? ProcessorState::manual_mask
                           : (impl_->ready && impl_->generation_set
                                  ? ProcessorState::awaiting_input
                                  : ProcessorState::initializing);
}

SubmitResult CoreProcessor::submit(Frame frame,
                                   ProcessorClock::time_point submitted_at) {
  const FrameKey key = frame_key(frame);
  SubmitResult result;
  result.submitted = key;

  if (!valid_frame_layout(frame)) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    ++impl_->metrics_value.rejected;
    result.reason = ProcessReason::invalid_frame;
    return result;
  }

  {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (impl_->stop_requested) {
      ++impl_->metrics_value.rejected;
      result.reason = ProcessReason::stopping;
      return result;
    }
    if (impl_->manual_mask) {
      ++impl_->metrics_value.rejected;
      result.reason = ProcessReason::manual_mask;
      return result;
    }
    if (!impl_->ready) {
      ++impl_->metrics_value.rejected;
      result.reason = ProcessReason::not_ready;
      return result;
    }
    if (!impl_->generation_set || frame.generation != impl_->generation) {
      ++impl_->metrics_value.rejected;
      result.reason = ProcessReason::stale_generation;
      return result;
    }
    if (impl_->last_submitted_id && frame.id <= *impl_->last_submitted_id) {
      ++impl_->metrics_value.rejected;
      result.reason = ProcessReason::non_monotonic_id;
      return result;
    }

    if (impl_->pending.size() >= impl_->config.max_pending_frames) {
      result.evicted = frame_key(impl_->pending.front().frame);
      impl_->pending.pop_front();
      ++impl_->metrics_value.queue_evictions;
      if (impl_->state_value == ProcessorState::recovering) {
        impl_->reset_recovery_progress_locked();
      }
    }

    if (impl_->outstanding_locked() >=
        impl_->config.max_completed_results) {
      ++impl_->metrics_value.rejected;
      ++impl_->metrics_value.output_rejections;
      result.reason = ProcessReason::output_queue_overflow;
      return result;
    }

    try {
      const auto enqueued_at = ProcessorClock::now();
      impl_->pending.push_back(Impl::WorkItem{std::move(frame), submitted_at,
                                             enqueued_at, impl_->epoch});
    } catch (...) {
      ++impl_->metrics_value.rejected;
      impl_->obscure_processing_failure_locked();
      result.reason = ProcessReason::processing_failure;
      return result;
    }
    if (impl_->state_value == ProcessorState::awaiting_input ||
        impl_->state_value == ProcessorState::obscured) {
      impl_->state_value = ProcessorState::recovering;
      impl_->reset_recovery_progress_locked();
    }
    impl_->last_submitted_id = key.id;
    ++impl_->metrics_value.submitted;
    result.accepted = true;
    result.reason = result.evicted ? ProcessReason::queue_overflow
                                   : ProcessReason::none;
  }
  impl_->pending_changed.notify_one();
  return result;
}

std::optional<ProcessResult> CoreProcessor::poll(ProcessorClock::time_point now) {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  if (impl_->completed.empty()) {
    return std::nullopt;
  }
  Impl::QueuedResult queued = std::move(impl_->completed.front());
  impl_->completed.pop_front();
  if (queued.result.kind == OutputKind::processed &&
      (now < queued.submitted_at ||
       now - queued.submitted_at > std::chrono::milliseconds(
           impl_->processing_deadline_ms.load(std::memory_order_relaxed)))) {
    queued.result.kind = OutputKind::drop;
    queued.result.frame.reset();
    queued.result.reason = ProcessReason::deadline_expired;
    ++impl_->metrics_value.deadline_drops;
    if (impl_->state_value == ProcessorState::running &&
        (!impl_->latest_processed ||
         queued.result.key.id >= impl_->latest_processed->id)) {
      impl_->state_value = ProcessorState::obscured;
      impl_->latest_processed.reset();
      impl_->reset_recovery_progress_locked();
    }
  } else if (queued.result.kind == OutputKind::processed &&
             impl_->generation_set &&
             queued.result.key.generation == impl_->generation &&
             impl_->state_value == ProcessorState::running) {
    impl_->latest_processed = queued.result.frame;
    impl_->latest_completed_at = queued.completed_at;
  }
  return std::move(queued.result);
}

ProcessResult CoreProcessor::snapshot(int width, int height, FrameKey output_key,
                                      ProcessorClock::time_point now) {
  ProcessResult result;
  result.kind = OutputKind::black;
  result.key = output_key;

  {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (width <= 0 || height <= 0) {
      result.reason = ProcessReason::no_frame_dimensions;
      return result;
    }

    if (impl_->generation_set && output_key.generation != impl_->generation) {
      result.reason = ProcessReason::stale_generation;
    } else if (impl_->manual_mask) {
      result.reason = ProcessReason::manual_mask;
    } else if (!impl_->ready) {
      result.reason = ProcessReason::not_ready;
    } else if (impl_->state_value == ProcessorState::running &&
               impl_->latest_processed &&
               impl_->latest_processed->generation == output_key.generation &&
               now >= impl_->latest_completed_at &&
               now - impl_->latest_completed_at <=
                   impl_->config.last_processed_hold) {
      result.kind = OutputKind::processed;
      result.key = frame_key(*impl_->latest_processed);
      result.frame = impl_->latest_processed;
      result.reason = ProcessReason::none;
      return result;
    } else {
      result.reason = impl_->state_value == ProcessorState::running
                          ? (impl_->latest_processed
                                 ? ProcessReason::stale_output
                                 : ProcessReason::awaiting_result)
                          : (impl_->state_value == ProcessorState::recovering
                                 ? ProcessReason::recovering
                                 : (impl_->state_value == ProcessorState::obscured
                                        ? ProcessReason::stale_output
                                        : ProcessReason::not_ready));
      if (impl_->state_value == ProcessorState::running) {
        impl_->state_value = ProcessorState::obscured;
        impl_->latest_processed.reset();
        impl_->reset_recovery_progress_locked();
      }
    }
  }

  try {
    result.frame = std::make_shared<const Frame>(make_black_frame(
        width, height, output_key.generation, output_key.id, output_key.pts_ns));
  } catch (...) {
    result.frame.reset();
    result.reason = ProcessReason::no_frame_dimensions;
  }
  return result;
}

ProcessorState CoreProcessor::state() const {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  return impl_->state_value;
}

ProcessorMetrics CoreProcessor::metrics() const {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  return impl_->metrics_value;
}

RecoveryProgressSnapshot CoreProcessor::recovery_progress() const {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  return RecoveryProgressSnapshot{
      impl_->generation_set, impl_->generation, impl_->state_value,
      impl_->config.recovery_successes, impl_->recovery_count,
      impl_->recovery_last_success_id};
}

void CoreProcessor::stop() { impl_->stop(); }

}  // namespace obs_face_mosaic::core
