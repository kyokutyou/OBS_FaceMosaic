#include "core/processor.hpp"
#include "inference/detector.hpp"
#include "obs/replay_policy.hpp"
#include "obs/buffer_limits.hpp"
#include "obs/generation_transition.hpp"
#include "obs/frame_converter.hpp"
#include "obs/settings.hpp"

#include <obs-module.h>
#include <obs.h>
#include <media-io/video-io.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace obs_face_mosaic::obs_plugin {
namespace {

using Clock = std::chrono::steady_clock;
using obs_face_mosaic::core::Frame;
using obs_face_mosaic::core::FrameKey;
using obs_face_mosaic::core::OutputKind;
using obs_face_mosaic::core::ProcessResult;
using obs_face_mosaic::core::ProcessorConfig;
using obs_face_mosaic::core::ProcessorState;
using obs_face_mosaic::core::CoreProcessor;
using DiagnosticOutput = obs_face_mosaic::diagnostics::Output;
using obs_face_mosaic::diagnostics::PerformanceMetrics;
using obs_face_mosaic::diagnostics::ScopedMeasurement;
using obs_face_mosaic::diagnostics::Stage;
using obs_face_mosaic::inference::Detector;
using obs_face_mosaic::inference::DetectorConfig;
using obs_face_mosaic::inference::Device;

constexpr char kFilterId[] = "obs_face_mosaic_filter";

struct Runtime {
  std::shared_ptr<Detector> detector;
  std::unique_ptr<CoreProcessor> processor;
  Device device{Device::directml};
  std::string adapter{"不明"};

  ~Runtime() {
    if (processor) {
      // CoreProcessor joins its inference owner. This destructor is dispatched
      // to RuntimeReaper so it never runs in an OBS video or UI callback.
      processor->stop();
      processor.reset();
    }
    detector.reset();
  }
};

class RuntimeReaper final {
 public:
  static RuntimeReaper& instance() {
    static RuntimeReaper reaper;
    return reaper;
  }

  void start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!worker_.joinable()) {
      stopping_ = false;
      worker_ = std::thread([this] { run(); });
    }
  }

  void post(std::function<void()> task) {
    start();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.push_back(std::move(task));
    }
    changed_.notify_one();
  }

  void stop_and_drain() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    changed_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

 private:
  RuntimeReaper() = default;

  void run() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
        if (tasks_.empty() && stopping_) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      try {
        task();
      } catch (...) {
        blog(LOG_ERROR, "OBS_FaceMosaic: background cleanup task failed");
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<std::function<void()>> tasks_;
  std::thread worker_;
  bool stopping_{false};
};

std::shared_ptr<Runtime> make_reaped_runtime(Runtime* runtime) {
  return std::shared_ptr<Runtime>(runtime, [](Runtime* value) {
    RuntimeReaper::instance().post([value] { delete value; });
  });
}

struct Control final {
#if defined(OBSFM_ENABLE_TEST_HOOKS)
  std::shared_ptr<std::atomic<bool>> test_failure{std::make_shared<std::atomic<bool>>(false)};
#endif
  std::mutex mutex;
  std::condition_variable changed;
  Settings settings;
  std::shared_ptr<Runtime> runtime;
  std::string status{"初期化中: モデルフォルダーを指定してください"};
  std::uint64_t settings_revision{1};
  std::uint64_t generation{1};
  std::atomic<std::uint64_t> detector_load_count{0};
  bool stopping{false};
};

struct PendingFrame {
  obs_source_frame* source_frame{nullptr};
  FrameKey key{};
  Clock::time_point submitted_at{};
  std::size_t estimated_source_bytes{0};
  std::optional<ProcessResult> result;
  bool force_black{false};
  BlackCause black_cause{BlackCause::none};
};

struct EmitResult {
  obs_source_frame* output{nullptr};
  obs_source_frame* release_after_gate{nullptr};
  bool renderable{false};
  bool newly_processed{false};
  std::uint64_t generation{0};
  std::uint64_t returned_at_ns{0};
  std::uint64_t output_id{0};
};

struct FilterData;
void filter_enabled_changed(void* data, calldata_t* params);
void source_media_stopped(void* data, calldata_t* params);
void source_media_ended(void* data, calldata_t* params);
void source_media_restart(void* data, calldata_t* params);
void source_media_started(void* data, calldata_t* params);
void source_shown(void* data, calldata_t* params);
void source_hidden(void* data, calldata_t* params);

struct FilterData {
  explicit FilterData(obs_source_t* source)
      : context(source), control(std::make_shared<Control>()),
        performance_metrics(std::make_shared<PerformanceMetrics>()) {
    char* verification_ids = nullptr;
    std::size_t verification_ids_size = 0;
    _dupenv_s(&verification_ids, &verification_ids_size,
              "OBS_FACE_MOSAIC_VERIFY_FRAME_IDS");
    verification_frame_id_logging_enabled =
        verification_ids && std::strcmp(verification_ids, "1") == 0;
    std::free(verification_ids);
    loader = std::thread([shared = control, metrics = performance_metrics,
                          transitions = lifecycle_epoch] {
      load_loop(shared, metrics, transitions);
    });
    enabled.store(obs_source_enabled(context), std::memory_order_relaxed);
    enable_signal_handler = obs_source_get_signal_handler(context);
    signal_handler_connect_ref(enable_signal_handler, "enable",
                               filter_enabled_changed, this);
  }

  ~FilterData() {
    // Disconnect before destroying any callback state. The signal callback
    // only touches atomics, so this cannot wait for video or inference locks.
    if (enable_signal_handler) {
      signal_handler_disconnect(enable_signal_handler, "enable",
                                filter_enabled_changed, this);
      enable_signal_handler = nullptr;
    }
    {
      std::lock_guard<std::mutex> lock(control->mutex);
      control->stopping = true;
    }
    control->changed.notify_all();
    if (loader.joinable()) {
      auto thread = std::make_shared<std::thread>(std::move(loader));
      RuntimeReaper::instance().post([thread] {
        if (thread->joinable()) {
          thread->join();
        }
      });
    }
    obs_source_t* retained_parent = nullptr;
    obs_weak_source_t* weak_parent_to_release = nullptr;
    std::deque<PendingFrame> frames_to_release;
    bool pending_without_parent = false;
    {
      std::lock_guard<std::mutex> lock(video_mutex);
      disconnect_lifecycle_signals_locked();
      retained_parent = weak_parent ? obs_weak_source_get_source(weak_parent) : nullptr;
      if (retained_parent) {
        frames_to_release = detach_pending();
      } else if (!pending.empty()) {
        pending_without_parent = true;
      }
      weak_parent_to_release = std::exchange(weak_parent, nullptr);
    }
    if (pending_without_parent) {
      blog(LOG_ERROR, "OBS_FaceMosaic: retained frames remained after parent removal");
    }
    if (retained_parent) {
      for (const auto& frame : frames_to_release) {
        if (frame.source_frame) {
          obs_source_release_frame(retained_parent, frame.source_frame);
        }
      }
      obs_source_release(retained_parent);
    }
    if (weak_parent_to_release) {
      obs_weak_source_release(weak_parent_to_release);
    }
  }

  static void load_loop(
      const std::shared_ptr<Control>& shared,
      const std::shared_ptr<PerformanceMetrics>& performance_metrics,
      const std::shared_ptr<LifecycleTransitionEpoch>& transitions) {
    std::uint64_t completed_revision = 0;
    for (;;) {
      Settings settings;
      std::uint64_t revision = 0;
      {
        std::unique_lock<std::mutex> lock(shared->mutex);
        shared->changed.wait(lock, [&] {
          return shared->stopping || shared->settings_revision != completed_revision;
        });
        if (shared->stopping) {
          return;
        }
        settings = shared->settings;
        revision = shared->settings_revision;
        completed_revision = revision;
      }

      if (settings.model_directory.empty()) {
        std::lock_guard<std::mutex> lock(shared->mutex);
        if (shared->settings_revision == revision) {
          shared->status = "待機中: モデルフォルダーを指定してください";
          shared->runtime.reset();
        }
        continue;
      }

      const std::string selected_filename = model_filename(settings);
      std::u8string model_directory_utf8;
      model_directory_utf8.reserve(settings.model_directory.size());
      for (const unsigned char character : settings.model_directory) {
        model_directory_utf8.push_back(static_cast<char8_t>(character));
      }
      const auto model_path = std::filesystem::path(model_directory_utf8) /
                              std::filesystem::path(selected_filename);
      try {
        DetectorConfig detector_config;
        detector_config.model_path = model_path;
        detector_config.device = settings.use_cpu ? Device::cpu : Device::directml;
        detector_config.gpu_adapter = settings.gpu_adapter;
        detector_config.score_threshold = settings.score_threshold;
        detector_config.nms_iou_threshold = settings.nms_threshold;
        detector_config.performance_metrics = performance_metrics;
        auto detector = std::make_shared<Detector>(detector_config);
        blog(LOG_INFO, "OBS_FaceMosaic: verified model=%s sha256=%s",
             selected_filename.c_str(), detector->model_sha256().c_str());
        shared->detector_load_count.fetch_add(1, std::memory_order_relaxed);

        ProcessorConfig processor_config;
        processor_config.processing_deadline =
            std::chrono::milliseconds(settings.latency_ms);
        processor_config.mosaic.expansion_ratio = settings.expansion_ratio;
        processor_config.mosaic.long_edge_cells = settings.mosaic_cells;
        processor_config.performance_metrics = performance_metrics;
        // Shared atomic-only gate outlives FilterData if asynchronous cleanup
        // is still joining the worker. No OBS/UI/video lock is acquired here.
        processor_config.on_processing_failure = [transitions]() noexcept {
          transitions->request();
        };
        auto* created = new Runtime();
        created->detector = detector;
        created->device = detector->device();
        created->adapter = detector->adapter_name();
        created->processor = std::make_unique<CoreProcessor>(
            [detector
#if defined(OBSFM_ENABLE_TEST_HOOKS)
             , test_failure = shared->test_failure
#endif
            ](const Frame& frame) {
#if defined(OBSFM_ENABLE_TEST_HOOKS)
              if (test_failure->load(std::memory_order_acquire)) {
                blog(LOG_WARNING, "OBS_FaceMosaic verification: injected inference exception");
                throw std::runtime_error("isolated inference exception test");
              }
#endif
              return detector->detect(frame);
            },
            processor_config);
        created->processor->set_ready(true);
        auto runtime = make_reaped_runtime(created);

        std::lock_guard<std::mutex> lock(shared->mutex);
        if (shared->stopping || shared->settings_revision != revision) {
          runtime.reset();
        } else {
          // L can change while the initial model is loading without creating a
          // new loader revision. Apply the latest values before publishing the
          // runtime, while no video callback can yet observe it.
          created->processor->set_processing_deadline(
              std::chrono::milliseconds(shared->settings.latency_ms));
          created->processor->set_manual_mask(shared->settings.manual_mask);
          shared->runtime = std::move(runtime);
          shared->status = shared->settings.manual_mask
                               ? "準備完了: 手動全画面遮蔽中"
                               : "準備完了: 入力フレームを待っています";
        }
      } catch (const obs_face_mosaic::inference::DeviceSelectionError&) {
        std::lock_guard<std::mutex> lock(shared->mutex);
        if (!shared->stopping && shared->settings_revision == revision) {
          shared->runtime.reset();
          shared->status = "選択したGPUを利用できません。GPUを選び直して再試行してください";
          blog(LOG_ERROR, "OBS_FaceMosaic: selected GPU unavailable; video remains fail-closed");
        }
      } catch (const obs_face_mosaic::inference::ModelIntegrityError& error) {
        std::lock_guard<std::mutex> lock(shared->mutex);
        if (!shared->stopping && shared->settings_revision == revision) {
          shared->runtime.reset();
          shared->status = "モデルのSHA-256照合に失敗しました。指定したn/mモデルを確認して再試行してください";
          // This exception contains only fixed text and hashes, never the path.
          blog(LOG_ERROR, "OBS_FaceMosaic: %s; video remains fail-closed", error.what());
        }
      } catch (const std::exception&) {
        std::lock_guard<std::mutex> lock(shared->mutex);
        if (!shared->stopping && shared->settings_revision == revision) {
          shared->runtime.reset();
          shared->status = "モデル読込または推論初期化に失敗しました。設定を確認して再試行してください";
          blog(LOG_ERROR, "OBS_FaceMosaic: ONNX Runtime initialization failed; video remains fail-closed");
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(shared->mutex);
        if (!shared->stopping && shared->settings_revision == revision) {
          shared->runtime.reset();
          shared->status = "モデル初期化に失敗しました。設定を確認して再試行してください";
          blog(LOG_ERROR, "OBS_FaceMosaic: unknown initialization failure; video remains fail-closed");
        }
      }
    }
  }

  std::shared_ptr<Runtime> runtime() {
    std::lock_guard<std::mutex> lock(control->mutex);
    return control->runtime;
  }

  std::deque<PendingFrame> detach_pending() noexcept {
    std::deque<PendingFrame> detached;
    detached.swap(pending);
    pending_count.store(0);
    pending_source_bytes = 0;
    latest_pending_bytes.store(0);
    return detached;
  }

  void disconnect_lifecycle_signals_locked() noexcept;

  std::deque<PendingFrame> detach_pending_for_transition(
      GenerationTransitionAction action) {
    auto detached = detach_pending_for_generation_transition(pending, action);
    pending_count.store(pending.size());
    if (action == GenerationTransitionAction::discard_pending_and_black_current) {
      pending_source_bytes = 0;
    }
    latest_pending_bytes.store(pending_source_bytes);
    return detached;
  }

  obs_source_t* context{nullptr};
  // The parent owns this filter. Keep only a weak handle here: a strong parent
  // reference would create a parent -> filter -> parent reference cycle and
  // prevent OBS from reaching filter_remove during shutdown.
  obs_weak_source_t* weak_parent{nullptr};
  signal_handler_t* parent_signal_handler{nullptr};
  signal_handler_t* enable_signal_handler{nullptr};
  std::atomic<bool> enabled{true};
  std::shared_ptr<LifecycleTransitionEpoch> lifecycle_epoch{
      std::make_shared<LifecycleTransitionEpoch>()};
  std::shared_ptr<Control> control;
  std::shared_ptr<PerformanceMetrics> performance_metrics;
  std::thread loader;
  std::deque<PendingFrame> pending;
  FrameConverter converter;
  std::uint64_t next_id{1};
  std::uint64_t last_pts{0};
  std::uint32_t last_width{0};
  std::uint32_t last_height{0};
  enum video_format last_format{VIDEO_FORMAT_NONE};
  FrameIntervalEstimator input_interval_estimator;
  // video_mutex guarded; this value is the estimate from the previous input,
  // before observing the current PTS. Discontinuity detection must not learn
  // its threshold from the jump it is trying to detect.
  std::uint64_t stable_input_interval_ns{kDefaultFrameIntervalNs};
  std::uint64_t active_generation{0};
  // Guarded by video_mutex. Advance the generation once per detected input
  // stall, then re-arm when the next input frame arrives.
  bool input_stall_generation_advanced{false};
  bool verification_frame_id_logging_enabled{false};
  std::size_t verification_frame_id_log_count{0};
  std::uint64_t recovery_progress_generation_logged{0};
  std::size_t recovery_progress_success_count_logged{0};
  std::uint64_t first_processed_output_generation_logged{0};
  std::weak_ptr<Runtime> active_runtime;
  std::atomic<std::uint64_t> last_input_time_ns{0};
  // OBS reports an inactive async source as 0x0. Keep the last valid input
  // size so the filter can still cover its scene item after input stops.
  std::atomic<std::uint32_t> render_width{0};
  std::atomic<std::uint32_t> render_height{0};
  // video_render runs on OBS's graphics thread. These atomics let it fail
  // closed without waiting for Control, video_mutex, or inference work.
  std::atomic<std::uint64_t> render_generation{1};
  std::atomic<std::uint64_t> renderable_generation{0};
  std::atomic<std::uint64_t> last_processed_output_time_ns{0};
  // Low-frequency A03 diagnostics: log only after input has been idle for at
  // least 500 ms, then once for stable black and once on recovery. Ordinary
  // startup, load, and processing-freshness transitions do not log here.
  std::atomic<bool> render_black_latched{false};
  std::atomic<bool> render_black_stop_logged{false};
  std::atomic<bool> render_black_stable_logged{false};
  std::atomic<std::uint64_t> render_black_started_at_ns{0};
  std::atomic<std::uint64_t> render_black_stop_started_at_ns{0};
  std::atomic<std::uint64_t> render_black_stop_output_count{0};
  std::atomic<std::uint64_t> last_properties_time_ns{0};
  std::atomic<std::uint64_t> pending_count{0};
  std::atomic<std::uint64_t> output_count{0};
  std::atomic<std::uint64_t> black_count{0};
  std::atomic<std::uint64_t> replay_count{0};
  std::atomic<std::uint64_t> drop_count{0};
  std::atomic<std::uint64_t> output_fps{0};
  std::atomic<std::int64_t> latest_latency_ms{0};
  std::atomic<std::int64_t> last_diagnostic_log_ms{0};
  std::atomic<std::uint64_t> latest_pending_limit{kMinimumOutstandingFrames};
  std::atomic<std::uint64_t> latest_pending_bytes{0};
  std::atomic<std::uint64_t> latest_input_interval_us{0};
  std::shared_ptr<const Frame> last_processed_frame;
  Clock::time_point last_processed_at{};
  std::size_t pending_source_bytes{0};
  std::mutex video_mutex;
  // Serializes settings/generation commits with the final OBS-frame write.
  // Neither side holds Control while doing the image conversion itself.
  std::mutex output_commit_mutex;
  std::deque<Clock::time_point> output_times;
};

void FilterData::disconnect_lifecycle_signals_locked() noexcept {
  auto* handler = std::exchange(parent_signal_handler, nullptr);
  if (!handler) {
    return;
  }

  // connect_ref keeps the handler alive. Disconnect synchronizes with an
  // in-flight OBS signal before FilterData can be destroyed; callbacks only
  // touch atomics and never acquire video_mutex.
  signal_handler_disconnect(handler, "media_stopped", source_media_stopped, this);
  signal_handler_disconnect(handler, "media_ended", source_media_ended, this);
  signal_handler_disconnect(handler, "media_restart", source_media_restart, this);
  signal_handler_disconnect(handler, "media_started", source_media_started, this);
  signal_handler_disconnect(handler, "show", source_shown, this);
  signal_handler_disconnect(handler, "hide", source_hidden, this);
}

void request_lifecycle_transition(FilterData* filter, const char* signal_name) noexcept {
  if (!filter) {
    return;
  }
  const auto epoch = filter->lifecycle_epoch->request();
  // Close the graphics-thread permit immediately. process_video will discard
  // old pending work, reset CoreProcessor, and acknowledge this epoch.
  filter->renderable_generation.store(0, std::memory_order_release);
  filter->last_processed_output_time_ns.store(0, std::memory_order_release);
  blog(LOG_INFO,
       "OBS_FaceMosaic media lifecycle signal: event=%s epoch=%llu",
       signal_name, static_cast<unsigned long long>(epoch));
}

void source_media_stopped(void* data, calldata_t*) {
  request_lifecycle_transition(static_cast<FilterData*>(data), "stopped");
}

void filter_enabled_changed(void* data, calldata_t* params) {
  auto* filter = static_cast<FilterData*>(data);
  if (!filter) {
    return;
  }
  const bool enabled = calldata_bool(params, "enabled");
  if (filter->enabled.exchange(enabled, std::memory_order_acq_rel) != enabled) {
    // OBS bypasses filter_video while disabled and can replace the parent's
    // async texture with raw pixels. Even a sub-frame toggle must invalidate
    // the old render permit before any new input arrives. Reuse the lifecycle
    // epoch so in-flight results cannot reopen it and the next callback resets
    // the queues/processor before beginning three-frame recovery.
    request_lifecycle_transition(filter, enabled ? "filter_enabled"
                                               : "filter_disabled");
  }
}

void source_shown(void* data, calldata_t*) {
  request_lifecycle_transition(static_cast<FilterData*>(data), "show");
}

void source_hidden(void* data, calldata_t*) {
  request_lifecycle_transition(static_cast<FilterData*>(data), "hide");
}

void source_media_ended(void* data, calldata_t*) {
  request_lifecycle_transition(static_cast<FilterData*>(data), "ended");
}

void source_media_restart(void* data, calldata_t*) {
  request_lifecycle_transition(static_cast<FilterData*>(data), "restart");
}

void source_media_started(void* data, calldata_t*) {
  request_lifecycle_transition(static_cast<FilterData*>(data), "started");
}

constexpr std::size_t kVerificationFrameIdLogLimit = 128;

bool reserve_verification_frame_id_log(FilterData* filter) noexcept {
  if (!filter->verification_frame_id_logging_enabled ||
      filter->verification_frame_id_log_count >= kVerificationFrameIdLogLimit) {
    return false;
  }
  ++filter->verification_frame_id_log_count;
  return true;
}

void log_verification_generation(FilterData* filter,
                                 std::uint64_t generation,
                                 std::uint64_t input_id) noexcept {
  if (reserve_verification_frame_id_log(filter)) {
    blog(LOG_INFO,
         "OBS_FaceMosaic verification generation: generation=%llu input_id=%llu",
         static_cast<unsigned long long>(generation),
         static_cast<unsigned long long>(input_id));
  }
}

void log_verification_recovery_progress(
    FilterData* filter, const std::shared_ptr<Runtime>& runtime,
    std::uint64_t generation) noexcept {
  if (!filter->verification_frame_id_logging_enabled || !runtime ||
      !runtime->processor) {
    return;
  }
  const auto progress = runtime->processor->recovery_progress();
  if (!progress.generation_set || progress.generation != generation ||
      !progress.last_successful_id || progress.consecutive_successes == 0 ||
      progress.consecutive_successes > progress.required_successes ||
      (filter->recovery_progress_generation_logged == generation &&
       filter->recovery_progress_success_count_logged ==
           progress.consecutive_successes)) {
    return;
  }
  if (!reserve_verification_frame_id_log(filter)) {
    return;
  }
  filter->recovery_progress_generation_logged = generation;
  filter->recovery_progress_success_count_logged =
      progress.consecutive_successes;
  blog(LOG_INFO,
       "OBS_FaceMosaic verification recovery: generation=%llu input_id=%llu "
       "successes=%llu required=%llu",
       static_cast<unsigned long long>(generation),
       static_cast<unsigned long long>(*progress.last_successful_id),
       static_cast<unsigned long long>(progress.consecutive_successes),
       static_cast<unsigned long long>(progress.required_successes));
}

void log_verification_first_processed_output(
    FilterData* filter, const EmitResult& emitted) noexcept {
  if (!emitted.newly_processed || emitted.output_id == 0 ||
      filter->first_processed_output_generation_logged == emitted.generation ||
      !reserve_verification_frame_id_log(filter)) {
    return;
  }
  filter->first_processed_output_generation_logged = emitted.generation;
  blog(LOG_INFO,
       "OBS_FaceMosaic verification first processed output: generation=%llu output_id=%llu",
       static_cast<unsigned long long>(emitted.generation),
       static_cast<unsigned long long>(emitted.output_id));
}

void release_frame_from_filter_callback(obs_source_t* parent, obs_source_frame* frame) {
  if (!frame) {
    return;
  }
  // OBS 32.2.2 calls async filters from async_tick while holding the parent's
  // recursive async_mutex. The built-in async-delay filter releases retained
  // frames from filter_video on timestamp resets, so balance dropped callback
  // references here instead of accumulating them until a possibly absent tick.
  if (parent) {
    obs_source_release_frame(parent, frame);
  }
}

std::uint64_t steady_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
          .count());
}

void invalidate_render_guard(FilterData* filter, std::uint64_t generation) noexcept {
  // Clear permission before publishing the new generation. A draw racing a
  // settings change must see either the old generation or a closed guard.
  filter->renderable_generation.store(0, std::memory_order_release);
  filter->last_processed_output_time_ns.store(0, std::memory_order_release);
  filter->render_generation.store(generation, std::memory_order_release);
}

bool advance_generation_for_input_stall(FilterData* filter) {
  if (!mark_input_stall_generation_transition(
          true, filter->input_stall_generation_advanced)) {
    return false;
  }

  filter->last_processed_frame.reset();
  filter->last_processed_at = {};
  std::lock_guard<std::mutex> lock(filter->control->mutex);
  const auto generation = ++filter->control->generation;
  invalidate_render_guard(filter, generation);
  filter->control->status =
      "入力停止を検出: 加工済み出力から500ms経過後、次の描画で黒画面に切り替えます";
  return true;
}

void record_render_output(FilterData* filter, const EmitResult& emitted) noexcept {
  const auto lifecycle_epoch = filter->lifecycle_epoch->requested();
  if (!emitted.output ||
      lifecycle_epoch != filter->lifecycle_epoch->applied() ||
      filter->render_generation.load(std::memory_order_acquire) != emitted.generation) {
    return;
  }
  if (!emitted.renderable) {
    filter->renderable_generation.store(0, std::memory_order_release);
    filter->last_processed_output_time_ns.store(0, std::memory_order_release);
    return;
  }
  if (emitted.newly_processed) {
    filter->last_processed_output_time_ns.store(emitted.returned_at_ns,
                                                std::memory_order_release);
    log_verification_first_processed_output(filter, emitted);
  }
  filter->renderable_generation.store(emitted.generation,
                                      std::memory_order_release);
  if (filter->render_generation.load(std::memory_order_acquire) !=
          emitted.generation ||
      filter->lifecycle_epoch->requested() != lifecycle_epoch ||
      filter->lifecycle_epoch->applied() != lifecycle_epoch) {
    auto expected_generation = emitted.generation;
    if (filter->renderable_generation.compare_exchange_strong(
            expected_generation, 0, std::memory_order_acq_rel)) {
      filter->last_processed_output_time_ns.store(0,
                                                 std::memory_order_release);
    }
  }
}

std::size_t saturating_add(std::size_t left, std::size_t right) noexcept {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  return right > maximum - left ? maximum : left + right;
}

std::size_t saturating_multiply(std::size_t left, std::size_t right) noexcept {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  return right != 0 && left > maximum / right ? maximum : left * right;
}

std::size_t conservative_frame_size(const obs_source_frame& frame) noexcept {
  const auto width = static_cast<std::size_t>(frame.width);
  const auto height = static_cast<std::size_t>(frame.height);
  const auto fallback = saturating_multiply(saturating_multiply(width, height), 8U);
  if (width == 0 || height == 0) {
    return fallback;
  }

  const auto half_height = height / 2U + height % 2U;
  std::array<std::size_t, MAX_AV_PLANES> plane_heights{};
  plane_heights[0] = height;
  switch (frame.format) {
    case VIDEO_FORMAT_I420:
    case VIDEO_FORMAT_I010:
      plane_heights[1] = half_height;
      plane_heights[2] = half_height;
      break;
    case VIDEO_FORMAT_NV12:
    case VIDEO_FORMAT_P010:
      plane_heights[1] = half_height;
      break;
    case VIDEO_FORMAT_I40A:
      plane_heights[1] = half_height;
      plane_heights[2] = half_height;
      plane_heights[3] = height;
      break;
    case VIDEO_FORMAT_I422:
    case VIDEO_FORMAT_I444:
    case VIDEO_FORMAT_I210:
    case VIDEO_FORMAT_I412:
    case VIDEO_FORMAT_I42A:
    case VIDEO_FORMAT_YUVA:
    case VIDEO_FORMAT_YA2L:
      plane_heights[1] = height;
      plane_heights[2] = height;
      plane_heights[3] = height;
      break;
    case VIDEO_FORMAT_P216:
    case VIDEO_FORMAT_P416:
      plane_heights[1] = height;
      break;
    default:
      break;
  }

  std::size_t bytes = 0;
  bool found_plane = false;
  for (std::size_t plane = 0; plane < plane_heights.size(); ++plane) {
    if (!frame.data[plane]) {
      continue;
    }
    found_plane = true;
    auto row_bytes = static_cast<std::size_t>(frame.linesize[plane]);
    if (row_bytes == 0 || plane_heights[plane] == 0) {
      return fallback;
    }
    bytes = saturating_add(
        bytes, saturating_multiply(row_bytes, plane_heights[plane]));
  }
  return found_plane ? bytes : fallback;
}

void append_pending(FilterData* filter, PendingFrame pending) {
  const auto source_bytes = pending.estimated_source_bytes;
  filter->pending.push_back(std::move(pending));
  filter->pending_source_bytes =
      saturating_add(filter->pending_source_bytes, source_bytes);
  filter->pending_count.store(filter->pending.size());
  filter->latest_pending_bytes.store(filter->pending_source_bytes);
}

void pop_front_pending(FilterData* filter) {
  if (filter->pending.empty()) {
    return;
  }
  const auto bytes = filter->pending.front().estimated_source_bytes;
  filter->pending_source_bytes = bytes > filter->pending_source_bytes
                                     ? 0
                                     : filter->pending_source_bytes - bytes;
  filter->pending.pop_front();
  filter->pending_count.store(filter->pending.size());
  filter->latest_pending_bytes.store(filter->pending_source_bytes);
}

void release_all_pending_from_callback(FilterData* filter, obs_source_t* parent) noexcept {
  while (!filter->pending.empty()) {
    auto* frame = filter->pending.front().source_frame;
    pop_front_pending(filter);
    release_frame_from_filter_callback(parent, frame);
  }
  filter->pending_source_bytes = 0;
  filter->latest_pending_bytes.store(0);
}

bool request_reload(FilterData* filter, const Settings& requested,
                   bool force = false) {
  std::shared_ptr<Runtime> old_runtime;
  bool reload = false;
  bool manual_changed = false;
  std::unique_lock<std::mutex> output_commit_lock(
      filter->output_commit_mutex);
  {
    std::lock_guard<std::mutex> lock(filter->control->mutex);
    manual_changed = filter->control->settings.manual_mask != requested.manual_mask;
    Settings comparable = requested;
    comparable.manual_mask = filter->control->settings.manual_mask;
    comparable.latency_ms = filter->control->settings.latency_ms;
    Settings current = filter->control->settings;
    current.manual_mask = filter->control->settings.manual_mask;
    reload = force || !(comparable == current);
    const bool latency_changed =
        filter->control->settings.latency_ms != requested.latency_ms;
    filter->control->settings = requested;
    if (reload) {
      old_runtime = filter->control->runtime;
      // Make the processor reject old and concurrent submissions before the
      // new generation becomes visible to the video callback.
      if (old_runtime && old_runtime->processor) {
        old_runtime->processor->set_ready(false);
      }
      ++filter->control->settings_revision;
      ++filter->control->generation;
      invalidate_render_guard(filter, filter->control->generation);
      filter->control->status = "設定を反映中: 入力を遮蔽しています";
      filter->control->runtime.reset();
    } else if (manual_changed || latency_changed) {
      const auto active_runtime = filter->control->runtime;
      if (latency_changed) {
        old_runtime = active_runtime;
        // Keep settings and processor readiness serialized under Control.
        // A callback cannot observe the new generation until old work has
        // been invalidated.
        if (old_runtime && old_runtime->processor) {
          old_runtime->processor->set_ready(false);
        }
      }
      if (active_runtime && active_runtime->processor) {
        active_runtime->processor->set_manual_mask(requested.manual_mask);
        filter->control->status = "設定を反映中: 入力を遮蔽しています";
      }
      // Invalidate every in-flight result. The next video callback sees the
      // new generation, discards pending/replay state, and starts black.
      ++filter->control->generation;
      invalidate_render_guard(filter, filter->control->generation);
    }
  }
  if (reload) {
    filter->control->changed.notify_one();
  }
  output_commit_lock.unlock();
  return reload;
}

void filter_update(void* data, obs_data_t* settings) {
  auto* filter = static_cast<FilterData*>(data);
#if defined(OBSFM_ENABLE_TEST_HOOKS)
  filter->control->test_failure->store(obs_data_get_bool(settings, "__test_fail_inference"), std::memory_order_release);
#endif
  migrate_settings(settings);
  const auto requested = read_settings(settings);
  request_reload(filter, requested);
}

bool retry_button(obs_properties_t*, obs_property_t*, void* data) {
  auto* filter = static_cast<FilterData*>(data);
  Settings settings;
  {
    std::lock_guard<std::mutex> lock(filter->control->mutex);
    settings = filter->control->settings;
  }
  request_reload(filter, settings, true);
  return true;
}

std::string state_name(ProcessorState state);

std::string display_status(FilterData* filter) {
  std::ostringstream stream;
  std::string status;
  Settings settings;
  auto runtime = filter->runtime();
  {
    std::lock_guard<std::mutex> lock(filter->control->mutex);
    status = filter->control->status;
    settings = filter->control->settings;
  }
  stream << status << "\nモデル: " << model_filename(settings)
         << " / 推論: " << (settings.use_cpu ? "CPU" : "DirectML")
         << " / L: " << settings.latency_ms << " ms"
         << " / 経過: " << filter->latest_latency_ms.load() << " ms"
         << "\n処理出力: " << filter->output_count.load() << " (直近FPS "
         << filter->output_fps.load() << ") / 黒遮蔽: " << filter->black_count.load()
         << " / 破棄・失敗: " << filter->drop_count.load()
         << " / 待機中: " << filter->pending_count.load();
  if (runtime && runtime->processor) {
    stream << "\n実行デバイス: " << runtime->adapter;
    stream << " / 状態: " << state_name(runtime->processor->state());
    const auto metrics = runtime->processor->metrics();
    stream << "\n推論投入: " << metrics.submitted
           << " / 推論完了: " << metrics.completed
           << " / キュー溢れ: " << metrics.queue_evictions
           << " / 期限超過: " << metrics.deadline_drops
           << " / 推論失敗: " << metrics.detector_failures;
  }
  return stream.str();
}

obs_properties_t* filter_properties(void* data) {
  auto* filter = static_cast<FilterData*>(data);
  std::string selected;
  if (filter) {
    std::lock_guard<std::mutex> lock(filter->control->mutex);
    selected = filter->control->settings.gpu_adapter;
  }
  const std::string status = filter
                                ? display_status(filter)
                                : "フィルター未作成: 追加後にモデルフォルダーを指定してください";
  return create_filter_properties(selected, status, retry_button, filter);
}

void mark_key_black(FilterData* filter, const FrameKey& key) {
  for (auto& pending : filter->pending) {
    if (pending.key.generation == key.generation && pending.key.id == key.id &&
        pending.key.pts_ns == key.pts_ns) {
      if (!pending.force_black) {
        pending.black_cause = BlackCause::queue_eviction;
      }
      pending.force_black = true;
      pending.result.reset();
      return;
    }
  }
}

void drain_results(FilterData* filter, const std::shared_ptr<Runtime>& runtime) {
  if (!runtime || !runtime->processor) {
    return;
  }
  while (auto result = runtime->processor->poll()) {
    const auto match = std::find_if(filter->pending.begin(), filter->pending.end(),
                                    [&](const PendingFrame& frame) {
      return frame.key.generation == result->key.generation &&
             frame.key.id == result->key.id && frame.key.pts_ns == result->key.pts_ns;
    });
    if (match == filter->pending.end()) {
      ++filter->drop_count;
      continue;
    }
    if (match->force_black || result->kind == OutputKind::drop || !result->frame) {
      match->black_cause = BlackCause::other;
      match->force_black = true;
      match->result.reset();
      if (result->kind == OutputKind::drop) {
        ++filter->drop_count;
      }
    } else {
      match->result = std::move(*result);
    }
  }
}

bool frame_ready(const PendingFrame& pending) {
  return pending.force_black || pending.result.has_value();
}

bool has_same_frame_processed_result(const PendingFrame& pending) {
  return !pending.force_black && pending.result &&
         pending.result->kind == OutputKind::processed &&
         pending.result->frame &&
         pending.result->key.generation == pending.key.generation &&
         pending.result->key.id == pending.key.id &&
         pending.result->key.pts_ns == pending.key.pts_ns;
}

EmitResult emit_front(FilterData* filter, std::uint64_t current_pts,
                      const Settings& settings,
                      const std::shared_ptr<Runtime>& runtime,
                      std::uint64_t current_generation,
                      bool current_manual_mask, bool pressure = false) {
  if (filter->pending.empty()) {
    return {};
  }
  auto& pending = filter->pending.front();
  const auto now = Clock::now();
  const auto elapsed = now - pending.submitted_at;
  if (!frame_ready(pending) && elapsed >= std::chrono::milliseconds(settings.latency_ms)) {
    pending.black_cause = BlackCause::other;
    pending.force_black = true;
    ++filter->drop_count;
  }

  const std::uint64_t latency_ns =
      static_cast<std::uint64_t>(settings.latency_ms) * kNanosecondsPerMillisecond;
  const bool media_delay_reached = current_pts >= pending.key.pts_ns &&
                                   current_pts - pending.key.pts_ns >= latency_ns;
  switch (pressure_action(pressure, has_same_frame_processed_result(pending),
                          pending.black_cause)) {
    case PressureAction::normal_timing:
    case PressureAction::emit_processed:
      // Pressure releases this frame immediately. Preserve a valid result for
      // this exact frame.
      break;
    case PressureAction::replay_if_eligible:
      // Queue eviction must remain an explicit black/replay candidate even
      // if an inconsistent caller left force_black unset.
      pending.force_black = true;
      break;
    case PressureAction::fail_closed:
      // The derived frame cap and hard byte budget remain strict bounds. A
      // frame with no valid same-frame result is never passed through raw.
      pending.black_cause = BlackCause::other;
      pending.force_black = true;
      pending.result.reset();
      break;
  }
  const bool immediate_emit =
      pressure ||
      (pending.force_black &&
       (settings.manual_mask ||
        elapsed >= std::chrono::milliseconds(settings.latency_ms)));
  if (!frame_ready(pending) ||
      (!media_delay_reached && !immediate_emit)) {
    return {};
  }

  bool written = false;
  bool replayed = false;
  if (!pending.force_black && pending.result && pending.result->frame) {
    ScopedMeasurement measurement(filter->performance_metrics.get(),
                                  Stage::obs_write_processed);
    written = filter->converter.write(*pending.source_frame, *pending.result->frame);
    if (!written) {
      measurement.fail();
    }
  } else if (pending.force_black && pending.black_cause == BlackCause::queue_eviction &&
             runtime && runtime->processor &&
             runtime->processor->state() == ProcessorState::running &&
             filter->last_processed_frame) {
    const auto cache_age = now - filter->last_processed_at;
    replayed = can_replay_cached_frame(ReplayCandidate{
        .cause = pending.black_cause,
        .generation = pending.key.generation,
        .current_generation = current_generation,
        .cached_generation = filter->last_processed_frame->generation,
        .width = static_cast<int>(pending.source_frame->width),
        .height = static_cast<int>(pending.source_frame->height),
        .cached_width = filter->last_processed_frame->width,
        .cached_height = filter->last_processed_frame->height,
        .cache_available = true,
        .runtime_running = true,
        .manual_mask = current_manual_mask,
        .cache_age = cache_age,
    });
    if (replayed) {
      ScopedMeasurement measurement(filter->performance_metrics.get(),
                                    Stage::obs_write_replay);
      written = filter->converter.write(*pending.source_frame,
                                        *filter->last_processed_frame);
      if (!written) {
        measurement.fail();
      }
    }
  }
  if (!written) {
    // A replay write failure falls back to a black write, so it must be
    // counted as black output rather than replayed output.
    replayed = false;
    ScopedMeasurement measurement(filter->performance_metrics.get(),
                                  Stage::obs_write_black);
    written = filter->converter.write_black(*pending.source_frame);
    if (!written) {
      measurement.fail();
    }
    pending.black_cause = BlackCause::other;
    pending.force_black = true;
  }
  if (!written) {
    // The frame may now be partly overwritten, so it must never pass through.
    auto* dropped_frame = pending.source_frame;
    pop_front_pending(filter);
    ++filter->drop_count;
    filter->performance_metrics->record_output(DiagnosticOutput::dropped);
    return EmitResult{nullptr, dropped_frame};
  }

  const auto output_frame = pending.source_frame;
  const auto output_generation = pending.key.generation;
  const auto output_id = pending.key.id;
  const auto output_submitted_at = pending.submitted_at;
  filter->latest_latency_ms.store(
      std::chrono::duration_cast<std::chrono::milliseconds>(now - pending.submitted_at).count());
  const bool is_black = !replayed &&
      (pending.force_black || !pending.result ||
       pending.result->kind != OutputKind::processed);
  if (!is_black && !replayed && pending.result && pending.result->frame) {
    filter->last_processed_frame = pending.result->frame;
    filter->last_processed_at = now;
  }
  pop_front_pending(filter);
  ++filter->output_count;
  if (is_black) {
    ++filter->black_count;
  }
  if (replayed) {
    ++filter->replay_count;
  }
  filter->performance_metrics->record_output(
      is_black ? DiagnosticOutput::black
               : (replayed ? DiagnosticOutput::replayed
                           : DiagnosticOutput::processed));
  try {
    filter->output_times.push_back(now);
    while (!filter->output_times.empty() &&
           now - filter->output_times.front() > std::chrono::seconds(1)) {
      filter->output_times.pop_front();
    }
    filter->output_fps.store(filter->output_times.size());
  } catch (...) {
    // Output accounting is best-effort; do not lose a complete frame after it
    // has been popped from the bounded pending queue.
    filter->output_fps.store(filter->output_times.size());
  }
  const auto returned_at = Clock::now();
  const auto return_stage = is_black
      ? Stage::obs_return_black
      : (replayed ? Stage::obs_return_replay : Stage::obs_return_processed);
  filter->performance_metrics->record(
      return_stage,
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          returned_at - output_submitted_at));
  return EmitResult{
      .output = output_frame,
      .release_after_gate = nullptr,
      .renderable = !is_black,
      .newly_processed = !is_black && !replayed,
      .generation = output_generation,
      .returned_at_ns = steady_ns(),
      .output_id = output_id,
  };
}

std::string state_name(ProcessorState state) {
  switch (state) {
    case ProcessorState::initializing: return "初期化中";
    case ProcessorState::awaiting_input: return "入力待ち";
    case ProcessorState::recovering: return "復帰確認";
    case ProcessorState::running: return "稼働中";
    case ProcessorState::obscured: return "遮蔽中";
    case ProcessorState::manual_mask: return "手動遮蔽";
    case ProcessorState::stopping: return "終了処理中";
  }
  return "不明";
}

obs_source_frame* process_video(void* data, obs_source_frame* input) {
  auto* filter = static_cast<FilterData*>(data);
  if (!filter || !input) {
    return nullptr;
  }
  auto* parent = obs_filter_get_parent(filter->context);

  std::lock_guard<std::mutex> video_lock(filter->video_mutex);
  const auto now = Clock::now();
  const auto input_time_ns = steady_ns();
  const auto previous_input_time_ns =
      filter->last_input_time_ns.load(std::memory_order_acquire);
  const auto stable_interval_before_input = filter->stable_input_interval_ns;
  const auto lifecycle_epoch = filter->lifecycle_epoch->requested();
  const bool lifecycle_transition_requested =
      lifecycle_epoch != filter->lifecycle_epoch->applied();
  const bool input_gap_exceeded = input_wall_gap_requires_generation_transition(
      previous_input_time_ns, input_time_ns, stable_interval_before_input);
  const bool input_stall_transition =
      filter->input_stall_generation_advanced || input_gap_exceeded;
  if (input_gap_exceeded) {
    advance_generation_for_input_stall(filter);
  }
  // A tick may already have advanced the generation during the gap. This
  // callback consumes that transition and allows the next stall to advance it.
  rearm_input_stall_generation_transition(
      filter->input_stall_generation_advanced);
  filter->last_input_time_ns.store(input_time_ns, std::memory_order_release);
  if (input->width > 0 && input->height > 0) {
    filter->render_width.store(input->width, std::memory_order_release);
    filter->render_height.store(input->height, std::memory_order_release);
  }
  Settings settings;
  std::shared_ptr<Runtime> runtime;
  std::uint64_t generation = 1;
  {
    std::lock_guard<std::mutex> lock(filter->control->mutex);
    if (filter->control->status.starts_with("入力停止を検出:")) {
      filter->control->status = "入力復帰中: 新しい世代で処理を再開します";
    }
    settings = filter->control->settings;
    generation = filter->control->generation;
    runtime = filter->control->runtime;
  }

  bool signature_changed = false;
  const bool converter_ok = filter->converter.configure(*input, &signature_changed);
  if (!converter_ok) {
    // Configuration failure prevents a pixel conversion attempt. Keep it in
    // the same stage's failure count so this path is visible in diagnostics.
    filter->performance_metrics->record(
        Stage::input_to_rgba, std::chrono::nanoseconds::zero(), true);
  }
  const bool dimensions_changed = filter->last_width != 0 &&
      (filter->last_width != input->width || filter->last_height != input->height ||
       filter->last_format != input->format);
  const bool timestamp_jump = pts_discontinuity_requires_generation_transition(
      filter->last_pts, input->timestamp, stable_interval_before_input);
  const bool generation_changed = filter->active_generation != generation;
  const bool interval_history_reset = dimensions_changed || timestamp_jump ||
      (signature_changed && filter->last_width != 0) || generation_changed ||
      lifecycle_transition_requested;
  const auto frame_interval_ns = stable_frame_interval_from_pts(
      filter->input_interval_estimator, input->timestamp, filter->last_pts,
      filter->last_width != 0 && !interval_history_reset,
      interval_history_reset);
  filter->stable_input_interval_ns = frame_interval_ns;
  if (!input_stall_transition &&
      (lifecycle_transition_requested || dimensions_changed || timestamp_jump ||
       (signature_changed && filter->last_width != 0))) {
    {
      std::lock_guard<std::mutex> lock(filter->control->mutex);
      // A settings update may have landed after the callback's initial
      // snapshot. Source-driven generation changes must use the current
      // settings/runtime, especially the latest processing deadline.
      settings = filter->control->settings;
      runtime = filter->control->runtime;
      generation = ++filter->control->generation;
      invalidate_render_guard(filter, generation);
    }
  }
  const auto transition_action =
      generation_transition_action(filter->active_generation, generation);
  const bool generation_transition =
      transition_action ==
      GenerationTransitionAction::discard_pending_and_black_current;
  if (generation_transition) {
    auto stale_frames = filter->detach_pending_for_transition(transition_action);
    for (const auto& stale : stale_frames) {
      release_frame_from_filter_callback(parent, stale.source_frame);
    }
    filter->drop_count.fetch_add(stale_frames.size());
    filter->last_processed_frame.reset();
    filter->last_processed_at = {};
    if (runtime && runtime->processor) {
      runtime->processor->reset_generation(
          generation, std::chrono::milliseconds(settings.latency_ms));
      runtime->processor->set_manual_mask(settings.manual_mask);
      runtime->processor->set_ready(true);
      filter->active_runtime = runtime;
    } else {
      filter->active_runtime.reset();
    }
    filter->active_generation = generation;
    filter->recovery_progress_generation_logged = generation;
    filter->recovery_progress_success_count_logged = 0;
  }
  if (lifecycle_transition_requested && generation_transition) {
    filter->lifecycle_epoch->acknowledge(lifecycle_epoch);
    blog(LOG_INFO,
         "OBS_FaceMosaic media lifecycle transition applied: epoch=%llu generation=%llu",
         static_cast<unsigned long long>(lifecycle_epoch),
         static_cast<unsigned long long>(generation));
  }
  filter->latest_input_interval_us.store(frame_interval_ns / 1'000ULL);
  filter->latest_pending_limit.store(
      outstanding_frame_limit(static_cast<std::uint64_t>(settings.latency_ms),
                              frame_interval_ns));
  filter->last_pts = input->timestamp;
  filter->last_width = input->width;
  filter->last_height = input->height;
  filter->last_format = input->format;

  if (runtime && runtime->processor) {
    if (filter->active_runtime.lock() != runtime || filter->active_generation != generation) {
      runtime->processor->reset_generation(
          generation, std::chrono::milliseconds(settings.latency_ms));
      runtime->processor->set_manual_mask(settings.manual_mask);
      runtime->processor->set_ready(true);
      filter->active_runtime = runtime;
      filter->active_generation = generation;
    }
    {
      std::lock_guard<std::mutex> lock(filter->control->mutex);
      if (filter->control->generation == generation &&
          filter->control->runtime == runtime &&
          filter->control->status.starts_with("設定を反映中:")) {
        filter->control->status = settings.manual_mask
                                      ? "準備完了: 手動全画面遮蔽中"
                                      : "準備完了: 新世代の復帰確認中";
      }
    }
  }

  Frame frame;
  frame.generation = generation;
  frame.id = filter->next_id++;
  frame.pts_ns = input->timestamp;
  frame.width = static_cast<int>(input->width);
  frame.height = static_cast<int>(input->height);
  const FrameKey key{frame.generation, frame.id, frame.pts_ns};
  if (generation_transition) {
    log_verification_generation(filter, generation, frame.id);
  }
  PendingFrame pending;
  pending.source_frame = input;
  pending.key = key;
  pending.submitted_at = now;
  pending.estimated_source_bytes = conservative_frame_size(*input);

  const char* parent_id = nullptr;
  if (parent) {
    parent_id = obs_source_get_id(parent);
  }
  const bool supported_source = parent_id && std::string(parent_id) == "ffmpeg_source";
  if (generation_transition) {
    // A settings/source generation change invalidates all old work. Present an
    // opaque black frame immediately, then begin processing on the next input.
    pending.black_cause = BlackCause::other;
    pending.force_black = true;
    ++filter->drop_count;
  } else if (settings.manual_mask || !runtime || !runtime->processor || !supported_source || !converter_ok) {
    pending.black_cause = BlackCause::other;
    pending.force_black = true;
    if (!supported_source) {
      ++filter->drop_count;
    }
  } else {
    ConvertResult converted = ConvertResult::failure;
    {
      ScopedMeasurement measurement(filter->performance_metrics.get(),
                                    Stage::input_to_rgba);
      converted = filter->converter.read(*input, frame);
      if (converted != ConvertResult::success) {
        measurement.fail();
      }
    }
    if (converted != ConvertResult::success) {
      pending.black_cause = BlackCause::other;
      pending.force_black = true;
      ++filter->drop_count;
    } else {
      const auto submit = runtime->processor->submit(std::move(frame), now);
      if (submit.evicted) {
        mark_key_black(filter, *submit.evicted);
      }
      if (!submit.accepted) {
        pending.black_cause = BlackCause::other;
        pending.force_black = true;
        ++filter->drop_count;
      }
    }
  }
  append_pending(filter, std::move(pending));

  if (runtime && runtime->processor) {
    runtime->processor->set_manual_mask(settings.manual_mask);
    drain_results(filter, runtime);
    log_verification_recovery_progress(filter, runtime, generation);
    if (runtime->processor->state() == ProcessorState::running) {
      // Initial startup and settings/source generation transitions both wait
      // for the processor's consecutive-success gate. Update only one of those
      // readiness messages when this exact runtime and generation are still
      // current. The allowlist and current manual-mask setting preserve
      // load-failure, input-stop, manual-mask, and newer-update messages.
      std::lock_guard<std::mutex> lock(filter->control->mutex);
      const bool awaiting_processed_frames =
          filter->control->status == "準備完了: 入力フレームを待っています" ||
          filter->control->status == "準備完了: 新世代の復帰確認中";
      if (filter->control->generation == generation &&
          filter->control->runtime == runtime &&
          !filter->control->settings.manual_mask &&
          awaiting_processed_frames) {
        filter->control->status = "準備完了: 処理中";
      }
    }
  }

  // Keep the number of borrowed OBS frames bounded. If latency/configuration
  // cannot be met under pressure, emit an opaque black frame for the oldest
  // matching OBS frame; never return an unprocessed input frame.
  const auto pending_limit = static_cast<std::size_t>(
      filter->latest_pending_limit.load());
  const bool pressure = filter->pending.size() > pending_limit ||
                        filter->pending_source_bytes > kMaxOutstandingFrameBytes;
  // Serialize the output write with settings updates. The update path never
  // takes video_mutex. This gate is independent of Control, so no settings
  // mutex or OBS source/frame ownership API is held during pixel conversion;
  // failed frame releases are deferred until after unlocking. If an update
  // won the gate after this callback's initial snapshot,
  // discard its stale adapter queue instead of returning old processed/replay
  // pixels after the new settings became visible.
  std::unique_lock<std::mutex> output_gate(filter->output_commit_mutex);
  Settings output_settings;
  std::uint64_t output_generation = 0;
  {
    std::lock_guard<std::mutex> control_lock(filter->control->mutex);
    output_settings = filter->control->settings;
    output_generation = filter->control->generation;
  }
  if (output_generation != generation || filter->lifecycle_epoch->pending()) {
    output_gate.unlock();
    auto stale_frames = filter->detach_pending();
    for (const auto& stale : stale_frames) {
      release_frame_from_filter_callback(parent, stale.source_frame);
    }
    filter->drop_count.fetch_add(stale_frames.size());
    filter->last_processed_frame.reset();
    filter->last_processed_at = {};
    return nullptr;
  }
  const auto emitted = emit_front(
      filter, input->timestamp, output_settings, runtime, generation,
      output_settings.manual_mask, pressure || generation_transition);
  output_gate.unlock();
  record_render_output(filter, emitted);
  if (emitted.release_after_gate) {
    release_frame_from_filter_callback(parent, emitted.release_after_gate);
  }
  if (emitted.output) {
    if (filter->lifecycle_epoch->pending()) {
      release_frame_from_filter_callback(parent, emitted.output);
      return nullptr;
    }
    return emitted.output;
  }

  return nullptr;
}

const char* filter_name(void*) {
  return "自動顔モザイク (OBS_FaceMosaic)";
}

void filter_update_wrapper(void* data, obs_data_t* settings) {
  if (data && settings) {
    filter_update(data, settings);
  }
}

void* filter_create_wrapper(obs_data_t* settings, obs_source_t* context) {
  try {
    auto* filter = new FilterData(context);
    if (settings) {
      filter_update(filter, settings);
    }
    return filter;
  } catch (...) {
    blog(LOG_ERROR, "OBS_FaceMosaic: filter initialization failed");
    return nullptr;
  }
}

void filter_destroy_wrapper(void* data) {
  delete static_cast<FilterData*>(data);
}

obs_properties_t* filter_properties_wrapper(void* data) {
  return filter_properties(data);
}

void filter_video_tick(void* data, float) {
  auto* filter = static_cast<FilterData*>(data);
  if (!filter) {
    return;
  }

  std::deque<PendingFrame> release_now;
  obs_source_t* parent = nullptr;
  const auto now_ns = steady_ns();
  {
    std::lock_guard<std::mutex> lock(filter->video_mutex);
    const auto last_input_ns = filter->last_input_time_ns.load();
    const bool input_stalled = input_stop_requires_generation_transition(
        last_input_ns, now_ns);
    if (input_stalled && filter->weak_parent) {
      // Acquire a transient strong ref only while frames are being released.
      // Persistent strong ownership would keep the parent alive through its
      // owned filter and prevent OBS teardown from reaching filter_remove.
      parent = obs_weak_source_get_source(filter->weak_parent);
    }
    if (input_stalled) {
      advance_generation_for_input_stall(filter);
      // filter_video no longer receives frames after an input stop. The render
      // callback independently checks the last processed output and paints
      // opaque black once it is older than 500 ms.
      if (parent) {
        release_now = filter->detach_pending();
      }
      {
        std::lock_guard<std::mutex> control_lock(filter->control->mutex);
        filter->control->status = "入力停止を検出: 加工済み出力から500ms経過後、次の描画で黒画面に切り替えます";
      }
    }
  }

  if (parent) {
    for (const auto& frame : release_now) {
      if (frame.source_frame) {
        obs_source_release_frame(parent, frame.source_frame);
      }
    }
    obs_source_release(parent);
  }
}

bool draw_opaque_black(std::uint32_t width, std::uint32_t height) {
  if (width == 0 || height == 0) {
    return false;
  }

  gs_effect_t* solid = obs_get_base_effect(OBS_EFFECT_SOLID);
  if (!solid) {
    return false;
  }
  gs_eparam_t* color_param = gs_effect_get_param_by_name(solid, "color");
  gs_technique_t* technique = gs_effect_get_technique(solid, "Solid");
  if (!color_param || !technique) {
    return false;
  }

  struct vec4 color;
  vec4_set(&color, 0.0F, 0.0F, 0.0F, 1.0F);
  gs_effect_set_vec4(color_param, &color);

  // Replace destination pixels so the privacy mask is opaque even when the
  // scene beneath this source contains a visible image.
  gs_blend_state_push();
  gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
  const size_t passes = gs_technique_begin(technique);
  for (size_t index = 0; index < passes; ++index) {
    gs_technique_begin_pass(technique, index);
    gs_draw_sprite(nullptr, 0, width, height);
    gs_technique_end_pass(technique);
  }
  gs_technique_end(technique);
  gs_blend_state_pop();
  return true;
}

void filter_video_render(void* data, gs_effect_t* effect) {
  UNUSED_PARAMETER(effect);
  auto* filter = static_cast<FilterData*>(data);
  if (!filter) {
    return;
  }

  const auto now_ns = steady_ns();
  const auto lifecycle_epoch = filter->lifecycle_epoch->requested();
  const auto generation = filter->render_generation.load(std::memory_order_acquire);
  const auto renderable_generation =
      filter->renderable_generation.load(std::memory_order_acquire);
  const auto last_processed_ns =
      filter->last_processed_output_time_ns.load(std::memory_order_acquire);
  constexpr std::uint64_t kRenderFreshnessLimitNs =
      500ULL * kNanosecondsPerMillisecond;
  const bool fresh_processed_output =
      generation != 0 && renderable_generation == generation &&
      lifecycle_epoch == filter->lifecycle_epoch->applied() &&
      last_processed_ns != 0 && now_ns >= last_processed_ns &&
      now_ns - last_processed_ns < kRenderFreshnessLimitNs;

  if (fresh_processed_output && obs_filter_get_parent(filter->context) &&
      obs_filter_get_target(filter->context) &&
      filter->lifecycle_epoch->requested() == lifecycle_epoch &&
      filter->lifecycle_epoch->applied() == lifecycle_epoch &&
      filter->render_generation.load(std::memory_order_acquire) == generation &&
      filter->renderable_generation.load(std::memory_order_acquire) == generation) {
    filter->render_black_latched.store(false, std::memory_order_release);
    if (filter->render_black_stop_logged.exchange(false,
                                                  std::memory_order_acq_rel)) {
      const auto stop_started_ns =
          filter->render_black_stop_started_at_ns.load(std::memory_order_acquire);
      const auto black_duration_ms = stop_started_ns != 0 && now_ns >= stop_started_ns
          ? (now_ns - stop_started_ns) / kNanosecondsPerMillisecond
          : 0;
      const auto black_output_count =
          filter->render_black_stop_output_count.load(std::memory_order_acquire);
      const auto current_output_count = filter->output_count.load(std::memory_order_relaxed);
      const auto output_delta = current_output_count >= black_output_count
          ? current_output_count - black_output_count
          : 0;
      blog(LOG_INFO,
           "OBS_FaceMosaic A03 black guard recovered: generation=%llu "
           "black_ms=%llu output_delta=%llu outputs=%llu",
           static_cast<unsigned long long>(generation),
           static_cast<unsigned long long>(black_duration_ms),
           static_cast<unsigned long long>(output_delta),
           static_cast<unsigned long long>(current_output_count));
    }
    filter->render_black_stable_logged.store(false, std::memory_order_release);
    filter->render_black_started_at_ns.store(0, std::memory_order_release);
    // OBS's standard filter delegation renders the target. For this async
    // input, filter_video has already replaced its cached frame with the
    // processed result before this render callback is reached.
    if (filter->lifecycle_epoch->requested() == lifecycle_epoch &&
        filter->lifecycle_epoch->applied() == lifecycle_epoch &&
        filter->render_generation.load(std::memory_order_acquire) == generation &&
        filter->renderable_generation.load(std::memory_order_acquire) == generation &&
        filter->last_processed_output_time_ns.load(std::memory_order_acquire) ==
            last_processed_ns) {
      obs_source_skip_video_filter(filter->context);
      return;
    }
  }

  const bool black_drawn = draw_opaque_black(
      filter->render_width.load(std::memory_order_acquire),
      filter->render_height.load(std::memory_order_acquire));
  if (!black_drawn) {
    return;
  }

  if (!filter->render_black_latched.load(std::memory_order_acquire)) {
    filter->render_black_started_at_ns.store(now_ns, std::memory_order_release);
    filter->render_black_latched.store(true, std::memory_order_release);
  }

  const auto last_input_ns = filter->last_input_time_ns.load(std::memory_order_acquire);
  const auto input_idle_ns = last_input_ns != 0 && now_ns >= last_input_ns
      ? now_ns - last_input_ns
      : 0;
  constexpr std::uint64_t kInputStopLogThresholdNs =
      500ULL * kNanosecondsPerMillisecond;
  if (input_idle_ns >= kInputStopLogThresholdNs &&
      !filter->render_black_stop_logged.load(std::memory_order_acquire)) {
    const auto stop_output_count = filter->output_count.load(std::memory_order_relaxed);
    filter->render_black_stop_started_at_ns.store(now_ns, std::memory_order_release);
    filter->render_black_stop_output_count.store(stop_output_count,
                                                 std::memory_order_release);
    filter->render_black_stable_logged.store(false, std::memory_order_release);
    filter->render_black_stop_logged.store(true, std::memory_order_release);
    const auto black_started_ns =
        filter->render_black_started_at_ns.load(std::memory_order_acquire);
    const auto black_duration_ms = black_started_ns != 0 && now_ns >= black_started_ns
        ? (now_ns - black_started_ns) / kNanosecondsPerMillisecond
        : 0;
    const auto processed_idle_ms = last_processed_ns != 0 && now_ns >= last_processed_ns
        ? (now_ns - last_processed_ns) / kNanosecondsPerMillisecond
        : 0;
    blog(LOG_INFO,
         "OBS_FaceMosaic A03 black guard entered: generation=%llu "
         "input_idle_ms=%llu processed_idle_ms=%llu black_ms=%llu "
         "outputs=%llu size=%ux%u",
         static_cast<unsigned long long>(generation),
         static_cast<unsigned long long>(input_idle_ns /
                                         kNanosecondsPerMillisecond),
         static_cast<unsigned long long>(processed_idle_ms),
         static_cast<unsigned long long>(black_duration_ms),
         static_cast<unsigned long long>(stop_output_count),
         filter->render_width.load(std::memory_order_relaxed),
         filter->render_height.load(std::memory_order_relaxed));
    return;
  }

  constexpr std::uint64_t kBlackStableLogIntervalNs =
      500ULL * kNanosecondsPerMillisecond;
  const auto stop_started_ns =
      filter->render_black_stop_started_at_ns.load(std::memory_order_acquire);
  if (filter->render_black_stop_logged.load(std::memory_order_acquire) &&
      stop_started_ns != 0 && now_ns >= stop_started_ns &&
      now_ns - stop_started_ns >= kBlackStableLogIntervalNs &&
      !filter->render_black_stable_logged.exchange(true, std::memory_order_acq_rel)) {
    const auto black_output_count =
        filter->render_black_stop_output_count.load(std::memory_order_acquire);
    const auto current_output_count = filter->output_count.load(std::memory_order_relaxed);
    const auto output_delta = current_output_count >= black_output_count
        ? current_output_count - black_output_count
        : 0;
    blog(LOG_INFO,
         "OBS_FaceMosaic A03 black guard stable: generation=%llu "
         "black_ms=%llu output_delta=%llu outputs=%llu",
         static_cast<unsigned long long>(generation),
         static_cast<unsigned long long>((now_ns - stop_started_ns) /
                                         kNanosecondsPerMillisecond),
         static_cast<unsigned long long>(output_delta),
         static_cast<unsigned long long>(current_output_count));
  }
}

std::uint32_t filter_width(void* data) {
  auto* filter = static_cast<FilterData*>(data);
  return filter ? filter->render_width.load(std::memory_order_acquire) : 0;
}

std::uint32_t filter_height(void* data) {
  auto* filter = static_cast<FilterData*>(data);
  return filter ? filter->render_height.load(std::memory_order_acquire) : 0;
}

enum gs_color_space filter_video_get_color_space(
    void* data, std::size_t count,
    const enum gs_color_space* preferred_spaces) {
  auto* filter = static_cast<FilterData*>(data);
  auto* target = filter ? obs_filter_get_target(filter->context) : nullptr;
  if (!target) {
    return count > 0 ? preferred_spaces[0] : GS_CS_SRGB;
  }
  return obs_source_get_color_space(target, count, preferred_spaces);
}

void log_diagnostics(FilterData* filter, std::uint32_t input_width,
                     std::uint32_t input_height, enum video_format input_format) {
  try {
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            Clock::now().time_since_epoch())
                            .count();
    auto previous_ms = filter->last_diagnostic_log_ms.load();
    if (previous_ms != 0 && now_ms - previous_ms < 5'000) {
      return;
    }
    if (!filter->last_diagnostic_log_ms.compare_exchange_strong(previous_ms, now_ms)) {
      return;
    }

    Settings settings;
    std::shared_ptr<Runtime> runtime;
    std::string status;
    std::uint64_t generation = 0;
    std::uint64_t detector_load_count = 0;
    {
      std::lock_guard<std::mutex> lock(filter->control->mutex);
      settings = filter->control->settings;
      runtime = filter->control->runtime;
      status = filter->control->status;
      generation = filter->control->generation;
      detector_load_count = filter->control->detector_load_count.load(
          std::memory_order_relaxed);
    }

    std::uint64_t submitted = 0;
    std::uint64_t completed = 0;
    std::uint64_t rejected = 0;
    std::uint64_t queue_evictions = 0;
    std::uint64_t deadline_drops = 0;
    std::uint64_t detector_failures = 0;
    int processor_state = -1;
    std::string adapter = settings.use_cpu ? "CPU" : "不明";
    if (runtime && runtime->processor) {
      const auto metrics = runtime->processor->metrics();
      submitted = metrics.submitted;
      completed = metrics.completed;
      rejected = metrics.rejected;
      queue_evictions = metrics.queue_evictions;
      deadline_drops = metrics.deadline_drops;
      detector_failures = metrics.detector_failures;
      processor_state = static_cast<int>(runtime->processor->state());
      adapter = runtime->adapter;
    }

    const auto output_count = filter->output_count.load();
    const auto black_count = filter->black_count.load();
    const auto replay_count = filter->replay_count.load();
    const auto processed_count = output_count >= black_count + replay_count
                                     ? output_count - black_count - replay_count
                                     : 0;
    blog(LOG_INFO,
     "OBS_FaceMosaic diagnostic: model=%s device=%s adapter=%s manual_mask=%d "
         "model_loads=%llu generation=%llu input=%ux%u format=%s(%d) output=%llu processed=%llu "
         "replayed=%llu output_fps=%llu black=%llu drop=%llu pending=%llu pending_limit=%llu "
         "pending_source_bytes=%llu input_interval_us=%llu last_latency_ms=%lld "
         "processor_state=%d submitted=%llu completed=%llu rejected=%llu "
         "queue_evictions=%llu deadline_drops=%llu detector_failures=%llu status=%s",
         settings.model.c_str(), settings.use_cpu ? "CPU" : "DirectML",
         adapter.c_str(),
         settings.manual_mask ? 1 : 0,
         static_cast<unsigned long long>(detector_load_count),
         static_cast<unsigned long long>(generation), input_width, input_height,
         get_video_format_name(input_format), static_cast<int>(input_format),
         static_cast<unsigned long long>(output_count),
         static_cast<unsigned long long>(processed_count),
         static_cast<unsigned long long>(replay_count),
         static_cast<unsigned long long>(filter->output_fps.load()),
         static_cast<unsigned long long>(black_count),
         static_cast<unsigned long long>(filter->drop_count.load()),
         static_cast<unsigned long long>(filter->pending_count.load()),
         static_cast<unsigned long long>(filter->latest_pending_limit.load()),
         static_cast<unsigned long long>(filter->latest_pending_bytes.load()),
         static_cast<unsigned long long>(filter->latest_input_interval_us.load()),
         static_cast<long long>(filter->latest_latency_ms.load()), processor_state,
         static_cast<unsigned long long>(submitted),
         static_cast<unsigned long long>(completed),
         static_cast<unsigned long long>(rejected),
         static_cast<unsigned long long>(queue_evictions),
         static_cast<unsigned long long>(deadline_drops),
         static_cast<unsigned long long>(detector_failures),
         status.c_str());

    // The legacy diagnostic line logs immediately once at startup. Keep the
    // first metrics window intact so the first timing report spans >=5s.
    if (previous_ms != 0) {
      const auto window = filter->performance_metrics->take_window();
      const auto append_stage = [](
        std::ostringstream& line, const char* name,
        const obs_face_mosaic::diagnostics::StageWindow& stage) {
        line << ' ' << name << "{n=" << stage.samples << ",f=" << stage.failures;
        if (stage.samples != 0) {
          line << ",p50_us=" << stage.p50_us << ",p95_us=" << stage.p95_us
               << ",p99_us=" << stage.p99_us;
        } else {
          line << ",p50_us=-,p95_us=-,p99_us=-";
        }
        line << '}';
      };
      std::ostringstream timing;
      timing << "OBS_FaceMosaic timing: window_ms=" << now_ms - previous_ms
             << " out_processed="
             << window.outputs[static_cast<std::size_t>(DiagnosticOutput::processed)]
             << " out_replay="
             << window.outputs[static_cast<std::size_t>(DiagnosticOutput::replayed)]
             << " out_black="
             << window.outputs[static_cast<std::size_t>(DiagnosticOutput::black)]
             << " out_drop="
             << window.outputs[static_cast<std::size_t>(DiagnosticOutput::dropped)];
      append_stage(timing, "input_to_rgba",
                   window.stages[static_cast<std::size_t>(Stage::input_to_rgba)]);
      append_stage(timing, "core_queue_wait",
                   window.stages[static_cast<std::size_t>(Stage::core_queue_wait)]);
      append_stage(timing, "core_success",
                   window.stages[static_cast<std::size_t>(Stage::core_processing_success)]);
      append_stage(timing, "core_deadline",
                   window.stages[static_cast<std::size_t>(Stage::core_processing_deadline)]);
      append_stage(timing, "detector_pre",
                   window.stages[static_cast<std::size_t>(Stage::detector_preprocess)]);
      append_stage(timing, "onnx_run",
                   window.stages[static_cast<std::size_t>(Stage::onnx_run)]);
      append_stage(timing, "detector_post",
                   window.stages[static_cast<std::size_t>(Stage::detector_postprocess)]);
      append_stage(timing, "mosaic",
                   window.stages[static_cast<std::size_t>(Stage::mosaic)]);
      append_stage(timing, "return_processed",
                   window.stages[static_cast<std::size_t>(Stage::obs_return_processed)]);
      append_stage(timing, "return_replay",
                   window.stages[static_cast<std::size_t>(Stage::obs_return_replay)]);
      append_stage(timing, "return_black",
                   window.stages[static_cast<std::size_t>(Stage::obs_return_black)]);
      append_stage(timing, "write_processed",
                   window.stages[static_cast<std::size_t>(Stage::obs_write_processed)]);
      append_stage(timing, "write_replay",
                   window.stages[static_cast<std::size_t>(Stage::obs_write_replay)]);
      append_stage(timing, "write_black",
                   window.stages[static_cast<std::size_t>(Stage::obs_write_black)]);
      blog(LOG_INFO, "%s", timing.str().c_str());
    }
  } catch (...) {
    // Diagnostics are best-effort and must never change frame ownership or
    // interfere with the filter's fail-closed output path.
  }
}

obs_source_frame* filter_video_callback(void* data, obs_source_frame* input) {
  auto* filter = static_cast<FilterData*>(data);
  if (!filter || !input) {
    return nullptr;
  }
  // Copy metadata before the callback can return or transfer the retained
  // OBS frame. Logging never dereferences an input frame after processing.
  const auto input_width = input->width;
  const auto input_height = input->height;
  const auto input_format = input->format;
  auto* parent = obs_filter_get_parent(filter->context);
  try {
    log_diagnostics(filter, input_width, input_height, input_format);
    return process_video(filter, input);
  } catch (const std::exception&) {
    blog(LOG_ERROR, "OBS_FaceMosaic: frame processing failed; raw frame blocked");
  } catch (...) {
    blog(LOG_ERROR, "OBS_FaceMosaic: unknown frame failure; raw frame blocked");
  }

  // Keep a held frame only if it was already registered in the bounded queue;
  // otherwise release it now. Never return it without a complete overwrite.
  std::lock_guard<std::mutex> lock(filter->video_mutex);
  const auto pending = std::find_if(filter->pending.begin(), filter->pending.end(),
                                   [input](const PendingFrame& frame) {
    return frame.source_frame == input;
  });
  if (pending != filter->pending.end()) {
    pending->black_cause = BlackCause::other;
    pending->force_black = true;
    pending->result.reset();
    try {
      std::unique_lock<std::mutex> output_gate(filter->output_commit_mutex);
      Settings settings;
      std::uint64_t generation = 0;
      {
        std::lock_guard<std::mutex> control_lock(filter->control->mutex);
        settings = filter->control->settings;
        generation = filter->control->generation;
      }
      if (pending->key.generation != generation) {
        output_gate.unlock();
        release_all_pending_from_callback(filter, parent);
        return nullptr;
      }
      const auto runtime = filter->runtime();
      const auto emitted = emit_front(
          filter, input->timestamp, settings, runtime,
          generation, settings.manual_mask, true);
      output_gate.unlock();
      if (emitted.release_after_gate) {
        release_frame_from_filter_callback(parent, emitted.release_after_gate);
      }
      if (emitted.output) {
        return emitted.output;
      }
    } catch (...) {
      // The rescue conversion may fail too. Drop every still-owned frame so
      // none can escape unprocessed and none is left for filter_remove to
      // release a second time.
      release_all_pending_from_callback(filter, parent);
    }
  } else {
    release_frame_from_filter_callback(parent, input);
  }
  return nullptr;
}

void filter_add_callback(void* data, obs_source_t* parent) {
  auto* filter = static_cast<FilterData*>(data);
  if (!filter || !parent) {
    return;
  }
  std::lock_guard<std::mutex> lock(filter->video_mutex);
  filter->disconnect_lifecycle_signals_locked();
  if (filter->weak_parent) {
    obs_weak_source_release(filter->weak_parent);
  }
  filter->weak_parent = obs_source_get_weak_source(parent);

  const char* parent_id = obs_source_get_id(parent);
  if (!parent_id || std::string(parent_id) != "ffmpeg_source") {
    return;
  }

  auto* handler = obs_source_get_signal_handler(parent);
  if (!handler) {
    return;
  }
  signal_handler_connect_ref(handler, "media_stopped", source_media_stopped, filter);
  signal_handler_connect_ref(handler, "media_ended", source_media_ended, filter);
  signal_handler_connect_ref(handler, "media_restart", source_media_restart, filter);
  signal_handler_connect_ref(handler, "media_started", source_media_started, filter);
  signal_handler_connect_ref(handler, "show", source_shown, filter);
  signal_handler_connect_ref(handler, "hide", source_hidden, filter);
  filter->parent_signal_handler = handler;
}

void filter_remove_callback(void* data, obs_source_t* parent) {
  auto* filter = static_cast<FilterData*>(data);
  if (!filter) {
    return;
  }
  obs_source_t* release_parent = parent;
  obs_source_t* transient_parent = nullptr;
  obs_weak_source_t* weak_parent_to_release = nullptr;
  std::deque<PendingFrame> frames_to_release;
  bool pending_without_parent = false;
  {
    std::lock_guard<std::mutex> lock(filter->video_mutex);
    filter->disconnect_lifecycle_signals_locked();
    if (!release_parent && filter->weak_parent) {
      transient_parent = obs_weak_source_get_source(filter->weak_parent);
      release_parent = transient_parent;
    }
    if (release_parent) {
      frames_to_release = filter->detach_pending();
    } else if (!filter->pending.empty()) {
      pending_without_parent = true;
    }
    weak_parent_to_release = std::exchange(filter->weak_parent, nullptr);
  }
  if (release_parent) {
    for (const auto& frame : frames_to_release) {
      if (frame.source_frame) {
        obs_source_release_frame(release_parent, frame.source_frame);
      }
    }
  } else if (pending_without_parent) {
    blog(LOG_ERROR, "OBS_FaceMosaic: cannot release retained frames after parent expiration");
  }
  if (transient_parent) {
    obs_source_release(transient_parent);
  }
  if (weak_parent_to_release) {
    obs_weak_source_release(weak_parent_to_release);
  }
}

void filter_defaults_wrapper(obs_data_t* settings) {
  filter_defaults(settings);
}

obs_source_info filter_info = {
    .id = kFilterId,
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC,
    .get_name = filter_name,
    .create = filter_create_wrapper,
    .destroy = filter_destroy_wrapper,
    .get_width = filter_width,
    .get_height = filter_height,
    .get_defaults = filter_defaults_wrapper,
    .get_properties = filter_properties_wrapper,
    .update = filter_update_wrapper,
    .video_tick = filter_video_tick,
    .video_render = filter_video_render,
    .filter_video = filter_video_callback,
    .filter_remove = filter_remove_callback,
    .video_get_color_space = filter_video_get_color_space,
    .filter_add = filter_add_callback,
};

}  // namespace

}  // namespace obs_face_mosaic::obs_plugin

OBS_DECLARE_MODULE()

bool obs_module_load(void) {
  obs_face_mosaic::obs_plugin::RuntimeReaper::instance().start();
  obs_register_source(&obs_face_mosaic::obs_plugin::filter_info);
  return true;
}

void obs_module_unload(void) {
  obs_face_mosaic::obs_plugin::RuntimeReaper::instance().stop_and_drain();
}
