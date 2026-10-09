#pragma once

#include "core/frame.hpp"
#include "core/mosaic.hpp"
#include "core/performance_metrics.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace obs_face_mosaic::core {

using ProcessorClock = std::chrono::steady_clock;
using DetectFunction = std::function<std::vector<FaceBox>(const Frame&)>;

struct FrameKey {
  std::uint64_t generation{0};
  std::uint64_t id{0};
  std::uint64_t pts_ns{0};
};

enum class OutputKind {
  processed,
  black,
  drop,
};

enum class ProcessReason {
  none,
  recovering,
  manual_mask,
  not_ready,
  stale_generation,
  invalid_frame,
  non_monotonic_id,
  queue_overflow,
  deadline_expired,
  detector_failure,
  processing_failure,
  stale_output,
  awaiting_result,
  stopping,
  no_frame_dimensions,
  output_queue_overflow,
};

enum class ProcessorState {
  initializing,
  awaiting_input,
  recovering,
  running,
  obscured,
  manual_mask,
  stopping,
};

struct RecoveryProgressSnapshot {
  bool generation_set{false};
  std::uint64_t generation{0};
  ProcessorState state{ProcessorState::initializing};
  std::size_t required_successes{0};
  std::size_t consecutive_successes{0};
  // ID of the latest successful frame in the current recovery streak. It is
  // empty before the first success and cleared when the streak is reset.
  std::optional<std::uint64_t> last_successful_id;
};

struct ProcessResult {
  OutputKind kind{OutputKind::drop};
  FrameKey key{};
  // Present only for processed or black outputs. The shared owner keeps pixels
  // alive after poll()/snapshot() returns while preventing caller mutation.
  std::shared_ptr<const Frame> frame;
  ProcessReason reason{ProcessReason::none};
};

struct SubmitResult {
  bool accepted{false};
  FrameKey submitted{};
  // When the bounded waiting queue is full, its oldest waiting frame is
  // discarded and reported here; the in-flight frame is never interrupted.
  std::optional<FrameKey> evicted;
  ProcessReason reason{ProcessReason::none};
};

struct ProcessorConfig {
  std::size_t max_pending_frames{2};
  std::size_t max_completed_results{8};
  std::chrono::milliseconds processing_deadline{150};
  std::chrono::milliseconds last_processed_hold{500};
  std::size_t recovery_successes{3};
  MosaicOptions mosaic{};
  std::shared_ptr<diagnostics::PerformanceMetrics> performance_metrics;
  // Called under the state mutex when a current inference/processing operation
  // fails. Must be nonblocking, nonthrowing, and must not reenter the processor.
  // Lets an adapter invalidate results it has already polled, even without input.
  std::function<void()> on_processing_failure;
};

struct ProcessorMetrics {
  std::uint64_t submitted{0};
  std::uint64_t completed{0};
  std::uint64_t rejected{0};
  std::uint64_t queue_evictions{0};
  std::uint64_t stale_generation_drops{0};
  std::uint64_t deadline_drops{0};
  std::uint64_t detector_failures{0};
  std::uint64_t output_evictions{0};
  std::uint64_t output_rejections{0};
};

class CoreProcessor final {
 public:
  explicit CoreProcessor(DetectFunction detect,
                         ProcessorConfig config = ProcessorConfig{});
  ~CoreProcessor();

  CoreProcessor(const CoreProcessor&) = delete;
  CoreProcessor& operator=(const CoreProcessor&) = delete;
  CoreProcessor(CoreProcessor&&) = delete;
  CoreProcessor& operator=(CoreProcessor&&) = delete;

  // Generation values must strictly increase after the first accepted value.
  // This invalidates queued, in-flight, and completed results from older input.
  bool reset_generation(std::uint64_t generation);
  bool reset_generation(std::uint64_t generation,
                       std::chrono::milliseconds processing_deadline);
  // Changes only the deadline used for future checks. Callers that change a
  // live stream should also invalidate/reset its generation before submitting
  // new frames so old work cannot be judged under the new policy.
  void set_processing_deadline(std::chrono::milliseconds processing_deadline);
  void set_ready(bool ready);
  void set_manual_mask(bool enabled);

  // Nonblocking. The worker owns the submitted pixels. It never returns the
  // input frame; only accepted metadata and any evicted frame key are exposed.
  SubmitResult submit(Frame frame,
                      ProcessorClock::time_point submitted_at =
                          ProcessorClock::now());

  // Returns a completed result for exactly its original generation/id/PTS.
  // drop has no pixels, black contains an opaque image, processed contains the
  // corresponding frame after detection and mosaic.
  std::optional<ProcessResult> poll(ProcessorClock::time_point now =
                                        ProcessorClock::now());

  // Safe draw fallback: returns only the latest result already accepted by
  // poll() from the active generation, for at most last_processed_hold;
  // otherwise returns opaque black for the requested output descriptor. It
  // never returns an input frame or a completed-but-unpolled result.
  ProcessResult snapshot(int width, int height, FrameKey output_key,
                         ProcessorClock::time_point now = ProcessorClock::now());

  ProcessorState state() const;
  ProcessorMetrics metrics() const;

  // Copies generation-scoped recovery counters under the state mutex. This
  // does not wait for inference or expose frame pixels, URLs, or queue storage.
  RecoveryProgressSnapshot recovery_progress() const;

  // Stop prevents new work and joins the worker. A detector call already in
  // progress cannot be force-cancelled by this platform-neutral layer.
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace obs_face_mosaic::core
