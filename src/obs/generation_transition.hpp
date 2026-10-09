#pragma once

#include <atomic>
#include <cstdint>

namespace obs_face_mosaic::obs_plugin {

enum class GenerationTransitionAction : std::uint8_t {
  keep_pending,
  discard_pending_and_black_current,
};

constexpr std::uint64_t kMinimumDiscontinuityThresholdNs = 150'000'000ULL;
constexpr std::uint64_t kStableIntervalThresholdMultiplier = 5ULL;
constexpr std::uint64_t kInputStopGenerationThresholdNs = 500'000'000ULL;

constexpr std::uint64_t discontinuity_threshold_ns(
    std::uint64_t stable_input_interval_ns) noexcept {
  const auto relative_threshold =
      stable_input_interval_ns * kStableIntervalThresholdMultiplier;
  return relative_threshold > kMinimumDiscontinuityThresholdNs
             ? relative_threshold
             : kMinimumDiscontinuityThresholdNs;
}

constexpr bool pts_discontinuity_requires_generation_transition(
    std::uint64_t previous_pts_ns,
    std::uint64_t current_pts_ns,
    std::uint64_t stable_input_interval_ns) noexcept {
  return previous_pts_ns != 0 &&
         (current_pts_ns < previous_pts_ns ||
          current_pts_ns - previous_pts_ns >
              discontinuity_threshold_ns(stable_input_interval_ns));
}

constexpr bool input_wall_gap_requires_generation_transition(
    std::uint64_t previous_input_ns,
    std::uint64_t current_input_ns,
    std::uint64_t stable_input_interval_ns) noexcept {
  return previous_input_ns != 0 && current_input_ns > previous_input_ns &&
         current_input_ns - previous_input_ns >
             discontinuity_threshold_ns(stable_input_interval_ns);
}

constexpr bool input_stop_requires_generation_transition(
    std::uint64_t previous_input_ns,
    std::uint64_t current_input_ns) noexcept {
  return previous_input_ns != 0 && current_input_ns > previous_input_ns &&
         current_input_ns - previous_input_ns >
             kInputStopGenerationThresholdNs;
}

// Signals from one source lifecycle boundary can arrive back-to-back (for
// example, media_ended followed by media_started) before the next video
// callback. process_video snapshots the request epoch and acknowledges it only
// after old pending work has been discarded and CoreProcessor reset. Requests
// arriving after that snapshot remain pending for the next callback.
class LifecycleTransitionEpoch final {
 public:
  std::uint64_t request() noexcept {
    return requested_.fetch_add(1, std::memory_order_acq_rel) + 1;
  }

  std::uint64_t requested() const noexcept {
    return requested_.load(std::memory_order_acquire);
  }

  std::uint64_t applied() const noexcept {
    return applied_.load(std::memory_order_acquire);
  }

  bool pending() const noexcept { return requested() != applied(); }

  bool outputs_permitted() const noexcept { return !pending(); }

  void acknowledge(std::uint64_t epoch) noexcept {
    applied_.store(epoch, std::memory_order_release);
  }

 private:
  std::atomic<std::uint64_t> requested_{0};
  std::atomic<std::uint64_t> applied_{0};
};

constexpr GenerationTransitionAction generation_transition_action(
    std::uint64_t active_generation,
    std::uint64_t current_generation) noexcept {
  return active_generation == current_generation
             ? GenerationTransitionAction::keep_pending
             : GenerationTransitionAction::discard_pending_and_black_current;
}

constexpr bool mark_input_stall_generation_transition(
    bool input_stalled,
    bool& transition_already_marked) noexcept {
  if (!input_stalled || transition_already_marked) {
    return false;
  }
  transition_already_marked = true;
  return true;
}

constexpr void rearm_input_stall_generation_transition(
    bool& transition_already_marked) noexcept {
  transition_already_marked = false;
}

template <typename PendingQueue>
PendingQueue detach_pending_for_generation_transition(
    PendingQueue& pending,
    GenerationTransitionAction action) {
  PendingQueue detached;
  if (action == GenerationTransitionAction::discard_pending_and_black_current) {
    detached.swap(pending);
  }
  return detached;
}

}  // namespace obs_face_mosaic::obs_plugin
