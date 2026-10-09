#pragma once

#include <chrono>
#include <cstdint>

namespace obs_face_mosaic::obs_plugin {

enum class BlackCause : std::uint8_t {
  none,
  queue_eviction,
  other,
};

enum class PressureAction : std::uint8_t {
  normal_timing,
  emit_processed,
  replay_if_eligible,
  fail_closed,
};

constexpr PressureAction pressure_action(bool pressure,
                                         bool same_frame_processed,
                                         BlackCause black_cause) noexcept {
  if (!pressure) {
    return PressureAction::normal_timing;
  }
  if (same_frame_processed) {
    return PressureAction::emit_processed;
  }
  if (black_cause == BlackCause::queue_eviction) {
    return PressureAction::replay_if_eligible;
  }
  return PressureAction::fail_closed;
}

struct ReplayCandidate {
  BlackCause cause{BlackCause::none};
  std::uint64_t generation{0};
  std::uint64_t current_generation{0};
  std::uint64_t cached_generation{0};
  int width{0};
  int height{0};
  int cached_width{0};
  int cached_height{0};
  bool cache_available{false};
  bool runtime_running{false};
  bool manual_mask{false};
  std::chrono::steady_clock::duration cache_age{};
};

inline constexpr auto kMaxProcessedReplayAge = std::chrono::milliseconds(500);

constexpr bool can_replay_cached_frame(const ReplayCandidate& candidate) noexcept {
  return candidate.cause == BlackCause::queue_eviction &&
         candidate.generation == candidate.current_generation &&
         candidate.generation == candidate.cached_generation &&
         candidate.width > 0 && candidate.height > 0 &&
         candidate.width == candidate.cached_width &&
         candidate.height == candidate.cached_height &&
         candidate.cache_available && candidate.runtime_running &&
         !candidate.manual_mask &&
         candidate.cache_age >= std::chrono::steady_clock::duration::zero() &&
         candidate.cache_age <= kMaxProcessedReplayAge;
}

}  // namespace obs_face_mosaic::obs_plugin
