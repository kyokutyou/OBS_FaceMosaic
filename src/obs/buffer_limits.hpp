#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace obs_face_mosaic::obs_plugin {

inline constexpr std::uint64_t kNanosecondsPerMillisecond = 1'000'000ULL;
inline constexpr std::uint64_t kDefaultFrameIntervalNs = 33'333'333ULL;
inline constexpr std::uint64_t kMinimumFrameIntervalNs = 1'000'000ULL;
inline constexpr std::uint64_t kMaximumFrameIntervalNs = 1'000'000'000ULL;
inline constexpr std::size_t kMinimumOutstandingFrames = 11;
inline constexpr std::size_t kAbsoluteMaxOutstandingFrames = 64;
inline constexpr std::size_t kMaxOutstandingFrameBytes = 128U * 1024U * 1024U;

inline std::uint64_t clamp_frame_interval_ns(std::uint64_t interval_ns) noexcept {
  return std::clamp(interval_ns, kMinimumFrameIntervalNs,
                    kMaximumFrameIntervalNs);
}

class FrameIntervalEstimator final {
 public:
  void reset() noexcept {
    sample_count_ = 0;
    next_sample_ = 0;
  }

  std::uint64_t observe(std::uint64_t current_pts_ns,
                        std::uint64_t previous_pts_ns,
                        bool previous_valid) noexcept {
    if (!previous_valid || current_pts_ns <= previous_pts_ns) {
      return kDefaultFrameIntervalNs;
    }

    samples_[next_sample_] =
        clamp_frame_interval_ns(current_pts_ns - previous_pts_ns);
    next_sample_ = (next_sample_ + 1U) % samples_.size();
    sample_count_ = std::min(sample_count_ + 1U, samples_.size());

    // The lower median is conservative for buffer sizing: one or two dropped
    // frame intervals cannot suddenly shrink the L-derived source-frame cap.
    auto sorted = samples_;
    std::sort(sorted.begin(), sorted.begin() +
                                  static_cast<std::ptrdiff_t>(sample_count_));
    if (sample_count_ < 3U) {
      return kDefaultFrameIntervalNs;
    }
    return sorted[(sample_count_ - 1U) / 2U];
  }

 private:
  std::array<std::uint64_t, 5> samples_{};
  std::size_t sample_count_{0};
  std::size_t next_sample_{0};
};

inline std::uint64_t stable_frame_interval_from_pts(
    FrameIntervalEstimator& estimator, std::uint64_t current_pts_ns,
    std::uint64_t previous_pts_ns, bool previous_valid,
    bool reset_history) noexcept {
  if (reset_history) {
    estimator.reset();
    return kDefaultFrameIntervalNs;
  }
  return estimator.observe(current_pts_ns, previous_pts_ns, previous_valid);
}

inline std::size_t outstanding_frame_limit(std::uint64_t latency_ms,
                                           std::uint64_t frame_interval_ns) noexcept {
  const auto interval_ns = clamp_frame_interval_ns(frame_interval_ns);
  const auto latency_ns = latency_ms * kNanosecondsPerMillisecond;
  // The two-frame slack covers the boundary between a nominal period and
  // integer nanosecond PTS values (for 30fps, 30 periods are 10ns under 1s).
  const auto frames_for_latency = latency_ns / interval_ns;
  const auto with_slack = frames_for_latency + 2ULL;
  return static_cast<std::size_t>(std::clamp<std::uint64_t>(
      with_slack, kMinimumOutstandingFrames, kAbsoluteMaxOutstandingFrames));
}

}  // namespace obs_face_mosaic::obs_plugin
