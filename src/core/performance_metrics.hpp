#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>

namespace obs_face_mosaic::diagnostics {

// The logarithmic histogram uses 16 buckets per power-of-two microsecond
// range. It is fixed-size, allocation-free, and lock-free on supported x64
// Windows builds. Percentiles are reported as the upper edge of a bucket.
enum class Stage : std::size_t {
  input_to_rgba,
  core_queue_wait,
  core_processing_success,
  core_processing_deadline,
  detector_preprocess,
  onnx_run,
  detector_postprocess,
  mosaic,
  obs_return_processed,
  obs_return_replay,
  obs_return_black,
  obs_write_processed,
  obs_write_replay,
  obs_write_black,
  count,
};

enum class Output : std::size_t {
  processed,
  replayed,
  black,
  dropped,
  count,
};

inline constexpr std::size_t kStageCount = static_cast<std::size_t>(Stage::count);
inline constexpr std::size_t kOutputCount = static_cast<std::size_t>(Output::count);
inline constexpr std::size_t kHistogramBucketCount = 512;

struct StageWindow {
  std::uint64_t samples{0};
  std::uint64_t failures{0};
  std::uint64_t p50_us{0};
  std::uint64_t p95_us{0};
  std::uint64_t p99_us{0};
};

struct MetricsWindow {
  std::array<StageWindow, kStageCount> stages{};
  std::array<std::uint64_t, kOutputCount> outputs{};
};

class PerformanceMetrics final {
 public:
  PerformanceMetrics() noexcept {
    for (auto& histogram : histograms_) {
      for (auto& bucket : histogram) {
        bucket.store(0, std::memory_order_relaxed);
      }
    }
    for (auto& failures : failures_) {
      failures.store(0, std::memory_order_relaxed);
    }
    for (auto& output : outputs_) {
      output.store(0, std::memory_order_relaxed);
    }
  }

  PerformanceMetrics(const PerformanceMetrics&) = delete;
  PerformanceMetrics& operator=(const PerformanceMetrics&) = delete;

  void record(Stage stage, std::chrono::nanoseconds duration,
              bool failed = false) noexcept {
    const auto stage_index = static_cast<std::size_t>(stage);
    if (stage_index >= kStageCount) {
      return;
    }
    std::uint64_t nanoseconds = duration.count() > 0
                                    ? static_cast<std::uint64_t>(duration.count())
                                    : 0;
    std::uint64_t microseconds = nanoseconds / 1000U;
    if (nanoseconds % 1000U != 0) {
      ++microseconds;
    }
    if (microseconds == 0) {
      microseconds = 1;
    }
    const auto bucket_index = bucket_for(microseconds);
    histograms_[stage_index][bucket_index].fetch_add(1,
                                                     std::memory_order_relaxed);
    if (failed) {
      failures_[stage_index].fetch_add(1, std::memory_order_relaxed);
    }
  }

  void record_output(Output output) noexcept {
    const auto index = static_cast<std::size_t>(output);
    if (index < kOutputCount) {
      outputs_[index].fetch_add(1, std::memory_order_relaxed);
    }
  }

  MetricsWindow take_window() noexcept {
    MetricsWindow window;
    for (std::size_t stage = 0; stage < kStageCount; ++stage) {
      auto& summary = window.stages[stage];
      std::array<std::uint64_t, kHistogramBucketCount> counts{};
      for (std::size_t bucket = 0; bucket < kHistogramBucketCount; ++bucket) {
        counts[bucket] = histograms_[stage][bucket].exchange(
            0, std::memory_order_relaxed);
        summary.samples += counts[bucket];
      }
      summary.failures = failures_[stage].exchange(0, std::memory_order_relaxed);
      if (summary.samples != 0) {
        summary.p50_us = percentile_upper_bound(counts, summary.samples, 50);
        summary.p95_us = percentile_upper_bound(counts, summary.samples, 95);
        summary.p99_us = percentile_upper_bound(counts, summary.samples, 99);
      }
    }
    for (std::size_t output = 0; output < kOutputCount; ++output) {
      window.outputs[output] = outputs_[output].exchange(
          0, std::memory_order_relaxed);
    }
    return window;
  }

 private:
  static std::size_t bucket_for(std::uint64_t microseconds) noexcept {
    std::uint32_t exponent = 0;
    std::uint64_t base = 1;
    while (exponent < 31 && microseconds >= base * 2U) {
      base *= 2U;
      ++exponent;
    }
    const std::uint64_t remainder = microseconds - base;
    const std::size_t sub_bucket = static_cast<std::size_t>(
        (remainder * 16U) / base);
    return std::min<std::size_t>(
        static_cast<std::size_t>(exponent) * 16U + sub_bucket,
        kHistogramBucketCount - 1U);
  }

  static std::uint64_t percentile_upper_bound(
      const std::array<std::uint64_t, kHistogramBucketCount>& counts,
      std::uint64_t samples, std::uint64_t percentile) noexcept {
    const std::uint64_t target = (samples / 100U) * percentile +
        (((samples % 100U) * percentile + 99U) / 100U);
    std::uint64_t cumulative = 0;
    for (std::size_t bucket = 0; bucket < kHistogramBucketCount; ++bucket) {
      cumulative += counts[bucket];
      if (cumulative >= target) {
        const auto exponent = static_cast<std::uint32_t>(bucket / 16U);
        const auto sub_bucket = static_cast<std::uint64_t>(bucket % 16U);
        const std::uint64_t base = std::uint64_t{1} << exponent;
        const std::uint64_t upper_offset =
            (base * (sub_bucket + 1U) + 15U) / 16U;
        return base + upper_offset - 1U;
      }
    }
    return 0;
  }

  std::array<std::array<std::atomic<std::uint64_t>, kHistogramBucketCount>,
             kStageCount> histograms_{};
  std::array<std::atomic<std::uint64_t>, kStageCount> failures_{};
  std::array<std::atomic<std::uint64_t>, kOutputCount> outputs_{};
};

class ScopedMeasurement final {
 public:
  ScopedMeasurement(PerformanceMetrics* metrics, Stage stage) noexcept
      : metrics_(metrics), stage_(stage),
        started_(metrics ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point{}),
        uncaught_exceptions_(std::uncaught_exceptions()) {}

  ~ScopedMeasurement() {
    if (metrics_) {
      metrics_->record(stage_, std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - started_),
                       failed_ || std::uncaught_exceptions() > uncaught_exceptions_);
    }
  }

  ScopedMeasurement(const ScopedMeasurement&) = delete;
  ScopedMeasurement& operator=(const ScopedMeasurement&) = delete;

  void fail() noexcept { failed_ = true; }

 private:
  PerformanceMetrics* metrics_;
  Stage stage_;
  std::chrono::steady_clock::time_point started_;
  int uncaught_exceptions_;
  bool failed_{false};
};

}  // namespace obs_face_mosaic::diagnostics
