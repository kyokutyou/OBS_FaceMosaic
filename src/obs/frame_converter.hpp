#pragma once

#include "core/frame.hpp"
#include <obs.h>
#include <media-io/video-scaler.h>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace obs_face_mosaic::obs_plugin {

enum class ConvertResult { success, unsupported_color, failure };

// Owned by one filter; calls are serialized by its video mutex.
// OBS frame ownership and generation handling remain with the caller.
class FrameConverter final {
 public:
  FrameConverter() = default;
  ~FrameConverter();
  FrameConverter(const FrameConverter&) = delete;
  FrameConverter& operator=(const FrameConverter&) = delete;

  bool configure(const obs_source_frame& frame, bool* changed = nullptr);
  ConvertResult read(const obs_source_frame& frame, core::Frame& output);
  bool write(obs_source_frame& destination, const core::Frame& source);
  bool write_black(obs_source_frame& destination);

 private:
  struct ScaleSignature {
    enum video_format format{VIDEO_FORMAT_NONE};
    int width{0};
    int height{0};
    enum video_colorspace colorspace{VIDEO_CS_709};
    enum video_range_type range{VIDEO_RANGE_PARTIAL};
    std::uint8_t trc{VIDEO_TRC_DEFAULT};
    bool flip{false};
    std::array<float, 16> matrix{};

    friend bool operator==(const ScaleSignature&, const ScaleSignature&) = default;
  };

  static void flip_rows(std::vector<std::uint8_t>& pixels, int width, int height);
  static bool make_signature(const obs_source_frame& frame, ScaleSignature& signature);
  void reset();

  video_scaler_t* to_rgba_{nullptr};
  video_scaler_t* from_rgba_{nullptr};
  std::optional<ScaleSignature> signature_;
  std::vector<std::uint8_t> rgba_;
  std::vector<std::uint8_t> output_scratch_;
  std::vector<std::uint8_t> black_;
};

}  // namespace obs_face_mosaic::obs_plugin
