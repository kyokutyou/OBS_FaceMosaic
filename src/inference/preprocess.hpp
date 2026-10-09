#pragma once

#include "core/frame.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace obs_face_mosaic::inference {

inline constexpr int kInputSize = 640;

struct LetterboxTransform {
  int resized_width{};
  int resized_height{};
  int pad_left{};
  int pad_top{};
  float scale_x{};
  float scale_y{};
};

struct PreparedInput {
  LetterboxTransform transform;
  std::span<const float> values;
};

// RGB planar float input for the verified n/m model contract. The returned
// view belongs to this workspace and expires on the next prepare/destruction.
// One detector owns one workspace and serializes prepare + inference together.
class Preprocessor final {
 public:
  PreparedInput prepare(const core::Frame& frame);

 private:
  struct AxisSample {
    std::size_t first{};
    std::size_t second{};
    float weight{};
  };

  int source_width_{};
  int source_height_{};
  LetterboxTransform transform_;
  std::array<AxisSample, kInputSize> columns_{};
  std::array<AxisSample, kInputSize> rows_{};
  std::vector<float> input_;
};

}  // namespace obs_face_mosaic::inference
