#include "inference/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace obs_face_mosaic::inference {
namespace {

constexpr float kLetterboxValue = 114.0F / 255.0F;

#if defined(_M_X64) || defined(__SSE2__)
// Read exactly one RGBA pixel, including at the end of a one-pixel image.
// memcpy avoids alignment and aliasing requirements on the byte vector.
__m128 rgba_as_float(const std::uint8_t* pixel) {
  std::int32_t packed;
  std::memcpy(&packed, pixel, sizeof(packed));
  const auto bytes = _mm_cvtsi32_si128(packed);
  const auto words = _mm_unpacklo_epi8(bytes, _mm_setzero_si128());
  return _mm_cvtepi32_ps(_mm_unpacklo_epi16(words, _mm_setzero_si128()));
}
#endif

std::size_t rgba_byte_count(int width, int height) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Frame dimensions must be positive.");
  }

  const auto max_size = std::numeric_limits<std::size_t>::max();
  const auto pixel_width = static_cast<std::size_t>(width);
  const auto pixel_height = static_cast<std::size_t>(height);
  if (pixel_width > max_size / pixel_height || pixel_width * pixel_height > max_size / 4U) {
    throw std::invalid_argument("Frame dimensions overflow the RGBA buffer size.");
  }
  return pixel_width * pixel_height * 4U;
}

LetterboxTransform make_transform(int width, int height) {
  const float gain = std::min(static_cast<float>(kInputSize) / static_cast<float>(width),
                              static_cast<float>(kInputSize) / static_cast<float>(height));
  const int resized_width = std::clamp(static_cast<int>(std::round(width * gain)), 1, kInputSize);
  const int resized_height = std::clamp(static_cast<int>(std::round(height * gain)), 1, kInputSize);
  const int pad_x = kInputSize - resized_width;
  const int pad_y = kInputSize - resized_height;
  const int pad_left = static_cast<int>(std::round(static_cast<float>(pad_x) * 0.5F - 0.1F));
  const int pad_top = static_cast<int>(std::round(static_cast<float>(pad_y) * 0.5F - 0.1F));
  return LetterboxTransform{resized_width,
                            resized_height,
                            pad_left,
                            pad_top,
                            static_cast<float>(resized_width) / static_cast<float>(width),
                            static_cast<float>(resized_height) / static_cast<float>(height)};
}

}  // namespace

PreparedInput Preprocessor::prepare(const core::Frame& frame) {
  if (frame.rgba.size() != rgba_byte_count(frame.width, frame.height)) {
    throw std::invalid_argument("Frame RGBA byte count does not match its dimensions.");
  }
  constexpr std::size_t plane_size = static_cast<std::size_t>(kInputSize) * kInputSize;
  if (source_width_ != frame.width || source_height_ != frame.height) {
    // Same rounding, half-pixel centers and edge clamping as the scalar path.
    // Allocate first so a failed allocation cannot publish a partial cache.
    input_.resize(plane_size * 3U);
    const auto transform = make_transform(frame.width, frame.height);
    const auto build_axis = [](auto& axis, int count, int source_extent,
                               float scale, std::size_t stride) {
      for (int index = 0; index < count; ++index) {
        const float source = (static_cast<float>(index) + 0.5F) / scale - 0.5F;
        const float clamped = std::clamp(source, 0.0F, static_cast<float>(source_extent - 1));
        const int first = static_cast<int>(std::floor(clamped));
        const int second = std::min(first + 1, source_extent - 1);
        axis[index] = AxisSample{static_cast<std::size_t>(first) * stride,
                                 static_cast<std::size_t>(second) * stride,
                                 clamped - static_cast<float>(first)};
      }
    };
    build_axis(columns_, transform.resized_width, frame.width, transform.scale_x, 4U);
    build_axis(rows_, transform.resized_height, frame.height, transform.scale_y,
               static_cast<std::size_t>(frame.width) * 4U);
    // Only padding persists between same-size frames; every image pixel below
    // is overwritten. Reset all planes when the aspect ratio/dimensions change.
    std::fill(input_.begin(), input_.end(), kLetterboxValue);
    transform_ = transform;
    source_width_ = frame.width;
    source_height_ = frame.height;
  }

  for (int y = 0; y < transform_.resized_height; ++y) {
    const auto& row = rows_[y];
    const auto* top_row = frame.rgba.data() + row.first;
    const auto* bottom_row = frame.rgba.data() + row.second;
    const std::size_t destination = static_cast<std::size_t>(y + transform_.pad_top) *
                                       kInputSize + transform_.pad_left;
#if defined(_M_X64) || defined(__SSE2__)
    const auto vertical_weight = _mm_set1_ps(row.weight);
    const auto normalization = _mm_set1_ps(255.0F);
#endif
    for (int x = 0; x < transform_.resized_width; ++x) {
      const auto& column = columns_[x];
#if defined(_M_X64) || defined(__SSE2__)
      const auto p00 = rgba_as_float(top_row + column.first);
      const auto p10 = rgba_as_float(top_row + column.second);
      const auto p01 = rgba_as_float(bottom_row + column.first);
      const auto p11 = rgba_as_float(bottom_row + column.second);
      const auto horizontal_weight = _mm_set1_ps(column.weight);
      // Preserve the scalar operation order and division; do not approximate
      // normalization or fuse operations. Only the RGB lanes are written.
      const auto top = _mm_add_ps(p00, _mm_mul_ps(_mm_sub_ps(p10, p00), horizontal_weight));
      const auto bottom = _mm_add_ps(p01, _mm_mul_ps(_mm_sub_ps(p11, p01), horizontal_weight));
      const auto rgb = _mm_div_ps(
          _mm_add_ps(top, _mm_mul_ps(_mm_sub_ps(bottom, top), vertical_weight)), normalization);
      _mm_store_ss(input_.data() + destination + x, rgb);
      _mm_store_ss(input_.data() + plane_size + destination + x,
                   _mm_shuffle_ps(rgb, rgb, _MM_SHUFFLE(1, 1, 1, 1)));
      _mm_store_ss(input_.data() + 2U * plane_size + destination + x,
                   _mm_shuffle_ps(rgb, rgb, _MM_SHUFFLE(2, 2, 2, 2)));
#else
      for (std::size_t channel = 0; channel < 3; ++channel) {
        const float p00 = static_cast<float>(top_row[column.first + channel]);
        const float p10 = static_cast<float>(top_row[column.second + channel]);
        const float p01 = static_cast<float>(bottom_row[column.first + channel]);
        const float p11 = static_cast<float>(bottom_row[column.second + channel]);
        const float top = p00 + ((p10 - p00) * column.weight);
        const float bottom = p01 + ((p11 - p01) * column.weight);
        input_[channel * plane_size + destination + x] =
            (top + ((bottom - top) * row.weight)) / 255.0F;
      }
#endif
    }
  }
  return {transform_, input_};
}

}  // namespace obs_face_mosaic::inference
