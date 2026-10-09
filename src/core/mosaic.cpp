#include "core/mosaic.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace obs_face_mosaic::core {
namespace {

std::size_t checked_pixel_bytes(int width, int height) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("frame dimensions must be positive");
  }

  const auto w = static_cast<std::size_t>(width);
  const auto h = static_cast<std::size_t>(height);
  constexpr std::size_t channels = 4;
  if (w > std::numeric_limits<std::size_t>::max() / h / channels) {
    throw std::invalid_argument("frame dimensions overflow RGBA8 storage");
  }
  return w * h * channels;
}

bool has_valid_pixels(const Frame& frame) {
  return frame.rgba.size() == checked_pixel_bytes(frame.width, frame.height);
}

int bounded_cell_count(int extent, int long_edge_cells) {
  const int maximum = std::min(32, extent);
  return maximum < 2 ? maximum : std::clamp(long_edge_cells, 2, maximum);
}

void pixelate_rect(Frame& frame, int left, int top, int right, int bottom,
                   int long_edge_cells) {
  const int rect_width = right - left;
  const int rect_height = bottom - top;
  if (rect_width <= 0 || rect_height <= 0) {
    return;
  }

  int cells_x = 1;
  int cells_y = 1;
  if (rect_width >= rect_height) {
    cells_x = bounded_cell_count(rect_width, long_edge_cells);
    const double scaled = static_cast<double>(cells_x) * rect_height / rect_width;
    cells_y = std::clamp(static_cast<int>(std::lround(scaled)), 1,
                         std::min(32, rect_height));
  } else {
    cells_y = bounded_cell_count(rect_height, long_edge_cells);
    const double scaled = static_cast<double>(cells_y) * rect_width / rect_height;
    cells_x = std::clamp(static_cast<int>(std::lround(scaled)), 1,
                         std::min(32, rect_width));
  }

  const std::size_t frame_width = static_cast<std::size_t>(frame.width);
  for (int cell_y = 0; cell_y < cells_y; ++cell_y) {
    const int y0 = top + static_cast<int>(
                             (static_cast<std::int64_t>(cell_y) * rect_height) /
                             cells_y);
    const int y1 = top + static_cast<int>(
                             (static_cast<std::int64_t>(cell_y + 1) * rect_height) /
                             cells_y);
    for (int cell_x = 0; cell_x < cells_x; ++cell_x) {
      const int x0 = left + static_cast<int>(
                               (static_cast<std::int64_t>(cell_x) * rect_width) /
                               cells_x);
      const int x1 = left + static_cast<int>(
                               (static_cast<std::int64_t>(cell_x + 1) * rect_width) /
                               cells_x);
      if (x1 <= x0 || y1 <= y0) {
        continue;
      }

      std::uint64_t sums[4] = {0, 0, 0, 0};
      std::uint64_t count = 0;
      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
          const std::size_t offset =
              (static_cast<std::size_t>(y) * frame_width +
               static_cast<std::size_t>(x)) *
              4;
          for (std::size_t channel = 0; channel < 4; ++channel) {
            sums[channel] += frame.rgba[offset + channel];
          }
          ++count;
        }
      }

      std::uint8_t average[4]{};
      for (std::size_t channel = 0; channel < 4; ++channel) {
        average[channel] = static_cast<std::uint8_t>(sums[channel] / count);
      }
      average[3] = 255;
      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
          const std::size_t offset =
              (static_cast<std::size_t>(y) * frame_width +
               static_cast<std::size_t>(x)) *
              4;
          for (std::size_t channel = 0; channel < 4; ++channel) {
            frame.rgba[offset + channel] = average[channel];
          }
        }
      }
    }
  }
}

}  // namespace

void apply_face_mosaic(Frame& frame, const std::vector<FaceBox>& faces,
                       const MosaicOptions& options) {
  if (!has_valid_pixels(frame)) {
    throw std::invalid_argument("frame pixel storage does not match its dimensions");
  }

  const double expansion = std::isfinite(options.expansion_ratio)
                               ? std::clamp(static_cast<double>(options.expansion_ratio),
                                            0.0, 1.0)
                               : 0.20;
  const int cells = std::clamp(options.long_edge_cells, 2, 32);
  const double frame_width = frame.width;
  const double frame_height = frame.height;

  for (const FaceBox& face : faces) {
    if (!std::isfinite(face.x) || !std::isfinite(face.y) ||
        !std::isfinite(face.width) || !std::isfinite(face.height) ||
        !std::isfinite(face.score) ||
        face.width <= 0.0F || face.height <= 0.0F) {
      continue;
    }

    const double box_width = face.width;
    const double box_height = face.height;
    const double left_value = std::clamp(
        static_cast<double>(face.x) - box_width * expansion, 0.0, frame_width);
    const double top_value = std::clamp(
        static_cast<double>(face.y) - box_height * expansion, 0.0, frame_height);
    const double right_value = std::clamp(
        static_cast<double>(face.x) + box_width * (1.0 + expansion), 0.0,
        frame_width);
    const double bottom_value = std::clamp(
        static_cast<double>(face.y) + box_height * (1.0 + expansion), 0.0,
        frame_height);

    const int left = static_cast<int>(std::floor(left_value));
    const int top = static_cast<int>(std::floor(top_value));
    const int right = static_cast<int>(std::ceil(right_value));
    const int bottom = static_cast<int>(std::ceil(bottom_value));
    if (right <= left || bottom <= top) {
      continue;
    }
    pixelate_rect(frame, left, top, right, bottom, cells);
  }
}

Frame make_black_frame(int width, int height, std::uint64_t generation,
                       std::uint64_t id, std::uint64_t pts_ns) {
  const std::size_t bytes = checked_pixel_bytes(width, height);
  Frame frame;
  frame.generation = generation;
  frame.id = id;
  frame.pts_ns = pts_ns;
  frame.width = width;
  frame.height = height;
  frame.rgba.resize(bytes, 0);
  for (std::size_t alpha = 3; alpha < bytes; alpha += 4) {
    frame.rgba[alpha] = 255;
  }
  return frame;
}

}  // namespace obs_face_mosaic::core
