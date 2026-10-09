#pragma once

#include <cstdint>
#include <vector>

namespace obs_face_mosaic::core {

// CPU-owned video frame used across the processing boundary. Pixels are tightly
// packed RGBA8, row-major, with exactly width * height * 4 bytes.
struct Frame {
  std::uint64_t generation{0};
  std::uint64_t id{0};
  std::uint64_t pts_ns{0};
  int width{0};
  int height{0};
  std::vector<std::uint8_t> rgba;
};

// Coordinates are in pixels in the corresponding Frame's coordinate space.
struct FaceBox {
  float x{0.0F};
  float y{0.0F};
  float width{0.0F};
  float height{0.0F};
  float score{0.0F};
};

}  // namespace obs_face_mosaic::core
