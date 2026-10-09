#pragma once

#include "core/frame.hpp"

namespace obs_face_mosaic::core {

struct MosaicOptions {
  float expansion_ratio{0.20F};
  int long_edge_cells{8};
};

// Applies the configured pixelation in place. Invalid face boxes are ignored;
// an invalid frame layout is rejected with std::invalid_argument.
void apply_face_mosaic(Frame& frame, const std::vector<FaceBox>& faces,
                       const MosaicOptions& options = {});

// Produces an opaque RGBA8 black frame with the requested identity and size.
Frame make_black_frame(int width, int height, std::uint64_t generation,
                       std::uint64_t id, std::uint64_t pts_ns);

}  // namespace obs_face_mosaic::core
