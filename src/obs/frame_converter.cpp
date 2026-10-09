#include "obs/frame_converter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace obs_face_mosaic::obs_plugin {
using core::Frame;

FrameConverter::~FrameConverter() { reset(); }

bool FrameConverter::configure(const obs_source_frame& frame, bool* changed) {
  if (changed) {
    *changed = false;
  }
  ScaleSignature signature;
  if (!make_signature(frame, signature)) {
    reset();
    return false;
  }
  if (signature_ && *signature_ == signature && to_rgba_ && from_rgba_) {
    return true;
  }

  reset();
  video_scale_info source_info{};
  source_info.format = signature.format;
  source_info.width = static_cast<std::uint32_t>(signature.width);
  source_info.height = static_cast<std::uint32_t>(signature.height);
  source_info.range = signature.range;
  source_info.colorspace = signature.colorspace;

  video_scale_info rgba_info{};
  rgba_info.format = VIDEO_FORMAT_RGBA;
  rgba_info.width = source_info.width;
  rgba_info.height = source_info.height;
  rgba_info.range = VIDEO_RANGE_FULL;
  rgba_info.colorspace = signature.colorspace;

  if (video_scaler_create(&to_rgba_, &rgba_info, &source_info,
                          VIDEO_SCALE_FAST_BILINEAR) != VIDEO_SCALER_SUCCESS) {
    reset();
    return false;
  }
  if (video_scaler_create(&from_rgba_, &source_info, &rgba_info,
                          VIDEO_SCALE_FAST_BILINEAR) != VIDEO_SCALER_SUCCESS) {
    reset();
    return false;
  }

  signature_ = signature;
  const auto pixel_count = static_cast<std::size_t>(signature.width) *
                           static_cast<std::size_t>(signature.height);
  if (pixel_count > static_cast<std::size_t>(-1) / 4) {
    reset();
    return false;
  }
  const auto byte_count = pixel_count * 4;
  rgba_.resize(byte_count);
  output_scratch_.resize(byte_count);
  black_.resize(byte_count);
  for (std::size_t offset = 0; offset < byte_count; offset += 4) {
    black_[offset + 0] = 0;
    black_[offset + 1] = 0;
    black_[offset + 2] = 0;
    black_[offset + 3] = 255;
  }
  if (changed) {
    *changed = true;
  }
  return true;
}

ConvertResult FrameConverter::read(const obs_source_frame& frame, Frame& output) {
  if (!configure(frame)) {
    return (frame.trc == VIDEO_TRC_PQ || frame.trc == VIDEO_TRC_HLG)
               ? ConvertResult::unsupported_color
               : ConvertResult::failure;
  }
  output.width = frame.width;
  output.height = frame.height;
  output.rgba.resize(rgba_.size());

  const std::uint8_t* input[MAX_AV_PLANES]{};
  std::uint8_t* destination[MAX_AV_PLANES]{};
  std::uint32_t input_lines[MAX_AV_PLANES]{};
  std::uint32_t destination_lines[MAX_AV_PLANES]{};
  for (std::size_t plane = 0; plane < MAX_AV_PLANES; ++plane) {
    input[plane] = frame.data[plane];
    input_lines[plane] = frame.linesize[plane];
  }
  destination[0] = output.rgba.data();
  destination_lines[0] = static_cast<std::uint32_t>(frame.width) * 4U;
  if (!video_scaler_scale(to_rgba_, destination, destination_lines, input,
                          input_lines)) {
    return ConvertResult::failure;
  }
  if (frame.flip) {
    flip_rows(output.rgba, frame.width, frame.height);
  }
  return ConvertResult::success;
}

bool FrameConverter::write(obs_source_frame& destination, const Frame& source) {
  if (source.width != static_cast<int>(destination.width) ||
      source.height != static_cast<int>(destination.height) ||
      !configure(destination) ||
      source.rgba.size() != rgba_.size()) {
    return false;
  }
  const std::uint8_t* input[MAX_AV_PLANES]{};
  std::uint8_t* output[MAX_AV_PLANES]{};
  std::uint32_t input_lines[MAX_AV_PLANES]{};
  std::uint32_t output_lines[MAX_AV_PLANES]{};

  const std::vector<std::uint8_t>* pixels = &source.rgba;
  if (destination.flip) {
    output_scratch_ = source.rgba;
    flip_rows(output_scratch_, source.width, source.height);
    pixels = &output_scratch_;
  }
  input[0] = pixels->data();
  input_lines[0] = static_cast<std::uint32_t>(source.width) * 4U;
  for (std::size_t plane = 0; plane < MAX_AV_PLANES; ++plane) {
    output[plane] = destination.data[plane];
    output_lines[plane] = destination.linesize[plane];
  }
  return video_scaler_scale(from_rgba_, output, output_lines, input,
                            input_lines);
}

bool FrameConverter::write_black(obs_source_frame& destination) {
  if (!configure(destination)) {
    return false;
  }
  Frame black;
  black.width = destination.width;
  black.height = destination.height;
  black.rgba = black_;
  return write(destination, black);
}

void FrameConverter::flip_rows(std::vector<std::uint8_t>& pixels, int width, int height) {
  const std::size_t row_size = static_cast<std::size_t>(width) * 4;
  if (height <= 1 || pixels.size() < row_size * static_cast<std::size_t>(height)) {
    return;
  }
  for (int top = 0, bottom = height - 1; top < bottom; ++top, --bottom) {
    auto* top_row = pixels.data() + static_cast<std::size_t>(top) * row_size;
    auto* bottom_row = pixels.data() + static_cast<std::size_t>(bottom) * row_size;
    std::swap_ranges(top_row, top_row + row_size, bottom_row);
  }
}

bool FrameConverter::make_signature(const obs_source_frame& frame,
                           ScaleSignature& signature) {
  if (!frame.data[0] || frame.width == 0 || frame.height == 0 ||
      frame.width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      frame.height > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  if (frame.trc == VIDEO_TRC_PQ || frame.trc == VIDEO_TRC_HLG) {
    return false;
  }
  const std::size_t w = frame.width;
  const std::size_t h = frame.height;
  if (w > static_cast<std::size_t>(-1) / h / 4) {
    return false;
  }

  signature.format = frame.format;
  signature.width = static_cast<int>(frame.width);
  signature.height = static_cast<int>(frame.height);
  signature.trc = frame.trc;
  signature.flip = frame.flip;
  signature.range = frame.full_range ? VIDEO_RANGE_FULL : VIDEO_RANGE_PARTIAL;
  signature.colorspace = format_is_yuv(frame.format) ? VIDEO_CS_709 : VIDEO_CS_SRGB;
  if (frame.trc == VIDEO_TRC_SRGB) {
    signature.colorspace = VIDEO_CS_SRGB;
  } else if (format_is_yuv(frame.format)) {
    // OBS frames carry the actual matrix. Match it against the official
    // libobs 601/709 conversion matrices instead of deriving it from width.
    double best_distance = std::numeric_limits<double>::infinity();
    for (const auto candidate : {VIDEO_CS_601, VIDEO_CS_709}) {
      float matrix[16]{};
      float range_min[3]{};
      float range_max[3]{};
      if (!video_format_get_parameters_for_format(candidate, signature.range,
                                                  frame.format, matrix,
                                                  range_min, range_max)) {
        continue;
      }
      double distance = 0.0;
      for (std::size_t index = 0; index < signature.matrix.size(); ++index) {
        distance += std::abs(static_cast<double>(matrix[index]) -
                             static_cast<double>(frame.color_matrix[index]));
      }
      if (distance < best_distance) {
        best_distance = distance;
        signature.colorspace = candidate;
        std::copy(std::begin(matrix), std::end(matrix), signature.matrix.begin());
      }
    }
    if (!std::isfinite(best_distance) || best_distance > 1.0) {
      // A custom/unknown matrix is not silently treated as a confident
      // detection input. The caller will keep the frame fail-closed.
      return false;
    }
    for (std::size_t index = 0; index < signature.matrix.size(); ++index) {
      if (std::abs(static_cast<double>(signature.matrix[index]) -
                   static_cast<double>(frame.color_matrix[index])) > 0.0001) {
        return false;
      }
    }
  }
  if (!format_is_yuv(frame.format)) {
    std::copy(std::begin(frame.color_matrix), std::end(frame.color_matrix),
              signature.matrix.begin());
  }
  return true;
}

void FrameConverter::reset() {
  if (to_rgba_) {
    video_scaler_destroy(to_rgba_);
    to_rgba_ = nullptr;
  }
  if (from_rgba_) {
    video_scaler_destroy(from_rgba_);
    from_rgba_ = nullptr;
  }
  signature_.reset();
}

}  // namespace obs_face_mosaic::obs_plugin
