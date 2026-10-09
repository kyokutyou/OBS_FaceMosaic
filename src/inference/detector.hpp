#pragma once

#include "core/frame.hpp"
#include "core/performance_metrics.hpp"
#include "inference/model_integrity.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace obs_face_mosaic::inference {

enum class Device {
  directml,
  cpu,
};

struct GpuAdapter {
  std::string id;  // DXGI LUID, stable for this Windows boot.
  std::string name;
  int device_index;
};

class DeviceSelectionError final : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

// Hardware adapters, with the OS high-performance preference first.
// Does not create a model/session or initialize a GPU for inference.
std::vector<GpuAdapter> available_gpu_adapters();

struct DetectorConfig {
  std::filesystem::path model_path;
  Device device{Device::directml};
  // DXGI adapter LUID, not an enumeration index. Empty selects the initial GPU.
  std::string gpu_adapter;
  float score_threshold{0.25F};
  float nms_iou_threshold{0.45F};
  // The checked-in models expose face (0) and face_anime (1). Real human faces
  // are the initial target; callers can opt into additional classes later.
  std::vector<int> class_ids{0};
  std::shared_ptr<diagnostics::PerformanceMetrics> performance_metrics;
};

// Synchronous detector. Create and call it on a dedicated inference worker;
// model loading and Run can take longer than one OBS video callback. Calls on
// one instance are serialized for DirectML compatibility.
class Detector final {
 public:
  explicit Detector(const DetectorConfig& config);
  ~Detector();

  Detector(const Detector&) = delete;
  Detector& operator=(const Detector&) = delete;
  Detector(Detector&&) = delete;
  Detector& operator=(Detector&&) = delete;

  // Returns coordinates in the same pixel space as frame. A successful empty
  // vector means only that this inference produced no selected-class boxes.
  std::vector<core::FaceBox> detect(const core::Frame& frame);

  Device device() const noexcept;
  const std::string& adapter_name() const noexcept;
  const std::string& model_sha256() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace obs_face_mosaic::inference
