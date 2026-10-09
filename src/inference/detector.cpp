#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#if defined(_WIN32)
#include <windows.h>
#endif

#include "inference/detector.hpp"
#include "inference/preprocess.hpp"

#include <dml_provider_factory.h>
#include <dxgi1_6.h>
#include <onnxruntime_cxx_api.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cwchar>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace obs_face_mosaic::inference {
namespace {

constexpr int kModelClassCount = 2;
constexpr std::size_t kCandidateCount = 8400;

struct Candidate {
  core::FaceBox box;
  int class_id{};
  std::size_t source_index{};
};

void require_tensor(const Ort::TypeInfo& type_info, const std::vector<std::int64_t>& expected_shape,
                    const char* name) {
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    throw std::runtime_error(std::string("Model ") + name + " must be a tensor.");
  }

  const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(std::string("Model ") + name + " must use float32 values.");
  }
  if (tensor_info.GetShape() != expected_shape) {
    throw std::runtime_error(std::string("Model ") + name + " has an unsupported tensor shape.");
  }
}

std::string metadata_value(const Ort::ModelMetadata& metadata, const char* key, OrtAllocator* allocator) {
  auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
  return value ? std::string(value.get()) : std::string{};
}

float intersection_over_union(const core::FaceBox& lhs, const core::FaceBox& rhs) {
  const float left = std::max(lhs.x, rhs.x);
  const float top = std::max(lhs.y, rhs.y);
  const float right = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
  const float bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
  const float intersection = std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
  const float union_area = (lhs.width * lhs.height) + (rhs.width * rhs.height) - intersection;
  return union_area > 0.0F ? intersection / union_area : 0.0F;
}

std::vector<core::FaceBox> suppress_overlaps(std::vector<Candidate> candidates, float threshold) {
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
    if (lhs.box.score == rhs.box.score) {
      return lhs.source_index < rhs.source_index;
    }
    return lhs.box.score > rhs.box.score;
  });

  std::vector<bool> suppressed(candidates.size(), false);
  std::vector<core::FaceBox> boxes;
  boxes.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    boxes.push_back(candidates[i].box);
    for (std::size_t j = i + 1U; j < candidates.size(); ++j) {
      if (!suppressed[j] && candidates[i].class_id == candidates[j].class_id &&
          intersection_over_union(candidates[i].box, candidates[j].box) > threshold) {
        suppressed[j] = true;
      }
    }
  }
  return boxes;
}

#if defined(_WIN32)
std::string dml_adapter_name(const OrtDmlApi* directml_api,
                             Ort::SessionOptions& options) {
  if (!directml_api) {
    return "不明";
  }

  IDMLDevice* dml_device = nullptr;
  OrtStatus* status = directml_api->GetDMLDevice(options, &dml_device);
  if (status) {
    Ort::GetApi().ReleaseStatus(status);
    return "不明";
  }
  if (!dml_device) {
    return "不明";
  }

  Microsoft::WRL::ComPtr<ID3D12Device> d3d12_device;
  if (FAILED(dml_device->GetParentDevice(IID_PPV_ARGS(&d3d12_device)))) {
    return "不明";
  }
  const LUID luid = d3d12_device->GetAdapterLuid();
  const std::string luid_text = "LUID=" + std::to_string(luid.HighPart) + ":" +
                                std::to_string(luid.LowPart);

  Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
    return "不明 (" + luid_text + ")";
  }
  Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
  if (FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) {
    return "不明 (" + luid_text + ")";
  }
  DXGI_ADAPTER_DESC description{};
  if (FAILED(adapter->GetDesc(&description))) {
    return "不明 (" + luid_text + ")";
  }

  const int wide_length = static_cast<int>(std::wcslen(description.Description));
  const int utf8_length = WideCharToMultiByte(CP_UTF8, 0, description.Description,
                                               wide_length, nullptr, 0,
                                               nullptr, nullptr);
  if (utf8_length <= 0) {
    return "不明 (" + luid_text + ")";
  }
  std::string name(static_cast<std::size_t>(utf8_length), '\0');
  if (WideCharToMultiByte(CP_UTF8, 0, description.Description, wide_length,
                          name.data(), utf8_length, nullptr, nullptr) <= 0) {
    return "不明 (" + luid_text + ")";
  }
  return name + " (" + luid_text + ")";
}
#endif

}  // namespace

std::vector<GpuAdapter> available_gpu_adapters() {
  Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
    throw DeviceSelectionError("GPU adapter enumeration failed.");
  std::vector<GpuAdapter> result;
  for (UINT index = 0; ; ++index) {
    Microsoft::WRL::ComPtr<IDXGIAdapter1> gpu;
    const auto status = factory->EnumAdapters1(index, &gpu);
    if (status == DXGI_ERROR_NOT_FOUND) break;
    if (FAILED(status)) throw DeviceSelectionError("GPU adapter enumeration failed.");
    DXGI_ADAPTER_DESC1 desc{};
    if (FAILED(gpu->GetDesc1(&desc))) throw DeviceSelectionError("GPU adapter description failed.");
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
    const int length = static_cast<int>(std::wcslen(desc.Description));
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, desc.Description, length, nullptr, 0, nullptr, nullptr);
    std::string name(static_cast<std::size_t>(std::max(bytes, 0)), '\0');
    if (bytes > 0) WideCharToMultiByte(CP_UTF8, 0, desc.Description, length, name.data(), bytes, nullptr, nullptr);
    result.push_back({std::to_string(desc.AdapterLuid.HighPart) + ":" +
                        std::to_string(desc.AdapterLuid.LowPart), std::move(name), static_cast<int>(index)});
  }
  Microsoft::WRL::ComPtr<IDXGIFactory6> preferences;
  Microsoft::WRL::ComPtr<IDXGIAdapter1> preferred;
  if (SUCCEEDED(factory.As(&preferences)) &&
      SUCCEEDED(preferences->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                       IID_PPV_ARGS(&preferred)))) {
    DXGI_ADAPTER_DESC1 desc{};
    if (SUCCEEDED(preferred->GetDesc1(&desc))) {
      const auto id = std::to_string(desc.AdapterLuid.HighPart) + ":" + std::to_string(desc.AdapterLuid.LowPart);
      const auto found = std::find_if(result.begin(), result.end(), [&](const auto& gpu) { return gpu.id == id; });
      if (found != result.end()) std::rotate(result.begin(), found, found + 1);
    }
  }
  return result;
}

struct Detector::Impl {
  explicit Impl(const DetectorConfig& config)
      : env(ORT_LOGGING_LEVEL_WARNING, "OBS_FaceMosaic"), device(config.device), score_threshold(config.score_threshold),
        nms_iou_threshold(config.nms_iou_threshold), class_ids(config.class_ids),
        performance_metrics(config.performance_metrics),
        adapter(device == Device::cpu ? "CPU" : "不明") {
    if (config.model_path.empty()) {
      throw std::invalid_argument("A model path is required.");
    }
    if (!std::filesystem::is_regular_file(config.model_path)) {
      throw std::invalid_argument("The selected model file does not exist.");
    }
    if (!std::isfinite(score_threshold) || score_threshold < 0.0F || score_threshold > 1.0F) {
      throw std::invalid_argument("The score threshold must be finite and within [0, 1].");
    }
    if (!std::isfinite(nms_iou_threshold) || nms_iou_threshold < 0.0F || nms_iou_threshold > 1.0F) {
      throw std::invalid_argument("The NMS IoU threshold must be finite and within [0, 1].");
    }
    if (class_ids.empty()) {
      throw std::invalid_argument("At least one model class must be selected.");
    }
    std::unordered_set<int> unique_class_ids;
    for (int class_id : class_ids) {
      if (class_id < 0 || class_id >= kModelClassCount || !unique_class_ids.insert(class_id).second) {
        throw std::invalid_argument("Selected class ids must be unique values from this model's [0, 1] classes.");
      }
    }

    auto verified_model = read_verified_model(config.model_path);
    model_hash = verified_model.sha256;
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.DisableMemPattern();

    if (device == Device::directml) {
#if defined(_WIN32)
      const void* provider_api = nullptr;
      Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, &provider_api));
      const auto* directml_api = static_cast<const OrtDmlApi*>(provider_api);
      const auto adapters = available_gpu_adapters();
      if (adapters.empty()) throw DeviceSelectionError("No hardware GPU is available.");
      const auto selected_id = config.gpu_adapter.empty() ? adapters.front().id : config.gpu_adapter;
      const auto selected = std::find_if(adapters.begin(), adapters.end(),
          [&](const auto& gpu) { return gpu.id == selected_id; });
      if (selected == adapters.end()) throw DeviceSelectionError("The selected GPU is unavailable; select a GPU again.");
      // ORT 1.24.4 defines this index as IDXGIFactory::EnumAdapters order.
      Ort::ThrowOnError(directml_api->SessionOptionsAppendExecutionProvider_DML(options, selected->device_index));
      adapter = dml_adapter_name(directml_api, options);
      if (adapter.find("(LUID=" + selected->id + ")") == std::string::npos)
        throw DeviceSelectionError("DirectML GPU identity could not be verified.");
#else
      throw std::runtime_error("DirectML is available only on Windows.");
#endif
    }

    session = std::make_unique<Ort::Session>(env, verified_model.bytes.data(),
                                            verified_model.bytes.size(), options);
    validate_model_contract();
  }

  void validate_model_contract() {
    Ort::AllocatorWithDefaultOptions allocator;
    if (session->GetInputCount() != 1U || session->GetOutputCount() != 1U) {
      throw std::runtime_error("The face model must have one input and one output.");
    }

    auto input_name = session->GetInputNameAllocated(0U, allocator);
    auto output_name = session->GetOutputNameAllocated(0U, allocator);
    input_name_storage = input_name.get();
    output_name_storage = output_name.get();
    if (input_name_storage != "images" || output_name_storage != "output0") {
      throw std::runtime_error("The selected model has unexpected input or output names.");
    }
    require_tensor(session->GetInputTypeInfo(0U), {1, 3, kInputSize, kInputSize}, "input");
    require_tensor(session->GetOutputTypeInfo(0U), {1, 6, static_cast<std::int64_t>(kCandidateCount)}, "output");

    if (session->GetOpset("") != 19) {
      throw std::runtime_error("The selected face model must use ONNX default-domain opset 19.");
    }
    const Ort::ModelMetadata metadata = session->GetModelMetadata();
    if (metadata_value(metadata, "task", allocator) != "detect") {
      throw std::runtime_error("The selected model is not tagged as a detection model.");
    }
    const std::string class_names = metadata_value(metadata, "names", allocator);
    if (class_names.find("0: 'face'") == std::string::npos ||
        class_names.find("1: 'face_anime'") == std::string::npos) {
      throw std::runtime_error("The selected model has an unsupported class ordering.");
    }
  }

  std::vector<core::FaceBox> detect(const core::Frame& frame) {
    // The input tensor borrows workspace pixels. Keep the same lock through
    // preprocessing, Run and result decoding so another call cannot overwrite
    // them while DirectML is reading them.
    std::lock_guard<std::mutex> lock(run_mutex);
    PreparedInput input;
    std::array<std::int64_t, 4> input_shape{1, 3, kInputSize, kInputSize};
    static const Ort::MemoryInfo cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor;
    {
      diagnostics::ScopedMeasurement measurement(
          performance_metrics.get(), diagnostics::Stage::detector_preprocess);
      input = preprocessor.prepare(frame);
      input_tensor = Ort::Value::CreateTensor<float>(
          // ORT takes a mutable pointer for a borrowed tensor; Run treats this
          // model input as read-only. The workspace view stays const elsewhere.
          cpu_memory_info, const_cast<float*>(input.values.data()), input.values.size(), input_shape.data(),
          input_shape.size());
    }
    const auto& transform = input.transform;
    const char* input_names[] = {input_name_storage.c_str()};
    const char* output_names[] = {output_name_storage.c_str()};
    std::vector<Ort::Value> outputs;
    {
      diagnostics::ScopedMeasurement measurement(
          performance_metrics.get(), diagnostics::Stage::onnx_run);
      outputs = session->Run(Ort::RunOptions{nullptr}, input_names,
                             &input_tensor, 1U, output_names, 1U);
    }

    diagnostics::ScopedMeasurement postprocess_measurement(
        performance_metrics.get(), diagnostics::Stage::detector_postprocess);
    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<std::int64_t> output_shape = output_info.GetShape();
    if (output_shape != std::vector<std::int64_t>{1, 6, static_cast<std::int64_t>(kCandidateCount)}) {
      throw std::runtime_error("Runtime output shape does not match the validated face model contract.");
    }
    const float* output = outputs[0].GetTensorData<float>();
    std::vector<Candidate> candidates;
    for (std::size_t index = 0; index < kCandidateCount; ++index) {
      const float center_x = output[index];
      const float center_y = output[kCandidateCount + index];
      const float width = output[(kCandidateCount * 2U) + index];
      const float height = output[(kCandidateCount * 3U) + index];
      if (!std::isfinite(center_x) || !std::isfinite(center_y) || !std::isfinite(width) || !std::isfinite(height) ||
          width <= 0.0F || height <= 0.0F) {
        continue;
      }

      for (const int class_id : class_ids) {
        const float score = output[(kCandidateCount * static_cast<std::size_t>(4 + class_id)) + index];
        if (!std::isfinite(score) || score < score_threshold) {
          continue;
        }
        const float left = (center_x - (width * 0.5F) - static_cast<float>(transform.pad_left)) / transform.scale_x;
        const float top = (center_y - (height * 0.5F) - static_cast<float>(transform.pad_top)) / transform.scale_y;
        const float right = (center_x + (width * 0.5F) - static_cast<float>(transform.pad_left)) / transform.scale_x;
        const float bottom = (center_y + (height * 0.5F) - static_cast<float>(transform.pad_top)) / transform.scale_y;
        const float clipped_left = std::clamp(left, 0.0F, static_cast<float>(frame.width));
        const float clipped_top = std::clamp(top, 0.0F, static_cast<float>(frame.height));
        const float clipped_right = std::clamp(right, 0.0F, static_cast<float>(frame.width));
        const float clipped_bottom = std::clamp(bottom, 0.0F, static_cast<float>(frame.height));
        if (clipped_right <= clipped_left || clipped_bottom <= clipped_top) {
          continue;
        }
        candidates.push_back(Candidate{core::FaceBox{clipped_left, clipped_top, clipped_right - clipped_left,
                                                      clipped_bottom - clipped_top, score},
                                       class_id, index});
      }
    }
    return suppress_overlaps(std::move(candidates), nms_iou_threshold);
  }

  Ort::Env env;
  Device device;
  float score_threshold;
  float nms_iou_threshold;
  std::vector<int> class_ids;
  std::unique_ptr<Ort::Session> session;
  std::string input_name_storage;
  std::string output_name_storage;
  std::shared_ptr<diagnostics::PerformanceMetrics> performance_metrics;
  std::string adapter;
  std::string model_hash;
  Preprocessor preprocessor;
  std::mutex run_mutex;
};

Detector::Detector(const DetectorConfig& config) : impl_(std::make_unique<Impl>(config)) {}

Detector::~Detector() = default;

std::vector<core::FaceBox> Detector::detect(const core::Frame& frame) { return impl_->detect(frame); }

Device Detector::device() const noexcept { return impl_->device; }

const std::string& Detector::adapter_name() const noexcept { return impl_->adapter; }
const std::string& Detector::model_sha256() const noexcept { return impl_->model_hash; }

}  // namespace obs_face_mosaic::inference
