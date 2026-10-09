#include "obs/settings.hpp"
#include "inference/detector.hpp"
#include <obs-module.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string_view>

namespace obs_face_mosaic::obs_plugin {

namespace {
std::string default_model_directory() noexcept {
  try {
    const auto* data_path = obs_get_module_data_path(obs_current_module());
    if (!data_path || !data_path[0]) return {};
    // OBS supplies UTF-8 paths, including for portable/custom installations.
    // Do not require the directory to exist before showing the default in UI.
    const auto path = std::filesystem::absolute(
        std::filesystem::u8path(data_path) / "models").lexically_normal().generic_u8string();
    return {reinterpret_cast<const char*>(path.data()), path.size()};
  } catch (...) {
    return {};  // The loader keeps output black when the path cannot be resolved.
  }
}
}  // namespace

const char* model_filename(const Settings& settings) noexcept {
  if (settings.model == "n-fp16") return "face-mosaic-lite-fp16.onnx";
  if (settings.model == "m-fp16") return "face-mosaic-detail-fp16.onnx";
  return settings.model == "m" ? "face-mosaic-detail-fp32.onnx" : "face-mosaic-lite-fp32.onnx";
}

Settings read_settings(obs_data_t* data) {
  Settings settings;
  const char* model = obs_data_get_string(data, "model");
  settings.model = model && model[0] == 'm' ? "m" : "n";
  // Preserve saved precision choices, including legacy n/m (FP32).
  if (model && (std::string_view(model) == "n-fp16" || std::string_view(model) == "m-fp16"))
    settings.model = model;
  if (!model || !model[0]) settings.model = "n-fp16";
  const char* directory = obs_data_get_string(data, "model_directory");
  if (directory) {
    settings.model_directory = directory;
  }
  settings.use_cpu = obs_data_get_bool(data, "use_cpu");
  settings.gpu_adapter = obs_data_get_string(data, "gpu_adapter");
  settings.score_threshold = std::clamp(
      static_cast<float>(obs_data_get_double(data, "score_threshold")), 0.01F, 0.90F);
  settings.nms_threshold = std::clamp(
      static_cast<float>(obs_data_get_double(data, "nms_threshold")), 0.10F, 0.90F);
  settings.expansion_ratio = std::clamp(
      static_cast<float>(obs_data_get_double(data, "expansion_ratio")), 0.0F, 1.0F);
  settings.mosaic_cells = static_cast<int>(std::clamp<std::int64_t>(
      obs_data_get_int(data, "mosaic_cells"), 2, 32));
  settings.latency_ms = static_cast<int>(std::clamp<std::int64_t>(
      obs_data_get_int(data, "latency_ms"), 50, 1000));
  settings.manual_mask = obs_data_get_bool(data, "manual_mask");
  return settings;
}

void filter_defaults(obs_data_t* settings) {
  obs_data_set_default_string(settings, "model", "n-fp16");
  const auto directory = default_model_directory();
  obs_data_set_default_string(settings, "model_directory", directory.c_str());
  obs_data_set_default_bool(settings, "use_cpu", false);
  obs_data_set_default_string(settings, "gpu_adapter", "");
  obs_data_set_default_double(settings, "score_threshold", 0.25);
  obs_data_set_default_double(settings, "nms_threshold", 0.45);
  obs_data_set_default_double(settings, "expansion_ratio", 0.20);
  obs_data_set_default_int(settings, "mosaic_cells", 8);
  obs_data_set_default_int(settings, "latency_ms", 150);
  obs_data_set_default_bool(settings, "manual_mask", false);
  obs_data_set_default_int(settings, "schema_version", 3);
}

void migrate_settings(obs_data_t* settings) {
  // Empty saved selections mean automatic placement. Keep the default-only
  // value out of JSON so moving OBS does not retain an obsolete absolute path.
  if (!obs_data_get_string(settings, "model_directory")[0])
    obs_data_unset_user_value(settings, "model_directory");
  if (!obs_data_has_user_value(settings, "model")) {
    // OBS omits default-only values from saved JSON. Older schema versions
    // therefore need their old default pinned before applying today's default.
    const bool legacy = obs_data_has_user_value(settings, "schema_version") &&
                        obs_data_get_int(settings, "schema_version") < 3;
    obs_data_set_string(settings, "model", legacy ? "n" : "n-fp16");
  }
  if (!obs_data_get_bool(settings, "use_cpu") &&
      !obs_data_has_user_value(settings, "gpu_adapter")) {
    // One-time migration: persist a concrete GPU rather than resolving the
    // automatic preference on every load. Missing saved GPUs never migrate.
    try {
      const auto adapters = obs_face_mosaic::inference::available_gpu_adapters();
      if (!adapters.empty()) {
        obs_data_set_string(settings, "gpu_adapter", adapters.front().id.c_str());
        blog(LOG_INFO, "OBS_FaceMosaic: initial GPU selection saved (LUID=%s)", adapters.front().id.c_str());
      }
    } catch (...) { /* Loader reports failure while the output stays black. */ }
  }
  obs_data_set_int(settings, "schema_version", 3);
}

obs_properties_t* create_filter_properties(
    const std::string& selected, const std::string& status,
    obs_property_clicked_t retry_callback, void* retry_data) {
  auto* properties = obs_properties_create();
  auto* model = obs_properties_add_list(properties, "model", "モデル",
                                        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
  obs_property_list_add_string(model, "軽量モデル（GPU負荷を抑える・推奨）", "n-fp16");
  obs_property_list_add_string(model, "検出重視モデル（GPUに余裕がある場合）", "m-fp16");
  obs_property_list_add_string(model, "軽量モデル・互換版（通常版で問題が出る場合）", "n");
  obs_property_list_add_string(model, "検出重視モデル・互換版（通常版で問題が出る場合）", "m");
  obs_properties_add_path(properties, "model_directory", "モデルフォルダー",
                          OBS_PATH_DIRECTORY, nullptr, nullptr);

  auto* device = obs_properties_add_list(properties, "use_cpu", "推論デバイス",
                                         OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_BOOL);
  obs_property_list_add_bool(device, "DirectML GPU", false);
  obs_property_list_add_bool(device, "CPU (診断用)", true);
  auto* gpu = obs_properties_add_list(properties, "gpu_adapter", "DirectML GPU",
                                      OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
  bool found_selected = false;
  try {
    for (const auto& adapter : obs_face_mosaic::inference::available_gpu_adapters()) {
      const auto label = adapter.name + " (LUID=" + adapter.id + ")";
      obs_property_list_add_string(gpu, label.c_str(), adapter.id.c_str());
      found_selected |= adapter.id == selected;
    }
  } catch (...) { /* Keep saved selection visible even when enumeration fails. */ }
  if (!selected.empty() && !found_selected)
    obs_property_list_add_string(gpu, "保存済みGPUが見つかりません（選び直してください）", selected.c_str());
  if (obs_property_list_item_count(gpu) == 0)
    obs_property_list_add_string(gpu, "GPUが見つかりません", "");
  obs_properties_add_float_slider(properties, "score_threshold", "検出スコア閾値",
                                  0.01, 0.90, 0.01);
  obs_properties_add_float_slider(properties, "nms_threshold", "NMS IoU閾値",
                                  0.10, 0.90, 0.01);
  obs_properties_add_float_slider(properties, "expansion_ratio", "顔矩形の拡張率",
                                  0.0, 1.0, 0.01);
  obs_properties_add_int_slider(properties, "mosaic_cells", "モザイク長辺セル数",
                                2, 32, 1);
  obs_properties_add_int_slider(properties, "latency_ms", "処理待ち遅延 L (ms)",
                                 50, 1000, 10);
  obs_properties_add_bool(properties, "manual_mask", "全画面を隠す");
  // obs_get_source_properties requests the type's settings with data == null.
  // Only a live instance can expose runtime counters or reload its model.
  if (retry_data) {
    obs_properties_add_button2(properties, "retry", "モデルを再読み込み",
                               retry_callback, retry_data);
  }
  obs_properties_add_text(properties, "runtime_status", status.c_str(), OBS_TEXT_INFO);
  obs_properties_add_text(properties, "safety_warning",
                           "顔検出の漏れはあり得ます。フィルターを無効化・削除すると未加工映像が表示されます。",
                           OBS_TEXT_INFO);
  obs_properties_add_text(properties, "audio_note",
                           "このフィルターは音声を変更しません。遅延はOBS側の同期オフセットで実測調整してください。",
                           OBS_TEXT_INFO);
  return properties;
}

}  // namespace obs_face_mosaic::obs_plugin
