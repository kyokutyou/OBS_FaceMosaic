#pragma once

#include <obs.h>
#include <string>

namespace obs_face_mosaic::obs_plugin {

struct Settings {
  std::string model{"n-fp16"};
  std::string model_directory;
  bool use_cpu{false};
  std::string gpu_adapter;
  float score_threshold{0.25F};
  float nms_threshold{0.45F};
  float expansion_ratio{0.20F};
  int mosaic_cells{8};
  int latency_ms{150};
  bool manual_mask{false};

  friend bool operator==(const Settings&, const Settings&) = default;
};

Settings read_settings(obs_data_t* data);
const char* model_filename(const Settings& settings) noexcept;
void filter_defaults(obs_data_t* settings);
void migrate_settings(obs_data_t* settings);
// Caller supplies a status snapshot and retry callback; this module owns no
// runtime state and never acquires frame/worker locks.
obs_properties_t* create_filter_properties(
    const std::string& selected_gpu, const std::string& status,
    obs_property_clicked_t retry_callback, void* retry_data);

}  // namespace obs_face_mosaic::obs_plugin
