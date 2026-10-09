#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace obs_face_mosaic::inference {

class ModelIntegrityError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct VerifiedModel {
  std::vector<std::uint8_t> bytes;
  std::string sha256;
};

// Read once, verify the selected n/m identity, and give ORT these exact bytes.
// Never re-open the path after verification (it may have been replaced).
VerifiedModel read_verified_model(const std::filesystem::path& path);

}  // namespace obs_face_mosaic::inference
