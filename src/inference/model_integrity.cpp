#include "inference/model_integrity.hpp"

#include <windows.h>
#include <bcrypt.h>
#include <array>
#include <fstream>

namespace obs_face_mosaic::inference {

VerifiedModel read_verified_model(const std::filesystem::path& path) {
  const auto filename = path.filename();
  const char* expected = nullptr;
  if (filename == "face-mosaic-lite-fp32.onnx")
    expected = "A4DFFE5F031E476186E3EAB59BB0BC1201364DA02FD505A2949C79FD278E30E2";
  else if (filename == "face-mosaic-detail-fp32.onnx")
    expected = "325F267E8F24C22D67179A77AFE8B93D179098E787112F64E4784A4A0AD5D7EB";
  else if (filename == "face-mosaic-lite-fp16.onnx")
    expected = "96BD42768D704814E7386F4C38E056F05BEF2B2C0224F5E02AB8B4A3B6125F76";
  else if (filename == "face-mosaic-detail-fp16.onnx")
    expected = "8FCE465D4AD0F584B1E7E739EFE5E76AC417F5FA5AE7F9240A8A2A9F9E448D53";
  else
    throw ModelIntegrityError("Unsupported model identity; select the verified n or m model.");

  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto size = input.tellg();
  // All supported files are smaller than 81 MB. Bound unexpected allocations.
  if (!input || size <= 0 || size > 128 * 1024 * 1024)
    throw ModelIntegrityError("Model is unreadable, empty, or exceeds the supported size limit.");
  VerifiedModel result;
  result.bytes.resize(static_cast<std::size_t>(size));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(result.bytes.data()), size);
  if (!input || input.peek() != std::char_traits<char>::eof())
    throw ModelIntegrityError("Model read failed or file size changed during loading.");

  struct Algorithm {
    BCRYPT_ALG_HANDLE handle{nullptr};
    ~Algorithm() { if (handle) BCryptCloseAlgorithmProvider(handle, 0); }
  } algorithm;
  if (BCryptOpenAlgorithmProvider(&algorithm.handle, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
    throw ModelIntegrityError("SHA-256 provider could not be initialized.");
  std::array<UCHAR, 32> digest{};
  if (BCryptHash(algorithm.handle, nullptr, 0, result.bytes.data(),
                 static_cast<ULONG>(result.bytes.size()), digest.data(),
                 static_cast<ULONG>(digest.size())) < 0)
    throw ModelIntegrityError("SHA-256 calculation failed.");
  constexpr char hex[] = "0123456789ABCDEF";
  for (const auto value : digest) {
    result.sha256 += hex[value >> 4];
    result.sha256 += hex[value & 15];
  }
  if (result.sha256 != expected)
    throw ModelIntegrityError(std::string("Model SHA-256 mismatch: expected=") +
                              expected + " actual=" + result.sha256);
  return result;
}

}  // namespace obs_face_mosaic::inference
