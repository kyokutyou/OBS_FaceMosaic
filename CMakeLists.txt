cmake_minimum_required(VERSION 3.28)
project(obs_face_mosaic VERSION 0.1.1 LANGUAGES CXX)

if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "OBS_FaceMosaic currently supports Windows x64 only.")
endif()

set(OBS_TARGET_VERSION "32.2.2" CACHE STRING "Exact OBS/libobs version to build against (supported: 32.2.2, 30.2.3)")
set_property(CACHE OBS_TARGET_VERSION PROPERTY STRINGS "32.2.2" "30.2.3")
if(NOT OBS_TARGET_VERSION MATCHES "^(32\\.2\\.2|30\\.2\\.3)$")
  message(FATAL_ERROR "OBS_TARGET_VERSION must be 32.2.2 or 30.2.3 (got '${OBS_TARGET_VERSION}').")
endif()

set(OBS_BUILD_DIR "" CACHE PATH "OBS ${OBS_TARGET_VERSION} build directory containing libobs/libobsConfig.cmake")
set(OBS_LIBOBS_DIR "" CACHE PATH "Directory containing libobsConfig.cmake from OBS ${OBS_TARGET_VERSION}")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Extracted Microsoft.ML.OnnxRuntime.DirectML package (optional; fetched at the pinned version when empty)")
set(DIRECTML_ROOT "" CACHE PATH "Extracted Microsoft.AI.DirectML package (optional; fetched at the pinned version when empty)")
set(OBSFM_FETCH_NATIVE_DEPENDENCIES ON CACHE BOOL "Download pinned ONNX Runtime and DirectML NuGet packages when roots are omitted")

if(OBS_BUILD_DIR AND NOT OBS_LIBOBS_DIR)
  set(OBS_LIBOBS_DIR "${OBS_BUILD_DIR}/libobs")
endif()
if(OBS_LIBOBS_DIR)
  list(PREPEND CMAKE_PREFIX_PATH "${OBS_LIBOBS_DIR}")
endif()
find_package(libobs ${OBS_TARGET_VERSION} EXACT CONFIG REQUIRED)
if(NOT TARGET OBS::libobs)
  message(FATAL_ERROR "The selected libobs package does not export OBS::libobs.")
endif()

include(cmake/FetchNuget.cmake)
if(NOT ONNXRUNTIME_ROOT)
  if(NOT OBSFM_FETCH_NATIVE_DEPENDENCIES)
    message(FATAL_ERROR "Set ONNXRUNTIME_ROOT or enable OBSFM_FETCH_NATIVE_DEPENDENCIES.")
  endif()
  obsfm_resolve_nuget_package(
    microsoft.ml.onnxruntime.directml 1.24.4
    "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/1.24.4/microsoft.ml.onnxruntime.directml.1.24.4.nupkg"
    57E9F11B73437BEF7A309496135D4C1F96B1A8E9DDBA60013FA27BFC1D788681
    ONNXRUNTIME_ROOT
  )
endif()
if(NOT DIRECTML_ROOT)
  if(NOT OBSFM_FETCH_NATIVE_DEPENDENCIES)
    message(FATAL_ERROR "Set DIRECTML_ROOT or enable OBSFM_FETCH_NATIVE_DEPENDENCIES.")
  endif()
  obsfm_resolve_nuget_package(
    microsoft.ai.directml 1.15.4
    "https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg"
    4E7CB7DDCE8CF837A7A75DC029209B520CA0101470FCDF275C1F49736A3615B9
    DIRECTML_ROOT
  )
endif()

set(_ort_include "${ONNXRUNTIME_ROOT}/build/native/include")
set(_ort_native "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native")
set(_ort_lib "${_ort_native}/onnxruntime.lib")
set(_ort_dll "${_ort_native}/onnxruntime.dll")
set(_directml_dll "${DIRECTML_ROOT}/bin/x64-win/DirectML.dll")
foreach(_required_file IN ITEMS "${_ort_include}/onnxruntime_cxx_api.h" "${_ort_lib}" "${_ort_dll}" "${_directml_dll}")
  if(NOT EXISTS "${_required_file}")
    message(FATAL_ERROR "Missing required ONNX Runtime/DirectML package file: ${_required_file}")
  endif()
endforeach()

add_library(obs_face_mosaic_core STATIC
  src/core/mosaic.cpp
  src/core/processor.cpp
)
target_compile_features(obs_face_mosaic_core PUBLIC cxx_std_20)
target_include_directories(obs_face_mosaic_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
set_target_properties(obs_face_mosaic_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(obs_face_mosaic_inference STATIC
  src/inference/detector.cpp
  src/inference/preprocess.cpp
  src/inference/model_integrity.cpp
)
target_compile_features(obs_face_mosaic_inference PUBLIC cxx_std_20)
target_include_directories(obs_face_mosaic_inference PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${_ort_include}"
)
target_link_libraries(obs_face_mosaic_inference PUBLIC obs_face_mosaic_core "${_ort_lib}")
if(WIN32)
  target_link_libraries(obs_face_mosaic_inference PUBLIC dxgi bcrypt)
endif()
set_target_properties(obs_face_mosaic_inference PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(obs-face-mosaic MODULE
  src/obs/filter.cpp
  src/obs/frame_converter.cpp
  src/obs/settings.cpp
)
option(OBSFM_ENABLE_TEST_HOOKS "Enable isolated OBS inference fault injection (never for distribution)" OFF)
if(OBSFM_ENABLE_TEST_HOOKS)
  target_compile_definitions(obs-face-mosaic PRIVATE OBSFM_ENABLE_TEST_HOOKS=1)
endif()
target_compile_features(obs-face-mosaic PRIVATE cxx_std_20)
target_include_directories(obs-face-mosaic PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(obs-face-mosaic PRIVATE OBS::libobs obs_face_mosaic_core obs_face_mosaic_inference)
target_compile_definitions(obs-face-mosaic PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
if(MSVC)
  target_compile_options(obs_face_mosaic_core PRIVATE /W4 /permissive- /utf-8)
  target_compile_options(obs_face_mosaic_inference PRIVATE /W4 /permissive- /utf-8)
  target_compile_options(obs-face-mosaic PRIVATE /W4 /permissive- /utf-8)
endif()
set_target_properties(obs-face-mosaic PROPERTIES PREFIX "" OUTPUT_NAME "obs-face-mosaic")

add_custom_command(TARGET obs-face-mosaic POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_ort_dll}" "$<TARGET_FILE_DIR:obs-face-mosaic>/onnxruntime.dll"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_directml_dll}" "$<TARGET_FILE_DIR:obs-face-mosaic>/DirectML.dll"
  VERBATIM
)

include(GNUInstallDirs)
install(TARGETS obs-face-mosaic LIBRARY DESTINATION "obs-plugins/64bit" RUNTIME DESTINATION "obs-plugins/64bit")
install(FILES "${_ort_dll}" "${_directml_dll}" DESTINATION "obs-plugins/64bit")
