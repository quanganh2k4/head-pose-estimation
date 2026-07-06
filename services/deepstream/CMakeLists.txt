cmake_minimum_required(VERSION 3.13)
project(deepstream_gRPC_service CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Tìm các thư viện hệ thống ─────────────────────────────────────────────────
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# Protobuf (sử dụng cmake find_package module)
find_package(Protobuf REQUIRED)
message(STATUS "Found Protobuf: ${Protobuf_VERSION}")

# gRPC (sử dụng PkgConfig vì gói cài từ apt không đi kèm gRPCConfig.cmake)
pkg_check_modules(GRPC REQUIRED grpc++ grpc)
message(STATUS "Found gRPC via PkgConfig")

# GStreamer & GLib (sử dụng PkgConfig)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-video-1.0 gstreamer-app-1.0)
pkg_check_modules(GLIB REQUIRED glib-2.0 gobject-2.0)

# Mosquitto (sử dụng PkgConfig)
pkg_check_modules(MOSQUITTO REQUIRED libmosquitto)

# OpenCV (sử dụng find_package)
find_package(OpenCV REQUIRED)

# DeepStream SDK (nvdsmeta & nvbufsurface)
# Tự động tìm thư mục DeepStream trong hệ thống
file(GLOB DEEPSTREAM_DIRS "/opt/nvidia/deepstream/deepstream-*")
list(GET DEEPSTREAM_DIRS 0 DEEPSTREAM_DIR)
message(STATUS "Found DeepStream directory: ${DEEPSTREAM_DIR}")

# Cấu hình include directory cho DeepStream, CUDA
include_directories(${DEEPSTREAM_DIR}/sources/includes /usr/local/cuda/include)

# Tự động tìm thư viện TensorRT & CUDA
find_library(NVINFER NAMES nvinfer REQUIRED)
find_library(CUDART NAMES cudart PATHS /usr/local/cuda/lib64 /usr/local/cuda/lib REQUIRED)

# Chỉ định trực tiếp các thư viện DeepStream để link
set(NVDS_GST_META "${DEEPSTREAM_DIR}/lib/libnvdsgst_meta.so")
set(NVBUF_SURFACE "${DEEPSTREAM_DIR}/lib/libnvbufsurface.so")
set(NVBUF_TRANSFORM "${DEEPSTREAM_DIR}/lib/libnvbufsurftransform.so")

# ── Tự động sinh code C++ từ file .proto ──────────────────────────────────────
set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/gens")
file(MAKE_DIRECTORY ${GEN_DIR})

set(PROTO_FILE "${PROTO_DIR}/camera.proto")
set(PROTO_SRC "${GEN_DIR}/camera.pb.cc")
set(PROTO_HDR "${GEN_DIR}/camera.pb.h")
set(GRPC_SRC "${GEN_DIR}/camera.grpc.pb.cc")
set(GRPC_HDR "${GEN_DIR}/camera.grpc.pb.h")

# Lấy vị trí của protoc compiler
if(TARGET protobuf::protoc)
  get_target_property(PROTOC_LOC protobuf::protoc IMPORTED_LOCATION_RELEASE)
  if(NOT PROTOC_LOC)
    get_target_property(PROTOC_LOC protobuf::protoc IMPORTED_LOCATION)
  endif()
endif()
if(NOT PROTOC_LOC AND Protobuf_PROTOC_EXECUTABLE)
  set(PROTOC_LOC ${Protobuf_PROTOC_EXECUTABLE})
endif()
if(NOT PROTOC_LOC)
  find_program(PROTOC_LOC NAMES protoc)
endif()
if(NOT PROTOC_LOC)
  set(PROTOC_LOC protoc)
endif()

# Lấy vị trí của grpc_cpp_plugin
find_program(GRPC_PLUGIN_LOC NAMES grpc_cpp_plugin)
if(NOT GRPC_PLUGIN_LOC)
  set(GRPC_PLUGIN_LOC grpc_cpp_plugin)
endif()

# Thiết lập custom command để biên dịch proto lúc compile
add_custom_command(
  OUTPUT "${PROTO_SRC}" "${PROTO_HDR}" "${GRPC_SRC}" "${GRPC_HDR}"
  COMMAND ${PROTOC_LOC}
  ARGS --proto_path "${PROTO_DIR}"
       --cpp_out "${GEN_DIR}"
       --grpc_out "${GEN_DIR}"
       --plugin=protoc-gen-grpc=${GRPC_PLUGIN_LOC}
       "${PROTO_FILE}"
  DEPENDS "${PROTO_FILE}"
  COMMENT "Generating C++ gRPC stubs from ${PROTO_FILE}"
  VERBATIM
)

# ── Định nghĩa Executable Target ──────────────────────────────────────────────
add_executable(deepstream_gRPC_service
  src/main.cpp
  src/pipeline.cpp
  src/pipeline.hpp
  src/probe_processor.cpp
  src/probe_processor.hpp
  ${PROTO_SRC}
  ${PROTO_HDR}
  ${GRPC_SRC}
  ${GRPC_HDR}
)

# Cấu hình các include directories cho target
target_include_directories(deepstream_gRPC_service PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${GEN_DIR}
  ${GST_INCLUDE_DIRS}
  ${GLIB_INCLUDE_DIRS}
  ${GRPC_INCLUDE_DIRS}
  ${MOSQUITTO_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Cấu hình compile flags bổ sung từ PkgConfig
target_compile_options(deepstream_gRPC_service PRIVATE
  ${GST_CFLAGS_OTHER}
  ${GLIB_CFLAGS_OTHER}
  ${GRPC_CFLAGS_OTHER}
  ${MOSQUITTO_CFLAGS_OTHER}
)

# Link các thư viện liên kết động
target_link_libraries(deepstream_gRPC_service PRIVATE
  ${GRPC_LIBRARIES}
  protobuf::libprotobuf
  ${GST_LIBRARIES}
  ${GLIB_LIBRARIES}
  ${MOSQUITTO_LIBRARIES}
  ${OpenCV_LIBS}
  ${NVDS_GST_META}
  ${NVBUF_SURFACE}
  ${NVBUF_TRANSFORM}
  ${NVINFER}
  ${CUDART}
  Threads::Threads
)
