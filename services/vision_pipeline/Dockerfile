FROM nvcr.io/nvidia/deepstream-l4t:7.0-triton-multiarch

ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt compiler, cmake, gRPC, Protobuf và các thư viện cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libgrpc++-dev \
    protobuf-compiler-grpc \
    libprotobuf-dev \
    libmosquitto-dev \
    libglib2.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy toàn bộ mã nguồn của service (bao gồm CMakeLists.txt, proto/, src/)
COPY services/vision_pipeline/ /app/

# Copy binary đã biên dịch sẵn từ host (được compile trong container có GPU/runtime=nvidia)
COPY services/vision_pipeline/build/deepstream_gRPC_service /app/deepstream_gRPC_service

# Expose cổng gRPC mặc định
EXPOSE 50051

# Khởi chạy binary
ENTRYPOINT ["/app/deepstream_gRPC_service"]
