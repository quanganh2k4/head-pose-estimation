# Headpose Camera IP

Hệ thống phân tích **head pose** và **gaze direction** theo thời gian thực từ nhiều camera IP. Dự án được tối ưu cho NVIDIA Jetson, sử dụng DeepStream/TensorRT ở tầng inference và FastAPI ở tầng business logic.

[![CI Backend](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-backend.yml/badge.svg)](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-backend.yml)
[![CI Pipeline](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-pipeline.yml/badge.svg)](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-pipeline.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-NVIDIA%20Jetson-76B900?logo=nvidia&logoColor=white)
![DeepStream](https://img.shields.io/badge/DeepStream-7.0-76B900)
![TensorRT](https://img.shields.io/badge/TensorRT-C%2B%2B%20API-76B900)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20%2B%20WebSocket-009688?logo=fastapi&logoColor=white)

## Mục tiêu

- Xử lý đồng thời nhiều luồng RTSP.
- Tận dụng GPU/NVMM zero-copy trên NVIDIA Jetson.
- Cho phép thêm và xóa camera khi hệ thống đang chạy.
- Tách inference nặng bằng C++ khỏi business logic bằng Python.
- Cung cấp dữ liệu pose/gaze qua REST, WebSocket và MQTT.

## Kiến trúc

Hệ thống có bốn thành phần chính:

| Thành phần | Vai trò | Cổng |
| --- | --- | --- |
| `mediamtx` | Kéo RTSP từ camera một lần, proxy RTSP và phát WebRTC/WHEP | `8554`, `8889`, `9997` |
| `vision-pipeline` | DeepStream, PeopleNet, NvDCF, FaceMesh TensorRT và GPU processing | gRPC `50051` |
| `analytics-api` | FastAPI, head pose/gaze, filter, alert và WebSocket | HTTP `8080` |
| `mqtt-broker` | Message bus giữa pipeline và analytics API | MQTT `1883` |

```mermaid
flowchart LR
    Camera[Camera IP<br/>RTSP] -->|RTSP| MTX[MediaMTX<br/>single pull point]
    MTX -->|RTSP nội bộ| Vision[vision-pipeline<br/>C++ / DeepStream / TensorRT]
    Vision -->|gaze/cam_id/metadata| MQTT[(Mosquitto MQTT)]
    MQTT --> Analytics[analytics-api<br/>Python / FastAPI]
    Analytics -->|REST + WebSocket| Client[Dashboard / Client]
    MTX -->|WebRTC / WHEP| Client
    Analytics -->|MediaMTX API| MTX
    Analytics -->|gRPC Add/Remove/ListCamera| Vision
```

### Luồng dữ liệu

1. MediaMTX kéo camera thật và tạo path RTSP nội bộ.
2. `vision-pipeline` đọc path đó bằng GStreamer/DeepStream.
3. Pad probe lấy metadata, crop khuôn mặt trên GPU và chạy FaceMesh TensorRT.
4. Landmark được publish lên `gaze/<camera_id>/metadata`.
5. `analytics-api` tính pose/gaze, lọc nhiễu và publish lên `gaze/<camera_id>/calculated`.
6. Client nhận telemetry qua WebSocket và nhận video trực tiếp qua WebRTC/WHEP.

`gRPC` chỉ dùng cho command như `AddCamera`, `RemoveCamera`, `ListCameras`. `MQTT` dùng cho stream/event liên tục; không dùng MQTT để quản lý lifecycle camera.

Tài liệu architecture đầy đủ, quy tắc mở rộng và cách debug nằm tại [docs/architecture.md](docs/architecture.md).

## Công nghệ

- C++17, GStreamer, NVIDIA DeepStream 7.0, TensorRT
- Python 3.10+, FastAPI, Paho MQTT, gRPC
- MediaMTX, Eclipse Mosquitto, Docker Compose
- NVIDIA Jetson Orin/Xavier/Nano hoặc Linux NVIDIA GPU tương thích

## Cấu trúc thư mục

```text
.
├── api/proto/                 # Proto dùng chung cho gRPC
├── configs/                   # MediaMTX, Mosquitto, pipeline config
├── deployments/               # Docker Compose và file môi trường
├── docs/                      # Architecture, API và tài liệu kỹ thuật
├── models/                    # Model/config inference
├── services/
│   ├── vision_pipeline/       # C++ DeepStream/TensorRT service
│   └── analytics_api/         # Python FastAPI và thuật toán
├── tools/                     # RTSP simulator, visualizer, benchmark
├── benchmarks/                # Baseline và benchmark
├── Makefile
└── README.md
```

## Yêu cầu

- Docker và Docker Compose.
- NVIDIA Container Toolkit nếu chạy production pipeline trên Jetson/GPU.
- Python 3.10+ nếu chạy test hoặc tool ngoài container.
- Camera IP hỗ trợ RTSP, hoặc file video để dùng với RTSP simulator.

## Cấu hình

Tạo file môi trường local từ mẫu:

```bash
cp deployments/.env.example deployments/.env
```

Điền URL camera và địa chỉ Jetson mà client bên ngoài có thể truy cập. Không commit username/password camera hoặc file `.env` vào Git.

## Chạy hệ thống

### Production trên Jetson

```bash
make prod-up
```

Xem log:

```bash
docker compose -f deployments/docker/docker-compose.prod.yml logs -f
```

Dừng hệ thống:

```bash
make prod-down
```

### Development/local

Khởi động Mosquitto, MediaMTX và analytics API:

```bash
make dev-up
```

Dừng môi trường dev:

```bash
make dev-down
```

Nếu không có camera thật, có thể dùng simulator:

```bash
python tools/rtsp_simulator.py \
  --input sample_video.mp4 \
  --url rtsp://localhost:8554/cam0
```

## API và protocol

Các endpoint và topic đầy đủ được ghi tại [docs/api_reference.md](docs/api_reference.md).

| Loại | Địa chỉ | Mục đích |
| --- | --- | --- |
| REST | `GET /cameras` | Liệt kê camera |
| REST | `POST /cameras/add` | Thêm camera runtime |
| REST | `DELETE /cameras/{cam_id}` | Xóa camera runtime |
| REST | `GET /history/{cam_id}` | Lấy telemetry lịch sử |
| WebSocket | `/ws/gaze` | Stream pose/gaze và alert |
| MQTT | `gaze/{cam_id}/metadata` | Landmark thô từ pipeline |
| MQTT | `gaze/{cam_id}/calculated` | Pose/gaze đã filter và alert |
| gRPC | `:50051` | Quản lý lifecycle camera |

Ví dụ thêm camera:

```bash
curl -X POST http://localhost:8080/cameras/add \
  -H 'Content-Type: application/json' \
  -d '{"url":"rtsp://user:password@192.168.1.50:554/stream"}'
```

## Kiểm thử và chất lượng

```bash
make test       # Python unit tests
make lint       # Ruff và các kiểm tra lint
make format     # Format Python/tooling code
```

Khi thay đổi C++ pipeline, cần build Docker image tương ứng và kiểm tra model, batch size, GPU memory, MQTT topic và gRPC contract.

## Nguyên tắc bảo trì

- Sửa proto tại `api/proto/`, không sửa trực tiếp file generated.
- Không cho `vision-pipeline` kéo trực tiếp camera thật; luôn đi qua MediaMTX.
- Giữ business logic trong `analytics-api`, không đưa alert/filter vào pad probe.
- Dùng environment variable cho IP, port, model path và ngưỡng cảnh báo.
- Khi thêm endpoint/topic/service, cập nhật tài liệu tương ứng trong `docs/`.
- Khi debug, kiểm tra theo thứ tự: MediaMTX → GStreamer/DeepStream → MQTT → analytics API → WebSocket/WebRTC.

## Tài liệu liên quan

- [Architecture và data flow](docs/architecture.md)
- [API và protocol reference](docs/api_reference.md)
- [Dynamic pad manipulation](docs/dynamic_pad_manipulation.md)
- [C++/DeepStream design](docs/design_and_architecture.md)
- [Benchmark](docs/benchmarks.md)
- [Technical deep dive](docs/codebase_line_by_line_architecture_deep_dive.md)

## License

Dự án được phát hành theo [MIT License](LICENSE).
