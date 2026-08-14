# Headpose Camera IP — Production Edge AI & DeepStream Microservices

[![CI Backend](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-backend.yml/badge.svg)](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-backend.yml)
[![CI Pipeline](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-pipeline.yml/badge.svg)](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-pipeline.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-NVIDIA%20Jetson-76B900?logo=nvidia&logoColor=white)
![DeepStream](https://img.shields.io/badge/DeepStream-7.0-76B900)
![TensorRT](https://img.shields.io/badge/TensorRT-C%2B%2B%20API-76B900)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20%2B%20WebSocket-009688?logo=fastapi&logoColor=white)

Hệ thống phân tích hướng nhìn (gaze) và tư thế đầu (head pose) thời gian thực đa luồng RTSP camera IP, được xây dựng trên nền tảng **NVIDIA DeepStream SDK 7.0**, **TensorRT C++ API**, **GStreamer**, **gRPC**, **FastAPI**, và **MQTT**, tối ưu hóa chuyên sâu cho các thiết bị Edge AI như NVIDIA Jetson Orin / Xavier / Nano.

Hệ thống áp dụng kiến trúc **Decoupled Microservices** giúp tách biệt hoàn toàn tầng AI inference nặng phần cứng GPU khỏi tầng Business logic/Alert engine nhẹ trên CPU, cho phép quản lý thêm/xóa camera động tại runtime mà không gián đoạn pipeline.

---

## ⚡ Bảng So Sánh Hiệu Năng (Benchmarks)

| Pipeline Architecture | Latency (End-to-End) | Max Concurrent Streams (1080p@30fps) | GPU Utilization | CPU Load | Memory (VRAM/RAM) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Python Monolithic (PoC)** | ~85 ms | 2 Streams (GIL Bottleneck) | ~45% | ~85% (High) | ~2.1 GB |
| **C++ Native Microservices** | **~18 ms** | **6 - 8 Streams (Zero-copy GPU)** | **~85% (Optimal)** | **~22% (Low)** | **~1.2 GB** |

> Xem chi tiết báo cáo đo kiểm và sizing phần cứng tại [docs/benchmarks.md](docs/benchmarks.md).

---

## 🏛️ Kiến Trúc Hệ Thống (Architecture)

``![Diagram](https://mermaid.ink/img/eyJjb2RlIjogImdyYXBoIExSXG4gICAgc3ViZ3JhcGggTEFOW1wiTVx1MWVhMW5nIExBTiAvIEVkZ2UgTmV0d29ya1wiXVxuICAgICAgICBDQU1bXCJcdWQ4M2RcdWRjZjcgQ2FtZXJhIElQIDEuLk48YnIvPihSVFNQLCBIMjY0L0gyNjUpXCJdXG4gICAgICAgIFZJRVdFUltcIlx1ZDgzZFx1ZGNiYiBDbGllbnQgLyBWaWV3ZXIgR1VJPGJyLz5SRVNUIFx1MDBiNyBXZWJTb2NrZXQgXHUwMGI3IFdlYlJUQ1wiXVxuICAgIGVuZFxuXG4gICAgc3ViZ3JhcGggSkVUU09OW1wiTlZJRElBIEpldHNvbiBIb3N0IChEb2NrZXIgQ29tcG9zZSlcIl1cbiAgICAgICAgTVRYW1wiTWVkaWFNVFggR2F0ZXdheTxici8+UlRTUCBQcm94eSA6ODU1NCBcdTAwYjcgQVBJIDo5OTk3PGJyLz5XZWJSVEMvV0hFUCA6ODg4OVwiXVxuXG4gICAgICAgIHN1YmdyYXBoIERTW1widmlzaW9uLXBpcGVsaW5lIChDKysxNylcIl1cbiAgICAgICAgICAgIEdSUENbXCJnUlBDIFNlcnZlciA6NTAwNTE8YnIvPkFkZC9SZW1vdmUvTGlzdCBDYW1lcmFcIl1cbiAgICAgICAgICAgIFBJUEVbXCJHU3RyZWFtZXIgUGlwZWxpbmU8YnIvPnJ0c3BzcmMgXHUyMTkyIG52c3RyZWFtbXV4IFx1MjE5Mjxici8+UGVvcGxlTmV0IChudmluZmVyKSBcdTIxOTIgTnZEQ0YgdHJhY2tlclwiXVxuICAgICAgICAgICAgUFJPQkVbXCJDdXN0b20gUGFkIFByb2JlPGJyLz5HUFUgQ3JvcCAoTnZCdWZTdXJmVHJhbnNmb3JtKTxici8+RmFjZU1lc2ggVGVuc29yUlQgQysrIEFQSVwiXVxuICAgICAgICBlbmRcblxuICAgICAgICBNUVRUW1wiTVFUVCBCcm9rZXI8YnIvPk1vc3F1aXR0byA6MTg4M1wiXVxuXG4gICAgICAgIHN1YmdyYXBoIEFQUFtcImFuYWx5dGljcy1hcGkgKFB5dGhvbiAvIEZhc3RBUEkpXCJdXG4gICAgICAgICAgICBBTEdPW1wiU3BoZXJpY2FsIE1vcnBoaW5nPGJyLz4rIE9uZS1FdXJvIEZpbHRlclwiXVxuICAgICAgICAgICAgQUxFUlRbXCJBbGVydCBFbmdpbmUgKERpc3RyYWN0aW9uKVwiXVxuICAgICAgICAgICAgQVBJW1wiUkVTVCBBUEkgOjgwODA8YnIvPldlYlNvY2tldCAvd3MvZ2F6ZVwiXVxuICAgICAgICBlbmRcbiAgICBlbmRcblxuICAgIENBTSAtLT58XCJSVFNQIEluZ2VzdFwifCBNVFhcbiAgICBNVFggLS0+fFwiUHJveGllZCBSVFNQXCJ8IFBJUEVcbiAgICBQSVBFIC0tPiBQUk9CRVxuICAgIFBST0JFIC0tPnxcIkxhbmRtYXJrIEpTT04gKGdhemUvKy9tZXRhZGF0YSlcInwgTVFUVFxuICAgIE1RVFQgLS0+IEFMR09cbiAgICBBTEdPIC0tPiBBTEVSVFxuICAgIEFMR08gLS0+fFwiU21vb3RoZWQgUG9zZSAoZ2F6ZS8rL2NhbGN1bGF0ZWQpXCJ8IE1RVFRcbiAgICBBUEkgLS0+fFwiXHUwMTEwXHUwMTAzbmcga1x1MDBmZCBwYXRoXCJ8IE1UWFxuICAgIEFQSSAtLT58XCJBZGRDYW1lcmEgLyBSZW1vdmVDYW1lcmFcInwgR1JQQ1xuICAgIEdSUEMgLS4tPnxcIkR5bmFtaWMgUGFkIExpbmsvVW5saW5rXCJ8IFBJUEVcbiAgICBWSUVXRVIgPC0tPiBBUElcbiAgICBNVFggLS0+fFwiV2ViUlRDIFZpZGVvIChXSEVQKVwifCBWSUVXRVIiLCAibWVybWFpZCI6IHsidGhlbWUiOiAiZGVmYXVsdCJ9fQ==)

<details><summary>🔍 Xem mã nguồn Mermaid</summary>

`mermaid
graph LR
    subgraph LAN["Mạng LAN / Edge Network"]
        CAM["📷 Camera IP 1..N<br/>(RTSP, H264/H265)"]
        VIEWER["💻 Client / Viewer GUI<br/>REST · WebSocket · WebRTC"]
    end

    subgraph JETSON["NVIDIA Jetson Host (Docker Compose)"]
        MTX["MediaMTX Gateway<br/>RTSP Proxy :8554 · API :9997<br/>WebRTC/WHEP :8889"]

        subgraph DS["vision-pipeline (C++17)"]
            GRPC["gRPC Server :50051<br/>Add/Remove/List Camera"]
            PIPE["GStreamer Pipeline<br/>rtspsrc → nvstreammux →<br/>PeopleNet (nvinfer) → NvDCF tracker"]
            PROBE["Custom Pad Probe<br/>GPU Crop (NvBufSurfTransform)<br/>FaceMesh TensorRT C++ API"]
        end

        MQTT["MQTT Broker<br/>Mosquitto :1883"]

        subgraph APP["analytics-api (Python / FastAPI)"]
            ALGO["Spherical Morphing<br/>+ One-Euro Filter"]
            ALERT["Alert Engine (Distraction)"]
            API["REST API :8080<br/>WebSocket /ws/gaze"]
        end
    end

    CAM -->|"RTSP Ingest"| MTX
    MTX -->|"Proxied RTSP"| PIPE
    PIPE --> PROBE
    PROBE -->|"Landmark JSON (gaze/+/metadata)"| MQTT
    MQTT --> ALGO
    ALGO --> ALERT
    ALGO -->|"Smoothed Pose (gaze/+/calculated)"| MQTT
    API -->|"Đăng ký path"| MTX
    API -->|"AddCamera / RemoveCamera"| GRPC
    GRPC -.->|"Dynamic Pad Link/Unlink"| PIPE
    VIEWER <--> API
    MTX -->|"WebRTC Video (WHEP)"| VIEWER
`

</details>``

### Các Service Độc Lập
1. **`vision-pipeline` (C++17 / TensorRT / GStreamer):**
   - Ingestion đa camera qua GStreamer pipeline tối ưu phần cứng.
   - Nhận diện người bằng PeopleNet + Tracking bằng NvDCF.
   - Crop khuôn mặt và chuyển đổi định dạng trực tiếp trên GPU qua `NvBufSurfTransform` (Zero-copy).
   - Suy luận 468/478 Face Landmarks bằng FaceMesh qua TensorRT C++ Native API.
   - gRPC Server (:50051) cho phép gắn/gỡ dynamic pad vào `nvstreammux` mà không cần khởi động lại pipeline.
2. **`analytics-api` (Python 3.10 / FastAPI):**
   - Subscribe message từ MQTT broker.
   - Tính toán tư thế đầu (Head Pose - Yaw, Pitch, Roll) bằng thuật toán **Spherical Morphing**.
   - Khử rung thích ứng bằng **One-Euro Filter**.
   - Cung cấp REST APIs, WebSocket streaming và WebRTC player.
3. **`mediamtx`:** Proxy RTSP và Gateway WebRTC WHEP, tránh quá tải kết nối trực tiếp đến camera IP.
4. **`mqtt-broker` (Eclipse Mosquitto):** Message bus trung gian độ trễ siêu thấp giữa C++ và Python.

---

## 📁 Cấu Trúc Thư Mục Chuẩn Production

```text
headpose-cameraIP/
├── .github/                         # CI/CD Workflows
│   └── workflows/
│       ├── ci-backend.yml           # Linting (Ruff), Pytest unit tests
│       ├── ci-pipeline.yml          # Format check (Clang-format)
│       └── docker-build.yml         # Container build smoke check
├── api/                             # Single Source of Truth cho gRPC / Protobuf
│   └── proto/
│       └── camera_service.proto     # Schema định nghĩa gRPC API
├── assets/                          # Hình ảnh sơ đồ kiến trúc & Dashboard UI
│   ├── system_architecture.png
│   ├── demo_realtime.png
│   └── dashboard_preview.png
├── configs/                         # Cấu hình tập trung cho toàn bộ hệ thống
│   ├── mediamtx.yml
│   ├── mosquitto.conf
│   └── pipeline_config.example.yaml
├── deployments/                     # Môi trường triển khai
│   ├── docker/
│   │   ├── docker-compose.prod.yml  # Compose production cho NVIDIA Jetson
│   │   └── docker-compose.dev.yml   # Compose dev/local (mock RTSP, broker)
│   ├── systemd/
│   │   └── headpose.service         # Systemd service unit cho Jetson bare-metal
│   └── .env.example                 # Mẫu biến môi trường
├── docs/                            # Tài liệu kỹ thuật chi tiết
│   ├── architecture.md              # Thiết kế chi tiết kiến trúc & data flow
│   ├── benchmarks.md                # Báo cáo đo kiểm hiệu năng chi tiết
│   ├── dynamic_pad_manipulation.md  # Kỹ thuật gRPC Add/Remove stream động
│   ├── api_reference.md             # Đặc tả REST / WebSocket / gRPC / MQTT
│   └── design_and_architecture.md   # Thiết kế C++ pipeline & probe internals
├── models/                          # Cấu hình inference & helper scripts
│   ├── download_weights.sh          # Script tải/kiểm tra model weights
│   ├── peoplenet/                   # PeopleNet configs & labels
│   ├── mediapipe_pose/              # FaceMesh ONNX & parser configs
│   ├── retinaface/                  # RetinaFace fallback configs
│   └── scrfd/                       # SCRFD fallback configs
├── services/                        # Microservices mã nguồn chính
│   ├── vision_pipeline/             # [C++17] Native DeepStream & TensorRT Service
│   │   ├── CMakeLists.txt
│   │   ├── Dockerfile
│   │   ├── proto/
│   │   └── src/                     # main.cpp, pipeline.cpp, probe_processor.cpp
│   └── analytics_api/               # [Python] FastAPI Business Logic & Alert Engine
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── main.py
│       ├── modules/                 # head_pose_algo.py (Spherical Morphing & Filter)
│       ├── static/                  # Web dashboard UI
│       └── tests/                   # Pytest unit tests cho thuật toán
├── tools/                           # Developer & Evaluation Tools
│   ├── gaze_visualizer_gui.py       # GUI xem trực tiếp video và overlay pose debug
│   ├── rtsp_simulator.py            # Giả lập camera RTSP từ file video MP4
│   └── benchmark_latency.py         # Công cụ đo độ trễ MQTT/gRPC end-to-end
├── benchmarks/                      # PoC baseline lưu trữ để so sánh hiệu năng
│   └── poc_monolithic_pipeline.py
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml          # Quản lý pre-commit linting (Ruff, Clang-format)
├── Makefile                         # Lệnh điều khiển 1 chạm (make prod-up, make test, ...)
└── README.md
```

---

## 🚀 Hướng Dẫn Khởi Chạy (Quickstart)

### 1. Yêu Cầu Phần Cứng & Môi Trường
- Thiết bị **NVIDIA Jetson** (JetPack 6.x / DeepStream 7.0+) hoặc máy Linux x86_64 có NVIDIA GPU hỗ trợ TensorRT.
- Docker & NVIDIA Container Toolkit.
- Python 3.10+ (nếu chạy test hoặc công cụ phụ trợ ngoài container).

### 2. Cài Đặt và Khởi Chạy Nhanh với Makefile

```bash
# 1. Cấu hình biến môi trường
cp deployments/.env.example deployments/.env
# Điền thông tin RTSP camera vào deployments/.env

# 2. Khởi chạy toàn bộ hệ thống (Production)
make prod-up

# 3. Xem log hoạt động
docker compose -f deployments/docker/docker-compose.prod.yml logs -f
```

### 3. Thêm / Xóa Camera Động tại Runtime

```bash
# Thêm camera (không restart service):
curl -X POST http://localhost:8080/cameras/add \
     -H 'Content-Type: application/json' \
     -d '{"url": "rtsp://user:password@192.168.1.50:554/stream"}'

# Xóa camera:
curl -X DELETE http://localhost:8080/cameras/1
```

### 4. Sử Dụng Công Cụ Giả Lập & Visualizer (Tools)

- **Giả lập RTSP từ file video (không cần camera thật):**
  ```bash
  python tools/rtsp_simulator.py --input sample_video.mp4 --url rtsp://localhost:8554/cam0
  ```
- **Mở Visualizer Debug GUI:**
  ```bash
  python tools/gaze_visualizer_gui.py
  ```
- **Đo Benchmark độ trễ:**
  ```bash
  python tools/benchmark_latency.py --broker 127.0.0.1
  ```

---

## 🧪 Kiểm Thử & Đảm Bảo Chất Lượng Mã Nguồn (QA & Testing)

```bash
# Chạy Unit Tests thuật toán (One-Euro filter, Spherical Morphing, Geom Check)
make test

# Kiểm tra định dạng code & linting
make lint

# Tự động format code
make format
```

---

## 📄 Tài Liệu Kỹ Thuật Bổ Sung
- [Chi tiết Thiết kế Kiến trúc](docs/architecture.md)
- [Báo cáo Hiệu năng & Đo kiểm](docs/benchmarks.md)
- [Cơ chế Link/Unlink Dynamic Pad](docs/dynamic_pad_manipulation.md)
- [Tài liệu API & Protocol Specs](docs/api_reference.md)
- [Ghi chú Thiết kế C++ DeepStream](docs/design_and_architecture.md)

---

## 📜 Giấy Phép (License)
Dự án được phân phối dưới giấy phép [MIT License](LICENSE).
