# Headpose Camera IP

[English](README.md) | **Tiếng Việt**

[![CI Backend](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-backend.yml/badge.svg)](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-backend.yml)
[![CI Pipeline](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-pipeline.yml/badge.svg)](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-pipeline.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-NVIDIA%20Jetson-76B900?logo=nvidia&logoColor=white)
![DeepStream](https://img.shields.io/badge/DeepStream-7.0-76B900)
![TensorRT](https://img.shields.io/badge/TensorRT-C%2B%2B%20API-76B900)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20%2B%20WebSocket-009688?logo=fastapi&logoColor=white)

Phân tích tư thế đầu và hướng nhìn theo thời gian thực từ nhiều camera IP. Suy luận video nặng chạy trong service C++ dùng DeepStream/TensorRT, còn Python/FastAPI xử lý logic ứng dụng, smoothing, cảnh báo, quản lý camera và cung cấp dữ liệu cho client.

## Mục tiêu

- Xử lý nhiều luồng RTSP thông qua một điểm pull của MediaMTX cho mỗi camera.
- Tận dụng xử lý GPU/NVMM zero-copy khi môi trường hỗ trợ.
- Phát hiện người/khuôn mặt, tracking object và chạy FaceMesh TensorRT.
- Thêm, xóa và liệt kê camera khi pipeline đang chạy.
- Publish landmark thô và dữ liệu pose/gaze đã tính qua MQTT.
- Cung cấp REST, WebSocket và WebRTC/WHEP.
- Giữ credential, ngưỡng, địa chỉ và đường dẫn model phụ thuộc triển khai ở dạng cấu hình.

## Kiến trúc

Runtime được chia thành bốn thành phần:

| Thành phần | Trách nhiệm | Giao diện |
| --- | --- | --- |
| mediamtx | Một điểm pull camera, proxy RTSP và phân phối WebRTC/WHEP | RTSP :8554, WebRTC :8889, API :9997 |
| vision-pipeline | GStreamer/DeepStream C++17, PeopleNet, tracking và FaceMesh TensorRT | gRPC :50051, MQTT |
| analytics-api | FastAPI, thuật toán pose/gaze, filter, alert, proxy camera và WebSocket | HTTP :8080, gRPC, MQTT |
| mqtt-broker | Message bus Mosquitto cho metadata và kết quả liên tục | MQTT :1883 |

~~~mermaid
flowchart LR
    Camera[Camera IP / RTSP] --> MTX[MediaMTX<br/>điểm pull duy nhất]
    MTX -->|RTSP nội bộ| Vision[vision-pipeline<br/>C++ / DeepStream / TensorRT]
    Vision -->|gaze/cam_id/metadata| MQTT[(Mosquitto MQTT)]
    MQTT --> Analytics[analytics-api<br/>Python / FastAPI]
    Analytics -->|REST + WebSocket| Client[Dashboard / Client]
    MTX -->|WebRTC / WHEP| Client
    Analytics -->|MediaMTX API| MTX
    Analytics -->|gRPC Add/Remove/ListCamera| Vision
~~~

### Sơ đồ tương tác

| Sơ đồ | Nội dung | Bản live | Source |
| --- | --- | --- | --- |
| Architecture | Thành phần, protocol và data/control plane | [Mở trên GitHub Pages](https://quanganh2k4.github.io/head-pose-estimation/headpose-architecture.html) | [HTML](headpose-architecture.html) · [JSON](headpose-architecture.archify.json) |
| Sequence | Đăng ký camera và telemetry realtime | [Mở trên GitHub Pages](https://quanganh2k4.github.io/head-pose-estimation/headpose-sequence.html) | [HTML](headpose-sequence.html) · [JSON](headpose-sequence.archify.json) |

Các link live được tự động deploy bởi workflow [Deploy Archify diagrams](.github/workflows/deploy-diagrams.yml). Với repository này, chỉ cần bật **Settings → Pages → Source: GitHub Actions** một lần.

Thiết kế control plane/data plane, ownership, debug và mở rộng chi tiết nằm tại [docs/architecture.md](docs/architecture.md).

### Luồng dữ liệu

1. MediaMTX kéo stream camera thật và tạo path nội bộ.
2. vision-pipeline đọc path đó qua GStreamer/DeepStream.
3. Pipeline phát hiện người/khuôn mặt, tracking object, crop vùng mặt và chạy FaceMesh TensorRT.
4. Pad probe publish metadata landmark lên gaze/<camera_id>/metadata.
5. analytics-api validate landmark, tính pose/gaze, smoothing, áp dụng alert và publish gaze/<camera_id>/calculated.
6. Client nhận telemetry qua WebSocket và video từ MediaMTX bằng WebRTC/WHEP.

gRPC xử lý lệnh lifecycle như AddCamera, RemoveCamera và ListCameras. MQTT xử lý dữ liệu liên tục/sự kiện, không dùng cho lifecycle camera. Camera thật nên đi qua MediaMTX thay vì được pipeline pull trực tiếp.

### Trách nhiệm trong repository

- services/vision_pipeline/src/main.cpp: startup, MQTT, gRPC server và GLib main loop.
- services/vision_pipeline/src/pipeline.cpp: tạo GStreamer/DeepStream, nvstreammux, detector/tracker và source động.
- services/vision_pipeline/src/probe_processor.cpp: metadata DeepStream, face crop, TensorRT, validate landmark và MQTT.
- services/analytics_api/main.py: FastAPI, MQTT subscriber, gRPC client, đăng ký MediaMTX, state và alert.
- services/analytics_api/modules/head_pose_algo.py: validate landmark, spherical morphing, Euler và One Euro filter.
- services/analytics_api/tests/: test analytics độc lập phần cứng.

## Công nghệ

- C++17, CMake, GStreamer, NVIDIA DeepStream 7.0, TensorRT, OpenCV và Mosquitto.
- Python 3.10+, FastAPI, Paho MQTT, gRPC, NumPy và SciPy.
- MediaMTX và Docker Compose cho hạ tầng.
- NVIDIA Jetson Orin/Xavier/Nano hoặc Linux NVIDIA GPU tương thích.

## Cấu trúc repository

~~~text
.
├── api/proto/                 # Contract gRPC dùng chung
├── configs/                   # Cấu hình MediaMTX, Mosquitto và pipeline
├── deployments/               # Docker Compose, systemd và ví dụ môi trường
├── docs/                      # Tài liệu architecture, API và vận hành
├── models/                    # Cấu hình model và model local
├── services/
│   ├── vision_pipeline/       # Service C++ DeepStream/TensorRT
│   └── analytics_api/         # Service Python FastAPI và thuật toán
├── tools/                     # RTSP simulator, viewer và utility
├── benchmarks/                # Benchmark legacy chỉ dùng local
├── Makefile
├── README.md                 # Tài liệu chính tiếng Anh
└── README.vi.md              # Bản dịch tiếng Việt
~~~

## Yêu cầu

- Docker và Docker Compose.
- NVIDIA Container Toolkit và NVIDIA runtime tương thích cho production GPU.
- Python 3.10+ khi chạy test hoặc tool ngoài container.
- Camera IP hỗ trợ RTSP hoặc nguồn video cho simulator.
- Thư viện DeepStream/TensorRT và model local cho C++ pipeline.
- Network cho analytics-api, MediaMTX, MQTT broker và vision-pipeline kết nối nhau.

## Cấu hình

Tạo file môi trường local:

~~~bash
cp deployments/.env.example deployments/.env
~~~

Thiết lập CAMERA_URLS, MEDIAMTX_API, MEDIAMTX_RTSP_HOST, DEEPSTREAM_GRPC_SERVER, MQTT_BROKER, MQTT_PORT, YAW_ALERT_DEG, PITCH_ALERT_DEG và ALERT_DURATION_S.

Viewer độc lập lấy URL camera proxy từ analytics-api. Nếu API không khả dụng, đặt CAM0_RTSP_URL và/hoặc CAM1_RTSP_URL ở máy local; các giá trị này không được commit.

Nguồn lấy URL camera:

- Ưu tiên: thêm camera bằng `POST /cameras/add`; viewer sẽ lấy URL proxy MediaMTX từ `--app-api` (mặc định `http://127.0.0.1:8080`).
- Viewer độc lập: lấy URL RTSP trong trang quản trị camera/NVR rồi đặt vào biến môi trường local. Không đưa credential thật vào source code:

~~~bash
export CAM0_RTSP_URL='rtsp://<user>:<password>@<camera-host>:<port>/<stream-path>'
python tools/gaze_visualizer_gui.py --camera cam0
~~~

Trên PowerShell dùng `$env:CAM0_RTSP_URL = 'rtsp://<user>:<password>@<camera-host>:<port>/<stream-path>'`. Chạy `python tools/gaze_visualizer_gui.py --camera-url-help` để xem hướng dẫn tương tự.

Đường dẫn model/log có thể override bằng environment variable khi được hỗ trợ. Không commit credential camera, file .env, TensorRT engine sinh tự động, model weights hoặc artifact riêng tư.

## Chạy hệ thống

### Production trên Jetson

~~~bash
make prod-up
docker compose -f deployments/docker/docker-compose.prod.yml logs -f
make prod-down
~~~

### Development/local

~~~bash
make dev-up
make dev-down
~~~

Không có camera thật thì chạy simulator:

~~~bash
python tools/rtsp_simulator.py \
  --input sample_video.mp4 \
  --url rtsp://localhost:8554/cam0
~~~

Môi trường development phù hợp cho API/hạ tầng. Production pipeline cần Jetson/NVIDIA runtime và model DeepStream local.

## API và protocol

Contract đầy đủ nằm tại [docs/api_reference.md](docs/api_reference.md).

| Loại | Địa chỉ/topic | Mục đích |
| --- | --- | --- |
| REST | GET /health | Health và trạng thái kết nối |
| REST | GET /cameras | Liệt kê camera active |
| REST | POST /cameras/add | Thêm camera runtime |
| REST | DELETE /cameras/{src_id} | Xóa camera runtime |
| REST | GET /gaze/latest | Payload mới nhất của mọi camera |
| REST | GET /gaze/{cam_id}/history | Frame đã tính gần đây của một camera |
| REST | GET /alerts | Các alert gần đây |
| WebSocket | /ws/gaze/{cam_id} | Cập nhật pose/gaze trực tiếp |
| MQTT | gaze/{cam_id}/metadata | Landmark thô và metadata frame |
| MQTT | gaze/{cam_id}/calculated | Pose/gaze đã filter và alert |
| gRPC | :50051 | Quản lý lifecycle camera |

Ví dụ đăng ký camera:

~~~bash
curl -X POST http://localhost:8080/cameras/add \
  -H 'Content-Type: application/json' \
  -d '{"url":"rtsp://camera-host:554/stream"}'
~~~

Repository không chứa video camera thật. Chỉ thêm media demo khi có quyền công bố; khi test local nên dùng dữ liệu tổng hợp hoặc video có giấy phép phù hợp.

## Kiểm thử và chất lượng

~~~bash
make test
make lint
make format
~~~

Test độc lập phần cứng bao phủ One Euro filter, validate landmark, xử lý occlusion, spherical pose, dấu góc, rotation matrix và độ ổn định trên mặt tổng hợp đứng yên/chuyển động.

Khi đổi C++ cần build Docker image và kiểm tra tensor shape, batch size, GPU memory, MQTT topic, gRPC contract và lifecycle source.

## Quy tắc bảo trì

- Sửa proto nguồn; không sửa tay generated stub Python/C++.
- Giữ C++ tập trung vào throughput, ownership buffer, GPU/NVMM và lifecycle source.
- Để business logic, smoothing, alert và HTTP/WebSocket trong analytics-api.
- Route việc pull camera thật qua MediaMTX.
- Dùng environment variable cho địa chỉ, port, model path, credential và threshold.
- Cập nhật docs/api_reference.md khi contract endpoint/topic thay đổi.
- Thêm test độc lập phần cứng trong services/analytics_api/tests/.
- Không commit deep-dive notes private, benchmark legacy, credential, model weights hoặc build artifact.

## Tài liệu liên quan

- [Architecture và design](docs/architecture.md)
- [API reference](docs/api_reference.md)
- [Dynamic pad manipulation](docs/dynamic_pad_manipulation.md)
- [Benchmarks](docs/benchmarks.md)
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## License

Project phát hành theo [MIT License](LICENSE). License không áp dụng cho sample code proprietary của DeepStream hoặc model artifact bên thứ ba; xem [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
