# Headpose Camera IP

[![CI Backend](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-backend.yml/badge.svg)](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-backend.yml)
[![CI Pipeline](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-pipeline.yml/badge.svg)](https://github.com/quanganh2k4/head-pose-estimation/actions/workflows/ci-pipeline.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-NVIDIA%20Jetson-76B900?logo=nvidia&logoColor=white)
![DeepStream](https://img.shields.io/badge/DeepStream-7.0-76B900)
![TensorRT](https://img.shields.io/badge/TensorRT-C%2B%2B%20API-76B900)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20%2B%20WebSocket-009688?logo=fastapi&logoColor=white)

**English** | [Tiếng Việt](README.vi.md)

Real-time head-pose and gaze-direction estimation from multiple IP cameras. GPU-heavy video inference runs in a C++ DeepStream/TensorRT service, while Python/FastAPI handles application logic, smoothing, alerts, camera management, and client delivery.

## Goals

- Process multiple RTSP streams through one MediaMTX pull point per camera.
- Use NVIDIA GPU/NVMM zero-copy processing where supported.
- Detect people/faces, track objects, and run FaceMesh TensorRT inference.
- Add, remove, and list cameras while the pipeline is running.
- Publish raw landmarks and calculated pose/gaze data through MQTT.
- Provide REST, WebSocket, and WebRTC/WHEP integrations.
- Keep credentials, thresholds, addresses, and model paths deployment-configurable.

## Architecture

The runtime is split into four components:

| Component | Responsibility | Interfaces |
| --- | --- | --- |
| mediamtx | Single camera pull point, RTSP proxy, WebRTC/WHEP delivery | RTSP :8554, WebRTC :8889, API :9997 |
| vision-pipeline | C++17 GStreamer/DeepStream, PeopleNet, tracking, FaceMesh TensorRT | gRPC :50051, MQTT |
| analytics-api | FastAPI, pose/gaze algorithms, filters, alerts, camera proxy, WebSocket | HTTP :8080, gRPC, MQTT |
| mqtt-broker | Mosquitto message bus for continuous metadata and results | MQTT :1883 |

~~~mermaid
flowchart LR
    Camera[IP Camera / RTSP] --> MTX[MediaMTX<br/>single pull point]
    MTX -->|Internal RTSP| Vision[vision-pipeline<br/>C++ / DeepStream / TensorRT]
    Vision -->|gaze/cam_id/metadata| MQTT[(Mosquitto MQTT)]
    MQTT --> Analytics[analytics-api<br/>Python / FastAPI]
    Analytics -->|REST + WebSocket| Client[Dashboard / Client]
    MTX -->|WebRTC / WHEP| Client
    Analytics -->|MediaMTX API| MTX
    Analytics -->|gRPC Add/Remove/ListCamera| Vision
~~~

### Interactive diagrams

| Diagram | Purpose | Live view | Source |
| --- | --- | --- | --- |
| Architecture | Components, protocols, and data/control planes | [Open on GitHub Pages](https://quanganh2k4.github.io/head-pose-estimation/headpose-architecture.html) | [HTML](headpose-architecture.html) · [JSON](headpose-architecture.archify.json) |
| Sequence | Camera registration and real-time telemetry | [Open on GitHub Pages](https://quanganh2k4.github.io/head-pose-estimation/headpose-sequence.html) | [HTML](headpose-sequence.html) · [JSON](headpose-sequence.archify.json) |

The live links are published automatically by [Deploy Archify diagrams](.github/workflows/deploy-diagrams.yml). Enable **Settings → Pages → Source: GitHub Actions** once for the repository.

Detailed control-plane/data-plane design, ownership rules, debugging flow, and extension guidance are in [docs/architecture.md](docs/architecture.md).

### Data flow

1. MediaMTX pulls the physical camera stream and exposes one internal path.
2. vision-pipeline reads that path through GStreamer/DeepStream.
3. The pipeline detects people/faces, tracks objects, crops face regions, and runs FaceMesh TensorRT.
4. The pad probe publishes landmark metadata to gaze/<camera_id>/metadata.
5. analytics-api validates landmarks, calculates pose/gaze, applies smoothing and alert rules, and publishes gaze/<camera_id>/calculated.
6. Clients receive telemetry over WebSocket and video from MediaMTX through WebRTC/WHEP.

gRPC handles lifecycle commands such as AddCamera, RemoveCamera, and ListCameras. MQTT carries continuous/event data; it is not used for camera lifecycle control. Physical cameras should be pulled through MediaMTX rather than directly by the pipeline.

### Repository responsibilities

- services/vision_pipeline/src/main.cpp: startup, MQTT, gRPC server, and GLib main loop.
- services/vision_pipeline/src/pipeline.cpp: GStreamer/DeepStream construction, nvstreammux, detector/tracker, and dynamic sources.
- services/vision_pipeline/src/probe_processor.cpp: DeepStream metadata, face crops, TensorRT inference, landmark validation, and MQTT.
- services/analytics_api/main.py: FastAPI, MQTT subscription, gRPC client, MediaMTX registration, state, and alerts.
- services/analytics_api/modules/head_pose_algo.py: landmark validation, spherical morphing, Euler conversion, and One Euro filtering.
- services/analytics_api/tests/: hardware-independent analytics tests.

## Technology

- C++17, CMake, GStreamer, NVIDIA DeepStream 7.0, TensorRT, OpenCV, and Mosquitto.
- Python 3.10+, FastAPI, Paho MQTT, gRPC, NumPy, and SciPy.
- MediaMTX and Docker Compose for infrastructure.
- NVIDIA Jetson Orin/Xavier/Nano or a compatible Linux NVIDIA GPU.

## Repository layout

~~~text
.
├── api/proto/                 # Shared gRPC contract
├── configs/                   # MediaMTX, Mosquitto, and pipeline configuration
├── deployments/               # Docker Compose, systemd, and environment examples
├── docs/                      # Public architecture, API, and operational docs
├── models/                    # Model configuration and local model artifacts
├── services/
│   ├── vision_pipeline/       # C++ DeepStream/TensorRT service
│   └── analytics_api/         # Python FastAPI service and algorithms
├── tools/                     # RTSP simulator, viewer, and utilities
├── benchmarks/                # Local-only legacy benchmark material
├── Makefile
├── README.md                 # Main English documentation
└── README.vi.md              # Vietnamese translation
~~~

## Requirements

- Docker and Docker Compose.
- NVIDIA Container Toolkit and a compatible NVIDIA runtime for production GPU deployment.
- Python 3.10+ for tests or tools outside containers.
- An RTSP-capable IP camera or a video source for the simulator.
- DeepStream/TensorRT libraries and locally supplied model files for the C++ pipeline.
- Network access between analytics-api, MediaMTX, the MQTT broker, and vision-pipeline.

## Configuration

Create a local environment file:

~~~bash
cp deployments/.env.example deployments/.env
~~~

Configure camera URLs and deployment-specific endpoints. Important settings include CAMERA_URLS, MEDIAMTX_API, MEDIAMTX_RTSP_HOST, DEEPSTREAM_GRPC_SERVER, MQTT_BROKER, MQTT_PORT, YAW_ALERT_DEG, PITCH_ALERT_DEG, and ALERT_DURATION_S.

The standalone viewer gets proxied camera URLs from analytics-api. If the API is unavailable, set CAM0_RTSP_URL and/or CAM1_RTSP_URL locally; these values are never committed.

Camera URL sources:

- Preferred: add a camera with `POST /cameras/add`; the viewer reads the resulting MediaMTX-proxied URL from `--app-api` (default `http://127.0.0.1:8080`).
- Standalone viewer: copy the RTSP URL from the camera/NVR administration page into local environment variables. Do not put real credentials in source code:

~~~bash
export CAM0_RTSP_URL='rtsp://<user>:<password>@<camera-host>:<port>/<stream-path>'
python tools/gaze_visualizer_gui.py --camera cam0
~~~

On PowerShell, use `$env:CAM0_RTSP_URL = 'rtsp://<user>:<password>@<camera-host>:<port>/<stream-path>'`. Run `python tools/gaze_visualizer_gui.py --camera-url-help` for the same guidance.

Model and log paths support environment overrides where applicable. Never commit camera credentials, .env files, generated TensorRT engines, model weights, or private deployment artifacts.

## Running the system

### Production on Jetson

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

With no physical camera, use the simulator:

~~~bash
python tools/rtsp_simulator.py \
  --input sample_video.mp4 \
  --url rtsp://localhost:8554/cam0
~~~

The development stack is intended for API/infrastructure work. The production pipeline additionally requires the Jetson/NVIDIA runtime and local DeepStream model artifacts.

## API and protocol

The complete contract is documented in [docs/api_reference.md](docs/api_reference.md).

| Type | Address/topic | Purpose |
| --- | --- | --- |
| REST | GET /health | Health and connectivity status |
| REST | GET /cameras | List active cameras |
| REST | POST /cameras/add | Add a camera at runtime |
| REST | DELETE /cameras/{src_id} | Remove a camera at runtime |
| REST | GET /gaze/latest | Latest payload for all cameras |
| REST | GET /gaze/{cam_id}/history | Recent calculated frames for one camera |
| REST | GET /alerts | Recent alert records |
| WebSocket | /ws/gaze/{cam_id} | Live pose/gaze updates |
| MQTT | gaze/{cam_id}/metadata | Raw landmarks and frame metadata |
| MQTT | gaze/{cam_id}/calculated | Filtered pose/gaze and alert data |
| gRPC | :50051 | Camera lifecycle management |

Example camera registration:

~~~bash
curl -X POST http://localhost:8080/cameras/add \
  -H 'Content-Type: application/json' \
  -d '{"url":"rtsp://camera-host:554/stream"}'
~~~

No real camera footage is included in this repository. Demo media should only be added when you have permission to publish it; use the RTSP simulator with synthetic or properly licensed footage for local testing.

## Testing and quality

~~~bash
make test
make lint
make format
~~~

The hardware-independent tests cover One Euro filtering, landmark validation, occlusion handling, spherical pose estimation, angle signs, rotation matrices, and stability on synthetic static/moving faces.

When changing C++ code, rebuild the relevant Docker image and verify tensor shapes, batch size, GPU memory, MQTT topics, gRPC contracts, and source lifecycle behavior.

## Maintenance rules

- Edit source proto files; never hand-edit generated Python/C++ stubs.
- Keep C++ focused on throughput, buffer ownership, GPU/NVMM handling, and source lifecycle.
- Keep business logic, smoothing, alerts, and HTTP/WebSocket concerns in analytics-api.
- Route physical camera pulls through MediaMTX.
- Use environment variables for addresses, ports, model paths, credentials, and thresholds.
- Update docs/api_reference.md when endpoint/topic contracts change.
- Add hardware-independent tests under services/analytics_api/tests/.
- Do not commit private deep-dive notes, legacy benchmarks, credentials, model weights, or build artifacts.

## Related documentation

- [Architecture and design](docs/architecture.md)
- [API reference](docs/api_reference.md)
- [Dynamic pad manipulation](docs/dynamic_pad_manipulation.md)
- [Benchmarks](docs/benchmarks.md)
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## License

Released under the [MIT License](LICENSE). The license does not cover proprietary third-party DeepStream sample code or model artifacts; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
