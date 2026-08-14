# System Architecture & Data Flow

This document details the decoupled microservices architecture of the Real-Time Multi-Camera Head Pose and Gaze Estimation System on NVIDIA Jetson.

```mermaid
graph LR
    subgraph LAN["Local Network (LAN)"]
        CAM["📷 IP Cameras 1..N<br/>(RTSP, H264/H265)"]
        VIEWER["💻 Client / Viewer GUI<br/>REST · WebSocket · WebRTC"]
    end

    subgraph JETSON["NVIDIA Jetson Host (Docker Compose)"]
        MTX["MediaMTX Gateway<br/>RTSP Proxy :8554 · API :9997<br/>WebRTC/WHEP :8889"]

        subgraph VISION["vision-pipeline (C++17)"]
            GRPC["gRPC Server :50051<br/>Add/Remove/List Camera"]
            PIPE["GStreamer Pipeline<br/>rtspsrc → nvstreammux →<br/>PeopleNet (nvinfer) → NvDCF tracker"]
            PROBE["Custom Pad Probe<br/>GPU Crop (NvBufSurfTransform)<br/>FaceMesh TensorRT C++ API"]
        end

        MQTT["MQTT Broker<br/>Mosquitto :1883"]

        subgraph APP["analytics-api (Python / FastAPI)"]
            ALGO["Spherical Morphing<br/>+ One-Euro Filter"]
            ALERT["Distraction Alert Engine"]
            API["REST API :8080<br/>WebSocket /ws/gaze"]
        end
    end

    CAM -->|"Single RTSP Ingest Point"| MTX
    MTX -->|"Proxied Internal RTSP"| PIPE
    PIPE --> PROBE
    PROBE -->|"Landmark JSON (gaze/+/metadata)"| MQTT
    MQTT --> ALGO
    ALGO --> ALERT
    ALGO -->|"Smoothed Pose (gaze/+/calculated)"| MQTT
    API -->|"Register Stream Path"| MTX
    API -->|"AddCamera / RemoveCamera"| GRPC
    GRPC -.->|"Dynamic Pad Link/Unlink"| PIPE
    VIEWER <--> API
    MTX -->|"WebRTC Video Stream"| VIEWER
```

## Core Design Principles

1. **Decoupling AI Inference from Business Logic:**
   - **`vision-pipeline` (C++):** Dedicated to high-throughput hardware-accelerated video decoding, multi-stream multiplexing, object detection, and landmark inference. Zero Python GIL interference.
   - **`analytics-api` (Python):** Handles alert logic, pose smoothing filtering, REST endpoints, and WebSocket broadcasting.

2. **Zero-Copy Hardware Acceleration:**
   - Uses `NvBufSurfTransform` to perform cropping and color space conversion directly in NVMM memory on the GPU before passing tensors to TensorRT.

3. **Dynamic Stream Management:**
   - Addition or removal of RTSP streams happens at runtime via gRPC and dynamic GStreamer pad linking without interrupting existing streams.
