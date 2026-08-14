# System Architecture & Data Flow

This document details the decoupled microservices architecture of the Real-Time Multi-Camera Head Pose and Gaze Estimation System on NVIDIA Jetson.

``![Diagram](https://mermaid.ink/img/eyJjb2RlIjogImdyYXBoIExSXG4gICAgc3ViZ3JhcGggTEFOW1wiTG9jYWwgTmV0d29yayAoTEFOKVwiXVxuICAgICAgICBDQU1bXCJcdWQ4M2RcdWRjZjcgSVAgQ2FtZXJhcyAxLi5OPGJyLz4oUlRTUCwgSDI2NC9IMjY1KVwiXVxuICAgICAgICBWSUVXRVJbXCJcdWQ4M2RcdWRjYmIgQ2xpZW50IC8gVmlld2VyIEdVSTxici8+UkVTVCBcdTAwYjcgV2ViU29ja2V0IFx1MDBiNyBXZWJSVENcIl1cbiAgICBlbmRcblxuICAgIHN1YmdyYXBoIEpFVFNPTltcIk5WSURJQSBKZXRzb24gSG9zdCAoRG9ja2VyIENvbXBvc2UpXCJdXG4gICAgICAgIE1UWFtcIk1lZGlhTVRYIEdhdGV3YXk8YnIvPlJUU1AgUHJveHkgOjg1NTQgXHUwMGI3IEFQSSA6OTk5Nzxici8+V2ViUlRDL1dIRVAgOjg4ODlcIl1cblxuICAgICAgICBzdWJncmFwaCBWSVNJT05bXCJ2aXNpb24tcGlwZWxpbmUgKEMrKzE3KVwiXVxuICAgICAgICAgICAgR1JQQ1tcImdSUEMgU2VydmVyIDo1MDA1MTxici8+QWRkL1JlbW92ZS9MaXN0IENhbWVyYVwiXVxuICAgICAgICAgICAgUElQRVtcIkdTdHJlYW1lciBQaXBlbGluZTxici8+cnRzcHNyYyBcdTIxOTIgbnZzdHJlYW1tdXggXHUyMTkyPGJyLz5QZW9wbGVOZXQgKG52aW5mZXIpIFx1MjE5MiBOdkRDRiB0cmFja2VyXCJdXG4gICAgICAgICAgICBQUk9CRVtcIkN1c3RvbSBQYWQgUHJvYmU8YnIvPkdQVSBDcm9wIChOdkJ1ZlN1cmZUcmFuc2Zvcm0pPGJyLz5GYWNlTWVzaCBUZW5zb3JSVCBDKysgQVBJXCJdXG4gICAgICAgIGVuZFxuXG4gICAgICAgIE1RVFRbXCJNUVRUIEJyb2tlcjxici8+TW9zcXVpdHRvIDoxODgzXCJdXG5cbiAgICAgICAgc3ViZ3JhcGggQVBQW1wiYW5hbHl0aWNzLWFwaSAoUHl0aG9uIC8gRmFzdEFQSSlcIl1cbiAgICAgICAgICAgIEFMR09bXCJTcGhlcmljYWwgTW9ycGhpbmc8YnIvPisgT25lLUV1cm8gRmlsdGVyXCJdXG4gICAgICAgICAgICBBTEVSVFtcIkRpc3RyYWN0aW9uIEFsZXJ0IEVuZ2luZVwiXVxuICAgICAgICAgICAgQVBJW1wiUkVTVCBBUEkgOjgwODA8YnIvPldlYlNvY2tldCAvd3MvZ2F6ZVwiXVxuICAgICAgICBlbmRcbiAgICBlbmRcblxuICAgIENBTSAtLT58XCJTaW5nbGUgUlRTUCBJbmdlc3QgUG9pbnRcInwgTVRYXG4gICAgTVRYIC0tPnxcIlByb3hpZWQgSW50ZXJuYWwgUlRTUFwifCBQSVBFXG4gICAgUElQRSAtLT4gUFJPQkVcbiAgICBQUk9CRSAtLT58XCJMYW5kbWFyayBKU09OIChnYXplLysvbWV0YWRhdGEpXCJ8IE1RVFRcbiAgICBNUVRUIC0tPiBBTEdPXG4gICAgQUxHTyAtLT4gQUxFUlRcbiAgICBBTEdPIC0tPnxcIlNtb290aGVkIFBvc2UgKGdhemUvKy9jYWxjdWxhdGVkKVwifCBNUVRUXG4gICAgQVBJIC0tPnxcIlJlZ2lzdGVyIFN0cmVhbSBQYXRoXCJ8IE1UWFxuICAgIEFQSSAtLT58XCJBZGRDYW1lcmEgLyBSZW1vdmVDYW1lcmFcInwgR1JQQ1xuICAgIEdSUEMgLS4tPnxcIkR5bmFtaWMgUGFkIExpbmsvVW5saW5rXCJ8IFBJUEVcbiAgICBWSUVXRVIgPC0tPiBBUElcbiAgICBNVFggLS0+fFwiV2ViUlRDIFZpZGVvIFN0cmVhbVwifCBWSUVXRVIiLCAibWVybWFpZCI6IHsidGhlbWUiOiAiZGVmYXVsdCJ9fQ==)

<details><summary>🔍 Xem mã nguồn Mermaid</summary>

`mermaid
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
`

</details>``

## Core Design Principles

1. **Decoupling AI Inference from Business Logic:**
   - **`vision-pipeline` (C++):** Dedicated to high-throughput hardware-accelerated video decoding, multi-stream multiplexing, object detection, and landmark inference. Zero Python GIL interference.
   - **`analytics-api` (Python):** Handles alert logic, pose smoothing filtering, REST endpoints, and WebSocket broadcasting.

2. **Zero-Copy Hardware Acceleration:**
   - Uses `NvBufSurfTransform` to perform cropping and color space conversion directly in NVMM memory on the GPU before passing tensors to TensorRT.

3. **Dynamic Stream Management:**
   - Addition or removal of RTSP streams happens at runtime via gRPC and dynamic GStreamer pad linking without interrupting existing streams.
