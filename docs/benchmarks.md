# Performance Benchmarks & Hardware Sizing

Performance evaluation comparing the initial Monolithic Python prototype with the current Native C++ Microservices Architecture on NVIDIA Jetson.

## Comparative Performance Metrics

| Architecture | End-to-End Latency | Max Camera Streams (1080p @ 30fps) | GPU Utilization | Host CPU Load | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Python Monolithic (PoC)** | ~85 ms | 2 Streams (GIL Bottleneck) | ~45% | ~85% (High) | ~2.1 GB |
| **C++ Native Microservices** | **~18 ms** | **6 - 8 Streams** | **~85% (Optimal)**| **~22% (Low)** | **~1.2 GB** |

## Profiling Breakdown (per frame on Jetson Orin Nano / NX)

1. **Hardware Decode & NVMM Buffering (`nvv4l2decoder`):** ~2.5 ms
2. **Primary Detector (`PeopleNet` INT8/FP16):** ~4.2 ms
3. **NvDCF Multi-Object Tracking:** ~1.8 ms
4. **GPU Crop & Normalize (`NvBufSurfTransform`):** ~0.9 ms
5. **Secondary Inference (`FaceMesh` TensorRT FP16):** ~3.4 ms
6. **MQTT Pub/Sub Event Transport:** ~1.5 ms
7. **Spherical Morphing & One-Euro Filter (FastAPI):** ~3.2 ms
8. **Total End-to-End Pipeline Latency:** **~17.5 ms**
