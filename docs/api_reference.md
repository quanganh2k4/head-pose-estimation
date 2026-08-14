# API Reference & Protocols

## 1. REST Endpoints (`analytics-api` :8080)

- `GET /cameras`
  - Returns a list of active cameras and their stream URLs.
- `POST /cameras/add`
  - Body: `{"url": "rtsp://camera-ip/stream"}`
  - Registers stream with MediaMTX and invokes gRPC `AddCamera` on `vision-pipeline`.
- `DELETE /cameras/{cam_id}`
  - Removes stream from MediaMTX and invokes gRPC `RemoveCamera`.
- `GET /history/{cam_id}`
  - Returns historical pose & alert telemetry.

## 2. WebSocket Endpoint

- `GET /ws/gaze`
  - Streams real-time head pose (pitch, yaw, roll), landmark coordinates, and distraction alert statuses.

## 3. gRPC Service (`vision-pipeline` :50051)

Defined in `api/proto/camera_service.proto`:
- `rpc AddCamera(CameraAddRequest) returns (CameraAddResponse)`
- `rpc RemoveCamera(CameraRemoveRequest) returns (CameraRemoveResponse)`
- `rpc ListCameras(CameraListRequest) returns (CameraListResponse)`

## 4. MQTT Topics (Mosquitto :1883)

- `gaze/{cam_id}/metadata`: Raw facial landmarks detected by DeepStream pad probe.
- `gaze/{cam_id}/calculated`: Smoothed head pose angles and distraction alert notifications.
