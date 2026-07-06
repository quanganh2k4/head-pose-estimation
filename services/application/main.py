#!/usr/bin/env python3
"""
application/main.py — Application Service (python:3.10-slim, không cần GPU)
Chức năng:
  - Subscribe MQTT: gaze/+/metadata → business logic
  - REST API (FastAPI): proxy camera management → deepstream service
  - Dashboard data endpoint
  - Log shipper đến VPS
"""
import os
import json
import time
import asyncio
import hashlib
import threading
import requests
from collections import deque, defaultdict

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import grpc
import camera_pb2
import camera_pb2_grpc
from modules.head_pose_algo import SphericalHeadPoseEstimator

# ── Config ───────────────────────────────────────────────────────────────────────
MQTT_BROKER            = os.environ.get("MQTT_BROKER", "mqtt-broker")
MQTT_PORT              = int(os.environ.get("MQTT_PORT", "1883"))
DEEPSTREAM_GRPC_SERVER = os.environ.get("DEEPSTREAM_GRPC_SERVER", "127.0.0.1:50051")
APP_PORT               = int(os.environ.get("APP_PORT", "8080"))
LOG_SERVER_URL         = os.environ.get("LOG_SERVER_URL", "")
DEVICE_ID              = os.environ.get("DEVICE_ID", "jetson-edge")
MEDIAMTX_API           = os.environ.get("MEDIAMTX_API", "http://127.0.0.1:9997")
# Dùng cho deepstream-service (cùng host, network_mode: host) khi gọi AddCamera
MEDIAMTX_RTSP_HOST     = os.environ.get("MEDIAMTX_RTSP_HOST", "127.0.0.1:8554")
# Dùng cho viewer bên ngoài (ví dụ gaze_viewer.py chạy trên laptop khác) — phải là
# IP/hostname LAN thật của Jetson, KHÔNG phải 127.0.0.1
MEDIAMTX_PUBLIC_RTSP_HOST = os.environ.get("MEDIAMTX_PUBLIC_RTSP_HOST", MEDIAMTX_RTSP_HOST)
# WebRTC (WHEP) — browser trên laptop kết nối thẳng vào mediamtx, không qua application
MEDIAMTX_PUBLIC_WEBRTC_HOST = os.environ.get("MEDIAMTX_PUBLIC_WEBRTC_HOST",
                                              MEDIAMTX_PUBLIC_RTSP_HOST.split(":")[0] + ":8889")
CAMERA_URLS            = [u for u in os.environ.get("CAMERA_URLS", "").split(",") if u]
# Hệ thống chạy duy nhất phương pháp gốc Spherical Morphing của bài báo.

# ── gRPC Client setup ─────────────────────────────────────────────────────────────
_grpc_channel = grpc.insecure_channel(DEEPSTREAM_GRPC_SERVER)
_grpc_stub    = camera_pb2_grpc.CameraServiceStub(_grpc_channel)

# ── In-memory store ───────────────────────────────────────────────────────────────
_latest: dict             = {}  # {cam_id: last payload}
_history: dict            = defaultdict(lambda: deque(maxlen=300))  # {cam_id: deque of payloads}
_alerts: list             = []  # recent alerts
_store_lock               = threading.Lock()

# Alert thresholds
YAW_ALERT_DEG      = float(os.environ.get("YAW_ALERT_DEG",   "45.0"))
PITCH_ALERT_DEG    = float(os.environ.get("PITCH_ALERT_DEG", "30.0"))
ALERT_DURATION_S   = float(os.environ.get("ALERT_DURATION_S", "3.0"))

# Track consecutive alert frames per (cam, person)
_alert_tracker: dict = defaultdict(lambda: {"count": 0, "since": 0.0, "fired": False})
# Track YoloHeadPoseEstimator instance per (cam, person)
_pose_estimators: dict = {}

# ── WebSocket relay (cho viewer WebRTC/WHEP, Phase 3) ──────────────────────────────
# MQTT callback (_on_message) chạy trên thread riêng của paho-mqtt, không phải asyncio
# event loop của uvicorn — cần asyncio.run_coroutine_threadsafe để gửi qua WebSocket
# một cách an toàn giữa 2 thread.
_ws_clients: dict = defaultdict(set)  # {cam_id: set of WebSocket}
_ws_lock = threading.Lock()
_main_loop: asyncio.AbstractEventLoop = None


def _broadcast_ws(cam_id: str, payload: dict):
    if _main_loop is None:
        return
    with _ws_lock:
        clients = list(_ws_clients.get(cam_id, ()))
    if not clients:
        return
    data = json.dumps(payload, separators=(',', ':'))

    async def _send_all():
        dead = []
        for ws in clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        if dead:
            with _ws_lock:
                for ws in dead:
                    _ws_clients[cam_id].discard(ws)

    asyncio.run_coroutine_threadsafe(_send_all(), _main_loop)


def _check_alerts(cam_id: str, detections: list, ts_ms: int):
    ts_s = ts_ms / 1000.0
    for i, det in enumerate(detections):
        person_id = det.get("id", i)
        key   = (cam_id, person_id)
        yaw   = abs(det.get("yaw",   0.0))
        pitch = abs(det.get("pitch", 0.0))
        alert_condition = yaw > YAW_ALERT_DEG or pitch > PITCH_ALERT_DEG

        tr = _alert_tracker[key]
        if alert_condition:
            if tr["count"] == 0: tr["since"] = ts_s
            tr["count"] += 1
            duration = ts_s - tr["since"]
            if duration >= ALERT_DURATION_S and not tr["fired"]:
                tr["fired"] = True
                alert = {
                    "cam": cam_id, "person": person_id,
                    "yaw": round(det.get("yaw", 0.0), 1),
                    "pitch": round(det.get("pitch", 0.0), 1),
                    "duration_s": round(duration, 1),
                    "ts": ts_ms,
                }
                with _store_lock:
                    _alerts.append(alert)
                    if len(_alerts) > 500: _alerts.pop(0)
                print(f"[ALERT] cam={cam_id} person={person_id} "
                      f"yaw={alert['yaw']} pitch={alert['pitch']} for {alert['duration_s']}s")
        else:
            tr["count"] = 0; tr["fired"] = False


# ── MQTT subscriber ───────────────────────────────────────────────────────────────
_mqtt_client = mqtt.Client(CallbackAPIVersion.VERSION1, client_id="app-service", clean_session=True)

def _on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe("gaze/+/metadata", qos=0)
        client.subscribe("gaze/+/health",   qos=0)
        print(f"[MQTT] Connected, subscribed gaze/+/metadata")
    else:
        print(f"[MQTT] Connect failed rc={rc}")

def _on_message(client, userdata, msg):
    try:
        cam_id = msg.topic.split("/")[1]  # "gaze/cam0/metadata" → "cam0"
        payload = json.loads(msg.payload)
        
        detections = payload.get("d", [])
        for i, det in enumerate(detections):
            pts = det.get("pts")
            if pts is not None:
                person_id = det.get("id", i)
                key = (cam_id, person_id)
                estimator = _pose_estimators.get(key)
                if estimator is None:
                    # Chạy duy nhất thuật toán Spherical Morphing của bài báo gốc
                    estimator = SphericalHeadPoseEstimator(smooth_alpha=0.20, point_alpha=0.65)
                    _pose_estimators[key] = estimator

                def pt(name):
                    v = np.array(pts[name], dtype=np.float64)
                    v[1] = -v[1]  # ảnh Y hướng xuống, model Y hướng lên
                    return v
                try:
                    # Phương pháp gốc bài báo: 5 điểm 2D (bỏ Z), khớp với PAPER_3D_MODEL
                    m_pts = np.vstack([pt("nose"), pt("chin"), pt("left_eye"), pt("right_eye"), pt("bridge")])

                    pose = estimator.update_points(m_pts, ts_ms=payload.get("ts"))
                    yaw = pose["yaw"]
                    pitch = pose["pitch"]
                    R = pose["R_smooth"]
                    conf = pose["confidence"]

                    det["yaw"] = round(float(yaw), 1)
                    det["pitch"] = round(float(pitch), 1)
                    det["gaze"] = [round(float(R[0,2]), 3), round(-float(R[1,2]), 3), round(float(R[2,2]), 3)]
                    det["conf"] = round(float(conf), 2)

                    # In log rõ ràng cho thuật toán bao gồm confidence
                    print(f"[MP:spherical] cam={cam_id} obj={person_id} yaw={yaw:+.1f} pitch={pitch:+.1f} conf={conf:.1f}")
                except Exception as ex:
                    print(f"[ERROR] Pose estimation failed for cam={cam_id} obj={person_id}: {ex}")

        with _store_lock:
            _latest[cam_id] = payload
            _history[cam_id].append(payload)
        _check_alerts(cam_id, detections, payload.get("ts", 0))
        _broadcast_ws(cam_id, payload)

        try:
            client.publish(f"gaze/{cam_id}/calculated", json.dumps(payload, separators=(',',':')), qos=0)
        except Exception as pe:
            print(f"[MQTT Publish Error] {cam_id}: {pe}")
    except Exception as e:
        print(f"[MQTT] Parse error: {e}")

_mqtt_client.on_connect = _on_connect
_mqtt_client.on_message = _on_message
_mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)


# ── FastAPI app ───────────────────────────────────────────────────────────────────
api = FastAPI(title="HeadPose Application API", version="2.0.0")
api.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@api.on_event("startup")
async def _capture_event_loop():
    global _main_loop
    _main_loop = asyncio.get_running_loop()


@api.websocket("/ws/gaze/{cam_id}")
async def ws_gaze(websocket: WebSocket, cam_id: str):
    await websocket.accept()
    with _ws_lock:
        _ws_clients[cam_id].add(websocket)
    try:
        while True:
            # Không cần dữ liệu từ client, chỉ giữ kết nối sống và phát hiện khi đóng.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            _ws_clients[cam_id].discard(websocket)


@api.get("/config")
def get_config():
    """Cấu hình cho frontend viewer (WHEP host, v.v.) — tránh hardcode trong JS."""
    return {"mediamtx_webrtc_host": MEDIAMTX_PUBLIC_WEBRTC_HOST}


@api.get("/viewer", response_class=HTMLResponse)
def viewer_page():
    static_path = os.path.join(os.path.dirname(__file__), "static", "viewer.html")
    return FileResponse(static_path)


# ── Dashboard endpoints ────────────────────────────────────────────────────────────
@api.get("/health")
def health():
    return {"status": "ok", "mqtt_connected": _mqtt_client.is_connected(),
            "active_cameras": list(_latest.keys())}

@api.get("/gaze/latest")
def gaze_latest():
    with _store_lock:
        return {"data": dict(_latest)}

@api.get("/gaze/{cam_id}/latest")
def gaze_cam_latest(cam_id: str):
    with _store_lock:
        if cam_id not in _latest:
            raise HTTPException(404, f"Camera {cam_id} not found")
        return _latest[cam_id]

@api.get("/gaze/{cam_id}/history")
def gaze_cam_history(cam_id: str, n: int = 60):
    with _store_lock:
        data = list(_history.get(cam_id, []))[-n:]
    return {"cam_id": cam_id, "frames": data}

@api.get("/alerts")
def get_alerts(limit: int = 50):
    with _store_lock:
        return {"alerts": _alerts[-limit:]}


# ── MediaMTX camera routing ─────────────────────────────────────────────────────
# mediamtx là điểm pull RTSP DUY NHẤT từ camera thật. Trước khi deepstream-service
# nhận một camera, ta đăng ký path tương ứng trên mediamtx rồi đưa nó URL đã proxy
# (rtsp://mediamtx:8554/<path>) thay vì URL camera gốc — tránh camera bị pull 2 lần
# (DeepStream + bất kỳ viewer nào khác) và tránh cạn kết nối đồng thời trên camera.
_camera_mediamtx_paths: dict = {}  # {src_id: mediamtx_path_name}


def _mediamtx_path_name(url: str) -> str:
    # Tên path phải ổn định theo URL (idempotent khi add lại cùng 1 camera) và
    # không lộ credential ra ngoài (path name không chứa user/pass của URL gốc).
    return "cam_" + hashlib.md5(url.encode("utf-8")).hexdigest()[:10]


def _mediamtx_register(path_name: str, source_url: str) -> bool:
    body = {"source": source_url, "sourceOnDemand": False}
    try:
        r = requests.post(f"{MEDIAMTX_API}/v3/config/paths/add/{path_name}", json=body, timeout=5)
        if r.status_code < 300:
            return True
        # Path đã tồn tại từ lần add trước (ví dụ application restart) → cập nhật lại.
        r = requests.patch(f"{MEDIAMTX_API}/v3/config/paths/patch/{path_name}", json=body, timeout=5)
        return r.status_code < 300
    except requests.RequestException as e:
        print(f"[MediaMTX] Register path {path_name} failed: {e}")
        return False


def _mediamtx_unregister(path_name: str):
    try:
        requests.delete(f"{MEDIAMTX_API}/v3/config/paths/delete/{path_name}", timeout=5)
    except requests.RequestException as e:
        print(f"[MediaMTX] Unregister path {path_name} failed: {e}")


def _add_camera(url: str, retries: int = 1, retry_delay_s: float = 2.0) -> dict:
    """Đăng ký path trên mediamtx rồi gọi gRPC AddCamera xuống deepstream-service
    với URL đã proxy. Dùng chung cho cả camera khởi động (CAMERA_URLS) và REST API."""
    path_name = _mediamtx_path_name(url)
    if not _mediamtx_register(path_name, url):
        raise RuntimeError(f"Failed to register mediamtx path for {url}")

    proxied_url = f"rtsp://{MEDIAMTX_RTSP_HOST}/{path_name}"
    last_err = None
    for attempt in range(retries + 1):
        try:
            response = _grpc_stub.AddCamera(camera_pb2.CameraAddRequest(url=proxied_url))
            _camera_mediamtx_paths[response.src_id] = path_name
            return {"src_id": response.src_id, "url": response.url, "status": response.status}
        except grpc.RpcError as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_delay_s)
    _mediamtx_unregister(path_name)
    raise last_err


# ── Camera management (proxy → deepstream service, qua mediamtx) ──────────────────
class CameraAddRequest(BaseModel):
    url: str
    name: str = ""

class CameraRemoveRequest(BaseModel):
    src_id: int

@api.get("/cameras")
def cameras_list():
    try:
        response = _grpc_stub.ListCameras(camera_pb2.CameraListRequest())
        cameras = []
        for c in response.cameras:
            path_name = _camera_mediamtx_paths.get(c.src_id)
            public_url = f"rtsp://{MEDIAMTX_PUBLIC_RTSP_HOST}/{path_name}" if path_name else None
            cameras.append({
                "src_id": c.src_id, "url": c.url,
                "mediamtx_path": path_name,
                "mediamtx_rtsp_url": public_url,
            })
        return {"cameras": cameras}
    except grpc.RpcError as e:
        raise HTTPException(status_code=503, detail=f"DeepStream gRPC error: {e.details()}")

@api.post("/cameras/add")
def camera_add(req: CameraAddRequest):
    try:
        return _add_camera(req.url)
    except grpc.RpcError as e:
        raise HTTPException(status_code=503, detail=f"DeepStream gRPC error: {e.details()}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

@api.delete("/cameras/{src_id}")
def camera_remove(src_id: int):
    try:
        response = _grpc_stub.RemoveCamera(camera_pb2.CameraRemoveRequest(src_id=src_id))
        path_name = _camera_mediamtx_paths.pop(src_id, None)
        if path_name:
            _mediamtx_unregister(path_name)
        return {
            "src_id": response.src_id,
            "status": response.status
        }
    except grpc.RpcError as e:
        raise HTTPException(status_code=503, detail=f"DeepStream gRPC error: {e.details()}")

@api.post("/cameras/{src_id}/remove")
def camera_remove_post(src_id: int):
    return camera_remove(src_id)


# ── Log shipper ────────────────────────────────────────────────────────────────────
def _log_shipper():
    if not LOG_SERVER_URL:
        return
    while True:
        time.sleep(5)
        try:
            with _store_lock:
                alerts_snap = list(_alerts[-20:])
                latest_snap = dict(_latest)
            requests.post(LOG_SERVER_URL, json={
                "device_id": DEVICE_ID,
                "ts": int(time.time() * 1000),
                "alerts": alerts_snap,
                "cameras": list(latest_snap.keys()),
            }, timeout=5)
        except Exception as e:
            print(f"[SHIPPER] {e}")


# ── Entry point ──────────────────────────────────────────────────────────────────
def main():
    # MQTT
    try:
        _mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        _mqtt_client.loop_start()
    except Exception as e:
        print(f"[MQTT] Initial connect failed: {e} (will retry...)")
        threading.Thread(target=lambda: _mqtt_client.reconnect(), daemon=True).start()

    # Log shipper
    threading.Thread(target=_log_shipper, daemon=True, name="log-shipper").start()

    # Camera khởi động: đăng ký qua mediamtx rồi add vào deepstream-service.
    # Retry vì deepstream-service/mediamtx có thể chưa sẵn sàng (race điều kiện khởi
    # động container, dù docker-compose đã có depends_on).
    for url in CAMERA_URLS:
        try:
            result = _add_camera(url, retries=5, retry_delay_s=3.0)
            print(f"[APP] Initial camera added: {result}")
        except Exception as e:
            print(f"[APP] Failed to add initial camera {url}: {e}")

    print(f"[APP] Starting on port {APP_PORT}")
    uvicorn.run(api, host="0.0.0.0", port=APP_PORT, log_level="warning")


if __name__ == "__main__":
    main()
