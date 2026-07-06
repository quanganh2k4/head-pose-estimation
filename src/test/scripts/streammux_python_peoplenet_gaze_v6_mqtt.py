#!/usr/bin/env python3
"""
streammux_python_peoplenet_gaze_v5.py
Pipeline DeepStream + pyds OSD với thuật toán Spherical Parametrization
& 3D Morphing (Yuan et al., arXiv:1907.09217).

V5 TEST MODE: Chỉ dùng MediaPipe FaceMesh để lấy landmark
(nose_tip, chin, left/right eye canthus) → tính head-pose và gaze vector.
Không có YOLO sgie fallback để giảm tải tối đa trên Jetson Nano.

Thiết kế cache để tránh lag:
  - Mỗi PROCESS_EVERY_N_FRAMES frame: chạy MediaPipe, lưu pitch/yaw/roll + landmark.
  - Mọi frame: vẽ lại từ cache với face bbox hiện tại.
"""

import sys
import os
import math
import signal
import datetime
import threading
import base64
import queue as _queue_mod
import json
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds
import numpy as np
import paho.mqtt.client as mqtt


# ─── Auto log to file với rotation (tee stdout+stderr → log file) ────────────
_LOG_DIR      = os.environ.get("LOG_DIR", "/home/ivision/headpose-cameraIP/log")
_LOG_PATH     = os.path.join(_LOG_DIR, "gaze_v5_test.log")
_LOG_MAX_MB   = 50   # rotate khi file vượt 50MB

_unshipped_logs = []
_unshipped_logs_lock = threading.Lock()

class _Tee:
    """Ghi đồng thời ra stream gốc và file log, có rotation theo size."""
    def __init__(self, stream, log_path):
        self._stream   = stream
        self._path     = log_path
        self._fh       = open(log_path, "a", encoding="utf-8", buffering=8192)
        self._written  = 0
    def _rotate(self):
        self._fh.close()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.rename(self._path, self._path.replace(".log", f"_{ts}.log"))
        self._fh      = open(self._path, "a", encoding="utf-8", buffering=8192)
        self._written = 0
    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        self._written += len(data)
        if self._written > _LOG_MAX_MB * 1024 * 1024:
            self._rotate()
        
        # Save lines for sending to dashboard
        if data.strip():
            for line in data.strip().split('\n'):
                if line.strip():
                    with _unshipped_logs_lock:
                        _unshipped_logs.append(line.strip())
    def flush(self):
        self._stream.flush()
        self._fh.flush()
    def fileno(self):
        return self._stream.fileno()

os.makedirs(_LOG_DIR, exist_ok=True)
_tee_out   = _Tee(sys.__stdout__, _LOG_PATH)
_tee_out._fh.write(f"\n{'='*60}\n[START] {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n{'='*60}\n")
_tee_out._fh.flush()
sys.stdout = _tee_out
sys.stderr = _Tee(sys.__stderr__, _LOG_PATH)
try:
    import cv2
except Exception:
    cv2 = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import tensorrt as trt
    import pycuda.driver as cuda
except Exception:
    trt = None
    cuda = None

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

if cuda is not None:
    try:
        cuda.init()
    except Exception as e:
        print(f"[CUDA INIT FAILED] {e}")
        cuda = None

USE_GPU_LANDMARK = os.environ.get("USE_GPU_LANDMARK", "1") == "1"

# ─── Hằng số ─────────────────────────────────────────────────────────────────
PROCESS_EVERY_N_FRAMES = 4     # Chạy ONNX landmark mỗi N frame (8 = ~3.75fps ở 30fps input, tiết kiệm CPU)
USE_SPHERICAL_MORPHING = False  # False = linear fast, nhanh hơn cho Jetson Nano
DISABLE_MEDIAPIPE      = "--disable-mediapipe" in sys.argv  # mặc định bật, tắt bằng --disable-mediapipe
FACE_BBOX_PADDING      = 0.5
LANDMARK_SCORE_THRESH  = -0.1  # logit -0.3 ≈ 43% — chấp nhận face nghiêng/góc cao hơn
ONNX_MODEL_PATH = os.environ.get(
    "ONNX_MODEL_PATH",
    "/home/ivision/headpose-cameraIP/models/mediapipe_pose/"
    "20_new_onnx_postprocess_N-batch/face_mesh_192x192_post.onnx",
)
PEOPLENET_CONFIG = os.environ.get(
    "PEOPLENET_CONFIG",
    "/home/ivision/headpose-cameraIP/models/peoplenet/config_infer_peoplenet.txt",
)
HEAD_VERTICAL_SHIFT    = 0.25
LINE_WIDTH             = 3
AXIS_SIZE_RATIO        = 0.40
GAZE_SIZE_RATIO        = 0.55

# RTSP: tcp ổn định hơn udp (tránh vỡ ảnh H.265), latency cao hơn chút
RTSP_URLS = [
    u.strip() for u in os.environ.get(
        "RTSP_URLS",
        "rtsp://user:password@192.168.1.10:1904/stream,"
        "rtsp://user:password@192.168.1.11:1904/stream",
    ).split(",") if u.strip()
]
RTSP_PROTOCOLS = os.environ.get("RTSP_PROTOCOLS", "tcp+udp").lower()
RTSP_PUBLISH_URL = os.environ.get("RTSP_PUBLISH_URL", "")

# ─── Cấu hình MQTT ─────────────────────────────────────────────
MQTT_BROKER = os.environ.get("MQTT_BROKER", "127.0.0.1") # IP Laptop
MQTT_PORT   = int(os.environ.get("MQTT_PORT", "1883"))

mqtt_client = mqtt.Client(client_id="jetson-nano-gaze", clean_session=True)
mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Kết nối thành công tới Broker {MQTT_BROKER}:{MQTT_PORT}")
    else:
        print(f"[MQTT] Kết nối thất bại, rc={rc}")

mqtt_client.on_connect = on_mqtt_connect

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    mqtt_client.loop_start()
except Exception as e:
    print(f"[MQTT] Lỗi kết nối tới broker: {e}")


# Không dùng nvtracker (tiết kiệm CPU/GPU). obj_id=UINT64_MAX → dùng grid position.
_UINT64_MAX    = 18446744073709551615
_ID_GRID_PX    = 80   # Kích thước ô grid pixel để tạo pseudo-ID từ tâm bbox


# ─── State toàn cục ──────────────────────────────────────────────────────────
_frame_counter     = {}
_pose_cache        = {}
_mp_estimators     = {}   # {(src_id, obj_id): YoloHeadPoseEstimator} – PAPER_3D_MODEL
_ort_session       = None
_ort_status_printed = False
_latest_video_feeds = {}
_latest_video_feeds_lock = threading.Lock()
_last_pose_log     = {}
_probe_hits        = 0

# ─── Base64 encode thread pool (1 thread/camera, tách khỏi GStreamer probe) ──
# probe chỉ push RGBA array vào queue, thread này encode JPEG + base64 async.
_encode_queues: dict = {}   # src_id → queue.Queue(maxsize=2)
_encode_queues_lock = threading.Lock()


def _encode_worker(src_id: int, q: "_queue_mod.Queue"):
    """Encode frame RGBA→BGR→JPEG→base64 trong thread riêng, không block probe."""
    while True:
        item = q.get()
        if item is None:
            break
        if cv2 is None:
            continue
        try:
            frame_bgr = cv2.cvtColor(item, cv2.COLOR_RGBA2BGR)
            frame_rs  = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)
            _, buf    = cv2.imencode('.jpg', frame_rs, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            b64       = base64.b64encode(buf).decode('utf-8')
            with _latest_video_feeds_lock:
                _latest_video_feeds[src_id] = f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            print(f"[ENCODE-THREAD] src={src_id}: {e}")


def _get_encode_queue(src_id: int) -> "_queue_mod.Queue":
    """Lấy (hoặc tạo mới) encode queue + thread cho src_id."""
    with _encode_queues_lock:
        if src_id not in _encode_queues:
            q = _queue_mod.Queue(maxsize=2)
            _encode_queues[src_id] = q
            t = threading.Thread(
                target=_encode_worker, args=(src_id, q),
                daemon=True, name=f"encode-{src_id}"
            )
            t.start()
        return _encode_queues[src_id]


class _CamState:
    """Per-camera mutable state — passed as u_data to sgie_src_pad_probe.
    Mỗi CameraWorker giữ instance riêng → các camera hoàn toàn độc lập nhau."""
    __slots__ = ("frame_counter", "pose_cache", "mp_estimators",
                 "last_pose_log", "probe_hits", "restarting", "cam_idx")

    def __init__(self, cam_idx: int):
        self.cam_idx        = cam_idx
        self.frame_counter: dict = {}
        self.pose_cache:    dict = {}
        self.mp_estimators: dict = {}
        self.last_pose_log: dict = {}
        self.probe_hits:     int = 0
        self.restarting:    bool = False


    def reset(self):
        self.frame_counter.clear()
        self.pose_cache.clear()
        self.mp_estimators.clear()
        self.last_pose_log.clear()
        self.probe_hits = 0
        self.restarting = False


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1: THUẬT TOÁN SPHERICAL MORPHING (Yuan et al. 2020)
# ─────────────────────────────────────────────────────────────────────────────

# Mô hình 3D khuôn mặt – fallback default cho algorithm functions
GENERIC_3D_MODEL = np.array([
    [ 0.0,   0.0,  45.0],
    [ 0.0, -65.0, -15.0],
    [-65.0,  35.0, -25.0],
    [ 65.0,  35.0, -25.0],
], dtype=np.float64)

# Mô hình 4-point: nhìn thẳng (|yaw| < 25°)
PAPER_3D_MODEL = np.array([
    [ 0.0,    0.0,  50.0],   # nose_tip   pt(1)
    [ 0.0, -115.0, -35.0],   # chin       pt(152)
    [-48.0,   35.0, -25.0],   # left_eye
    [ 48.0,   35.0, -25.0],   # right_eye
], dtype=np.float64)

# Mô hình 5-point: quay phải (yaw > 25°) — mắt trái rõ, mắt phải bị che
# Dùng: nose + chin + left_eye + left_cheek(pt234) + nose_bridge(pt168)
MODEL_TURN_RIGHT = np.array([
    [  0.0,    0.0,  50.0],   # nose_tip    pt(1)
    [  0.0, -115.0, -35.0],   # chin        pt(152)
    [-48.0,   35.0, -25.0],   # left_eye
    [-75.0,   -5.0, -45.0],   # left_cheek  pt(234)
    [  0.0,   30.0,   5.0],   # nose_bridge pt(168)
], dtype=np.float64)

# Mô hình 5-point: quay trái (yaw < -25°) — mắt phải rõ, mắt trái bị che
# Dùng: nose + chin + right_eye + right_cheek(pt454) + nose_bridge(pt168)
MODEL_TURN_LEFT = np.array([
    [ 0.0,    0.0,  50.0],   # nose_tip    pt(1)
    [ 0.0, -115.0, -35.0],   # chin        pt(152)
    [48.0,   35.0, -25.0],   # right_eye
    [75.0,   -5.0, -45.0],   # right_cheek pt(454)
    [ 0.0,   30.0,   5.0],   # nose_bridge pt(168)
], dtype=np.float64)


def solve_sphere(M):
    X, Y, Z = M[:, 0], M[:, 1], M[:, 2]
    P = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(3):
        P[i, 0] = X[i+1] - X[0]
        P[i, 1] = Y[i+1] - Y[0]
        P[i, 2] = Z[i+1] - Z[0]
        b[i] = 0.5 * (
            (X[i+1]**2 + Y[i+1]**2 + Z[i+1]**2)
            - (X[0]**2 + Y[0]**2 + Z[0]**2)
        )
    try:
        Sc = np.linalg.solve(P, b)
        x0, y0, z0 = Sc
        l = float(np.sqrt((X[0]-x0)**2 + (Y[0]-y0)**2 + (Z[0]-z0)**2))
        return x0, y0, z0, l
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, 1.0


def _normalize_by_centroid(points):
    center = points.mean(axis=0)
    centered = points - center
    scale = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / (scale + 1e-8), center


def compute_morph_loss(v, phi, theta, x0, y0, z0, l,
                       R1_2d, m_norm, M_norm, eta):
    mp_ = phi.copy()
    mt  = theta.copy()
    mp_[0] += v[0]; mp_[1] += v[1]
    mp_[2] += v[2]; mp_[3] += v[2]
    mt[2]  += v[3]; mt[3]  -= v[3]
    M_morph = np.empty((4, 3))
    M_morph[:, 0] = x0 + l * np.sin(mp_) * np.cos(mt)
    M_morph[:, 1] = y0 + l * np.sin(mp_) * np.sin(mt)
    M_morph[:, 2] = z0 + l * np.cos(mp_)
    err_proj = np.sum((m_norm - M_morph.dot(R1_2d.T))**2)
    err_def  = eta * np.sum((M_morph - M_norm)**2)
    return err_proj + err_def, M_morph


def estimate_pose_spherical_morphing(m_points, eta=1.77, initial_v=None,
                                     model_3d=GENERIC_3D_MODEL):
    m_norm, m0 = _normalize_by_centroid(m_points)
    M_norm, _  = _normalize_by_centroid(model_3d)
    R1_raw, _, _, _ = np.linalg.lstsq(M_norm, m_norm, rcond=None)
    R1_2d = R1_raw.T
    x0, y0, z0, l = solve_sphere(M_norm)
    n_pts = M_norm.shape[0]
    phi   = np.array([math.acos(max(min((M_norm[i,2]-z0)/(l+1e-8),1.0),-1.0)) for i in range(n_pts)])
    theta = np.array([math.atan2(M_norm[i,1]-y0, M_norm[i,0]-x0) for i in range(n_pts)])

    def loss_fn(v):
        loss, _ = compute_morph_loss(v, phi, theta, x0, y0, z0, l, R1_2d, m_norm, M_norm, eta)
        return loss

    v0 = np.zeros(4) if initial_v is None else np.asarray(initial_v, dtype=np.float64)
    result = minimize(loss_fn, v0, method='L-BFGS-B',
                      bounds=[(-0.35,0.35),(-0.35,0.35),(-0.35,0.35),(-0.50,0.50)],
                      options={'maxiter': 15, 'ftol': 1e-8, 'gtol': 1e-5})
    v = result.x
    _, M_morph = compute_morph_loss(v, phi, theta, x0, y0, z0, l, R1_2d, m_norm, M_norm, eta)
    R_opt_raw, _, _, _ = np.linalg.lstsq(M_morph, m_norm, rcond=None)
    R_opt_raw = R_opt_raw.T
    ro1 = R_opt_raw[0] / (np.linalg.norm(R_opt_raw[0]) + 1e-8)
    ro2 = R_opt_raw[1] - np.dot(R_opt_raw[1], ro1) * ro1
    ro2 = ro2 / (np.linalg.norm(ro2) + 1e-8)
    ro3 = np.cross(ro1, ro2)
    R_opt = np.vstack([ro1, ro2, ro3])
    pitch = math.atan2(R_opt[2,1], R_opt[2,2])
    yaw   = math.atan2(-R_opt[2,0], math.sqrt(R_opt[2,1]**2 + R_opt[2,2]**2))
    roll  = math.atan2(R_opt[1,0], R_opt[0,0])
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll), m0, v, R_opt


def estimate_pose_linear_fast(m_points, model_3d=GENERIC_3D_MODEL):
    m_norm, m0 = _normalize_by_centroid(m_points)
    M_norm, _  = _normalize_by_centroid(model_3d)
    R_raw, _, _, _ = np.linalg.lstsq(M_norm, m_norm, rcond=None)
    R_raw = R_raw.T
    ro1 = R_raw[0] / (np.linalg.norm(R_raw[0]) + 1e-8)
    ro2 = R_raw[1] - np.dot(R_raw[1], ro1) * ro1
    ro2 = ro2 / (np.linalg.norm(ro2) + 1e-8)
    ro3 = np.cross(ro1, ro2)
    R_opt = np.vstack([ro1, ro2, ro3])
    pitch = math.atan2(R_opt[2,1], R_opt[2,2])
    yaw   = math.atan2(-R_opt[2,0], math.sqrt(R_opt[2,1]**2 + R_opt[2,2]**2))
    roll  = math.atan2(R_opt[1,0], R_opt[0,0])
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll), m0, None, R_opt


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2: TIỆN ÍCH GÓC & ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def _wrap180(delta):
    return (delta + 180.0) % 360.0 - 180.0

def _unwrap(angle, ref):
    return ref + _wrap180(angle - ref)

def _clamp_step(angle, prev, max_step):
    d = _wrap180(angle - prev)
    return prev + max(-max_step, min(max_step, d))


def _refs_from_paper_points(points):
    """Tính x_ref / y_ref từ 4 điểm [nose, chin, left_eye, right_eye]."""
    nose, chin, left_eye, right_eye = points
    eye_mid = 0.5 * (left_eye + right_eye)
    return {
        "x_ref":     right_eye - left_eye,
        "y_ref":     eye_mid - chin,
        "mp_points": points,
    }


class FaceMeshTRT:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(FaceMeshTRT, cls).__new__(cls)
                cls._instance._init_trt()
            return cls._instance

    def _init_trt(self):
        if cuda is None or trt is None:
            raise RuntimeError("pycuda or tensorrt not available")
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        self.engine_path = os.environ.get(
            "TRT_ENGINE_PATH",
            "/home/ivision/headpose-cameraIP/models/mediapipe_pose/"
            "20_new_onnx_postprocess_N-batch/face_mesh_192x192.onnx_b1_gpu0_fp16.engine"
        )
        print(f"[FaceMeshTRT INIT] Loading engine from {self.engine_path}")
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        self.d_input = cuda.mem_alloc(1 * 3 * 192 * 192 * 4) # float32
        self.d_landmarks = cuda.mem_alloc(1404 * 4)
        self.d_score = cuda.mem_alloc(1 * 4)
        self.stream = cuda.Stream()
        
        self.lock = threading.Lock()
        self.ctx.pop()
        print("[FaceMeshTRT INIT] Completed successfully!")

    def infer(self, inp):
        with self.lock:
            self.ctx.push()
            try:
                # Copy input to GPU
                cuda.memcpy_htod_async(self.d_input, inp.ravel(), self.stream)
                
                # Setup tensor addresses (TRT 10.3)
                self.context.set_tensor_address("input", int(self.d_input))
                self.context.set_tensor_address("landmarks", int(self.d_landmarks))
                self.context.set_tensor_address("score", int(self.d_score))
                
                # Execute inference
                self.context.execute_async_v3(stream_handle=self.stream.handle)
                
                # Retrieve output async
                landmarks = np.empty((1404,), dtype=np.float32)
                score = np.empty((1,), dtype=np.float32)
                cuda.memcpy_dtoh_async(landmarks, self.d_landmarks, self.stream)
                cuda.memcpy_dtoh_async(score, self.d_score, self.stream)
                
                self.stream.synchronize()
                return landmarks, score
            finally:
                self.ctx.pop()


def get_ort_session():
    """Khởi tạo lazy ONNX Runtime session."""
    global _ort_session, _ort_status_printed
    if ort is None or cv2 is None:
        if not _ort_status_printed:
            reasons = []
            if ort is None:  reasons.append("onnxruntime không cài được")
            if cv2 is None:  reasons.append("cv2 không cài được")
            print("[ONNX DISABLED] " + ", ".join(reasons))
            _ort_status_printed = True
        return None
    if _ort_session is None:
        print(f"[ONNX INIT] {ONNX_MODEL_PATH}")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ort_session = ort.InferenceSession(
            ONNX_MODEL_PATH,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        ins  = [x.name for x in _ort_session.get_inputs()]
        outs = [x.name for x in _ort_session.get_outputs()]
        print(f"[ONNX INIT] inputs={ins} outputs={outs}")
    return _ort_session


def extract_onnx_landmark_points(frame_np, face_left, face_top, face_w, face_h,
                                  prev_yaw=0.0):
    """
    Chạy face_mesh_192x192_post.onnx trên face crop.
    prev_yaw: góc yaw từ frame trước → chọn bộ landmark phù hợp.
    Trả về (points, model_3d, refs).
    """
    if frame_np is None or cv2 is None:
        return None, None, None

    if face_w < 28 or face_h < 28:
        return None, None, None

    fh, fw = frame_np.shape[:2]
    pad_x     = 0.28 * face_w
    pad_y_top = 0.45 * face_h
    pad_y_bot = 0.18 * face_h
    x1 = int(max(0, face_left - pad_x))
    y1 = int(max(0, face_top  - pad_y_top))
    x2 = int(min(fw, face_left + face_w + pad_x))
    y2 = int(min(fh, face_top  + face_h + pad_y_bot))
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        return None, None, None

    crop_rgb = cv2.cvtColor(frame_np[y1:y2, x1:x2], cv2.COLOR_RGBA2RGB)
    crop_h, crop_w = crop_rgb.shape[:2]
    crop_192 = cv2.resize(crop_rgb, (192, 192), interpolation=cv2.INTER_LINEAR)
    inp = (crop_192.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]

    gpu_success = False
    if USE_GPU_LANDMARK and trt is not None and cuda is not None:
        try:
            face_mesh_gpu = FaceMeshTRT()
            lm_raw, score_out = face_mesh_gpu.infer(inp)
            score_val = float(score_out[0])
            if score_val < LANDMARK_SCORE_THRESH:
                return None, None, None
            
            lm_reshaped = lm_raw.reshape(468, 3)
            lm = np.empty((468, 3), dtype=np.float32)
            lm[:, 0] = x1 + (lm_reshaped[:, 0] / 192.0) * crop_w
            lm[:, 1] = y1 + (lm_reshaped[:, 1] / 192.0) * crop_h
            lm[:, 2] = lm_reshaped[:, 2]
            gpu_success = True
        except Exception as e:
            print(f"[FaceMeshTRT ERROR] Fallback to CPU ORT: {e}")
            gpu_success = False

    if not gpu_success:
        sess = get_ort_session()
        if sess is None:
            return None, None, None

        feeds = {
            "input":       inp,
            "crop_x1":     np.array([[x1]], dtype=np.int32),
            "crop_y1":     np.array([[y1]], dtype=np.int32),
            "crop_width":  np.array([[crop_w]], dtype=np.int32),
            "crop_height": np.array([[crop_h]], dtype=np.int32),
        }
        score_out, lm_out = sess.run(["score", "final_landmarks"], feeds)

        if float(score_out[0, 0]) < LANDMARK_SCORE_THRESH:
            return None, None, None

        lm = lm_out[0]  # [468, 3]

    def pt(idx):
        return np.array([lm[idx, 0], lm[idx, 1]], dtype=np.float64)

    left_eye   = (pt(33)  + pt(133)) * 0.5
    right_eye  = (pt(263) + pt(362)) * 0.5
    nose_tip   = pt(1)
    chin       = pt(152)
    nose_bridge = pt(168)
    left_cheek  = pt(234)
    right_cheek = pt(454)

    # Chọn bộ landmark theo hướng quay đầu hiện tại
    if prev_yaw > 25.0:
        # Quay phải: mắt phải bị che → dùng mắt trái + má trái + sống mũi
        points   = np.vstack([nose_tip, chin, left_eye, left_cheek, nose_bridge])
        model_3d = MODEL_TURN_RIGHT
    elif prev_yaw < -25.0:
        # Quay trái: mắt trái bị che → dùng mắt phải + má phải + sống mũi
        points   = np.vstack([nose_tip, chin, right_eye, right_cheek, nose_bridge])
        model_3d = MODEL_TURN_LEFT
    else:
        # Nhìn thẳng: dùng 4-point chuẩn
        points   = np.vstack([nose_tip, chin, left_eye, right_eye])
        model_3d = PAPER_3D_MODEL

    return points, model_3d, _refs_from_paper_points(
        np.vstack([nose_tip, chin, left_eye, right_eye])
    )


class YoloHeadPoseEstimator:
    """Realtime head-pose estimator tuân thủ bài báo khoa học."""
    def __init__(self, eta=1.77, smooth_alpha=0.18, point_alpha=0.35,
                 max_angle_step=8.0, model_3d=GENERIC_3D_MODEL):
        self.eta            = eta
        self.smooth_alpha   = smooth_alpha
        self.point_alpha    = point_alpha
        self.max_angle_step = max_angle_step
        self.model_3d       = model_3d
        self._v             = None
        self._smooth_pts    = None
        self._smooth_ang    = None
        self._active_model  = id(model_3d)  # track model identity để reset khi đổi

    def update_points(self, m_pts, model_3d=None):
        active = model_3d if model_3d is not None else self.model_3d
        model_id = id(active)

        # Reset point smoother khi đổi bộ landmark (số điểm thay đổi)
        if self._smooth_pts is None or model_id != self._active_model:
            self._smooth_pts   = m_pts.copy()
            self._active_model = model_id

        # Ước lượng tốc độ quay bằng cách so sánh landmark hiện tại với
        # landmark frame trước đã smooth — không cần chạy lstsq lần 2.
        if self._smooth_ang is not None and self._smooth_pts is not None:
            pts_delta = np.linalg.norm(m_pts - self._smooth_pts, axis=1).max()
            # Quy đổi thô: mỗi pixel landmark dịch ~0.5° góc ở muxer 1920x1080
            delta_max = float(pts_delta) * 0.5
        else:
            delta_max = 0.0

        # Khi quay nhanh: bỏ smooth landmark, dùng raw để solver không bị trễ
        if delta_max > 12.0:
            pts_for_solver = m_pts
            self._smooth_pts = m_pts.copy()
        else:
            a = self.point_alpha
            self._smooth_pts = a * m_pts + (1.0 - a) * self._smooth_pts
            pts_for_solver = self._smooth_pts

        m_pts_paper = pts_for_solver.copy()
        m_pts_paper[:, 1] = -m_pts_paper[:, 1]

        if USE_SPHERICAL_MORPHING:
            pitch, yaw, roll, _, self._v, R_opt = estimate_pose_spherical_morphing(
                m_pts_paper, eta=self.eta, initial_v=self._v, model_3d=active)
        else:
            pitch, yaw, roll, _, self._v, R_opt = estimate_pose_linear_fast(
                m_pts_paper, model_3d=active)

        raw = np.array([pitch, yaw, roll])
        if self._smooth_ang is None:
            self._smooth_ang = raw.copy()
        else:
            if delta_max > 25.0:
                # Quay rất nhanh: snap tức thì, không clamp
                alpha = 1.0
                step  = 999.0
            elif delta_max > 12.0:
                # Quay nhanh: bắt kịp ngay
                alpha = 0.90
                step  = 45.0
            elif delta_max > 5.0:
                # Quay trung bình
                alpha = 0.55
                step  = 20.0
            else:
                # Ổn định: smoothing tối đa để giảm rung
                alpha = self.smooth_alpha   # 0.18
                step  = self.max_angle_step  # 8.0

            raw = np.array([
                _clamp_step(_unwrap(raw[0], self._smooth_ang[0]), self._smooth_ang[0], step),
                _clamp_step(_unwrap(raw[1], self._smooth_ang[1]), self._smooth_ang[1], step),
                _clamp_step(_unwrap(raw[2], self._smooth_ang[2]), self._smooth_ang[2], step * 0.5),
            ])
            self._smooth_ang = alpha * raw + (1.0 - alpha) * self._smooth_ang

        pitch, yaw, roll = self._smooth_ang
        pr = math.radians(pitch); yr = math.radians(yaw); rr = math.radians(roll)
        Rx = np.array([[1,0,0],[0,math.cos(pr),-math.sin(pr)],[0,math.sin(pr),math.cos(pr)]])
        Ry = np.array([[math.cos(yr),0,math.sin(yr)],[0,1,0],[-math.sin(yr),0,math.cos(yr)]])
        Rz = np.array([[math.cos(rr),-math.sin(rr),0],[math.sin(rr),math.cos(rr),0],[0,0,1]])
        return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll),
                "R_smooth": Rz @ Ry @ Rx}


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3: VẼ OSD (pyds display meta)
# ─────────────────────────────────────────────────────────────────────────────

def _add_line(dm, x1, y1, x2, y2, r, g, b, a=1.0, w=LINE_WIDTH):
    idx = dm.num_lines
    if idx >= 16:
        return False
    lp = dm.line_params[idx]
    lp.x1 = max(0, int(float(x1))); lp.y1 = max(0, int(float(y1)))
    lp.x2 = max(0, int(float(x2))); lp.y2 = max(0, int(float(y2)))
    lp.line_width = w
    lp.line_color.red = r; lp.line_color.green = g
    lp.line_color.blue = b; lp.line_color.alpha = a
    dm.num_lines = idx + 1
    return True


def _unit2(vec):
    if vec is None:
        return None
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    if vec.size < 2:
        return None
    vec = vec[:2]
    if not np.all(np.isfinite(vec)):
        return None
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return None
    return vec / norm


def _blend_dir(primary, reference, ref_weight):
    primary   = _unit2(primary)
    reference = _unit2(reference)
    if primary is None:   return reference
    if reference is None: return primary
    if float(np.dot(primary, reference)) < 0.0:
        primary = -primary
    blended = (1.0 - ref_weight) * primary + ref_weight * reference
    blended_u = _unit2(blended)
    return blended_u if blended_u is not None else primary


def draw_pose_osd(batch_meta, frame_meta,
                  head_left, head_top, head_w, head_h,
                  pitch_deg, yaw_deg, ox, oy, R,
                  axis_size, gaze_size, x_ref=None, y_ref=None):
    dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    if dm is None:
        return
    dm.num_rects = 0
    dm.num_lines = 0

    dm.num_rects = 0

    x_axis = np.array([float(R[0,0]), -float(R[1,0])], dtype=np.float64)
    y_axis = np.array([float(R[0,1]), -float(R[1,1])], dtype=np.float64)

    x_axis = _blend_dir(x_axis, x_ref, 0.60)
    if y_ref is not None and x_axis is not None:
        y_orth  = np.array([-x_axis[1], x_axis[0]], dtype=np.float64)
        y_ref_u = _unit2(y_ref)
        if y_ref_u is not None and float(np.dot(y_orth, y_ref_u)) < 0.0:
            y_orth = -y_orth
        y_axis = _blend_dir(y_axis, y_orth, 0.70)
    else:
        y_axis = _unit2(y_axis)

    if x_axis is None: x_axis = np.array([1.0, 0.0], dtype=np.float64)
    if y_axis is None: y_axis = np.array([0.0,-1.0], dtype=np.float64)

    _add_line(dm, ox, oy, ox + axis_size*float(x_axis[0]), oy + axis_size*float(x_axis[1]), 1.0,0.0,0.0)
    _add_line(dm, ox, oy, ox + axis_size*float(y_axis[0]), oy + axis_size*float(y_axis[1]), 0.0,1.0,0.0)
    gdx = float(R[0,2]); gdy = -float(R[1,2])
    _add_line(dm, ox, oy, ox + gaze_size*gdx, oy + gaze_size*gdy, 0.0,0.0,1.0, 1.0, LINE_WIDTH+2)

    pyds.nvds_add_display_meta_to_frame(frame_meta, dm)


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 4: PAD PROBE
# ─────────────────────────────────────────────────────────────────────────────

_pipeline_restarting = False  # guard tránh probe chạy khi pipeline đang restart


def sgie_src_pad_probe(pad, info, u_data):
    st: _CamState = u_data

    if st.restarting:
        return Gst.PadProbeReturn.DROP

    try:
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        st.probe_hits += 1

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        pyds.nvds_acquire_meta_lock(batch_meta)

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            src_id = frame_meta.source_id
            st.frame_counter[src_id] = st.frame_counter.get(src_id, 0) + 1
            do_detect = (not DISABLE_MEDIAPIPE) and (st.frame_counter[src_id] % PROCESS_EVERY_N_FRAMES == 0)
            face_seen = 0

            # Periodic purge mỗi 300 frame (~10s) — dọn cache stale, gọi GC
            if st.frame_counter[src_id] % 300 == 0:
                import gc
                cur_frame = st.frame_counter[src_id]
                stale = [k for k, v in st.pose_cache.items()
                         if k[0] == src_id and cur_frame - v.get("last_seen", 0) > 60]
                for k in stale:
                    del st.pose_cache[k]
                    st.mp_estimators.pop(k, None)
                    st.last_pose_log.pop(k, None)
                for k in list(st.mp_estimators.keys()):
                    if k not in st.pose_cache:
                        del st.mp_estimators[k]
                for k in list(st.last_pose_log.keys()):
                    if k not in st.pose_cache:
                        del st.last_pose_log[k]
                gc.collect()
                if stale:
                    print(f"[PURGE] src={src_id} removed={len(stale)} cache={len(st.pose_cache)} est={len(st.mp_estimators)}")

            # Fetch RGBA frame một lần duy nhất mỗi frame_meta
            frame_np = None
            if do_detect:
                try:
                    raw_surf = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    frame_np = np.array(raw_surf)
                except Exception as exc:
                    print(f"[WARN] get_surface src={src_id}: {exc}")

            # Push frame vào encode thread async — chỉ ở display mode (không có RTSP output)
            # Probe chỉ copy RGBA array (nhanh); encode JPEG/base64 nặng chạy ở thread riêng.
            if not RTSP_PUBLISH_URL and st.frame_counter[src_id] % 3 == 0:
                stream_frame_np = frame_np
                if stream_frame_np is None:
                    try:
                        raw_surf = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                        stream_frame_np = np.array(raw_surf)
                    except Exception as e:
                        print(f"[DEBUG VIDEO] get_nvds_buf_surface failed for src={src_id}: {e}")

                if stream_frame_np is not None:
                    try:
                        eq = _get_encode_queue(src_id)
                        eq.put_nowait(stream_frame_np)   # non-blocking; bỏ qua nếu thread đang bận
                    except _queue_mod.Full:
                        pass   # encode thread chưa kịp → bỏ frame này, lấy frame sau

            # Đếm tổng số face PeopleNet detect để debug multi-person
            total_faces = 0
            tmp_obj = frame_meta.obj_meta_list
            while tmp_obj is not None:
                try:
                    tmp_meta = pyds.NvDsObjectMeta.cast(tmp_obj.data)
                    if tmp_meta.class_id == 2 or tmp_meta.obj_label == "face":
                        total_faces += 1
                    tmp_obj = tmp_obj.next
                except StopIteration:
                    break
            if do_detect and st.frame_counter[src_id] % 90 == 0:
                print(f"[DETECT] src={src_id} faces_detected={total_faces}")

            detections = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                if obj_meta.class_id == 2 or obj_meta.obj_label == "face":
                    face_seen += 1
                    fr        = obj_meta.rect_params
                    face_left = fr.left
                    face_top  = fr.top
                    face_w    = fr.width
                    face_h    = fr.height

                    px = face_w * FACE_BBOX_PADDING
                    py = face_h * FACE_BBOX_PADDING
                    hl = max(0.0, face_left - px)
                    ht = max(0.0, face_top  - py)
                    hw = (face_left + face_w + px) - hl
                    hh = (face_top  + face_h + py) - ht
                    ht = max(0.0, ht - face_h * HEAD_VERTICAL_SHIFT)

                    obj_id = obj_meta.object_id
                    if obj_id == _UINT64_MAX:
                        cx_bin = int(face_left + face_w * 0.5) // _ID_GRID_PX
                        cy_bin = int(face_top  + face_h * 0.5) // _ID_GRID_PX
                        obj_id = cx_bin * 1000 + cy_bin
                    cache_key = (src_id, obj_id)

                    if do_detect:
                        try:
                            prev_yaw = st.pose_cache[cache_key]["yaw"] \
                                if cache_key in st.pose_cache else 0.0
                            mp_pts, mp_model, mp_refs = extract_onnx_landmark_points(
                                frame_np, face_left, face_top, face_w, face_h,
                                prev_yaw=prev_yaw)
                            if mp_pts is not None:
                                if cache_key not in st.mp_estimators:
                                    st.mp_estimators[cache_key] = YoloHeadPoseEstimator(
                                        model_3d=PAPER_3D_MODEL,
                                        smooth_alpha=0.20,
                                        point_alpha=0.65,
                                    )
                                pose = st.mp_estimators[cache_key].update_points(
                                    mp_pts, model_3d=mp_model)
                                st.pose_cache[cache_key] = {
                                    "pitch":      pose["pitch"],
                                    "yaw":        pose["yaw"],
                                    "roll":       pose["roll"],
                                    "R_smooth":   pose["R_smooth"],
                                    "miss_count": 0,
                                    "last_seen":  st.frame_counter[src_id],
                                    "mp_points":  mp_refs["mp_points"],
                                    "x_ref":      mp_refs["x_ref"],
                                    "y_ref":      mp_refs["y_ref"],
                                }
                                log_key = (src_id, obj_id)
                                if st.frame_counter[src_id] - st.last_pose_log.get(log_key, -999) >= 150:
                                    st.last_pose_log[log_key] = st.frame_counter[src_id]
                                    print(
                                        f"[MP] src={src_id} obj={obj_id} "
                                        f"yaw={pose['yaw']:+.1f} "
                                        f"pitch={pose['pitch']:+.1f} "
                                        f"roll={pose['roll']:+.1f}"
                                    )
                            else:
                                if cache_key in st.pose_cache:
                                    st.pose_cache[cache_key]["miss_count"] = \
                                        st.pose_cache[cache_key].get("miss_count", 0) + 1
                                    if st.pose_cache[cache_key]["miss_count"] > 15:
                                        del st.pose_cache[cache_key]
                                        if cache_key in st.mp_estimators:
                                            del st.mp_estimators[cache_key]
                                if st.frame_counter[src_id] % 90 == 0:
                                    print(f"[MISS] src={src_id} obj={obj_id}: ONNX score thấp/không detect mặt")
                        except Exception as exc:
                            print(f"[WARN] ONNX landmark: {exc}")

                    det = {
                        "box": [
                            round(float(face_left)),
                            round(float(face_top)),
                            round(float(face_left + face_w)),
                            round(float(face_top + face_h))
                        ],
                        "score": round(float(obj_meta.confidence), 2)
                    }

                    if cache_key in st.pose_cache:
                        c = st.pose_cache[cache_key]
                        R = c["R_smooth"]
                        gdx = float(R[0, 2])
                        gdy = -float(R[1, 2])
                        gdz = float(R[2, 2])
                        det["gaze"] = [round(gdx, 3), round(gdy, 3), round(gdz, 3)]
                        det["yaw"] = round(float(c["yaw"]), 1)
                        det["pitch"] = round(float(c["pitch"]), 1)
                    
                    detections.append(det)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # Publish JSON metadata qua MQTT
            payload = {
                "f": st.frame_counter[src_id],
                "ts": int(time.time() * 1000),
                "w": 1920,
                "h": 1080,
                "d": detections
            }
            topic = f"gaze/cam{st.cam_idx}/metadata"
            try:
                mqtt_client.publish(topic, json.dumps(payload, separators=(',', ':')), qos=0)
            except Exception as e:
                print(f"[MQTT ERROR] Publish failed for cam {st.cam_idx}: {e}")

            if do_detect and face_seen == 0 and st.frame_counter[src_id] % 30 == 0:
                print(f"[NO FACE] src={src_id}: không thấy object class_id=2/label=face")

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        pyds.nvds_release_meta_lock(batch_meta)

    except Exception:
        import traceback
        print("[LỖI TRONG PROBE]:")
        traceback.print_exc()
        try:
            pyds.nvds_release_meta_lock(batch_meta)
        except Exception:
            pass

    return Gst.PadProbeReturn.OK


def log_shipper_run():
    """Background thread to ship logs and latest video frames to VPS."""
    log_server_url = os.environ.get("LOG_SERVER_URL")
    if not log_server_url:
        return
        
    import requests
    import time
    
    device_id = os.environ.get("DEVICE_ID", "jetson_nano_edge")
    push_interval = float(os.environ.get("LOG_PUSH_INTERVAL", "0.3"))
    
    while True:
        try:
            # 1. Pop logs
            to_send_logs = []
            with _unshipped_logs_lock:
                if _unshipped_logs:
                    to_send_logs = list(_unshipped_logs)
                    _unshipped_logs.clear()
            
            # 2. Pop video feeds
            to_send_feeds = {}
            with _latest_video_feeds_lock:
                if _latest_video_feeds:
                    to_send_feeds = dict(_latest_video_feeds)
                    _latest_video_feeds.clear()
            
            # 3. Send payload
            if to_send_logs or to_send_feeds:
                payload = {
                    "device_id": device_id,
                    "lines": to_send_logs,
                    "video_feeds": to_send_feeds
                }
                resp = requests.post(log_server_url, json=payload, timeout=3.0)
                print(f"[DEBUG SHIPPER] Sent to {log_server_url}: status={resp.status_code}, logs={len(to_send_logs)}, feeds={len(to_send_feeds)}")
        except Exception as e:
            print(f"[DEBUG SHIPPER] Error: {e}")
            
        time.sleep(push_interval)


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 5: PIPELINE BUILDER (RTSP mode — nvstreamdemux)
# ─────────────────────────────────────────────────────────────────────────────

def _nvenc_set_quality(enc):
    """Đặt các property nâng cao cho nvv4l2h264enc trên Jetson.
    Dùng try/except vì tên property có thể khác giữa các phiên bản driver."""
    try:
        enc.set_property("profile", 2)           # Main profile (thay Baseline) — hỗ trợ B-frame
    except Exception:
        pass
    try:
        enc.set_property("preset-level", 1)      # 1=UltraFast..4=Slow; giữ 1 vì Jetson Nano yếu
    except Exception:
        pass
    try:
        enc.set_property("control-rate", 1)      # 1=CBR — ổn định bitrate cho streaming
    except Exception:
        pass


def _build_rtsp_pipeline(source_count, rtsp_publish_url, rtsp_urls, peoplenet_config):
    """
    Xây dựng pipeline lập trình với nvstreamdemux để xuất 2 RTSP stream riêng biệt.
    Mỗi camera → rtsp_publish_url_{i}  (ví dụ: .../live/jetson_nano_0, _1)
    Không dùng Gst.parse_launch() vì nvstreamdemux cần request pad thủ công.
    """
    has_nvenc = Gst.ElementFactory.find("nvv4l2h264enc") is not None
    pipeline = Gst.Pipeline.new("gaze-pipeline")

    def _make(factory, name):
        el = Gst.ElementFactory.make(factory, name)
        if not el:
            raise RuntimeError(f"Không tạo được element '{factory}' (name={name})")
        return el

    # ── Shared chain: muxer → pgie → mpconv (probe RGBA) → nvstreamdemux ──
    muxer = _make("nvstreammux", "muxer")
    muxer.set_property("batch-size",           source_count)
    muxer.set_property("width",                1920)
    muxer.set_property("height",               1080)
    muxer.set_property("batched-push-timeout", 33000)
    muxer.set_property("live-source",          1)
    try:
        muxer.set_property("sync-inputs", 0)
    except Exception:
        pass

    pgie = _make("nvinfer", "pgie")
    pgie.set_property("batch-size",       source_count)
    pgie.set_property("config-file-path", peoplenet_config)

    mpconv    = _make("nvvideoconvert", "mpconv")
    caps_rgba = _make("capsfilter",     "caps_rgba")
    caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))

    demux = _make("nvstreamdemux", "demux")

    for el in [muxer, pgie, mpconv, caps_rgba, demux]:
        pipeline.add(el)

    if not muxer.link(pgie):          raise RuntimeError("link muxer→pgie thất bại")
    if not pgie.link(mpconv):         raise RuntimeError("link pgie→mpconv thất bại")
    if not mpconv.link(caps_rgba):    raise RuntimeError("link mpconv→caps_rgba thất bại")
    if not caps_rgba.link(demux):     raise RuntimeError("link caps_rgba→demux thất bại")

    # ── Per-source branches: demux.src_i → q_i → osd_i → conv_i → enc_i → sink_i ──
    # Tạo từng element riêng lẻ và link trực tiếp (không dùng parse_bin_from_description)
    # để tránh ghost-pad làm rtspclientsink không kết nối được MediaMTX.
    for i in range(source_count):
        cam_url = f"{rtsp_publish_url}_{i}"

        q = _make("queue", f"q_{i}")
        q.set_property("leaky", 1)           # drop oldest frame, không drop frame mới
        q.set_property("max-size-buffers", 6)
        q.set_property("max-size-time",    0)
        q.set_property("max-size-bytes",   0)

        osd  = _make("nvdsosd",        f"osd_{i}")
        conv = _make("nvvideoconvert", f"conv_{i}")

        vrate = None
        q_enc = _make("queue", f"q_enc_{i}")
        q_enc.set_property("leaky", 2)
        q_enc.set_property("max-size-buffers", 4)
        q_enc.set_property("max-size-time", 0)
        q_enc.set_property("max-size-bytes", 0)

        if has_nvenc:
            caps_enc = _make("capsfilter", f"caps_enc_{i}")
            caps_enc.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))
            enc = _make("nvv4l2h264enc", f"enc_{i}")
            enc.set_property("bitrate",        6000000)   # 6 Mbps cho 1920×1080
            enc.set_property("insert-sps-pps", 1)
            enc.set_property("iframeinterval", 15)        # keyframe mỗi 0.5s — WebRTC recover nhanh hơn
            _nvenc_set_quality(enc)
        else:
            vrate = _make("videorate", f"vrate_{i}")
            vrate.set_property("drop-only", True)
            caps_enc = _make("capsfilter", f"caps_enc_{i}")
            caps_enc.set_property("caps", Gst.Caps.from_string("video/x-raw,format=I420,width=1280,height=720,framerate=25/1"))
            enc = _make("x264enc", f"enc_{i}")
            enc.set_property("bitrate", 2500)             # 2.5 Mbps cho 720p
            enc.set_property("key-int-max", 25)           # keyframe mỗi 1s ở 25fps
            enc.set_property("threads", 3)
            enc.set_property("sliced-threads", True)
            enc.set_property("bframes", 0)
            enc.set_property("b-adapt", False)
            enc.set_property("aud", True)
            enc.set_property("byte-stream", True)
            Gst.util_set_object_arg(enc, "speed-preset", "ultrafast")
            Gst.util_set_object_arg(enc, "tune",         "zerolatency")

        h264p = _make("h264parse",      f"h264p_{i}")
        h264p.set_property("config-interval", -1)
        sink  = _make("rtspclientsink", f"rtsp_sink_{i}")
        sink.set_property("protocols", "tcp")
        sink.set_property("location",  cam_url)

        elements = [q, osd, conv]
        if vrate:
            elements.append(vrate)
        elements.extend([caps_enc, q_enc, enc, h264p, sink])

        for el in elements:
            pipeline.add(el)

        # Link nội bộ branch
        if not q.link(osd):         raise RuntimeError(f"q_{i}→osd_{i}")
        if not osd.link(conv):      raise RuntimeError(f"osd_{i}→conv_{i}")
        if vrate:
            if not conv.link(vrate):      raise RuntimeError(f"conv_{i}→vrate_{i}")
            if not vrate.link(caps_enc):  raise RuntimeError(f"vrate_{i}→caps_enc_{i}")
        else:
            if not conv.link(caps_enc):   raise RuntimeError(f"conv_{i}→caps_enc_{i}")
        if not caps_enc.link(q_enc): raise RuntimeError(f"caps_enc_{i}→q_enc_{i}")
        if not q_enc.link(enc):      raise RuntimeError(f"q_enc_{i}→enc_{i}")
        if not enc.link(h264p):     raise RuntimeError(f"enc_{i}→h264p_{i}")
        if not h264p.link(sink):    raise RuntimeError(f"h264p_{i}→rtsp_sink_{i}")

        # Link trực tiếp demux.src_i → q_i.sink (không qua ghost pad)
        demux_src = demux.get_request_pad(f"src_{i}")
        if not demux_src:
            raise RuntimeError(f"Không lấy được demux request pad src_{i}")
        q_sink = q.get_static_pad("sink")
        ret = demux_src.link(q_sink)
        if ret != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Link demux.src_{i} → q_{i}.sink thất bại: {ret}")

        print(f"  Branch {i}: → {cam_url}")

    # ── Source bins (nvurisrcbin có dynamic pad → dùng pad-added signal) ──
    for idx, rtsp_url in enumerate(rtsp_urls):
        src = _make("nvurisrcbin", f"src_{idx}")
        src.set_property("uri",                    rtsp_url)
        src.set_property("type",                   2)
        src.set_property("select-rtp-protocol",    4)
        src.set_property("latency",                200)
        try:
            src.set_property("drop-on-latency",        True)
        except Exception:
            pass
        src.set_property("rtsp-reconnect-interval", 10)
        src.set_property("rtsp-reconnect-attempts", 0)
        pipeline.add(src)

        mux_sink = muxer.get_request_pad(f"sink_{idx}")
        if not mux_sink:
            raise RuntimeError(f"Không lấy được muxer sink_{idx}")

        def _on_pad_added(element, pad, _sink=mux_sink):
            if pad.get_direction() != Gst.PadDirection.SRC:
                return
            if _sink.is_linked():
                return
            caps = pad.get_current_caps() or pad.query_caps(None)
            if caps and caps.get_size() > 0:
                if "video" in caps.get_structure(0).get_name():
                    ret = pad.link(_sink)
                    if ret != Gst.PadLinkReturn.OK:
                        print(f"[WARN] pad-added link thất bại: {ret}")

        src.connect("pad-added", _on_pad_added)

    enc_name = "nvv4l2h264enc" if has_nvenc else "x264enc"
    print(f"[RTSP] {enc_name} → {rtsp_publish_url}_[0..{source_count - 1}]  (nvstreamdemux, riêng từng cam)")
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 5B: SINGLE-CAMERA PIPELINE (dùng bởi CameraWorker)
# ─────────────────────────────────────────────────────────────────────────────

def _build_single_cam_pipeline(cam_idx, rtsp_url, publish_url, peoplenet_config):
    """Pipeline độc lập cho 1 camera: không dùng nvstreamdemux (batch=1).
    Chỉ chạy AI (PeopleNet + FaceMesh TRT), không vẽ hình, không encode video."""
    pipeline = Gst.Pipeline.new(f"gaze-pipeline-{cam_idx}")

    def _make(factory, name):
        el = Gst.ElementFactory.make(factory, name)
        if not el:
            raise RuntimeError(f"Không tạo được '{factory}' (name={name})")
        return el

    src = _make("nvurisrcbin", f"src_{cam_idx}")
    src.set_property("uri",                     rtsp_url)
    src.set_property("type",                    2)
    src.set_property("select-rtp-protocol",     4)
    src.set_property("latency",                 200)
    try:
        src.set_property("drop-on-latency",         True)
    except Exception:
        pass
    src.set_property("rtsp-reconnect-interval", 10)
    src.set_property("rtsp-reconnect-attempts", 0)

    muxer = _make("nvstreammux", "muxer")
    muxer.set_property("batch-size",           1)
    muxer.set_property("width",                1920) # Giữ 1920x1080 cho độ chính xác AI cao nhất
    muxer.set_property("height",               1080)
    muxer.set_property("batched-push-timeout", 33000)
    muxer.set_property("live-source",          1)
    try:
        muxer.set_property("sync-inputs", 0)
    except Exception:
        pass

    pgie = _make("nvinfer", "pgie")
    pgie.set_property("batch-size",       1)
    pgie.set_property("config-file-path", peoplenet_config)

    mpconv    = _make("nvvideoconvert", "mpconv")
    caps_rgba = _make("capsfilter",     "caps_rgba")
    caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))

    sink = _make("fakesink", "sink")
    sink.set_property("sync",  False)
    sink.set_property("async", False)

    elements = [src, muxer, pgie, mpconv, caps_rgba, sink]

    for el in elements:
        pipeline.add(el)

    if not muxer.link(pgie):         raise RuntimeError("muxer→pgie")
    if not pgie.link(mpconv):        raise RuntimeError("pgie→mpconv")
    if not mpconv.link(caps_rgba):   raise RuntimeError("mpconv→caps_rgba")
    if not caps_rgba.link(sink):     raise RuntimeError("caps_rgba→sink")

    mux_sink = muxer.get_request_pad("sink_0")
    if not mux_sink:
        raise RuntimeError(f"Không lấy được muxer sink_0 (cam {cam_idx})")

    def _on_pad_added(element, pad, _sink=mux_sink):
        if pad.get_direction() != Gst.PadDirection.SRC:
            return
        if _sink.is_linked():
            return
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps and caps.get_size() > 0:
            if "video" in caps.get_structure(0).get_name():
                ret = pad.link(_sink)
                if ret != Gst.PadLinkReturn.OK:
                    print(f"[WARN] cam-{cam_idx} pad-added link thất bại: {ret}")

    src.connect("pad-added", _on_pad_added)

    print(f"[CAM-{cam_idx}] Pipeline AI (metadata-only) khởi tạo xong.")
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 5C: CAMERA WORKER — mỗi camera chạy hoàn toàn độc lập trong thread riêng
# ─────────────────────────────────────────────────────────────────────────────

class CameraWorker(threading.Thread):
    """Chạy pipeline DeepStream cho 1 camera trong thread riêng.
    Nếu pipeline lỗi → tự restart, không ảnh hưởng camera khác."""

    _THERMAL_ZONES = {
        "CPU": "/sys/devices/virtual/thermal/thermal_zone0/temp",
        "GPU": "/sys/devices/virtual/thermal/thermal_zone1/temp",
    }

    def __init__(self, cam_idx: int, rtsp_url: str, publish_url: str, peoplenet_config: str):
        super().__init__(daemon=True, name=f"cam-{cam_idx}")
        self.cam_idx        = cam_idx
        self.rtsp_url       = rtsp_url
        self.publish_url    = publish_url
        self.peoplenet_config = peoplenet_config
        self.stop_event     = threading.Event()
        self.state          = _CamState(cam_idx)
        self.loop           = None

    # ── Public API ────────────────────────────────────────────────────────────

    def stop(self):
        self.stop_event.set()
        if self.loop:
            GLib.idle_add(self.loop.quit)

    # ── Thread entry point ────────────────────────────────────────────────────

    def run(self):
        retry_delay = 5
        while not self.stop_event.is_set():
            try:
                self._run_once()
                retry_delay = 5          # reset delay sau lần chạy thành công
            except Exception as exc:
                print(f"[CAM-{self.cam_idx}] Lỗi worker: {exc}")
            if not self.stop_event.is_set():
                print(f"[CAM-{self.cam_idx}] Restart sau {retry_delay}s...")
                self.stop_event.wait(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
        print(f"[CAM-{self.cam_idx}] Worker dừng.")

    # ── Single pipeline run ────────────────────────────────────────────────────

    def _run_once(self):
        st = self.state
        st.reset()

        pipeline = _build_single_cam_pipeline(
            self.cam_idx, self.rtsp_url, self.publish_url, self.peoplenet_config
        )

        mpconv     = pipeline.get_by_name("mpconv")
        mpconv_src = mpconv.get_static_pad("src")
        mpconv_src.add_probe(Gst.PadProbeType.BUFFER, sgie_src_pad_probe, st)

        loop = GLib.MainLoop()
        self.loop = loop
        bus  = pipeline.get_bus()
        bus.add_signal_watch()

        restart_pending = {"flag": False}

        def _do_restart():
            if restart_pending["flag"]:
                return False
            restart_pending["flag"] = True
            st.restarting = True
            print(f"[CAM-{self.cam_idx}] Đang restart pipeline...")
            try:
                pipeline.set_state(Gst.State.NULL)
                pipeline.get_state(Gst.CLOCK_TIME_NONE)
            except Exception:
                pass
            loop.quit()
            return False

        def _on_bus_msg(bus, msg, _loop):
            t = msg.type
            if t == Gst.MessageType.EOS:
                print(f"[CAM-{self.cam_idx}] EOS — restart sau 3s")
                if not restart_pending["flag"]:
                    GLib.timeout_add_seconds(3, _do_restart)
            elif t == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                src_name = msg.src.get_name() if msg.src else "unknown"
                # Lỗi RTSP (nguồn hoặc đích) → restart sau 15s cho MediaMTX dọn session cũ
                # Lỗi khác (codec, GPU) → restart nhanh hơn sau 5s
                delay = 15 if "rtsp" in src_name else 5
                print(f"[CAM-{self.cam_idx}] ERROR {src_name}: {err.message} — restart sau {delay}s")
                if not restart_pending["flag"]:
                    GLib.timeout_add_seconds(delay, _do_restart)
            elif t == Gst.MessageType.WARNING:
                warn, _ = msg.parse_warning()
                src_name = msg.src.get_name() if msg.src else "unknown"
                print(f"[CAM-{self.cam_idx}] WARN {src_name}: {warn.message}")
            elif t == Gst.MessageType.STATE_CHANGED and msg.src == pipeline:
                old, new, _ = msg.parse_state_changed()
                print(f"[CAM-{self.cam_idx}] {old.value_nick} → {new.value_nick}")

        bus.connect("message", _on_bus_msg, loop)

        last_wd = {"hits": -1, "same": 0}

        def _watchdog():
            if self.stop_event.is_set():
                return False        # dừng GLib timer khi worker bị stop()
            if st.probe_hits == last_wd["hits"]:
                last_wd["same"] += 1
            else:
                last_wd["hits"] = st.probe_hits
                last_wd["same"] = 0
            stall = last_wd["same"] >= 3
            temps = " ".join(
                f"{n}={self._read_temp(p):.0f}°C"
                for n, p in self._THERMAL_ZONES.items()
            )
            print(f"[CAM-{self.cam_idx}]{'[STALL]' if stall else ''} "
                  f"probe={st.probe_hits} frames={dict(st.frame_counter)} | {temps}")
            if stall and not restart_pending["flag"]:
                print(f"[CAM-{self.cam_idx}] STALL {last_wd['same']*5}s — restart...")
                last_wd["same"] = 0
                GLib.timeout_add_seconds(2, _do_restart)
            return True

        GLib.timeout_add_seconds(5, _watchdog)

        pipeline.set_state(Gst.State.PLAYING)
        print(f"[CAM-{self.cam_idx}] PLAYING → {self.publish_url}")

        loop.run()

        pipeline.set_state(Gst.State.NULL)
        print(f"[CAM-{self.cam_idx}] Pipeline đã dừng.")

    @staticmethod
    def _read_temp(path):
        try:
            return int(open(path).read().strip()) / 1000.0
        except Exception:
            return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 6: MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    Gst.init(None)

    print(f"[MAIN] Khởi động {len(RTSP_URLS)} camera (chỉ lấy metadata AI, gửi về MQTT)")
    print(f"[MAIN] MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")

    workers = [
        CameraWorker(i, url, "", PEOPLENET_CONFIG)
        for i, url in enumerate(RTSP_URLS)
    ]
    for w in workers:
        w.start()

    import signal as _sig
    def _stop_all(signum=None, frame=None):
        print("\nDừng tất cả cameras...")
        for w in workers:
            w.stop()

    _sig.signal(_sig.SIGINT,  _stop_all)
    _sig.signal(_sig.SIGTERM, _stop_all)

    while any(w.is_alive() for w in workers):
        for w in workers:
            w.join(timeout=1.0)

    # Dừng MQTT client loop
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("Tất cả cameras đã dừng.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
