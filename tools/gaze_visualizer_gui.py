#!/usr/bin/env python3
"""
gaze_viewer.py — Render RTSP video stream overlayed with MQTT headpose metadata.
Supports local GUI window or MJPEG web streaming server.
Calculates headpose and gaze direction vector locally from raw landmarks if needed,
and supports side-by-side grid view for both cameras.
"""
import argparse
import json
import math
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver

import cv2
import numpy as np
import requests
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

import multiprocessing
import queue

try:
    import tensorrt
    import pycuda.driver
    HAS_SCRFD_DEPS = True
except ImportError:
    HAS_SCRFD_DEPS = False

# Implementation note.
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
CAMERA_RTSP_URLS = {
    "cam0": "rtsp://user:password@192.168.1.10:1904/stream",
    "cam1": "rtsp://user:password@192.168.1.11:1904/stream",
}


def fetch_camera_urls(app_api_url: str) -> dict:
    """Fetch camera URLs from the application service."""
    try:
        r = requests.get(f"{app_api_url}/cameras", timeout=5)
        r.raise_for_status()
        cameras = r.json().get("cameras", [])
        urls = {}
        for c in cameras:
            url = c.get("mediamtx_rtsp_url") or c.get("url")
            if url:
                urls[f"cam{c['src_id']}"] = url
        return urls
    except Exception as e:
        print(f"[Viewer] Failed to fetch camera list from {app_api_url}: {e}")
        return {}

# Global thread-safe states

# Frame buffer: {cam_id: deque of (wall_clock_ms, frame)}
# Implementation note.
from collections import deque
FRAME_BUFFER_SIZE = 60  # Implementation note.
frame_buffers: dict = {"cam0": deque(maxlen=FRAME_BUFFER_SIZE),
                       "cam1": deque(maxlen=FRAME_BUFFER_SIZE)}
frame_locks = {"cam0": threading.Lock(), "cam1": threading.Lock()}

latest_metadatas = {"cam0": None, "cam1": None}
metadata_lock = threading.Lock()

current_preview_frame = None
preview_lock = threading.Lock()

stop_event = threading.Event()

# Implementation note.
# Implementation note.
#
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
PIPELINE_DELAY_MS: float = 60.0


try:
    from scipy.optimize import minimize as _scipy_minimize
except Exception:
    _scipy_minimize = None

# ── 3D Face Models ─────────────────────────────────────────────────────────────
FACE_3D_MODEL = np.array([
    [  0.0,    0.0,   50.0],   # nose tip
    [  0.0, -115.0,  -35.0],   # chin
    [-48.0,   35.0,  -25.0],   # Implementation note.
    [ 48.0,   35.0,  -25.0],   # right eye
    [  0.0,   20.0,    5.0],   # Implementation note.
    [-70.0,   30.0,  -40.0],   # left cheek
    [ 70.0,   30.0,  -40.0],   # right cheek
], dtype=np.float64)

PAPER_3D_MODEL = np.array([
    [  0.0,    0.0,   50.0],   # nose tip
    [  0.0, -115.0,  -35.0],   # chin
    [-48.0,   35.0,  -25.0],   # left eye
    [ 48.0,   35.0,  -25.0],   # right eye
    [  0.0,   20.0,    5.0],   # nose bridge
], dtype=np.float64)


# ── SCRFD 3D Model ─────────────────────────────────────────────────────────────
SCRFD_3D_MODEL = np.array([
    [-48.0,  35.0, -25.0],   # left eye
    [ 48.0,  35.0, -25.0],   # right eye
    [  0.0,   0.0,  50.0],   # nose tip
    [-35.0, -45.0, -15.0],   # left mouth corner
    [ 35.0, -45.0, -15.0],   # right mouth corner
], dtype=np.float64)

def init_scrfd_detector(engine_path):
    # This is kept as a checker stub
    if not HAS_SCRFD_DEPS:
        raise RuntimeError("Không thể chạy SCRFD vì thiếu thư viện PyCUDA hoặc TensorRT.")
    return None

class SCRFDProcess(multiprocessing.get_context("spawn").Process):
    def __init__(self, engine_path, input_queue, output_queue):
        super().__init__(daemon=True)
        self.engine_path = engine_path
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        # Import CUDA dependencies ONLY inside the child process to keep main process CUDA-free
        global trt, cuda
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        import traceback
        
        print(f"[SCRFD Process] Loading TensorRT engine inside child process: {self.engine_path}...")
        try:
            detector = SCRFDTRT(self.engine_path)
        except Exception as e:
            print(f"[SCRFD Process] FATAL: Cannot load engine: {e}")
            traceback.print_exc()
            return
        print("[SCRFD Process] SCRFD Engine loaded successfully!")
        
        frame_count = 0
        while True:
            try:
                task = self.input_queue.get(timeout=10.0)
                if task is None:
                    break
                cam, ts, frame = task
                frame_count += 1
                
                # Run inference
                dets, kps_all = detector.detect(frame, thr=0.35, nms=0.4)
                
                # Implementation note.
                if frame_count % 10 == 1:
                    print(f"[SCRFD Process] Frame {frame_count}: cam={cam}, detected {len(dets)} faces")
                
                # Send result back to parent process
                self.output_queue.put((cam, ts, frame, dets, kps_all))
            except Exception as e:
                if "Empty" not in type(e).__name__:
                    print(f"[SCRFD Process Run Error] {e}")
                    traceback.print_exc()

def _make_scrfd_anchors(input_size=640):
    centers, strides = [], []
    for stride in [8, 16, 32]:
        n = input_size // stride
        gy, gx = np.mgrid[0:n, 0:n]
        c = np.stack([gx, gy], -1).reshape(-1, 2) * stride
        c = np.repeat(c, 2, axis=0)
        centers.append(c)
        strides.append(np.full((c.shape[0], 1), stride, np.float32))
    return np.vstack(centers).astype(np.float32), np.vstack(strides)

_SCRFD_ANCHORS, _SCRFD_STRIDES = _make_scrfd_anchors()

def _decode_scrfd_boxes(loc, c, s):
    return np.hstack([c[:, :1]-loc[:, :1]*s, c[:, 1:2]-loc[:, 1:2]*s,
                      c[:, :1]+loc[:, 2:3]*s, c[:, 1:2]+loc[:, 3:4]*s])

def _decode_scrfd_kps(pre, c, s):
    out = np.empty_like(pre)
    for i in range(5):
        out[:, i*2  ] = c[:, 0] + pre[:, i*2  ] * s[:, 0]
        out[:, i*2+1] = c[:, 1] + pre[:, i*2+1] * s[:, 0]
    return out

def _scrfd_nms(dets, thresh=0.4):
    x1,y1,x2,y2,sc = dets[:,0],dets[:,1],dets[:,2],dets[:,3],dets[:,4]
    areas = (x2-x1+1)*(y2-y1+1)
    order = sc.argsort()[::-1]; keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        w=np.maximum(0.,xx2-xx1+1); h=np.maximum(0.,yy2-yy1+1)
        ovr=(w*h)/(areas[i]+areas[order[1:]]-w*h)
        order=order[np.where(ovr<=thresh)[0]+1]
    return keep

def _get_trt_tensor_index(engine, name):
    for i in range(engine.num_io_tensors):
        if engine.get_tensor_name(i) == name:
            return i
    raise ValueError(f"Tensor không có trong engine: {name!r}")

class SCRFDTRT:
    W = H = 640

    def __init__(self, path):
        lg = trt.Logger(trt.Logger.WARNING)
        with open(path,"rb") as f, trt.Runtime(lg) as rt:
            self.eng = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.eng.create_execution_context()

        def alloc(n):
            h = np.empty(n, dtype=np.float32)
            return h, cuda.mem_alloc(h.nbytes)

        self.h_in,  self.d_in  = alloc(3*self.H*self.W)
        self.h_s8,  self.d_s8  = alloc(12800)
        self.h_b8,  self.d_b8  = alloc(12800*4)
        self.h_l8,  self.d_l8  = alloc(12800*10)
        self.h_s16, self.d_s16 = alloc(3200)
        self.h_b16, self.d_b16 = alloc(3200*4)
        self.h_l16, self.d_l16 = alloc(3200*10)
        self.h_s32, self.d_s32 = alloc(800)
        self.h_b32, self.d_b32 = alloc(800*4)
        self.h_l32, self.d_l32 = alloc(800*10)

        binds = {"input.1":self.d_in,
                 "448":self.d_s8,  "451":self.d_b8,  "454":self.d_l8,
                 "471":self.d_s16, "474":self.d_b16,  "477":self.d_l16,
                 "494":self.d_s32, "497":self.d_b32,  "500":self.d_l32}
        self.bindings = [None]*self.eng.num_io_tensors
        for name,ptr in binds.items():
            self.bindings[_get_trt_tensor_index(self.eng, name)] = int(ptr)

    def detect(self, frame, thr=0.40, nms=0.4):
        oh, ow = frame.shape[:2]
        img = cv2.cvtColor(cv2.resize(frame,(self.W,self.H)), cv2.COLOR_BGR2RGB).astype(np.float32)
        np.copyto(self.h_in, ((img-127.5)*0.007843137).transpose(2,0,1).ravel())
        cuda.memcpy_htod(self.d_in, self.h_in)
        self.ctx.execute_v2(self.bindings)
        pairs = [(self.h_s8,self.d_s8),(self.h_b8,self.d_b8),(self.h_l8,self.d_l8),
                 (self.h_s16,self.d_s16),(self.h_b16,self.d_b16),(self.h_l16,self.d_l16),
                 (self.h_s32,self.d_s32),(self.h_b32,self.d_b32),(self.h_l32,self.d_l32)]
        for h,d in pairs: cuda.memcpy_dtoh(h,d)

        sc  = 1./(1.+np.exp(-np.concatenate([self.h_s8,self.h_s16,self.h_s32])))
        bx  = np.concatenate([self.h_b8.reshape(-1,4),self.h_b16.reshape(-1,4),self.h_b32.reshape(-1,4)])
        lm  = np.concatenate([self.h_l8.reshape(-1,10),self.h_l16.reshape(-1,10),self.h_l32.reshape(-1,10)])
        ii  = np.where(sc > thr)[0]
        if not len(ii): return np.empty((0,5)), np.empty((0,10))

        c=_SCRFD_ANCHORS[ii]; s=_SCRFD_STRIDES[ii]
        boxes = _decode_scrfd_boxes(bx[ii], c, s)
        boxes[:,[0,2]] *= ow/self.W; boxes[:,[1,3]] *= oh/self.H
        kps   = _decode_scrfd_kps(lm[ii], c, s)
        kps[:,0::2] *= ow/self.W; kps[:,1::2] *= oh/self.H

        dets = np.hstack([boxes, sc[ii,None]]).astype(np.float32)
        keep = _scrfd_nms(dets, nms)
        return dets[keep], kps[keep]

def estimate_pose_scrfd(kps5, v0=None):
    mn, _ = _normalize_by_centroid(np.hstack([kps5, np.zeros((5, 1))]))
    Mn, _ = _normalize_by_centroid(SCRFD_3D_MODEL)
    R1_raw, _, _, _ = np.linalg.lstsq(Mn, mn, rcond=None)
    R1_2d = R1_raw.T
    x0, y0, z0, l = solve_sphere(Mn)
    n = 5
    phi = np.array([math.acos(max(min((Mn[i, 2]-z0)/(l+1e-8), 1.), -1.)) for i in range(n)])
    theta = np.array([math.atan2(Mn[i, 1]-y0, Mn[i, 0]-x0) for i in range(n)])
    def morph(v):
        mp = phi.copy()
        mt = theta.copy()
        mp[0] += v[0]
        mp[1] += v[0]
        mp[2] += v[1]
        mp[3] += v[2]
        mp[4] += v[2]
        mt[0] += v[3]
        mt[1] -= v[3]
        Mm = np.empty((n, 3))
        Mm[:, 0] = x0 + l*np.sin(mp)*np.cos(mt)
        Mm[:, 1] = y0 + l*np.sin(mp)*np.sin(mt)
        Mm[:, 2] = z0 + l*np.cos(mp)
        return 1.77*np.sum((Mm-Mn)**2) + np.sum((mn-Mm.dot(R1_2d.T))**2), Mm
    res = _scipy_minimize(lambda v: morph(v)[0], np.zeros(4) if v0 is None else v0,
                          method='L-BFGS-B', bounds=[(-0.35,0.35)]*3+[(-0.5,0.5)],
                          options={'maxiter':8,'ftol':1e-8,'gtol':1e-5})
    _, Mm = morph(res.x)
    R_opt_raw, _, _, _ = np.linalg.lstsq(Mm, mn, rcond=None)
    pitch, yaw, roll, R = _rotation_from_matrix_gs(R_opt_raw.T)
    return pitch, yaw, roll, R, res.x

def draw_scrfd_overlay(frame, dets, kps_all, pose_cache):
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d[:4].astype(int)
        kps = kps_all[i].reshape(5, 2)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for kp in kps:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 200, 255), -1)
            
        label_y = max(25, y1 - 10)
        label_text = f"SCRFD ID:{i} LMs:5"
        
        pts = kps.copy()
        pts[:, 1] = -pts[:, 1]
        
        try:
            v_cache = pose_cache.get(i) if pose_cache is not None else None
            pitch, yaw, roll, R, v_opt = estimate_pose_scrfd(pts, v_cache)
            if pose_cache is not None:
                pose_cache[i] = v_opt
                
            label_text += f" Y:{yaw:+.1f} P:{pitch:+.1f}"
            cv2.putText(frame, label_text, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            
            gx, gy = float(R[0, 2]), -float(R[1, 2])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            gaze_scale = max(100, (x2 - x1) * 1.0)
            ex = int(cx + gx * gaze_scale)
            ey = int(cy + gy * gaze_scale)
            
            cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 0, 255), 4, tipLength=0.25, line_type=cv2.LINE_AA)
        except Exception as e:
            cv2.putText(frame, label_text, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)



# ── One-Euro Filter ──────────────────────────────────────────────────────────
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.t_prev = t0
        self.dx_prev = 0.0

    def __call__(self, t, x):
        dt = (t - self.t_prev) / 1000.0  # Implementation note.
        if dt <= 0.0:
            return self.x_prev
        
        # Implementation note.
        dx = (x - self.x_prev) / dt
        
        # Implementation note.
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        
        # Implementation note.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Implementation note.
        alpha_x = self._alpha(dt, cutoff)
        x_hat = alpha_x * x + (1.0 - alpha_x) * self.x_prev
        
        # Implementation note.
        self.x_prev = x_hat
        self.t_prev = t
        self.dx_prev = dx_hat
        
        return x_hat

    def _alpha(self, dt, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


# Implementation note.
def validate_landmarks(pts):
    """
    pts: Nx2 array of landmarks [nose, chin, left_eye, right_eye, bridge]
    Documentation for this component.
    """
    pts = np.asarray(pts, dtype=np.float64)[:, :2]
    
    # Implementation note.
    d_eyes = np.linalg.norm(pts[2] - pts[3])
    d_nose_chin = np.linalg.norm(pts[0] - pts[1])
    d_left_bridge = np.linalg.norm(pts[2] - pts[4])
    d_right_bridge = np.linalg.norm(pts[3] - pts[4])
    
    # Implementation note.
    ratio_chin = d_nose_chin / (d_eyes + 1e-8)
    chin_ok = 0.7 <= ratio_chin <= 3.5
    
    # Implementation note.
    left_eye_ok = d_left_bridge > 0.15 * d_eyes
    right_eye_ok = d_right_bridge > 0.15 * d_eyes
    
    active_mask = [0, 4]  # Implementation note.
    if chin_ok:
        active_mask.append(1)
    if left_eye_ok:
        active_mask.append(2)
    if right_eye_ok:
        active_mask.append(3)
        
    # Implementation note.
    if len(active_mask) < 3:
        if d_left_bridge > d_right_bridge:
            active_mask = [0, 2, 4]
        else:
            active_mask = [0, 3, 4]
            
    confidence = len(active_mask) / 5.0
    return active_mask, confidence


# ── Rotation to Euler ────────────────────────────────────────────────────────
def _rotation_to_euler(R):
    pitch = math.atan2(R[2, 1], R[2, 2])
    yaw   = math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll  = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def _euler_to_rotation_matrix(pitch, yaw, roll):
    pr, yr, rr = math.radians(pitch), math.radians(yaw), math.radians(roll)
    Rx = np.array([[1, 0, 0], [0, math.cos(pr), -math.sin(pr)], [0, math.sin(pr), math.cos(pr)]])
    Ry = np.array([[math.cos(yr), 0, math.sin(yr)], [0, 1, 0], [-math.sin(yr), 0, math.cos(yr)]])
    Rz = np.array([[math.cos(rr), -math.sin(rr), 0], [math.sin(rr), math.cos(rr), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ── Spherical parameterization + 3D morphing (Yuan et al., arXiv:1907.09217) ───
def solve_sphere(M):
    X, Y, Z = M[:, 0], M[:, 1], M[:, 2]
    P = np.zeros((3, 3)); b = np.zeros(3)
    for i in range(3):
        P[i] = [X[i+1]-X[0], Y[i+1]-Y[0], Z[i+1]-Z[0]]
        b[i] = 0.5*((X[i+1]**2+Y[i+1]**2+Z[i+1]**2)-(X[0]**2+Y[0]**2+Z[0]**2))
    try:
        Sc = np.linalg.solve(P, b)
        x0, y0, z0 = Sc
        l = float(np.sqrt((X[0]-x0)**2+(Y[0]-y0)**2+(Z[0]-z0)**2))
        return x0, y0, z0, l
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, 1.0


def _normalize_by_centroid(points):
    center = points.mean(axis=0)
    centered = points - center
    scale = float(np.linalg.norm(centered, axis=1).mean())
    return centered / (scale + 1e-8), center


def _rotation_from_matrix_gs(R_raw):
    ro1 = R_raw[0] / (np.linalg.norm(R_raw[0]) + 1e-8)
    ro2 = R_raw[1] - np.dot(R_raw[1], ro1) * ro1
    ro2 = ro2 / (np.linalg.norm(ro2) + 1e-8)
    ro3 = np.cross(ro1, ro2)
    R = np.vstack([ro1, ro2, ro3])
    pitch, yaw, roll = _rotation_to_euler(R)
    return pitch, yaw, roll, R


def estimate_pose_spherical_morphing(m_points_2d, eta=1.77, initial_v=None,
                                      model_3d=PAPER_3D_MODEL, active_mask=[0, 1, 2, 3, 4]):
    """
    Documentation for this component.
    """
    if _scipy_minimize is None:
        raise RuntimeError("scipy is not installed; spherical morphing requires L-BFGS-B")
        
    # Implementation note.
    M_full_norm, _ = _normalize_by_centroid(model_3d)
    x0, y0, z0, l = solve_sphere(M_full_norm)
    n_full = M_full_norm.shape[0]
    phi_full = np.array([math.acos(max(min((M_full_norm[i, 2]-z0)/(l+1e-8), 1.), -1.)) for i in range(n_full)])
    theta_full = np.array([math.atan2(M_full_norm[i, 1]-y0, M_full_norm[i, 0]-x0) for i in range(n_full)])

    # Implementation note.
    m_points_2d = np.asarray(m_points_2d, dtype=np.float64)[:, :2]
    m_active = m_points_2d[active_mask]
    m_norm, m0 = _normalize_by_centroid(np.hstack([m_active, np.zeros((len(m_active), 1))]))
    
    M_active_norm = M_full_norm[active_mask]
    
    # Implementation note.
    R1_raw, _, _, _ = np.linalg.lstsq(M_active_norm, m_norm, rcond=None)
    R1_2d = R1_raw.T

    def _morph(v):
        mp_ = phi_full.copy(); mt = theta_full.copy()
        mp_[0] += v[0]  # nose
        mp_[1] += v[1]  # chin
        mp_[2] += v[2]  # left eye
        mp_[3] += v[2]  # right eye
        # Implementation note.
        
        mt[2] += v[3]  # left eye
        mt[3] -= v[3]  # right eye
        
        Mm_full = np.empty((n_full, 3))
        Mm_full[:, 0] = x0 + l*np.sin(mp_)*np.cos(mt)
        Mm_full[:, 1] = y0 + l*np.sin(mp_)*np.sin(mt)
        Mm_full[:, 2] = z0 + l*np.cos(mp_)
        
        Mm_active = Mm_full[active_mask]
        return eta*np.sum((Mm_active-M_active_norm)**2) + np.sum((m_norm-Mm_active.dot(R1_2d.T))**2), Mm_active, Mm_full

    v0 = np.zeros(4) if initial_v is None else np.asarray(initial_v, dtype=np.float64)
    res = _scipy_minimize(lambda v: _morph(v)[0], v0, method='L-BFGS-B',
                           bounds=[(-0.35, 0.35)]*3+[(-0.5, 0.5)],
                           options={'maxiter': 8, 'ftol': 1e-8, 'gtol': 1e-5})
    _, Mm_active_opt, Mm_full_opt = _morph(res.x)
    
    # Implementation note.
    R_opt_raw, _, _, _ = np.linalg.lstsq(Mm_active_opt, m_norm, rcond=None)
    pitch, yaw, roll, R = _rotation_from_matrix_gs(R_opt_raw.T)
    
    # Implementation note.
    if 2 not in active_mask and 3 in active_mask:
        # Implementation note.
        if yaw < 5.0:
            yaw = max(5.0, abs(yaw))
            R = _euler_to_rotation_matrix(pitch, yaw, roll)
    elif 3 not in active_mask and 2 in active_mask:
        # Implementation note.
        if yaw > -5.0:
            yaw = min(-5.0, -abs(yaw))
            R = _euler_to_rotation_matrix(pitch, yaw, roll)
            
    return pitch, yaw, roll, res.x, R


class SphericalHeadPoseEstimator:
    def __init__(self, eta=1.77, smooth_alpha=0.18, point_alpha=0.35,
                 max_angle_step=8.0, model_3d=PAPER_3D_MODEL):
        self.eta = eta
        self.point_alpha = point_alpha
        self.model_3d = model_3d
        
        self._v = None
        self._smooth_pts = None
        
        # Implementation note.
        self.filter_pitch = None
        self.filter_yaw = None
        self.filter_roll = None
        self.filter_v = [None, None, None, None]
        self.t_prev = None

    def update_points(self, m_pts, ts_ms=None):
        """
        Documentation for this component.
        """
        m_pts = np.asarray(m_pts, dtype=np.float64)[:, :2]
        
        # Implementation note.
        active_mask, confidence = validate_landmarks(m_pts)
        
        # Implementation note.
        if self._smooth_pts is None:
            self._smooth_pts = m_pts.copy()
        
        a = self.point_alpha
        self._smooth_pts = a * m_pts + (1. - a) * self._smooth_pts
        pts = self._smooth_pts

        # Implementation note.
        pitch, yaw, roll, solved_v, R = estimate_pose_spherical_morphing(
            pts, eta=self.eta, initial_v=self._v, model_3d=self.model_3d, active_mask=active_mask)

        # Implementation note.
        t = ts_ms if ts_ms is not None else (time.time() * 1000.0)
        
        if self.t_prev is None or (t - self.t_prev) <= 0.0:
            # Implementation note.
            self.filter_pitch = OneEuroFilter(t, pitch, min_cutoff=0.8, beta=0.015)
            self.filter_yaw = OneEuroFilter(t, yaw, min_cutoff=0.8, beta=0.015)
            self.filter_roll = OneEuroFilter(t, roll, min_cutoff=1.5, beta=0.01)
            
            for idx in range(4):
                self.filter_v[idx] = OneEuroFilter(t, solved_v[idx], min_cutoff=0.05, beta=0.002)
                
            self._v = solved_v
            self.t_prev = t
        else:
            # Implementation note.
            pitch = self.filter_pitch(t, pitch)
            yaw = self.filter_yaw(t, yaw)
            roll = self.filter_roll(t, roll)
            
            # Implementation note.
            filtered_v = np.empty(4)
            for idx in range(4):
                filtered_v[idx] = self.filter_v[idx](t, solved_v[idx])
            self._v = filtered_v
            self.t_prev = t

        # Implementation note.
        R_smooth = _euler_to_rotation_matrix(pitch, yaw, roll)
        
        return {
            "pitch": float(pitch), 
            "yaw": float(yaw), 
            "roll": float(roll), 
            "R_smooth": R_smooth,
            "confidence": confidence
        }



# ── Angle utilities ──────────────────────────────────────────────────────────
def _wrap180(d):    return (d + 180.) % 360. - 180.
def _unwrap(a, r):   return r + _wrap180(a - r)
def _clamp_step(a, p, s): d = _wrap180(a - p); return p + max(-s, min(s, d))


# Global estimator cache
_pose_estimators_spherical = {}


def get_frame_for_metadata(cam_id: str, meta_ts_ms: float):
    """
    Documentation for this component.
    Documentation for this component.
      Documentation for this component.
      Documentation for this component.
      Documentation for this component.
    """
    target_ts = meta_ts_ms + PIPELINE_DELAY_MS
    with frame_locks[cam_id]:
        buf = list(frame_buffers[cam_id])  # [(wall_ms, frame), ...]

    if not buf:
        return None, None

    # Implementation note.
    best_idx = min(range(len(buf)), key=lambda i: abs(buf[i][0] - target_ts))
    best_ts, best_frame = buf[best_idx]
    return best_frame, best_ts


# ── RTSP Stream Threaded Captures ──────────────────────────────────────────────
def rtsp_capture_worker(cam_id, rtsp_url):
    print(f"[RTSP] Starting capture thread for {cam_id}...")
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    reconnect_delay = 2
    while not stop_event.is_set():
        if not cap.isOpened():
            print(f"[RTSP] {cam_id} disconnected. Reconnecting in {reconnect_delay}s...")
            time.sleep(reconnect_delay)
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        ret, frame = cap.read()
        if not ret:
            print(f"[RTSP] {cam_id} read error. Reconnecting...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        # Implementation note.
        wall_ms = time.time() * 1000.0
        with frame_locks[cam_id]:
            frame_buffers[cam_id].append((wall_ms, frame))

    cap.release()
    print(f"[RTSP] Capture thread for {cam_id} stopped.")


# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
def gst_capture_worker(cam_id, rtsp_url):
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst, GLib
    except ImportError:
        print(f"[GST] PyGObject/GStreamer chưa cài. Cài đặt:\n"
              f"  Windows: tải GStreamer runtime+devel (MSVC) tại gstreamer.freedesktop.org,\n"
              f"           rồi 'pip install PyGObject' (hoặc dùng gói vendor kèm sẵn gi).\n"
              f"  Linux:   sudo apt install python3-gi gstreamer1.0-plugins-{{good,bad,ugly}} "
              f"gstreamer1.0-libav\n"
              f"Hoặc chạy lại với --backend opencv để dùng OpenCV như cũ.")
        stop_event.set()
        return

    Gst.init(None)
    print(f"[GST] Starting capture pipeline for {cam_id}...")

    # Implementation note.
    # Implementation note.
    # Implementation note.
    # Implementation note.
    #
    # Implementation note.
    # Implementation note.
    # Implementation note.
    # Implementation note.
    has_nvvidconv = Gst.ElementFactory.find("nvvideoconvert") is not None
    convert_chain = "nvvideoconvert ! video/x-raw,format=BGRx ! videoconvert" if has_nvvidconv else "videoconvert"
    pipeline_str = (
        f"rtspsrc location={rtsp_url} latency=200 protocols=tcp ! "
        f"decodebin ! {convert_chain} ! video/x-raw,format=BGR ! "
        f"appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
    )
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name("sink")

    anchor = {"wall_ms": None, "pts_ns": None}

    def on_new_sample(sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        width, height = s.get_value("width"), s.get_value("height")

        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            frame = np.ndarray((height, width, 3), buffer=map_info.data, dtype=np.uint8).copy()
        finally:
            buf.unmap(map_info)

        pts_ns = buf.pts
        if pts_ns is None or pts_ns == Gst.CLOCK_TIME_NONE:
            wall_ms = time.time() * 1000.0  # Implementation note.
        else:
            if anchor["pts_ns"] is None:
                anchor["wall_ms"] = time.time() * 1000.0
                anchor["pts_ns"] = pts_ns
            wall_ms = anchor["wall_ms"] + (pts_ns - anchor["pts_ns"]) / 1e6

        with frame_locks[cam_id]:
            frame_buffers[cam_id].append((wall_ms, frame))
        return Gst.FlowReturn.OK

    appsink.connect("new-sample", on_new_sample)

    bus = pipeline.get_bus()

    def on_bus_message(bus, message, loop):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"[GST] {cam_id} error: {err} ({dbg})")
        elif t == Gst.MessageType.EOS:
            print(f"[GST] {cam_id} end of stream.")
        return True

    loop = GLib.MainLoop()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, loop)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        while not stop_event.is_set():
            loop.get_context().iteration(True)
    finally:
        pipeline.set_state(Gst.State.NULL)
    print(f"[GST] Capture pipeline for {cam_id} stopped.")


# ── MQTT Client ────────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    topics = userdata["topics"]
    if rc == 0:
        for t in topics:
            client.subscribe(t, qos=0)
        print(f"[MQTT] Connected, subscribed to: {topics}")
    else:
        print(f"[MQTT] Connect failed with code {rc}")


def on_message(client, userdata, msg):
    global latest_metadatas
    try:
        parts = msg.topic.split("/")
        if len(parts) >= 2:
            cam_id = parts[1]
            data = json.loads(msg.payload.decode("utf-8"))
            with metadata_lock:
                latest_metadatas[cam_id] = data
    except Exception as e:
        print(f"[MQTT] Parse message error on topic {msg.topic}: {e}")


def start_mqtt(broker, port, topics):
    client = mqtt.Client(CallbackAPIVersion.VERSION1, client_id="gaze-viewer-script")
    client.user_data_set({"topics": topics})
    client.on_connect = on_connect
    client.on_message = on_message
    
    print(f"[MQTT] Connecting to broker {broker}:{port}...")
    client.connect(broker, port, 60)
    
    t = threading.Thread(target=client.loop_forever, daemon=True)
    t.start()
    return client


# ── MJPEG HTTP Streaming Server ───────────────────────────────────────────────
class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global current_preview_frame
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with preview_lock:
                        if current_preview_frame is None:
                            img = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(img, "Waiting for video streams...", (130, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            _, jpeg = cv2.imencode('.jpg', img)
                        else:
                            h, w = current_preview_frame.shape[:2]
                            scale = min(1920 / w, 1080 / h, 1.0)
                            preview = cv2.resize(current_preview_frame,
                                                 (int(w * scale), int(h * scale)))
                            _, jpeg = cv2.imencode('.jpg', preview,
                                                   [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                        frame_bytes = jpeg.tobytes()

                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.04)  # ~25 FPS
            except Exception:
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Implementation note.


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_web_server(port):
    server = ThreadedHTTPServer(('0.0.0.0', port), StreamingHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[WEB] MJPEG Streamer started at http://localhost:{port}")
    return server


# ── Rendering overlay ─────────────────────────────────────────────────────────
def draw_overlay(frame, metadata, cam_id=""):
    global _pose_estimators_spherical
    if not metadata or "d" not in metadata:
        return

    frame_h, frame_w = frame.shape[:2]

    for i, det in enumerate(metadata["d"]):
        box = det.get("box")
        if not box:
            continue

        x1, y1, x2, y2 = box
        person_id = det.get("id", i)

        rx1, ry1, rx2, ry2 = x1, y1, x2, y2

        # Implementation note.
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        
        # Details
        pts = det.get("pts")
        
        yaw_s, pitch_s, gaze_s, conf_s = None, None, None, det.get("conf")
        meta_ts = metadata.get("ts")

        if pts is not None:
            key = (cam_id, person_id)
            
            # Implementation note.
            def pt(name):
                v = np.array(pts[name], dtype=np.float64)
                v[1] = -v[1]
                return v

            # Implementation note.
            if _scipy_minimize is not None:
                est_s = _pose_estimators_spherical.get(key)
                if est_s is None:
                    est_s = SphericalHeadPoseEstimator()
                    _pose_estimators_spherical[key] = est_s
                
                try:
                    m_pts_s = np.vstack([
                        pt("nose"), pt("chin"), pt("left_eye"), pt("right_eye"), pt("bridge")
                    ])
                    pose_s = est_s.update_points(m_pts_s, ts_ms=meta_ts)
                    yaw_s = pose_s["yaw"]
                    pitch_s = pose_s["pitch"]
                    R_s = pose_s["R_smooth"]
                    conf_s = pose_s["confidence"]
                    gaze_s = [float(R_s[0,2]), -float(R_s[1,2]), float(R_s[2,2])]
                except Exception as ex:
                    pass

        # Implementation note.
        num_landmarks = 0
        if pts is not None:
            for name, coords in pts.items():
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    num_landmarks += 1

        # Implementation note.
        label_y = max(25, ry1 - 10)
        label_text = f"ID:{person_id} LMs:{num_landmarks}"
        if _scipy_minimize is None:
            label_text += " (scipy missing)"
        elif yaw_s is not None and pitch_s is not None:
            label_text += f" Y:{yaw_s:+.1f} P:{pitch_s:+.1f}"
            
        if conf_s is not None:
            label_text += f" C:{conf_s:.1f}"

        cv2.putText(frame, label_text, (rx1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # Implementation note.
        if gaze_s is not None:
            gx, gy, gz = gaze_s
            ox = int((rx1 + rx2) / 2)
            oy = int((ry1 + ry2) / 2)
            cv2.circle(frame, (ox, oy), 5, (0, 255, 255), -1)  # Implementation note.
            
            box_width = rx2 - rx1
            gaze_scale = max(100, box_width * 1.0)
            ex = int(ox + gx * gaze_scale)
            ey = int(oy + gy * gaze_scale)
            
            # Implementation note.
            # Implementation note.
            # Implementation note.
            # Implementation note.
            arrow_color = (0, 0, 255)
            if conf_s is not None:
                if abs(conf_s - 0.8) < 0.05:
                    arrow_color = (0, 255, 255)
                elif conf_s <= 0.65:
                    arrow_color = (0, 140, 255)
                    
            cv2.arrowedLine(frame, (ox, oy), (ex, ey), arrow_color, 4,
                            tipLength=0.25, line_type=cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global PIPELINE_DELAY_MS
    parser = argparse.ArgumentParser(description="Gaze and Headpose overlay viewer.")
    parser.add_argument("--camera", type=str, default="cam0", choices=["cam0", "cam1", "all"],
                        help="Camera identifier (cam0, cam1, or all)")
    parser.add_argument("--mqtt-host", type=str, default="127.0.0.1",
                        help="MQTT broker host IP")
    parser.add_argument("--mqtt-port", type=int, default=1883,
                        help="MQTT broker port")
    parser.add_argument("--app-api", type=str, default="http://127.0.0.1:8080",
                        help="application service REST API, dùng để lấy URL camera "
                             "đã proxy qua mediamtx thay vì pull thẳng camera")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Open a local GUI display window (cv2.imshow)")
    parser.add_argument("--web", dest="web", action="store_true", default=True,
                        help="Start MJPEG web streaming server (default: True)")
    parser.add_argument("--no-web", dest="web", action="store_false",
                        help="Tắt MJPEG web server — dùng khi chỉ cần --gui, tránh "
                             "cv2.imshow (main thread) and cv2.imencode (web thread) "
                             "cùng đụng vào Qt internals gây spam 'QObject::killTimer'.")
    parser.add_argument("--web-port", type=int, default=5000,
                        help="MJPEG web server port")
    parser.add_argument("--pipeline-delay", type=float, default=PIPELINE_DELAY_MS,
                        help="Ước tính độ trễ pipeline Jetson->laptop (ms), CHỈ đúng nếu "
                             "2 máy đã NTP-sync đồng hồ. Tăng nếu bbox chạy trước mặt, "
                             f"giảm nếu bbox chạy sau. (default: {PIPELINE_DELAY_MS:.0f})")
    parser.add_argument("--backend", type=str, default="opencv", choices=["opencv", "gstreamer"],
                        help="Cách đọc RTSP: 'opencv' (mặc định, cũ) hay 'gstreamer' "
                             "(PTS thật per-frame thay vì đoán theo giờ nhận, thử nghiệm "
                             "để so sánh độ đồng bộ). Cần cài GStreamer+PyGObject riêng "
                             "nếu chọn 'gstreamer'.")
    parser.add_argument("--scrfd", action="store_true", default=False,
                        help="Chạy nhận diện mặt và landmark bằng SCRFD TensorRT cục bộ thay vì dùng MediaPipe từ MQTT.")
    parser.add_argument("--scrfd-engine", type=str, default=None,
                        help="Đường dẫn tới file .engine của SCRFD. Mặc định tự động tìm theo các vị trí phổ biến.")
    args = parser.parse_args()

    PIPELINE_DELAY_MS = args.pipeline_delay
    print(f"[Viewer] Pipeline delay compensation: {PIPELINE_DELAY_MS:.0f}ms")
    
    # Identify which camera channels to activate
    active_cams = ["cam0", "cam1"] if args.camera == "all" else [args.camera]

    # Implementation note.
    # Implementation note.
    fetched = fetch_camera_urls(args.app_api)
    camera_urls = dict(CAMERA_RTSP_URLS)
    camera_urls.update(fetched)
    if not fetched:
        print("[Viewer] WARNING: dùng URL camera trực tiếp (fallback) — "
              "sẽ pull RTSP song song với deepstream-service.")

    # Configure topics
    mqtt_topics = [f"gaze/{cam}/calculated" for cam in active_cams]

    print(f"=== Starting Gaze Viewer (Camera: {args.camera}) ===")
    for cam in active_cams:
        print(f" - {cam} RTSP Source: {camera_urls[cam]}")
    print(f"MQTT Topics : {mqtt_topics}")
    
    # Start MQTT client
    mqtt_client = start_mqtt(args.mqtt_host, args.mqtt_port, mqtt_topics)

    # Start Background RTSP capture threads
    capture_fn = gst_capture_worker if args.backend == "gstreamer" else rtsp_capture_worker
    print(f"[Viewer] RTSP backend: {args.backend}")
    capture_threads = []
    for cam in active_cams:
        t = threading.Thread(target=capture_fn, args=(cam, camera_urls[cam]), daemon=True)
        t.start()
        capture_threads.append(t)

    # Initialize SCRFD Process if requested (runs in a separate spawned process to isolate GStreamer NVDEC from TensorRT CUDA contexts)
    scrfd_in_q = None
    scrfd_out_q = None
    scrfd_proc = None
    pose_caches = {}
    if args.scrfd:
        scrfd_in_q = multiprocessing.Queue(maxsize=2)
        scrfd_out_q = multiprocessing.Queue()

        # Implementation note.
        if args.scrfd_engine:
            engine_path = args.scrfd_engine
        else:
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            _candidates = [
                os.path.join(_script_dir, "..", "models", "scrfd", "det_10g.engine"),
                os.path.join(os.getcwd(), "models", "scrfd", "det_10g.engine"),
            ]
            engine_path = next((p for p in _candidates if os.path.exists(p)), None)

        if not engine_path or not os.path.exists(engine_path):
            print("[TRT] ERROR: Không tìm thấy SCRFD engine. Hãy chỉ định bằng --scrfd-engine /path/to/det_10g.engine")
            sys.exit(1)

        print(f"[TRT] Dùng SCRFD engine: {engine_path}")
        scrfd_proc = SCRFDProcess(engine_path, scrfd_in_q, scrfd_out_q)
        scrfd_proc.start()
        print("[Viewer] Spawned SCRFD child process successfully.")
        
        for cam in active_cams:
            pose_caches[cam] = {}
        
    # Start Web Preview Server
    web_server = None
    if args.web:
        web_server = start_web_server(args.web_port)
        
    print("[Viewer] System initialized. Press Ctrl+C to stop.")
    
    fps_time = time.time()
    frames_count = 0
    
    last_rendered_ts = {cam: -1 for cam in active_cams}
    display_frames = {cam: None for cam in active_cams}
    # Implementation note.
    last_scrfd_dets = {cam: (np.empty((0,5)), np.empty((0,10))) for cam in active_cams}

    try:
        while True:
            # Implementation note.
            with metadata_lock:
                metas = {cam: latest_metadatas[cam] for cam in active_cams}

            # Implementation note.
            for cam in active_cams:
                if args.scrfd:
                    with frame_locks[cam]:
                        buf = frame_buffers[cam]
                        if buf:
                            ts, frame = buf[-1]
                            if ts != last_rendered_ts[cam]:
                                # Implementation note.
                                try:
                                    scrfd_in_q.put((cam, ts, frame), block=False)
                                except queue.Full:
                                    pass
                                last_rendered_ts[cam] = ts
                                
                                # Implementation note.
                                draw_frame = frame.copy()
                                dets, kps_all = last_scrfd_dets[cam]
                                if len(dets) > 0:
                                    draw_scrfd_overlay(draw_frame, dets, kps_all, pose_caches[cam])
                                # Implementation note.
                                status = f"SCRFD: {len(dets)} face(s)"
                                cv2.putText(draw_frame, status, (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                                display_frames[cam] = draw_frame
                else:
                    meta = metas.get(cam)
                    if meta is not None:
                        meta_ts = meta.get("ts", 0)
                        frame, matched_ts = get_frame_for_metadata(cam, meta_ts)
                        if frame is not None:
                            draw_frame = frame.copy()
                            draw_overlay(draw_frame, meta, cam_id=cam)
                            display_frames[cam] = draw_frame
                    else:
                        with frame_locks[cam]:
                            buf = frame_buffers[cam]
                            if buf:
                                draw_frame = buf[-1][1].copy()
                                draw_overlay(draw_frame, None, cam_id=cam)
                                display_frames[cam] = draw_frame

            # Implementation note.
            if args.scrfd:
                while True:
                    try:
                        cam, ts, _frame, dets, kps_all = scrfd_out_q.get_nowait()
                        last_scrfd_dets[cam] = (dets, kps_all)
                    except queue.Empty:
                        break

            # Implementation note.
            has_any = any(display_frames[cam] is not None for cam in active_cams)
            if not has_any:
                time.sleep(0.02)
                continue

            # Implementation note.
            if args.camera == "all":
                f0 = display_frames.get("cam0")
                f1 = display_frames.get("cam1")
                
                target_h, target_w = 540, 960  # Implementation note.
                
                if f0 is not None:
                    f0_resized = cv2.resize(f0, (target_w, target_h))
                else:
                    f0_resized = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    cv2.putText(f0_resized, "Camera 0 offline", (300, 270),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                
                if f1 is not None:
                    f1_resized = cv2.resize(f1, (target_w, target_h))
                else:
                    f1_resized = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    cv2.putText(f1_resized, "Camera 1 offline", (300, 270),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                
                # Implementation note.
                composite_frame = np.hstack((f0_resized, f1_resized))
            else:
                composite_frame = display_frames.get(args.camera)
                
            if composite_frame is None:
                time.sleep(0.01)
                continue
                
            # Update web preview frame
            if args.web:
                with preview_lock:
                    global current_preview_frame
                    current_preview_frame = composite_frame.copy()
                    
            # Local GUI Window
            if args.gui:
                cv2.imshow("Gaze Unified Viewer", composite_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            # Calculate FPS
            frames_count += 1
            if time.time() - fps_time >= 5.0:
                fps = frames_count / (time.time() - fps_time)
                print(f"[Viewer] Render FPS: {fps:.1f}")
                frames_count = 0
                fps_time = time.time()
                
            time.sleep(0.01)  # Throttle rendering loop to avoid 100% CPU usage
            
    except KeyboardInterrupt:
        print("\nStopping Gaze Viewer...")
    finally:
        stop_event.set()
        # Implementation note.
        if scrfd_proc is not None and scrfd_proc.is_alive():
            try:
                scrfd_in_q.put(None, timeout=1)  # Implementation note.
            except Exception:
                pass
            scrfd_proc.terminate()
            scrfd_proc.join(timeout=3)
        if scrfd_in_q is not None:
            scrfd_in_q.cancel_join_thread()
        if scrfd_out_q is not None:
            scrfd_out_q.cancel_join_thread()
        cv2.destroyAllWindows()
        print("Gaze Viewer stopped.")


if __name__ == "__main__":
    main()
