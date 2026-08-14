"""
head_pose_algo.py — robust head-pose estimator using 5-point Spherical Morphing.
Includes:
  - OneEuroFilter for dynamic adaptive noise filtering
  - Geometric Consistency Check for landmark outlier/occlusion detection
  - Dynamic Subset Projector for estimation with missing landmarks
  - Yaw direction constraints under extreme profile angles
"""
import math
import numpy as np
import time

try:
    from scipy.optimize import minimize as _scipy_minimize
except Exception:
    _scipy_minimize = None


# ── 3D face model gốc của bài báo (5 điểm: nose, chin, left_eye, right_eye, bridge) ─────
PAPER_3D_MODEL = np.array([
    [  0.0,    0.0,   50.0],   # nose tip
    [  0.0, -115.0,  -35.0],   # chin
    [-48.0,   35.0,  -25.0],   # left eye
    [ 48.0,   35.0,  -25.0],   # right eye
    [  0.0,   20.0,    5.0],   # nose bridge
], dtype=np.float64)


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
        dt = (t - self.t_prev) / 1000.0  # Chuyển sang giây
        if dt <= 0.0:
            return self.x_prev
        
        # Tính toán đạo hàm
        dx = (x - self.x_prev) / dt
        
        # Lọc đạo hàm
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        
        # Cắt tần số tự động điều chỉnh theo vận tốc thay đổi
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Lọc giá trị tín hiệu
        alpha_x = self._alpha(dt, cutoff)
        x_hat = alpha_x * x + (1.0 - alpha_x) * self.x_prev
        
        # Lưu vết trạng thái
        self.x_prev = x_hat
        self.t_prev = t
        self.dx_prev = dx_hat
        
        return x_hat

    def _alpha(self, dt, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


# ── Phân tích và phát hiện mốc bất thường ─────────────────────────────────────
def validate_landmarks(pts):
    """
    pts: Nx2 array of landmarks [nose, chin, left_eye, right_eye, bridge]
    Trả về: (active_mask, confidence)
    """
    pts = np.asarray(pts, dtype=np.float64)[:, :2]
    
    # 5 điểm indices: 0: nose, 1: chin, 2: left_eye, 3: right_eye, 4: bridge
    d_eyes = np.linalg.norm(pts[2] - pts[3])
    d_nose_chin = np.linalg.norm(pts[0] - pts[1])
    d_left_bridge = np.linalg.norm(pts[2] - pts[4])
    d_right_bridge = np.linalg.norm(pts[3] - pts[4])
    
    # Kiểm tra 1: Che miệng/cằm (d_nose_chin so với d_eyes)
    ratio_chin = d_nose_chin / (d_eyes + 1e-8)
    chin_ok = 0.7 <= ratio_chin <= 3.5
    
    # Kiểm tra 2: Quay mặt nghiêng quá sâu (mắt tiến sát sống mũi)
    left_eye_ok = d_left_bridge > 0.15 * d_eyes
    right_eye_ok = d_right_bridge > 0.15 * d_eyes
    
    active_mask = [0, 4]  # Mũi và sống mũi luôn luôn là neo cứng đáng tin cậy
    if chin_ok:
        active_mask.append(1)
    if left_eye_ok:
        active_mask.append(2)
    if right_eye_ok:
        active_mask.append(3)
        
    # Bảo đảm tối thiểu 3 điểm không thẳng hàng để giải PnP
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
    Hỗ trợ giải tối ưu trên tập con động thông qua active_mask.
    """
    if _scipy_minimize is None:
        raise RuntimeError("scipy chưa cài — cần cho spherical morphing (L-BFGS-B)")
        
    # Tính toán thông số cầu trên mô hình đầy đủ 5 điểm để giữ vững cấu hình gốc
    M_full_norm, _ = _normalize_by_centroid(model_3d)
    x0, y0, z0, l = solve_sphere(M_full_norm)
    n_full = M_full_norm.shape[0]
    phi_full = np.array([math.acos(max(min((M_full_norm[i, 2]-z0)/(l+1e-8), 1.), -1.)) for i in range(n_full)])
    theta_full = np.array([math.atan2(M_full_norm[i, 1]-y0, M_full_norm[i, 0]-x0) for i in range(n_full)])

    # Lọc ra các điểm mốc và mô hình con theo active_mask
    m_points_2d = np.asarray(m_points_2d, dtype=np.float64)[:, :2]
    m_active = m_points_2d[active_mask]
    m_norm, m0 = _normalize_by_centroid(np.hstack([m_active, np.zeros((len(m_active), 1))]))
    
    M_active_norm = M_full_norm[active_mask]
    
    # Giải sơ bộ ma trận xoay
    R1_raw, _, _, _ = np.linalg.lstsq(M_active_norm, m_norm, rcond=None)
    R1_2d = R1_raw.T

    def _morph(v):
        mp_ = phi_full.copy(); mt = theta_full.copy()
        mp_[0] += v[0]  # nose
        mp_[1] += v[1]  # chin
        mp_[2] += v[2]  # left eye
        mp_[3] += v[2]  # right eye
        # index 4 (bridge) giữ cố định
        
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
    
    # Giải ma trận xoay tối ưu dựa trên phân bổ thực tế của các điểm hoạt động
    R_opt_raw, _, _, _ = np.linalg.lstsq(Mm_active_opt, m_norm, rcond=None)
    pitch, yaw, roll, R = _rotation_from_matrix_gs(R_opt_raw.T)
    
    # Ràng buộc góc xoay Yaw nếu một bên mắt bị khuất (Yaw Bound Constraint)
    if 2 not in active_mask and 3 in active_mask:
        # Mắt trái bị che khuất -> Đầu quay sang phải (yaw dương)
        if yaw < 5.0:
            yaw = max(5.0, abs(yaw))
            R = _euler_to_rotation_matrix(pitch, yaw, roll)
    elif 3 not in active_mask and 2 in active_mask:
        # Mắt phải bị che khuất -> Đầu quay sang trái (yaw âm)
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
        
        # Bộ lọc One-Euro Filters thích ứng động cho góc Pose và tham số v
        self.filter_pitch = None
        self.filter_yaw = None
        self.filter_roll = None
        self.filter_v = [None, None, None, None]
        self.t_prev = None

    def update_points(self, m_pts, ts_ms=None):
        """
        Cập nhật landmarks, tự động phân tích và áp dụng One-Euro Filter động.
        """
        m_pts = np.asarray(m_pts, dtype=np.float64)[:, :2]
        
        # 1. Phát hiện điểm mốc bất thường bằng Geometric Consistency
        active_mask, confidence = validate_landmarks(m_pts)
        
        # 2. Làm mượt điểm mốc đầu vào
        if self._smooth_pts is None:
            self._smooth_pts = m_pts.copy()
        
        a = self.point_alpha
        self._smooth_pts = a * m_pts + (1. - a) * self._smooth_pts
        pts = self._smooth_pts

        # 3. Tính toán góc quay bằng Spherical Morphing
        pitch, yaw, roll, solved_v, R = estimate_pose_spherical_morphing(
            pts, eta=self.eta, initial_v=self._v, model_3d=self.model_3d, active_mask=active_mask)

        # 4. Sử dụng bộ lọc One-Euro Filter làm mượt thích ứng
        t = ts_ms if ts_ms is not None else (time.time() * 1000.0)
        
        if self.t_prev is None or (t - self.t_prev) <= 0.0:
            # Khởi tạo giá trị ban đầu cho các bộ lọc
            self.filter_pitch = OneEuroFilter(t, pitch, min_cutoff=0.8, beta=0.015)
            self.filter_yaw = OneEuroFilter(t, yaw, min_cutoff=0.8, beta=0.015)
            self.filter_roll = OneEuroFilter(t, roll, min_cutoff=1.5, beta=0.01)
            
            for idx in range(4):
                self.filter_v[idx] = OneEuroFilter(t, solved_v[idx], min_cutoff=0.05, beta=0.002)
                
            self._v = solved_v
            self.t_prev = t
        else:
            # Lọc góc quay đầu ra
            pitch = self.filter_pitch(t, pitch)
            yaw = self.filter_yaw(t, yaw)
            roll = self.filter_roll(t, roll)
            
            # Lọc tham số hình học v
            filtered_v = np.empty(4)
            for idx in range(4):
                filtered_v[idx] = self.filter_v[idx](t, solved_v[idx])
            self._v = filtered_v
            self.t_prev = t

        # Tái dựng lại ma trận xoay mượt mà cuối cùng
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
