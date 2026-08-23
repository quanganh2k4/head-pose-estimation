"""
Unit tests for modules/head_pose_algo.py. They run on any machine with numpy and
scipy, without GPU/DeepStream, and are used as the algorithm CI check.

The main strategy projects the standard PAPER_3D_MODEL at known yaw/pitch angles
onto 2D, feeds those points to the estimator, and checks sign and magnitude.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.head_pose_algo import (  # noqa: E402
    PAPER_3D_MODEL,
    OneEuroFilter,
    SphericalHeadPoseEstimator,
    _euler_to_rotation_matrix,
    _wrap180,
    estimate_pose_spherical_morphing,
    validate_landmarks,
)


def project_model(pitch=0.0, yaw=0.0, roll=0.0):
    """Orthographically project a rotated 3D model onto 2D, dropping Z."""
    R = _euler_to_rotation_matrix(pitch, yaw, roll)
    rotated = PAPER_3D_MODEL @ R.T
    return rotated[:, :2]


# ── _wrap180 ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("deg,expected", [(0, 0), (180, -180), (190, -170), (-190, 170), (360, 0)])
def test_wrap180(deg, expected):
    assert _wrap180(deg) == pytest.approx(expected)


# ── OneEuroFilter ────────────────────────────────────────────────────────────
def test_one_euro_constant_signal_unchanged():
    f = OneEuroFilter(t0=0.0, x0=10.0)
    for t_ms in range(33, 1000, 33):
        out = f(float(t_ms), 10.0)
    assert out == pytest.approx(10.0, abs=1e-6)

def test_one_euro_reduces_noise_variance():
    rng = np.random.default_rng(42)
    signal = 5.0 + rng.normal(0.0, 2.0, size=200)
    f = OneEuroFilter(t0=0.0, x0=signal[0], min_cutoff=0.5, beta=0.01)
    filtered = [f(float((i + 1) * 33), x) for i, x in enumerate(signal[1:])]
    assert np.std(filtered) < np.std(signal) * 0.6

def test_one_euro_non_positive_dt_returns_previous():
    f = OneEuroFilter(t0=100.0, x0=1.0)
    assert f(100.0, 99.0) == pytest.approx(1.0)   # dt = 0: keep the old value
    assert f(50.0, 99.0) == pytest.approx(1.0)    # dt < 0: keep the old value


# ── validate_landmarks (Geometric Consistency Check) ────────────────────────
def test_validate_landmarks_frontal_face_all_points_active():
    pts = project_model(0.0, 0.0, 0.0)
    active_mask, confidence = validate_landmarks(pts)
    assert sorted(active_mask) == [0, 1, 2, 3, 4]
    assert confidence == pytest.approx(1.0)

def test_validate_landmarks_occluded_eye_is_dropped():
    pts = project_model(0.0, 0.0, 0.0).copy()
    pts[2] = pts[4]  # left eye overlaps bridge: severe profile / invalid landmark
    active_mask, confidence = validate_landmarks(pts)
    assert 2 not in active_mask
    assert confidence < 1.0

def test_validate_landmarks_always_at_least_three_points():
    pts = project_model(0.0, 0.0, 0.0).copy()
    pts[1] = pts[0]  # invalid chin
    pts[2] = pts[4]  # invalid left eye
    pts[3] = pts[4]  # invalid right eye
    active_mask, _ = validate_landmarks(pts)
    assert len(active_mask) >= 3


# ── estimate_pose_spherical_morphing ─────────────────────────────────────────
def test_estimate_frontal_face_near_zero_angles():
    pts = project_model(0.0, 0.0, 0.0)
    pitch, yaw, roll, _, _ = estimate_pose_spherical_morphing(pts)
    assert abs(yaw) < 5.0
    assert abs(pitch) < 5.0
    assert abs(roll) < 5.0

@pytest.mark.parametrize("true_yaw", [-25.0, -15.0, 15.0, 25.0])
def test_estimate_recovers_yaw_sign_and_magnitude(true_yaw):
    pts = project_model(yaw=true_yaw)
    _, yaw, _, _, _ = estimate_pose_spherical_morphing(pts)
    assert np.sign(yaw) == np.sign(true_yaw)
    assert abs(yaw - true_yaw) < 12.0

@pytest.mark.parametrize("true_pitch", [-20.0, 20.0])
def test_estimate_recovers_pitch_sign(true_pitch):
    pts = project_model(pitch=true_pitch)
    pitch, _, _, _, _ = estimate_pose_spherical_morphing(pts)
    assert np.sign(pitch) == np.sign(true_pitch)


# ── SphericalHeadPoseEstimator (full validation + morphing + filtering pipeline) ─
def test_estimator_stable_on_static_frontal_face():
    est = SphericalHeadPoseEstimator()
    rng = np.random.default_rng(7)
    pts_base = project_model(0.0, 0.0, 0.0)
    yaws = []
    for i in range(30):
        noisy = pts_base + rng.normal(0.0, 0.5, size=pts_base.shape)
        pose = est.update_points(noisy, ts_ms=float(i * 33))
        yaws.append(pose["yaw"])
    # The final 10 frames should be near zero with little jitter after filtering.
    tail = np.array(yaws[-10:])
    assert np.all(np.abs(tail) < 6.0)
    assert np.std(tail) < 1.5

def test_estimator_tracks_head_turn():
    est = SphericalHeadPoseEstimator()
    for i in range(15):
        est.update_points(project_model(yaw=0.0), ts_ms=float(i * 33))
    pose = None
    for i in range(15, 60):
        # Turn the head gradually to 30 degrees for about 0.5 seconds, then hold.
        target = min(30.0, (i - 14) * 2.0)
        pose = est.update_points(project_model(yaw=target), ts_ms=float(i * 33))
    assert pose["yaw"] > 15.0

def test_estimator_returns_rotation_matrix_and_confidence():
    est = SphericalHeadPoseEstimator()
    pose = est.update_points(project_model(yaw=10.0), ts_ms=0.0)
    R = pose["R_smooth"]
    assert R.shape == (3, 3)
    # R must be a valid rotation matrix (orthogonal, determinant approximately 1).
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < pose["confidence"] <= 1.0
