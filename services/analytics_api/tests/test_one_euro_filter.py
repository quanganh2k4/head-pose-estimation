import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.head_pose_algo import OneEuroFilter


def test_one_euro_filter_converges_towards_constant_input():
    filt = OneEuroFilter(0.0, 0.0)
    outputs = [filt(float(i) * 100.0, 10.0) for i in range(1, 30)]

    assert outputs[-1] > outputs[0]
    assert outputs[-1] == 10.0 or outputs[-1] > 9.0


def test_one_euro_filter_keeps_previous_value_for_non_monotonic_time():
    filt = OneEuroFilter(100.0, 3.5)

    assert filt(100.0, 99.0) == 3.5
    assert filt(50.0, 99.0) == 3.5
