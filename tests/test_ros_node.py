import math

import numpy as np
from builtin_interfaces.msg import Time

from ros_node import build_command_message, build_state_vector


def test_build_state_vector_uses_absolute_pose_only() -> None:
    current_pose = np.array([1.0, 0.5, 0.2], dtype=np.float32)

    state = build_state_vector(current_pose)

    assert state.shape == (4,)
    assert np.allclose(state[:2], np.array([1.0, 0.5], dtype=np.float32))
    assert np.allclose(state[2:4], np.array([math.sin(0.2), math.cos(0.2)], dtype=np.float32))


def test_build_command_message_sets_header_stamp() -> None:
    stamp = Time(sec=12, nanosec=34)

    msg = build_command_message(0.25, 0.75, stamp=stamp)

    assert msg.drive.steering_angle == 0.25
    assert msg.drive.speed == 0.75
    assert msg.header.stamp == stamp
