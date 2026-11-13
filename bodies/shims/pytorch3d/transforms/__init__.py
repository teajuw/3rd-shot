from .rotation_conversions import *  # noqa: F401,F403
from .rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle


def so3_exp_map(log_rot, eps=1e-4):
    return axis_angle_to_matrix(log_rot)


def so3_log_map(R, eps=1e-4, cos_bound=1e-4):
    return matrix_to_axis_angle(R)
