import math

import numpy as np


def perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    matrix[3, 3] = 0.0
    return matrix


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0:3] = right
    matrix[1, 0:3] = true_up
    matrix[2, 0:3] = -forward
    matrix[0, 3] = -float(np.dot(right, eye))
    matrix[1, 3] = -float(np.dot(true_up, eye))
    matrix[2, 3] = float(np.dot(forward, eye))
    return matrix


def ortho(width, height):
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = 2.0 / max(width, 1)
    matrix[1, 1] = 2.0 / max(height, 1)
    matrix[0, 3] = -1.0
    matrix[1, 3] = -1.0
    return matrix


def write_matrix(uniform, matrix):
    uniform.write(np.ascontiguousarray(matrix.T).tobytes())
