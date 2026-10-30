import math
import numpy as np

from server.pig_model import (
    PIXEL,
    build_box_mesh,
    transform_vertices,
    _rotation_x,
    _rotation_y,
    _translation,
)

# Sheep boxes defined in pixels (16 pixels = 1.0 block).
SHEEP_PARTS = {
    "head": {
        "pivot": (0.0, 14.0, -6.0),
        "box": (-3.0, -3.0, -6.0, 3.0, 3.0, 0.0),
        "uv": {
            "east": (0, 8, 8, 16),
            "west": (16, 8, 24, 16),
            "up": (8, 0, 16, 8),
            "down": (16, 0, 24, 8),
            "south": (24, 8, 32, 16),
            "north": (8, 8, 16, 16),
        },
    },
    "head_wool": {
        "pivot": (0.0, 14.0, -6.0),
        "box": (-3.5, -2.0, -4.5, 3.5, 4.0, 0.5),
        "uv": {
            "east": (28, 16, 36, 24),
            "west": (28, 16, 36, 24),
            "up": (36, 16, 44, 24),
            "down": (36, 16, 44, 24),
            "south": (36, 8, 44, 16),
            "north": (36, 8, 44, 16),
        },
    },
    "body": {
        "pivot": (0.0, 11.0, 1.0),
        "box": (-4.0, -3.0, -7.0, 4.0, 3.0, 7.0),
        "uv": {
            "up": (36, 16, 46, 32),
            "down": (46, 16, 56, 32),
            "east": (28, 16, 36, 32),
            "west": (46, 16, 54, 32),
            "north": (46, 8, 56, 16),
            "south": (36, 8, 46, 16),
        },
    },
    "wool_body": {
        "pivot": (0.0, 11.0, 1.0),
        "box": (-5.5, -4.5, -8.0, 5.5, 4.5, 8.0),
        "uv": {
            "up": (36, 16, 46, 32),
            "down": (46, 16, 56, 32),
            "east": (28, 16, 36, 32),
            "west": (46, 16, 54, 32),
            "north": (46, 8, 56, 16),
            "south": (36, 8, 46, 16),
        },
    },
    "leg_front_left": {
        "pivot": (3.0, 8.0, -4.0),
        "box": (-1.5, -8.0, -1.5, 1.5, 0.0, 1.5),
        "uv": {
            "east": (0, 20, 4, 26),
            "west": (8, 20, 12, 26),
            "up": (4, 16, 8, 20),
            "down": (8, 16, 12, 20),
            "north": (4, 20, 8, 26),
            "south": (12, 20, 16, 26),
        },
    },
    "leg_front_right": {
        "pivot": (-3.0, 8.0, -4.0),
        "box": (-1.5, -8.0, -1.5, 1.5, 0.0, 1.5),
        "uv": {
            "east": (0, 20, 4, 26),
            "west": (8, 20, 12, 26),
            "up": (4, 16, 8, 20),
            "down": (8, 16, 12, 20),
            "north": (4, 20, 8, 26),
            "south": (12, 20, 16, 26),
        },
    },
    "leg_back_left": {
        "pivot": (3.0, 8.0, 6.0),
        "box": (-1.5, -8.0, -1.5, 1.5, 0.0, 1.5),
        "uv": {
            "east": (0, 20, 4, 26),
            "west": (8, 20, 12, 26),
            "up": (4, 16, 8, 20),
            "down": (8, 16, 12, 20),
            "north": (4, 20, 8, 26),
            "south": (12, 20, 16, 26),
        },
    },
    "leg_back_right": {
        "pivot": (-3.0, 8.0, 6.0),
        "box": (-1.5, -8.0, -1.5, 1.5, 0.0, 1.5),
        "uv": {
            "east": (0, 20, 4, 26),
            "west": (8, 20, 12, 26),
            "up": (4, 16, 8, 20),
            "down": (8, 16, 12, 20),
            "north": (4, 20, 8, 26),
            "south": (12, 20, 16, 26),
        },
    },
}

SHEEP_PART_ORDER_WOOL = (
    "body",
    "wool_body",
    "head",
    "head_wool",
    "leg_front_left",
    "leg_front_right",
    "leg_back_left",
    "leg_back_right",
)

SHEEP_PART_ORDER_SHEARED = (
    "body",
    "head",
    "leg_front_left",
    "leg_front_right",
    "leg_back_left",
    "leg_back_right",
)


def build_sheep_meshes():
    meshes = {}
    for name, part in SHEEP_PARTS.items():
        meshes[name] = build_box_mesh(part["box"], part["uv"])
    return meshes


def sheep_pose(time, speed, yaw, pitch=0.0):
    if speed < 0.05:
        swing = 0.0
    else:
        swing = math.sin(time * 10.0) * min(1.0, speed * 1.5) * 0.7
    return {
        "head": (pitch, 0.0, 0.0),
        "head_wool": (pitch, 0.0, 0.0),
        "body": (0.0, 0.0, 0.0),
        "wool_body": (0.0, 0.0, 0.0),
        "leg_front_left": (swing, 0.0, 0.0),
        "leg_front_right": (-swing, 0.0, 0.0),
        "leg_back_left": (-swing, 0.0, 0.0),
        "leg_back_right": (swing, 0.0, 0.0),
    }


def sheep_part_matrix(pos, yaw, part_name, pitch=0.0):
    pivot = SHEEP_PARTS[part_name]["pivot"]
    px, py, pz = pivot[0] * PIXEL, pivot[1] * PIXEL, pivot[2] * PIXEL
    if part_name in ("head", "head_wool") and pitch < -0.2:
        # Lower head pivot toward the ground when grazing
        py += pitch * 6.0 * PIXEL
    t_world = _translation(pos[0], pos[1], pos[2])
    ry = _rotation_y(yaw)
    t_pivot = _translation(px, py, pz)
    rx = _rotation_x(pitch)
    return t_world @ ry @ t_pivot @ rx


class SheepMeshes:
    def __init__(self):
        self.parts = build_sheep_meshes()

    def posed_vertices(self, position, yaw, poses, sheared=False):
        chunks = []
        index_chunks = []
        base = 0
        order = SHEEP_PART_ORDER_SHEARED if sheared else SHEEP_PART_ORDER_WOOL
        for name in order:
            vertices, indices = self.parts[name]
            pitch, _roll, _yaw = poses.get(name, (0.0, 0.0, 0.0))
            matrix = sheep_part_matrix(position, yaw, name, pitch=pitch)
            transformed = transform_vertices(vertices, matrix)
            chunks.append(transformed)
            index_chunks.append(indices + base)
            base += transformed.shape[0] // 8
        if not chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint32)
        return np.concatenate(chunks), np.concatenate(index_chunks)

