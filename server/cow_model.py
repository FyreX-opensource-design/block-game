"""Box cow: white body, horns, snout, and an udder. 16 pixels is one block."""

import math

import numpy as np

PIXEL = 1.0 / 16.0
SKIN_W = 64.0
SKIN_H = 64.0

COW_PARTS = {
    "body": {
        "pivot": (0.0, 14.0, 1.0),
        "box": (-6.0, -5.0, -9.0, 6.0, 5.0, 9.0),
        "uv": {
            "up": (0, 0, 18, 12),
            "down": (18, 0, 36, 12),
            "north": (0, 12, 12, 22),
            "south": (12, 12, 24, 22),
            "east": (24, 12, 42, 22),
            "west": (42, 12, 60, 22),
        },
    },
    "udder": {
        "pivot": (0.0, 9.0, 4.0),
        "box": (-3.0, -4.0, -3.0, 3.0, 0.0, 3.0),
        "uv": {
            "north": (48, 34, 54, 38),
            "south": (54, 34, 60, 38),
            "east": (48, 38, 54, 42),
            "west": (54, 38, 60, 42),
            "up": (48, 42, 54, 48),
            "down": (54, 42, 60, 48),
        },
    },
    "tail": {
        "pivot": (0.0, 17.0, 10.0),
        "box": (-1.0, -7.0, -1.0, 1.0, 0.0, 1.0),
        "uv": {
            "north": (0, 40, 2, 47),
            "south": (2, 40, 4, 47),
            "east": (4, 40, 6, 47),
            "west": (6, 40, 8, 47),
            "up": (0, 47, 2, 49),
            "down": (2, 47, 4, 49),
        },
    },
    "head": {
        "pivot": (0.0, 18.0, -8.0),
        "box": (-4.0, -4.0, -8.0, 4.0, 4.0, 0.0),
        "uv": {
            "north": (0, 24, 8, 32),
            "south": (8, 24, 16, 32),
            "east": (16, 24, 24, 32),
            "west": (24, 24, 32, 32),
            "up": (32, 24, 40, 32),
            "down": (40, 24, 48, 32),
        },
    },
    "snout": {
        "pivot": (0.0, 18.0, -8.0),
        "box": (-3.0, -4.0, -11.0, 3.0, 0.0, -8.0),
        "uv": {
            "north": (0, 34, 6, 38),
            "south": (6, 34, 12, 38),
            "east": (12, 34, 15, 38),
            "west": (15, 34, 18, 38),
            "up": (18, 34, 24, 37),
            "down": (24, 34, 30, 37),
        },
    },
    "horn_left": {
        "pivot": (0.0, 18.0, -8.0),
        "box": (2.0, 3.0, -5.0, 4.0, 8.0, -3.0),
        "uv": {
            "north": (32, 34, 34, 39),
            "south": (34, 34, 36, 39),
            "east": (36, 34, 38, 39),
            "west": (38, 34, 40, 39),
            "up": (32, 39, 34, 41),
            "down": (34, 39, 36, 41),
        },
    },
    "horn_right": {
        "pivot": (0.0, 18.0, -8.0),
        "box": (-4.0, 3.0, -5.0, -2.0, 8.0, -3.0),
        "uv": {
            "north": (40, 34, 42, 39),
            "south": (42, 34, 44, 39),
            "east": (44, 34, 46, 39),
            "west": (46, 34, 48, 39),
            "up": (40, 39, 42, 41),
            "down": (42, 39, 44, 41),
        },
    },
    "ear_left": {
        "pivot": (0.0, 18.0, -8.0),
        "box": (4.0, 1.0, -5.0, 7.0, 3.0, -3.0),
        "uv": {
            "north": (48, 24, 51, 26),
            "south": (51, 24, 54, 26),
            "east": (54, 24, 56, 26),
            "west": (56, 24, 58, 26),
            "up": (48, 26, 51, 28),
            "down": (51, 26, 54, 28),
        },
    },
    "ear_right": {
        "pivot": (0.0, 18.0, -8.0),
        "box": (-7.0, 1.0, -5.0, -4.0, 3.0, -3.0),
        "uv": {
            "north": (48, 28, 51, 30),
            "south": (51, 28, 54, 30),
            "east": (54, 28, 56, 30),
            "west": (56, 28, 58, 30),
            "up": (48, 30, 51, 32),
            "down": (51, 30, 54, 32),
        },
    },
    "leg_front_left": {
        "pivot": (4.0, 9.0, -6.0),
        "box": (-1.5, -9.0, -1.5, 1.5, 0.0, 1.5),
        "uv": {
            "north": (8, 40, 11, 49),
            "south": (11, 40, 14, 49),
            "east": (14, 40, 17, 49),
            "west": (17, 40, 20, 49),
            "up": (8, 49, 11, 52),
            "down": (11, 49, 14, 52),
        },
    },
    "leg_front_right": {
        "pivot": (-4.0, 9.0, -6.0),
        "box": (-1.5, -9.0, -1.5, 1.5, 0.0, 1.5),
        "uv": {
            "north": (8, 40, 11, 49),
            "south": (11, 40, 14, 49),
            "east": (14, 40, 17, 49),
            "west": (17, 40, 20, 49),
            "up": (8, 49, 11, 52),
            "down": (11, 49, 14, 52),
        },
    },
    "leg_back_left": {
        "pivot": (4.0, 9.0, 7.0),
        "box": (-1.5, -9.0, -1.5, 1.5, 0.0, 1.5),
        "uv": {
            "north": (8, 40, 11, 49),
            "south": (11, 40, 14, 49),
            "east": (14, 40, 17, 49),
            "west": (17, 40, 20, 49),
            "up": (8, 49, 11, 52),
            "down": (11, 49, 14, 52),
        },
    },
    "leg_back_right": {
        "pivot": (-4.0, 9.0, 7.0),
        "box": (-1.5, -9.0, -1.5, 1.5, 0.0, 1.5),
        "uv": {
            "north": (8, 40, 11, 49),
            "south": (11, 40, 14, 49),
            "east": (14, 40, 17, 49),
            "west": (17, 40, 20, 49),
            "up": (8, 49, 11, 52),
            "down": (11, 49, 14, 52),
        },
    },
}

COW_PART_ORDER = (
    "body",
    "udder",
    "tail",
    "head",
    "snout",
    "horn_left",
    "horn_right",
    "ear_left",
    "ear_right",
    "leg_front_left",
    "leg_front_right",
    "leg_back_left",
    "leg_back_right",
)

HEAD_PARTS = ("head", "snout", "horn_left", "horn_right", "ear_left", "ear_right")


def _face_uv(u0, v0, u1, v1):
    su0 = (u0 + 0.5) / SKIN_W
    su1 = (u1 - 0.5) / SKIN_W
    sv_top = 1.0 - (v0 + 0.5) / SKIN_H
    sv_bot = 1.0 - (v1 - 0.5) / SKIN_H
    return (
        (su0, sv_bot),
        (su1, sv_bot),
        (su1, sv_top),
        (su0, sv_top),
    )


def build_box_mesh(box, uv_dict):
    x0, y0, z0, x1, y1, z1 = box
    faces = (
        ("east", (x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (1, 0, 0)),
        ("west", (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (-1, 0, 0)),
        ("up", (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0), (0, 1, 0)),
        ("down", (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (0, -1, 0)),
        ("south", (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0, 0, 1)),
        ("north", (x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (0, 0, -1)),
    )
    vertices = []
    indices = []
    for face, a, b, c, d, normal in faces:
        if face not in uv_dict:
            continue
        u0, v0, u1, v1 = uv_dict[face]
        uv = _face_uv(u0, v0, u1, v1)
        base = len(vertices) // 8
        for corner, (u, v) in zip((a, b, c, d), uv):
            vertices.extend(
                (
                    corner[0] * PIXEL,
                    corner[1] * PIXEL,
                    corner[2] * PIXEL,
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                    u,
                    v,
                )
            )
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.uint32)


def build_cow_meshes():
    return {name: build_box_mesh(COW_PARTS[name]["box"], COW_PARTS[name]["uv"]) for name in COW_PART_ORDER}


def cow_pose(time, speed, yaw, pitch=0.0):
    if speed < 0.05:
        swing = 0.0
    else:
        swing = math.sin(time * 9.0) * min(1.0, speed * 1.4) * 0.65
    poses = {name: (pitch, 0.0, 0.0) for name in HEAD_PARTS}
    poses["body"] = (0.0, 0.0, 0.0)
    poses["udder"] = (0.0, 0.0, 0.0)
    poses["tail"] = (0.35 + swing * 0.15, 0.0, 0.0)
    poses["leg_front_left"] = (swing, 0.0, 0.0)
    poses["leg_front_right"] = (-swing, 0.0, 0.0)
    poses["leg_back_left"] = (-swing, 0.0, 0.0)
    poses["leg_back_right"] = (swing, 0.0, 0.0)
    return poses


def _rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _translation(x, y, z):
    return np.array(
        [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def cow_part_matrix(pos, yaw, part_name, pitch=0.0):
    pivot = COW_PARTS[part_name]["pivot"]
    px, py, pz = pivot[0] * PIXEL, pivot[1] * PIXEL, pivot[2] * PIXEL
    return _translation(pos[0], pos[1], pos[2]) @ _rotation_y(yaw) @ _translation(px, py, pz) @ _rotation_x(pitch)


def transform_vertices(vertices, matrix):
    if len(vertices) == 0:
        return vertices
    v = vertices.reshape(-1, 8).copy()
    positions = v[:, 0:3]
    normals = v[:, 3:6]
    ones = np.ones((len(positions), 1), dtype=np.float32)
    pos_h = np.hstack([positions, ones])
    pos_trans = (matrix @ pos_h.T).T
    v[:, 0:3] = pos_trans[:, 0:3]
    rot = matrix[0:3, 0:3]
    norm_trans = (rot @ normals.T).T
    lengths = np.linalg.norm(norm_trans, axis=1, keepdims=True)
    lengths[lengths < 1e-6] = 1.0
    v[:, 3:6] = norm_trans / lengths
    return v.reshape(-1)


class CowMeshes:
    def __init__(self):
        self.parts = build_cow_meshes()

    def posed_vertices(self, position, yaw, poses):
        chunks = []
        index_chunks = []
        base = 0
        for name in COW_PART_ORDER:
            vertices, indices = self.parts[name]
            pitch, _roll, _yaw = poses.get(name, (0.0, 0.0, 0.0))
            matrix = cow_part_matrix(position, yaw, name, pitch=pitch)
            transformed = transform_vertices(vertices, matrix)
            chunks.append(transformed)
            index_chunks.append(indices + base)
            base += transformed.shape[0] // 8
        if not chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint32)
        return np.concatenate(chunks), np.concatenate(index_chunks)
