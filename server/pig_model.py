import math
import numpy as np

PIXEL = 1.0 / 16.0
SKIN_W = 64.0
SKIN_H = 32.0

# Pig boxes defined in pixels (16 pixels = 1.0 block).
# Pivot locations in world space (relative to bottom center of feet at y=0).
PIG_PARTS = {
    "head": {
        "pivot": (0.0, 12.0, -6.0),
        # Head box: 8 wide (-4 to 4), 8 high (-4 to 4), 8 deep (-8 to 0 relative to pivot)
        "box": (-4.0, -4.0, -8.0, 4.0, 4.0, 0.0),
        "uv": {
            "east": (0, 8, 8, 16),
            "west": (16, 8, 24, 16),
            "up": (8, 0, 16, 8),
            "down": (16, 0, 24, 8),
            "south": (24, 8, 32, 16),
            "north": (8, 8, 16, 16),  # face with eyes
        },
    },
    "snout": {
        "pivot": (0.0, 12.0, -6.0),
        # Snout box: 4 wide (-2 to 2), 3 high (-3 to 0), 1 deep (-9 to -8)
        "box": (-2.0, -3.0, -9.0, 2.0, 0.0, -8.0),
        "uv": {
            "east": (16, 17, 17, 20),
            "west": (21, 17, 22, 20),
            "up": (17, 16, 21, 17),
            "down": (21, 16, 25, 17),
            "south": (22, 17, 26, 20),
            "north": (17, 17, 21, 20),
        },
    },
    "body": {
        "pivot": (0.0, 10.0, 1.0),
        # Body box: 10 wide (-5 to 5), 8 high (-4 to 4), 16 long (-8 to 8)
        "box": (-5.0, -4.0, -8.0, 5.0, 4.0, 8.0),
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
        "pivot": (3.0, 6.0, -5.0),
        "box": (-2.0, -6.0, -2.0, 2.0, 0.0, 2.0),
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
        "pivot": (-3.0, 6.0, -5.0),
        "box": (-2.0, -6.0, -2.0, 2.0, 0.0, 2.0),
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
        "pivot": (3.0, 6.0, 7.0),
        "box": (-2.0, -6.0, -2.0, 2.0, 0.0, 2.0),
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
        "pivot": (-3.0, 6.0, 7.0),
        "box": (-2.0, -6.0, -2.0, 2.0, 0.0, 2.0),
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

PIG_PART_ORDER = (
    "body",
    "head",
    "snout",
    "leg_front_left",
    "leg_front_right",
    "leg_back_left",
    "leg_back_right",
)


def _face_uv_pig(u0, v0, u1, v1):
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
        uv = _face_uv_pig(u0, v0, u1, v1)
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


def build_pig_meshes():
    meshes = {}
    for name in PIG_PART_ORDER:
        part = PIG_PARTS[name]
        meshes[name] = build_box_mesh(part["box"], part["uv"])
    return meshes


def pig_pose(time, speed, yaw, pitch=0.0):
    if speed < 0.05:
        swing = 0.0
    else:
        swing = math.sin(time * 10.0) * min(1.0, speed * 1.5) * 0.7
    return {
        "head": (pitch, 0.0, 0.0),
        "snout": (pitch, 0.0, 0.0),
        "body": (0.0, 0.0, 0.0),
        "leg_front_left": (swing, 0.0, 0.0),
        "leg_front_right": (-swing, 0.0, 0.0),
        "leg_back_left": (-swing, 0.0, 0.0),
        "leg_back_right": (swing, 0.0, 0.0),
    }


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


def pig_part_matrix(pos, yaw, part_name, pitch=0.0, roll=0.0):
    pivot = PIG_PARTS[part_name]["pivot"]
    px, py, pz = pivot[0] * PIXEL, pivot[1] * PIXEL, pivot[2] * PIXEL
    t_world = _translation(pos[0], pos[1], pos[2])
    ry = _rotation_y(yaw)
    t_pivot = _translation(px, py, pz)
    rx = _rotation_x(pitch)
    return t_world @ ry @ t_pivot @ rx


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


class PigMeshes:
    def __init__(self):
        self.parts = build_pig_meshes()

    def posed_vertices(self, position, yaw, poses):
        chunks = []
        index_chunks = []
        base = 0
        for name in PIG_PART_ORDER:
            vertices, indices = self.parts[name]
            pitch, _roll, _yaw = poses.get(name, (0.0, 0.0, 0.0))
            matrix = pig_part_matrix(position, yaw, name, pitch=pitch)
            transformed = transform_vertices(vertices, matrix)
            chunks.append(transformed)
            index_chunks.append(indices + base)
            base += transformed.shape[0] // 8
        if not chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint32)
        return np.concatenate(chunks), np.concatenate(index_chunks)

