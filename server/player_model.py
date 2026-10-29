import math

import numpy as np

PIXEL = 1.0 / 16.0
SKIN = 64.0

# Classic Steve boxes in pixels. Feet at y=0. Each part is local to its pivot.
PARTS = {
    "head": {
        "pivot": (0.0, 24.0, 0.0),
        "box": (-4.0, 0.0, -4.0, 4.0, 8.0, 4.0),
        "uv": {"east": (0, 8, 8, 16), "west": (16, 8, 24, 16), "up": (8, 0, 16, 8), "down": (16, 0, 24, 8), "south": (24, 8, 32, 16), "north": (8, 8, 16, 16)},
    },
    "hat": {
        "pivot": (0.0, 24.0, 0.0),
        "box": (-4.5, -0.5, -4.5, 4.5, 8.5, 4.5),
        "uv": {"east": (32, 8, 40, 16), "west": (48, 8, 56, 16), "up": (40, 0, 48, 8), "down": (48, 0, 56, 8), "south": (56, 8, 64, 16), "north": (40, 8, 48, 16)},
    },
    "body": {
        "pivot": (0.0, 18.0, 0.0),
        "box": (-4.0, -6.0, -2.0, 4.0, 6.0, 2.0),
        "uv": {"east": (16, 20, 20, 32), "west": (28, 20, 32, 32), "up": (20, 16, 28, 20), "down": (28, 16, 36, 20), "south": (32, 20, 40, 32), "north": (20, 20, 28, 32)},
    },
    "arm_r": {
        "pivot": (-6.0, 22.0, 0.0),
        "box": (-2.0, -10.0, -2.0, 2.0, 2.0, 2.0),
        "uv": {"east": (40, 20, 44, 32), "west": (48, 20, 52, 32), "up": (44, 16, 48, 20), "down": (48, 16, 52, 20), "south": (52, 20, 56, 32), "north": (44, 20, 48, 32)},
    },
    "arm_l": {
        "pivot": (6.0, 22.0, 0.0),
        "box": (-2.0, -10.0, -2.0, 2.0, 2.0, 2.0),
        "uv": {"east": (32, 52, 36, 64), "west": (40, 52, 44, 64), "up": (36, 48, 40, 52), "down": (40, 48, 44, 52), "south": (44, 52, 48, 64), "north": (36, 52, 40, 64)},
    },
    "leg_r": {
        "pivot": (-2.0, 12.0, 0.0),
        "box": (-2.0, -12.0, -2.0, 2.0, 0.0, 2.0),
        "uv": {"east": (0, 20, 4, 32), "west": (8, 20, 12, 32), "up": (4, 16, 8, 20), "down": (8, 16, 12, 20), "south": (12, 20, 16, 32), "north": (4, 20, 8, 32)},
    },
    "leg_l": {
        "pivot": (2.0, 12.0, 0.0),
        "box": (-2.0, -12.0, -2.0, 2.0, 0.0, 2.0),
        "uv": {"east": (16, 52, 20, 64), "west": (24, 52, 28, 64), "up": (20, 48, 24, 52), "down": (24, 48, 28, 52), "south": (28, 52, 32, 64), "north": (20, 52, 24, 64)},
    },
}

PART_ORDER = ("body", "head", "hat", "arm_r", "arm_l", "leg_r", "leg_l")


def part_names():
    return PART_ORDER


def limb_pose(time, speed, yaw, pitch=0.0):
    if speed < 0.1:
        swing = 0.0
    else:
        swing = math.sin(time * 10.0) * min(1.0, speed * 1.6) * 0.7
    head_pitch = pitch
    return {
        "head": (head_pitch, 0.0, 0.0),
        "hat": (head_pitch, 0.0, 0.0),
        "body": (0.0, 0.0, 0.0),
        "arm_r": (swing, 0.0, 0.0),
        "arm_l": (-swing, 0.0, 0.0),
        "leg_r": (-swing, 0.0, 0.0),
        "leg_l": (swing, 0.0, 0.0),
    }


def build_part_mesh(name):
    part = PARTS[name]
    x0, y0, z0, x1, y1, z1 = part["box"]
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
        u0, v0, u1, v1 = part["uv"][face]
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


def build_player_meshes():
    return {name: build_part_mesh(name) for name in PART_ORDER}


def build_fp_arm_mesh():
    """FP forearm: keep the hand cap, omit the shoulder cap (open end stays off-screen)."""
    part = PARTS["arm_r"]
    x0, y0, z0, x1, y1, z1 = part["box"]
    # Extend past the pivot so the open proximal cut is outside the frustum.
    y1 = 8.0
    faces = (
        ("east", (x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (1, 0, 0)),
        ("west", (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (-1, 0, 0)),
        ("down", (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (0, -1, 0)),
        ("south", (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0, 0, 1)),
        ("north", (x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (0, 0, -1)),
    )
    vertices = []
    indices = []
    for face, a, b, c, d, normal in faces:
        u0, v0, u1, v1 = part["uv"][face]
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


def part_pivot(name):
    px, py, pz = PARTS[name]["pivot"]
    return (px * PIXEL, py * PIXEL, pz * PIXEL)


def hand_anchor():
    """Right-hand tip in arm_r local space (holds the item)."""
    return (0.0, -10.0 * PIXEL, -1.0 * PIXEL)


def _face_uv(u0, v0, u1, v1):
    # Skin image row 0 is the visual top. Texture is uploaded flipud, so GL v=0 is the visual bottom.
    su0 = (u0 + 0.5) / SKIN
    su1 = (u1 - 0.5) / SKIN
    sv_top = 1.0 - (v0 + 0.5) / SKIN
    sv_bot = 1.0 - (v1 - 0.5) / SKIN
    return (
        (su0, sv_bot),
        (su1, sv_bot),
        (su1, sv_top),
        (su0, sv_top),
    )


def transform_vertices(vertices, matrix):
    out = np.array(vertices, dtype=np.float32, copy=True)
    count = out.shape[0] // 8
    positions = out.reshape(count, 8)[:, 0:3]
    normals = out.reshape(count, 8)[:, 3:6]
    ones = np.ones((count, 1), dtype=np.float32)
    transformed = (matrix @ np.concatenate([positions, ones], axis=1).T).T[:, :3]
    rotation = matrix[:3, :3]
    transformed_n = (rotation @ normals.T).T
    lengths = np.linalg.norm(transformed_n, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-6)
    out.reshape(count, 8)[:, 0:3] = transformed
    out.reshape(count, 8)[:, 3:6] = transformed_n / lengths
    return out


def model_matrix(position, yaw, part, pitch=0.0, roll=0.0):
    pivot = part_pivot(part)
    # Local rotation around pivot, then place pivot in body space, then yaw body around feet.
    local = _rotation_xyz(pitch, roll, 0.0)
    body = np.eye(4, dtype=np.float32)
    body[:3, :3] = _rotation_y(yaw)[:3, :3]
    body[0, 3] = position[0]
    body[1, 3] = position[1]
    body[2, 3] = position[2]
    pivot_mat = np.eye(4, dtype=np.float32)
    pivot_mat[0, 3] = pivot[0]
    pivot_mat[1, 3] = pivot[1]
    pivot_mat[2, 3] = pivot[2]
    return body @ pivot_mat @ local


def _rotation_y(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = c
    matrix[0, 2] = s
    matrix[2, 0] = -s
    matrix[2, 2] = c
    return matrix


def _rotation_xyz(pitch, roll, yaw):
    # Pitch around X (arm swing), roll around Z, yaw around Y.
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cz, sz = math.cos(roll), math.sin(roll)
    matrix = np.eye(4, dtype=np.float32)
    # Rz * Ry * Rx
    matrix[0, 0] = cz * cy
    matrix[0, 1] = cz * sy * sx - sz * cx
    matrix[0, 2] = cz * sy * cx + sz * sx
    matrix[1, 0] = sz * cy
    matrix[1, 1] = sz * sy * sx + cz * cx
    matrix[1, 2] = sz * sy * cx - cz * sx
    matrix[2, 0] = -sy
    matrix[2, 1] = cy * sx
    matrix[2, 2] = cy * cx
    return matrix


def held_item_matrix(position, yaw, arm_pitch, item_scale=None, flat=False):
    if item_scale is None:
        item_scale = 0.55 if flat else 0.75
    arm = model_matrix(position, yaw, "arm_r", pitch=arm_pitch)
    tip = np.array([*hand_anchor(), 1.0], dtype=np.float32)
    world = arm @ tip
    matrix = np.eye(4, dtype=np.float32)
    if flat:
        # Orient flat tool so shaft points forward (-Z) and blade stands upright.
        # R_base rotates from texture XY plane (shaft at -45°) to -Z (forward).
        # Tilted 15° upward so the tool points forward in front of the hand.
        r_flat = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.5, math.sqrt(3) / 2.0, 0.0],
                [math.sqrt(3) / 2.0, -0.5, 0.0],
            ],
            dtype=np.float32,
        )
        matrix[:3, :3] = (arm[:3, :3] @ r_flat) * item_scale
    else:
        matrix[:3, :3] = arm[:3, :3] * item_scale
    matrix[0, 3] = float(world[0])
    matrix[1, 3] = float(world[1])
    matrix[2, 3] = float(world[2])
    return matrix


def _camera_basis(yaw, pitch):
    forward = _dir(yaw, pitch)
    right = (math.cos(yaw), 0.0, -math.sin(yaw))
    up = _cross(right, forward)
    return forward, right, up


def _view_to_world(eye, yaw, pitch, view_x, view_y, view_z):
    """view: +X right, +Y up, +Z forward (in front of the camera)."""
    forward, right, up = _camera_basis(yaw, pitch)
    return (
        eye[0] + right[0] * view_x + up[0] * view_y + forward[0] * view_z,
        eye[1] + right[1] * view_x + up[1] * view_y + forward[1] * view_z,
        eye[2] + right[2] * view_x + up[2] * view_y + forward[2] * view_z,
    )


def _fp_anchors(arm_pitch=0.0, arm_roll=0.0):
    bob = arm_pitch * 0.04
    # Minecraft-ish: shoulder below the bottom edge, fist in the lower-right.
    shoulder_v = (0.55 + arm_roll * 0.03, -1.05 + bob, 0.38)
    tip_v = (0.34, -0.28 + bob, 0.62)
    return shoulder_v, tip_v


def first_person_arm_matrix(eye, yaw, pitch, arm_pitch=0.0, arm_roll=0.0):
    """Viewmodel arm: shoulder off-screen bottom-right, hand tip in the lower-right."""
    shoulder_v, tip_v = _fp_anchors(arm_pitch, arm_roll)
    shoulder = _view_to_world(eye, yaw, pitch, *shoulder_v)
    tip = _view_to_world(eye, yaw, pitch, *tip_v)
    hy = hand_anchor()[1]
    along = np.array(tip, dtype=np.float32) - np.array(shoulder, dtype=np.float32)
    length = float(np.linalg.norm(along))
    if length < 1e-5:
        along = np.array(_dir(yaw, pitch), dtype=np.float32)
        length = 1.0
    y_axis = -along / length
    forward = np.array(_dir(yaw, pitch), dtype=np.float32)
    # Prefer camera-right as the arm's "thickness" side so the sleeve reads as a limb.
    right = np.array((math.cos(yaw), 0.0, -math.sin(yaw)), dtype=np.float32)
    x_axis = right - y_axis * float(np.dot(right, y_axis))
    x_len = float(np.linalg.norm(x_axis))
    if x_len < 1e-5:
        x_axis = np.cross(forward, y_axis)
        x_len = float(np.linalg.norm(x_axis)) or 1.0
    x_axis /= x_len
    z_axis = np.cross(x_axis, y_axis)
    # Stretch length only — uniform scale made a fat floating plank.
    scale_len = length / max(abs(hy), 1e-5)
    scale_thick = 0.95
    matrix = np.eye(4, dtype=np.float32)
    matrix[0:3, 0] = x_axis * scale_thick
    matrix[0:3, 1] = y_axis * scale_len
    matrix[0:3, 2] = z_axis * scale_thick
    matrix[0, 3] = shoulder[0]
    matrix[1, 3] = shoulder[1]
    matrix[2, 3] = shoulder[2]
    return matrix


def first_person_held_matrix(eye, yaw, pitch, arm_pitch=0.0, arm_roll=0.0, item_scale=0.35, flat=False):
    """Place the held item in camera space (arm is hidden while holding)."""
    del arm_pitch, arm_roll  # bob reserved for a later swing anim
    eye = np.array(eye, dtype=np.float32)
    forward, right, up = _camera_basis(yaw, pitch)
    forward = np.array(forward, dtype=np.float32)
    right = np.array(right, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    if flat:
        # Minecraft FP: origin in the BR screen corner; item extends steeply upward (~72-75 deg)
        # with face turned inward ~38 deg towards center to show 3D extrusion thickness.
        hold = _view_to_world(eye, yaw, pitch, 0.96, -0.74, 0.84)
        base = np.column_stack((right, up, -forward))
        ry = _rotation_y(math.radians(-38))[:3, :3]
        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(math.radians(-14)), -math.sin(math.radians(-14))],
                [0.0, math.sin(math.radians(-14)), math.cos(math.radians(-14))],
            ],
            dtype=np.float32,
        )
        rz = np.array(
            [
                [math.cos(math.radians(-15)), -math.sin(math.radians(-15)), 0.0],
                [math.sin(math.radians(-15)), math.cos(math.radians(-15)), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        mc = ry @ rx @ rz
        basis = base @ mc
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = basis * item_scale
        matrix[0, 3] = float(hold[0])
        matrix[1, 3] = float(hold[1])
        matrix[2, 3] = float(hold[2])
        return matrix

    # Half-block: same BR corner bias.
    hold = _view_to_world(eye, yaw, pitch, 0.95, -0.68, 0.95)
    base = np.column_stack((right, up, -forward))
    tilt = _rotation_xyz(math.radians(14), math.radians(24), math.radians(-52))[:3, :3]
    basis = base @ tilt
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = basis * item_scale
    matrix[0, 3] = float(hold[0])
    matrix[1, 3] = float(hold[1])
    matrix[2, 3] = float(hold[2])
    return matrix


def _uniform_scale(value):
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = value
    matrix[1, 1] = value
    matrix[2, 2] = value
    return matrix


def _dir(yaw, pitch):
    cp = math.cos(pitch)
    return (-math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
