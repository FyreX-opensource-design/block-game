import math
import numpy as np

PIXEL = 1.0 / 16.0
SKIN_W = 64.0
SKIN_H = 32.0

SPIDER_PARTS = {
    "head": {
        "pivot": (0.0, 4.0, -3.0),
        "box": (-4.0, -4.0, -8.0, 4.0, 4.0, 0.0),
        "uv": {
            "up": (40, 0, 48, 8),
            "down": (48, 0, 56, 8),
            "east": (32, 8, 40, 16),
            "north": (40, 8, 48, 16),
            "west": (48, 8, 56, 16),
            "south": (56, 8, 64, 16),
        },
    },
    "thorax": {
        "pivot": (0.0, 5.0, 0.0),
        "box": (-3.0, -3.0, -3.0, 3.0, 3.0, 3.0),
        "uv": {
            "up": (6, 0, 12, 6),
            "down": (12, 0, 18, 6),
            "east": (0, 6, 6, 12),
            "north": (6, 6, 12, 12),
            "west": (12, 6, 18, 12),
            "south": (18, 6, 24, 12),
        },
    },
    "abdomen": {
        "pivot": (0.0, 5.0, 3.0),
        "box": (-5.0, -4.0, 0.0, 5.0, 4.0, 12.0),
        "uv": {
            "up": (12, 12, 22, 24),
            "down": (22, 12, 32, 24),
            "east": (0, 24, 12, 32),
            "north": (12, 24, 22, 32),
            "west": (22, 24, 34, 32),
            "south": (34, 24, 44, 32),
        },
    },
    "leg_r1": {
        "pivot": (-3.0, 5.0, -2.0),
        "box": (-12.0, -1.0, -1.0, 0.0, 1.0, 1.0),
        "uv": {
            "up": (48, 20, 60, 22),
            "down": (48, 22, 60, 24),
            "north": (48, 20, 60, 22),
            "south": (48, 22, 60, 24),
            "east": (48, 20, 50, 22),
            "west": (58, 20, 60, 22),
        },
    },
    "leg_l1": {
        "pivot": (3.0, 5.0, -2.0),
        "box": (0.0, -1.0, -1.0, 12.0, 1.0, 1.0),
        "uv": {
            "up": (48, 20, 60, 22),
            "down": (48, 22, 60, 24),
            "north": (48, 20, 60, 22),
            "south": (48, 22, 60, 24),
            "east": (48, 20, 50, 22),
            "west": (58, 20, 60, 22),
        },
    },
    "leg_r2": {
        "pivot": (-3.0, 5.0, -0.7),
        "box": (-12.0, -1.0, -1.0, 0.0, 1.0, 1.0),
        "uv": {
            "up": (48, 20, 60, 22),
            "down": (48, 22, 60, 24),
            "north": (48, 20, 60, 22),
            "south": (48, 22, 60, 24),
            "east": (48, 20, 50, 22),
            "west": (58, 20, 60, 22),
        },
    },
    "leg_l2": {
        "pivot": (3.0, 5.0, -0.7),
        "box": (0.0, -1.0, -1.0, 12.0, 1.0, 1.0),
        "uv": {
            "up": (48, 20, 60, 22),
            "down": (48, 22, 60, 24),
            "north": (48, 20, 60, 22),
            "south": (48, 22, 60, 24),
            "east": (48, 20, 50, 22),
            "west": (58, 20, 60, 22),
        },
    },
    "leg_r3": {
        "pivot": (-3.0, 5.0, 0.7),
        "box": (-12.0, -1.0, -1.0, 0.0, 1.0, 1.0),
        "uv": {
            "up": (48, 20, 60, 22),
            "down": (48, 22, 60, 24),
            "north": (48, 20, 60, 22),
            "south": (48, 22, 60, 24),
            "east": (48, 20, 50, 22),
            "west": (58, 20, 60, 22),
        },
    },
    "leg_l3": {
        "pivot": (3.0, 5.0, 0.7),
        "box": (0.0, -1.0, -1.0, 12.0, 1.0, 1.0),
        "uv": {
            "up": (48, 20, 60, 22),
            "down": (48, 22, 60, 24),
            "north": (48, 20, 60, 22),
            "south": (48, 22, 60, 24),
            "east": (48, 20, 50, 22),
            "west": (58, 20, 60, 22),
        },
    },
    "leg_r4": {
        "pivot": (-3.0, 5.0, 2.0),
        "box": (-12.0, -1.0, -1.0, 0.0, 1.0, 1.0),
        "uv": {
            "up": (48, 20, 60, 22),
            "down": (48, 22, 60, 24),
            "north": (48, 20, 60, 22),
            "south": (48, 22, 60, 24),
            "east": (48, 20, 50, 22),
            "west": (58, 20, 60, 22),
        },
    },
    "leg_l4": {
        "pivot": (3.0, 5.0, 2.0),
        "box": (0.0, -1.0, -1.0, 12.0, 1.0, 1.0),
        "uv": {
            "up": (48, 20, 60, 22),
            "down": (48, 22, 60, 24),
            "north": (48, 20, 60, 22),
            "south": (48, 22, 60, 24),
            "east": (48, 20, 50, 22),
            "west": (58, 20, 60, 22),
        },
    },
}

SPIDER_PART_ORDER = (
    "head",
    "thorax",
    "abdomen",
    "leg_r1",
    "leg_l1",
    "leg_r2",
    "leg_l2",
    "leg_r3",
    "leg_l3",
    "leg_r4",
    "leg_l4",
)


def _face_uv_spider(u0, v0, u1, v1):
    su0 = (u0 + 0.05) / SKIN_W
    su1 = (u1 - 0.05) / SKIN_W
    sv_top = 1.0 - (v0 + 0.05) / SKIN_H
    sv_bot = 1.0 - (v1 - 0.05) / SKIN_H
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
        uv = _face_uv_spider(u0, v0, u1, v1)
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


def build_spider_meshes():
    meshes = {}
    for name in SPIDER_PART_ORDER:
        part = SPIDER_PARTS[name]
        meshes[name] = build_box_mesh(part["box"], part["uv"])
    return meshes


def spider_pose(time=0.0, speed=0.0, yaw=0.0, pitch=0.0, climbing=False, on_ceiling=False, walk_time=None):
    """Calculate poses for spider parts: 8 scuttling legs and surface orientation."""
    if walk_time is not None:
        time = walk_time

    if on_ceiling:
        surface_roll = math.pi
        surface_pitch = 0.0
    elif climbing:
        surface_pitch = 1.45  # ~83 degrees pitched up against wall (head points UP, belly on wall)
        surface_roll = 0.0
    else:
        surface_pitch = 0.0
        surface_roll = 0.0

    # Leg scuttle swing
    if speed < 0.04 and not climbing:
        swing = 0.0
    else:
        # Smooth scuttle cycle
        scuttle_speed = max(speed, 1.2 if climbing else 0.0)
        swing = math.sin(time * 5.0) * min(0.32, max(0.12, scuttle_speed * 0.25))

    # Base resting droop (tilt down toward floor)
    r_droop = 0.38
    l_droop = -0.38

    # Alternating leg gait
    # Group A: R1, L2, R3, L4 (swing positive forward)
    # Group B: L1, R2, L3, R4 (swing negative backward)
    poses = {
        "head": (pitch * 0.5, 0.0, 0.0),
        "thorax": (0.0, 0.0, 0.0),
        "abdomen": (0.0, 0.0, 0.0),
        # Leg poses: (pitch, roll, yaw_offset)
        # pitch=0 (no longitudinal spin jitter!)
        # roll=droop + slight step lift
        # yaw_offset=spread + swing
        "leg_r1": (0.0, r_droop - max(0.0, swing) * 0.15, -0.75 + swing),
        "leg_l1": (0.0, l_droop + max(0.0, -swing) * 0.15, 0.75 - swing),
        "leg_r2": (0.0, r_droop - max(0.0, -swing) * 0.15, -0.25 - swing),
        "leg_l2": (0.0, l_droop + max(0.0, swing) * 0.15, 0.25 + swing),
        "leg_r3": (0.0, r_droop - max(0.0, swing) * 0.15, 0.25 + swing),
        "leg_l3": (0.0, l_droop + max(0.0, -swing) * 0.15, -0.25 - swing),
        "leg_r4": (0.0, r_droop - max(0.0, -swing) * 0.15, 0.75 - swing),
        "leg_l4": (0.0, l_droop + max(0.0, swing) * 0.15, -0.75 + swing),
        "__surface__": (surface_pitch, surface_roll),
    }
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


def _rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _translation(x, y, z):
    return np.array(
        [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def spider_part_matrix(pos, yaw, part_name, pitch=0.0, roll=0.0, yaw_offset=0.0, surface_pitch=0.0, surface_roll=0.0):
    pivot = SPIDER_PARTS[part_name]["pivot"]
    px, py, pz = pivot[0] * PIXEL, pivot[1] * PIXEL, pivot[2] * PIXEL

    # When on ceiling (surface_roll ~ pi), offset render position by spider height (0.8)
    # so the inverted feet touch the ceiling surface directly rather than hovering a block below it.
    render_y = pos[1] + (0.8 if abs(surface_roll) > 0.1 else 0.0)

    # When climbing a wall (surface_pitch ~ 1.45), offset render position toward the wall
    # along the facing direction so the climbing feet and belly cling flush against the wall face
    # rather than hovering away from it due to horizontal collision radius.
    wall_offset = 0.45 if surface_pitch > 0.5 else 0.0
    fx = -math.sin(yaw)
    fz = -math.cos(yaw)
    render_x = pos[0] + fx * wall_offset
    render_z = pos[2] + fz * wall_offset

    t_world = _translation(render_x, render_y, render_z)
    ry = _rotation_y(yaw)
    rx_surf = _rotation_x(surface_pitch)
    rz_surf = _rotation_z(surface_roll)
    base_matrix = t_world @ ry @ rx_surf @ rz_surf

    t_pivot = _translation(px, py, pz)
    ry_off = _rotation_y(yaw_offset)
    rz = _rotation_z(roll)
    rx = _rotation_x(pitch)
    return base_matrix @ t_pivot @ ry_off @ rz @ rx


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


class SpiderMeshes:
    def __init__(self):
        self.parts = build_spider_meshes()

    def posed_vertices(self, position, yaw, poses, surface_pitch=None, surface_roll=None):
        if surface_pitch is None or surface_roll is None:
            surf_p, surf_r = poses.get("__surface__", (0.0, 0.0))
            if surface_pitch is None:
                surface_pitch = surf_p
            if surface_roll is None:
                surface_roll = surf_r

        chunks = []
        index_chunks = []
        base = 0
        for name in SPIDER_PART_ORDER:
            vertices, indices = self.parts[name]
            pitch, roll, yaw_off = poses.get(name, (0.0, 0.0, 0.0))
            matrix = spider_part_matrix(
                position,
                yaw,
                name,
                pitch=pitch,
                roll=roll,
                yaw_offset=yaw_off,
                surface_pitch=surface_pitch,
                surface_roll=surface_roll,
            )
            transformed = transform_vertices(vertices, matrix)
            chunks.append(transformed)
            index_chunks.append(indices + base)
            base += transformed.shape[0] // 8
        if not chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint32)
        return np.concatenate(chunks), np.concatenate(index_chunks)
