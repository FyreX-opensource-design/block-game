"""Bone skeleton: a skull, an open ribcage, and thin limbs. 16 pixels is one block.

Joint pivots match the player model so the bow stays in the right hand.
Texture tiles, painted by assets/textures/entity/skeleton.png:
  (0, 0)-(4, 4) bone, (4, 0)-(8, 4) mid, (0, 4)-(4, 8) shade, (4, 4)-(8, 8) joint,
  (8, 8)-(16, 16) skull face, (8, 18)-(14, 20) teeth.
"""

import numpy as np

from server.player_model import model_matrix, transform_vertices

PIXEL = 1.0 / 16.0
SKIN = 64.0

_BONE = (0, 0, 4, 4)
_MID = (4, 0, 8, 4)
_SHADE = (0, 4, 4, 8)
_JOINT = (4, 4, 8, 8)
_FACE = (8, 8, 16, 16)
_TEETH = (8, 18, 14, 20)

PART_ORDER = ("body", "head", "arm_r", "arm_l", "leg_r", "leg_l")


def _faces(rect, **overrides):
    uv = {name: rect for name in ("east", "west", "up", "down", "south", "north")}
    uv.update(overrides)
    return uv


def _face_uv(u0, v0, u1, v1):
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


def _box_mesh(box, uv_dict):
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
    return vertices, indices


def _combine(boxes):
    vertices = []
    indices = []
    base = 0
    for box, uv_dict in boxes:
        part_vertices, part_indices = _box_mesh(box, uv_dict)
        vertices.extend(part_vertices)
        indices.extend(index + base for index in part_indices)
        base += len(part_vertices) // 8
    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.uint32)


def _ribs():
    boxes = []
    for y in (-3.4, -1.0, 1.4, 3.6):
        top = y + 1.05
        boxes.append(((-3.5, y, 1.15, 3.5, top, 2.15), _faces(_MID)))
        boxes.append(((-3.5, y, -2.15, 3.5, top, -1.15), _faces(_SHADE)))
        boxes.append(((-3.85, y, -1.3, -2.65, top, 1.3), _faces(_MID)))
        boxes.append(((2.65, y, -1.3, 3.85, top, 1.3), _faces(_MID)))
    return boxes


def _body_boxes():
    return [
        ((-0.75, -6.0, -0.75, 0.75, 6.0, 0.75), _faces(_BONE)),
        ((-3.3, -6.3, -1.7, 3.3, -4.1, 1.7), _faces(_SHADE)),
        ((-5.6, 4.5, -0.85, 5.6, 5.9, 0.85), _faces(_BONE)),
        *_ribs(),
    ]


def _arm_boxes():
    return [
        ((-0.85, -5.1, -0.85, 0.85, 1.1, 0.85), _faces(_BONE)),
        ((-1.25, -6.15, -1.25, 1.25, -4.55, 1.25), _faces(_JOINT)),
        ((-0.75, -10.15, -0.75, 0.75, -5.5, 0.75), _faces(_MID)),
        ((-1.15, -11.15, -1.7, 1.15, -9.35, 0.55), _faces(_BONE)),
    ]


def _leg_boxes():
    return [
        ((-0.9, -6.1, -0.9, 0.9, 0.3, 0.9), _faces(_BONE)),
        ((-1.2, -7.05, -1.2, 1.2, -5.55, 1.2), _faces(_JOINT)),
        ((-0.7, -12.05, -0.7, 0.7, -6.45, 0.7), _faces(_MID)),
        ((-1.15, -12.35, -2.3, 1.15, -11.15, 0.9), _faces(_SHADE)),
    ]


def _head_boxes():
    cranium = _faces(_BONE, north=_FACE, down=_JOINT, south=_SHADE)
    jaw = _faces(_SHADE, north=_TEETH, up=_BONE)
    hinge = _faces(_JOINT)
    # The jaw overlaps the cranium and rises into hinges at each cheek.
    return [
        ((-3.6, 2.2, -3.6, 3.6, 8.0, 3.6), cranium),
        ((-3.0, 0.0, -4.3, 3.0, 2.7, 2.6), jaw),
        ((-3.6, 0.8, -1.2, -2.5, 3.6, 2.2), hinge),
        ((2.5, 0.8, -1.2, 3.6, 3.6, 2.2), hinge),
    ]


SKELETON_BOXES = {
    "head": _head_boxes(),
    "body": _body_boxes(),
    "arm_r": _arm_boxes(),
    "arm_l": _arm_boxes(),
    "leg_r": _leg_boxes(),
    "leg_l": _leg_boxes(),
}


def build_skeleton_meshes():
    return {name: _combine(SKELETON_BOXES[name]) for name in PART_ORDER}


class SkeletonMeshes:
    def __init__(self):
        self.parts = build_skeleton_meshes()

    def posed_vertices(self, position, yaw, poses):
        chunks = []
        index_chunks = []
        base = 0
        for name in PART_ORDER:
            vertices, indices = self.parts[name]
            pitch, roll, _yaw = poses.get(name, (0.0, 0.0, 0.0))
            matrix = model_matrix(position, yaw, name, pitch=pitch, roll=roll)
            transformed = transform_vertices(vertices, matrix)
            chunks.append(transformed)
            index_chunks.append(indices + base)
            base += transformed.shape[0] // 8
        if not chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint32)
        return np.concatenate(chunks), np.concatenate(index_chunks)
