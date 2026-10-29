import math

import numpy as np

from server.blocks import ASSETS, BLOCKS, FACE_INDEX
from server.images import read_png
from server.items import ITEMS
from server.player_model import (
    PART_ORDER,
    build_fp_arm_mesh,
    build_player_meshes,
    first_person_arm_matrix,
    first_person_held_matrix,
    held_item_matrix,
    limb_pose,
    model_matrix,
    transform_vertices,
)
from server.stacks import stack_id

VIEW_FIRST = 0
VIEW_THIRD_BACK = 1
VIEW_THIRD_FRONT = 2
VIEW_DISTANCE = 4.0

# Minecraft item units: 16 texture pixels = 1.0 local; thickness = 1 pixel.
_ITEM_PIXEL = 1.0 / 16.0
_EXTRUDE_CACHE = {}


class PlayerMeshes:
    def __init__(self):
        self.parts = build_player_meshes()
        self.fp_arm = build_fp_arm_mesh()

    def posed_vertices(self, position, yaw, poses, parts=None):
        names = PART_ORDER if parts is None else parts
        chunks = []
        index_chunks = []
        base = 0
        for name in names:
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

    def first_person_arm(self, eye, yaw, pitch, poses):
        arm_pitch, arm_roll, _unused = poses.get("arm_r", (0.0, 0.0, 0.0))
        matrix = first_person_arm_matrix(eye, yaw, pitch, arm_pitch, arm_roll)
        vertices, indices = self.fp_arm
        return transform_vertices(vertices, matrix), indices.copy()


def walk_speed(player):
    if hasattr(player, "_move_speed"):
        hspeed = math.hypot(player.vel[0], player.vel[2])
        if hspeed < 0.1:
            return 0.0
        mspeed = getattr(player, "_move_speed", 0.0)
        return mspeed if mspeed > 0.1 else 0.0
    hspeed = math.hypot(player.vel[0], player.vel[2])
    actual = getattr(player, "actual_speed", None)
    if actual is not None:
        speed = min(hspeed, actual)
    else:
        speed = hspeed
    return speed if speed > 0.1 else 0.0


def third_person_pose(player, time):
    pitch = getattr(player, "pitch", 0.0)
    return limb_pose(time, walk_speed(player), player.yaw, pitch=pitch)


def first_person_pose(player, time):
    speed = walk_speed(player)
    bob = math.sin(time * 10.0) * min(1.0, speed * 1.4) * 0.06 if speed > 0.1 else 0.0
    return {
        "arm_r": (bob, 0.0, 0.0),
    }


def held_item_mesh(item_id, matrix, atlas="block"):
    item = ITEMS.get(item_id)
    if item is None:
        return None
    if item.model == "block" and item.block in BLOCKS:
        return _block_cube(item.block, matrix), "block"
    if item.model == "flat":
        return _flat_item(item, matrix), "item"
    if item.model == "axes":
        return _debug_axes_mesh(item, matrix), "item"
    return None


def _debug_axes_mesh(item, matrix):
    """RGB gizmo: +X red, +Y green, +Z blue (local item axes after the hold matrix)."""
    u0, v0, u1, v1 = item.icon
    du, dv = u1 - u0, v1 - v0

    def swatch(frac_lo, frac_hi):
        # Sample a solid stripe from the debug_axes texture (R|G|B columns).
        su0 = u0 + frac_lo * du
        su1 = u0 + frac_hi * du
        sv0 = v0 + 0.35 * dv
        sv1 = v0 + 0.65 * dv
        return ((su0, sv0), (su1, sv0), (su1, sv1), (su0, sv1))

    red = swatch(0.05, 0.28)
    green = swatch(0.38, 0.62)
    blue = swatch(0.72, 0.95)
    white = swatch(0.45, 0.55)  # center mark; fine as origin stub

    vertices = []
    indices = []

    def add_box(x0, y0, z0, x1, y1, z1, uvs):
        faces = (
            ((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (1, 0, 0)),
            ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (-1, 0, 0)),
            ((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0), (0, 1, 0)),
            ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (0, -1, 0)),
            ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0, 0, 1)),
            ((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (0, 0, -1)),
        )
        for a, b, c, d, normal in faces:
            base = len(vertices) // 8
            for corner, uv in zip((a, b, c, d), uvs):
                vertices.extend((*corner, *normal, *uv))
            indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))

    t = 0.04
    length = 0.75
    add_box(0.0, -t, -t, length, t, t, red)  # +X
    add_box(-t, 0.0, -t, t, length, t, green)  # +Y
    add_box(-t, -t, 0.0, t, t, length, blue)  # +Z
    add_box(-t * 1.4, -t * 1.4, -t * 1.4, t * 1.4, t * 1.4, t * 1.4, white)
    raw = np.asarray(vertices, dtype=np.float32)
    # Flat lighting so axis colors stay readable.
    raw.reshape(-1, 8)[:, 3:6] = (0.0, 0.0, 1.0)
    return transform_vertices(raw, matrix), np.asarray(indices, dtype=np.uint32)


def _block_cube(block_id, matrix):
    block = BLOCKS[block_id]
    # Held block ~0.5 of a world block so it reads large in FP like Minecraft.
    # Inset faces slightly along their normals so shared edges never Z-fight.
    inset = 0.001
    vertices = []
    indices = []

    is_half_slab = bool(block.properties.get("slab") and not block.properties.get("double_slab"))
    y_center_offset = 0.125 if is_half_slab else 0.0

    elements = block.elements if (hasattr(block, "elements") and block.elements) else None
    if not elements:
        elements = [{"from": [0, 0, 0], "to": [16, 16, 16], "faces": {i: block.face_uvs[i] for i in range(6) if block.face_uvs[i] is not None}}]

    for element in elements:
        orig = element["from"]
        ext = element["to"]
        x0 = -0.25 + (orig[0] / 16.0) * 0.5 + inset
        x1 = -0.25 + (ext[0] / 16.0) * 0.5 - inset
        y0 = -0.25 + (orig[1] / 16.0) * 0.5 + y_center_offset + inset
        y1 = -0.25 + (ext[1] / 16.0) * 0.5 + y_center_offset - inset
        z0 = -0.25 + (orig[2] / 16.0) * 0.5 + inset
        z1 = -0.25 + (ext[2] / 16.0) * 0.5 - inset

        faces = (
            ("east", (x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (1, 0, 0)),
            ("west", (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (-1, 0, 0)),
            ("up", (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0), (0, 1, 0)),
            ("down", (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (0, -1, 0)),
            ("south", (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0, 0, 1)),
            ("north", (x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (0, 0, -1)),
        )
        for face, a, b, c, d, normal in faces:
            f_idx = FACE_INDEX[face]
            if "faces" in element and f_idx not in element["faces"]:
                continue
            uv = element.get("faces", {}).get(f_idx) or block.face_uvs[f_idx] or block.face_uvs[FACE_INDEX["up"]]
            nx, ny, nz = normal
            push = 0.002
            a = (a[0] + nx * push, a[1] + ny * push, a[2] + nz * push)
            b = (b[0] + nx * push, b[1] + ny * push, b[2] + nz * push)
            c = (c[0] + nx * push, c[1] + ny * push, c[2] + nz * push)
            d = (d[0] + nx * push, d[1] + ny * push, d[2] + nz * push)
            base = len(vertices) // 8
            for corner, (u, v) in zip((a, b, c, d), uv):
                vertices.extend((*corner, *normal, u, v))
            indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))

    raw = np.asarray(vertices, dtype=np.float32)
    return transform_vertices(raw, matrix), np.asarray(indices, dtype=np.uint32)


def falling_block_mesh(block_id, pos):
    block = BLOCKS.get(block_id)
    if block is None:
        return None
    px, py, pz = pos
    x0, y0, z0 = px - 0.5, py, pz - 0.5
    x1, y1, z1 = px + 0.5, py + 1.0, pz + 0.5
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
        uv = block.face_uvs[FACE_INDEX[face]] or block.face_uvs[FACE_INDEX["up"]]
        base = len(vertices) // 8
        for corner, (u, v) in zip((a, b, c, d), uv):
            vertices.extend((*corner, *normal, u, v))
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.uint32)


def _flat_item(item, matrix):
    vertices, indices = _extruded_item_mesh(item)
    return transform_vertices(vertices, matrix), indices.copy()


def _extruded_item_mesh(item):
    """Minecraft-style flat item: 1px extrusion with stair-steps filled so FP can't see through."""
    cached = _EXTRUDE_CACHE.get(item.id)
    if cached is not None:
        return cached
    path = ASSETS / "textures" / f"{item.texture}.png"
    image = read_png(path)
    height, width = image.shape[:2]
    opaque = image[:, :, 3] >= 128
    # Fill diagonal stair notches so angled FP views can't see through the slab.
    solid = opaque.copy()
    uv_source = np.empty((height, width, 2), dtype=np.int32)
    uv_source[:, :, 0] = np.arange(width)[None, :]
    uv_source[:, :, 1] = np.arange(height)[:, None]
    for py in range(height):
        for px in range(width):
            if not opaque[py, px]:
                continue
            for dx, dy in ((1, -1), (1, 1), (-1, -1), (-1, 1)):
                nx, ny = px + dx, py + dy
                if not (0 <= nx < width and 0 <= ny < height and opaque[ny, nx]):
                    continue
                for cx, cy in ((px + dx, py), (px, py + dy)):
                    if 0 <= cx < width and 0 <= cy < height and not opaque[cy, cx]:
                        solid[cy, cx] = True
                        uv_source[cy, cx] = (px, py)
    u0, v0, u1, v1 = item.icon
    vertices = []
    indices = []
    half = 0.5 * _ITEM_PIXEL
    # Side walls only grow outward — front/back stay exact so neighbors don't Z-fight.
    side_pad = _ITEM_PIXEL * 0.05
    ox = width * 0.5
    oy = height * 0.5

    def add_quad(a, b, c, d, normal, uvs):
        base = len(vertices) // 8
        for corner, (u, v) in zip((a, b, c, d), uvs):
            vertices.extend((*corner, *normal, u, v))
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))

    def is_solid(px, py):
        return 0 <= px < width and 0 <= py < height and bool(solid[py, px])

    def pixel_uv(px, py):
        sx, sy = (int(uv_source[py, px, 0]), int(uv_source[py, px, 1]))
        su0 = u0 + ((sx + 0.05) / width) * (u1 - u0)
        su1 = u0 + ((sx + 0.95) / width) * (u1 - u0)
        sv1 = v0 + ((height - sy - 0.05) / height) * (v1 - v0)
        sv0 = v0 + ((height - sy - 0.95) / height) * (v1 - v0)
        return su0, sv0, su1, sv1

    # Emit backs, then fronts, then sides so depth-disabled FP draws still layer right.
    z0, z1 = -half, half
    quads_back = []
    quads_front = []
    quads_side = []
    for py in range(height):
        for px in range(width):
            if not solid[py, px]:
                continue
            lx0 = (px - ox) * _ITEM_PIXEL
            lx1 = (px + 1 - ox) * _ITEM_PIXEL
            ly1 = (oy - py) * _ITEM_PIXEL
            ly0 = (oy - py - 1) * _ITEM_PIXEL
            su0, sv0, su1, sv1 = pixel_uv(px, py)
            front = ((su0, sv0), (su1, sv0), (su1, sv1), (su0, sv1))
            back = ((su1, sv0), (su0, sv0), (su0, sv1), (su1, sv1))
            rim_u = (su0 + su1) * 0.5
            rim_v = (sv0 + sv1) * 0.5
            rim = ((rim_u, rim_v), (rim_u, rim_v), (rim_u, rim_v), (rim_u, rim_v))
            quads_back.append(((lx1, ly0, z0), (lx0, ly0, z0), (lx0, ly1, z0), (lx1, ly1, z0), (0, 0, -1), back))
            quads_front.append(((lx0, ly0, z1), (lx1, ly0, z1), (lx1, ly1, z1), (lx0, ly1, z1), (0, 0, 1), front))
            if not is_solid(px - 1, py):
                quads_side.append(
                    (
                        (lx0, ly0 - side_pad, z0),
                        (lx0, ly0 - side_pad, z1),
                        (lx0, ly1 + side_pad, z1),
                        (lx0, ly1 + side_pad, z0),
                        (-1, 0, 0),
                        rim,
                    )
                )
            if not is_solid(px + 1, py):
                quads_side.append(
                    (
                        (lx1, ly0 - side_pad, z1),
                        (lx1, ly0 - side_pad, z0),
                        (lx1, ly1 + side_pad, z0),
                        (lx1, ly1 + side_pad, z1),
                        (1, 0, 0),
                        rim,
                    )
                )
            if not is_solid(px, py + 1):
                quads_side.append(
                    (
                        (lx0 - side_pad, ly0, z0),
                        (lx1 + side_pad, ly0, z0),
                        (lx1 + side_pad, ly0, z1),
                        (lx0 - side_pad, ly0, z1),
                        (0, -1, 0),
                        rim,
                    )
                )
            if not is_solid(px, py - 1):
                quads_side.append(
                    (
                        (lx0 - side_pad, ly1, z1),
                        (lx1 + side_pad, ly1, z1),
                        (lx1 + side_pad, ly1, z0),
                        (lx0 - side_pad, ly1, z0),
                        (0, 1, 0),
                        rim,
                    )
                )
    for a, b, c, d, normal, uvs in quads_back + quads_front + quads_side:
        add_quad(a, b, c, d, normal, uvs)

    raw = np.asarray(vertices, dtype=np.float32)
    ys, xs = np.where(opaque)
    if len(xs):
        score = xs.astype(np.float32) + ys.astype(np.float32)
        pick = int(np.argmax(score))
        hx = (float(xs[pick]) + 0.5 - ox) * _ITEM_PIXEL
        hy = (oy - float(ys[pick]) - 0.5) * _ITEM_PIXEL
        raw.reshape(-1, 8)[:, 0] -= hx
        raw.reshape(-1, 8)[:, 1] -= hy
    # Flat lighting — dark lambert on 1px rims reads as see-through seams.
    raw.reshape(-1, 8)[:, 3:6] = (0.0, 0.0, 1.0)
    mesh = (raw, np.asarray(indices, dtype=np.uint32))
    _EXTRUDE_CACHE[item.id] = mesh
    return mesh


def item_hold_matrix(player, poses, first_person=False, eye=None, item_id=""):
    arm_pitch, arm_roll, _unused = poses.get("arm_r", (0.0, 0.0, 0.0))
    item = ITEMS.get(item_id)
    flat = item is not None and item.model in ("flat", "axes")
    if first_person:
        return first_person_held_matrix(
            eye,
            player.yaw,
            player.pitch,
            arm_pitch,
            arm_roll,
            # Flat ~MC firstperson size; block mesh is already 0.5 world units.
            item_scale=0.68 if flat else 0.6,
            flat=flat,
        )
    return held_item_matrix(
        player.pos,
        player.yaw,
        arm_pitch,
        item_scale=0.55 if flat else 0.75,
        flat=flat,
    )


def selected_item_id(hotbar, selected):
    if selected < 0 or selected >= len(hotbar):
        return ""
    return stack_id(hotbar[selected])


def dropped_item_matrix(pos, time_val, scale=0.4):
    x, y, z = pos
    bob = math.sin(time_val * 3.5) * 0.05
    yaw = time_val * 1.8
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.array(
        [
            [scale * cy, 0.0, scale * sy, x],
            [0.0, scale, 0.0, y + 0.12 + bob],
            [-scale * sy, 0.0, scale * cy, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _look(yaw, pitch):
    cp = math.cos(pitch)
    return (-math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp)
