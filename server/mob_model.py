"""Box and bone models for asset mobs.

A lua model is a list of boxes with pivots, the same shape the builtin mobs use.
A GeckoLib or AzureLib model is a Bedrock geometry file plus an animation file.
Both libraries share that format. Rotations are degrees. ``math.sin`` and
``math.cos`` take degrees, matching Molang. Numeric keyframes are interpolated.
"""

import math
import re
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"\s*(\d+\.\d+|\d+|query\.[A-Za-z_]+|math\.[A-Za-z_]+|[A-Za-z_][A-Za-z0-9_]*|.)")


def _face_uv(u0, v0, u1, v1, width, height):
    if u1 < u0:
        u0, u1 = u1, u0
    if v1 < v0:
        v0, v1 = v1, v0
    su0 = (u0 + 0.5) / width
    su1 = (u1 - 0.5) / width
    sv_top = 1.0 - (v0 + 0.5) / height
    sv_bot = 1.0 - (v1 - 0.5) / height
    return (
        (su0, sv_bot),
        (su1, sv_bot),
        (su1, sv_top),
        (su0, sv_top),
    )


def _box_vertices(box, uv_dict, texture_size):
    width, height = texture_size
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
        uv = _face_uv(u0, v0, u1, v1, width, height)
        base = len(vertices) // 8
        for corner, (u, v) in zip((a, b, c, d), uv):
            vertices.extend(
                (
                    float(corner[0]),
                    float(corner[1]),
                    float(corner[2]),
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                    u,
                    v,
                )
            )
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    return vertices, indices


def _translation(x, y, z):
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    return matrix


def _rotation_y(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = c
    matrix[0, 2] = s
    matrix[2, 0] = -s
    matrix[2, 2] = c
    return matrix


def _rotation_xyz(pitch, yaw, roll):
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cz, sz = math.cos(roll), math.sin(roll)
    matrix = np.eye(4, dtype=np.float32)
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


def _transform(vertices, matrix):
    if len(vertices) == 0:
        return vertices
    out = np.array(vertices, dtype=np.float32, copy=True)
    count = out.shape[0] // 8
    positions = out.reshape(count, 8)[:, 0:3]
    normals = out.reshape(count, 8)[:, 3:6]
    ones = np.ones((count, 1), dtype=np.float32)
    transformed = (matrix @ np.concatenate([positions, ones], axis=1).T).T[:, :3]
    rotated = (matrix[:3, :3] @ normals.T).T
    lengths = np.linalg.norm(rotated, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-6)
    out.reshape(count, 8)[:, 0:3] = transformed
    out.reshape(count, 8)[:, 3:6] = rotated / lengths
    return out


def _mob_matrix(position, yaw):
    matrix = _rotation_y(yaw)
    matrix[0, 3] = position[0]
    matrix[1, 3] = position[1]
    matrix[2, 3] = position[2]
    return matrix


class LuaBoxModel:
    """Parts are boxes in pixel space. ``pose`` is still, swing, -swing, or look."""

    def __init__(self, parts, texture_size):
        self.parts = parts
        self.texture_size = texture_size
        built = []
        for part in parts:
            vertices, indices = _box_vertices(part["box"], part["uv"], texture_size)
            scale = np.array(vertices, dtype=np.float32)
            if len(scale):
                scale.reshape(-1, 8)[:, 0:3] *= np.float32(1.0 / 16.0)
            built.append((np.asarray(scale, dtype=np.float32), np.asarray(indices, dtype=np.uint32)))
        self._built = built

    def posed_vertices(self, position, yaw, time, speed, pitch=0.0):
        if speed < 0.05:
            swing = 0.0
        else:
            swing = math.sin(time * 10.0) * min(1.0, speed * 1.6) * 0.7
        body = _mob_matrix(position, yaw)
        chunks = []
        index_chunks = []
        base = 0
        for part, (vertices, indices) in zip(self.parts, self._built):
            if len(vertices) == 0:
                continue
            role = part.get("pose") or "still"
            if role == "swing":
                part_pitch = swing
            elif role == "-swing":
                part_pitch = -swing
            elif role == "look":
                part_pitch = pitch
            else:
                part_pitch = 0.0
            px, py, pz = (float(v) / 16.0 for v in part["pivot"])
            local = _translation(px, py, pz) @ _rotation_xyz(part_pitch, 0.0, 0.0) @ _translation(-px, -py, -pz)
            transformed = _transform(vertices, body @ local)
            chunks.append(transformed)
            index_chunks.append(indices + base)
            base += transformed.shape[0] // 8
        if not chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint32)
        return np.concatenate(chunks), np.concatenate(index_chunks)


def _eval_molang(expr, time):
    tokens = _TOKEN.findall(str(expr))
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else ""

    def take():
        nonlocal pos
        token = tokens[pos]
        pos += 1
        return token

    def parse_expr():
        value = parse_term()
        while peek() in ("+", "-"):
            op = take()
            right = parse_term()
            value = value + right if op == "+" else value - right
        return value

    def parse_term():
        value = parse_factor()
        while peek() in ("*", "/"):
            op = take()
            right = parse_factor()
            value = value * right if op == "*" else value / right
        return value

    def parse_factor():
        token = peek()
        if token in ("+", "-"):
            take()
            value = parse_factor()
            return value if token == "+" else -value
        if token == "(":
            take()
            value = parse_expr()
            if peek() == ")":
                take()
            return value
        take()
        if token in ("math.sin", "math.cos", "math.abs", "math.lerp"):
            if peek() == "(":
                take()
            args = [parse_expr()]
            while peek() == ",":
                take()
                args.append(parse_expr())
            if peek() == ")":
                take()
            if token == "math.sin":
                return math.sin(math.radians(args[0]))
            if token == "math.cos":
                return math.cos(math.radians(args[0]))
            if token == "math.abs":
                return abs(args[0])
            if token == "math.lerp" and len(args) >= 3:
                return args[0] + (args[1] - args[0]) * args[2]
            return 0.0
        if token in ("query.anim_time", "query.life_time", "query.anim_time_abs"):
            return float(time)
        try:
            return float(token)
        except ValueError:
            return 0.0

    try:
        return float(parse_expr())
    except (IndexError, ZeroDivisionError, ValueError):
        return 0.0


def _component(value, time):
    if isinstance(value, str):
        return _eval_molang(value, time)
    return float(value)


def _vector(value, time):
    if isinstance(value, dict):
        if "vector" in value:
            value = value["vector"]
        elif "post" in value:
            value = value["post"]
        else:
            return (0.0, 0.0, 0.0)
    if isinstance(value, str):
        return (_eval_molang(value, time), 0.0, 0.0)
    if not isinstance(value, (list, tuple)):
        return (0.0, 0.0, 0.0)
    comps = [_component(item, time) for item in list(value)[:3]]
    while len(comps) < 3:
        comps.append(0.0)
    return tuple(comps)


def _keyframes(channel):
    if not isinstance(channel, dict) or "vector" in channel or "post" in channel:
        return None
    frames = []
    for key, value in channel.items():
        try:
            frames.append((float(key), value))
        except (TypeError, ValueError):
            continue
    if not frames:
        return None
    frames.sort(key=lambda item: item[0])
    return frames


def _sample_channel(channel, time, length, loop):
    if channel is None:
        return None
    frames = _keyframes(channel)
    if frames is None:
        return _vector(channel, time)
    if loop and length and length > 0:
        time = time % length
    if time <= frames[0][0]:
        return _vector(frames[0][1], time)
    if time >= frames[-1][0]:
        return _vector(frames[-1][1], time)
    for index in range(len(frames) - 1):
        t0, v0 = frames[index]
        t1, v1 = frames[index + 1]
        if t0 <= time <= t1:
            span = t1 - t0
            blend = 0.0 if span <= 0 else (time - t0) / span
            a = _vector(v0, time)
            b = _vector(v1, time)
            return tuple(a[i] + (b[i] - a[i]) * blend for i in range(3))
    return _vector(frames[-1][1], time)


def _box_uv(uv, size):
    """Bedrock box unwrap. ``uv`` is the top-left of the map, ``size`` is pixels."""
    u, v = float(uv[0]), float(uv[1])
    sx, sy, sz = (abs(float(n)) for n in size)
    return {
        "up": (u + sz, v, u + sz + sx, v + sz),
        "down": (u + sz + sx, v, u + sz + sx + sx, v + sz),
        "west": (u, v + sz, u + sz, v + sz + sy),
        "north": (u + sz, v + sz, u + sz + sx, v + sz + sy),
        "east": (u + sz + sx, v + sz, u + sz + sx + sz, v + sz + sy),
        "south": (u + sz + sx + sz, v + sz, u + sz + sx + sz + sx, v + sz + sy),
    }


def _face_rect(face):
    if isinstance(face, (list, tuple)) and len(face) >= 4:
        return tuple(float(n) for n in face[:4])
    if not isinstance(face, dict):
        return None
    uv = face.get("uv", (0, 0))
    size = face.get("uv_size")
    u, v = float(uv[0]), float(uv[1])
    if size is None:
        if len(uv) >= 4:
            return (u, v, float(uv[2]), float(uv[3]))
        return (u, v, u + 1.0, v + 1.0)
    return (u, v, u + float(size[0]), v + float(size[1]))


def _cube_uvs(cube):
    uv = cube.get("uv", (0, 0))
    size = cube.get("size", (0, 0, 0))
    if isinstance(uv, dict):
        rects = {}
        for face, value in uv.items():
            rect = _face_rect(value)
            if rect is not None:
                rects[str(face)] = rect
        return rects
    if isinstance(uv, (list, tuple)) and len(uv) >= 2:
        return _box_uv(uv, size)
    return {}


class GeckoModel:
    """Bedrock geometry posed by a GeckoLib or AzureLib animation clip."""

    def __init__(self, bones, animations, texture_size, idle, walk):
        self.bones = bones
        self.animations = animations
        self.texture_size = texture_size
        self.idle = idle
        self.walk = walk
        self._by_name = {bone["name"]: bone for bone in bones}

    def _clip(self, speed):
        name = self.walk if speed >= 0.05 and self.walk in self.animations else self.idle
        if name not in self.animations and self.animations:
            name = next(iter(self.animations))
        return self.animations.get(name)

    def posed_vertices(self, position, yaw, time, speed, pitch=0.0):
        clip = self._clip(speed)
        bone_rots = {}
        if clip is not None:
            length = float(clip.get("length") or 0.0)
            loop = bool(clip.get("loop", True))
            for name, channels in clip.get("bones", {}).items():
                rotation = _sample_channel(channels.get("rotation"), time, length, loop)
                offset = _sample_channel(channels.get("position"), time, length, loop)
                if rotation is None:
                    rotation = (0.0, 0.0, 0.0)
                if offset is None:
                    offset = (0.0, 0.0, 0.0)
                bone_rots[name] = (
                    math.radians(rotation[0]),
                    math.radians(rotation[1]),
                    math.radians(rotation[2]),
                    offset[0] / 16.0,
                    offset[1] / 16.0,
                    offset[2] / 16.0,
                )
        cache = {}

        def bone_matrix(name, guard):
            if name in cache:
                return cache[name]
            if name in guard:
                return np.eye(4, dtype=np.float32)
            guard.add(name)
            bone = self._by_name[name]
            rx, ry, rz, ox, oy, oz = bone_rots.get(name, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            px, py, pz = bone["pivot"]
            local = _translation(px + ox, py + oy, pz + oz) @ _rotation_xyz(rx, ry, rz) @ _translation(-px, -py, -pz)
            parent = bone.get("parent")
            if parent and parent in self._by_name:
                matrix = bone_matrix(parent, guard) @ local
            else:
                matrix = local
            cache[name] = matrix
            return matrix

        body = _mob_matrix(position, yaw)
        chunks = []
        index_chunks = []
        base = 0
        for bone in self.bones:
            vertices = bone["vertices"]
            indices = bone["indices"]
            if len(vertices) == 0:
                continue
            matrix = body @ bone_matrix(bone["name"], set())
            transformed = _transform(vertices, matrix)
            chunks.append(transformed)
            index_chunks.append(indices + base)
            base += transformed.shape[0] // 8
        if not chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint32)
        return np.concatenate(chunks), np.concatenate(index_chunks)


def _pixels_to_blocks(box):
    return tuple(float(value) / 16.0 for value in box)


def build_lua_model(parts, texture_size):
    return LuaBoxModel(parts, texture_size)


def _cube_box(cube):
    origin = cube.get("origin", (0, 0, 0))
    size = cube.get("size", (0, 0, 0))
    inflate = float(cube.get("inflate", 0) or 0)
    x0 = float(origin[0]) - inflate
    y0 = float(origin[1]) - inflate
    z0 = float(origin[2]) - inflate
    x1 = x0 + float(size[0]) + inflate * 2.0
    y1 = y0 + float(size[1]) + inflate * 2.0
    z1 = z0 + float(size[2]) + inflate * 2.0
    return (x0, y0, z0, x1, y1, z1)


def build_gecko_model(geometry, animations, texture_size, idle, walk):
    """``geometry`` is one minecraft:geometry entry. Animations are the file's clips."""
    description = geometry.get("description") or {}
    width = float(description.get("texture_width", texture_size[0]) or texture_size[0])
    height = float(description.get("texture_height", texture_size[1]) or texture_size[1])
    bones = []
    for raw in geometry.get("bones") or []:
        name = str(raw.get("name") or "bone")
        pivot = raw.get("pivot") or (0, 0, 0)
        vertices = []
        indices = []
        base = 0
        for cube in raw.get("cubes") or []:
            uvs = _cube_uvs(cube)
            if cube.get("mirror"):
                uvs = dict(uvs)
                uvs["east"], uvs["west"] = uvs.get("west", (0, 0, 1, 1)), uvs.get("east", (0, 0, 1, 1))
            box = _pixels_to_blocks(_cube_box(cube))
            part_vertices, part_indices = _box_vertices(box, uvs, (width, height))
            if cube.get("rotation"):
                cube_pivot = cube.get("pivot") or raw.get("pivot") or (0, 0, 0)
                px, py, pz = (float(cube_pivot[0]) / 16.0, float(cube_pivot[1]) / 16.0, float(cube_pivot[2]) / 16.0)
                rotation = cube["rotation"]
                spin = _rotation_xyz(
                    math.radians(float(rotation[0])),
                    math.radians(float(rotation[1])) if len(rotation) > 1 else 0.0,
                    math.radians(float(rotation[2])) if len(rotation) > 2 else 0.0,
                )
                baked = _translation(px, py, pz) @ spin @ _translation(-px, -py, -pz)
                part_vertices = _transform(part_vertices, baked).tolist()
            vertices.extend(part_vertices)
            indices.extend(index + base for index in part_indices)
            base += len(part_vertices) // 8
        bones.append(
            {
                "name": name,
                "parent": raw.get("parent") or "",
                "pivot": (
                    float(pivot[0]) / 16.0,
                    float(pivot[1]) / 16.0,
                    float(pivot[2]) / 16.0,
                ),
                "vertices": np.asarray(vertices, dtype=np.float32),
                "indices": np.asarray(indices, dtype=np.uint32),
            }
        )
    clips = {}
    for name, clip in (animations or {}).items():
        if not isinstance(clip, dict):
            continue
        loop = clip.get("loop", True)
        clips[str(name)] = {
            "length": float(clip.get("animation_length", 0) or 0),
            "loop": False if loop in (False, "hold_on_last_frame", "false") else True,
            "bones": clip.get("bones") or {},
        }
    return GeckoModel(bones, clips, (width, height), idle, walk)


def load_gecko_files(geometry_path, animation_path, texture_size, idle, walk):
    import json

    geometry_doc = json.loads(Path(geometry_path).read_text(encoding="utf-8"))
    entries = geometry_doc.get("minecraft:geometry") or []
    if not entries:
        raise ValueError(f"{geometry_path}: minecraft:geometry is missing")
    animation_doc = {}
    if animation_path:
        animation_doc = json.loads(Path(animation_path).read_text(encoding="utf-8"))
    animations = animation_doc.get("animations") or animation_doc
    return build_gecko_model(entries[0], animations, texture_size, idle, walk)
