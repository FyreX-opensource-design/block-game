import numpy as np

ICON_SIZE = 32
_SCALE = 15.0 / 16.0


def render_block_icon(top, side, height_scale=1.0, width_scale=1.0):
    image = np.zeros((ICON_SIZE, ICON_SIZE, 4), dtype=np.uint8)
    # Left = south, right = east, top last (painter order).
    _paint_face(image, side, "south", 0.84, height_scale=height_scale, width_scale=width_scale)
    _paint_face(image, side, "east", 0.62, height_scale=height_scale, width_scale=width_scale)
    _paint_face(image, top, "up", 1.0, height_scale=height_scale, width_scale=width_scale)
    return _seal(image)


def render_model_icon(elements, textures, default_texture=None):
    image = np.zeros((ICON_SIZE, ICON_SIZE, 4), dtype=np.uint8)
    depth_buf = np.full((ICON_SIZE, ICON_SIZE), -1e9, dtype=np.float32)
    shades = {"south": 0.84, "east": 0.62, "up": 1.0}

    face_aliases = {
        "south": ("south", 3),
        "east": ("east", 5),
        "up": ("up", 1),
    }

    for el in elements:
        x0, y0, z0 = el["from"]
        x1, y1, z1 = el["to"]
        el_faces = el.get("faces", {})

        for face_name, (str_key, int_key) in face_aliases.items():
            f_info = el_faces.get(str_key)
            if f_info is None and int_key in el_faces:
                f_info = el_faces[int_key]
            if f_info is None:
                continue

            if face_name == "south":
                corners3 = [
                    (x0, y1, z1),
                    (x1, y1, z1),
                    (x1, y0, z1),
                    (x0, y0, z1),
                ]
                default_uv = [x0, 16.0 - y1, x1, 16.0 - y0]
            elif face_name == "east":
                corners3 = [
                    (x1, y1, z1),
                    (x1, y1, z0),
                    (x1, y0, z0),
                    (x1, y0, z1),
                ]
                default_uv = [16.0 - z1, 16.0 - y1, 16.0 - z0, 16.0 - y0]
            else:  # "up"
                corners3 = [
                    (x0, y1, z0),
                    (x1, y1, z0),
                    (x1, y1, z1),
                    (x0, y1, z1),
                ]
                default_uv = [x0, z0, x1, z1]

            tex = None
            if isinstance(f_info, dict):
                tex_key = str(f_info.get("texture", "")).lstrip("#")
                if tex_key and tex_key in textures:
                    tex = textures[tex_key]
                elif tex_key and hasattr(textures, "get"):
                    tex = textures.get(tex_key)
                if tex is None:
                    for k in ("texture", "side", "top", "all"):
                        if k in textures:
                            tex = textures[k]
                            break
                if tex is None:
                    tex = default_texture
            elif isinstance(f_info, np.ndarray):
                tex = f_info
            if tex is None:
                tex = default_texture
            if tex is None and textures:
                tex = next(iter(textures.values()))
            if tex is None:
                continue

            if isinstance(f_info, dict) and "uv" in f_info:
                raw_uv = f_info["uv"]
                if len(raw_uv) == 4 and not isinstance(raw_uv[0], (list, tuple)):
                    u0, v0, u1, v1 = raw_uv
                    scale = 16.0 if max(abs(u0), abs(v0), abs(u1), abs(v1)) > 1.0 else 1.0
                    uvs = [
                        (u0 / scale, v0 / scale),
                        (u1 / scale, v0 / scale),
                        (u1 / scale, v1 / scale),
                        (u0 / scale, v1 / scale),
                    ]
                elif len(raw_uv) == 4:
                    uvs = [tuple(p) for p in raw_uv]
                else:
                    u0, v0, u1, v1 = [v / 16.0 for v in default_uv]
                    uvs = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            elif isinstance(f_info, (list, tuple)) and len(f_info) == 4 and isinstance(f_info[0], (list, tuple)):
                uvs = [tuple(p) for p in f_info]
            else:
                u0, v0, u1, v1 = [v / 16.0 for v in default_uv]
                uvs = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]

            _paint_face_depth(image, depth_buf, corners3, uvs, tex, shades[face_name])

    return _seal(image)


def render_flat_icon(texture):
    image = np.zeros((ICON_SIZE, ICON_SIZE, 4), dtype=np.uint8)
    scale = max(1, ICON_SIZE // max(int(texture.shape[0]), int(texture.shape[1])))
    scaled = np.repeat(np.repeat(texture, scale, axis=0), scale, axis=1)
    y = (ICON_SIZE - scaled.shape[0]) // 2
    x = (ICON_SIZE - scaled.shape[1]) // 2
    image[y : y + scaled.shape[0], x : x + scaled.shape[1]] = scaled[: ICON_SIZE - y, : ICON_SIZE - x]
    return image


def _paint_face(image, texture, face, shade, height_scale=1.0, width_scale=1.0):
    corners3, uvs = _face(face, height_scale=height_scale, width_scale=width_scale)
    corners = np.asarray([_project(*point) for point in corners3], dtype=np.float32)
    # Grow the quad a half-pixel so face seams do not leave holes.
    center = corners.mean(axis=0)
    corners = center + (corners - center) * 1.04
    uvs = np.asarray(uvs, dtype=np.float32)
    tint = np.array([shade, shade, shade, 1.0], dtype=np.float32)
    _fill_textured(image, corners, uvs, texture, tint)


def _seal(image):
    """Fill single-pixel holes left between projected faces."""
    alpha = image[:, :, 3] > 0
    filled = alpha.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        shifted = np.roll(np.roll(alpha, dy, axis=0), dx, axis=1)
        filled |= shifted
    holes = filled & ~alpha
    if not np.any(holes):
        return image
    sealed = image.copy()
    for y, x in zip(*np.where(holes)):
        samples = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < ICON_SIZE and 0 <= xx < ICON_SIZE and alpha[yy, xx]:
                    samples.append(image[yy, xx])
        if samples:
            sealed[y, x] = np.mean(samples, axis=0).astype(np.uint8)
            sealed[y, x, 3] = 255
    return sealed


def _face(face, height_scale=1.0, width_scale=1.0):
    # Corners CCW from outside. UV (0,0) = texture top-left.
    # width_scale narrows the block along X (centered), e.g. 0.5 for a vertical slab.
    h = 16.0 * height_scale
    y_off = (1.0 - height_scale) * 4.0
    y0 = y_off
    y1 = y_off + h
    x0 = (1.0 - width_scale) * 8.0
    x1 = 16.0 - x0
    u0 = x0 / 16.0
    u1 = x1 / 16.0
    if face == "south":
        return (
            ((x0, y1, 16.0), (x1, y1, 16.0), (x1, y0, 16.0), (x0, y0, 16.0)),
            ((u0, 1.0 - height_scale), (u1, 1.0 - height_scale), (u1, 1.0), (u0, 1.0)),
        )
    if face == "east":
        # u=0 at high z so the left edge of the east face (screen-left of that face) matches texture left.
        return (
            ((x1, y1, 16.0), (x1, y1, 0.0), (x1, y0, 0.0), (x1, y0, 16.0)),
            ((0.0, 1.0 - height_scale), (1.0, 1.0 - height_scale), (1.0, 1.0), (0.0, 1.0)),
        )
    return (
        ((x0, y1, 0.0), (x1, y1, 0.0), (x1, y1, 16.0), (x0, y1, 16.0)),
        ((u0, 0.0), (u1, 0.0), (u1, 1.0), (u0, 1.0)),
    )


def _project(x, y, z):
    sx = (x - z) * _SCALE + 16.0
    sy = (-y + (x + z) * 0.5) * _SCALE + 16.0
    return (sx, sy)


def _fill_textured(image, corners, uvs, texture, tint):
    _fill_tri_tex(image, corners[0], corners[1], corners[2], uvs[0], uvs[1], uvs[2], texture, tint)
    _fill_tri_tex(image, corners[0], corners[2], corners[3], uvs[0], uvs[2], uvs[3], texture, tint)


def _fill_tri_tex(image, a, b, c, uv_a, uv_b, uv_c, texture, tint):
    points = np.stack((a, b, c))
    min_x = max(int(np.floor(points[:, 0].min())), 0)
    max_x = min(int(np.ceil(points[:, 0].max())), ICON_SIZE - 1)
    min_y = max(int(np.floor(points[:, 1].min())), 0)
    max_y = min(int(np.ceil(points[:, 1].max())), ICON_SIZE - 1)
    if min_x > max_x or min_y > max_y:
        return
    xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    v0x, v0y = float(c[0] - a[0]), float(c[1] - a[1])
    v1x, v1y = float(b[0] - a[0]), float(b[1] - a[1])
    v2x = grid_x - float(a[0])
    v2y = grid_y - float(a[1])
    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot11 = v1x * v1x + v1y * v1y
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-8:
        return
    inv = 1.0 / denom
    dot02 = v0x * v2x + v0y * v2y
    dot12 = v1x * v2x + v1y * v2y
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    mask = (u >= -1e-4) & (v >= -1e-4) & (u + v <= 1.0 + 1e-4)
    if not np.any(mask):
        return
    height, width = texture.shape[:2]
    uu = float(uv_a[0]) + (float(uv_c[0]) - float(uv_a[0])) * u + (float(uv_b[0]) - float(uv_a[0])) * v
    vv = float(uv_a[1]) + (float(uv_c[1]) - float(uv_a[1])) * u + (float(uv_b[1]) - float(uv_a[1])) * v
    tx = np.clip(np.floor(uu * width).astype(np.int32), 0, width - 1)
    ty = np.clip(np.floor(vv * height).astype(np.int32), 0, height - 1)
    samples = texture[ty, tx]
    write = mask & (samples[:, :, 3] >= 128)
    if not np.any(write):
        return
    color = np.clip(samples.astype(np.float32) * tint, 0, 255).astype(np.uint8)
    patch = image[min_y : max_y + 1, min_x : max_x + 1]
    patch[write] = color[write]


def _paint_face_depth(image, depth_buf, corners3, uvs, texture, shade):
    corners = np.asarray([_project(*point) for point in corners3], dtype=np.float32)
    depths = np.asarray([point[0] + point[1] + point[2] for point in corners3], dtype=np.float32)
    center = corners.mean(axis=0)
    corners = center + (corners - center) * 1.04
    uvs = np.asarray(uvs, dtype=np.float32)
    tint = np.array([shade, shade, shade, 1.0], dtype=np.float32)
    _fill_tri_depth(
        image, depth_buf, corners[0], corners[1], corners[2],
        depths[0], depths[1], depths[2],
        uvs[0], uvs[1], uvs[2], texture, tint,
    )
    _fill_tri_depth(
        image, depth_buf, corners[0], corners[2], corners[3],
        depths[0], depths[2], depths[3],
        uvs[0], uvs[2], uvs[3], texture, tint,
    )


def _fill_tri_depth(image, depth_buf, a, b, c, da, db, dc, uv_a, uv_b, uv_c, texture, tint):
    points = np.stack((a, b, c))
    min_x = max(int(np.floor(points[:, 0].min())), 0)
    max_x = min(int(np.ceil(points[:, 0].max())), ICON_SIZE - 1)
    min_y = max(int(np.floor(points[:, 1].min())), 0)
    max_y = min(int(np.ceil(points[:, 1].max())), ICON_SIZE - 1)
    if min_x > max_x or min_y > max_y:
        return
    xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    v0x, v0y = float(c[0] - a[0]), float(c[1] - a[1])
    v1x, v1y = float(b[0] - a[0]), float(b[1] - a[1])
    v2x = grid_x - float(a[0])
    v2y = grid_y - float(a[1])
    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot11 = v1x * v1x + v1y * v1y
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-8:
        return
    inv = 1.0 / denom
    dot02 = v0x * v2x + v0y * v2y
    dot12 = v1x * v2x + v1y * v2y
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    mask = (u >= -1e-4) & (v >= -1e-4) & (u + v <= 1.0 + 1e-4)
    if not np.any(mask):
        return

    d = float(da) + (float(dc) - float(da)) * u + (float(db) - float(da)) * v
    patch_depth = depth_buf[min_y : max_y + 1, min_x : max_x + 1]
    depth_mask = mask & (d >= patch_depth - 1e-3)
    if not np.any(depth_mask):
        return

    height, width = texture.shape[:2]
    uu = float(uv_a[0]) + (float(uv_c[0]) - float(uv_a[0])) * u + (float(uv_b[0]) - float(uv_a[0])) * v
    vv = float(uv_a[1]) + (float(uv_c[1]) - float(uv_a[1])) * u + (float(uv_b[1]) - float(uv_a[1])) * v
    tx = np.clip(np.floor(uu * width).astype(np.int32), 0, width - 1)
    ty = np.clip(np.floor(vv * height).astype(np.int32), 0, height - 1)
    samples = texture[ty, tx]
    write = depth_mask & (samples[:, :, 3] >= 128)
    if not np.any(write):
        return
    color = np.clip(samples.astype(np.float32) * tint, 0, 255).astype(np.uint8)
    patch = image[min_y : max_y + 1, min_x : max_x + 1]
    patch[write] = color[write]
    patch_depth[write] = d[write]
