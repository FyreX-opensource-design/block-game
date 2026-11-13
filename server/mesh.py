import math
import threading
import numpy as np

from server.blocks import (
    BLOCKS,
    CHUNK_SIZE,
    FACE_UV_VECTORS,
    GLASS_CTM_UVS,
    SECTION_COUNT,
    SUB_CELL_BLOCK,
    _map_uv,
    block_from_index,
    block_palette_index,
    block_palette_indices_grid,
    get_block_ctm_uv,
    get_glass_ctm_uv,
    get_offset_info,
    get_stair_elements,
    has_half_offset_top,
    is_face_glass,
    is_glass,
    is_offset_block,
    is_stairs,
    is_sub_cell,
    is_vertical_slab_offset,
)
from server.noise import hash2, hash2_grid

_MESH_TL = threading.local()

CORNERS = np.array(
    [
        [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
        [[0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1]],
    ],
    dtype=np.float32,
)
NORMALS = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [0, 1, 0],
        [0, 1, 0],
    ],
    dtype=np.float32,
)

class ChunkMesh:
    def __init__(
        self,
        vertices,
        indices,
        cutout=None,
        translucent=None,
        lights=None,
        cutout_lights=None,
        translucent_lights=None,
        colors=None,
        cutout_colors=None,
        translucent_colors=None,
    ):
        self.vertices = vertices
        self.indices = indices
        self.cutout = cutout
        self.translucent = translucent
        self.lights = lights
        self.cutout_lights = cutout_lights
        self.translucent_lights = translucent_lights
        self.colors = colors
        self.cutout_colors = cutout_colors
        self.translucent_colors = translucent_colors


def _build_grass_grid(world, cx, cz):
    if world is None or not getattr(world, "biomes", None):
        return np.full((CHUNK_SIZE, CHUNK_SIZE, 3), (0.51, 0.72, 0.29), dtype=np.float32)
    from server.biomes import biome_index_grid
    x0 = cx * CHUNK_SIZE
    z0 = cz * CHUNK_SIZE
    xs = np.arange(x0 - 1, x0 + CHUNK_SIZE + 1)
    zs = np.arange(z0 - 1, z0 + CHUNK_SIZE + 1)
    xx, zz = np.meshgrid(xs, zs)
    biome_indices, catalog = biome_index_grid(xx, zz, getattr(world, "seed", 0), world.biomes)

    raw = np.zeros((CHUNK_SIZE + 2, CHUNK_SIZE + 2, 3), dtype=np.float32)
    raw[:] = getattr(catalog[-1], "grass_color", (0.51, 0.72, 0.29))
    for index, biome in enumerate(catalog):
        raw[biome_indices == index] = getattr(biome, "grass_color", (0.51, 0.72, 0.29))

    smoothed = (
        raw[:-2, :-2] + raw[:-2, 1:-1] + raw[:-2, 2:] +
        raw[1:-1, :-2] + raw[1:-1, 1:-1] + raw[1:-1, 2:] +
        raw[2:, :-2] + raw[2:, 1:-1] + raw[2:, 2:]
    ) * (1.0 / 9.0)
    return smoothed.astype(np.float32)


def _sample_light(world, wx, wy, wz):
    if world is None:
        return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sky_fn = getattr(world, "get_skylight", None)
    bl_fn = getattr(world, "get_blocklight", None)
    s = (sky_fn(wx, wy, wz) / 15.0) if sky_fn else 1.0
    b = bl_fn(wx, wy, wz) if bl_fn else (0, 0, 0, 0, 0)
    return (s, b[0] / 15.0, b[1] / 15.0, b[2] / 15.0, b[3] / 15.0, b[4] / 15.0)


def _build_light_grid(world, cx, cy, cz, chunk):
    grid = np.zeros((CHUNK_SIZE + 2, CHUNK_SIZE + 2, CHUNK_SIZE + 2, 6), dtype=np.float32)
    if world is None:
        grid[:, :, :, 0] = 1.0
        return grid

    lighting = getattr(world, "lighting", None)
    if lighting is not None:
        lighting.ensure_chunk_lit(chunk)

    grid[1:-1, 1:-1, 1:-1, 0] = chunk.skylight * (1.0 / 15.0)
    grid[1:-1, 1:-1, 1:-1, 1:] = chunk.blocklight * (1.0 / 15.0)

    # 6 Neighbor boundaries
    below = world.chunks.get((cx, cy - 1, cz))
    above = world.chunks.get((cx, cy + 1, cz))
    left = world.chunks.get((cx - 1, cy, cz))
    right = world.chunks.get((cx + 1, cy, cz))
    near = world.chunks.get((cx, cy, cz - 1))
    far = world.chunks.get((cx, cy, cz + 1))

    if below is not None:
        if lighting is not None and not getattr(below, "light_valid", False):
            lighting.ensure_chunk_lit(below)
        grid[0, 1:-1, 1:-1, 0] = below.skylight[-1, :, :] * (1.0 / 15.0)
        grid[0, 1:-1, 1:-1, 1:] = below.blocklight[-1, :, :] * (1.0 / 15.0)
    else:
        grid[0, 1:-1, 1:-1, :] = grid[1, 1:-1, 1:-1, :] if cy > 0 else 0.0

    if above is not None:
        if lighting is not None and not getattr(above, "light_valid", False):
            lighting.ensure_chunk_lit(above)
        grid[-1, 1:-1, 1:-1, 0] = above.skylight[0, :, :] * (1.0 / 15.0)
        grid[-1, 1:-1, 1:-1, 1:] = above.blocklight[0, :, :] * (1.0 / 15.0)
    else:
        if cy >= SECTION_COUNT - 1:
            grid[-1, 1:-1, 1:-1, 0] = 1.0
        else:
            grid[-1, 1:-1, 1:-1, :] = grid[-2, 1:-1, 1:-1, :]

    if left is not None:
        if lighting is not None and not getattr(left, "light_valid", False):
            lighting.ensure_chunk_lit(left)
        grid[1:-1, 1:-1, 0, 0] = left.skylight[:, :, -1] * (1.0 / 15.0)
        grid[1:-1, 1:-1, 0, 1:] = left.blocklight[:, :, -1] * (1.0 / 15.0)
    else:
        grid[1:-1, 1:-1, 0, :] = grid[1:-1, 1:-1, 1, :]

    if right is not None:
        if lighting is not None and not getattr(right, "light_valid", False):
            lighting.ensure_chunk_lit(right)
        grid[1:-1, 1:-1, -1, 0] = right.skylight[:, :, 0] * (1.0 / 15.0)
        grid[1:-1, 1:-1, -1, 1:] = right.blocklight[:, :, 0] * (1.0 / 15.0)
    else:
        grid[1:-1, 1:-1, -1, :] = grid[1:-1, 1:-1, -2, :]

    if near is not None:
        if lighting is not None and not getattr(near, "light_valid", False):
            lighting.ensure_chunk_lit(near)
        grid[1:-1, 0, 1:-1, 0] = near.skylight[:, -1, :] * (1.0 / 15.0)
        grid[1:-1, 0, 1:-1, 1:] = near.blocklight[:, -1, :] * (1.0 / 15.0)
    else:
        grid[1:-1, 0, 1:-1, :] = grid[1:-1, 1, 1:-1, :]

    if far is not None:
        if lighting is not None and not getattr(far, "light_valid", False):
            lighting.ensure_chunk_lit(far)
        grid[1:-1, -1, 1:-1, 0] = far.skylight[:, 0, :] * (1.0 / 15.0)
        grid[1:-1, -1, 1:-1, 1:] = far.blocklight[:, 0, :] * (1.0 / 15.0)
    else:
        grid[1:-1, -1, 1:-1, :] = grid[1:-1, -2, 1:-1, :]

    return grid


def _mask(data, cutout):
    mask = np.zeros(data.shape, dtype=bool)
    for value in np.unique(data):
        block = block_from_index(int(value))
        if block is None or block.cutout != cutout:
            continue
        if getattr(block, "translucent", False):
            continue
        if block.properties.get("fluid"):
            continue
        if not block.full_cube:
            continue
        if block.opacity < 1.0:
            continue
        mask |= data == value
    return mask


def _mask_translucent(data):
    mask = np.zeros(data.shape, dtype=bool)
    for value in np.unique(data):
        block = block_from_index(int(value))
        if block is None or not getattr(block, "translucent", False):
            continue
        if block.properties.get("fluid"):
            continue
        if not block.full_cube:
            continue
        mask |= data == value
    return mask


def _mask_glass(data):
    mask = np.zeros(data.shape, dtype=bool)
    for value in np.unique(data):
        block = block_from_index(int(value))
        if block is not None and is_glass(block) and block.full_cube and not getattr(block, "translucent", False):
            mask |= data == value
    return mask


def _fluid_height_grid(data):
    grid = np.zeros(data.shape, dtype=np.uint8)
    for value in np.unique(data):
        block = block_from_index(int(value))
        if block is None:
            continue
        if block.properties.get("waterlogged"):
            level = block.properties.get("level", 8)
            h = int(level * 2)
            if block.properties.get("vertical_slab"):
                grid[data == value] = h
                continue
            half = block.properties.get("half", getattr(block, "slab_half", "bottom"))
            if half == "top":
                h = min(8, h)
            else:
                h = max(8, h)
            grid[data == value] = h
            continue
        if not block.properties.get("fluid"):
            continue
        level = block.properties.get("level", 8)
        grid[data == value] = int(level * 2)
    return grid


def _occupancy(world, cx, cy, cz):
    chunk = world.chunks[(cx, cy, cz)]
    occ = np.zeros((CHUNK_SIZE + 2, CHUNK_SIZE + 2, CHUNK_SIZE + 2), dtype=bool)
    see = np.zeros_like(occ)
    glass = np.zeros_like(occ)
    fluid_h = np.zeros((CHUNK_SIZE + 2, CHUNK_SIZE + 2, CHUNK_SIZE + 2), dtype=np.uint8)
    occ[1:-1, 1:-1, 1:-1] = _mask(chunk.data, False)
    see[1:-1, 1:-1, 1:-1] = _mask(chunk.data, True)
    glass[1:-1, 1:-1, 1:-1] = _mask_glass(chunk.data)
    fluid_h[1:-1, 1:-1, 1:-1] = _fluid_height_grid(chunk.data)
    below = world.chunks.get((cx, cy - 1, cz))
    above = world.chunks.get((cx, cy + 1, cz))
    left = world.chunks.get((cx - 1, cy, cz))
    right = world.chunks.get((cx + 1, cy, cz))
    near = world.chunks.get((cx, cy, cz - 1))
    far = world.chunks.get((cx, cy, cz + 1))
    if cy == 0:
        occ[0, 1:-1, 1:-1] = True
    elif below is not None:
        occ[0, 1:-1, 1:-1] = _mask(below.data[-1, :, :], False)
        see[0, 1:-1, 1:-1] = _mask(below.data[-1, :, :], True)
        glass[0, 1:-1, 1:-1] = _mask_glass(below.data[-1, :, :])
        fluid_h[0, 1:-1, 1:-1] = _fluid_height_grid(below.data[-1, :, :])
    if above is not None:
        occ[-1, 1:-1, 1:-1] = _mask(above.data[0, :, :], False)
        see[-1, 1:-1, 1:-1] = _mask(above.data[0, :, :], True)
        glass[-1, 1:-1, 1:-1] = _mask_glass(above.data[0, :, :])
        fluid_h[-1, 1:-1, 1:-1] = _fluid_height_grid(above.data[0, :, :])
    if left is not None:
        occ[1:-1, 1:-1, 0] = _mask(left.data[:, :, -1], False)
        see[1:-1, 1:-1, 0] = _mask(left.data[:, :, -1], True)
        glass[1:-1, 1:-1, 0] = _mask_glass(left.data[:, :, -1])
        fluid_h[1:-1, 1:-1, 0] = _fluid_height_grid(left.data[:, :, -1])
    if right is not None:
        occ[1:-1, 1:-1, -1] = _mask(right.data[:, :, 0], False)
        see[1:-1, 1:-1, -1] = _mask(right.data[:, :, 0], True)
        glass[1:-1, 1:-1, -1] = _mask_glass(right.data[:, :, 0])
        fluid_h[1:-1, 1:-1, -1] = _fluid_height_grid(right.data[:, :, 0])
    if near is not None:
        occ[1:-1, 0, 1:-1] = _mask(near.data[:, -1, :], False)
        see[1:-1, 0, 1:-1] = _mask(near.data[:, -1, :], True)
        glass[1:-1, 0, 1:-1] = _mask_glass(near.data[:, -1, :])
        fluid_h[1:-1, 0, 1:-1] = _fluid_height_grid(near.data[:, -1, :])
    if far is not None:
        occ[1:-1, -1, 1:-1] = _mask(far.data[:, 0, :], False)
        see[1:-1, -1, 1:-1] = _mask(far.data[:, 0, :], True)
        glass[1:-1, -1, 1:-1] = _mask_glass(far.data[:, 0, :])
        fluid_h[1:-1, -1, 1:-1] = _fluid_height_grid(far.data[:, 0, :])
    return chunk, occ, see, fluid_h, glass


def _exposed(cells, occ, see=None, glass=None):
    hidden = (
        occ[1:-1, 1:-1, 2:],
        occ[1:-1, 1:-1, :-2],
        occ[2:, 1:-1, 1:-1],
        occ[:-2, 1:-1, 1:-1],
        occ[1:-1, 2:, 1:-1],
        occ[1:-1, :-2, 1:-1],
    )
    shared = (None, None, None, None, None, None)
    if see is not None:
        shared = (
            see[1:-1, 1:-1, 2:],
            see[1:-1, 1:-1, :-2],
            see[2:, 1:-1, 1:-1],
            see[:-2, 1:-1, 1:-1],
            see[1:-1, 2:, 1:-1],
            see[1:-1, :-2, 1:-1],
        )
    glass_shared = (None, None, None, None, None, None)
    glass_cells = None
    if glass is not None:
        glass_cells = glass[1:-1, 1:-1, 1:-1]
        glass_shared = (
            glass[1:-1, 1:-1, 2:],
            glass[1:-1, 1:-1, :-2],
            glass[2:, 1:-1, 1:-1],
            glass[:-2, 1:-1, 1:-1],
            glass[1:-1, 2:, 1:-1],
            glass[1:-1, :-2, 1:-1],
        )
    faces = []
    total = 0
    for normal_i, cover in enumerate(hidden):
        mask = cells & ~cover
        if shared[normal_i] is not None and normal_i in (1, 3, 5):
            mask &= ~shared[normal_i]
        if glass_shared[normal_i] is not None:
            mask &= ~(glass_cells & glass_shared[normal_i])
        ys, zs, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        faces.append((ys, zs, xs, normal_i))
        total += len(ys)
    return faces, total


def _is_opaque_offset(block_id):
    if not block_id or is_vertical_slab_offset(block_id):
        return False
    o_info = get_offset_info(block_id)
    if o_info is None:
        return False
    base_b = BLOCKS.get(o_info["base_id"])
    return base_b is not None and not base_b.cutout


def _has_opaque_half_offset_top(block_id):
    if not has_half_offset_top(block_id):
        return False
    b = BLOCKS.get(block_id)
    return b is not None and not b.cutout


def _slice_uv(uv, y_bot, y_top):
    (u0, v_bot), (u1, _), (_, v_top), _ = uv
    vb = v_bot + y_bot * (v_top - v_bot)
    vt = v_bot + y_top * (v_top - v_bot)
    return ((u0, vb), (u1, vb), (u1, vt), (u0, vt))


def octant_face_uvs(mat_block, normal_i, ox, oy, oz, off_x=None, off_y=None, off_z=None, base_uvs=None):
    if base_uvs is None:
        base_uvs = mat_block.face_uvs[normal_i]
    if base_uvs is None:
        return ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    (u0, v0), (u1, v1), (u2, v2), (u3, v3) = base_uvs
    corners = CORNERS[normal_i]
    uvs = []
    for k in range(4):
        X = (off_x + corners[k, 0] * 0.5) if off_x is not None else (ox + corners[k, 0]) * 0.5
        Y = (off_y + corners[k, 1] * 0.5) if off_y is not None else (oy + corners[k, 1]) * 0.5
        Z = (off_z + corners[k, 2] * 0.5) if off_z is not None else (oz + corners[k, 2]) * 0.5
        if normal_i == 0:
            s, t = 1.0 - Z, Y
        elif normal_i == 1:
            s, t = Z, Y
        elif normal_i == 2:
            s, t = X, 1.0 - Z
        elif normal_i == 3:
            s, t = X, Z
        elif normal_i == 4:
            s, t = X, Y
        else:
            s, t = 1.0 - X, Y
        w0 = (1.0 - s) * (1.0 - t)
        w1 = s * (1.0 - t)
        w2 = s * t
        w3 = (1.0 - s) * t
        u = w0 * u0 + w1 * u1 + w2 * u2 + w3 * u3
        v = w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3
        uvs.append((float(u), float(v)))
    return uvs


def _mesh_sub_cell(
    sub,
    lx, ly, lz,
    wx, wy, wz,
    x0, y0, z0,
    world, occ, l_grid, grass_grid, shade,
    extra_vertices, extra_indices, extra_lights, extra_colors,
    cutout_extra_vert, cutout_extra_ind, cutout_extra_light, cutout_extra_color,
    translucent_vert, translucent_ind, translucent_light, translucent_color,
    cursor_opaque, cursor_cutout, cursor_translucent,
):
    from server.sub_cell import OCTANT_COORDS, full_block_for

    for idx, mat_id in enumerate(sub.octants):
        if not mat_id:
            continue
        full_id = full_block_for(mat_id)
        mat_block = (BLOCKS.get(full_id) if full_id else None) or BLOCKS.get(mat_id)
        if mat_block is None:
            continue

        ox, oy, oz = OCTANT_COORDS[idx]
        is_translucent = bool(getattr(mat_block, "translucent", False))
        is_cutout = False if is_translucent else bool(mat_block.cutout)
        mat_is_glass = is_glass(mat_block)
        mat_glass_group = _glass_group(mat_block) if mat_is_glass else None
        is_grass_tinted = getattr(mat_block, "tint", None) == "grass" or mat_id in ("grass", "muddy_grass", "tall_grass", "fern")
        if is_grass_tinted and grass_grid is not None:
            pt_color = tuple(grass_grid[lz, lx])
        else:
            pt_color = (1.0, 1.0, 1.0)

        # Check if this octant is part of a horizontally/vertically offset block paired with an adjacent cell
        off_x = None
        off_y = None
        off_z = None

        if world is not None:
            # Z axis offset pairing
            if all(sub.get_octant(cx, cy, oz) == mat_id for cx in (0, 1) for cy in (0, 1)):
                if oz == 1:
                    adj_sub = world.get_sub_cell(wx, wy, wz + 1)
                    if adj_sub is not None and all(adj_sub.get_octant(cx, cy, 0) == mat_id for cx in (0, 1) for cy in (0, 1)):
                        off_z = 0.0
                elif oz == 0:
                    adj_sub = world.get_sub_cell(wx, wy, wz - 1)
                    if adj_sub is not None and all(adj_sub.get_octant(cx, cy, 1) == mat_id for cx in (0, 1) for cy in (0, 1)):
                        off_z = 0.5

            # X axis offset pairing
            if all(sub.get_octant(ox, cy, cz) == mat_id for cy in (0, 1) for cz in (0, 1)):
                if ox == 1:
                    adj_sub = world.get_sub_cell(wx + 1, wy, wz)
                    if adj_sub is not None and all(adj_sub.get_octant(0, cy, cz) == mat_id for cy in (0, 1) for cz in (0, 1)):
                        off_x = 0.0
                elif ox == 0:
                    adj_sub = world.get_sub_cell(wx - 1, wy, wz)
                    if adj_sub is not None and all(adj_sub.get_octant(1, cy, cz) == mat_id for cy in (0, 1) for cz in (0, 1)):
                        off_x = 0.5

            # Y axis offset pairing
            if all(sub.get_octant(cx, oy, cz) == mat_id for cx in (0, 1) for cz in (0, 1)):
                if oy == 1:
                    adj_sub = world.get_sub_cell(wx, wy + 1, wz)
                    if adj_sub is not None and all(adj_sub.get_octant(cx, 0, cz) == mat_id for cx in (0, 1) for cz in (0, 1)):
                        off_y = 0.0
                elif oy == 0:
                    adj_sub = world.get_sub_cell(wx, wy - 1, wz)
                    if adj_sub is not None and all(adj_sub.get_octant(cx, 1, cz) == mat_id for cx in (0, 1) for cz in (0, 1)):
                        off_y = 0.5

        offset_box = None
        if mat_is_glass:
            if off_z is not None:
                bx0, bx1 = float(wx), float(wx + 1)
                by0, by1 = float(wy), float(wy + 1)
                bz0, bz1 = (float(wz) + 0.5, float(wz) + 1.5) if off_z == 0.0 else (float(wz) - 0.5, float(wz) + 0.5)
                offset_box = (bx0, bx1, by0, by1, bz0, bz1)
            elif off_x is not None:
                bx0, bx1 = (float(wx) + 0.5, float(wx + 1.5)) if off_x == 0.0 else (float(wx) - 0.5, float(wx) + 0.5)
                by0, by1 = float(wy), float(wy + 1)
                bz0, bz1 = float(wz), float(wz + 1)
                offset_box = (bx0, bx1, by0, by1, bz0, bz1)
            elif off_y is not None:
                bx0, bx1 = float(wx), float(wx + 1)
                by0, by1 = (float(wy) + 0.5, float(wy + 1.5)) if off_y == 0.0 else (float(wy) - 0.5, float(wy) + 0.5)
                bz0, bz1 = float(wz), float(wz + 1)
                offset_box = (bx0, bx1, by0, by1, bz0, bz1)

        for normal_i in range(6):
            normal = NORMALS[normal_i]
            nx = int(normal[0])
            ny = int(normal[1])
            nz = int(normal[2])

            nox = ox + nx
            noy = oy + ny
            noz = oz + nz

            if 0 <= nox <= 1 and 0 <= noy <= 1 and 0 <= noz <= 1:
                adj_mat = sub.get_octant(nox, noy, noz)
                if adj_mat:
                    adj_blk = BLOCKS.get(adj_mat)
                    if adj_blk is not None:
                        if not adj_blk.cutout and not getattr(adj_blk, "translucent", False) and adj_blk.opacity >= 1.0:
                            continue
                        if (is_cutout or is_translucent) and (adj_blk.cutout or getattr(adj_blk, "translucent", False)) and (
                            adj_mat == mat_id or (mat_is_glass and is_glass(adj_blk) and _glass_group(adj_blk) == mat_glass_group)
                        ):
                            continue
                l_sample = tuple(
                    np.maximum(
                        l_grid[ly + 1, lz + 1, lx + 1],
                        l_grid[ly + 2, lz + 1, lx + 1],
                    )
                )
            else:
                if _neighbor_solid(occ, ly, lz, lx, normal_i):
                    continue

                adj_wx = wx + nx
                adj_wy = wy + ny
                adj_wz = wz + nz
                neigh_sub = world.get_sub_cell(adj_wx, adj_wy, adj_wz) if world is not None else None
                if neigh_sub is not None:
                    neigh_ox = nox % 2
                    neigh_oy = noy % 2
                    neigh_oz = noz % 2
                    neigh_mat = neigh_sub.get_octant(neigh_ox, neigh_oy, neigh_oz)
                    if neigh_mat:
                        neigh_blk = BLOCKS.get(neigh_mat)
                        if neigh_blk is not None:
                            if not neigh_blk.cutout and not getattr(neigh_blk, "translucent", False) and neigh_blk.opacity >= 1.0:
                                continue
                            if (is_cutout or is_translucent) and (neigh_blk.cutout or getattr(neigh_blk, "translucent", False)) and (
                                neigh_mat == mat_id or (mat_is_glass and is_glass(neigh_blk) and _glass_group(neigh_blk) == mat_glass_group)
                            ):
                                continue

                if is_glass(mat_block):
                    oct_el = {
                        "from": [ox * 8.0, oy * 8.0, oz * 8.0],
                        "to": [(ox + 1) * 8.0, (oy + 1) * 8.0, (oz + 1) * 8.0],
                    }
                    if _should_cull_glass_element_face(world, wx, wy, wz, mat_block, oct_el, normal_i):
                        continue

                if ny >= 0:
                    l_sample = tuple(
                        np.maximum(
                            l_grid[ly + 1 + ny, lz + 1 + nz, lx + 1 + nx],
                            l_grid[min(CHUNK_SIZE + 1, ly + 2 + ny), lz + 1 + nz, lx + 1 + nx],
                        )
                    )
                else:
                    l_sample = tuple(l_grid[ly + 1 + ny, lz + 1 + nz, lx + 1 + nx])

            base_uvs = None
            if mat_is_glass:
                comp_box = offset_box
                if comp_box is None:
                    norm_axis = 0 if nx != 0 else (1 if ny != 0 else 2)
                    t1_axis = (norm_axis + 1) % 3
                    t2_axis = (norm_axis + 2) % 3
                    ocoord = (ox, oy, oz)
                    def _same_oct(a1, a2):
                        c = [0, 0, 0]
                        c[norm_axis] = ocoord[norm_axis]
                        c[t1_axis] = a1
                        c[t2_axis] = a2
                        return sub.get_octant(c[0], c[1], c[2]) == mat_id
                    if all(_same_oct(a1, a2) for a1 in (0, 1) for a2 in (0, 1)):
                        r1 = (0.0, 1.0)
                        r2 = (0.0, 1.0)
                    elif all(_same_oct(a1, ocoord[t2_axis]) for a1 in (0, 1)):
                        r1 = (0.0, 1.0)
                        r2 = (ocoord[t2_axis] * 0.5, (ocoord[t2_axis] + 1) * 0.5)
                    elif all(_same_oct(ocoord[t1_axis], a2) for a2 in (0, 1)):
                        r1 = (ocoord[t1_axis] * 0.5, (ocoord[t1_axis] + 1) * 0.5)
                        r2 = (0.0, 1.0)
                    else:
                        r1 = (ocoord[t1_axis] * 0.5, (ocoord[t1_axis] + 1) * 0.5)
                        r2 = (ocoord[t2_axis] * 0.5, (ocoord[t2_axis] + 1) * 0.5)
                    w_base = (wx, wy, wz)
                    mins = [0.0, 0.0, 0.0]
                    maxs = [0.0, 0.0, 0.0]
                    mins[norm_axis] = w_base[norm_axis] + ocoord[norm_axis] * 0.5
                    maxs[norm_axis] = mins[norm_axis] + 0.5
                    mins[t1_axis] = w_base[t1_axis] + r1[0]
                    maxs[t1_axis] = w_base[t1_axis] + r1[1]
                    mins[t2_axis] = w_base[t2_axis] + r2[0]
                    maxs[t2_axis] = w_base[t2_axis] + r2[1]
                    comp_box = (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
                bx0, bx1, by0, by1, bz0, bz1 = comp_box
                comp_el = {
                    "from": [(bx0 - float(wx)) * 16.0, (by0 - float(wy)) * 16.0, (bz0 - float(wz)) * 16.0],
                    "to": [(bx1 - float(wx)) * 16.0, (by1 - float(wy)) * 16.0, (bz1 - float(wz)) * 16.0],
                }
                ctm_mask = _get_glass_face_ctm_mask(world, wx, wy, wz, normal_i, block=mat_block, element=comp_el)
                base_uvs = get_block_ctm_uv(mat_block, ctm_mask)

            uvs = octant_face_uvs(mat_block, normal_i, ox, oy, oz, off_x=off_x, off_y=off_y, off_z=off_z, base_uvs=base_uvs)
            corners = CORNERS[normal_i]
            pts = []
            for k in range(4):
                px = lx + (ox + corners[k, 0]) * 0.5
                py = ly + (oy + corners[k, 1]) * 0.5
                pz = lz + (oz + corners[k, 2]) * 0.5
                pts.append((px, py, pz))

            if is_translucent:
                base = cursor_translucent
                for pt, uv in zip(pts, uvs):
                    translucent_vert.extend((
                        pt[0], pt[1], pt[2],
                        normal[0], normal[1], normal[2],
                        uv[0], uv[1],
                        shade,
                    ))
                    translucent_light.append(l_sample)
                    translucent_color.append(pt_color)
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))
            elif is_cutout:
                base = cursor_cutout
                for pt, uv in zip(pts, uvs):
                    cutout_extra_vert.extend((
                        pt[0], pt[1], pt[2],
                        normal[0], normal[1], normal[2],
                        uv[0], uv[1],
                        shade,
                    ))
                    cutout_extra_light.append(l_sample)
                    cutout_extra_color.append(pt_color)
                    cursor_cutout += 1
                cutout_extra_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))
            else:
                base = cursor_opaque
                for pt, uv in zip(pts, uvs):
                    extra_vertices.extend((
                        pt[0], pt[1], pt[2],
                        normal[0], normal[1], normal[2],
                        uv[0], uv[1],
                        shade,
                    ))
                    extra_lights.append(l_sample)
                    extra_colors.append(pt_color)
                    cursor_opaque += 1
                extra_indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))

    return cursor_opaque, cursor_cutout, cursor_translucent


def build_chunk_mesh(world, cx, cy, cz):
    prev_cache = getattr(_MESH_TL, "elem_cache", None)
    prev_nearby = getattr(_MESH_TL, "nearby_cache", None)
    _MESH_TL.elem_cache = {}
    _MESH_TL.nearby_cache = {}
    try:
        return _build_chunk_mesh_inner(world, cx, cy, cz)
    finally:
        _MESH_TL.elem_cache = prev_cache
        _MESH_TL.nearby_cache = prev_nearby


def _build_chunk_mesh_inner(world, cx, cy, cz):
    if (cx, cy, cz) not in world.chunks:
        return None
    chunk, occ, see, fluid_h, glass = _occupancy(world, cx, cy, cz)
    present = [int(block_id) for block_id in np.unique(chunk.data)]
    partial_ids = [
        block_id
        for block_id in present
        if (block := block_from_index(block_id)) is not None
        and (
            not block.full_cube
            or block.properties.get("double_slab")
            or block.properties.get("double_vertical_slab")
        )
        and (not block.properties.get("fluid") or block.properties.get("waterlogged"))
    ]
    own = occ[1:-1, 1:-1, 1:-1].copy()
    own_see = see[1:-1, 1:-1, 1:-1].copy()
    own_trans = _mask_translucent(chunk.data)
    if partial_ids:
        partial_mask = np.isin(chunk.data, np.asarray(partial_ids, dtype=chunk.data.dtype))
        own[partial_mask] = False
        own_see[partial_mask] = False
        own_trans[partial_mask] = False
    faces, total = _exposed(own, occ)
    cutout_faces, cutout_total = _exposed(own_see, occ, see, glass=glass)
    trans_faces, trans_total = _exposed(own_trans, occ)
    partial_cells = (
        np.argwhere(np.isin(chunk.data, np.asarray(partial_ids, dtype=chunk.data.dtype))) if partial_ids else ()
    )
    fluid_mask = fluid_h[1:-1, 1:-1, 1:-1] > 0
    fluid_ys, fluid_zs, fluid_xs = np.nonzero(fluid_mask) if np.any(fluid_mask) else ((), (), ())

    if total == 0 and cutout_total == 0 and trans_total == 0 and len(partial_cells) == 0 and len(fluid_ys) == 0:
        return None

    l_grid = _build_light_grid(world, cx, cy, cz, chunk)
    grass_grid = _build_grass_grid(world, cx, cz)

    vertices = np.empty((total * 4, 9), dtype=np.float32)
    indices = np.empty(total * 6, dtype=np.uint32)
    lights = np.empty((total * 4, 6), dtype=np.float32)
    colors = np.ones((total * 4, 3), dtype=np.float32)
    x0 = cx * CHUNK_SIZE
    y0 = cy * CHUNK_SIZE
    z0 = cz * CHUNK_SIZE
    cursor = 0
    face_i = 0
    data = chunk.data
    for ys, zs, xs, normal_i in faces:
        ids = data[ys, zs, xs]
        normal = NORMALS[normal_i]
        corners = CORNERS[normal_i]
        nx = int(normal[0])
        ny = int(normal[1])
        nz = int(normal[2])
        for block_id in np.unique(ids):
            block = block_from_index(int(block_id))
            picked = ids == block_id
            yy = ys[picked]
            zz = zs[picked]
            xx = xs[picked]
            count = len(yy)
            shade = 0.92 + 0.08 * hash2_grid(np.int64(x0) + xx.astype(np.int64), np.int64(z0) + zz.astype(np.int64), 0)
            origin = np.stack((xx, yy, zz), axis=1).astype(np.float32)
            positions = (origin[:, None, :] + corners[None, :, :]).reshape(-1, 3)

            has_palettes = bool(getattr(block, "palette_face_uvs", None) and len(block.palette_face_uvs) > 1)
            has_ctm = bool(getattr(block, "ctm", False) and getattr(block, "palette_ctm_uvs", None) and len(block.palette_ctm_uvs) > 0)

            if has_palettes or has_ctm:
                wx = np.int64(x0) + xx.astype(np.int64)
                wy = np.int64(y0) + yy.astype(np.int64)
                wz = np.int64(z0) + zz.astype(np.int64)
                seed = world.seed if world is not None else 0
                num_p = len(block.palette_face_uvs) if getattr(block, "palette_face_uvs", None) else 1

                p_indices = block_palette_indices_grid(wx, wy, wz, seed, num_p)
                uvs = np.empty((count * 4, 2), dtype=np.float32)
                for i in range(count):
                    p_idx = int(p_indices[i])
                    if has_ctm:
                        ctm_mask = get_block_face_ctm_mask(world, int(wx[i]), int(wy[i]), int(wz[i]), normal_i, block)
                        b_uv = block.palette_ctm_uvs[p_idx][ctm_mask]
                    else:
                        b_uv = block.palette_face_uvs[p_idx][normal_i]
                    uvs[i * 4 : (i + 1) * 4] = b_uv
            else:
                uvs = np.broadcast_to(np.asarray(block.face_uvs[normal_i], dtype=np.float32), (count, 4, 2)).reshape(-1, 2)

            span = count * 4
            vertices[cursor : cursor + span, 0:3] = positions
            vertices[cursor : cursor + span, 3:6] = normal
            vertices[cursor : cursor + span, 6:8] = uvs
            vertices[cursor : cursor + span, 8] = np.repeat(shade.astype(np.float32), 4)

            face_light = l_grid[yy + 1 + ny, zz + 1 + nz, xx + 1 + nx]
            lights[cursor : cursor + span] = np.repeat(face_light, 4, axis=0)

            if getattr(block, "tint", None) == "grass" or getattr(block, "id", "") in ("grass", "muddy_grass"):
                if normal_i != 3:  # Not bottom face (which is dirt/mud)
                    face_color = grass_grid[zz, xx]
                    colors[cursor : cursor + span] = np.repeat(face_color, 4, axis=0)

            quads = np.arange(count, dtype=np.uint32) * 4 + np.uint32(cursor)
            slot = indices[face_i * 6 : (face_i + count) * 6].reshape(count, 6)
            slot[:, 0] = quads
            slot[:, 1] = quads + 1
            slot[:, 2] = quads + 2
            slot[:, 3] = quads
            slot[:, 4] = quads + 2
            slot[:, 5] = quads + 3
            cursor += span
            face_i += count
    cutout = (
        _emit_cutout(world, chunk.data, cutout_faces, cutout_total, x0, y0, z0, l_grid, grass_grid)
        if cutout_total
        else None
    )
    translucent_base = (
        _emit_cutout(world, chunk.data, trans_faces, trans_total, x0, y0, z0, l_grid, grass_grid)
        if trans_total
        else None
    )
    if translucent_base is not None and len(translucent_base[0]) == 0:
        translucent_base = None

    extra_vertices = []
    extra_indices = []
    extra_lights = []
    extra_colors = []
    cutout_extra_vert = []
    cutout_extra_ind = []
    cutout_extra_light = []
    cutout_extra_color = []
    translucent_vert = []
    translucent_ind = []
    translucent_light = []
    translucent_color = []
    cursor_opaque = total * 4
    cursor_cutout = len(cutout[0]) if cutout is not None else 0
    cursor_translucent = len(translucent_base[0]) if translucent_base is not None else 0

    from server.blocks import (
        has_half_offset_top,
        is_offset_block,
        is_vertical_slab_offset,
    )

    for y, z, x in partial_cells:
        block = block_from_index(data[int(y), int(z), int(x)])
        if block is None:
            continue
        shade = 0.92 + 0.08 * hash2(x0 + int(x), z0 + int(z))
        wx = x0 + int(x)
        wy = y0 + int(y)
        wz = z0 + int(z)
        if block.id == SUB_CELL_BLOCK or is_sub_cell(block):
            sub = chunk.sub_cells.get((int(x), int(y), int(z)))
            if sub is None and world is not None:
                sub = world.get_sub_cell(wx, wy, wz)
            if sub is not None:
                cursor_opaque, cursor_cutout, cursor_translucent = _mesh_sub_cell(
                    sub,
                    int(x),
                    int(y),
                    int(z),
                    wx,
                    wy,
                    wz,
                    x0,
                    y0,
                    z0,
                    world,
                    occ,
                    l_grid,
                    grass_grid,
                    shade,
                    extra_vertices,
                    extra_indices,
                    extra_lights,
                    extra_colors,
                    cutout_extra_vert,
                    cutout_extra_ind,
                    cutout_extra_light,
                    cutout_extra_color,
                    translucent_vert,
                    translucent_ind,
                    translucent_light,
                    translucent_color,
                    cursor_opaque,
                    cursor_cutout,
                    cursor_translucent,
                )
            continue
        from server.blocks import (
            is_fence,
            get_fence_elements,
            is_wall,
            get_wall_elements,
            is_stairs,
            get_stair_elements,
            is_dripstone,
            get_dripstone_elements,
        )
        if is_fence(block):
            elements = get_fence_elements(world, wx, wy, wz, block)
        elif is_wall(block):
            elements = get_wall_elements(world, wx, wy, wz, block)
        elif is_stairs(block):
            elements = get_stair_elements(world, wx, wy, wz, block)
        elif is_dripstone(block):
            elements = get_dripstone_elements(world, wx, wy, wz, block)
        else:
            elements = block.elements
        for element in elements:
            el_tint = element.get("tint", getattr(block, "tint", None))
            is_grass_tinted = el_tint == "grass" or getattr(block, "id", "") in ("tall_grass", "fern")
            from_y = float(element["from"][1])
            to_y = float(element["to"][1])
            for normal_i, uvs in element["faces"].items():
                if is_grass_tinted and (normal_i != 3 or getattr(block, "id", "") in ("tall_grass", "fern")):
                    pt_color = tuple(grass_grid[int(z), int(x)])
                else:
                    pt_color = (1.0, 1.0, 1.0)

                el_translucent = (
                    bool(element["translucent"])
                    if "translucent" in element
                    else bool(getattr(block, "translucent", False))
                )
                el_cutout = False if el_translucent else element.get("cutout", block.cutout)
                el_is_glass = element.get("glass") if "glass" in element else is_glass(block)
                ctm_blk = (BLOCKS.get(element["source_block"]) if "source_block" in element else None) or block
                if from_y < -1e-3:
                    if el_is_glass and _should_cull_glass_element_face(world, wx, wy, wz, ctm_blk, element, normal_i):
                        continue
                    nx = int(NORMALS[normal_i, 0])
                    nz = int(NORMALS[normal_i, 2])
                    if normal_i == 2:
                        above_b = world.get_block(wx, wy + 1, wz) if world is not None else None
                        if _is_opaque_offset(above_b):
                            continue
                        l_sample = tuple(
                            np.maximum(
                                l_grid[int(y) + 1, int(z) + 1, int(x) + 1],
                                l_grid[int(y) + 2, int(z) + 1, int(x) + 1],
                            )
                        )
                    elif normal_i == 3:
                        below_b = world.get_block(wx, wy - 1, wz) if world is not None else None
                        if _has_opaque_half_offset_top(below_b) or (el_cutout and has_half_offset_top(below_b)):
                            continue
                        l_sample = tuple(l_grid[int(y), int(z) + 1, int(x) + 1])
                    else:
                        if not _boundary(element, normal_i):
                            l_sample = tuple(
                                np.maximum(
                                    l_grid[int(y) + 1, int(z) + 1, int(x) + 1],
                                    l_grid[int(y), int(z) + 1, int(x) + 1],
                                )
                            )
                        else:
                            nbr_b = world.get_block(wx + nx, wy, wz + nz) if world is not None else None
                            if _is_opaque_offset(nbr_b) or (not el_is_glass and el_cutout and is_offset_block(nbr_b) and not is_vertical_slab_offset(nbr_b) and normal_i in (1, 5)):
                                continue
                            if (
                                _neighbor_solid(occ, int(y), int(z), int(x), normal_i)
                                and int(y) > 0
                                and _neighbor_solid(occ, int(y) - 1, int(z), int(x), normal_i)
                            ):
                                continue
                            l_sample = tuple(
                                np.maximum(
                                    l_grid[int(y) + 1, int(z) + 1 + nz, int(x) + 1 + nx],
                                    l_grid[int(y), int(z) + 1 + nz, int(x) + 1 + nx],
                                )
                            )
                else:
                    if normal_i == 2 and abs(to_y - 8.0) <= 1e-3:
                        above_b = world.get_block(wx, wy + 1, wz) if world is not None else None
                        if _is_opaque_offset(above_b):
                            continue
                    if el_is_glass and _should_cull_glass_element_face(world, wx, wy, wz, ctm_blk, element, normal_i):
                        continue
                    if _boundary(element, normal_i) and _neighbor_solid(occ, int(y), int(z), int(x), normal_i):
                        continue
                    if _boundary(element, normal_i):
                        l_sample = tuple(
                            l_grid[
                                int(y) + 1 + int(NORMALS[normal_i, 1]),
                                int(z) + 1 + int(NORMALS[normal_i, 2]),
                                int(x) + 1 + int(NORMALS[normal_i, 0]),
                            ]
                        )
                    else:
                        l_sample = tuple(
                            l_grid[
                                int(y) + 1,
                                int(z) + 1,
                                int(x) + 1,
                            ]
                        )
                base_spec = element.get("face_uv_specs", {}).get(normal_i, [0, 0, 16, 16])
                emit_list = []
                if el_is_glass and GLASS_CTM_UVS:
                    parts = _resolve_glass_face_parts(world, wx, wy, wz, ctm_blk, element, normal_i)
                    if not parts:
                        continue
                    if len(parts) == 1 and parts[0][:4] == (0.0, 1.0, 0.0, 1.0):
                        m = parts[0][4]
                        emit_list.append((element, get_block_ctm_uv(ctm_blk, m, base_spec) or uvs))
                    else:
                        for ua, ub, va, vb, pm in parts:
                            sub_el = _get_element_sub_box(element, normal_i, ua, ub, va, vb)
                            m = pm if pm is not None else _get_glass_face_ctm_mask(world, wx, wy, wz, normal_i, block=ctm_blk, element=sub_el)
                            spec_uv = _slice_spec_uv(base_spec, ua, ub, va, vb)
                            sub_uvs = get_block_ctm_uv(ctm_blk, m, spec_uv) or uvs
                            emit_list.append((sub_el, sub_uvs))
                else:
                    if getattr(ctm_blk, "ctm", False) and getattr(ctm_blk, "palette_ctm_uvs", None) and len(ctm_blk.palette_ctm_uvs) > 0:
                        seed = world.seed if world is not None else 0
                        num_p = len(ctm_blk.palette_face_uvs) if getattr(ctm_blk, "palette_face_uvs", None) else 1
                        p_idx = block_palette_index(wx, wy, wz, seed, num_p)
                        m = get_block_face_ctm_mask(world, wx, wy, wz, normal_i, block=ctm_blk, element=element)
                        rect = getattr(ctm_blk, "palette_ctm_rects", {}).get((p_idx, m))
                        if rect is not None:
                            uvs = _map_uv(rect, base_spec)
                        elif ctm_blk.palette_ctm_uvs and m < len(ctm_blk.palette_ctm_uvs[p_idx]):
                            uvs = ctm_blk.palette_ctm_uvs[p_idx][m]
                    elif getattr(ctm_blk, "palette_face_uvs", None) and len(ctm_blk.palette_face_uvs) > 1:
                        seed = world.seed if world is not None else 0
                        num_p = len(ctm_blk.palette_face_uvs)
                        p_idx = block_palette_index(wx, wy, wz, seed, num_p)
                        rect = getattr(ctm_blk, "palette_texture_rects", {}).get((p_idx, normal_i))
                        if rect is not None:
                            uvs = _map_uv(rect, base_spec)
                        else:
                            uvs = ctm_blk.palette_face_uvs[p_idx][normal_i]
                    emit_list.append((element, uvs))

                for emit_el, emit_uvs in emit_list:
                    points = _element_corners(emit_el, normal_i)
                    if el_translucent:
                        base = cursor_translucent
                        target_vert, target_ind, target_light, target_color = (
                            translucent_vert, translucent_ind, translucent_light, translucent_color
                        )
                        cursor_translucent += 4
                    elif el_cutout:
                        base = cursor_cutout
                        target_vert, target_ind, target_light, target_color = (
                            cutout_extra_vert, cutout_extra_ind, cutout_extra_light, cutout_extra_color
                        )
                        cursor_cutout += 4
                    else:
                        base = cursor_opaque
                        target_vert, target_ind, target_light, target_color = (
                            extra_vertices, extra_indices, extra_lights, extra_colors
                        )
                        cursor_opaque += 4
                    for corner, uv in zip(points, emit_uvs):
                        target_vert.extend(
                            (
                                int(x) + corner[0],
                                int(y) + corner[1],
                                int(z) + corner[2],
                                NORMALS[normal_i, 0],
                                NORMALS[normal_i, 1],
                                NORMALS[normal_i, 2],
                                uv[0],
                                uv[1],
                                shade,
                            )
                        )
                        target_light.append(l_sample)
                        target_color.append(pt_color)
                    target_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

    water_block = BLOCKS.get("water")

    for y, z, x in zip(fluid_ys, fluid_zs, fluid_xs):
        cur_h = int(fluid_h[y + 1, z + 1, x + 1])
        cur_top = cur_h / 16.0
        block = block_from_index(data[y, z, x])
        if block is None:
            continue
        shade = 0.92 + 0.08 * hash2(x0 + int(x), z0 + int(z))
        is_slab = bool(
            block.properties.get("slab")
            or getattr(block, "slab", False)
            or block.properties.get("waterlogged")
        )
        fluid_uvs = (water_block.face_uvs if (is_slab and water_block is not None) else block.face_uvs)
        slab_half = block.properties.get("half", getattr(block, "slab_half", None)) if is_slab else None

        if slab_half == "bottom":
            cur_base = 0.5
            cur_top = max(0.5, cur_top)
            # 1. Top face (normal_i = 2, Up, [0, 1, 0])
            above_h = fluid_h[y + 2, z + 1, x + 1]
            if above_h == 0 and cur_top > cur_base:
                base = cursor_translucent
                pts = (
                    (int(x), int(y) + cur_top, int(z) + 1),
                    (int(x) + 1, int(y) + cur_top, int(z) + 1),
                    (int(x) + 1, int(y) + cur_top, int(z)),
                    (int(x), int(y) + cur_top, int(z)),
                )
                l_sample = tuple(l_grid[y + 2, z + 1, x + 1])
                for pt, uv in zip(pts, fluid_uvs[2]):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 0.0, 1.0, 0.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # Bottom face of fluid touches top of bottom slab at y + 0.5, so culled.

            # Side faces (y_bot is at least 0.5)
            # 3. East face (normal_i = 0, +X, [1, 0, 0])
            nbr_occ = occ[y + 1, z + 1, x + 2]
            nbr_h = int(fluid_h[y + 1, z + 1, x + 2])
            y_bot = max(0.5, nbr_h / 16.0)
            if not nbr_occ and y_bot < cur_top:
                base = cursor_translucent
                pts = (
                    (int(x) + 1, int(y) + y_bot, int(z) + 1),
                    (int(x) + 1, int(y) + y_bot, int(z)),
                    (int(x) + 1, int(y) + cur_top, int(z)),
                    (int(x) + 1, int(y) + cur_top, int(z) + 1),
                )
                uvs = _slice_uv(fluid_uvs[0], y_bot, cur_top)
                l_sample = tuple(l_grid[y + 1, z + 1, x + 2])
                for pt, uv in zip(pts, uvs):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 1.0, 0.0, 0.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # 4. West face (normal_i = 1, -X, [-1, 0, 0])
            nbr_occ = occ[y + 1, z + 1, x]
            nbr_h = int(fluid_h[y + 1, z + 1, x])
            y_bot = max(0.5, nbr_h / 16.0)
            if not nbr_occ and y_bot < cur_top:
                base = cursor_translucent
                pts = (
                    (int(x), int(y) + y_bot, int(z)),
                    (int(x), int(y) + y_bot, int(z) + 1),
                    (int(x), int(y) + cur_top, int(z) + 1),
                    (int(x), int(y) + cur_top, int(z)),
                )
                uvs = _slice_uv(fluid_uvs[1], y_bot, cur_top)
                l_sample = tuple(l_grid[y + 1, z + 1, x])
                for pt, uv in zip(pts, uvs):
                    translucent_vert.extend((pt[0], pt[1], pt[2], -1.0, 0.0, 0.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # 5. South face (normal_i = 4, +Z, [0, 0, 1])
            nbr_occ = occ[y + 1, z + 2, x + 1]
            nbr_h = int(fluid_h[y + 1, z + 2, x + 1])
            y_bot = max(0.5, nbr_h / 16.0)
            if not nbr_occ and y_bot < cur_top:
                base = cursor_translucent
                pts = (
                    (int(x), int(y) + y_bot, int(z) + 1),
                    (int(x) + 1, int(y) + y_bot, int(z) + 1),
                    (int(x) + 1, int(y) + cur_top, int(z) + 1),
                    (int(x), int(y) + cur_top, int(z) + 1),
                )
                uvs = _slice_uv(fluid_uvs[4], y_bot, cur_top)
                l_sample = tuple(l_grid[y + 1, z + 2, x + 1])
                for pt, uv in zip(pts, uvs):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 0.0, 0.0, 1.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # 6. North face (normal_i = 5, -Z, [0, 0, -1])
            nbr_occ = occ[y + 1, z, x + 1]
            nbr_h = int(fluid_h[y + 1, z, x + 1])
            y_bot = max(0.5, nbr_h / 16.0)
            if not nbr_occ and y_bot < cur_top:
                base = cursor_translucent
                pts = (
                    (int(x) + 1, int(y) + y_bot, int(z)),
                    (int(x), int(y) + y_bot, int(z)),
                    (int(x), int(y) + cur_top, int(z)),
                    (int(x) + 1, int(y) + cur_top, int(z)),
                )
                uvs = _slice_uv(fluid_uvs[5], y_bot, cur_top)
                l_sample = tuple(l_grid[y + 1, z, x + 1])
                for pt, uv in zip(pts, uvs):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 0.0, 0.0, -1.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

        elif slab_half == "top":
            cur_base = 0.0
            cur_top = min(0.5, cur_top)
            # 1. Top face (normal_i = 2, Up, [0, 1, 0])
            if cur_top < 0.5 and cur_top > 0.0:
                base = cursor_translucent
                pts = (
                    (int(x), int(y) + cur_top, int(z) + 1),
                    (int(x) + 1, int(y) + cur_top, int(z) + 1),
                    (int(x) + 1, int(y) + cur_top, int(z)),
                    (int(x), int(y) + cur_top, int(z)),
                )
                l_sample = tuple(l_grid[y + 1, z + 1, x + 1])
                for pt, uv in zip(pts, fluid_uvs[2]):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 0.0, 1.0, 0.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # 2. Bottom face (normal_i = 3, Down, [0, -1, 0])
            below_occ = occ[y, z + 1, x + 1]
            below_h = fluid_h[y, z + 1, x + 1]
            if not below_occ and below_h == 0 and cur_top > 0.0:
                base = cursor_translucent
                pts = (
                    (int(x), int(y), int(z)),
                    (int(x) + 1, int(y), int(z)),
                    (int(x) + 1, int(y), int(z) + 1),
                    (int(x), int(y), int(z) + 1),
                )
                l_sample = tuple(l_grid[y, z + 1, x + 1])
                for pt, uv in zip(pts, fluid_uvs[3]):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 0.0, -1.0, 0.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # Side faces
            # 3. East face (normal_i = 0, +X, [1, 0, 0])
            nbr_occ = occ[y + 1, z + 1, x + 2]
            nbr_h = int(fluid_h[y + 1, z + 1, x + 2])
            y_bot = min(cur_top, nbr_h / 16.0)
            if not nbr_occ and y_bot < cur_top:
                base = cursor_translucent
                pts = (
                    (int(x) + 1, int(y) + y_bot, int(z) + 1),
                    (int(x) + 1, int(y) + y_bot, int(z)),
                    (int(x) + 1, int(y) + cur_top, int(z)),
                    (int(x) + 1, int(y) + cur_top, int(z) + 1),
                )
                uvs = _slice_uv(fluid_uvs[0], y_bot, cur_top)
                l_sample = tuple(l_grid[y + 1, z + 1, x + 2])
                for pt, uv in zip(pts, uvs):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 1.0, 0.0, 0.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # 4. West face (normal_i = 1, -X, [-1, 0, 0])
            nbr_occ = occ[y + 1, z + 1, x]
            nbr_h = int(fluid_h[y + 1, z + 1, x])
            y_bot = min(cur_top, nbr_h / 16.0)
            if not nbr_occ and y_bot < cur_top:
                base = cursor_translucent
                pts = (
                    (int(x), int(y) + y_bot, int(z)),
                    (int(x), int(y) + y_bot, int(z) + 1),
                    (int(x), int(y) + cur_top, int(z) + 1),
                    (int(x), int(y) + cur_top, int(z)),
                )
                uvs = _slice_uv(fluid_uvs[1], y_bot, cur_top)
                l_sample = tuple(l_grid[y + 1, z + 1, x])
                for pt, uv in zip(pts, uvs):
                    translucent_vert.extend((pt[0], pt[1], pt[2], -1.0, 0.0, 0.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # 5. South face (normal_i = 4, +Z, [0, 0, 1])
            nbr_occ = occ[y + 1, z + 2, x + 1]
            nbr_h = int(fluid_h[y + 1, z + 2, x + 1])
            y_bot = min(cur_top, nbr_h / 16.0)
            if not nbr_occ and y_bot < cur_top:
                base = cursor_translucent
                pts = (
                    (int(x), int(y) + y_bot, int(z) + 1),
                    (int(x) + 1, int(y) + y_bot, int(z) + 1),
                    (int(x) + 1, int(y) + cur_top, int(z) + 1),
                    (int(x), int(y) + cur_top, int(z) + 1),
                )
                uvs = _slice_uv(fluid_uvs[4], y_bot, cur_top)
                l_sample = tuple(l_grid[y + 1, z + 2, x + 1])
                for pt, uv in zip(pts, uvs):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 0.0, 0.0, 1.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # 6. North face (normal_i = 5, -Z, [0, 0, -1])
            nbr_occ = occ[y + 1, z, x + 1]
            nbr_h = int(fluid_h[y + 1, z, x + 1])
            y_bot = min(cur_top, nbr_h / 16.0)
            if not nbr_occ and y_bot < cur_top:
                base = cursor_translucent
                pts = (
                    (int(x) + 1, int(y) + y_bot, int(z)),
                    (int(x), int(y) + y_bot, int(z)),
                    (int(x), int(y) + cur_top, int(z)),
                    (int(x) + 1, int(y) + cur_top, int(z)),
                )
                uvs = _slice_uv(fluid_uvs[5], y_bot, cur_top)
                l_sample = tuple(l_grid[y + 1, z, x + 1])
                for pt, uv in zip(pts, uvs):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 0.0, 0.0, -1.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

        else:
            # 1. Top face (normal_i = 2, Up, [0, 1, 0])
            above_h = fluid_h[y + 2, z + 1, x + 1]
            if above_h == 0:
                base = cursor_translucent
                pts = (
                    (int(x), int(y) + cur_top, int(z) + 1),
                    (int(x) + 1, int(y) + cur_top, int(z) + 1),
                    (int(x) + 1, int(y) + cur_top, int(z)),
                    (int(x), int(y) + cur_top, int(z)),
                )
                l_sample = tuple(l_grid[y + 2, z + 1, x + 1])
                for pt, uv in zip(pts, fluid_uvs[2]):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 0.0, 1.0, 0.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # 2. Bottom face (normal_i = 3, Down, [0, -1, 0])
            below_occ = occ[y, z + 1, x + 1]
            below_h = fluid_h[y, z + 1, x + 1]
            if not below_occ and below_h == 0:
                base = cursor_translucent
                pts = (
                    (int(x), int(y), int(z)),
                    (int(x) + 1, int(y), int(z)),
                    (int(x) + 1, int(y), int(z) + 1),
                    (int(x), int(y), int(z) + 1),
                )
                l_sample = tuple(l_grid[y, z + 1, x + 1])
                for pt, uv in zip(pts, fluid_uvs[3]):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 0.0, -1.0, 0.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # 3. East face (normal_i = 0, +X, [1, 0, 0])
            nbr_occ = occ[y + 1, z + 1, x + 2]
            nbr_h = int(fluid_h[y + 1, z + 1, x + 2])
            if not nbr_occ and nbr_h < cur_h:
                y_bot = nbr_h / 16.0
                base = cursor_translucent
                pts = (
                    (int(x) + 1, int(y) + y_bot, int(z) + 1),
                    (int(x) + 1, int(y) + y_bot, int(z)),
                    (int(x) + 1, int(y) + cur_top, int(z)),
                    (int(x) + 1, int(y) + cur_top, int(z) + 1),
                )
                uvs = _slice_uv(fluid_uvs[0], y_bot, cur_top)
                l_sample = tuple(l_grid[y + 1, z + 1, x + 2])
                for pt, uv in zip(pts, uvs):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 1.0, 0.0, 0.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # 4. West face (normal_i = 1, -X, [-1, 0, 0])
            nbr_occ = occ[y + 1, z + 1, x]
            nbr_h = int(fluid_h[y + 1, z + 1, x])
            if not nbr_occ and nbr_h < cur_h:
                y_bot = nbr_h / 16.0
                base = cursor_translucent
                pts = (
                    (int(x), int(y) + y_bot, int(z)),
                    (int(x), int(y) + y_bot, int(z) + 1),
                    (int(x), int(y) + cur_top, int(z) + 1),
                    (int(x), int(y) + cur_top, int(z)),
                )
                uvs = _slice_uv(fluid_uvs[1], y_bot, cur_top)
                l_sample = tuple(l_grid[y + 1, z + 1, x])
                for pt, uv in zip(pts, uvs):
                    translucent_vert.extend((pt[0], pt[1], pt[2], -1.0, 0.0, 0.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # 5. South face (normal_i = 4, +Z, [0, 0, 1])
            nbr_occ = occ[y + 1, z + 2, x + 1]
            nbr_h = int(fluid_h[y + 1, z + 2, x + 1])
            if not nbr_occ and nbr_h < cur_h:
                y_bot = nbr_h / 16.0
                base = cursor_translucent
                pts = (
                    (int(x), int(y) + y_bot, int(z) + 1),
                    (int(x) + 1, int(y) + y_bot, int(z) + 1),
                    (int(x) + 1, int(y) + cur_top, int(z) + 1),
                    (int(x), int(y) + cur_top, int(z) + 1),
                )
                uvs = _slice_uv(fluid_uvs[4], y_bot, cur_top)
                l_sample = tuple(l_grid[y + 1, z + 2, x + 1])
                for pt, uv in zip(pts, uvs):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 0.0, 0.0, 1.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

            # 6. North face (normal_i = 5, -Z, [0, 0, -1])
            nbr_occ = occ[y + 1, z, x + 1]
            nbr_h = int(fluid_h[y + 1, z, x + 1])
            if not nbr_occ and nbr_h < cur_h:
                y_bot = nbr_h / 16.0
                base = cursor_translucent
                pts = (
                    (int(x) + 1, int(y) + y_bot, int(z)),
                    (int(x), int(y) + y_bot, int(z)),
                    (int(x), int(y) + cur_top, int(z)),
                    (int(x) + 1, int(y) + cur_top, int(z)),
                )
                uvs = _slice_uv(fluid_uvs[5], y_bot, cur_top)
                l_sample = tuple(l_grid[y + 1, z, x + 1])
                for pt, uv in zip(pts, uvs):
                    translucent_vert.extend((pt[0], pt[1], pt[2], 0.0, 0.0, -1.0, uv[0], uv[1], shade))
                    translucent_light.append(l_sample)
                    translucent_color.append((1.0, 1.0, 1.0))
                    cursor_translucent += 1
                translucent_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))

    if cutout_extra_vert:
        cev = np.asarray(cutout_extra_vert, dtype=np.float32).reshape(-1, 9)
        cei = np.asarray(cutout_extra_ind, dtype=np.uint32)
        cel = np.asarray(cutout_extra_light, dtype=np.float32).reshape(-1, 6)
        cec = np.asarray(cutout_extra_color, dtype=np.float32).reshape(-1, 3)
        if cutout is not None:
            cutout_v = np.vstack((cutout[0], cev))
            cutout_i = np.concatenate((cutout[1], cei))
            cutout_l = np.vstack((cutout[2], cel))
            cutout_c = np.vstack((cutout[3], cec))
        else:
            cutout_v = cev
            cutout_i = cei
            cutout_l = cel
            cutout_c = cec
        cutout = (cutout_v, cutout_i)
        cutout_lights = cutout_l
        cutout_colors = cutout_c
    elif cutout is not None:
        cutout_lights = cutout[2]
        cutout_colors = cutout[3]
        cutout = (cutout[0], cutout[1])
    else:
        cutout_lights = None
        cutout_colors = None

    translucent = None
    translucent_lights = None
    translucent_colors = None
    if translucent_vert:
        tv = np.asarray(translucent_vert, dtype=np.float32).reshape(-1, 9)
        ti = np.asarray(translucent_ind, dtype=np.uint32)
        tl = np.asarray(translucent_light, dtype=np.float32).reshape(-1, 6)
        tc = np.asarray(translucent_color, dtype=np.float32).reshape(-1, 3)
        if translucent_base is not None:
            translucent = (np.vstack((translucent_base[0], tv)), np.concatenate((translucent_base[1], ti)))
            translucent_lights = np.vstack((translucent_base[2], tl))
            translucent_colors = np.vstack((translucent_base[3], tc))
        else:
            translucent = (tv, ti)
            translucent_lights = tl
            translucent_colors = tc
    elif translucent_base is not None:
        translucent = (translucent_base[0], translucent_base[1])
        translucent_lights = translucent_base[2]
        translucent_colors = translucent_base[3]

    if extra_vertices:
        vertices = np.vstack((vertices, np.asarray(extra_vertices, dtype=np.float32).reshape(-1, 9)))
        indices = np.concatenate((indices, np.asarray(extra_indices, dtype=np.uint32)))
        lights = np.vstack((lights, np.asarray(extra_lights, dtype=np.float32).reshape(-1, 6)))
        colors = np.vstack((colors, np.asarray(extra_colors, dtype=np.float32).reshape(-1, 3)))

    if total == 0 and not extra_vertices and cutout is None and translucent is None:
        return None
    return ChunkMesh(
        vertices,
        indices,
        cutout=cutout,
        translucent=translucent,
        lights=lights,
        cutout_lights=cutout_lights,
        translucent_lights=translucent_lights,
        colors=colors,
        cutout_colors=cutout_colors,
        translucent_colors=translucent_colors,
    )


def _get_world_elements(world, bx, by, bz):
    if world is None:
        return ()
    cache = getattr(_MESH_TL, "elem_cache", None)
    if cache is not None:
        c_key = (id(world), bx, by, bz)
        cached = cache.get(c_key)
        if cached is not None:
            return cached
    block = world.get_block(bx, by, bz)
    if block is None or block == "air":
        if cache is not None:
            cache[c_key] = ()
        return ()
    b_obj = BLOCKS.get(block) if isinstance(block, str) else block
    if b_obj is None:
        if cache is not None:
            cache[c_key] = ()
        return ()
    if getattr(b_obj, "full_cube", False) and not b_obj.properties.get("double_slab") and not b_obj.properties.get("double_vertical_slab"):
        is_gl = is_glass(b_obj)
        is_op = b_obj.solid and not b_obj.cutout and b_obj.opacity >= 1.0
        res = ({
            "box": (float(bx), float(bx + 1), float(by), float(by + 1), float(bz), float(bz + 1)),
            "faces": {0, 1, 2, 3, 4, 5},
            "glass": is_gl,
            "glass_group": _glass_group(b_obj) if is_gl else None,
            "opaque": is_op,
            "block": b_obj,
            "element": None,
        },)
        if cache is not None:
            cache[c_key] = res
        return res
    if block == SUB_CELL_BLOCK or is_sub_cell(block):
        sub = world.get_sub_cell(bx, by, bz)
        if sub is None:
            if cache is not None:
                cache[c_key] = ()
            return ()
        from server.sub_cell import OCTANT_COORDS
        results = []
        for idx, mat_id in enumerate(sub.octants):
            if not mat_id:
                continue
            mat_blk = BLOCKS.get(mat_id)
            if mat_blk is None:
                continue
            ox, oy, oz = OCTANT_COORDS[idx]
            x0 = bx + ox * 0.5
            x1 = x0 + 0.5
            y0 = by + oy * 0.5
            y1 = y0 + 0.5
            z0 = bz + oz * 0.5
            z1 = z0 + 0.5
            is_gl = is_glass(mat_blk)
            is_op = bool(not mat_blk.cutout and not is_gl and getattr(mat_blk, "solid", False) and mat_blk.opacity >= 1.0)
            faces = set()
            for normal_i in range(6):
                nx = int(NORMALS[normal_i, 0])
                ny = int(NORMALS[normal_i, 1])
                nz = int(NORMALS[normal_i, 2])
                nox = ox + nx
                noy = oy + ny
                noz = oz + nz
                if 0 <= nox <= 1 and 0 <= noy <= 1 and 0 <= noz <= 1:
                    adj_mat = sub.get_octant(nox, noy, noz)
                    if adj_mat:
                        adj_b = BLOCKS.get(adj_mat)
                        if adj_b and not adj_b.cutout and not getattr(adj_b, "translucent", False) and not is_glass(adj_b) and adj_b.opacity >= 1.0:
                            continue
                        if is_gl and is_glass(adj_b) and _glass_group(adj_b) == _glass_group(mat_blk):
                            continue
                else:
                    adj_sub = world.get_sub_cell(bx + nx, by + ny, bz + nz)
                    if adj_sub is not None:
                        adj_mat = adj_sub.get_octant(nox % 2, noy % 2, noz % 2)
                        if adj_mat:
                            adj_b = BLOCKS.get(adj_mat)
                            if adj_b and not adj_b.cutout and not getattr(adj_b, "translucent", False) and not is_glass(adj_b) and adj_b.opacity >= 1.0:
                                continue
                            if is_gl and is_glass(adj_b) and _glass_group(adj_b) == _glass_group(mat_blk):
                                continue
                    else:
                        adj_block = world.get_block(bx + nx, by + ny, bz + nz)
                        if adj_block is not None and adj_block != "air":
                            adj_b = BLOCKS.get(adj_block) if isinstance(adj_block, str) else adj_block
                            if adj_b is not None and getattr(adj_b, "full_cube", False):
                                if not adj_b.cutout and not getattr(adj_b, "translucent", False) and not is_glass(adj_b) and adj_b.opacity >= 1.0:
                                    continue
                                if is_gl and is_glass(adj_b) and _glass_group(adj_b) == _glass_group(mat_blk):
                                    continue
                faces.add(normal_i)
            if faces:
                results.append({
                    "box": (x0, x1, y0, y1, z0, z1),
                    "faces": faces,
                    "glass": is_gl,
                    "glass_group": _glass_group(mat_blk) if is_gl else None,
                    "opaque": is_op,
                    "block": mat_blk,
                    "element": None,
                })
        res = tuple(results)
        if cache is not None:
            cache[c_key] = res
        return res
    b_glass = is_glass(b_obj)
    results = []
    els = get_stair_elements(world, bx, by, bz, b_obj) if is_stairs(b_obj) else getattr(b_obj, "elements", ())
    for el in els:
        x0 = bx + el["from"][0] / 16.0
        x1 = bx + el["to"][0] / 16.0
        y0 = by + el["from"][1] / 16.0
        y1 = by + el["to"][1] / 16.0
        z0 = bz + el["from"][2] / 16.0
        z1 = bz + el["to"][2] / 16.0
        el_blk = (BLOCKS.get(el["source_block"]) if "source_block" in el else None) or b_obj
        is_gl = bool(el.get("glass") or (b_glass if "glass" not in el else False))
        is_op = bool(not el.get("cutout") and not el.get("translucent") and not is_gl and getattr(el_blk, "solid", False) and getattr(el_blk, "opacity", 1.0) >= 1.0)
        results.append({
            "box": (x0, x1, y0, y1, z0, z1),
            "faces": set(el.get("faces", {}).keys()),
            "glass": is_gl,
            "glass_group": _glass_group(el_blk) if is_gl else None,
            "opaque": is_op,
            "block": el_blk,
            "element": el,
        })
    res = tuple(results)
    if cache is not None:
        cache[c_key] = res
    return res


def _query_nearby_elements(world, px, py, pz):
    bx = int(math.floor(px))
    by = int(math.floor(py))
    bz = int(math.floor(pz))
    dx_dir = -1 if px - bx < 0.05 else (1 if (bx + 1) - px < 0.05 else 0)
    dz_dir = -1 if pz - bz < 0.05 else (1 if (bz + 1) - pz < 0.05 else 0)
    nb_cache = getattr(_MESH_TL, "nearby_cache", None)
    if nb_cache is not None:
        n_key = (id(world), bx, by, bz, dx_dir, dz_dir)
        cached = nb_cache.get(n_key)
        if cached is not None:
            return cached
    els = []
    for dy in (0, 1, -1):
        sub = _get_world_elements(world, bx, by + dy, bz)
        if sub:
            els.extend(sub)
    if dx_dir:
        tx = bx + dx_dir
        for dy in (-1, 0, 1):
            sub = _get_world_elements(world, tx, by + dy, bz)
            if sub:
                els.extend(sub)
    if dz_dir:
        tz = bz + dz_dir
        for dy in (-1, 0, 1):
            sub = _get_world_elements(world, bx, by + dy, tz)
            if sub:
                els.extend(sub)
    res = tuple(els)
    if nb_cache is not None:
        nb_cache[n_key] = res
    return res


def _glass_group(block):
    if block is None:
        return None
    props = getattr(block, "properties", {})
    if props.get("ctm_group"):
        return props["ctm_group"]
    if props.get("stained_glass") or getattr(block, "translucent", False):
        return getattr(block, "base_id", None) or block.id
    return "glass"


def _is_glass_face_culled(world, wx, wy, wz, normal_i, block=None):
    return _should_cull_glass_element_face(world, wx, wy, wz, block or BLOCKS.get("glass"), None, normal_i)


def _glass_face_connects(world, wx, wy, wz, normal_i, d_vec, block=None, element=None):
    if world is None:
        return False
    nx = int(NORMALS[normal_i, 0])
    ny = int(NORMALS[normal_i, 1])
    nz = int(NORMALS[normal_i, 2])
    norm_axis = 0 if nx != 0 else (1 if ny != 0 else 2)

    if element is None:
        bounds = [
            (float(wx), float(wx + 1)),
            (float(wy), float(wy + 1)),
            (float(wz), float(wz + 1)),
        ]
    else:
        bounds = [
            (wx + element["from"][0] / 16.0, wx + element["to"][0] / 16.0),
            (wy + element["from"][1] / 16.0, wy + element["to"][1] / 16.0),
            (wz + element["from"][2] / 16.0, wz + element["to"][2] / 16.0),
        ]

    # Plane coordinate along norm_axis
    plane = bounds[norm_axis][1] if (nx + ny + nz) > 0 else bounds[norm_axis][0]

    # Direction axis and step
    dx, dy, dz = int(d_vec[0]), int(d_vec[1]), int(d_vec[2])
    d_axis = 0 if dx != 0 else (1 if dy != 0 else 2)
    d_sign = dx + dy + dz
    d_test = bounds[d_axis][1] + 0.05 if d_sign > 0 else bounds[d_axis][0] - 0.05

    # Perpendicular axis
    p_axis = 3 - norm_axis - d_axis
    p_min, p_max = bounds[p_axis]
    p_probes = (p_min + 0.25 * (p_max - p_min), p_min + 0.75 * (p_max - p_min))

    for p_val in p_probes:
        pt = [0.0, 0.0, 0.0]
        pt[norm_axis] = plane
        pt[d_axis] = d_test
        pt[p_axis] = p_val

        # Query nearby elements inside the block of the face first; skip front occlusion check if none match
        q_x = pt[0] - nx * 0.01
        q_y = pt[1] - ny * 0.01
        q_z = pt[2] - nz * 0.01
        found = False
        my_group = _glass_group(block) if block is not None else None
        for nbr in _query_nearby_elements(world, q_x, q_y, q_z):
            if not nbr["glass"]:
                continue
            if my_group is not None and nbr["glass_group"] != my_group:
                continue
            if normal_i not in nbr["faces"]:
                continue
            nbr_plane = nbr["box"][norm_axis * 2 + 1] if (nx + ny + nz) > 0 else nbr["box"][norm_axis * 2]
            if abs(nbr_plane - plane) > 1e-3:
                continue
            bx0, bx1, by0, by1, bz0, bz1 = nbr["box"]
            if (bx0 - 1e-3 <= pt[0] <= bx1 + 1e-3 and
                by0 - 1e-3 <= pt[1] <= by1 + 1e-3 and
                bz0 - 1e-3 <= pt[2] <= bz1 + 1e-3):
                found = True
                break
        if not found:
            return False

        # Check if front is occluded by an opaque non-glass element right in front of the plane
        front_qx = pt[0] + nx * 0.05
        front_qy = pt[1] + ny * 0.05
        front_qz = pt[2] + nz * 0.05
        for f_nbr in _query_nearby_elements(world, front_qx, front_qy, front_qz):
            if not f_nbr["opaque"] or f_nbr["glass"]:
                continue
            bx0, bx1, by0, by1, bz0, bz1 = f_nbr["box"]
            if (bx0 - 1e-3 <= front_qx <= bx1 + 1e-3 and
                by0 - 1e-3 <= front_qy <= by1 + 1e-3 and
                bz0 - 1e-3 <= front_qz <= bz1 + 1e-3):
                return False
    return True


def _get_glass_face_ctm_mask(world, wx, wy, wz, normal_i, block=None, element=None):
    if world is None:
        return 0
    u_vec, v_vec = FACE_UV_VECTORS[normal_i]
    up = _glass_face_connects(world, wx, wy, wz, normal_i, v_vec, block, element)
    down = _glass_face_connects(world, wx, wy, wz, normal_i, (-v_vec[0], -v_vec[1], -v_vec[2]), block, element)
    left = _glass_face_connects(world, wx, wy, wz, normal_i, (-u_vec[0], -u_vec[1], -u_vec[2]), block, element)
    right = _glass_face_connects(world, wx, wy, wz, normal_i, u_vec, block, element)
    return (1 if up else 0) | (2 if down else 0) | (4 if left else 0) | (8 if right else 0)


def get_block_face_ctm_mask(world, wx, wy, wz, normal_i, block=None, element=None):
    if world is None:
        return 0
    if is_glass(block):
        return _get_glass_face_ctm_mask(world, wx, wy, wz, normal_i, block=block, element=element)
    u_vec, v_vec = FACE_UV_VECTORS[normal_i]
    up = _block_face_connects(world, wx, wy, wz, normal_i, v_vec, block, element)
    down = _block_face_connects(world, wx, wy, wz, normal_i, (-v_vec[0], -v_vec[1], -v_vec[2]), block, element)
    left = _block_face_connects(world, wx, wy, wz, normal_i, (-u_vec[0], -u_vec[1], -u_vec[2]), block, element)
    right = _block_face_connects(world, wx, wy, wz, normal_i, u_vec, block, element)
    return (1 if up else 0) | (2 if down else 0) | (4 if left else 0) | (8 if right else 0)


def _block_face_connects(world, wx, wy, wz, normal_i, d_vec, block=None, element=None):
    if world is None:
        return False
    if getattr(block, "_is_glass", False) or is_glass(block):
        return _glass_face_connects(world, wx, wy, wz, normal_i, d_vec, block, element)

    nx = int(NORMALS[normal_i, 0])
    ny = int(NORMALS[normal_i, 1])
    nz = int(NORMALS[normal_i, 2])
    dx, dy, dz = int(d_vec[0]), int(d_vec[1]), int(d_vec[2])

    get_obj = getattr(world, "get_block_obj", None)
    if get_obj is not None:
        nbr_b = get_obj(wx + dx, wy + dy, wz + dz)
    else:
        nbr_b_id = world.get_block(wx + dx, wy + dy, wz + dz)
        nbr_b = BLOCKS.get(nbr_b_id) if isinstance(nbr_b_id, str) else nbr_b_id
    if nbr_b is None:
        return False

    if block is not None:
        my_b = block
    elif get_obj is not None:
        my_b = get_obj(wx, wy, wz)
    else:
        my_id = world.get_block(wx, wy, wz)
        my_b = BLOCKS.get(my_id) if isinstance(my_id, str) else my_id
    if my_b is None:
        return False

    my_group = getattr(my_b, "ctm_group", None) or my_b.properties.get("ctm_group") or my_b.base_id or my_b.id
    nbr_group = getattr(nbr_b, "ctm_group", None) or nbr_b.properties.get("ctm_group") or nbr_b.base_id or nbr_b.id
    if my_group != nbr_group:
        return False

    if get_obj is not None:
        fb = get_obj(wx + dx + nx, wy + dy + ny, wz + dz + nz)
    else:
        front_b_id = world.get_block(wx + dx + nx, wy + dy + ny, wz + dz + nz)
        fb = BLOCKS.get(front_b_id) if isinstance(front_b_id, str) else front_b_id
    if fb is not None and getattr(fb, "solid", False) and getattr(fb, "full_cube", False) and not getattr(fb, "cutout", False):
        return False

    return True


def _should_cull_glass_element_face(world, wx, wy, wz, block, element, normal_i):
    if world is None:
        return False
    nx = int(NORMALS[normal_i, 0])
    ny = int(NORMALS[normal_i, 1])
    nz = int(NORMALS[normal_i, 2])
    norm_axis = 0 if nx != 0 else (1 if ny != 0 else 2)
    n_sign = nx + ny + nz

    if element is None:
        bounds = [
            (float(wx), float(wx + 1)),
            (float(wy), float(wy + 1)),
            (float(wz), float(wz + 1)),
        ]
        my_box = (float(wx), float(wx + 1), float(wy), float(wy + 1), float(wz), float(wz + 1))
    else:
        bounds = [
            (wx + element["from"][0] / 16.0, wx + element["to"][0] / 16.0),
            (wy + element["from"][1] / 16.0, wy + element["to"][1] / 16.0),
            (wz + element["from"][2] / 16.0, wz + element["to"][2] / 16.0),
        ]
        my_box = (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1], bounds[2][0], bounds[2][1])

    plane = bounds[norm_axis][1] if n_sign > 0 else bounds[norm_axis][0]

    # Two tangent axes
    t1_axis = (norm_axis + 1) % 3
    t2_axis = (norm_axis + 2) % 3
    t1_min, t1_max = bounds[t1_axis]
    t2_min, t2_max = bounds[t2_axis]

    t1_probes = (t1_min + 0.25 * (t1_max - t1_min), t1_min + 0.75 * (t1_max - t1_min))
    t2_probes = (t2_min + 0.25 * (t2_max - t2_min), t2_min + 0.75 * (t2_max - t2_min))
    my_group = _glass_group(block) if block is not None else None

    for p1 in t1_probes:
        for p2 in t2_probes:
            pt = [0.0, 0.0, 0.0]
            pt[norm_axis] = plane
            pt[t1_axis] = p1
            pt[t2_axis] = p2

            # Step slightly in direction +normal across the plane
            q_x = pt[0] + nx * 0.01
            q_y = pt[1] + ny * 0.01
            q_z = pt[2] + nz * 0.01

            covered = False
            for nbr in _query_nearby_elements(world, q_x, q_y, q_z):
                # Skip self element
                if nbr["box"] == my_box and (element is None or nbr["element"] is element):
                    continue
                # Must be matching glass group or opaque solid
                if nbr["glass"]:
                    if my_group is not None and nbr["glass_group"] != my_group:
                        continue
                elif not nbr["opaque"]:
                    continue

                bx0, bx1, by0, by1, bz0, bz1 = nbr["box"]
                b_t1 = (bx0, bx1) if t1_axis == 0 else ((by0, by1) if t1_axis == 1 else (bz0, bz1))
                b_t2 = (bx0, bx1) if t2_axis == 0 else ((by0, by1) if t2_axis == 1 else (bz0, bz1))
                if not (b_t1[0] - 1e-3 <= pt[t1_axis] <= b_t1[1] + 1e-3):
                    continue
                if not (b_t2[0] - 1e-3 <= pt[t2_axis] <= b_t2[1] + 1e-3):
                    continue

                b_norm = (bx0, bx1) if norm_axis == 0 else ((by0, by1) if norm_axis == 1 else (bz0, bz1))
                q_val = pt[norm_axis] + n_sign * 0.01
                if not (b_norm[0] - 1e-3 <= q_val <= b_norm[1] + 1e-3):
                    continue

                if n_sign > 0:
                    if b_norm[0] > plane + 1e-3:
                        continue
                else:
                    if b_norm[1] < plane - 1e-3:
                        continue

                covered = True
                break
            if not covered:
                return False
    return True


def _get_element_sub_box(element, normal_i, ua, ub, va, vb):
    if element is None:
        fx, fy, fz = 0.0, 0.0, 0.0
        tx, ty, tz = 16.0, 16.0, 16.0
    else:
        fx, fy, fz = float(element["from"][0]), float(element["from"][1]), float(element["from"][2])
        tx, ty, tz = float(element["to"][0]), float(element["to"][1]), float(element["to"][2])
    dx, dy, dz = tx - fx, ty - fy, tz - fz
    if normal_i == 0:  # +X East: u = -Z, v = +Y
        return {"from": [fx, fy + va * dy, fz + (1.0 - ub) * dz], "to": [tx, fy + vb * dy, fz + (1.0 - ua) * dz]}
    if normal_i == 1:  # -X West: u = +Z, v = +Y
        return {"from": [fx, fy + va * dy, fz + ua * dz], "to": [tx, fy + vb * dy, fz + ub * dz]}
    if normal_i == 2:  # +Y Up: u = +X, v = -Z
        return {"from": [fx + ua * dx, fy, fz + (1.0 - vb) * dz], "to": [fx + ub * dx, ty, fz + (1.0 - va) * dz]}
    if normal_i == 3:  # -Y Down: u = +X, v = +Z
        return {"from": [fx + ua * dx, fy, fz + va * dz], "to": [fx + ub * dx, ty, fz + vb * dz]}
    if normal_i == 4:  # +Z South: u = +X, v = +Y
        return {"from": [fx + ua * dx, fy + va * dy, fz], "to": [fx + ub * dx, fy + vb * dy, tz]}
    return {"from": [fx + (1.0 - ub) * dx, fy + va * dy, fz], "to": [fx + (1.0 - ua) * dx, fy + vb * dy, tz]}


def _get_face_sub_element(normal_i, ua, ub, va, vb):
    return _get_element_sub_box(None, normal_i, ua, ub, va, vb)


def _slice_spec_uv(spec_uv, ua, ub, va, vb):
    su0, sv0, su1, sv1 = spec_uv
    return [
        su0 + ua * (su1 - su0),
        sv0 + (1.0 - vb) * (sv1 - sv0),
        su0 + ub * (su1 - su0),
        sv0 + (1.0 - va) * (sv1 - sv0),
    ]


def _resolve_glass_face_parts(world, wx, wy, wz, block, element, normal_i):
    """Return tuple of (ua, ub, va, vb, mask_or_None) for visible sub-quads of a glass face."""
    if world is None:
        return ((0.0, 1.0, 0.0, 1.0, 0),)

    nx = int(NORMALS[normal_i, 0])
    ny = int(NORMALS[normal_i, 1])
    nz = int(NORMALS[normal_i, 2])
    norm_axis = 0 if nx != 0 else (1 if ny != 0 else 2)
    n_sign = nx + ny + nz
    opp_normal_i = (1, 0, 3, 2, 5, 4)[normal_i]

    if element is None:
        bounds = ((float(wx), float(wx + 1)), (float(wy), float(wy + 1)), (float(wz), float(wz + 1)))
    else:
        bounds = (
            (wx + element["from"][0] / 16.0, wx + element["to"][0] / 16.0),
            (wy + element["from"][1] / 16.0, wy + element["to"][1] / 16.0),
            (wz + element["from"][2] / 16.0, wz + element["to"][2] / 16.0),
        )
    my_box = (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1], bounds[2][0], bounds[2][1])
    plane = bounds[norm_axis][1] if n_sign > 0 else bounds[norm_axis][0]

    u_vec, v_vec = FACE_UV_VECTORS[normal_i]
    u_axis = 0 if u_vec[0] else (1 if u_vec[1] else 2)
    v_axis = 0 if v_vec[0] else (1 if v_vec[1] else 2)
    u_sign = u_vec[u_axis]
    v_sign = v_vec[v_axis]
    u0, u1 = bounds[u_axis] if u_sign > 0 else (bounds[u_axis][1], bounds[u_axis][0])
    v0, v1 = bounds[v_axis] if v_sign > 0 else (bounds[v_axis][1], bounds[v_axis][0])
    my_group = _glass_group(block) if block is not None else None

    def _is_covered(u_frac, v_frac):
        pt = [0.0, 0.0, 0.0]
        pt[norm_axis] = plane
        pt[u_axis] = u0 + u_frac * (u1 - u0)
        pt[v_axis] = v0 + v_frac * (v1 - v0)
        qx = pt[0] + nx * 0.01
        qy = pt[1] + ny * 0.01
        qz = pt[2] + nz * 0.01
        for nbr in _query_nearby_elements(world, qx, qy, qz):
            if nbr["box"] == my_box and (element is None or nbr["element"] is element):
                continue
            if nbr["glass"]:
                if my_group is not None and nbr["glass_group"] != my_group:
                    continue
            elif not nbr["opaque"]:
                continue

            bx0, bx1, by0, by1, bz0, bz1 = nbr["box"]
            t1_axis = u_axis
            t2_axis = v_axis
            b_t1 = (bx0, bx1) if t1_axis == 0 else ((by0, by1) if t1_axis == 1 else (bz0, bz1))
            b_t2 = (bx0, bx1) if t2_axis == 0 else ((by0, by1) if t2_axis == 1 else (bz0, bz1))
            if not (b_t1[0] - 1e-3 <= pt[t1_axis] <= b_t1[1] + 1e-3):
                continue
            if not (b_t2[0] - 1e-3 <= pt[t2_axis] <= b_t2[1] + 1e-3):
                continue

            b_norm = (bx0, bx1) if norm_axis == 0 else ((by0, by1) if norm_axis == 1 else (bz0, bz1))
            q_val = pt[norm_axis] + n_sign * 0.01
            if not (b_norm[0] - 1e-3 <= q_val <= b_norm[1] + 1e-3):
                continue

            if n_sign > 0:
                if b_norm[0] > plane + 1e-3:
                    continue
            else:
                if b_norm[1] < plane - 1e-3:
                    continue

            return True
        return False

    q00 = _is_covered(0.25, 0.25)
    q10 = _is_covered(0.75, 0.25)
    q01 = _is_covered(0.25, 0.75)
    q11 = _is_covered(0.75, 0.75)
    if q00 and q10 and q01 and q11:
        return ()

    cull_u0 = q00 and q01
    cull_u1 = q10 and q11
    cull_v0 = q00 and q10
    cull_v1 = q01 and q11

    def _edge_connects(d_axis, d_val, p_axis, p_val):
        pt = [0.0, 0.0, 0.0]
        pt[norm_axis] = plane
        pt[d_axis] = d_val
        pt[p_axis] = p_val
        qx = pt[0] - nx * 0.01
        qy = pt[1] - ny * 0.01
        qz = pt[2] - nz * 0.01
        found = False
        for nbr in _query_nearby_elements(world, qx, qy, qz):
            if not nbr["glass"]:
                continue
            if my_group is not None and nbr["glass_group"] != my_group:
                continue
            if normal_i not in nbr["faces"]:
                continue
            nbr_plane = nbr["box"][norm_axis * 2 + 1] if n_sign > 0 else nbr["box"][norm_axis * 2]
            if abs(nbr_plane - plane) > 1e-3:
                continue
            bx0, bx1, by0, by1, bz0, bz1 = nbr["box"]
            if (bx0 - 1e-3 <= pt[0] <= bx1 + 1e-3 and
                by0 - 1e-3 <= pt[1] <= by1 + 1e-3 and
                bz0 - 1e-3 <= pt[2] <= bz1 + 1e-3):
                found = True
                break
        if not found:
            return False
        fqx = pt[0] + nx * 0.05
        fqy = pt[1] + ny * 0.05
        fqz = pt[2] + nz * 0.05
        for f_nbr in _query_nearby_elements(world, fqx, fqy, fqz):
            if not f_nbr["opaque"] or f_nbr["glass"]:
                continue
            bx0, bx1, by0, by1, bz0, bz1 = f_nbr["box"]
            if (bx0 - 1e-3 <= fqx <= bx1 + 1e-3 and
                by0 - 1e-3 <= fqy <= by1 + 1e-3 and
                bz0 - 1e-3 <= fqz <= bz1 + 1e-3):
                return False
        return True

    up_0 = _edge_connects(v_axis, v1 + 0.05 * v_sign, u_axis, u0 + 0.25 * (u1 - u0))
    up_1 = _edge_connects(v_axis, v1 + 0.05 * v_sign, u_axis, u0 + 0.75 * (u1 - u0))
    down_0 = _edge_connects(v_axis, v0 - 0.05 * v_sign, u_axis, u0 + 0.25 * (u1 - u0))
    down_1 = _edge_connects(v_axis, v0 - 0.05 * v_sign, u_axis, u0 + 0.75 * (u1 - u0))
    left_0 = _edge_connects(u_axis, u0 - 0.05 * u_sign, v_axis, v0 + 0.25 * (v1 - v0))
    left_1 = _edge_connects(u_axis, u0 - 0.05 * u_sign, v_axis, v0 + 0.75 * (v1 - v0))
    right_0 = _edge_connects(u_axis, u1 + 0.05 * u_sign, v_axis, v0 + 0.25 * (v1 - v0))
    right_1 = _edge_connects(u_axis, u1 + 0.05 * u_sign, v_axis, v0 + 0.75 * (v1 - v0))

    split_u = (cull_u0 != cull_u1) or (up_0 != up_1) or (down_0 != down_1)
    split_v = (cull_v0 != cull_v1) or (left_0 != left_1) or (right_0 != right_1)
    if not split_u and not split_v:
        if q00 or q10 or q01 or q11:
            split_u = True
            split_v = True
        else:
            mask = (
                (1 if (up_0 and up_1) else 0)
                | (2 if (down_0 and down_1) else 0)
                | (4 if (left_0 and left_1) else 0)
                | (8 if (right_0 and right_1) else 0)
            )
            return ((0.0, 1.0, 0.0, 1.0, mask),)

    if split_u and split_v:
        quads = (
            ((0.0, 0.5, 0.0, 0.5, None) if not q00 else None),
            ((0.5, 1.0, 0.5, 1.0, None) if not q11 else None),
            ((0.5, 1.0, 0.0, 0.5, None) if not q10 else None),
            ((0.0, 0.5, 0.5, 1.0, None) if not q01 else None),
        )
        return tuple(q for q in quads if q is not None)
    if split_u:
        return tuple(
            p for p in (
                ((0.0, 0.5, 0.0, 1.0, None) if not cull_u0 else None),
                ((0.5, 1.0, 0.0, 1.0, None) if not cull_u1 else None),
            ) if p is not None
        )
    return tuple(
        p for p in (
            ((0.0, 1.0, 0.0, 0.5, None) if not cull_v0 else None),
            ((0.0, 1.0, 0.5, 1.0, None) if not cull_v1 else None),
        ) if p is not None
    )


def _emit_cutout(world, data, faces, total, x0, y0, z0, l_grid=None, grass_grid=None):
    if grass_grid is None:
        grass_grid = _build_grass_grid(world, x0 // CHUNK_SIZE, z0 // CHUNK_SIZE)
    vertices = np.empty((total * 8, 9), dtype=np.float32)
    indices = np.empty(total * 12, dtype=np.uint32)
    lights = np.empty((total * 8, 6), dtype=np.float32)
    colors = np.ones((total * 8, 3), dtype=np.float32)
    cursor = 0
    face_i = 0
    for ys, zs, xs, normal_i in faces:
        ids = data[ys, zs, xs]
        normal = NORMALS[normal_i]
        corners = CORNERS[normal_i]
        nx = int(normal[0])
        ny = int(normal[1])
        nz = int(normal[2])
        for block_id in np.unique(ids):
            block = block_from_index(int(block_id))
            if block is None:
                continue
            picked = ids == block_id
            yy = ys[picked]
            zz = zs[picked]
            xx = xs[picked]
            count = len(yy)
            shade = 0.92 + 0.08 * hash2_grid(np.int64(x0) + xx.astype(np.int64), np.int64(z0) + zz.astype(np.int64), 0)
            origin = np.stack((xx, yy, zz), axis=1).astype(np.float32)
            positions = origin[:, None, :] + corners[None, :, :] + normal * np.float32(0.001)

            if l_grid is not None:
                face_light = l_grid[yy + 1 + ny, zz + 1 + nz, xx + 1 + nx]
            else:
                face_light = np.empty((count, 6), dtype=np.float32)
                for i in range(count):
                    face_light[i] = _sample_light(world, x0 + int(xx[i]) + nx, y0 + int(yy[i]) + ny, z0 + int(zz[i]) + nz)

            if is_glass(block) and GLASS_CTM_UVS:
                for i in range(count):
                    wx = x0 + int(xx[i])
                    wy = y0 + int(yy[i])
                    wz = z0 + int(zz[i])
                    parts = _resolve_glass_face_parts(world, wx, wy, wz, block, None, normal_i)
                    if not parts:
                        continue

                    sub_quads = []
                    if len(parts) == 1 and parts[0][:4] == (0.0, 1.0, 0.0, 1.0):
                        m = parts[0][4]
                        default_uv = (
                            GLASS_CTM_UVS[m]
                            if getattr(block, "id", "") in ("glass", "glass_slab", "glass_vertical_slab")
                            else block.face_uvs[normal_i]
                        ) if 0 <= m < len(GLASS_CTM_UVS) else block.face_uvs[normal_i]
                        sub_quads.append((positions[i], get_block_ctm_uv(block, m) or default_uv))
                    else:
                        pos = positions[i]
                        v0, v1, v2, v3 = pos[0], pos[1], pos[2], pos[3]
                        for ua, ub, va, vb, pm in parts:
                            sub_el = _get_face_sub_element(normal_i, ua, ub, va, vb)
                            m = pm if pm is not None else _get_glass_face_ctm_mask(world, wx, wy, wz, normal_i, block=block, element=sub_el)
                            spec_uv = _slice_spec_uv((0, 0, 16, 16), ua, ub, va, vb)
                            def_uv = (
                                GLASS_CTM_UVS[m]
                                if getattr(block, "id", "") in ("glass", "glass_slab", "glass_vertical_slab")
                                else block.face_uvs[normal_i]
                            ) if 0 <= m < len(GLASS_CTM_UVS) else block.face_uvs[normal_i]
                            uv = get_block_ctm_uv(block, m, [int(round(c)) for c in spec_uv]) or def_uv
                            p0 = (1.0 - ua) * (1.0 - va) * v0 + ua * (1.0 - va) * v1 + ua * va * v2 + (1.0 - ua) * va * v3
                            p1 = (1.0 - ub) * (1.0 - va) * v0 + ub * (1.0 - va) * v1 + ub * va * v2 + (1.0 - ub) * va * v3
                            p2 = (1.0 - ub) * (1.0 - vb) * v0 + ub * (1.0 - vb) * v1 + ub * vb * v2 + (1.0 - ub) * vb * v3
                            p3 = (1.0 - ua) * (1.0 - vb) * v0 + ua * (1.0 - vb) * v1 + ua * vb * v2 + (1.0 - ua) * vb * v3
                            sub_quads.append((np.array([p0, p1, p2, p3]), uv))

                    for h_pos, h_uv in sub_quads:
                        vertices[cursor : cursor + 4, 0:3] = h_pos
                        vertices[cursor : cursor + 4, 3:6] = normal
                        vertices[cursor : cursor + 4, 6:8] = h_uv
                        vertices[cursor : cursor + 4, 8] = shade[i]
                        lights[cursor : cursor + 4] = face_light[i]
                        slot = indices[face_i * 6 : (face_i + 1) * 6]
                        quad = np.uint32(cursor)
                        slot[0:6] = (quad, quad + 1, quad + 2, quad, quad + 2, quad + 3)
                        cursor += 4
                        face_i += 1
                continue

            uvs = np.broadcast_to(np.asarray(block.face_uvs[normal_i], dtype=np.float32), (count, 4, 2)).reshape(-1, 2)
            span = count * 4
            vertices[cursor : cursor + span, 0:3] = positions.reshape(-1, 3)
            vertices[cursor : cursor + span, 3:6] = normal
            vertices[cursor : cursor + span, 6:8] = uvs
            vertices[cursor : cursor + span, 8] = np.repeat(shade.astype(np.float32), 4)
            lights[cursor : cursor + span] = np.repeat(face_light, 4, axis=0)

            if getattr(block, "tint", None) == "grass" or getattr(block, "id", "") in ("tall_grass", "fern"):
                face_color = grass_grid[zz, xx]
                colors[cursor : cursor + span] = np.repeat(face_color, 4, axis=0)

            quads = np.arange(count, dtype=np.uint32) * 4 + np.uint32(cursor)
            slot = indices[face_i * 6 : (face_i + count) * 6].reshape(count, 6)
            slot[:, 0] = quads
            slot[:, 1] = quads + 1
            slot[:, 2] = quads + 2
            slot[:, 3] = quads
            slot[:, 4] = quads + 2
            slot[:, 5] = quads + 3
            cursor += span
            face_i += count
    return vertices[:cursor], indices[: face_i * 6], lights[:cursor], colors[:cursor]


def _boundary(element, normal_i):
    if normal_i >= 6:
        return False
    origin = element["from"]
    extent = element["to"]
    if normal_i == 0:
        return extent[0] >= 16 - 1e-3
    if normal_i == 1:
        return origin[0] <= 1e-3
    if normal_i == 2:
        return extent[1] >= 16 - 1e-3
    if normal_i == 3:
        return origin[1] <= 1e-3
    if normal_i == 4:
        return extent[2] >= 16 - 1e-3
    return origin[2] <= 1e-3


def _neighbor_occ_or_fluid(occ, fluid, y, z, x, normal_i):
    if normal_i >= 6:
        return False
    y += 1
    z += 1
    x += 1
    if normal_i == 0:
        return bool(occ[y, z, x + 1] or fluid[y, z, x + 1])
    if normal_i == 1:
        return bool(occ[y, z, x - 1] or fluid[y, z, x - 1])
    if normal_i == 2:
        return bool(occ[y + 1, z, x] or fluid[y + 1, z, x])
    if normal_i == 3:
        return bool(occ[y - 1, z, x] or fluid[y - 1, z, x])
    if normal_i == 4:
        return bool(occ[y, z + 1, x] or fluid[y, z + 1, x])
    return bool(occ[y, z - 1, x] or fluid[y, z - 1, x])


def _neighbor_solid(occ, y, z, x, normal_i):
    if normal_i >= 6:
        return False
    y += 1
    z += 1
    x += 1
    if normal_i == 0:
        return bool(occ[y, z, x + 1])
    if normal_i == 1:
        return bool(occ[y, z, x - 1])
    if normal_i == 2:
        return bool(occ[y + 1, z, x])
    if normal_i == 3:
        return bool(occ[y - 1, z, x])
    if normal_i == 4:
        return bool(occ[y, z + 1, x])
    return bool(occ[y, z - 1, x])


def _element_corners(element, normal_i):
    origin = np.asarray(element["from"], dtype=np.float32) / 16.0
    extent = np.asarray(element["to"], dtype=np.float32) / 16.0
    return origin + CORNERS[normal_i] * (extent - origin)
