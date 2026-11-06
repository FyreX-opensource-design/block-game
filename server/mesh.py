import numpy as np

from server.blocks import (
    BLOCKS,
    CHUNK_SIZE,
    FACE_UV_VECTORS,
    GLASS_CTM_UVS,
    SECTION_COUNT,
    SUB_CELL_BLOCK,
    block_from_index,
    get_glass_ctm_uv,
    get_offset_info,
    has_half_offset_top,
    is_face_glass,
    is_glass,
    is_offset_block,
    is_sub_cell,
    is_vertical_slab_offset,
)
from server.noise import hash2, hash2_grid

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
        if block.properties.get("fluid"):
            continue
        if not block.full_cube:
            continue
        if block.opacity < 1.0:
            continue
        mask |= data == value
    return mask


def _mask_glass(data):
    mask = np.zeros(data.shape, dtype=bool)
    for value in np.unique(data):
        block = block_from_index(int(value))
        if block is not None and is_glass(block) and block.full_cube:
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


def octant_face_uvs(mat_block, normal_i, ox, oy, oz, off_x=None, off_y=None, off_z=None):
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
    cursor_opaque, cursor_cutout,
):
    from server.sub_cell import OCTANT_COORDS

    for idx, mat_id in enumerate(sub.octants):
        if not mat_id:
            continue
        mat_block = BLOCKS.get(mat_id)
        if mat_block is None:
            continue

        ox, oy, oz = OCTANT_COORDS[idx]
        is_cutout = bool(mat_block.cutout)
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
                        if not adj_blk.cutout and adj_blk.opacity >= 1.0:
                            continue
                        if is_cutout and adj_blk.cutout and adj_mat == mat_id:
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
                            if not neigh_blk.cutout and neigh_blk.opacity >= 1.0:
                                continue
                            if is_cutout and neigh_blk.cutout and neigh_mat == mat_id:
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

            uvs = octant_face_uvs(mat_block, normal_i, ox, oy, oz, off_x=off_x, off_y=off_y, off_z=off_z)
            corners = CORNERS[normal_i]
            pts = []
            for k in range(4):
                px = lx + (ox + corners[k, 0]) * 0.5
                py = ly + (oy + corners[k, 1]) * 0.5
                pz = lz + (oz + corners[k, 2]) * 0.5
                pts.append((px, py, pz))

            if is_cutout:
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

    return cursor_opaque, cursor_cutout


def build_chunk_mesh(world, cx, cy, cz):
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
    if partial_ids:
        partial_mask = np.isin(chunk.data, np.asarray(partial_ids, dtype=chunk.data.dtype))
        own[partial_mask] = False
        own_see[partial_mask] = False
    faces, total = _exposed(own, occ)
    cutout_faces, cutout_total = _exposed(own_see, occ, see, glass=glass)
    partial_cells = (
        np.argwhere(np.isin(chunk.data, np.asarray(partial_ids, dtype=chunk.data.dtype))) if partial_ids else ()
    )
    fluid_mask = fluid_h[1:-1, 1:-1, 1:-1] > 0
    fluid_ys, fluid_zs, fluid_xs = np.nonzero(fluid_mask) if np.any(fluid_mask) else ((), (), ())

    if total == 0 and cutout_total == 0 and len(partial_cells) == 0 and len(fluid_ys) == 0:
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
    cursor_translucent = 0

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
                cursor_opaque, cursor_cutout = _mesh_sub_cell(
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
                    cursor_opaque,
                    cursor_cutout,
                )
            continue
        from server.blocks import is_fence, get_fence_elements
        elements = get_fence_elements(world, wx, wy, wz, block) if is_fence(block) else block.elements
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

                el_cutout = element.get("cutout", block.cutout)
                el_is_glass = element.get("glass") if "glass" in element else is_glass(block)
                if from_y < -1e-3:
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
                            if _is_opaque_offset(nbr_b) or (el_cutout and is_offset_block(nbr_b) and not is_vertical_slab_offset(nbr_b) and normal_i in (1, 5)):
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
                    if el_is_glass and _should_cull_glass_element_face(world, wx, wy, wz, block, element, normal_i):
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
                if el_is_glass and GLASS_CTM_UVS:
                    m = _get_glass_face_ctm_mask(world, wx, wy, wz, normal_i, block=block, element=element)
                    spec_uv = element.get("face_uv_specs", {}).get(normal_i, [0, 0, 16, 16])
                    uvs = get_glass_ctm_uv(m, spec_uv) or uvs
                points = _element_corners(element, normal_i)
                if el_cutout:
                    base = cursor_cutout
                    for corner, uv in zip(points, uvs):
                        cutout_extra_vert.extend(
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
                        cutout_extra_light.append(l_sample)
                        cutout_extra_color.append(pt_color)
                        cursor_cutout += 1
                    cutout_extra_ind.extend((base, base + 1, base + 2, base, base + 2, base + 3))
                else:
                    base = cursor_opaque
                    for corner, uv in zip(points, uvs):
                        extra_vertices.extend(
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
                        extra_lights.append(l_sample)
                        extra_colors.append(pt_color)
                        cursor_opaque += 1
                    extra_indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))

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
        translucent = (
            np.asarray(translucent_vert, dtype=np.float32).reshape(-1, 9),
            np.asarray(translucent_ind, dtype=np.uint32),
        )
        translucent_lights = np.asarray(translucent_light, dtype=np.float32).reshape(-1, 6)
        translucent_colors = np.asarray(translucent_color, dtype=np.float32).reshape(-1, 3)

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


def _is_glass_face_culled(world, wx, wy, wz, normal_i):
    if world is None:
        return False
    nx = int(NORMALS[normal_i, 0])
    ny = int(NORMALS[normal_i, 1])
    nz = int(NORMALS[normal_i, 2])
    adj = world.get_block(wx + nx, wy + ny, wz + nz)
    if adj is None:
        return False
    adj_face = (1, 0, 3, 2, 5, 4)[normal_i]
    if not is_face_glass(adj, adj_face):
        return False
    adj_blk = BLOCKS.get(adj) if isinstance(adj, str) else adj
    if adj_blk is None:
        return False
    if getattr(adj_blk, "full_cube", False):
        return True
    props = getattr(adj_blk, "properties", {})
    if normal_i == 2:
        if props.get("slab") and props.get("half", getattr(adj_blk, "slab_half", "bottom")) in ("bottom", "double"):
            return True
        if props.get("double_vertical_slab"):
            return True
    elif normal_i == 3:
        if props.get("slab") and props.get("half", getattr(adj_blk, "slab_half", "bottom")) in ("top", "double"):
            return True
        if props.get("double_vertical_slab"):
            return True
    elif normal_i == 4:
        if props.get("vertical_slab") and props.get("facing") == "north":
            return True
        if props.get("double_vertical_slab"):
            return True
    elif normal_i == 5:
        if props.get("vertical_slab") and props.get("facing", "south") == "south":
            return True
        if props.get("double_vertical_slab"):
            return True
    elif normal_i == 0:
        if props.get("vertical_slab") and props.get("facing") == "west":
            return True
        if props.get("double_vertical_slab"):
            return True
    elif normal_i == 1:
        if props.get("vertical_slab") and props.get("facing") == "east":
            return True
        if props.get("double_vertical_slab"):
            return True
    return False


def _glass_face_connects(world, wx, wy, wz, normal_i, d_vec, block=None, element=None):
    if world is None:
        return False
    target_x = wx + int(d_vec[0])
    target_y = wy + int(d_vec[1])
    target_z = wz + int(d_vec[2])
    target_blk = world.get_block(target_x, target_y, target_z)
    if target_blk is None or not is_face_glass(target_blk, normal_i):
        return False

    nx = int(NORMALS[normal_i, 0])
    ny = int(NORMALS[normal_i, 1])
    nz = int(NORMALS[normal_i, 2])

    front_blk = world.get_block(target_x + nx, target_y + ny, target_z + nz)
    if front_blk is not None:
        fb_obj = BLOCKS.get(front_blk) if isinstance(front_blk, str) else front_blk
        if fb_obj is not None and getattr(fb_obj, "solid", False) and getattr(fb_obj, "full_cube", False) and not getattr(fb_obj, "cutout", False):
            return False

    b_obj = BLOCKS.get(block) if isinstance(block, str) else block
    t_obj = BLOCKS.get(target_blk) if isinstance(target_blk, str) else target_blk
    b_props = getattr(b_obj, "properties", {}) if b_obj is not None else {}
    t_props = getattr(t_obj, "properties", {}) if t_obj is not None else {}

    b_is_slab = b_props.get("slab", False) and not b_props.get("double_slab", False)
    t_is_slab = t_props.get("slab", False) and not t_props.get("double_slab", False)
    b_half = b_props.get("half", getattr(b_obj, "slab_half", "bottom")) if b_is_slab else None
    t_half = t_props.get("half", getattr(t_obj, "slab_half", "bottom")) if t_is_slab else None

    # Horizontal side faces (east, west, south, north)
    if normal_i in (0, 1, 4, 5):
        if b_is_slab:
            if b_half == "bottom":
                if d_vec[1] == 0:
                    if t_is_slab and t_half != "bottom":
                        return False
                elif d_vec[1] > 0:
                    return False
                elif d_vec[1] < 0:
                    if t_is_slab and t_half != "top":
                        return False
            elif b_half == "top":
                if d_vec[1] == 0:
                    if t_is_slab and t_half != "top":
                        return False
                elif d_vec[1] > 0:
                    if t_is_slab and t_half != "bottom":
                        return False
                elif d_vec[1] < 0:
                    return False
        else:
            if d_vec[1] > 0:
                if t_is_slab and t_half != "bottom":
                    return False
            elif d_vec[1] < 0:
                if t_is_slab and t_half != "top":
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


def _should_cull_glass_element_face(world, wx, wy, wz, block, element, normal_i):
    if world is None:
        return False
    nx = int(NORMALS[normal_i, 0])
    ny = int(NORMALS[normal_i, 1])
    nz = int(NORMALS[normal_i, 2])

    origin = element["from"]
    extent = element["to"]

    is_at_boundary = False
    if normal_i == 0 and extent[0] >= 16.0 - 1e-3:
        is_at_boundary = True
    elif normal_i == 1 and origin[0] <= 1e-3:
        is_at_boundary = True
    elif normal_i == 2 and extent[1] >= 16.0 - 1e-3:
        is_at_boundary = True
    elif normal_i == 3 and origin[1] <= 1e-3:
        is_at_boundary = True
    elif normal_i == 4 and extent[2] >= 16.0 - 1e-3:
        is_at_boundary = True
    elif normal_i == 5 and origin[2] <= 1e-3:
        is_at_boundary = True

    if not is_at_boundary:
        return False

    adj = world.get_block(wx + nx, wy + ny, wz + nz)
    if adj is None:
        return False
    adj_face = (1, 0, 3, 2, 5, 4)[normal_i]
    if not is_face_glass(adj, adj_face):
        return False

    adj_blk = BLOCKS.get(adj) if isinstance(adj, str) else adj
    if adj_blk is None:
        return False
    if getattr(adj_blk, "full_cube", False):
        return True

    props = getattr(adj_blk, "properties", {})
    b_obj = BLOCKS.get(block) if isinstance(block, str) else block
    b_props = getattr(b_obj, "properties", {}) if b_obj is not None else {}

    if normal_i == 3:
        if props.get("slab") and props.get("half", getattr(adj_blk, "slab_half", "bottom")) in ("top", "double"):
            return True
        if props.get("double_vertical_slab"):
            return True
    elif normal_i == 2:
        if props.get("slab") and props.get("half", getattr(adj_blk, "slab_half", "bottom")) in ("bottom", "double"):
            return True
        if props.get("double_vertical_slab"):
            return True
    elif normal_i in (0, 1, 4, 5):
        b_half = b_props.get("half", getattr(b_obj, "slab_half", "bottom"))
        adj_half = props.get("half", getattr(adj_blk, "slab_half", "bottom"))
        if b_props.get("slab") and props.get("slab"):
            if b_half == adj_half or adj_half == "double":
                return True
        b_facing = b_props.get("facing", "south" if b_props.get("vertical_slab") else None)
        adj_facing = props.get("facing", "south" if props.get("vertical_slab") else None)
        if b_facing:
            if normal_i == 5 and b_facing == "north":
                if getattr(adj_blk, "full_cube", False) or adj_facing == "south":
                    return True
            elif normal_i == 4 and b_facing == "south":
                if getattr(adj_blk, "full_cube", False) or adj_facing == "north":
                    return True
            elif normal_i == 1 and b_facing == "west":
                if getattr(adj_blk, "full_cube", False) or adj_facing == "east":
                    return True
            elif normal_i == 0 and b_facing == "east":
                if getattr(adj_blk, "full_cube", False) or adj_facing == "west":
                    return True
            elif adj_facing == b_facing:
                return True

    return False


def _emit_cutout(world, data, faces, total, x0, y0, z0, l_grid=None, grass_grid=None):
    if grass_grid is None:
        grass_grid = _build_grass_grid(world, x0 // CHUNK_SIZE, z0 // CHUNK_SIZE)
    vertices = np.empty((total * 4, 9), dtype=np.float32)
    indices = np.empty(total * 6, dtype=np.uint32)
    lights = np.empty((total * 4, 6), dtype=np.float32)
    colors = np.ones((total * 4, 3), dtype=np.float32)
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
                    if _is_glass_face_culled(world, wx, wy, wz, normal_i):
                        continue
                    m = _get_glass_face_ctm_mask(world, wx, wy, wz, normal_i, block=block)
                    face_uv = GLASS_CTM_UVS[m]
                    vertices[cursor : cursor + 4, 0:3] = positions[i]
                    vertices[cursor : cursor + 4, 3:6] = normal
                    vertices[cursor : cursor + 4, 6:8] = face_uv
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
