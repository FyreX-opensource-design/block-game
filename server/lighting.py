from concurrent.futures import ThreadPoolExecutor

import numpy as np

CHUNK_SIZE = 16
WORLD_HEIGHT = 128
SECTION_COUNT = WORLD_HEIGHT // CHUNK_SIZE

LIGHT_MIN = 0
LIGHT_MAX = 15
LIGHT_CHANNELS = 5
CHANNEL_R = 0
CHANNEL_G = 1
CHANNEL_B = 2
CHANNEL_UV = 3
CHANNEL_IR = 4
CHANNEL_NAMES = ("red", "green", "blue", "uv", "infrared")


def parse_light_levels(val):
    """Parse light levels into a 5-tuple of integers (0..15) for R, G, B, UV, IR."""
    if val is None:
        return (0, 0, 0, 0, 0)
    if isinstance(val, (int, float)):
        v = int(max(0, min(LIGHT_MAX, round(val))))
        return (v, v, v, v, v)
    if isinstance(val, dict):
        r = int(val.get("r", val.get("red", 0)))
        g = int(val.get("g", val.get("green", 0)))
        b = int(val.get("b", val.get("blue", 0)))
        uv = int(val.get("uv", 0))
        ir = int(val.get("ir", val.get("infrared", 0)))
        return (
            max(0, min(LIGHT_MAX, r)),
            max(0, min(LIGHT_MAX, g)),
            max(0, min(LIGHT_MAX, b)),
            max(0, min(LIGHT_MAX, uv)),
            max(0, min(LIGHT_MAX, ir)),
        )
    if isinstance(val, (list, tuple)):
        if len(val) == 3:
            return (
                max(0, min(LIGHT_MAX, int(val[0]))),
                max(0, min(LIGHT_MAX, int(val[1]))),
                max(0, min(LIGHT_MAX, int(val[2]))),
                0,
                0,
            )
        if len(val) >= 5:
            return tuple(max(0, min(LIGHT_MAX, int(val[i]))) for i in range(5))
        if len(val) == 1:
            v = int(max(0, min(LIGHT_MAX, val[0])))
            return (v, v, v, v, v)
    return (0, 0, 0, 0, 0)


def get_block_opacity(block_obj):
    """Return opacity 0..15 of a block."""
    if block_obj is None:
        return 0
    if hasattr(block_obj, "properties") and "light_opacity" in block_obj.properties:
        return int(block_obj.properties["light_opacity"])
    bid = getattr(block_obj, "id", "")
    base = getattr(block_obj, "base_id", bid)
    if bid == "air":
        return 0
    if bid == "leaves" or base == "leaves" or "leaves" in bid or (hasattr(block_obj, "properties") and block_obj.properties.get("leaf")):
        return 1
    if bid.startswith("water") or (hasattr(block_obj, "properties") and block_obj.properties.get("fluid") == "water"):
        return 2
    if not getattr(block_obj, "solid", True):
        return 0
    props = getattr(block_obj, "properties", None) or {}
    if props.get("slab") and not props.get("double_slab"):
        if props.get("waterlogged") or getattr(block_obj, "waterlogged", False):
            return 2
        return 0
    if getattr(block_obj, "full_cube", False):
        return 15
    opacity = getattr(block_obj, "opacity", 1.0)
    return int(round(opacity * 15))


def get_block_emission(block_obj):
    """Return emission (r, g, b, uv, ir) tuple of a block definition."""
    if block_obj is None:
        return (0, 0, 0, 0, 0)
    if isinstance(block_obj, str):
        from server.blocks import BLOCKS

        block_obj = BLOCKS.get(block_obj)
        if block_obj is None:
            return (0, 0, 0, 0, 0)
    if hasattr(block_obj, "emission") and block_obj.emission:
        return block_obj.emission
    if hasattr(block_obj, "light") and block_obj.light:
        return block_obj.light
    return (0, 0, 0, 0, 0)


_OPACITY_LUT = None
_EMISSION_LUT = None


def _get_luts():
    global _OPACITY_LUT, _EMISSION_LUT
    if _OPACITY_LUT is not None:
        return _OPACITY_LUT, _EMISSION_LUT
    from server.blocks import BY_INDEX

    op_lut = np.zeros(256, dtype=np.int16)
    em_lut = np.zeros((256, 5), dtype=np.int16)
    for i, b in enumerate(BY_INDEX):
        if b is not None:
            op_lut[i] = get_block_opacity(b)
            em_lut[i] = get_block_emission(b)
    _OPACITY_LUT = op_lut
    _EMISSION_LUT = em_lut
    return _OPACITY_LUT, _EMISSION_LUT


class LightingSystem:
    def __init__(self, world, max_workers=4):
        self.world = world
        self.light_sources = {}  # (x, y, z) -> (r, g, b, uv, ir)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lighting")

    def close(self):
        self._executor.shutdown(wait=False)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def compute_chunks_parallel(self, chunk_keys):
        """Compute lighting for multiple chunks concurrently in worker threads."""
        keys = [k for k in chunk_keys if k in self.world.chunks]
        if not keys:
            return
        if len(keys) == 1:
            self.compute_chunk_lighting(*keys[0])
            return
        futures = [self._executor.submit(self.compute_chunk_lighting, *k) for k in keys]
        for f in futures:
            f.result()

    def get_opacity(self, x, y, z):
        if y < 0:
            return 15
        if y >= WORLD_HEIGHT:
            return 0
        block_id = self.world.get_block(x, y, z)
        from server.blocks import BLOCKS

        block = BLOCKS.get(block_id)
        return get_block_opacity(block)

    def get_emission(self, x, y, z, block_id=None):
        key = (int(x), int(y), int(z))
        if key in self.light_sources:
            return self.light_sources[key]
        furnace = self.world.get_furnace(x, y, z)
        if furnace is not None and furnace.is_lit:
            # Lit furnace emits high warm RGB, low UV, max Infrared
            return (15, 11, 4, 2, 15)
        if block_id is None:
            block_id = self.world.get_block(x, y, z)
        from server.blocks import BLOCKS

        block = BLOCKS.get(block_id)
        return get_block_emission(block)

    def get_skylight(self, x, y, z):
        if y >= WORLD_HEIGHT:
            return 15
        if y < 0:
            return 0
        chunk = self.world.chunk_at(x, y, z)
        if chunk is None:
            top = self.world.column_top(x, z)
            if y < top:
                return 0
            from server.blocks import WATER_LEVEL

            if y <= WATER_LEVEL and top < WATER_LEVEL:
                depth = WATER_LEVEL - y
                return max(0, 15 - depth * 2)
            return 15
        self.ensure_chunk_lit(chunk)
        lx = x - chunk.cx * CHUNK_SIZE
        ly = y - chunk.cy * CHUNK_SIZE
        lz = z - chunk.cz * CHUNK_SIZE
        return int(chunk.skylight[ly, lz, lx])

    def get_blocklight(self, x, y, z):
        if y < 0 or y >= WORLD_HEIGHT:
            return (0, 0, 0, 0, 0)
        chunk = self.world.chunk_at(x, y, z)
        if chunk is None:
            return (0, 0, 0, 0, 0)
        self.ensure_chunk_lit(chunk)
        lx = x - chunk.cx * CHUNK_SIZE
        ly = y - chunk.cy * CHUNK_SIZE
        lz = z - chunk.cz * CHUNK_SIZE
        return tuple(int(v) for v in chunk.blocklight[ly, lz, lx])

    def get_light(self, x, y, z):
        sky = self.get_skylight(x, y, z)
        bl = self.get_blocklight(x, y, z)
        return {
            "sky": sky,
            "block": bl,
            "r": bl[0],
            "g": bl[1],
            "b": bl[2],
            "uv": bl[3],
            "ir": bl[4],
        }

    def set_blocklight(self, x, y, z, r, g, b, uv=0, ir=0):
        key = (int(x), int(y), int(z))
        levels = parse_light_levels((r, g, b, uv, ir))
        if any(v > 0 for v in levels):
            self.light_sources[key] = levels
        else:
            self.light_sources.pop(key, None)
        self.on_block_changed(x, y, z)

    def clear_blocklight(self, x, y, z):
        key = (int(x), int(y), int(z))
        if key in self.light_sources:
            del self.light_sources[key]
            self.on_block_changed(x, y, z)

    def ensure_chunk_lit(self, chunk):
        if not getattr(chunk, "light_valid", False):
            cx, cy, cz = chunk.cx, chunk.cy, chunk.cz
            # Ensure chunks directly above in column are lit first (top-down)
            for ty in range(SECTION_COUNT - 1, cy, -1):
                ach = self.world.chunks.get((cx, ty, cz))
                if ach is not None and not getattr(ach, "light_valid", False):
                    self.compute_chunk_lighting(cx, ty, cz)
            self.compute_chunk_lighting(chunk.cx, chunk.cy, chunk.cz)

    def compute_chunk_lighting(self, cx, cy, cz):
        chunk = self.world.chunks.get((cx, cy, cz))
        if chunk is None:
            return
        if not hasattr(chunk, "skylight"):
            chunk.skylight = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
        if not hasattr(chunk, "blocklight"):
            chunk.blocklight = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, 5), dtype=np.uint8)

        x0 = cx * CHUNK_SIZE
        y0 = cy * CHUNK_SIZE
        z0 = cz * CHUNK_SIZE

        op_lut, em_lut = _get_luts()

        # 1. Build padded opacity array: (18, 18, 18)
        op_grid = np.zeros((CHUNK_SIZE + 2, CHUNK_SIZE + 2, CHUNK_SIZE + 2), dtype=np.int16)
        op_grid[1:-1, 1:-1, 1:-1] = op_lut[chunk.data]

        below = self.world.chunks.get((cx, cy - 1, cz))
        above = self.world.chunks.get((cx, cy + 1, cz))
        left = self.world.chunks.get((cx - 1, cy, cz))
        right = self.world.chunks.get((cx + 1, cy, cz))
        near = self.world.chunks.get((cx, cy, cz - 1))
        far = self.world.chunks.get((cx, cy, cz + 1))

        if below is not None:
            op_grid[0, 1:-1, 1:-1] = op_lut[below.data[-1, :, :]]
        elif cy == 0:
            op_grid[0, 1:-1, 1:-1] = 15

        if above is not None:
            op_grid[-1, 1:-1, 1:-1] = op_lut[above.data[0, :, :]]

        if left is not None:
            op_grid[1:-1, 1:-1, 0] = op_lut[left.data[:, :, -1]]

        if right is not None:
            op_grid[1:-1, 1:-1, -1] = op_lut[right.data[:, :, 0]]

        if near is not None:
            op_grid[1:-1, 0, 1:-1] = op_lut[near.data[:, -1, :]]

        if far is not None:
            op_grid[1:-1, -1, 1:-1] = op_lut[far.data[:, 0, :]]

        # 2. Skylight direct vertical raycast
        sky_grid = np.zeros((CHUNK_SIZE + 2, CHUNK_SIZE + 2, CHUNK_SIZE + 2), dtype=np.int16)

        # Top boundary incoming light
        if above is not None and getattr(above, "light_valid", False):
            sky_grid[-1, 1:-1, 1:-1] = above.skylight[0, :, :]
        elif cy >= SECTION_COUNT - 1:
            sky_grid[-1, 1:-1, 1:-1] = 15
        else:
            col_chunks = [self.world.chunks.get((cx, ty, cz)) for ty in range(SECTION_COUNT - 1, cy, -1)]
            col_sky = np.full((CHUNK_SIZE, CHUNK_SIZE), 15, dtype=np.int16)
            for ch in col_chunks:
                if ch is not None:
                    op_ch = op_lut[ch.data]
                    for ly in range(CHUNK_SIZE - 1, -1, -1):
                        cur_op = op_ch[ly]
                        atten = np.where(cur_op == 0, 0, cur_op)
                        col_sky = np.maximum(0, col_sky - atten)
            sky_grid[-1, 1:-1, 1:-1] = col_sky

        # Copy boundary skylight from neighbors
        if below is not None and getattr(below, "light_valid", False):
            sky_grid[0, 1:-1, 1:-1] = below.skylight[-1, :, :]
        if left is not None and getattr(left, "light_valid", False):
            sky_grid[1:-1, 1:-1, 0] = left.skylight[:, :, -1]
        if right is not None and getattr(right, "light_valid", False):
            sky_grid[1:-1, 1:-1, -1] = right.skylight[:, :, 0]
        if near is not None and getattr(near, "light_valid", False):
            sky_grid[1:-1, 0, 1:-1] = near.skylight[:, -1, :]
        if far is not None and getattr(far, "light_valid", False):
            sky_grid[1:-1, -1, 1:-1] = far.skylight[:, 0, :]

        # Vertical rays downwards
        for ly in range(CHUNK_SIZE, 0, -1):
            cur_op = op_grid[ly, 1:-1, 1:-1]
            above_val = sky_grid[ly + 1, 1:-1, 1:-1]
            atten = np.where(cur_op == 0, 0, cur_op)
            sky_grid[ly, 1:-1, 1:-1] = np.maximum(0, above_val - atten)

        # Vectorized skylight relaxation (diffuse into shadows/caves)
        sky_step = np.maximum(1, op_grid[1:-1, 1:-1, 1:-1])
        for _ in range(15):
            prev = sky_grid[1:-1, 1:-1, 1:-1].copy()
            cands = [
                sky_grid[1:-1, 1:-1, 1:-1],
                sky_grid[2:, 1:-1, 1:-1] - sky_step,
                sky_grid[:-2, 1:-1, 1:-1] - sky_step,
                sky_grid[1:-1, 2:, 1:-1] - sky_step,
                sky_grid[1:-1, :-2, 1:-1] - sky_step,
                sky_grid[1:-1, 1:-1, 2:] - sky_step,
                sky_grid[1:-1, 1:-1, :-2] - sky_step,
            ]
            sky_grid[1:-1, 1:-1, 1:-1] = np.maximum(0, np.maximum.reduce(cands))
            if np.array_equal(sky_grid[1:-1, 1:-1, 1:-1], prev):
                break

        chunk.skylight = sky_grid[1:-1, 1:-1, 1:-1].astype(np.uint8)

        # 3. Blocklight calculation
        bl_grid = np.zeros((CHUNK_SIZE + 2, CHUNK_SIZE + 2, CHUNK_SIZE + 2, 5), dtype=np.int16)
        has_emit = False

        em_chunk = em_lut[chunk.data]
        if np.any(em_chunk > 0):
            bl_grid[1:-1, 1:-1, 1:-1] = np.maximum(bl_grid[1:-1, 1:-1, 1:-1], em_chunk)
            has_emit = True

        # Check lit furnaces in chunk bounds
        if getattr(self.world, "furnaces", None):
            for (fx, fy, fz), furnace in self.world.furnaces.items():
                if x0 <= fx < x0 + CHUNK_SIZE and y0 <= fy < y0 + CHUNK_SIZE and z0 <= fz < z0 + CHUNK_SIZE:
                    if getattr(furnace, "is_lit", False):
                        bl_grid[fy - y0 + 1, fz - z0 + 1, fx - x0 + 1] = np.maximum(
                            bl_grid[fy - y0 + 1, fz - z0 + 1, fx - x0 + 1],
                            (15, 11, 4, 2, 15),
                        )
                        has_emit = True

        # Check registered light sources
        if self.light_sources:
            for (sx, sy, sz), emit in self.light_sources.items():
                if x0 <= sx < x0 + CHUNK_SIZE and y0 <= sy < y0 + CHUNK_SIZE and z0 <= sz < z0 + CHUNK_SIZE:
                    bl_grid[sy - y0 + 1, sz - z0 + 1, sx - x0 + 1] = np.maximum(
                        bl_grid[sy - y0 + 1, sz - z0 + 1, sx - x0 + 1], emit
                    )
                    has_emit = True

        # Copy boundary blocklight from neighbors if present
        if below is not None and getattr(below, "light_valid", False):
            if np.any(below.blocklight[-1, :, :] > 0):
                bl_grid[0, 1:-1, 1:-1] = below.blocklight[-1, :, :]
                has_emit = True
        if above is not None and getattr(above, "light_valid", False):
            if np.any(above.blocklight[0, :, :] > 0):
                bl_grid[-1, 1:-1, 1:-1] = above.blocklight[0, :, :]
                has_emit = True
        if left is not None and getattr(left, "light_valid", False):
            if np.any(left.blocklight[:, :, -1] > 0):
                bl_grid[1:-1, 1:-1, 0] = left.blocklight[:, :, -1]
                has_emit = True
        if right is not None and getattr(right, "light_valid", False):
            if np.any(right.blocklight[:, :, 0] > 0):
                bl_grid[1:-1, 1:-1, -1] = right.blocklight[:, :, 0]
                has_emit = True
        if near is not None and getattr(near, "light_valid", False):
            if np.any(near.blocklight[:, -1, :] > 0):
                bl_grid[1:-1, 0, 1:-1] = near.blocklight[:, -1, :]
                has_emit = True
        if far is not None and getattr(far, "light_valid", False):
            if np.any(far.blocklight[:, 0, :] > 0):
                bl_grid[1:-1, -1, 1:-1] = far.blocklight[:, 0, :]
                has_emit = True

        if has_emit:
            bl_step = np.maximum(1, op_grid[1:-1, 1:-1, 1:-1, None])
            for _ in range(15):
                prev = bl_grid[1:-1, 1:-1, 1:-1].copy()
                cands = [
                    bl_grid[1:-1, 1:-1, 1:-1],
                    bl_grid[2:, 1:-1, 1:-1] - bl_step,
                    bl_grid[:-2, 1:-1, 1:-1] - bl_step,
                    bl_grid[1:-1, 2:, 1:-1] - bl_step,
                    bl_grid[1:-1, :-2, 1:-1] - bl_step,
                    bl_grid[1:-1, 1:-1, 2:] - bl_step,
                    bl_grid[1:-1, 1:-1, :-2] - bl_step,
                ]
                bl_grid[1:-1, 1:-1, 1:-1] = np.maximum(0, np.maximum.reduce(cands))
                if np.array_equal(bl_grid[1:-1, 1:-1, 1:-1], prev):
                    break
            chunk.blocklight = bl_grid[1:-1, 1:-1, 1:-1].astype(np.uint8)
        else:
            chunk.blocklight.fill(0)

        chunk.light_valid = True

        # Propagate boundary skylight spillover into neighbors
        sky_spill = (
            (cx + 1, cy, cz, chunk.skylight[:, :, -1], lambda ch: ch.skylight[:, :, 0], lambda ch: ch.data[:, :, 0]),
            (cx - 1, cy, cz, chunk.skylight[:, :, 0], lambda ch: ch.skylight[:, :, -1], lambda ch: ch.data[:, :, -1]),
            (cx, cy, cz + 1, chunk.skylight[:, -1, :], lambda ch: ch.skylight[:, 0, :], lambda ch: ch.data[:, 0, :]),
            (cx, cy, cz - 1, chunk.skylight[:, 0, :], lambda ch: ch.skylight[:, -1, :], lambda ch: ch.data[:, -1, :]),
            (cx, cy - 1, cz, chunk.skylight[0, :, :], lambda ch: ch.skylight[-1, :, :], lambda ch: ch.data[-1, :, :]),
            (cx, cy + 1, cz, chunk.skylight[-1, :, :], lambda ch: ch.skylight[0, :, :], lambda ch: ch.data[0, :, :]),
        )
        for ncx, ncy, ncz, edge, n_fn, n_data_fn in sky_spill:
            if 0 <= ncy < SECTION_COUNT:
                nch = self.world.chunks.get((ncx, ncy, ncz))
                if nch is not None and getattr(nch, "light_valid", False):
                    n_step = np.maximum(1, op_lut[n_data_fn(nch)])
                    potential = np.maximum(0, edge.astype(np.int16) - n_step)
                    diff = potential - n_fn(nch).astype(np.int16)
                    if np.any(diff > 0):
                        nch.light_valid = False
                        self.world.mark_dirty(ncx, ncy, ncz)

        # Propagate boundary blocklight spillover into neighbors
        if has_emit:
            bl_spill = (
                (cx + 1, cy, cz, chunk.blocklight[:, :, -1], lambda ch: ch.blocklight[:, :, 0], lambda ch: ch.data[:, :, 0]),
                (cx - 1, cy, cz, chunk.blocklight[:, :, 0], lambda ch: ch.blocklight[:, :, -1], lambda ch: ch.data[:, :, -1]),
                (cx, cy, cz + 1, chunk.blocklight[:, -1, :], lambda ch: ch.blocklight[:, 0, :], lambda ch: ch.data[:, 0, :]),
                (cx, cy, cz - 1, chunk.blocklight[:, 0, :], lambda ch: ch.blocklight[:, -1, :], lambda ch: ch.data[:, -1, :]),
                (cx, cy - 1, cz, chunk.blocklight[0, :, :], lambda ch: ch.blocklight[-1, :, :], lambda ch: ch.data[-1, :, :]),
                (cx, cy + 1, cz, chunk.blocklight[-1, :, :], lambda ch: ch.blocklight[0, :, :], lambda ch: ch.data[0, :, :]),
            )
            for ncx, ncy, ncz, edge, n_fn, n_data_fn in bl_spill:
                if 0 <= ncy < SECTION_COUNT:
                    nch = self.world.chunks.get((ncx, ncy, ncz))
                    if nch is not None and getattr(nch, "light_valid", False):
                        n_step = np.maximum(1, op_lut[n_data_fn(nch)])[..., None]
                        potential = np.maximum(0, edge.astype(np.int16) - n_step)
                        diff = potential - n_fn(nch).astype(np.int16)
                        if np.any(diff > 0):
                            nch.light_valid = False
                            self.world.mark_dirty(ncx, ncy, ncz)

    def on_block_changed(self, x, y, z):
        """Called when a block is placed, broken, or updated at (x, y, z)."""
        cx = x // CHUNK_SIZE
        cy = y // CHUNK_SIZE
        cz = z // CHUNK_SIZE
        lx = x % CHUNK_SIZE
        ly = y % CHUNK_SIZE
        lz = z % CHUNK_SIZE

        affected = {(cx, cy, cz)}

        # Direct face neighbors across boundaries
        if lx == 0:
            affected.add((cx - 1, cy, cz))
        elif lx == CHUNK_SIZE - 1:
            affected.add((cx + 1, cy, cz))

        if lz == 0:
            affected.add((cx, cy, cz - 1))
        elif lz == CHUNK_SIZE - 1:
            affected.add((cx, cy, cz + 1))

        if ly == 0 and cy > 0:
            affected.add((cx, cy - 1, cz))
        elif ly == CHUNK_SIZE - 1 and cy < SECTION_COUNT - 1:
            affected.add((cx, cy + 1, cz))

        # Check if block emits light or is a registered light source
        emit = self.get_emission(x, y, z)
        is_emitter = any(v > 0 for v in emit) or (int(x), int(y), int(z)) in self.light_sources
        if is_emitter:
            for dcy, dcz, dcx in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
                ty = cy + dcy
                if 0 <= ty < SECTION_COUNT:
                    affected.add((cx + dcx, ty, cz + dcz))

        # Vertical skylight downward propagation (only if sunlight can reach this height)
        if y >= WORLD_HEIGHT - 1 or self.get_skylight(x, y + 1, z) > 0:
            for dcy in range(1, 4):
                if cy - dcy >= 0:
                    affected.add((cx, cy - dcy, cz))

        valid_keys = [k for k in affected if k in self.world.chunks]
        for ckey in valid_keys:
            self.world.chunks[ckey].light_valid = False
            self.world.mark_dirty(*ckey)

        # Immediately recompute lighting for the primary chunk so light queries on it are instant
        self.compute_chunk_lighting(cx, cy, cz)

