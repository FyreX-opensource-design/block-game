"""Snowfall on biomes whose precipitation is snow.

Rain strength still comes from the weather grid. In a snow biome that wetness
lays snow layers on open ground instead of rain, one layer at a time.
"""

import math
import random

from server.blocks import AIR, BLOCKS, CHUNK_SIZE, WORLD_HEIGHT
from server.weather import rain_at

MAX_SNOW_LAYERS = 8


def snow_layer_block(layers):
    layers = max(1, min(MAX_SNOW_LAYERS, int(layers)))
    if layers == 1:
        return "snow_layer"
    return f"snow_layer_{layers}"


def _layers(block_id):
    block = BLOCKS.get(block_id)
    if block is None:
        return 0
    return int((block.properties or {}).get("snow_layers") or 0)


def snow_rise(world, x, y, z, dx, dz, radius):
    """How far snow on a one-block step sticks up above ``y``.

    ``y`` is the feet height after clearing the block itself. A non-snow block
    in that space means the step is a real wall, so the extra rise is zero.
    """
    min_x = min(x, x + dx) - radius
    max_x = max(x, x + dx) + radius
    min_z = min(z, z + dz) - radius
    max_z = max(z, z + dz) + radius
    ix0 = math.floor(min_x)
    ix1 = math.floor(max_x - 1e-6)
    iz0 = math.floor(min_z)
    iz1 = math.floor(max_z - 1e-6)
    iy0 = math.floor(y)
    iy1 = math.floor(y + 1.0 - 1e-6)
    extra = 0.0
    for by in range(iy0, iy1 + 1):
        for bz in range(iz0, iz1 + 1):
            for bx in range(ix0, ix1 + 1):
                block = BLOCKS.get(world.get_block(bx, by, bz))
                if block is None or not block.solid:
                    continue
                for box in world.get_collision_boxes(bx, by, bz):
                    if box[4] <= y + 1e-4 or box[1] >= y + 1.0 - 1e-4:
                        continue
                    if max_x <= box[0] or min_x >= box[3] or max_z <= box[2] or min_z >= box[5]:
                        continue
                    if not int((block.properties or {}).get("snow_layers") or 0):
                        return 0.0
                    extra = max(extra, box[4] - y)
    return extra


def _holds_snow(block_id):
    block = BLOCKS.get(block_id)
    if block is None or not block.solid or not block.full_cube:
        return False
    props = block.properties or {}
    if props.get("leaf") or props.get("snow_layers") or props.get("fluid") or block.plant:
        return False
    return True


class SnowSystem:
    """Adds a snow layer to a few open columns while snow is falling."""

    def __init__(self, world):
        self.world = world
        self.tick_timer = 0.0
        self.tick_interval = 0.4
        self._rng = random.Random()

    def update(self, dt):
        self.tick_timer += dt
        if self.tick_timer < self.tick_interval:
            return []
        self.tick_timer = 0.0
        changed = []
        for x, z in self._sample_columns():
            hit = self.accumulate(x, z)
            if hit is None:
                continue
            changed.append(hit)
            if len(changed) >= 12:
                break
        return changed

    def _sample_columns(self):
        chunks = list(self.world.chunks)
        if not chunks:
            return []
        picks = []
        for _ in range(24):
            cx, _cy, cz = self._rng.choice(chunks)
            lx = self._rng.randrange(CHUNK_SIZE)
            lz = self._rng.randrange(CHUNK_SIZE)
            picks.append((cx * CHUNK_SIZE + lx, cz * CHUNK_SIZE + lz))
        return picks

    def accumulate(self, x, z):
        """Lay or thicken one snow layer at this column. Returns the edit, or None."""
        world = self.world
        x, z = int(x), int(z)
        if rain_at(getattr(world, "weather", None), x, z) < 0.12:
            return None
        biome = world.get_biome(x, z)
        if getattr(biome, "precipitation", "rain") != "snow":
            return None
        y = world.column_top(x, z)
        for _ in range(48):
            if y <= 1:
                return None
            block_id = world.get_block(x, y, z)
            layers = _layers(block_id)
            if layers:
                if layers >= MAX_SNOW_LAYERS:
                    return None
                above = world.get_block(x, y + 1, z)
                if above not in (AIR, "", None):
                    return None
                placed = snow_layer_block(layers + 1)
                if not world.set_block(x, y, z, placed):
                    return None
                return (x, y, z, placed)
            block = BLOCKS.get(block_id)
            props = (block.properties if block else {}) or {}
            if block is not None and (props.get("leaf") or block.plant or (block.solid and not block.full_cube)):
                y -= 1
                continue
            if not _holds_snow(block_id):
                return None
            if y + 1 >= WORLD_HEIGHT:
                return None
            above = world.get_block(x, y + 1, z)
            if above not in (AIR, "", None):
                return None
            if not world.set_block(x, y + 1, z, "snow_layer"):
                return None
            return (x, y + 1, z, "snow_layer")
        return None
