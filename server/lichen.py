"""Fluorescent lichen spreads slowly while it sits in UV light.

A patch already on a block can take another face of the same cell, or step
into a neighboring empty cell that has something solid to hold it.
"""

import random

import numpy as np

from server.blocks import AIR, VINE_BITS, block_index, vine_block_id, vine_info, vine_supported_mask

MIN_UV = 4
MIN_TIME = 50.0
MAX_TIME = 110.0
_NEIGHBORS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _uv_level(world, x, y, z):
    light = world.get_blocklight(x, y, z)
    if light is None:
        return 0
    if isinstance(light, dict):
        return int(light.get("uv", 0) or 0)
    return int(light[3])


def spread_targets(world, x, y, z):
    """Cells and masks this lichen can grow into. Empty when it has no UV."""
    info = vine_info(world.get_block(x, y, z))
    if info is None or info["base_id"] != "fluor_lichen":
        return []
    if _uv_level(world, x, y, z) < MIN_UV:
        return []
    found = []
    mask = int(info["mask"])
    for flag, _name, _offset, _origin, _extent in VINE_BITS:
        if mask & flag:
            continue
        if vine_supported_mask(world, x, y, z, flag) == flag:
            found.append((x, y, z, mask | flag))
    for dx, dy, dz in _NEIGHBORS:
        nx, ny, nz = x + dx, y + dy, z + dz
        if ny <= 0:
            continue
        there = world.get_block(nx, ny, nz)
        if there not in (AIR, "", None):
            continue
        for flag, _name, _offset, _origin, _extent in VINE_BITS:
            if vine_supported_mask(world, nx, ny, nz, flag) == flag:
                found.append((nx, ny, nz, flag))
                break
    return found


def grow_once(world, x, y, z):
    """Add one face or one neighboring patch. Returns the new block id, or None."""
    choices = spread_targets(world, x, y, z)
    if not choices:
        return None
    nx, ny, nz, mask = random.choice(choices)
    block_id = vine_block_id("fluor_lichen", mask)
    if world.set_block(nx, ny, nz, block_id):
        return block_id
    return None


class LichenGrowthSystem:
    """Counts down patches that are actually lit, then lets one of them spread."""

    def __init__(self, world):
        self.world = world
        self.timers = {}
        self._wait = 0.0
        self._cursor = 0

    def update(self, dt):
        self._wait += dt
        if self._wait < 1.0:
            return []
        step = self._wait
        self._wait = 0.0
        self._scan()
        grown = []
        for pos, remaining in list(self.timers.items()):
            if _uv_level(self.world, *pos) < MIN_UV:
                continue
            remaining -= step
            if remaining > 0:
                self.timers[pos] = remaining
                continue
            made = grow_once(self.world, *pos)
            self.timers[pos] = random.uniform(MIN_TIME, MAX_TIME)
            if made is not None:
                grown.append((pos, made))
        return grown

    def _scan(self):
        chunks = list(self.world.chunks)
        if not chunks:
            return
        if self._cursor % 8 == 0:
            live = set(chunks)
            self.timers = {
                pos: remaining
                for pos, remaining in self.timers.items()
                if (pos[0] // 16, pos[1] // 16, pos[2] // 16) in live
            }
        ids = _lichen_indices()
        for _ in range(2):
            key = chunks[self._cursor % len(chunks)]
            self._cursor += 1
            chunk = self.world.chunks.get(key)
            if chunk is None:
                continue
            cx, cy, cz = key
            for ly, lz, lx in np.argwhere(np.isin(chunk.data, ids)):
                pos = (cx * 16 + int(lx), cy * 16 + int(ly), cz * 16 + int(lz))
                if pos not in self.timers:
                    self.timers[pos] = random.uniform(MIN_TIME, MAX_TIME)


def _lichen_indices():
    found = []
    base = block_index("fluor_lichen")
    if base is not None:
        found.append(base)
    for mask in range(1, 64):
        index = block_index(f"fluor_lichen_v{mask}")
        if index is not None:
            found.append(index)
    return np.array(found or [0], dtype=np.uint16)
