"""Data-driven crops that advance through growth stages.

Every file in ``assets/crops`` (``.json`` or ``.lua``) describes one crop.
``stages`` is either a list of block ids or a count, which registers
``{id}_0`` .. ``{id}_{n-1}`` along with the seed item, produce item, and loot.
See ``server/crop_data`` for seeds, produce, and wild plants.

    {
      "id": "wheat",
      "stages": 8,
      "min_time": 180,
      "max_time": 340,
      "needs_water": true,
      "water_radius": 4,
      "water_vertical": 1
    }

A stage timer only counts down while the crop's chunk is loaded. Crops that
need water wait, without losing progress, until fresh water is close enough.
One day cycle is 1200 seconds, so seven steps of 180–340 seconds take about
one to two days.
"""

import math
import random
import re

import numpy as np

from server.blocks import BLOCKS, CHUNK_SIZE, block_index, can_sustain_plant
from server.fluid import fresh_water_nearby

_ID = re.compile(r"[a-z0-9_]+")
SAVE_FILE = "crops.json"


class Crop:
    def __init__(self, crop_id, stages, min_time, max_time, needs_water, water_radius, water_vertical, source):
        self.id = crop_id
        self.stages = tuple(stages)
        self.stage_index = {block_id: index for index, block_id in enumerate(self.stages)}
        self.min_time = float(min_time)
        self.max_time = float(max_time)
        self.needs_water = bool(needs_water)
        self.water_radius = int(water_radius)
        self.water_vertical = int(water_vertical)
        self.source = str(source)

    def roll_delay(self):
        return random.uniform(self.min_time, self.max_time)

    def next_stage(self, block_id):
        index = self.stage_index.get(block_id)
        if index is None or index + 1 >= len(self.stages):
            return None
        return self.stages[index + 1]


def load_crops(folder=None):
    from server.crop_data import load_crop_defs

    crops = []
    seen_blocks = set()
    for spec in load_crop_defs(folder):
        for block_id in spec.stages:
            if block_id not in BLOCKS:
                raise ValueError(f"{spec.source}: unknown crop stage {block_id}")
            if block_id in seen_blocks:
                raise ValueError(f"{spec.source}: block {block_id} is already a crop stage")
            seen_blocks.add(block_id)
        crops.append(
            Crop(
                spec.id,
                spec.stages,
                spec.min_time,
                spec.max_time,
                spec.needs_water,
                spec.water_radius,
                spec.water_vertical,
                spec.source,
            )
        )
    return tuple(crops)


def crop_from_mapping(fields, source):
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: crop file must be an object")
    crop_id = fields.get("id")
    if not isinstance(crop_id, str) or _ID.fullmatch(crop_id) is None:
        raise ValueError(f"{source}: crop id must be a lowercase name")
    stages = fields.get("stages")
    if isinstance(stages, str):
        stages = [stages]
    if not isinstance(stages, (list, tuple)) or len(stages) < 2:
        raise ValueError(f"{source}: stages must list at least two block ids")
    resolved = []
    for block_id in stages:
        if not isinstance(block_id, str) or block_id not in BLOCKS:
            raise ValueError(f"{source}: unknown crop stage {block_id!r}")
        if block_id in resolved:
            raise ValueError(f"{source}: stage {block_id} is listed twice")
        resolved.append(block_id)
    min_time = _number(fields.get("min_time", 180), "min_time", source)
    max_time = _number(fields.get("max_time", 340), "max_time", source)
    if min_time <= 0 or max_time < min_time:
        raise ValueError(f"{source}: times must satisfy 0 < min_time <= max_time")
    radius = _number(fields.get("water_radius", 4), "water_radius", source)
    vertical = _number(fields.get("water_vertical", 1), "water_vertical", source)
    if int(radius) != radius or radius < 0 or int(vertical) != vertical or vertical < 0:
        raise ValueError(f"{source}: water radius must be a whole number of zero or more")
    return Crop(
        crop_id,
        resolved,
        min_time,
        max_time,
        bool(fields.get("needs_water", True)),
        int(radius),
        int(vertical),
        source,
    )


def _number(value, name, source):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: {name} must be a number")
    return float(value)


CROPS = load_crops()


class CropGrowthSystem:
    """Advances placed crops one stage at a time."""

    def __init__(self, world, crops=None):
        self.world = world
        self.crops = tuple(CROPS if crops is None else crops)
        self.by_block = {}
        for crop in self.crops:
            for block_id in crop.stages:
                self.by_block[block_id] = crop
        self.timers = {}  # (x, y, z) -> [crop, remaining]
        self.tick_timer = 0.0
        self.tick_interval = 0.5
        self.chunks_per_tick = 2
        self._cursor = 0
        self.load()

    def _store(self):
        return getattr(self.world, "store", None)

    def load(self):
        store = self._store()
        if store is None or not hasattr(store, "read_meta"):
            return
        payload = store.read_meta(SAVE_FILE)
        if not isinstance(payload, dict):
            return
        entries = payload.get("timers")
        if not isinstance(entries, list):
            return
        by_id = {crop.id: crop for crop in self.crops}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                pos = (int(entry["x"]), int(entry["y"]), int(entry["z"]))
                crop = by_id[str(entry["crop"])]
                remaining = float(entry["remaining"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(remaining) or remaining <= 0:
                remaining = 0.01
            self.timers[pos] = [crop, remaining]

    def save(self):
        store = self._store()
        if store is None or not hasattr(store, "write_meta"):
            return
        entries = [
            {
                "x": x,
                "y": y,
                "z": z,
                "crop": crop.id,
                "remaining": round(float(remaining), 3),
            }
            for (x, y, z), (crop, remaining) in sorted(self.timers.items())
        ]
        store.write_meta(SAVE_FILE, {"version": 1, "timers": entries})

    def schedule(self, x, y, z):
        pos = (int(x), int(y), int(z))
        block_id = self.world.get_block(*pos)
        crop = self.by_block.get(block_id)
        if crop is None or crop.next_stage(block_id) is None:
            self.timers.pop(pos, None)
            return False
        if not can_sustain_plant(self.world.get_block(pos[0], pos[1] - 1, pos[2]), block_id):
            return False
        existing = self.timers.get(pos)
        if existing is not None and existing[0] is crop:
            return False
        self.timers[pos] = [crop, crop.roll_delay()]
        return True

    def on_block_set(self, x, y, z, block_id):
        pos = (int(x), int(y), int(z))
        self.timers.pop(pos, None)
        if block_id in self.by_block:
            self.schedule(*pos)

    def scan_chunk(self, key):
        chunk = self.world.chunks.get(key)
        if chunk is None:
            return
        cx, cy, cz = key
        for crop in self.crops:
            for block_id in crop.stages:
                if crop.next_stage(block_id) is None:
                    continue
                for ly, lz, lx in np.argwhere(chunk.data == block_index(block_id)):
                    self.schedule(
                        cx * CHUNK_SIZE + int(lx),
                        cy * CHUNK_SIZE + int(ly),
                        cz * CHUNK_SIZE + int(lz),
                    )

    def _hydrated(self, crop, x, y, z):
        if not crop.needs_water:
            return True
        return fresh_water_nearby(
            self.world,
            x,
            y - 1,
            z,
            radius=crop.water_radius,
            vertical=crop.water_vertical,
        )

    def update(self, dt):
        if not self.crops:
            return []
        self.tick_timer += dt
        if self.tick_timer < self.tick_interval:
            return []
        step = self.tick_timer
        self.tick_timer = 0.0
        keys = list(self.world.chunks.keys())
        if keys:
            for _ in range(min(self.chunks_per_tick, len(keys))):
                self._cursor %= len(keys)
                self.scan_chunk(keys[self._cursor])
                self._cursor += 1

        grown = []
        for pos, entry in list(self.timers.items()):
            x, y, z = pos
            if self.world.chunk_key(x, y, z) not in self.world.chunks:
                continue
            crop, remaining = entry
            block_id = self.world.get_block(x, y, z)
            if self.by_block.get(block_id) is not crop:
                del self.timers[pos]
                if block_id in self.by_block:
                    self.schedule(x, y, z)
                continue
            if not can_sustain_plant(self.world.get_block(x, y - 1, z), block_id):
                del self.timers[pos]
                continue
            if not self._hydrated(crop, x, y, z):
                continue
            remaining -= step
            if remaining > 0:
                entry[1] = remaining
                continue
            nxt = crop.next_stage(block_id)
            del self.timers[pos]
            if nxt and self.world.set_block(x, y, z, nxt):
                grown.append((x, y, z, nxt))
        return grown
