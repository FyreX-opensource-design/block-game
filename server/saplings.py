"""Data-driven saplings that grow into trees after a random delay.

Every file in ``assets/saplings`` (``.json`` or ``.lua``) describes one sapling:

    {
      "id": "oak_sapling",
      "block": "oak_sapling",                 # placed block
      "trees": ["oak", "fancy_oak", "tall_oak"],
      "min_time": 900,                        # seconds before growth (15 min)
      "max_time": 1800                        # seconds before growth (30 min)
    }

Saplings are tracked by world position. A timer only advances while the sapling's
chunk is loaded, so a sapling that is left alone keeps its remaining growth time.
"""

import json
import math
import random
import re
from pathlib import Path

import numpy as np

from server.blocks import ASSETS, BLOCKS, CHUNK_SIZE, block_index, can_sustain_plant
from server.lua import load_table
from server.trees import DECORATORS

_ID = re.compile(r"[a-z0-9_]+")
DEFAULT_MIN_TIME = 15 * 60.0
DEFAULT_MAX_TIME = 30 * 60.0
SAVE_FILE = "saplings.json"


class Sapling:
    def __init__(self, sapling_id, block, trees, min_time, max_time, source):
        self.id = sapling_id
        self.block = block
        self.trees = tuple(trees)
        self.tree_ids = tuple(tree.id for tree in self.trees)
        self.min_time = float(min_time)
        self.max_time = float(max_time)
        self.source = str(source)

    def roll_delay(self):
        return random.uniform(self.min_time, self.max_time)


def load_saplings(folder=None, trees=None):
    root = Path(folder) if folder is not None else ASSETS / "saplings"
    if not root.is_dir():
        return ()
    catalog = {tree.id: tree for tree in (DECORATORS if trees is None else trees)}
    saplings = []
    seen_ids = set()
    seen_blocks = set()
    for path in sorted(root.iterdir()):
        fields = _read_file(path)
        if fields is None:
            continue
        sapling = sapling_from_mapping(fields, path, catalog.values())
        if sapling.id in seen_ids:
            raise ValueError(f"{path}: duplicate sapling id {sapling.id}")
        if sapling.block in seen_blocks:
            raise ValueError(f"{path}: block {sapling.block} already has a sapling")
        seen_ids.add(sapling.id)
        seen_blocks.add(sapling.block)
        saplings.append(sapling)
    return tuple(saplings)


def sapling_from_mapping(fields, source, trees=None):
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: sapling file must be an object")
    sapling_id = fields.get("id")
    if not isinstance(sapling_id, str) or _ID.fullmatch(sapling_id) is None:
        raise ValueError(f"{source}: sapling id must be a lowercase name")
    block = fields.get("block", sapling_id)
    if not isinstance(block, str) or block not in BLOCKS:
        raise ValueError(f"{source}: block must be a known block id, got {block!r}")

    catalog = {tree.id: tree for tree in (DECORATORS if trees is None else trees)}
    raw_trees = fields.get("trees")
    if isinstance(raw_trees, str):
        raw_trees = [raw_trees]
    if not isinstance(raw_trees, (list, tuple)) or not raw_trees:
        raise ValueError(f"{source}: trees must be a non-empty list of tree ids")
    resolved = []
    seen_trees = set()
    for tree_id in raw_trees:
        if not isinstance(tree_id, str) or tree_id not in catalog:
            raise ValueError(f"{source}: unknown tree {tree_id!r}")
        if tree_id not in seen_trees:
            seen_trees.add(tree_id)
            resolved.append(catalog[tree_id])

    min_time = _number(fields.get("min_time", DEFAULT_MIN_TIME), "min_time", source)
    max_time = _number(fields.get("max_time", DEFAULT_MAX_TIME), "max_time", source)
    if min_time <= 0 or max_time < min_time:
        raise ValueError(f"{source}: times must satisfy 0 < min_time <= max_time")

    return Sapling(sapling_id, block, resolved, min_time, max_time, source)


def _read_file(path):
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".lua":
        return load_table(path)
    return None


def _number(value, name, source):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: {name} must be a number")
    return float(value)


SAPLINGS = load_saplings()


class SaplingGrowthSystem:
    """Schedules placed saplings and grows them into a randomly chosen tree."""

    def __init__(self, world, saplings=None):
        self.world = world
        self.saplings = tuple(SAPLINGS if saplings is None else saplings)
        self.by_block = {sapling.block: sapling for sapling in self.saplings}
        self.timers = {}  # (x, y, z) -> [sapling, remaining_seconds]
        self.tick_timer = 0.0
        self.tick_interval = 0.5
        self.chunks_per_tick = 2
        self._cursor = 0
        self.load()

    def _store(self):
        return getattr(self.world, "store", None)

    def load(self):
        """Restore saved growth timers from the world's sidecar file."""
        store = self._store()
        if store is None or not hasattr(store, "read_meta"):
            return
        payload = store.read_meta(SAVE_FILE)
        if not isinstance(payload, dict):
            return
        entries = payload.get("timers")
        if not isinstance(entries, list):
            return
        by_id = {sapling.id: sapling for sapling in self.saplings}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                x = int(entry["x"])
                y = int(entry["y"])
                z = int(entry["z"])
                sapling = by_id[str(entry["sapling"])]
                remaining = float(entry["remaining"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(remaining) or remaining <= 0:
                remaining = 0.01
            self.timers[(x, y, z)] = [sapling, remaining]

    def save(self):
        """Write current growth timers to the world's sidecar file."""
        store = self._store()
        if store is None or not hasattr(store, "write_meta"):
            return
        entries = [
            {
                "x": x,
                "y": y,
                "z": z,
                "sapling": sapling.id,
                "remaining": round(float(remaining), 3),
            }
            for (x, y, z), (sapling, remaining) in sorted(self.timers.items())
        ]
        store.write_meta(SAVE_FILE, {"version": 1, "timers": entries})

    def schedule(self, x, y, z):
        """Start a growth timer for the sapling at (x, y, z), if it can grow."""
        pos = (int(x), int(y), int(z))
        block_id = self.world.get_block(*pos)
        sapling = self.by_block.get(block_id)
        if sapling is None:
            return False
        if not can_sustain_plant(self.world.get_block(pos[0], pos[1] - 1, pos[2]), block_id):
            return False
        existing = self.timers.get(pos)
        if existing is not None and existing[0].block == block_id:
            return False
        self.timers[pos] = [sapling, sapling.roll_delay()]
        return True

    def on_block_set(self, x, y, z, block_id):
        """Keep timers in sync when a block is placed, broken, or replaced."""
        pos = (int(x), int(y), int(z))
        self.timers.pop(pos, None)
        if block_id in self.by_block:
            self.schedule(*pos)

    def scan_chunk(self, key):
        """Schedule any saplings already present in a chunk (e.g. after loading a save)."""
        chunk = self.world.chunks.get(key)
        if chunk is None:
            return
        cx, cy, cz = key
        for sapling in self.saplings:
            for ly, lz, lx in np.argwhere(chunk.data == block_index(sapling.block)):
                pos = (
                    cx * CHUNK_SIZE + int(lx),
                    cy * CHUNK_SIZE + int(ly),
                    cz * CHUNK_SIZE + int(lz),
                )
                self.schedule(*pos)

    def update(self, dt):
        self.tick_timer += dt
        if self.tick_timer < self.tick_interval:
            return []

        step = self.tick_timer
        self.tick_timer = 0.0
        self._scan_chunks()
        grown = []

        for pos, entry in list(self.timers.items()):
            x, y, z = pos
            # Pause timers while the sapling's chunk is unloaded.
            if self.world.chunk_key(x, y, z) not in self.world.chunks:
                continue
            sapling, remaining = entry
            block_id = self.world.get_block(x, y, z)
            if block_id != sapling.block:
                del self.timers[pos]
                replacement = self.by_block.get(block_id)
                if replacement is not None:
                    self.timers[pos] = [replacement, replacement.roll_delay()]
                continue
            if not can_sustain_plant(self.world.get_block(x, y - 1, z), block_id):
                del self.timers[pos]
                continue

            remaining -= step
            if remaining > 0:
                entry[1] = remaining
                continue

            del self.timers[pos]
            if not sapling.trees:
                continue
            tree = random.choice(sapling.trees)
            if self.world.grow_tree(x, y - 1, z, tree):
                grown.append((x, y, z, tree.id))

        return grown

    def _scan_chunks(self):
        keys = list(self.world.chunks.keys())
        if not keys:
            return
        for _ in range(min(self.chunks_per_tick, len(keys))):
            self._cursor %= len(keys)
            self.scan_chunk(keys[self._cursor])
            self._cursor += 1
