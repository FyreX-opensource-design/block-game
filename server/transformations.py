"""Data-driven block transformations (for example grass slowly spreading over dirt).

Every file in ``assets/transformations`` (``.json`` or ``.lua``) describes one rule:

    {
      "id": "dirt_to_grass",
      "from": "dirt",                      # block that gets transformed
      "to": "grass",                       # block it becomes
      "sources": ["grass", "#some_tag"],   # nearby blocks that cause the spread ("#" = block tag)
      "min_time": 120,                     # seconds before the change happens (random in range)
      "max_time": 180,
      "radius": 1,                         # horizontal reach of a source block
      "vertical_radius": 1,                # vertical reach of a source block
      "require_clear_above": true,         # the block above must let light through
      "max_above_opacity": 1,              # light opacity (0-15) allowed above the block
      "enabled": true
    }

Only ``id``, ``from``, ``to`` and ``sources`` are required.
"""

import json
import random
import re
from pathlib import Path

import numpy as np

from server.blocks import ASSETS, AIR, BLOCKS, CHUNK_SIZE, block_index
from server.lua import load_table

_ID = re.compile(r"[a-z0-9_]+")


class Transformation:
    def __init__(
        self,
        rule_id,
        from_block,
        to_block,
        sources,
        min_time=120.0,
        max_time=180.0,
        radius=1,
        vertical_radius=1,
        require_clear_above=True,
        max_above_opacity=1,
        enabled=True,
        source=None,
    ):
        self.id = rule_id
        self.from_block = from_block
        self.to_block = to_block
        self.sources = tuple(sources)
        self.min_time = float(min_time)
        self.max_time = float(max_time)
        self.radius = int(radius)
        self.vertical_radius = int(vertical_radius)
        self.require_clear_above = bool(require_clear_above)
        self.max_above_opacity = int(max_above_opacity)
        self.enabled = bool(enabled)
        self.file = str(source) if source else rule_id

    def is_source(self, block_id):
        """True if block_id is one of this rule's spread sources."""
        if not block_id or block_id == AIR:
            return False
        for entry in self.sources:
            if entry.startswith("#"):
                from server.tags import has_block_tag

                if has_block_tag(block_id, entry[1:]):
                    return True
            elif entry == block_id:
                return True
        return False

    def roll_delay(self):
        return random.uniform(self.min_time, self.max_time)


def load_transformations(folder=None):
    root = Path(folder) if folder is not None else ASSETS / "transformations"
    if not root.is_dir():
        return ()
    rules = []
    seen = set()
    for path in sorted(root.iterdir()):
        if path.suffix == ".json":
            fields = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".lua":
            fields = load_table(path)
        else:
            continue
        rule = transformation_from_mapping(fields, path)
        if rule.id in seen:
            raise ValueError(f"{path}: duplicate transformation id {rule.id}")
        seen.add(rule.id)
        rules.append(rule)
    return tuple(rules)


def transformation_from_mapping(fields, source):
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: transformation file must be an object")
    rule_id = fields.get("id")
    if not isinstance(rule_id, str) or _ID.fullmatch(rule_id) is None:
        raise ValueError(f"{source}: transformation id must be a lowercase name")
    from_block = fields.get("from")
    to_block = fields.get("to")
    for name, value in (("from", from_block), ("to", to_block)):
        if not isinstance(value, str) or value not in BLOCKS:
            raise ValueError(f"{source}: {name} must be a known block id, got {value!r}")
    sources = fields.get("sources")
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError(f"{source}: sources must be a non-empty list")
    sources = [str(s) for s in sources]
    for entry in sources:
        if not entry.startswith("#") and entry not in BLOCKS:
            raise ValueError(f"{source}: unknown source block {entry}")

    values = {
        "min_time": 120.0,
        "max_time": 180.0,
        "radius": 1,
        "vertical_radius": 1,
        "max_above_opacity": 1,
    }
    for name in values:
        if name in fields:
            values[name] = _number(fields[name], name, source)
    if values["min_time"] <= 0 or values["max_time"] < values["min_time"]:
        raise ValueError(f"{source}: times must satisfy 0 < min_time <= max_time")
    for name in ("radius", "vertical_radius"):
        if values[name] < 0 or int(values[name]) != values[name]:
            raise ValueError(f"{source}: {name} must be a whole number of zero or more")
    if not 0 <= values["max_above_opacity"] <= 15:
        raise ValueError(f"{source}: max_above_opacity must be from 0 to 15")

    return Transformation(
        rule_id,
        from_block,
        to_block,
        sources,
        min_time=values["min_time"],
        max_time=values["max_time"],
        radius=int(values["radius"]),
        vertical_radius=int(values["vertical_radius"]),
        require_clear_above=bool(fields.get("require_clear_above", True)),
        max_above_opacity=int(values["max_above_opacity"]),
        enabled=bool(fields.get("enabled", True)),
        source=source,
    )


def _number(value, name, source):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: {name} must be a number")
    return float(value)


class TransformationSystem:
    """Finds blocks eligible for a transformation and applies them after a random delay."""

    def __init__(self, world, rules=None):
        self.world = world
        self.rules = tuple(r for r in (load_transformations() if rules is None else rules) if r.enabled)
        self.timers = {}  # (x, y, z) -> [rule, remaining_seconds]
        self.tick_timer = 0.0
        self.tick_interval = 0.5
        self.chunks_per_tick = 8
        self._cursor = 0

    # --- eligibility -------------------------------------------------------------

    def is_eligible(self, rule, x, y, z):
        w = self.world
        if w.get_block(x, y, z) != rule.from_block:
            return False
        if rule.require_clear_above and not self._clear_above(rule, x, y, z):
            return False
        r, vr = rule.radius, rule.vertical_radius
        for dy in range(-vr, vr + 1):
            for dz in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    if rule.is_source(w.get_block(x + dx, y + dy, z + dz)):
                        return True
        return False

    def _clear_above(self, rule, x, y, z):
        from server.lighting import get_block_opacity

        above = self.world.get_block(x, y + 1, z)
        if above == AIR:
            return True
        return get_block_opacity(BLOCKS.get(above)) <= rule.max_above_opacity

    # --- scanning ----------------------------------------------------------------

    def _scan_chunk(self, key):
        chunk = self.world.chunks.get(key)
        if chunk is None:
            return
        cx, cy, cz = key
        for rule in self.rules:
            idx = block_index(rule.from_block)
            for ly, lz, lx in np.argwhere(chunk.data == idx):
                pos = (cx * CHUNK_SIZE + int(lx), cy * CHUNK_SIZE + int(ly), cz * CHUNK_SIZE + int(lz))
                if pos in self.timers:
                    continue
                if self.is_eligible(rule, *pos):
                    self.timers[pos] = [rule, rule.roll_delay()]

    def update(self, dt):
        if not self.rules:
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
                self._scan_chunk(keys[self._cursor])
                self._cursor += 1

        changed = []
        for pos, entry in list(self.timers.items()):
            entry[1] -= step
            if entry[1] > 0:
                continue
            del self.timers[pos]
            rule = entry[0]
            # Conditions may have changed while waiting (block placed above, grass removed...).
            if self.is_eligible(rule, *pos) and self.world.set_block(*pos, rule.to_block):
                changed.append((*pos, rule.to_block))
        return changed

