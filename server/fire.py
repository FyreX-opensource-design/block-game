"""Fire lit by lava. Flames spread a short way, then go out on their own.

Each flame is a file in ``assets/fire``. ``assets/fire.json`` sets how many
new flames a tick may start. Solid fuel takes the time in
``assets/tags/blocks/burnable.json``, or a block's own ``burn_time``
property. Plants are not solid, so the flame replaces them immediately.
"""

import json
import math
import random
from pathlib import Path

import numpy as np

from server.blocks import AIR, ASSETS, BLOCKS, BY_INDEX
from server.tags import has_block_tag

_NEIGHBORS = (
    (0, 1, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 0, 1),
    (0, 0, -1),
    (0, -1, 0),
)
_WOOD_TAGS = ("logs", "planks", "leaves")
_WOOD_PIECES = ("fence", "door", "trapdoor", "sapling", "planks", "log", "wood", "leaves")
_BURN_TIMES = None


def _span(value, default):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return (number, number)
    if isinstance(value, (list, tuple)) and value:
        try:
            lo = float(value[0])
            hi = float(value[1]) if len(value) > 1 else lo
        except (TypeError, ValueError):
            return default
        if hi < lo:
            lo, hi = hi, lo
        return (lo, hi)
    return default


def _number(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class FireType:
    """How one flame behaves. Loaded from ``assets/fire/<id>.json``."""

    def __init__(
        self,
        fire_id,
        priority=0,
        life=(4.5, 6.5),
        spread_life=(4.5, 6.0),
        ignite_life=(2.5, 5.5),
        brief_life=(1.0, 1.6),
        spread_chance=0.05,
        extinguish_chance=0.45,
        burn_window=4.2,
        cooldown=6.0,
        hold_life=0.5,
    ):
        self.id = str(fire_id)
        self.priority = int(priority)
        self.life = life
        self.spread_life = spread_life
        self.ignite_life = ignite_life
        self.brief_life = brief_life
        self.spread_chance = max(0.0, float(spread_chance))
        self.extinguish_chance = min(1.0, max(0.0, float(extinguish_chance)))
        self.burn_window = max(0.0, float(burn_window))
        self.cooldown = max(0.0, float(cooldown))
        self.hold_life = max(0.0, float(hold_life))

    def roll(self, span):
        lo, hi = span
        if hi <= lo:
            return float(lo)
        return random.uniform(lo, hi)


class FireSimulation:
    """How often the fire tick runs, and how much it may change at once."""

    def __init__(self, tick_interval=0.4, max_new_flames=4, flame_cap=12, lava_per_tick=16):
        self.tick_interval = max(0.05, float(tick_interval))
        self.max_new_flames = max(0, int(max_new_flames))
        self.flame_cap = max(1, int(flame_cap))
        self.lava_per_tick = max(1, int(lava_per_tick))


def _fire_type_from_mapping(entry, source=""):
    if not isinstance(entry, dict):
        raise ValueError(f"{source}: fire type must be an object")
    fire_id = str(entry.get("id") or (Path(source).stem if source else ""))
    if not fire_id:
        raise ValueError(f"{source}: fire type needs an id")
    return FireType(
        fire_id,
        priority=int(entry.get("priority", 0)),
        life=_span(entry.get("life"), (4.5, 6.5)),
        spread_life=_span(entry.get("spread_life"), (4.5, 6.0)),
        ignite_life=_span(entry.get("ignite_life"), (2.5, 5.5)),
        brief_life=_span(entry.get("brief_life"), (1.0, 1.6)),
        spread_chance=_number(entry.get("spread_chance", 0.05), 0.05),
        extinguish_chance=_number(entry.get("extinguish_chance", 0.45), 0.45),
        burn_window=_number(entry.get("burn_window", 4.2), 4.2),
        cooldown=_number(entry.get("cooldown", 6.0), 6.0),
        hold_life=_number(entry.get("hold_life", 0.5), 0.5),
    )


def load_fire_types(folder=None):
    root = Path(folder) if folder is not None else ASSETS / "fire"
    types = {}
    if root.is_dir():
        for path in sorted(root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            spec = _fire_type_from_mapping(data, path.name)
            types[spec.id] = spec
    if "fire" not in types:
        types["fire"] = FireType("fire")
    return types


def load_fire_simulation(path=None):
    file_path = Path(path) if path is not None else ASSETS / "fire.json"
    data = {}
    try:
        loaded = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded
    except (OSError, json.JSONDecodeError):
        data = {}
    return FireSimulation(
        tick_interval=_number(data.get("tick_interval", 0.4), 0.4),
        max_new_flames=int(_number(data.get("max_new_flames", 4), 4)),
        flame_cap=int(_number(data.get("flame_cap", 12), 12)),
        lava_per_tick=int(_number(data.get("lava_per_tick", 16), 16)),
    )


FIRE_TYPES = load_fire_types()
FIRE_SIMULATION = load_fire_simulation()
_DEFAULT_FIRE = FIRE_TYPES.get("fire") or FireType("fire")


def fire_type(kind):
    """The tuning for a flame. Unknown ids use ordinary fire."""
    return FIRE_TYPES.get(str(kind or ""), _DEFAULT_FIRE)


def _burn_times():
    global _BURN_TIMES
    if _BURN_TIMES is not None:
        return _BURN_TIMES
    times = {"default": 8.0}
    path = ASSETS / "tags" / "blocks" / "burnable.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    raw = data.get("times") if isinstance(data, dict) else None
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                times[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    _BURN_TIMES = times
    return times


def is_flammable(block_id):
    """Wood, leaves, and plants can catch fire. Stone and fluids cannot."""
    if not block_id or block_id == AIR:
        return False
    block = BLOCKS.get(block_id)
    if block is None:
        return False
    props = getattr(block, "properties", None) or {}
    if "burn_time" in props:
        return True
    if getattr(block, "plant", False):
        return True
    if has_block_tag(block_id, "burnable"):
        return True
    if any(has_block_tag(block_id, tag) for tag in _WOOD_TAGS):
        return True
    name = getattr(block, "base_id", "") or block_id
    return any(piece in name for piece in _WOOD_PIECES)


def burn_seconds(block_id):
    """How long a solid block burns before it is destroyed.

    ``None`` means the block cannot burn. ``0`` means a plant or other
    non-solid fuel, which the flame replaces on contact.
    """
    if not is_flammable(block_id):
        return None
    block = BLOCKS.get(block_id)
    if block is None:
        return None
    props = getattr(block, "properties", None) or {}
    if "burn_time" in props:
        try:
            return max(0.0, float(props["burn_time"]))
        except (TypeError, ValueError):
            return None
    if getattr(block, "plant", False) or not getattr(block, "solid", False):
        return 0.0
    times = _burn_times()
    for tag in _WOOD_TAGS:
        if tag in times and has_block_tag(block_id, tag):
            return float(times[tag])
    return float(times.get("default", 8.0))


def flame_kind(block_id):
    """The fire type id when this block is a flame or a lava that lights one."""
    block = BLOCKS.get(block_id)
    if block is None:
        return ""
    props = getattr(block, "properties", None) or {}
    kind = str(props.get("fire") or props.get("lava") or "")
    if kind in FIRE_TYPES:
        return kind
    return ""


def body_flame(world, pos, radius, height):
    """The flame a body is standing in. The higher ``priority`` wins."""
    if world is None or pos is None:
        return ""
    found = ""
    found_priority = None
    min_x = math.floor(pos[0] - radius)
    max_x = math.floor(pos[0] + radius - 1e-6)
    min_y = math.floor(pos[1])
    max_y = math.floor(pos[1] + height - 1e-4)
    min_z = math.floor(pos[2] - radius)
    max_z = math.floor(pos[2] + radius - 1e-6)
    for y in range(min_y, max_y + 1):
        for z in range(min_z, max_z + 1):
            for x in range(min_x, max_x + 1):
                kind = flame_kind(world.get_block(x, y, z))
                if not kind:
                    continue
                priority = fire_type(kind).priority
                if found_priority is None or priority > found_priority:
                    found = kind
                    found_priority = priority
    return found


class FireSystem:
    """Tracks lava and the flames it starts. Every flame has a lifetime."""

    def __init__(self, world):
        self.world = world
        self.lava = {}
        self.flames = {}
        self.burning = {}
        self.cooled = {}
        self._new_flames = 0
        self._acc = 0.0
        self._cursor = 0
        self._lava_index = {}
        self._fire_index = {}
        self._indexed = False

    def _index(self):
        if self._indexed:
            return
        lava = {}
        fire = {}
        for index, block in enumerate(BY_INDEX):
            if block is None:
                continue
            props = getattr(block, "properties", None) or {}
            kind = props.get("lava")
            if kind:
                lava[int(index)] = str(kind)
            flame = props.get("fire")
            if flame:
                fire[int(index)] = str(flame)
        self._lava_index = lava
        self._fire_index = fire
        self._indexed = True

    def scan_chunk(self, chunk):
        """Remember lava and fire already stored in a generated or loaded chunk."""
        self._index()
        if not self._lava_index and not self._fire_index:
            return
        data = chunk.data
        wanted = np.fromiter(set(self._lava_index) | set(self._fire_index), dtype=np.int32)
        if wanted.size == 0:
            return
        hits = np.isin(data, wanted)
        if not np.any(hits):
            return
        x0 = chunk.cx * data.shape[2]
        y0 = chunk.cy * data.shape[0]
        z0 = chunk.cz * data.shape[1]
        for ly, lz, lx in zip(*np.nonzero(hits)):
            index = int(data[ly, lz, lx])
            pos = (x0 + int(lx), y0 + int(ly), z0 + int(lz))
            kind = self._lava_index.get(index)
            if kind:
                self.lava[pos] = kind
                continue
            flame = self._fire_index.get(index)
            if flame and pos not in self.flames:
                self.flames[pos] = (fire_type(flame).roll(fire_type(flame).life), flame, 0.0, False)

    def on_block_set(self, x, y, z, block_id):
        self._index()
        pos = (int(x), int(y), int(z))
        block = BLOCKS.get(block_id)
        props = getattr(block, "properties", None) or {}
        kind = props.get("lava")
        if kind:
            self.lava[pos] = str(kind)
        else:
            self.lava.pop(pos, None)
        flame = props.get("fire")
        if flame:
            if pos not in self.flames:
                spec = fire_type(flame)
                self.flames[pos] = (spec.roll(spec.life), str(flame), 0.0, False)
        else:
            self.flames.pop(pos, None)
        if burn_seconds(block_id) is None:
            self.burning.pop(pos, None)

    def update(self, dt):
        if not getattr(self.world, "simulate_blocks", True):
            return []
        self._acc += max(0.0, float(dt))
        if self._acc < FIRE_SIMULATION.tick_interval:
            return []
        step = self._acc
        self._acc = 0.0
        changed = []
        touched = set()
        self._new_flames = 0
        for pos in list(self.cooled):
            self.cooled[pos] -= step
            if self.cooled[pos] <= 0.0:
                self.cooled.pop(pos, None)
        for pos in list(self.flames):
            life, kind, age, brief = self._flame_record(self.flames[pos])
            spec = fire_type(kind)
            if self.world.get_block(*pos) != kind:
                self.flames.pop(pos, None)
                continue
            life -= step
            age += step
            fed = self._burn_neighbors(pos, kind, step, changed, touched)
            # Stay through the burn window on fuel so the blocks actually
            # burn. After that the flame is likely to die, with wood left over.
            if not brief and fed and age < spec.burn_window:
                life = max(life, spec.hold_life)
            elif not brief and age >= spec.burn_window and random.random() < spec.extinguish_chance:
                life = 0.0
            if life <= 0.0:
                self._snuff(pos, changed)
                continue
            self.flames[pos] = (life, kind, age, brief)
            if (
                not brief
                and age < spec.burn_window
                and len(self.flames) < FIRE_SIMULATION.flame_cap
                and random.random() < spec.spread_chance
            ):
                self._spread(pos, kind, changed)
        if self.lava:
            keys = list(self.lava)
            count = min(FIRE_SIMULATION.lava_per_tick, len(keys))
            start = self._cursor % len(keys)
            self._cursor += count
            for offset in range(count):
                pos = keys[(start + offset) % len(keys)]
                kind = self.lava.get(pos)
                if kind and self.world.get_block(*pos) not in ("", AIR):
                    self._tick_lava_fuel(pos, kind, step, changed, touched)
                    self._ignite_around(pos, kind, changed)
        for spot in list(self.burning):
            if spot not in touched:
                self.burning.pop(spot, None)
        return changed

    def _flame_record(self, value):
        life, kind = value[0], value[1]
        age = value[2] if len(value) > 2 else 0.0
        brief = value[3] if len(value) > 3 else False
        return float(life), kind, float(age), bool(brief)

    def _snuff(self, pos, changed):
        self.flames.pop(pos, None)
        block_id = self.world.get_block(*pos)
        block = BLOCKS.get(block_id)
        props = getattr(block, "properties", None) or {}
        kind = str(props.get("fire") or "")
        if kind not in FIRE_TYPES:
            return
        self.cooled[pos] = fire_type(kind).cooldown
        if self.world.set_block(*pos, AIR):
            changed.append((*pos, AIR))

    def _tick_lava_fuel(self, pos, kind, step, changed, touched):
        """Lava keeps burning the blocks it touches, after the flames have gone."""
        for dx, dy, dz in _NEIGHBORS:
            spot = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
            if spot in touched:
                continue
            self._advance_burn(spot, kind, step, changed, touched)

    def _spread(self, pos, kind, changed):
        spec = fire_type(kind)
        order = list(_NEIGHBORS)
        random.shuffle(order)
        for dx, dy, dz in order:
            spot = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
            if self._catch(spot, kind, changed, life=spec.roll(spec.spread_life)):
                return

    def _ignite_around(self, pos, kind, changed):
        spec = fire_type(kind)
        for dx, dy, dz in _NEIGHBORS:
            self._catch(
                (pos[0] + dx, pos[1] + dy, pos[2] + dz),
                kind,
                changed,
                life=spec.roll(spec.ignite_life),
            )

    def _burn_neighbors(self, pos, kind, step, changed, touched):
        """Advance the burn clock on solid fuel touching this flame."""
        fed = False
        for dx, dy, dz in _NEIGHBORS:
            spot = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
            if self._advance_burn(spot, kind, step, changed, touched):
                fed = True
        return fed

    def _advance_burn(self, spot, kind, step, changed, touched):
        if spot in touched:
            return
        seconds = burn_seconds(self.world.get_block(*spot))
        if seconds is None or seconds <= 0.0:
            return False
        left = self.burning.get(spot, seconds) - step
        if left <= 0.0:
            self.burning.pop(spot, None)
            self._consume(spot, kind, changed)
            return True
        self.burning[spot] = left
        touched.add(spot)
        return True

    def _consume(self, pos, kind, changed):
        """The fuel is gone. The flame takes its place."""
        spec = fire_type(kind)
        if self._light(pos, kind, spec.roll(spec.brief_life), changed, limited=False, brief=True):
            return
        if self.world.set_block(*pos, AIR):
            changed.append((*pos, AIR))

    def _catch(self, pos, kind, changed, life):
        block_id = self.world.get_block(*pos)
        if not is_flammable(block_id):
            return False
        block = BLOCKS.get(block_id)
        if block is not None and not block.solid:
            return self._light(pos, kind, life, changed)
        x, y, z = pos
        for dx, dy, dz in ((0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)):
            spot = (x + dx, y + dy, z + dz)
            if self.world.get_block(*spot) == AIR:
                return self._light(spot, kind, life, changed)
        return False

    def _light(self, pos, kind, life, changed, limited=True, brief=False):
        if kind not in BLOCKS:
            return False
        if limited and self.cooled.get(pos, 0.0) > 0.0:
            return False
        if limited and self._new_flames >= FIRE_SIMULATION.max_new_flames:
            return False
        if pos in self.flames and self.world.get_block(*pos) == kind:
            return False
        self.flames[pos] = (float(life), kind, 0.0, bool(brief))
        if self.world.set_block(*pos, kind):
            if limited:
                self._new_flames += 1
            changed.append((*pos, kind))
            return True
        self.flames.pop(pos, None)
        return False
