"""Data-driven bows and arrows.

A file in ``assets/bows`` describes how a bow shoots. ``distance`` is how far a
full-draw, level shot travels before it has dropped about one block.
``accuracy`` is 0 to 1: 1 is perfectly straight at full draw.

    {
      "id": "wood_bow",
      "accuracy": 0.82,
      "distance": 26,
      "draw_time": 0.9
    }

A file in ``assets/arrows`` describes the projectile. ``effects`` is reserved
for later arrow types; unknown effect ids are kept and ignored.

    {
      "id": "arrow",
      "damage": 5,
      "speed": 1,
      "gravity": 12,
      "effects": []
    }

The asset id is the item id unless ``item`` says otherwise.
"""

import json
import math
import random
import re
from pathlib import Path

from server.blocks import ASSETS
from server.lua import load_table

_ID = re.compile(r"[a-z0-9_]+")
MAX_SPREAD = 0.18
MIN_POWER = 0.15
_LEVEL_DROP = 1.25


class Bow:
    def __init__(self, bow_id, item, accuracy, distance, draw_time, source):
        self.id = bow_id
        self.item = item
        self.accuracy = float(accuracy)
        self.distance = float(distance)
        self.draw_time = float(draw_time)
        self.source = str(source)


class Arrow:
    def __init__(self, arrow_id, item, damage, speed, gravity, effects, source):
        self.id = arrow_id
        self.item = item
        self.damage = float(damage)
        self.speed = float(speed)
        self.gravity = float(gravity)
        self.effects = tuple(dict(effect) for effect in effects)
        self.source = str(source)


def load_bows(folder=None):
    return _load(folder, "bows", bow_from_mapping)


def load_arrows(folder=None):
    return _load(folder, "arrows", arrow_from_mapping)


def _load(folder, kind, parse):
    root = Path(folder) if folder is not None else ASSETS / kind
    if not root.is_dir():
        return {}
    found = {}
    for path in sorted(root.iterdir()):
        fields = _read_file(path)
        if fields is None:
            continue
        record = parse(fields, path)
        if record.item in found:
            raise ValueError(f"{path}: {record.item} is already a {kind[:-1]}")
        found[record.item] = record
    return found


def bow_from_mapping(fields, source):
    bow_id, item = _identity(fields, source, "bow")
    accuracy = _number(fields.get("accuracy", 1), "accuracy", source)
    if not 0 <= accuracy <= 1:
        raise ValueError(f"{source}: accuracy must stay between 0 and 1")
    distance = _number(fields.get("distance", 20), "distance", source)
    draw_time = _number(fields.get("draw_time", 1), "draw_time", source)
    if distance <= 0 or draw_time <= 0:
        raise ValueError(f"{source}: distance and draw_time must be positive")
    return Bow(bow_id, item, accuracy, distance, draw_time, source)


def arrow_from_mapping(fields, source):
    arrow_id, item = _identity(fields, source, "arrow")
    damage = _number(fields.get("damage", 1), "damage", source)
    speed = _number(fields.get("speed", 1), "speed", source)
    gravity = _number(fields.get("gravity", 12), "gravity", source)
    if damage < 0 or speed <= 0 or gravity < 0:
        raise ValueError(f"{source}: damage, speed, and gravity must stay usable")
    raw = fields.get("effects", [])
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{source}: effects must be a list")
    effects = []
    for effect in raw:
        if not isinstance(effect, dict) or not isinstance(effect.get("id"), str):
            raise ValueError(f"{source}: each effect needs an id")
        if _ID.fullmatch(effect["id"]) is None:
            raise ValueError(f"{source}: effect id must be a lowercase name")
        effects.append(dict(effect))
    return Arrow(arrow_id, item, damage, speed, gravity, effects, source)


def launch_speed(bow, arrow, power):
    """Blocks per second for this draw. A level full draw covers ``bow.distance``."""
    power = max(0.0, min(1.0, float(power)))
    gravity = float(arrow.gravity)
    if gravity <= 0:
        return float(bow.distance) * float(arrow.speed) * power
    flight = math.sqrt((2.0 * _LEVEL_DROP) / gravity)
    return (float(bow.distance) / flight) * float(arrow.speed) * power


def aim_direction(bow, power, direction, rng=None):
    """Unit look vector after this bow's spread. Full draw and accuracy 1 is exact."""
    rng = rng or random
    power = max(0.0, min(1.0, float(power)))
    unit = _unit(direction)
    if unit is None:
        return (0.0, 0.0, -1.0)
    spread = (1.0 - float(bow.accuracy) * power) * MAX_SPREAD
    if spread <= 0:
        return unit
    from server.player import look_direction

    dx, dy, dz = unit
    horiz = math.hypot(dx, dz)
    yaw = math.atan2(-dx, -dz)
    pitch = math.atan2(dy, horiz)
    yaw += rng.uniform(-spread, spread)
    pitch += rng.uniform(-spread, spread)
    pitch = max(-1.5, min(1.5, pitch))
    return look_direction(yaw, pitch)


def shot_origin(pos, eye, direction):
    unit = _unit(direction) or (0.0, 0.0, -1.0)
    return [
        float(pos[0]) + unit[0] * 0.75,
        float(pos[1]) + float(eye) + unit[1] * 0.75,
        float(pos[2]) + unit[2] * 0.75,
    ]


def fire_arrow(world, bow, arrow, origin, direction, power, owner=""):
    """Spawn an arrow flying along ``direction``. Speed comes from the bow and arrow."""
    unit = _unit(direction)
    if unit is None or world is None or bow is None or arrow is None:
        return None
    power = max(0.0, min(1.0, float(power)))
    speed = launch_speed(bow, arrow, power)
    velocity = [unit[0] * speed, unit[1] * speed, unit[2] * speed]
    return world.spawn_arrow(
        arrow.item,
        origin,
        velocity,
        damage=arrow.damage,
        power=power,
        gravity=arrow.gravity,
        effects=arrow.effects,
        owner=owner,
    )


def apply_arrow_effects(target, effects, power, velocity):
    """Apply effect ids this game understands. Unknown ids are left for later."""
    if target is None:
        return
    power = max(0.0, min(1.0, float(power)))
    for effect in effects or ():
        kind = str(effect.get("id", ""))
        if kind == "knockback":
            amount = float(effect.get("amount", 1) or 0)
            scale = 1.0 + amount * power
            vel = getattr(target, "vel", None)
            if vel is not None and len(vel) >= 3:
                vel[0] *= scale
                vel[2] *= scale
                vel[1] += 0.5 * amount * power


def _identity(fields, source, kind):
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: {kind} file must be an object")
    record_id = fields.get("id")
    if not isinstance(record_id, str) or _ID.fullmatch(record_id) is None:
        raise ValueError(f"{source}: {kind} id must be a lowercase name")
    item = fields.get("item", record_id)
    if not isinstance(item, str) or _ID.fullmatch(item) is None:
        raise ValueError(f"{source}: item must be a lowercase name")
    from server.items import ITEMS

    if item not in ITEMS:
        raise ValueError(f"{source}: unknown item {item}")
    return record_id, item


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


def _unit(direction):
    if direction is None or len(direction) < 3:
        return None
    dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-8:
        return None
    return (dx / length, dy / length, dz / length)


BOWS = load_bows()
ARROWS = load_arrows()
