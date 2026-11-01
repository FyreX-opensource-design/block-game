"""Data-driven damage types.

Every file in ``assets/damage`` (``.json`` or ``.lua``) describes one damage
type. The engine currently understands two ``kind`` values:

``fall``
    Applied when a survival player lands after falling. ``min_distance`` is the
    number of blocks they can fall safely; with ``per_block`` true the damage
    scales with every additional block.

``suffocation``
    Applied repeatedly while a survival player's head is inside a solid block,
    once every ``interval`` seconds.

Example::

    {
      "id": "fall",
      "name": "Fall",
      "kind": "fall",
      "damage": 1.0,
      "min_distance": 3.0,
      "per_block": true,
      "message": "fell from a high place"
    }

If the asset folder is missing or empty, built-in versions of the two damage
types above are used so the game still works without assets.
"""

import json
from pathlib import Path

from server.blocks import ASSETS
from server.lua import load_table

DEFAULT_MAX_HEALTH = 20.0
KINDS = ("fall", "suffocation")

_BUILTIN = (
    {
        "id": "fall",
        "name": "Fall",
        "kind": "fall",
        "damage": 1.0,
        "min_distance": 3.0,
        "per_block": True,
        "message": "fell from a high place",
    },
    {
        "id": "suffocation",
        "name": "Suffocation",
        "kind": "suffocation",
        "damage": 1.0,
        "interval": 0.5,
        "message": "suffocated in a block",
    },
)


class DamageType:
    def __init__(
        self,
        damage_id,
        name,
        kind,
        damage,
        message="",
        min_distance=3.0,
        per_block=True,
        interval=0.5,
        source="",
    ):
        self.id = str(damage_id)
        self.name = str(name)
        self.kind = str(kind)
        self.damage = float(damage)
        self.message = str(message)
        self.min_distance = float(min_distance)
        self.per_block = bool(per_block)
        self.interval = float(interval)
        self.source = str(source)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "damage": self.damage,
            "message": self.message,
            "min_distance": self.min_distance,
            "per_block": self.per_block,
            "interval": self.interval,
        }

    def __repr__(self):
        return f"DamageType({self.id!r}, kind={self.kind!r}, damage={self.damage})"


def damage_type_from_mapping(fields, source="<builtin>"):
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: damage type file must be an object")

    damage_id = str(fields.get("id", "")).strip().lower()
    if not damage_id:
        raise ValueError(f"{source}: damage type id is required")

    kind = str(fields.get("kind", damage_id)).strip().lower()
    if kind not in KINDS:
        raise ValueError(f"{source}: unknown damage kind {kind!r} (expected one of {', '.join(KINDS)})")

    damage = _number(fields.get("damage", 1.0), "damage", source)
    if damage < 0:
        raise ValueError(f"{source}: damage must be zero or greater")

    min_distance = _number(fields.get("min_distance", 3.0), "min_distance", source)
    if min_distance < 0:
        raise ValueError(f"{source}: min_distance must be zero or greater")

    interval = _number(fields.get("interval", 0.5), "interval", source)
    if interval <= 0:
        raise ValueError(f"{source}: interval must be greater than zero")

    return DamageType(
        damage_id,
        fields.get("name", damage_id.replace("_", " ").title()),
        kind,
        damage,
        message=fields.get("message", ""),
        min_distance=min_distance,
        per_block=bool(fields.get("per_block", True)),
        interval=interval,
        source=source,
    )


def load_damage_types(folder=None):
    root = Path(folder) if folder is not None else ASSETS / "damage"
    types = {}
    if root.is_dir():
        for path in sorted(root.iterdir()):
            fields = _read_file(path)
            if fields is None:
                continue
            damage_type = damage_type_from_mapping(fields, path)
            if damage_type.id in types:
                raise ValueError(f"{path}: duplicate damage type id {damage_type.id}")
            types[damage_type.id] = damage_type
    if not types:
        for fields in _BUILTIN:
            damage_type = damage_type_from_mapping(fields)
            types[damage_type.id] = damage_type
    return types


DAMAGE_TYPES = {}


def get_damage_type(damage_id):
    return DAMAGE_TYPES.get(str(damage_id or "").strip().lower())


def damage_types_for_kind(kind):
    kind = str(kind or "").strip().lower()
    return tuple(damage_type for damage_type in DAMAGE_TYPES.values() if damage_type.kind == kind)


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


DAMAGE_TYPES = load_damage_types()
