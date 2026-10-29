import json
import re
from pathlib import Path

from server.blocks import ASSETS, WORLD_HEIGHT
from server.lua import load_table

_ID = re.compile(r"[a-z0-9_]+")
_INT_FIELDS = ("min_y", "max_y", "octaves", "seed_offset", "max_vein_size")
_FLOAT_FIELDS = ("scale", "threshold")


class Ore:
    def __init__(
        self,
        ore_id,
        block,
        target,
        min_y,
        max_y,
        scale,
        threshold,
        octaves,
        seed_offset,
        enabled=True,
        source=None,
        max_vein_size=0,
    ):
        self.id = ore_id
        self.block = block
        self.target = target
        self.min_y = min_y
        self.max_y = max_y
        self.scale = scale
        self.threshold = threshold
        self.octaves = octaves
        self.seed_offset = seed_offset
        self.enabled = enabled
        self.source = str(source) if source else ore_id
        self.max_vein_size = max_vein_size


def load_ores(folder=None):
    root = Path(folder) if folder is not None else ASSETS / "ores"
    if not root.is_dir():
        return ()
    ores = []
    seen = set()
    for path in sorted(root.iterdir()):
        if path.suffix == ".json":
            fields = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".lua":
            fields = load_table(path)
        else:
            continue
        ore = ore_from_mapping(fields, path)
        if ore.id in seen:
            raise ValueError(f"{path}: duplicate ore id {ore.id}")
        seen.add(ore.id)
        ores.append(ore)
    return tuple(ores)


def ore_from_mapping(fields, source):
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: ore file must be an object")
    ore_id = fields.get("id")
    if not isinstance(ore_id, str) or _ID.fullmatch(ore_id) is None:
        raise ValueError(f"{source}: ore id must be a lowercase name")

    block = str(fields.get("block", f"{ore_id}_ore"))
    target = fields.get("target", "stone")
    enabled = bool(fields.get("enabled", True))

    values = {
        "min_y": 1,
        "max_y": WORLD_HEIGHT - 1,
        "scale": 0.1,
        "threshold": 0.6,
        "octaves": 2,
        "seed_offset": 0,
        "max_vein_size": 0,
    }
    for name in _INT_FIELDS:
        if name in fields:
            values[name] = _whole(fields[name], name, source)
    for name in _FLOAT_FIELDS:
        if name in fields:
            values[name] = _number(fields[name], name, source)

    if values["max_vein_size"] < 0:
        raise ValueError(f"{source}: max_vein_size must be zero or more")
    if values["min_y"] < 1 or values["max_y"] > WORLD_HEIGHT or values["min_y"] > values["max_y"]:
        raise ValueError(f"{source}: ore y range must stay above the floor and inside the world")
    if values["octaves"] < 1 or values["octaves"] > 5:
        raise ValueError(f"{source}: octaves must be from 1 to 5")
    if values["scale"] <= 0:
        raise ValueError(f"{source}: ore scale must be positive")

    return Ore(
        ore_id,
        block,
        target,
        values["min_y"],
        values["max_y"],
        values["scale"],
        values["threshold"],
        values["octaves"],
        values["seed_offset"],
        enabled=enabled,
        source=str(source),
        max_vein_size=values["max_vein_size"],
    )


def _whole(value, name, source):
    number = _number(value, name, source)
    if int(number) != number:
        raise ValueError(f"{source}: {name} must be a whole number")
    return int(number)


def _number(value, name, source):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: {name} must be a number")
    return float(value)
