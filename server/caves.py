import json
import re
from pathlib import Path

from server.blocks import ASSETS, WORLD_HEIGHT
from server.lua import load_table
from server.noise import fbm, fbm3

_ID = re.compile(r"[a-z0-9_]+")
_INT_FIELDS = ("min_y", "max_y", "surface_margin", "octaves", "seed_offset", "breach_depth")
_FLOAT_FIELDS = (
    "horizontal_scale",
    "vertical_scale",
    "thickness",
    "threshold",
    "mask_scale",
    "mask_threshold",
    "breach_scale",
    "breach_threshold",
)


class Cave:
    def __init__(
        self,
        cave_id,
        style,
        min_y,
        max_y,
        surface_margin,
        horizontal_scale,
        vertical_scale,
        octaves,
        seed_offset,
        thickness,
        threshold,
        mask_scale,
        mask_threshold,
        breach_depth,
        breach_scale,
        breach_threshold,
        enabled,
        source,
    ):
        self.id = cave_id
        self.style = style
        self.min_y = min_y
        self.max_y = max_y
        self.surface_margin = surface_margin
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.octaves = octaves
        self.seed_offset = seed_offset
        self.thickness = thickness
        self.threshold = threshold
        self.mask_scale = mask_scale
        self.mask_threshold = mask_threshold
        self.breach_depth = breach_depth
        self.breach_scale = breach_scale
        self.breach_threshold = breach_threshold
        self.enabled = enabled
        self.source = source

    def opens(self, x, y, z, surface, seed):
        from server.blocks import WATER_LEVEL

        margin = max(self.surface_margin, 4) if surface <= WATER_LEVEL else self.surface_margin
        if not self.enabled or y < self.min_y or y > self.max_y or y > surface - margin:
            return False
        if self.mask_scale > 0 and fbm(x * self.mask_scale, z * self.mask_scale, 2, seed + self.seed_offset + 4096) <= self.mask_threshold:
            return False
        if self.breach_threshold > 0 and y > surface - self.breach_depth:
            if surface <= WATER_LEVEL:
                return False
            breach = fbm3(x * self.breach_scale, y * self.breach_scale, z * self.breach_scale, 1, seed + self.seed_offset + 77)
            if breach <= self.breach_threshold:
                return False
        noise = fbm3(
            x * self.horizontal_scale,
            y * self.vertical_scale,
            z * self.horizontal_scale,
            self.octaves,
            seed + self.seed_offset,
        )
        if self.style == "tunnel":
            return abs(noise - 0.5) < self.thickness
        return noise > self.threshold


def load_caves(folder=None):
    root = Path(folder) if folder is not None else ASSETS / "caves"
    if not root.is_dir():
        return ()
    caves = []
    seen = set()
    for path in sorted(root.iterdir()):
        if path.suffix == ".json":
            fields = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".lua":
            fields = load_table(path)
        else:
            continue
        cave = cave_from_mapping(fields, path)
        if cave.id in seen:
            raise ValueError(f"{path}: duplicate cave id {cave.id}")
        seen.add(cave.id)
        caves.append(cave)
    return tuple(caves)


def cave_from_mapping(fields, source):
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: cave file must be an object")
    cave_id = fields.get("id")
    if not isinstance(cave_id, str) or _ID.fullmatch(cave_id) is None:
        raise ValueError(f"{source}: cave id must be a lowercase name")
    style = fields.get("style", "tunnel")
    if style not in ("tunnel", "cavern"):
        raise ValueError(f"{source}: style must be tunnel or cavern")
    enabled = fields.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{source}: enabled must be true or false")
    values = {
        "min_y": 8,
        "max_y": 64,
        "surface_margin": 5,
        "horizontal_scale": 0.05,
        "vertical_scale": 0.08,
        "octaves": 2,
        "seed_offset": 0,
        "thickness": 0.07,
        "threshold": 0.68,
        "mask_scale": 0.0,
        "mask_threshold": 0.0,
        "breach_depth": 0,
        "breach_scale": 0.08,
        "breach_threshold": 0.0,
    }
    for name in _INT_FIELDS:
        if name in fields:
            values[name] = _whole(fields[name], name, source)
    for name in _FLOAT_FIELDS:
        if name in fields:
            values[name] = _number(fields[name], name, source)
    if values["min_y"] < 1 or values["max_y"] >= WORLD_HEIGHT or values["min_y"] > values["max_y"]:
        raise ValueError(f"{source}: cave y range must stay above the floor and inside the world")
    if values["surface_margin"] < 0:
        raise ValueError(f"{source}: surface_margin must be zero or more")
    if values["octaves"] < 1 or values["octaves"] > 5:
        raise ValueError(f"{source}: octaves must be from 1 to 5")
    if values["horizontal_scale"] <= 0 or values["vertical_scale"] <= 0:
        raise ValueError(f"{source}: cave scale must be positive")
    if values["mask_scale"] < 0:
        raise ValueError(f"{source}: mask_scale must be zero or more")
    if style == "tunnel" and values["thickness"] <= 0:
        raise ValueError(f"{source}: tunnel thickness must be positive")
    return Cave(
        cave_id,
        style,
        values["min_y"],
        values["max_y"],
        values["surface_margin"],
        values["horizontal_scale"],
        values["vertical_scale"],
        values["octaves"],
        values["seed_offset"],
        values["thickness"],
        values["threshold"],
        values["mask_scale"],
        values["mask_threshold"],
        values["breach_depth"],
        values["breach_scale"],
        values["breach_threshold"],
        enabled,
        str(source),
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
