"""Crop files and the blocks, items, and loot they register.

A file in ``assets/crops`` can list each stage block, or give a stage count
and let the game register ``{id}_0`` .. ``{id}_{n-1}``:

    {
      "id": "orange_carrot",
      "stages": 8,
      "min_time": 180,
      "max_time": 340,
      "needs_water": true,
      "seeds": "orange_carrot_seeds",
      "produce": "orange_carrot",
      "food": 4,
      "wild": {
        "block": "wild_orange_carrot",
        "chance": 0.01,
        "surfaces": ["grass"],
        "biomes": ["plains", "lowland_plains"]
      }
    }

One day cycle is 1200 seconds. Seven steps at 180–340 seconds each take about
one to two days. Stage blocks, seed and produce items, and loot tables are
created from the file when those assets are not already written by hand.
``wild`` places an untamed plant in the listed biomes.
"""

import json
import re
from pathlib import Path

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_ID = re.compile(r"[a-z0-9_]+")

# 24000 ticks per day at 20 ticks per second. Seven stage steps at these
# bounds finish in about one to two days.
DEFAULT_MIN_TIME = 180.0
DEFAULT_MAX_TIME = 340.0

_CACHE = None


class WildPlant:
    def __init__(self, block, chance, surfaces, biomes, seed_offset):
        self.block = block
        self.chance = float(chance)
        self.surfaces = tuple(surfaces)
        self.biomes = tuple(biomes)
        self.seed_offset = int(seed_offset)


class CropSpec:
    def __init__(
        self,
        crop_id,
        stages,
        min_time,
        max_time,
        needs_water,
        water_radius,
        water_vertical,
        seeds,
        produce,
        food,
        wild,
        source,
    ):
        self.id = crop_id
        self.stages = tuple(stages)
        self.min_time = float(min_time)
        self.max_time = float(max_time)
        self.needs_water = bool(needs_water)
        self.water_radius = int(water_radius)
        self.water_vertical = int(water_vertical)
        self.seeds = seeds
        self.produce = produce
        self.food = int(food)
        self.wild = wild
        self.source = str(source)

    @property
    def lang_block(self):
        return f"block.{self.id}"


def load_crop_defs(folder=None):
    global _CACHE
    if folder is None and _CACHE is not None:
        return _CACHE
    root = Path(folder) if folder is not None else _ASSETS / "crops"
    if not root.is_dir():
        specs = ()
    else:
        specs = tuple(_load_folder(root))
    if folder is None:
        _CACHE = specs
    return specs


def _load_folder(root):
    specs = []
    seen = set()
    seen_blocks = set()
    seen_items = set()
    for path in sorted(root.iterdir()):
        fields = _read_file(path)
        if fields is None:
            continue
        spec = spec_from_mapping(fields, path)
        if spec.id in seen:
            raise ValueError(f"{path}: duplicate crop id {spec.id}")
        for block_id in spec.stages:
            if block_id in seen_blocks:
                raise ValueError(f"{path}: block {block_id} is already a crop stage")
            seen_blocks.add(block_id)
        if spec.wild is not None:
            if spec.wild.block in seen_blocks:
                raise ValueError(f"{path}: wild block {spec.wild.block} is already used")
            seen_blocks.add(spec.wild.block)
        for item_id in (spec.seeds, spec.produce):
            if not item_id:
                continue
            if item_id in seen_items:
                raise ValueError(f"{path}: item {item_id} is already used by a crop")
            seen_items.add(item_id)
        seen.add(spec.id)
        specs.append(spec)
    return specs


def spec_from_mapping(fields, source):
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: crop file must be an object")
    crop_id = fields.get("id")
    if not isinstance(crop_id, str) or _ID.fullmatch(crop_id) is None:
        raise ValueError(f"{source}: crop id must be a lowercase name")
    stages = _stages(fields.get("stages"), crop_id, source)
    min_time = _number(fields.get("min_time", DEFAULT_MIN_TIME), "min_time", source)
    max_time = _number(fields.get("max_time", DEFAULT_MAX_TIME), "max_time", source)
    if min_time <= 0 or max_time < min_time:
        raise ValueError(f"{source}: times must satisfy 0 < min_time <= max_time")
    radius = _number(fields.get("water_radius", 4), "water_radius", source)
    vertical = _number(fields.get("water_vertical", 1), "water_vertical", source)
    if int(radius) != radius or radius < 0 or int(vertical) != vertical or vertical < 0:
        raise ValueError(f"{source}: water radius must be a whole number of zero or more")
    seeds = _optional_id(fields.get("seeds"), "seeds", source)
    produce = _optional_id(fields.get("produce"), "produce", source)
    if seeds and produce and seeds == produce:
        raise ValueError(f"{source}: seeds and produce must be different items")
    food = fields.get("food", 0)
    if isinstance(food, bool) or not isinstance(food, int) or food < 0:
        raise ValueError(f"{source}: food must be a whole number of zero or more")
    return CropSpec(
        crop_id,
        stages,
        min_time,
        max_time,
        bool(fields.get("needs_water", True)),
        int(radius),
        int(vertical),
        seeds,
        produce,
        food,
        _wild(fields.get("wild"), crop_id, source),
        source,
    )


def _stages(raw, crop_id, source):
    if isinstance(raw, bool):
        raise ValueError(f"{source}: stages must be a count or a list of block ids")
    if isinstance(raw, int):
        if raw < 2 or raw > 16:
            raise ValueError(f"{source}: a crop needs 2 to 16 stages")
        return tuple(f"{crop_id}_{index}" for index in range(raw))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError(f"{source}: stages must list at least two block ids")
    resolved = []
    for block_id in raw:
        if not isinstance(block_id, str) or _ID.fullmatch(block_id) is None:
            raise ValueError(f"{source}: unknown crop stage {block_id!r}")
        if block_id in resolved:
            raise ValueError(f"{source}: stage {block_id} is listed twice")
        resolved.append(block_id)
    return tuple(resolved)


def _wild(raw, crop_id, source):
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: wild must be an object")
    block = raw.get("block", f"wild_{crop_id}")
    if not isinstance(block, str) or _ID.fullmatch(block) is None:
        raise ValueError(f"{source}: wild block must be a lowercase name")
    chance = _number(raw.get("chance", 0.01), "chance", source)
    if not 0 <= chance <= 1:
        raise ValueError(f"{source}: wild chance must stay between 0 and 1")
    surfaces = raw.get("surfaces", ["grass"])
    if isinstance(surfaces, str):
        surfaces = [surfaces]
    if not isinstance(surfaces, (list, tuple)) or not surfaces:
        raise ValueError(f"{source}: wild surfaces must be a list of block ids")
    clean_surfaces = []
    for surface in surfaces:
        if not isinstance(surface, str) or _ID.fullmatch(surface) is None:
            raise ValueError(f"{source}: unknown wild surface {surface!r}")
        if surface not in clean_surfaces:
            clean_surfaces.append(surface)
    biomes = raw.get("biomes", [])
    if isinstance(biomes, str):
        biomes = [biomes]
    if not isinstance(biomes, (list, tuple)) or not biomes:
        raise ValueError(f"{source}: wild biomes must list at least one biome")
    clean_biomes = []
    for biome_id in biomes:
        if not isinstance(biome_id, str) or _ID.fullmatch(biome_id) is None:
            raise ValueError(f"{source}: unknown wild biome {biome_id!r}")
        if biome_id not in clean_biomes:
            clean_biomes.append(biome_id)
    offset = raw.get("seed_offset", sum(ord(char) for char in block))
    offset = _number(offset, "seed_offset", source)
    if int(offset) != offset:
        raise ValueError(f"{source}: seed_offset must be a whole number")
    return WildPlant(block, chance, clean_surfaces, clean_biomes, int(offset))


def block_entries(folder=None):
    """Stage and wild blocks declared by crop files and not written by hand."""
    entries = []
    root = _ASSETS / "blocks"
    for spec in load_crop_defs(folder):
        for index, block_id in enumerate(spec.stages):
            if (root / f"{block_id}.json").is_file():
                continue
            ripe = index + 1 == len(spec.stages)
            table = None
            if spec.seeds and spec.produce:
                table = f"crops/{spec.id}_{'ripe' if ripe else 'young'}"
            entries.append(
                _plant_block(
                    block_id,
                    spec.lang_block,
                    block_id,
                    table,
                    valid_ground=("farmland",),
                )
            )
        wild = spec.wild
        if wild is None or (root / f"{wild.block}.json").is_file():
            continue
        table = f"crops/{spec.id}_wild" if spec.seeds and spec.produce else None
        entries.append(_plant_block(wild.block, f"block.{wild.block}", wild.block, table, valid_ground=None))
    return entries


def item_entries(folder=None):
    """Seed and produce items declared by crop files and not written by hand."""
    entries = []
    root = _ASSETS / "items"
    for spec in load_crop_defs(folder):
        if spec.seeds and not (root / f"{spec.seeds}.json").is_file():
            entries.append(
                {
                    "id": spec.seeds,
                    "lang": f"item.{spec.seeds}",
                    "tab": ["natural"],
                    "model": "flat",
                    "texture": spec.seeds,
                    "block": spec.stages[0],
                    "max_stack": 64,
                    "left": "none",
                    "right": "place",
                    "food": 0,
                }
            )
        if spec.produce and not (root / f"{spec.produce}.json").is_file():
            entries.append(
                {
                    "id": spec.produce,
                    "lang": f"item.{spec.produce}",
                    "tab": ["materials"],
                    "model": "flat",
                    "texture": spec.produce,
                    "max_stack": 64,
                    "left": "none",
                    "right": "eat",
                    "food": spec.food,
                }
            )
    return entries


def loot_table_for(table_id, folder=None):
    """Loot implied by a crop file, or None when this id is not one of them."""
    if not table_id or not str(table_id).startswith("crops/"):
        return None
    for spec in load_crop_defs(folder):
        if not spec.seeds or not spec.produce:
            continue
        if table_id == f"crops/{spec.id}_young":
            return _pool(({"type": "item", "name": spec.seeds, "count": 1},))
        if table_id == f"crops/{spec.id}_ripe":
            return _pool(
                (
                    {"type": "item", "name": spec.produce, "count": 1},
                    {"type": "item", "name": spec.seeds, "count": [1, 2]},
                )
            )
        if spec.wild is not None and table_id == f"crops/{spec.id}_wild":
            return _pool(
                (
                    {"type": "item", "name": spec.seeds, "count": 1},
                    {"type": "item", "name": spec.produce, "count": 1, "chance": 0.35},
                )
            )
    return None


def wild_plants(folder=None):
    plants = []
    for spec in load_crop_defs(folder):
        if spec.wild is not None:
            plants.append(spec.wild)
    return tuple(plants)


def _plant_block(block_id, lang, texture, loot_table, valid_ground):
    entry = {
        "id": block_id,
        "lang": lang,
        "creative": False,
        "variant": True,
        "model": "cross",
        "render": "cutout",
        "solid": False,
        "hardness": 0.0,
        "tool": "none",
        "mining_level": 0,
        "fluid_breakable": True,
        "plant": True,
        "textures": {"cross": texture},
    }
    if loot_table:
        entry["loot_table"] = loot_table
    if valid_ground is not None:
        entry["valid_ground"] = list(valid_ground)
    return entry


def _pool(entries):
    return {"type": "block", "pools": [{"rolls": 1, "entries": [dict(entry) for entry in entries]}]}


def _optional_id(value, name, source):
    if value is None or value == "":
        return None
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{source}: {name} must be a lowercase name")
    return value


def _read_file(path):
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".lua":
        from server.lua import load_table

        return load_table(path)
    return None


def _number(value, name, source):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: {name} must be a number")
    return float(value)
