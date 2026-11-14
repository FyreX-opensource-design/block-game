import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from server.blocks import ASSETS, BLOCKS, STONE, WORLD_HEIGHT
from server.lua import load_table
from server.noise import fbm3, fbm3_grid

_ID = re.compile(r"[a-z0-9_]+")


class UndergroundBiome:
    def __init__(
        self,
        biome_id: str,
        name: Optional[str] = None,
        enabled: bool = True,
        min_y: int = 1,
        max_y: int = WORLD_HEIGHT - 1,
        scale: float = 0.02,
        octaves: int = 2,
        seed_offset: int = 0,
        threshold: float = 0.55,
        default_stone: str = STONE,
        stone_types: Optional[Sequence[str]] = None,
        stone_layers: Optional[Sequence[Dict[str, Any]]] = None,
        cave_fill: Optional[Dict[str, str]] = None,
        cave_generation: Optional[Any] = None,
        features: Optional[Sequence[Dict[str, Any]]] = None,
        ores: Optional[Sequence[Any]] = None,
        mob_spawns: Optional[Sequence[Dict[str, Any]]] = None,
        source: Optional[str] = None,
    ):
        self.id = biome_id
        self.name = name or biome_id.replace("_", " ").title()
        self.enabled = bool(enabled)
        self.min_y = int(min_y)
        self.max_y = int(max_y)
        self.scale = float(scale)
        self.octaves = int(octaves)
        self.seed_offset = int(seed_offset)
        self.threshold = float(threshold)
        self.default_stone = default_stone if default_stone in BLOCKS else STONE
        self.stone_types = tuple(stone_types) if stone_types is not None else None
        self.stone_layers = tuple(stone_layers or ())
        self.cave_fill = dict(cave_fill or {})
        self.cave_generation = cave_generation
        self.features = tuple(features or ())
        self.ores = tuple(ores or ())
        self.mob_spawns = tuple(mob_spawns) if mob_spawns is not None else None
        self.source = source

    def sample(self, x: float, y: float, z: float, seed: int) -> float:
        if not self.enabled or y < self.min_y or y > self.max_y:
            return 0.0
        return fbm3(
            x * self.scale,
            y * self.scale,
            z * self.scale,
            self.octaves,
            seed + self.seed_offset,
        )

    def stone_at(self, y: int) -> str:
        for layer in self.stone_layers:
            ly_min = layer.get("min_y", 0)
            ly_max = layer.get("max_y", WORLD_HEIGHT)
            if ly_min <= y <= ly_max:
                st = layer.get("stone", self.default_stone)
                if st in BLOCKS:
                    return st
        return self.default_stone


def load_underground_biomes(folder=None) -> Tuple[UndergroundBiome, ...]:
    root = Path(folder) if folder is not None else ASSETS / "underground_biomes"
    if not root.is_dir():
        return ()
    biomes = []
    seen = set()
    for path in sorted(root.iterdir()):
        if path.suffix == ".json":
            try:
                fields = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
        elif path.suffix == ".lua":
            try:
                fields = load_table(path)
            except Exception:
                continue
        else:
            continue

        if not isinstance(fields, dict):
            continue
        biome_id = fields.get("id")
        if not isinstance(biome_id, str) or not _ID.fullmatch(biome_id):
            continue
        if biome_id in seen:
            continue
        seen.add(biome_id)

        biomes.append(
            UndergroundBiome(
                biome_id=biome_id,
                name=fields.get("name"),
                enabled=fields.get("enabled", True),
                min_y=fields.get("min_y", 1),
                max_y=fields.get("max_y", WORLD_HEIGHT - 1),
                scale=fields.get("scale", 0.02),
                octaves=fields.get("octaves", 2),
                seed_offset=fields.get("seed_offset", 0),
                threshold=fields.get("threshold", 0.55),
                default_stone=fields.get("default_stone", STONE),
                stone_types=fields.get("stone_types"),
                stone_layers=fields.get("stone_layers"),
                cave_fill=fields.get("cave_fill"),
                cave_generation=fields.get("cave_generation"),
                features=fields.get("features"),
                ores=fields.get("ores"),
                mob_spawns=fields.get("mob_spawns"),
                source=str(path),
            )
        )
    return tuple(biomes)


def underground_biome_at(
    x: float, y: float, z: float, seed: int, biomes: Sequence[UndergroundBiome]
) -> Optional[UndergroundBiome]:
    """Return the active UndergroundBiome at (x, y, z), or None if none exceeds threshold."""
    best_biome = None
    best_score = -1.0
    for biome in biomes:
        if not biome.enabled or y < biome.min_y or y > biome.max_y:
            continue
        val = biome.sample(x, y, z, seed)
        if val >= biome.threshold and (val - biome.threshold) > best_score:
            best_score = val - biome.threshold
            best_biome = biome
    return best_biome


def default_stone_strata(y: int, seed: int = 0, x: float = 0.0, z: float = 0.0) -> str:
    """Return realistic geological stone strata based on elevation and subtle noise."""
    offset = (fbm3(x * 0.04, y * 0.04, z * 0.04, 1, seed + 109) - 0.5) * 6.0
    eff_y = y + offset
    if eff_y > 45:
        return "andesite" if (y % 4 < 2) else STONE
    elif eff_y > 32:
        return "diorite" if (y % 5 < 2) else STONE
    elif eff_y > 18:
        return "granite" if (y % 4 < 2) else STONE
    elif eff_y > 8:
        return "marble" if (y % 5 < 2) else STONE
    else:
        return "marble" if (y % 3 == 0) else STONE

