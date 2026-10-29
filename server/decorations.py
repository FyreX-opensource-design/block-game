import json
import re
from pathlib import Path

from server.blocks import ASSETS, BLOCKS
from server.lua import load_table
from server.noise import hash2

_ID = re.compile(r"[a-z0-9_]+")


class DecorationFeature:
    def __init__(
        self,
        feature_id,
        blocks,
        chance=0.1,
        surfaces=("grass",),
        seed_offset=0,
        source=None,
    ):
        self.id = feature_id
        self.blocks = tuple(blocks)  # tuple of (block_id, weight)
        self.chance = float(chance)
        self.surfaces = tuple(surfaces)
        self.seed_offset = int(seed_offset)
        self.source = str(source) if source else feature_id

    @property
    def block(self):
        """Convenience property for single-block features."""
        return self.blocks[0][0] if self.blocks else ""

    def pick_block(self, wx, wz, seed):
        if not self.blocks:
            return ""
        if len(self.blocks) == 1:
            return self.blocks[0][0]
        total_weight = sum(w for _, w in self.blocks)
        if total_weight <= 0:
            return self.blocks[0][0]
        roll = hash2(wx + 31, wz - 17, seed + self.seed_offset) * total_weight
        accum = 0.0
        for block_id, weight in self.blocks:
            accum += weight
            if roll <= accum:
                return block_id
        return self.blocks[-1][0]


def load_decorations(folder=None):
    root = Path(folder) if folder is not None else ASSETS / "decorations"
    if not root.is_dir():
        return ()
    decorations = []
    seen = set()
    for path in sorted(root.iterdir()):
        fields = _read_file(path)
        if fields is None:
            continue
        dec = decoration_from_mapping(fields, path)
        if dec.id in seen:
            raise ValueError(f"{path}: duplicate decoration id {dec.id}")
        seen.add(dec.id)
        decorations.append(dec)
    return tuple(decorations)


def decoration_from_mapping(fields, source):
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: decoration file must be an object")
    dec_id = fields.get("id")
    if not isinstance(dec_id, str) or _ID.fullmatch(dec_id) is None:
        raise ValueError(f"{source}: decoration id must be a lowercase name")

    chance = _number(fields.get("chance", 0.1), "chance", source)
    if not 0 <= chance <= 1:
        raise ValueError(f"{source}: chance must stay between 0 and 1")

    seed_offset = _whole(fields.get("seed_offset", 0), "seed_offset", source)

    raw_surfaces = fields.get("surfaces", ["grass"])
    if isinstance(raw_surfaces, str):
        raw_surfaces = [raw_surfaces]
    if not isinstance(raw_surfaces, (list, tuple)):
        raise ValueError(f"{source}: surfaces must be a list of block names")
    surfaces = []
    for s in raw_surfaces:
        if s not in BLOCKS:
            raise ValueError(f"{source}: unknown surface block {s}")
        if s not in surfaces:
            surfaces.append(s)

    blocks = []
    if "block" in fields:
        b = _block(fields["block"], "block", source)
        blocks.append((b, 1.0))
    elif "blocks" in fields:
        raw_blocks = fields["blocks"]
        if isinstance(raw_blocks, dict):
            for b_name, weight in raw_blocks.items():
                b = _block(b_name, "block", source)
                blocks.append((b, _number(weight, "weight", source)))
        elif isinstance(raw_blocks, list):
            for item in raw_blocks:
                if isinstance(item, str):
                    b = _block(item, "block", source)
                    blocks.append((b, 1.0))
                elif isinstance(item, dict):
                    b = _block(item.get("block"), "block", source)
                    w = _number(item.get("weight", 1.0), "weight", source)
                    blocks.append((b, w))
                else:
                    raise ValueError(f"{source}: invalid block spec in blocks list")
        else:
            raise ValueError(f"{source}: blocks must be a list or dict")
    else:
        raise ValueError(f"{source}: decoration requires 'block' or 'blocks'")

    if not blocks:
        raise ValueError(f"{source}: decoration must specify at least one block")

    return DecorationFeature(
        dec_id,
        blocks,
        chance=chance,
        surfaces=surfaces,
        seed_offset=seed_offset,
        source=str(source),
    )


def _read_file(path):
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".lua":
        return load_table(path)
    return None


def _block(value, name, source):
    if not isinstance(value, str) or value not in BLOCKS:
        raise ValueError(f"{source}: {name} must be a block")
    return value


def _whole(value, name, source):
    number = _number(value, name, source)
    if int(number) != number:
        raise ValueError(f"{source}: {name} must be a whole number")
    return int(number)


def _number(value, name, source):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: {name} must be a number")
    return float(value)


DECORATIONS = load_decorations()

