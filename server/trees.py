import json
import re
from pathlib import Path

from server.blocks import ASSETS, BLOCKS
from server.lua import load_table

_ID = re.compile(r"[a-z0-9_]+")


class TreeDecorator:
    def __init__(
        self,
        tree_id,
        trunk,
        leaves,
        trunk_min,
        trunk_extra,
        extra_threshold,
        radius,
        leaf_down,
        leaf_up,
        top_radius,
        skip_corners,
        max_manhattan,
        thin_above,
        thin_threshold,
        source,
        valid_blocks=("grass", "dirt", "muddy_grass", "mud"),
        shape="blob",
        branch_chance=0.6,
        branch_count=3,
        branch_length=2,
    ):
        self.id = tree_id
        self.trunk = trunk
        self.leaves = leaves
        self.trunk_min = trunk_min
        self.trunk_extra = trunk_extra
        self.extra_threshold = extra_threshold
        self.radius = radius
        self.leaf_down = leaf_down
        self.leaf_up = leaf_up
        self.top_radius = top_radius
        self.skip_corners = skip_corners
        self.max_manhattan = max_manhattan
        self.thin_above = thin_above
        self.thin_threshold = thin_threshold
        self.source = source
        self.valid_blocks = tuple(valid_blocks)
        self.shape = shape
        self.branch_chance = branch_chance
        self.branch_count = branch_count
        self.branch_length = branch_length


def load_trees(folder=None):
    root = Path(folder) if folder is not None else ASSETS / "trees"
    if not root.is_dir():
        return ()
    trees = []
    seen = set()
    for path in sorted(root.iterdir()):
        fields = _read_file(path)
        if fields is None:
            continue
        tree = tree_from_mapping(fields, path)
        if tree.id in seen:
            raise ValueError(f"{path}: duplicate tree id {tree.id}")
        seen.add(tree.id)
        trees.append(tree)
    return tuple(trees)


def tree_from_mapping(fields, source):
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: tree file must be an object")
    tree_id = fields.get("id")
    if not isinstance(tree_id, str) or _ID.fullmatch(tree_id) is None:
        raise ValueError(f"{source}: tree id must be a lowercase name")
    trunk = _block(fields.get("trunk", "log"), "trunk", source)
    leaves = _block(fields.get("leaves", "leaves"), "leaves", source)
    trunk_min = _whole(fields.get("trunk_min", 4), "trunk_min", source)
    trunk_extra = _whole(fields.get("trunk_extra", 1), "trunk_extra", source)
    extra_threshold = _number(fields.get("extra_threshold", 0.55), "extra_threshold", source)
    radius = _whole(fields.get("radius", 2), "radius", source)
    leaf_down = _whole(fields.get("leaf_down", 2), "leaf_down", source)
    leaf_up = _whole(fields.get("leaf_up", 2), "leaf_up", source)
    top_radius = _whole(fields.get("top_radius", 1), "top_radius", source)
    skip_corners = fields.get("skip_corners", True)
    max_manhattan = _whole(fields.get("max_manhattan", 4), "max_manhattan", source)
    thin_above = _whole(fields.get("thin_above", 2), "thin_above", source)
    thin_threshold = _number(fields.get("thin_threshold", 0.72), "thin_threshold", source)
    if not isinstance(skip_corners, bool):
        raise ValueError(f"{source}: skip_corners must be true or false")
    if trunk_min < 1 or trunk_extra < 0 or radius < 1 or leaf_down < 0 or leaf_up < 0 or top_radius < 0:
        raise ValueError(f"{source}: tree sizes must stay positive")
    if not 0 <= extra_threshold <= 1 or not 0 <= thin_threshold <= 1:
        raise ValueError(f"{source}: tree chances must stay between 0 and 1")
    shape = str(fields.get("shape", "blob")).lower()
    branch_chance = _number(fields.get("branch_chance", 0.6), "branch_chance", source)
    branch_count = _whole(fields.get("branch_count", 3), "branch_count", source)
    branch_length = _whole(fields.get("branch_length", 2), "branch_length", source)
    raw_valid = fields.get("valid_blocks", fields.get("valid_ground", ["grass", "dirt", "muddy_grass", "mud"]))
    valid_blocks = tuple(b if str(b).startswith("#") else _block(b, "valid_blocks", source) for b in raw_valid)
    return TreeDecorator(
        tree_id,
        trunk,
        leaves,
        trunk_min,
        trunk_extra,
        extra_threshold,
        radius,
        leaf_down,
        leaf_up,
        top_radius,
        skip_corners,
        max_manhattan,
        thin_above,
        thin_threshold,
        str(source),
        valid_blocks=valid_blocks,
        shape=shape,
        branch_chance=branch_chance,
        branch_count=branch_count,
        branch_length=branch_length,
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


DECORATORS = load_trees()
