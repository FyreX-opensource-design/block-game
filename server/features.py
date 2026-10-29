from pathlib import Path
import json
import re

from server.blocks import ASSETS
from server.lua import load_table

_ID = re.compile(r"[a-z0-9_]+")


class ConfiguredFeature:
    def __init__(self, feature_id, feature_type, data):
        self.id = feature_id
        self.type = feature_type
        self.data = dict(data)
        self.chance = float(data.get("chance", 0.025))
        self.fluid = str(data.get("fluid", "water"))
        self.radius_min = int(data.get("radius_min", data.get("min_radius", 4)))
        self.radius_max = int(data.get("radius_max", data.get("max_radius", 7)))
        self.depth = int(data.get("depth", 3))
        self.bed_blocks = tuple(str(b) for b in data.get("bed_blocks", ["dirt", "gravel", "clay"]))
        self.shore_blocks = tuple(str(b) for b in data.get("shore_blocks", ["mud", "muddy_grass"]))
        self.biomes = tuple(str(b) for b in data.get("biomes", ()))


def load_features(folder=None):
    root = Path(folder) if folder is not None else ASSETS / "features"
    if not root.is_dir():
        return ()
    features = []
    seen = set()
    for path in sorted(root.iterdir()):
        data = None
        if path.suffix == ".lua":
            data = load_table(path)
        elif path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = None
        if isinstance(data, dict):
            fid = str(data.get("id", path.stem))
            if not _ID.fullmatch(fid):
                continue
            if fid in seen:
                continue
            seen.add(fid)
            ftype = str(data.get("type", "lake"))
            features.append(ConfiguredFeature(fid, ftype, data))
    return tuple(features)


FEATURES = load_features()

