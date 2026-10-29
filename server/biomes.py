import json
import re
from pathlib import Path

from server.blocks import ASSETS, BLOCKS
from server.decorations import DECORATIONS, DecorationFeature, decoration_from_mapping
from server.lua import load_table
from server.noise import fbm, fbm_grid
from server.trees import DECORATORS

_ID = re.compile(r"[a-z0-9_]+")


class Biome:
    def __init__(
        self,
        biome_id,
        surface,
        soil,
        soil_depth,
        tree_chance,
        ceiling,
        scale,
        octaves,
        seed_offset,
        trees,
        source,
        decorations=(),
        height_offset=0.0,
        height_scale=1.0,
        cliff=False,
        cliff_step=0.0,
        blend=None,
        name=None,
        grass_color=None,
        foliage_color=None,
        colors=None,
        inland=None,
        transition=None,
        transition_ceiling=None,
        transition_width=None,
        floor=None,
        floor_detail=None,
        curve="smoothstep",
    ):
        self.id = biome_id
        self.name = str(name or biome_id.replace("_", " ").title())
        self.surface = surface
        self.soil = soil
        self.soil_depth = soil_depth
        self.tree_chance = tree_chance
        self.ceiling = ceiling
        self.scale = scale
        self.octaves = octaves
        self.seed_offset = seed_offset
        self.trees = trees
        self.source = source
        self.decorations = tuple(decorations)
        self.height_offset = float(height_offset)
        self.height_scale = float(height_scale)
        self.cliff = bool(cliff)
        self.cliff_step = float(cliff_step)
        self.transition = dict(transition or {})

        trans_blend = self.transition.get("blend")
        actual_blend = blend if blend is not None else trans_blend
        self.blend = float(actual_blend) if actual_blend is not None else (0.012 if self.cliff else 0.08)

        if inland is not None:
            self.inland = bool(inland)
        elif "inland" in self.transition:
            self.inland = bool(self.transition["inland"])
        else:
            self.inland = not (biome_id in ("ocean", "beach"))

        tc = transition_ceiling if transition_ceiling is not None else self.transition.get("ceiling")
        self.transition_ceiling = float(tc) if tc is not None else None

        tw = transition_width if transition_width is not None else self.transition.get("width")
        self.transition_width = float(tw) if tw is not None else None

        fl = floor if floor is not None else self.transition.get("floor")
        self.floor = float(fl) if fl is not None else None

        fd = floor_detail if floor_detail is not None else self.transition.get("floor_detail")
        self.floor_detail = float(fd) if fd is not None else None

        c_name = curve if curve != "smoothstep" else (self.transition.get("curve") or "smoothstep")
        self.curve = str(c_name).lower()

        self.grass_color = parse_color(grass_color, default=(0.51, 0.72, 0.29))
        self.foliage_color = parse_color(foliage_color, default=self.grass_color)
        self.colors = dict(colors or {})
        self.colors["grass"] = self.grass_color
        self.colors["foliage"] = self.foliage_color

    def get_color(self, tint_type="grass"):
        return self.colors.get(tint_type, self.grass_color)


def parse_color(val, default=(0.51, 0.72, 0.29)):
    if val is None:
        return default
    if isinstance(val, (tuple, list)):
        if len(val) >= 3:
            r, g, b = float(val[0]), float(val[1]), float(val[2])
            if r > 1.0 or g > 1.0 or b > 1.0:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            return (r, g, b)
        return default
    if isinstance(val, int):
        return (((val >> 16) & 0xFF) / 255.0, ((val >> 8) & 0xFF) / 255.0, (val & 0xFF) / 255.0)
    if isinstance(val, str):
        s = val.strip().lstrip("#")
        if len(s) == 6:
            try:
                return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)
            except ValueError:
                pass
    return default


def load_biomes(folder=None, trees=None, decorations=None):
    root = Path(folder) if folder is not None else ASSETS / "biomes"
    if not root.is_dir():
        return ()
    catalog = {tree.id: tree for tree in (DECORATORS if trees is None else trees)}
    dec_catalog = {dec.id: dec for dec in (DECORATIONS if decorations is None else decorations)}
    biomes = []
    seen = set()
    for path in sorted(root.iterdir()):
        fields = _read_file(path)
        if fields is None:
            continue
        biome = biome_from_mapping(fields, path, catalog, dec_catalog)
        if biome.id in seen:
            raise ValueError(f"{path}: duplicate biome id {biome.id}")
        seen.add(biome.id)
        biomes.append(biome)
    if not biomes:
        return ()
    ordered = tuple(sorted(biomes, key=lambda biome: (biome.ceiling, biome.id)))
    first = ordered[0]
    for earlier, later in zip(ordered, ordered[1:]):
        if earlier.ceiling >= later.ceiling:
            raise ValueError(f"{later.source}: biome bands must climb")
    if ordered[-1].ceiling < 1:
        raise ValueError("the highest biome band must reach 1")
    for biome in ordered[1:]:
        if (biome.scale, biome.octaves, biome.seed_offset) != (first.scale, first.octaves, first.seed_offset):
            raise ValueError(f"{biome.source}: biomes must share one climate noise")
    return ordered


def biome_from_mapping(fields, source, trees, decorations=None):
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: biome file must be an object")
    if decorations is None:
        decorations = {dec.id: dec for dec in DECORATIONS}
    biome_id = fields.get("id")
    if not isinstance(biome_id, str) or _ID.fullmatch(biome_id) is None:
        raise ValueError(f"{source}: biome id must be a lowercase name")
    surface = _block(fields.get("surface", "grass"), "surface", source)
    soil = _block(fields.get("soil", "dirt"), "soil", source)
    soil_depth = _whole(fields.get("soil_depth", 3), "soil_depth", source)
    tree_chance = _number(fields.get("tree_chance", 0), "tree_chance", source)
    ceiling = _number(fields["ceiling"], "ceiling", source) if "ceiling" in fields else None
    if ceiling is None:
        raise ValueError(f"{source}: ceiling must be a number")
    scale = _number(fields.get("scale", 0.006), "scale", source)
    octaves = _whole(fields.get("octaves", 2), "octaves", source)
    seed_offset = _whole(fields.get("seed_offset", 0), "seed_offset", source)
    names = fields.get("trees", [])
    if isinstance(names, str) or not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"{source}: trees must be a list of tree ids")
    resolved = []
    for name in names:
        if name not in trees:
            raise ValueError(f"{source}: unknown tree {name}")
        if name not in {tree.id for tree in resolved}:
            resolved.append(trees[name])
    if soil_depth < 1:
        raise ValueError(f"{source}: soil_depth must be at least 1")
    if not 0 <= tree_chance <= 1 or not 0 < ceiling <= 1:
        raise ValueError(f"{source}: tree_chance and ceiling must stay between 0 and 1")
    if scale <= 0 or octaves < 1 or octaves > 5:
        raise ValueError(f"{source}: climate noise must stay positive")

    raw_decs = fields.get("decorations", [])
    resolved_decs = []
    if isinstance(raw_decs, list):
        for item in raw_decs:
            if isinstance(item, str):
                if item not in decorations:
                    raise ValueError(f"{source}: unknown decoration {item}")
                resolved_decs.append(decorations[item])
            elif isinstance(item, dict):
                if "feature" in item or ("id" in item and item["id"] in decorations):
                    fid = item.get("feature") or item.get("id")
                    base_dec = decorations[fid]
                    chance = _number(item.get("chance", base_dec.chance), "chance", source)
                    surfaces = item.get("surfaces", base_dec.surfaces)
                    seed_offset = _whole(item.get("seed_offset", base_dec.seed_offset), "seed_offset", source)
                    resolved_decs.append(
                        DecorationFeature(
                            base_dec.id,
                            base_dec.blocks,
                            chance=chance,
                            surfaces=surfaces,
                            seed_offset=seed_offset,
                            source=f"{source}:{base_dec.id}",
                        )
                    )
                else:
                    if "id" not in item:
                        item = dict(item)
                        item["id"] = f"{biome_id}_dec_{len(resolved_decs)}"
                    resolved_decs.append(decoration_from_mapping(item, source))
            elif isinstance(item, DecorationFeature):
                resolved_decs.append(item)
            else:
                raise ValueError(f"{source}: decorations must be a list of decoration names or definitions")
    elif raw_decs:
        raise ValueError(f"{source}: decorations must be a list")

    height_offset = _number(fields.get("height_offset", 0.0), "height_offset", source)
    height_scale = _number(fields.get("height_scale", 1.0), "height_scale", source)
    cliff = bool(fields.get("cliff", False))
    cliff_step = _number(fields.get("cliff_step", 0.0), "cliff_step", source)
    blend = _number(fields["blend"], "blend", source) if "blend" in fields else None

    trans_map = fields.get("transition")
    if trans_map is not None and not isinstance(trans_map, dict):
        raise ValueError(f"{source}: transition must be an object")
    trans = dict(trans_map or {})

    inland_val = fields.get("inland") if "inland" in fields else trans.get("inland")
    trans_ceiling = fields.get("transition_ceiling") or trans.get("ceiling")
    trans_width = fields.get("transition_width") or trans.get("width")
    floor_val = fields.get("floor") or trans.get("floor")
    floor_detail_val = fields.get("floor_detail") or trans.get("floor_detail")
    curve_val = fields.get("curve") or trans.get("curve", "smoothstep")

    if blend is None and "blend" in trans:
        blend = _number(trans["blend"], "blend", source)
    if not cliff and "cliff" in trans:
        cliff = bool(trans["cliff"])
    if cliff_step == 0.0 and ("step" in trans or "cliff_step" in trans):
        cliff_step = _number(trans.get("cliff_step") or trans.get("step"), "cliff_step", source)

    name = fields.get("name")
    grass_color = parse_color(fields.get("grass_color"), default=None)
    foliage_color = parse_color(fields.get("foliage_color"), default=grass_color)
    raw_colors = fields.get("colors")
    colors = {}
    if isinstance(raw_colors, dict):
        for k, v in raw_colors.items():
            colors[k] = parse_color(v)

    return Biome(
        biome_id,
        surface,
        soil,
        soil_depth,
        tree_chance,
        ceiling,
        scale,
        octaves,
        seed_offset,
        tuple(resolved),
        str(source),
        decorations=tuple(resolved_decs),
        height_offset=height_offset,
        height_scale=height_scale,
        cliff=cliff,
        cliff_step=cliff_step,
        blend=blend,
        name=name,
        grass_color=grass_color,
        foliage_color=foliage_color,
        colors=colors,
        inland=inland_val,
        transition=trans,
        transition_ceiling=trans_ceiling,
        transition_width=trans_width,
        floor=floor_val,
        floor_detail=floor_detail_val,
        curve=curve_val,
    )


def climate_grid(x, z, seed, biomes):
    biome = biomes[0]
    return fbm_grid(x * biome.scale, z * biome.scale, biome.octaves, int(seed) + biome.seed_offset)


def biome_at(x, z, seed, biomes):
    biome = biomes[0]
    sample = fbm(int(x) * biome.scale, int(z) * biome.scale, biome.octaves, int(seed) + biome.seed_offset)
    return pick_biome(sample, biomes)


def pick_biome(sample, biomes):
    for biome in biomes:
        if sample <= biome.ceiling:
            return biome
    return biomes[-1]


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


BIOMES = load_biomes()
