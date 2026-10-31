import functools
import json
import re
from pathlib import Path

import numpy as np

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
        min_size=0.0,
        valid_spawn=None,
        variant_of=None,
        variant_noise=None,
        coast=None,
        variants=(),
        subsoil=None,
        subsoil_depth=0.0,
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
        self.min_size = float(min_size)
        self.valid_spawn = True if valid_spawn is None else bool(valid_spawn)
        self.variant_of = variant_of
        self.variant_noise = dict(variant_noise or {})
        self.coast = dict(coast or {})
        self.variants = tuple(variants)
        self.subsoil = subsoil
        self.subsoil_depth = float(subsoil_depth)

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

    def coast_noise_params(self):
        noise = self.coast.get("noise", {})
        return {
            "scale": float(noise.get("scale", 0.018)),
            "octaves": int(noise.get("octaves", 3)),
            "seed_offset": int(noise.get("seed_offset", 7331)),
            "offset": tuple(noise.get("offset", (7.3, -11.9))),
        }

    def _variant_value(self, x, z, seed):
        cfg = self.variant_noise
        scale = float(cfg.get("scale", 0.018))
        octaves = int(cfg.get("octaves", 3))
        seed_offset = int(cfg.get("seed_offset", 7331))
        offset = cfg.get("offset", (7.3, -11.9))
        ox, oz = float(offset[0]), float(offset[1])
        return fbm(x * scale + ox, z * scale + oz, octaves, seed + seed_offset)

    def matches(self, x, z, seed):
        if not self.variant_noise:
            return False
        value = self._variant_value(x, z, seed)
        threshold = float(self.variant_noise.get("threshold", 0.5))
        if bool(self.variant_noise.get("above", True)):
            return value >= threshold
        return value <= threshold

    def variant_mask_grid(self, x, z, seed):
        cfg = self.variant_noise
        if not cfg:
            return np.zeros(np.broadcast(x, z).shape, dtype=bool)
        scale = float(cfg.get("scale", 0.018))
        octaves = int(cfg.get("octaves", 3))
        seed_offset = int(cfg.get("seed_offset", 7331))
        offset = cfg.get("offset", (7.3, -11.9))
        ox, oz = float(offset[0]), float(offset[1])
        value = fbm_grid(x * scale + ox, z * scale + oz, octaves, seed + seed_offset)
        threshold = float(cfg.get("threshold", 0.5))
        if bool(cfg.get("above", True)):
            return value >= threshold
        return value <= threshold

    def variant_at(self, x, z, seed):
        for variant in self.variants:
            if variant.matches(x, z, seed):
                return variant
        return self


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
    by_id = {biome.id: biome for biome in biomes}
    base = [biome for biome in biomes if biome.variant_of is None]
    variants = [biome for biome in biomes if biome.variant_of is not None]
    if not base:
        raise ValueError("at least one biome must not be a variant")
    for variant in variants:
        parent = by_id.get(variant.variant_of)
        if parent is None:
            raise ValueError(f"{variant.source}: unknown variant_of biome {variant.variant_of}")
        if parent.variant_of is not None:
            raise ValueError(f"{variant.source}: variant_of must point at a base biome")
        if not variant.variant_noise:
            raise ValueError(f"{variant.source}: variant biomes need variant_noise")
        if variant.ceiling != parent.ceiling:
            raise ValueError(f"{variant.source}: variant ceiling must match {parent.id}")
        if (variant.scale, variant.octaves, variant.seed_offset) != (parent.scale, parent.octaves, parent.seed_offset):
            raise ValueError(f"{variant.source}: variants must share their parent's climate noise")
        if not variant.coast:
            variant.coast = dict(parent.coast)
        noise = dict(variant.variant_noise)
        uses_cliff = bool(noise.pop("coast_cliff", False))
        parent_noise = parent.coast.get("noise", {}) if parent.coast else {}
        defaults = {
            "scale": parent_noise.get("scale", 0.018),
            "octaves": parent_noise.get("octaves", 3),
            "seed_offset": parent_noise.get("seed_offset", 7331),
            "offset": parent_noise.get("offset", (7.3, -11.9)),
        }
        for key, value in defaults.items():
            noise.setdefault(key, value)
        if uses_cliff:
            cliff = parent.coast.get("cliff")
            if not cliff:
                raise ValueError(f"{variant.source}: coast_cliff requires {parent.id} to define coast.cliff")
            # Match the boundary where the parent's smoothstep cliff factor
            # crosses 0.5 so surface and terrain agree.
            threshold = float(cliff.get("threshold", 0.42))
            blend = float(cliff.get("blend", 0.10))
            noise["threshold"] = threshold - 0.5 * blend
            noise["above"] = False
        variant.variant_noise = noise
        parent.variants = parent.variants + (variant,)
    for biome in base:
        biome.variants = tuple(sorted(biome.variants, key=lambda item: item.id))

    ordered = tuple(sorted(base, key=lambda biome: (biome.ceiling, biome.id)))
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

    raw_subsoil = fields.get("subsoil")
    subsoil = _block(raw_subsoil, "subsoil", source) if raw_subsoil is not None else _block(soil, "soil", source)
    raw_subsoil_depth = fields.get("subsoil_depth")
    subsoil_depth = _whole(raw_subsoil_depth, "subsoil_depth", source) if raw_subsoil_depth is not None else 0
    if subsoil_depth < 0:
        raise ValueError(f"{source}: subsoil_depth must be non-negative")

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
    raw_min_size = fields.get("min_size", fields.get("min_biome_size", trans.get("min_size", 0.0)))
    min_size = _number(raw_min_size, "min_size", source) if raw_min_size is not None else 0.0
    if min_size < 0:
        raise ValueError(f"{source}: min_size must be non-negative")

    raw_valid_spawn = fields.get("valid_spawn", fields.get("can_spawn", fields.get("spawnable", None)))
    if raw_valid_spawn is not None and not isinstance(raw_valid_spawn, bool):
        raise ValueError(f"{source}: valid_spawn must be a boolean")

    variant_of = fields.get("variant_of")
    if variant_of is not None and (not isinstance(variant_of, str) or _ID.fullmatch(variant_of) is None):
        raise ValueError(f"{source}: variant_of must be a biome id")

    raw_variant_noise = fields.get("variant_noise")
    if raw_variant_noise is not None and not isinstance(raw_variant_noise, dict):
        raise ValueError(f"{source}: variant_noise must be an object")
    variant_noise = dict(raw_variant_noise or {})

    raw_coast = fields.get("coast")
    if raw_coast is not None and not isinstance(raw_coast, dict):
        raise ValueError(f"{source}: coast must be an object")
    coast = dict(raw_coast or {})

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
        min_size=min_size,
        valid_spawn=raw_valid_spawn,
        variant_of=variant_of,
        variant_noise=variant_noise,
        coast=coast,
        subsoil=subsoil,
        subsoil_depth=subsoil_depth,
    )


@functools.lru_cache(maxsize=64)
def _biome_climate_params(biomes_tuple):
    if not biomes_tuple:
        return 0.006, 0.0, (), ()
    b0 = biomes_tuple[0]
    s = float(b0.scale)
    c = tuple(float(b.ceiling) for b in biomes_tuple)
    pos_sizes = [float(getattr(b, "min_size", 0.0)) for b in biomes_tuple if getattr(b, "min_size", 0.0) > 0.0]
    if not pos_sizes:
        return s, 0.0, c, c

    g0 = 0.36
    n = len(biomes_tuple)
    dc = [c[0]] + [c[i] - c[i - 1] for i in range(1, n)]
    m = [max(0.0, float(getattr(b, "min_size", 0.0))) for b in biomes_tuple]

    if n >= 3:
        interior_span = max(1e-6, c[n - 2] - c[0])
        req = [max(dc[i], m[i] * g0 * s) for i in range(1, n - 1)]
        req_sum = sum(req)
        c0_eff = max(0.05, min(dc[0], 0.12))
        cn_eff = max(0.05, min(dc[n - 1], 0.10))
        stretch = max(
            1.0,
            req_sum / interior_span,
            (m[0] * g0 * s) / c0_eff,
            (m[n - 1] * g0 * s) / cn_eff,
        )
        r = [0.0] * n
        r[0] = c[0]
        cur = c[0]
        for idx, i in enumerate(range(1, n - 1)):
            cur += interior_span * (req[idx] / req_sum)
            r[i] = cur
        r[n - 1] = c[n - 1]
    elif n == 2:
        w0 = max(dc[0], m[0] * g0 * s)
        w1 = max(dc[1], m[1] * g0 * s)
        total_span = max(1e-6, c[1])
        stretch = max(1.0, (w0 + w1) / total_span)
        r = [total_span * (w0 / (w0 + w1)), c[1]]
    else:
        stretch = max(1.0, m[0] * g0 * s)
        r = [c[0]]

    s_eff = s / stretch
    d = min(14.0, 0.45 * (sum(pos_sizes) / len(pos_sizes)))
    return s_eff, d, tuple(r), c


def _remap_climate_scalar(val, r_tuple, c_tuple):
    if r_tuple == c_tuple:
        return val
    prev_r = 0.0
    prev_c = 0.0
    n = len(r_tuple)
    for i in range(n):
        ri = r_tuple[i]
        ci = c_tuple[i]
        if val <= ri or i == n - 1:
            span_r = max(1e-9, ri - prev_r)
            t = (val - prev_r) / span_r
            return prev_c + t * (ci - prev_c)
        prev_r = ri
        prev_c = ci
    return val


def _remap_climate_grid(grid, r_tuple, c_tuple):
    if r_tuple == c_tuple:
        return grid
    xp = (0.0,) + r_tuple
    fp = (0.0,) + c_tuple
    return np.interp(grid, xp, fp)


def _median3_scalar(a, b, c):
    if a > b:
        a, b = b, a
    if b > c:
        b = a if a > c else c
    return b


def _median3_grid(a, b, c):
    ab_min = np.minimum(a, b)
    ab_max = np.maximum(a, b)
    return np.maximum(ab_min, np.minimum(ab_max, c))


def climate_grid(x, z, seed, biomes):
    biomes_tuple = tuple(biomes)
    biome = biomes_tuple[0]
    s_eff, d, r_tuple, c_tuple = _biome_climate_params(biomes_tuple)
    f_seed = int(seed) + biome.seed_offset
    if d <= 0.0:
        raw = fbm_grid(x * s_eff, z * s_eff, biome.octaves, f_seed)
        return _remap_climate_grid(raw, r_tuple, c_tuple)

    c0 = fbm_grid(x * s_eff, z * s_eff, biome.octaves, f_seed)
    cx1 = fbm_grid((x - d) * s_eff, z * s_eff, biome.octaves, f_seed)
    cx2 = fbm_grid((x + d) * s_eff, z * s_eff, biome.octaves, f_seed)
    cz1 = fbm_grid(x * s_eff, (z - d) * s_eff, biome.octaves, f_seed)
    cz2 = fbm_grid(x * s_eff, (z + d) * s_eff, biome.octaves, f_seed)
    med_x = _median3_grid(cx1, c0, cx2)
    raw = _median3_grid(cz1, med_x, cz2)
    return _remap_climate_grid(raw, r_tuple, c_tuple)


@functools.lru_cache(maxsize=262144)
def _cached_climate_at(x, z, seed, biomes_tuple):
    biome = biomes_tuple[0]
    s_eff, d, r_tuple, c_tuple = _biome_climate_params(biomes_tuple)
    f_seed = seed + biome.seed_offset
    fx = float(x)
    fz = float(z)
    if d <= 0.0:
        raw = fbm(fx * s_eff, fz * s_eff, biome.octaves, f_seed)
        return _remap_climate_scalar(raw, r_tuple, c_tuple)

    c0 = fbm(fx * s_eff, fz * s_eff, biome.octaves, f_seed)
    cx1 = fbm((fx - d) * s_eff, fz * s_eff, biome.octaves, f_seed)
    cx2 = fbm((fx + d) * s_eff, fz * s_eff, biome.octaves, f_seed)
    cz1 = fbm(fx * s_eff, (fz - d) * s_eff, biome.octaves, f_seed)
    cz2 = fbm(fx * s_eff, (fz + d) * s_eff, biome.octaves, f_seed)
    med_x = _median3_scalar(cx1, c0, cx2)
    raw = _median3_scalar(cz1, med_x, cz2)
    return _remap_climate_scalar(raw, r_tuple, c_tuple)


def climate_at(x, z, seed, biomes):
    return _cached_climate_at(int(x), int(z), int(seed), tuple(biomes))


def biome_at(x, z, seed, biomes):
    sample = climate_at(x, z, seed, biomes)
    return pick_biome(sample, biomes).variant_at(x, z, seed)


def pick_biome(sample, biomes):
    for biome in biomes:
        if sample <= biome.ceiling:
            return biome
    return biomes[-1]


def all_biomes(biomes):
    result = []
    for biome in biomes:
        result.append(biome)
        result.extend(biome.variants)
    return tuple(result)


def biome_index_grid(x, z, seed, biomes):
    """Return a grid of indices into ``all_biomes(biomes)`` for each coordinate."""
    biomes = tuple(biomes)
    catalog = all_biomes(biomes)
    index_by_id = {biome.id: index for index, biome in enumerate(catalog)}
    climate = climate_grid(x, z, seed, biomes)
    grid = np.full(climate.shape, index_by_id[biomes[-1].id], dtype=np.int16)
    assigned = np.zeros(climate.shape, dtype=bool)
    for biome in biomes:
        mask = (climate <= biome.ceiling) & ~assigned
        grid[mask] = index_by_id[biome.id]
        assigned |= mask
    for base in biomes:
        base_index = index_by_id[base.id]
        for variant in base.variants:
            mask = (grid == base_index) & variant.variant_mask_grid(x, z, seed)
            grid[mask] = index_by_id[variant.id]
    return grid, catalog


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
