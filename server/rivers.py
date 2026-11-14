"""River generation, winding paths, terrain carving, and water type determination."""

import math
import numpy as np

from server.blocks import BRACKISH_WATER, SALT_WATER, WATER, WATER_LEVEL
from server.noise import fbm, fbm_grid

# Noise parameters for winding river meanders
WARP_SCALE = 0.0030
WARP_AMP = 75.0
RIVER_SCALE = 0.0032

# Base channel and valley widths in noise distance space
BASE_CHANNEL_W = 0.015
BASE_VALLEY_W = 0.075

# Climate thresholds
HEADWATER_MAX_CLIMATE = 0.88
HEADWATER_FADE = 0.04
COAST_UPPER = 0.36
COAST_LOWER = 0.30


class RiverBiome:
    """Lightweight biome representation for river channels and estuaries."""

    def __init__(self, biome_id="river", name="River", inland=True, surface="sand", temperature=None, touching=None):
        self.id = biome_id
        self.name = name
        self.surface = surface
        self.soil = "dirt"
        self.soil_depth = 3
        self.subsoil = "stone"
        self.subsoil_depth = 3
        self.valid_spawn = False
        self.inland = inland
        self.trees = ()
        self.decorations = ()
        self.grass_color = (0.45, 0.75, 0.35)
        self.foliage_color = (0.40, 0.70, 0.30)
        self.ceiling = 1.0
        self.variants = ()
        self.temperature = tuple(temperature) if temperature is not None else None
        self.touching = tuple(touching) if touching is not None else None


RIVER_BIOME = RiverBiome("river", "River", inland=True, surface="sand", temperature=(0.0, 0.50), touching=("plains", "savanna_river", "estuary"))
RIVER_MUD_BIOME = RiverBiome("river", "River", inland=True, surface="mud", temperature=(0.0, 0.50), touching=("plains", "savanna_river", "estuary"))
SAVANNA_RIVER_BIOME = RiverBiome(
    "savanna_river",
    "Savanna River",
    inland=True,
    surface="sand",
    temperature=(0.50, 0.75),
    touching=("river", "desert_river", "savanna", "estuary"),
)
SAVANNA_RIVER_BIOME.grass_color = (0.71, 0.69, 0.27)
SAVANNA_RIVER_BIOME.foliage_color = (0.61, 0.65, 0.22)
SAVANNA_RIVER_MUD_BIOME = RiverBiome(
    "savanna_river",
    "Savanna River",
    inland=True,
    surface="mud",
    temperature=(0.50, 0.75),
    touching=("river", "desert_river", "savanna", "estuary"),
)
SAVANNA_RIVER_MUD_BIOME.grass_color = (0.71, 0.69, 0.27)
SAVANNA_RIVER_MUD_BIOME.foliage_color = (0.61, 0.65, 0.22)
DESERT_RIVER_BIOME = RiverBiome(
    "desert_river",
    "Desert River",
    inland=True,
    surface="sand",
    temperature=(0.75, 1.0),
    touching=("savanna_river", "desert", "estuary"),
)
DESERT_RIVER_BIOME.soil = "sand"
DESERT_RIVER_BIOME.subsoil = "sandstone"
DESERT_RIVER_BIOME.grass_color = (0.78, 0.73, 0.38)
DESERT_RIVER_BIOME.foliage_color = (0.72, 0.70, 0.30)
ESTUARY_BIOME = RiverBiome("estuary", "River Mouth", inland=False, surface="sand")


def _river_dist_scalar(x, z, seed):
    wx = x + fbm(x * WARP_SCALE + 15.1, z * WARP_SCALE - 11.7, 2, seed + 10101) * WARP_AMP
    wz = z + fbm(x * WARP_SCALE - 23.5, z * WARP_SCALE + 31.9, 2, seed + 20202) * WARP_AMP
    r_noise = fbm(wx * RIVER_SCALE, wz * RIVER_SCALE, 3, seed + 30303)
    return abs(r_noise - 0.5)


def _river_dist_grid(xx, zz, seed):
    wx = xx + fbm_grid(xx * WARP_SCALE + 15.1, zz * WARP_SCALE - 11.7, 2, seed + 10101) * WARP_AMP
    wz = zz + fbm_grid(xx * WARP_SCALE - 23.5, zz * WARP_SCALE + 31.9, 2, seed + 20202) * WARP_AMP
    r_noise = fbm_grid(wx * RIVER_SCALE, wz * RIVER_SCALE, 3, seed + 30303)
    return np.abs(r_noise - 0.5)


def is_river(x, z, seed, climate=None):
    """Return True if (x, z) is inside a river water channel."""
    if climate is None:
        from server.biomes import BIOMES, climate_at

        climate = climate_at(x, z, seed, BIOMES)
    headwater = max(0.0, min(1.0, (HEADWATER_MAX_CLIMATE - climate) / HEADWATER_FADE))
    if headwater < 1.0:
        return False
    estuary = 1.0 + 0.4 * max(0.0, min(1.0, (COAST_UPPER - climate) / 0.10))
    channel_w = BASE_CHANNEL_W * estuary
    r_dist = _river_dist_scalar(x, z, seed)
    return r_dist <= channel_w


def is_river_mouth(x, z, seed, climate=None):
    """Return True if (x, z) is at a river-ocean meeting point (estuary)."""
    if climate is None:
        from server.biomes import BIOMES, climate_at

        climate = climate_at(x, z, seed, BIOMES)
    if not (COAST_LOWER <= climate <= COAST_UPPER):
        return False
    r_dist = _river_dist_scalar(x, z, seed)
    return r_dist <= BASE_VALLEY_W * 1.2


def river_water_type(x, z, seed, climate):
    """Return the water block type ('water', 'brackish_water', 'salt_water') for (x, z)."""
    r_dist = _river_dist_scalar(x, z, seed)
    headwater = max(0.0, min(1.0, (HEADWATER_MAX_CLIMATE - climate) / HEADWATER_FADE))
    estuary = 1.0 + 0.4 * max(0.0, min(1.0, (COAST_UPPER - climate) / 0.10))
    channel_w = BASE_CHANNEL_W * estuary

    # River channel or mouth meeting point
    if r_dist <= channel_w * 1.5 and headwater > 0.0:
        if COAST_LOWER < climate <= COAST_UPPER:
            return BRACKISH_WATER
        elif climate > COAST_UPPER:
            return WATER
        else:
            return SALT_WATER

    # Broader river mouth meeting zone into coast
    if r_dist <= BASE_VALLEY_W * estuary and (COAST_LOWER < climate <= COAST_UPPER):
        return BRACKISH_WATER

    # Default for open ocean/submerged land
    return SALT_WATER


def river_water_type_grid(xx, zz, seed, climate, tops):
    """Return a 2D array of water block IDs for grid (xx, zz)."""
    r_dist = _river_dist_grid(xx, zz, seed)
    headwater = np.clip((HEADWATER_MAX_CLIMATE - climate) / HEADWATER_FADE, 0.0, 1.0)
    estuary = 1.0 + 0.4 * np.clip((COAST_UPPER - climate) / 0.10, 0.0, 1.0)
    channel_w = BASE_CHANNEL_W * estuary

    in_channel = (r_dist <= channel_w * 1.5) & (headwater > 0.0)
    in_mouth = (r_dist <= BASE_VALLEY_W * estuary) & (climate > COAST_LOWER) & (climate <= COAST_UPPER)

    result = np.full(xx.shape, SALT_WATER, dtype=object)
    result[in_channel & (climate > COAST_UPPER)] = WATER
    result[(in_channel | in_mouth) & (climate > COAST_LOWER) & (climate <= COAST_UPPER)] = BRACKISH_WATER
    return result


def carve_river_scalar(x, z, h_raw, seed, climate):
    """Carve river valley and channel into terrain height for scalar coordinate (x, z)."""
    headwater = max(0.0, min(1.0, (HEADWATER_MAX_CLIMATE - climate) / HEADWATER_FADE))
    if headwater <= 0.0:
        return int(h_raw)

    r_dist = _river_dist_scalar(x, z, seed)
    estuary = 1.0 + 0.35 * max(0.0, min(1.0, (COAST_UPPER - climate) / 0.10))
    channel_w = BASE_CHANNEL_W * estuary

    # Dynamic bank width: scales with required carve depth (h_raw - 64)
    # Lowland/plains have tight, natural banks (3-6 blocks) to avoid flattening the world,
    # while mountains scale up to 0.055 to ensure slopes stay <= 2.5 blocks/m.
    dh = max(2.0, float(h_raw) - float(WATER_LEVEL))
    bank_w = max(0.010, min(0.055, dh * 0.0016)) * estuary
    valley_w = channel_w + bank_w

    if r_dist >= valley_w:
        return int(h_raw)

    bed_h = float(WATER_LEVEL - 3)
    if r_dist <= channel_w:
        v = r_dist / max(1e-6, channel_w)
        target_h = bed_h + (WATER_LEVEL - bed_h) * (v**2)
    else:
        u = (r_dist - channel_w) / max(1e-6, valley_w - channel_w)
        s = u**1.2
        target_h = float(WATER_LEVEL) + (float(h_raw) - float(WATER_LEVEL)) * s

    h_c = target_h * headwater + float(h_raw) * (1.0 - headwater)
    h_carved = min(float(h_raw), h_c)
    return int(max(50, min(122, round(h_carved))))


def carve_river_grid(xx, zz, heights, seed, climate):
    """Carve river valley and channel into terrain heights for a 2D grid."""
    headwater = np.clip((HEADWATER_MAX_CLIMATE - climate) / HEADWATER_FADE, 0.0, 1.0)
    estuary = 1.0 + 0.35 * np.clip((COAST_UPPER - climate) / 0.10, 0.0, 1.0)
    channel_w = BASE_CHANNEL_W * estuary

    dh = np.maximum(2.0, heights.astype(np.float64) - float(WATER_LEVEL))
    bank_w = np.clip(dh * 0.0016, 0.010, 0.055) * estuary
    valley_w = channel_w + bank_w

    r_dist = _river_dist_grid(xx, zz, seed)

    bed_h = float(WATER_LEVEL - 3)
    v = np.clip(r_dist / np.maximum(1e-6, channel_w), 0.0, 1.0)
    channel_bed = bed_h + (WATER_LEVEL - bed_h) * (v**2)

    u = np.clip((r_dist - channel_w) / np.maximum(1e-6, valley_w - channel_w), 0.0, 1.0)
    s = u**1.2
    bank_h = float(WATER_LEVEL) + (heights.astype(np.float64) - float(WATER_LEVEL)) * s

    in_channel = (r_dist <= channel_w) & (headwater > 0.0)
    in_valley = (r_dist <= valley_w) & (headwater > 0.0)

    target_h = np.where(in_channel, channel_bed, np.where(in_valley, bank_h, heights.astype(np.float64)))
    h_c = target_h * headwater + heights.astype(np.float64) * (1.0 - headwater)
    carved = np.minimum(heights.astype(np.float64), h_c)
    return np.clip(np.rint(carved), 50, 122).astype(heights.dtype)


def is_river_channel_grid(xx, zz, seed, climate):
    """Return boolean 2D mask of river channels and immediate banks."""
    r_dist = _river_dist_grid(xx, zz, seed)
    headwater = np.clip((HEADWATER_MAX_CLIMATE - climate) / HEADWATER_FADE, 0.0, 1.0)
    estuary = 1.0 + 0.35 * np.clip((COAST_UPPER - climate) / 0.10, 0.0, 1.0)
    channel_w = BASE_CHANNEL_W * estuary
    return (r_dist <= channel_w * 2.0) & (headwater > 0.0)


def river_bed_material_grid(xx, zz, seed):
    """Return 2D array of block IDs ('sand' or 'mud') for river bed columns."""
    from server.blocks import MUD, SAND

    bed_noise = fbm_grid(xx * 0.06 + 33.0, zz * 0.06 - 44.0, 2, seed + 888)
    res = np.full(xx.shape, SAND, dtype=object)
    res[bed_noise > 0.48] = MUD
    return res


def river_bed_material(x, z, seed):
    """Return 'sand' or 'mud' for river bed coordinate (x, z)."""
    from server.blocks import MUD, SAND

    bed_noise = fbm(x * 0.06 + 33.0, z * 0.06 - 44.0, 2, seed + 888)
    return MUD if bed_noise > 0.48 else SAND


def river_biome_at(x, y, z, seed, biomes):
    """Return a River or Estuary biome if (x, y, z) is in a river channel, else None."""
    from server.biomes import climate_at, temperature_at

    climate = climate_at(x, z, seed, biomes)
    if not is_river(x, z, seed, climate=climate):
        return None
    if y is not None and y > WATER_LEVEL + 4:
        return None
    if COAST_LOWER <= climate <= COAST_UPPER:
        return ESTUARY_BIOME

    temp = temperature_at(x, z, seed)
    is_mud = river_bed_material(x, z, seed) == "mud"
    if temp >= 0.75:
        return DESERT_RIVER_BIOME
    elif temp >= 0.50:
        return SAVANNA_RIVER_MUD_BIOME if is_mud else SAVANNA_RIVER_BIOME
    else:
        return RIVER_MUD_BIOME if is_mud else RIVER_BIOME

