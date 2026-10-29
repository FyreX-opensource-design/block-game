import functools
import math

import numpy as np

from server.blocks import STONE_BASE

MASK = 0xFFFFFFFF


def hash2(x, y, seed=0):
    n = ((x * 374761393) + (y * 668265263) + (seed * 1442695041)) & MASK
    n = ((n ^ (n >> 13)) * 1274126177) & MASK
    return (n ^ (n >> 16)) / 4294967295


def value_noise(x, y, seed=0):
    x0 = math.floor(x)
    y0 = math.floor(y)
    tx = x - x0
    ty = y - y0
    sx = tx * tx * (3 - 2 * tx)
    sy = ty * ty * (3 - 2 * ty)
    n00 = hash2(x0, y0, seed)
    n10 = hash2(x0 + 1, y0, seed)
    n01 = hash2(x0, y0 + 1, seed)
    n11 = hash2(x0 + 1, y0 + 1, seed)
    nx0 = n00 + (n10 - n00) * sx
    nx1 = n01 + (n11 - n01) * sx
    return nx0 + (nx1 - nx0) * sy


def fbm(x, y, octaves, seed=0):
    value = 0.0
    amplitude = 0.5
    frequency = 1.0
    total = 0.0
    for _ in range(octaves):
        value += amplitude * value_noise(x * frequency, y * frequency, seed)
        total += amplitude
        amplitude *= 0.5
        frequency *= 2
    return value / total


def hash2_grid(x, y, seed):
    n = (x * np.int64(374761393) + y * np.int64(668265263) + np.int64(seed) * np.int64(1442695041)) & np.int64(MASK)
    n = ((n ^ (n >> 13)) * np.int64(1274126177)) & np.int64(MASK)
    return (n ^ (n >> 16)) * (1.0 / 4294967295.0)


def value_noise_grid(x, y, seed):
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    tx = x - x0
    ty = y - y0
    sx = tx * tx * (3.0 - 2.0 * tx)
    sy = ty * ty * (3.0 - 2.0 * ty)
    n00 = hash2_grid(x0, y0, seed)
    n10 = hash2_grid(x0 + 1, y0, seed)
    n01 = hash2_grid(x0, y0 + 1, seed)
    n11 = hash2_grid(x0 + 1, y0 + 1, seed)
    nx0 = n00 + (n10 - n00) * sx
    nx1 = n01 + (n11 - n01) * sx
    return nx0 + (nx1 - nx0) * sy


def fbm_grid(x, y, octaves, seed):
    value = np.zeros(np.broadcast(x, y).shape, dtype=np.float64)
    amplitude = 0.5
    frequency = 1.0
    total = 0.0
    for _ in range(octaves):
        value += amplitude * value_noise_grid(x * frequency, y * frequency, seed)
        total += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    return value / total


def apply_curve_scalar(w, curve="smoothstep"):
    if curve == "linear":
        return w
    elif curve == "smootherstep":
        return w * w * w * (w * (w * 6.0 - 15.0) + 10.0)
    elif curve == "cosine":
        return (1.0 - math.cos(w * math.pi)) * 0.5
    return w * w * (3.0 - 2.0 * w)


def apply_curve_grid(w, curve="smoothstep"):
    if curve == "linear":
        return w
    elif curve == "smootherstep":
        return w * w * w * (w * (w * 6.0 - 15.0) + 10.0)
    elif curve == "cosine":
        return (1.0 - np.cos(w * np.pi)) * 0.5
    return w * w * (3.0 - 2.0 * w)


def _get_transition_params(biomes):
    beach_c = 0.42
    ocean_w = 0.16
    floor_base = 50.0
    floor_detail = 4.0
    ocean_curve = "smoothstep"
    inland_list = []

    for b in biomes:
        if getattr(b, "transition_ceiling", None) is not None:
            beach_c = float(b.transition_ceiling)
        if getattr(b, "transition_width", None) is not None:
            ocean_w = float(b.transition_width)
        if getattr(b, "floor", None) is not None:
            floor_base = float(b.floor)
        if getattr(b, "floor_detail", None) is not None:
            floor_detail = float(b.floor_detail)
        if not getattr(b, "inland", True):
            if getattr(b, "curve", None):
                ocean_curve = b.curve
        else:
            inland_list.append(b)

    if not inland_list:
        inland_list = [b for b in biomes if b.ceiling > beach_c]

    return beach_c, ocean_w, floor_base, floor_detail, ocean_curve, inland_list


def terrain_grid(x, z, seed, biomes=None):
    broad = fbm_grid(x * 0.014, z * 0.014, 4, seed)
    detail = fbm_grid(x * 0.05 + 12.3, z * 0.05 - 4.7, 3, seed)
    cover = 13.0 + broad * 16.0 + detail * detail * 7.0
    lift = np.rint(cover) - 11.0
    base_h = STONE_BASE + np.clip(lift, 4.0, 32.0)
    climate = fbm_grid(x * 0.006, z * 0.006, 2, int(seed) + 90)

    if biomes is None:
        from server.biomes import BIOMES

        biomes = BIOMES

    beach_c, ocean_w, floor_base, floor_detail, ocean_curve, inland = _get_transition_params(biomes)

    t_oc = np.clip((beach_c - climate) / ocean_w, 0.0, 1.0)
    s_oc = apply_curve_grid(t_oc, ocean_curve)
    floor = floor_base + floor_detail * detail

    if not inland:
        offset = 0.0
        scale = 1.0
        cliff_w = 0.0
        cliff_s = 0.0
    else:
        offset = np.full_like(climate, float(inland[0].height_offset))
        scale = np.full_like(climate, float(inland[0].height_scale))
        cliff_w = np.zeros_like(climate)
        cliff_s = np.zeros_like(climate)
        for k in range(1, len(inland)):
            prev_b = inland[k - 1]
            cur_b = inland[k]
            c_prev = prev_b.ceiling
            b_width = cur_b.blend
            w = np.clip((climate - c_prev) / b_width, 0.0, 1.0)
            w = apply_curve_grid(w, cur_b.curve)
            offset = offset + (cur_b.height_offset - prev_b.height_offset) * w
            scale = scale + (cur_b.height_scale - prev_b.height_scale) * w
            if cur_b.cliff:
                cliff_w = np.maximum(cliff_w, w)
                if cur_b.cliff_step > 0:
                    cliff_s = np.maximum(cliff_s, w * cur_b.cliff_step)

    terrace = np.rint((detail - 0.5) * 4.0) * 2.0
    h_inland = base_h + offset + (lift - 8.0) * (scale - 1.0) + terrace * cliff_w + cliff_s
    final_h = np.rint(h_inland + (floor - h_inland) * s_oc)
    return np.clip(final_h, 50, 122).astype(np.int16)


def hash3(x, y, z, seed=0):
    n = ((x * 374761393) + (y * 668265263) + (z * 1274126177) + (seed * 1442695041)) & MASK
    n = ((n ^ (n >> 13)) * 1274126177) & MASK
    return (n ^ (n >> 16)) / 4294967295


def value_noise3(x, y, z, seed=0):
    x0 = math.floor(x)
    y0 = math.floor(y)
    z0 = math.floor(z)
    tx = x - x0
    ty = y - y0
    tz = z - z0
    sx = tx * tx * (3 - 2 * tx)
    sy = ty * ty * (3 - 2 * ty)
    sz = tz * tz * (3 - 2 * tz)
    n000 = hash3(x0, y0, z0, seed)
    n100 = hash3(x0 + 1, y0, z0, seed)
    n010 = hash3(x0, y0 + 1, z0, seed)
    n110 = hash3(x0 + 1, y0 + 1, z0, seed)
    n001 = hash3(x0, y0, z0 + 1, seed)
    n101 = hash3(x0 + 1, y0, z0 + 1, seed)
    n011 = hash3(x0, y0 + 1, z0 + 1, seed)
    n111 = hash3(x0 + 1, y0 + 1, z0 + 1, seed)
    nx00 = n000 + (n100 - n000) * sx
    nx10 = n010 + (n110 - n010) * sx
    nx01 = n001 + (n101 - n001) * sx
    nx11 = n011 + (n111 - n011) * sx
    nxy0 = nx00 + (nx10 - nx00) * sy
    nxy1 = nx01 + (nx11 - nx01) * sy
    return nxy0 + (nxy1 - nxy0) * sz


def fbm3(x, y, z, octaves, seed=0):
    value = 0.0
    amplitude = 0.5
    frequency = 1.0
    total = 0.0
    for _ in range(octaves):
        value += amplitude * value_noise3(x * frequency, y * frequency, z * frequency, seed)
        total += amplitude
        amplitude *= 0.5
        frequency *= 2
    return value / total


def hash3_grid(x, y, z, seed):
    n = (x * np.int64(374761393) + y * np.int64(668265263) + z * np.int64(1274126177) + np.int64(seed) * np.int64(1442695041)) & np.int64(MASK)
    n = ((n ^ (n >> 13)) * np.int64(1274126177)) & np.int64(MASK)
    return (n ^ (n >> 16)) * (1.0 / 4294967295.0)


def value_noise3_grid(x, y, z, seed):
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    tx = x - x0
    ty = y - y0
    tz = z - z0
    sx = tx * tx * (3.0 - 2.0 * tx)
    sy = ty * ty * (3.0 - 2.0 * ty)
    sz = tz * tz * (3.0 - 2.0 * tz)

    def corner(dx, dy, dz):
        return hash3_grid(x0 + dx, y0 + dy, z0 + dz, seed)

    n000 = corner(0, 0, 0)
    n100 = corner(1, 0, 0)
    n010 = corner(0, 1, 0)
    n110 = corner(1, 1, 0)
    n001 = corner(0, 0, 1)
    n101 = corner(1, 0, 1)
    n011 = corner(0, 1, 1)
    n111 = corner(1, 1, 1)
    nx00 = n000 + (n100 - n000) * sx
    nx10 = n010 + (n110 - n010) * sx
    nx01 = n001 + (n101 - n001) * sx
    nx11 = n011 + (n111 - n011) * sx
    nxy0 = nx00 + (nx10 - nx00) * sy
    nxy1 = nx01 + (nx11 - nx01) * sy
    return nxy0 + (nxy1 - nxy0) * sz


def fbm3_grid(x, y, z, octaves, seed):
    value = np.zeros(np.broadcast(x, y, z).shape, dtype=np.float64)
    amplitude = 0.5
    frequency = 1.0
    total = 0.0
    for _ in range(octaves):
        value += amplitude * value_noise3_grid(x * frequency, y * frequency, z * frequency, seed)
        total += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    return value / total


@functools.lru_cache(maxsize=262144)
def _cached_terrain_height(x, z, seed, biomes_tuple):
    broad = fbm(x * 0.014, z * 0.014, 4, seed)
    detail = fbm(x * 0.05 + 12.3, z * 0.05 - 4.7, 3, seed)
    cover = 13.0 + broad * 16.0 + detail * detail * 7.0
    lift = round(cover) - 11.0
    base_h = STONE_BASE + max(4.0, min(32.0, lift))
    climate = fbm(x * 0.006, z * 0.006, 2, seed + 90)

    beach_c, ocean_w, floor_base, floor_detail, ocean_curve, inland = _get_transition_params(biomes_tuple)

    t_oc = min(1.0, max(0.0, (beach_c - climate) / ocean_w))
    s_oc = apply_curve_scalar(t_oc, ocean_curve)
    floor = floor_base + floor_detail * detail

    if not inland:
        offset = 0.0
        scale = 1.0
        cliff_w = 0.0
        cliff_s = 0.0
    else:
        offset = float(inland[0].height_offset)
        scale = float(inland[0].height_scale)
        cliff_w = 0.0
        cliff_s = 0.0
        for k in range(1, len(inland)):
            prev_b = inland[k - 1]
            cur_b = inland[k]
            c_prev = prev_b.ceiling
            b_width = cur_b.blend
            w = min(1.0, max(0.0, (climate - c_prev) / b_width))
            w = apply_curve_scalar(w, cur_b.curve)
            offset += (cur_b.height_offset - prev_b.height_offset) * w
            scale += (cur_b.height_scale - prev_b.height_scale) * w
            if cur_b.cliff:
                cliff_w = max(cliff_w, w)
                if cur_b.cliff_step > 0:
                    cliff_s = max(cliff_s, w * cur_b.cliff_step)

    terrace = round((detail - 0.5) * 4.0) * 2.0
    h_inland = base_h + offset + (lift - 8.0) * (scale - 1.0) + terrace * cliff_w + cliff_s
    final_h = round(h_inland + (floor - h_inland) * s_oc)
    return int(max(50, min(122, final_h)))


def terrain_height(x, z, seed=0, biomes=None):
    if biomes is None:
        from server.biomes import BIOMES

        biomes = BIOMES
    return _cached_terrain_height(int(x), int(z), int(seed), tuple(biomes))
