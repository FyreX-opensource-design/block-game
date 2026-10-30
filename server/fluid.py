"""Fluid simulation system matching Minecraft fluid physics."""

import collections
import json
import math
from pathlib import Path

from server.blocks import AIR, ASSETS, BLOCKS, BRACKISH_WATER, CHUNK_SIZE, SALT_WATER, WATER, WORLD_HEIGHT
from server.recipes import find_fluid_meet, find_fluid_meet_recipe, find_fluid_transform

WATER_SOURCE = "water"
WATER_FALLING = "water_falling"
SALT_WATER_SOURCE = "salt_water"
SALT_WATER_FALLING = "salt_water_falling"
BRACKISH_WATER_SOURCE = "brackish_water"
BRACKISH_WATER_FALLING = "brackish_water_falling"

CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DROPOFF_MAX_DIST = 4


class FluidDefinition:
    def __init__(
        self,
        fluid_id,
        block=None,
        flow_distance=7,
        speed=1.0,
        tick_interval=None,
        dropoff_distance=4,
        infinite_source=True,
        push_strength=1.0,
    ):
        self.id = str(fluid_id)
        self.block = str(block or fluid_id)
        self.flow_distance = max(0, min(7, int(flow_distance)))
        self.speed = max(0.01, float(speed))
        if tick_interval is not None:
            self.tick_interval = max(0.01, float(tick_interval))
        else:
            self.tick_interval = 0.15 / self.speed
        self.dropoff_distance = max(1, int(dropoff_distance))
        self.infinite_source = bool(infinite_source)
        self.push_strength = float(push_strength)


def fluid_from_mapping(entry, source=""):
    if not isinstance(entry, dict):
        raise ValueError(f"{source}: fluid definition must be an object")
    fluid_id = str(entry.get("id", Path(source).stem if source else ""))
    if not fluid_id:
        raise ValueError(f"{source}: fluid definition needs an id")
    block = str(entry.get("block", fluid_id))
    flow_distance = int(entry.get("flow_distance", entry.get("max_distance", 7)))
    speed = float(entry.get("speed", entry.get("flow_speed", 1.0)))
    tick_interval = entry.get("tick_interval")
    dropoff_distance = int(entry.get("dropoff_distance", DROPOFF_MAX_DIST))
    infinite_source = bool(entry.get("infinite_source", True))
    push_strength = float(entry.get("push_strength", 1.0))
    return FluidDefinition(
        fluid_id=fluid_id,
        block=block,
        flow_distance=flow_distance,
        speed=speed,
        tick_interval=tick_interval,
        dropoff_distance=dropoff_distance,
        infinite_source=infinite_source,
        push_strength=push_strength,
    )


def load_fluids(folder=None):
    root = Path(folder) if folder is not None else (Path(ASSETS) / "fluids")
    fluids = {}
    if root.is_dir():
        for path in sorted(root.glob("*.json")):
            fluid_def = fluid_from_mapping(json.loads(path.read_text()), path.name)
            fluids[fluid_def.id] = fluid_def
    for default_id in (WATER, SALT_WATER, BRACKISH_WATER):
        if default_id not in fluids:
            fluids[default_id] = FluidDefinition(default_id, block=default_id)
    return fluids


FLUIDS = load_fluids()


def _build_fluid_tables():
    levels = {}
    level_blocks = {}
    falling_blocks = {}
    sources = set()

    for fluid_id, fluid_def in FLUIDS.items():
        base = fluid_def.block
        sources.add(base)
        levels[base] = 8
        fall_id = f"{base}_falling"
        levels[fall_id] = 8
        falling_blocks[fluid_id] = fall_id
        l_map = {8: fall_id}
        for lvl in range(7, 0, -1):
            b_id = f"{base}_{lvl}"
            levels[b_id] = lvl
            l_map[lvl] = b_id
        level_blocks[fluid_id] = l_map

    for block_id, b_obj in BLOCKS.items():
        props = getattr(b_obj, "properties", None) or {}
        if props.get("waterlogged"):
            continue
        f_type = props.get("fluid")
        if isinstance(f_type, str) and f_type:
            lvl = int(props.get("level", 8))
            levels[block_id] = lvl
            if f_type not in FLUIDS:
                FLUIDS[f_type] = FluidDefinition(f_type, block=getattr(b_obj, "base_id", None) or block_id)
            if lvl == 8 and not props.get("falling", False):
                sources.add(block_id)
            if f_type not in level_blocks:
                level_blocks[f_type] = {}
            if props.get("falling", False):
                falling_blocks[f_type] = block_id
                level_blocks[f_type][8] = block_id
            elif 1 <= lvl <= 7:
                level_blocks[f_type][lvl] = block_id

    return levels, level_blocks, falling_blocks, sources


FLUID_LEVELS, FLUID_LEVEL_BLOCKS, FLUID_FALLING_BLOCKS, FLUID_SOURCES = _build_fluid_tables()

WATER_LEVEL_BLOCKS = FLUID_LEVEL_BLOCKS.get("water", {
    8: "water_falling",
    7: "water_7",
    6: "water_6",
    5: "water_5",
    4: "water_4",
    3: "water_3",
    2: "water_2",
    1: "water_1",
})

EMPTY_BUCKETS = {
    "wood_bucket": "wood_water_bucket",
    "clay_bucket": "clay_water_bucket",
    "iron_bucket": "water_bucket",
    "bucket": "water_bucket",
}

BUCKET_FLUID_MAP = {
    ("wood_bucket", "water"): "wood_water_bucket",
    ("clay_bucket", "water"): "clay_water_bucket",
    ("iron_bucket", "water"): "water_bucket",
    ("bucket", "water"): "water_bucket",
    ("wood_bucket", "salt_water"): "wood_salt_water_bucket",
    ("clay_bucket", "salt_water"): "clay_salt_water_bucket",
    ("iron_bucket", "salt_water"): "salt_water_bucket",
    ("bucket", "salt_water"): "salt_water_bucket",
    ("wood_bucket", "brackish_water"): "wood_brackish_water_bucket",
    ("clay_bucket", "brackish_water"): "clay_brackish_water_bucket",
    ("iron_bucket", "brackish_water"): "brackish_water_bucket",
    ("bucket", "brackish_water"): "brackish_water_bucket",
}

FILLED_BUCKETS = {
    "wood_water_bucket": ("water", "wood_bucket"),
    "water_wood_bucket": ("water", "wood_bucket"),
    "clay_water_bucket": ("water", "clay_bucket"),
    "water_clay_bucket": ("water", "clay_bucket"),
    "iron_water_bucket": ("water", "iron_bucket"),
    "water_bucket": ("water", "iron_bucket"),
    "wood_salt_water_bucket": ("salt_water", "wood_bucket"),
    "salt_water_wood_bucket": ("salt_water", "wood_bucket"),
    "clay_salt_water_bucket": ("salt_water", "clay_bucket"),
    "salt_water_clay_bucket": ("salt_water", "clay_bucket"),
    "iron_salt_water_bucket": ("salt_water", "iron_bucket"),
    "salt_water_bucket": ("salt_water", "iron_bucket"),
    "wood_brackish_water_bucket": ("brackish_water", "wood_bucket"),
    "brackish_water_wood_bucket": ("brackish_water", "wood_bucket"),
    "clay_brackish_water_bucket": ("brackish_water", "clay_bucket"),
    "brackish_water_clay_bucket": ("brackish_water", "clay_bucket"),
    "iron_brackish_water_bucket": ("brackish_water", "iron_bucket"),
    "brackish_water_bucket": ("brackish_water", "iron_bucket"),
}


def is_empty_bucket(item_id):
    return item_id in EMPTY_BUCKETS


def is_filled_bucket(item_id):
    return item_id in FILLED_BUCKETS


def get_filled_bucket(bucket_id, fluid_id="water"):
    norm_fluid = get_fluid_type(fluid_id) or str(fluid_id or "water")
    return BUCKET_FLUID_MAP.get((bucket_id, norm_fluid))


def get_bucket_fluid(bucket_id):
    if bucket_id in FILLED_BUCKETS:
        return FILLED_BUCKETS[bucket_id][0]
    return None


def get_bucket_empty(bucket_id):
    if bucket_id in FILLED_BUCKETS:
        return FILLED_BUCKETS[bucket_id][1]
    return None


def get_fluid_type(block_id):
    """Return the base fluid ID ('water', 'salt_water', 'brackish_water', etc.) for a block, or None."""
    if not block_id:
        return None
    if block_id in FLUIDS:
        return block_id
    b_obj = BLOCKS.get(block_id)
    if b_obj is not None:
        props = getattr(b_obj, "properties", None) or {}
        if props.get("waterlogged"):
            return str(props.get("fluid", "water"))
        f_prop = props.get("fluid")
        if isinstance(f_prop, str) and f_prop:
            return f_prop
    if block_id in FLUID_LEVELS:
        if block_id.endswith("_falling"):
            return block_id[: -len("_falling")]
        for lvl in range(1, 8):
            suffix = f"_{lvl}"
            if block_id.endswith(suffix):
                return block_id[: -len(suffix)]
        return block_id
    from server.blocks import is_waterlogged_slab

    if is_waterlogged_slab(block_id):
        return "water"
    return None


def get_fluid_def(block_or_fluid_id):
    fluid_type = get_fluid_type(block_or_fluid_id) or str(block_or_fluid_id or "")
    if fluid_type in FLUIDS:
        return FLUIDS[fluid_type]
    return FluidDefinition(fluid_type or "water")


def is_fluid(block_id):
    if not block_id:
        return False
    if block_id in FLUID_LEVELS:
        return True
    b_obj = BLOCKS.get(block_id)
    if b_obj is not None and (getattr(b_obj, "properties", None) or {}).get("fluid"):
        return True
    from server.blocks import is_waterlogged_slab

    return is_waterlogged_slab(block_id)


def is_water(block_id):
    return is_fluid(block_id)


def is_fluid_source(block_id):
    if not block_id:
        return False
    if block_id in FLUID_SOURCES:
        return True
    b_obj = BLOCKS.get(block_id)
    if b_obj is not None:
        props = getattr(b_obj, "properties", None) or {}
        if props.get("fluid") and not props.get("waterlogged") and not props.get("falling", False):
            if int(props.get("level", 8)) == 8:
                return True
    from server.blocks import get_slab_info

    s_info = get_slab_info(block_id)
    if s_info is not None and s_info["waterlogged"]:
        return s_info.get("level", 8) == 8
    return False


def is_water_source(block_id):
    return is_fluid_source(block_id)


def is_falling_fluid(block_id):
    if not block_id:
        return False
    if block_id.endswith("_falling"):
        return True
    b_obj = BLOCKS.get(block_id)
    if b_obj is not None:
        return bool((getattr(b_obj, "properties", None) or {}).get("falling", False))
    return False


def get_fluid_level(block_id):
    if not block_id:
        return 0
    if block_id in FLUID_LEVELS:
        return FLUID_LEVELS[block_id]
    b_obj = BLOCKS.get(block_id)
    if b_obj is not None:
        props = getattr(b_obj, "properties", None) or {}
        if props.get("fluid") and not props.get("waterlogged"):
            return int(props.get("level", 8))
    from server.blocks import get_slab_info

    s_info = get_slab_info(block_id)
    if s_info is not None and s_info["waterlogged"]:
        return s_info.get("level", 8)
    return 0


def _fluid_falling_block(fluid_type):
    if fluid_type in FLUID_FALLING_BLOCKS:
        return FLUID_FALLING_BLOCKS[fluid_type]
    cand = f"{fluid_type}_falling"
    return cand if cand in BLOCKS else WATER_FALLING


def _fluid_level_block(fluid_type, level):
    l_map = FLUID_LEVEL_BLOCKS.get(fluid_type)
    if l_map and level in l_map:
        return l_map[level]
    cand = f"{fluid_type}_{level}"
    if cand in BLOCKS:
        return cand
    return WATER_LEVEL_BLOCKS.get(level, f"water_{level}")


def _step_down_level(nbr_block, nbr_level, fluid_def):
    """Calculate the level a neighbor provides when spreading horizontally by 1 block."""
    max_flow = fluid_def.flow_distance
    if max_flow <= 0:
        return 0
    if is_fluid_source(nbr_block) or is_falling_fluid(nbr_block) or nbr_level >= 8:
        return min(7, max_flow)
    return min(nbr_level - 1, max_flow)


def can_flow_into(world, x, y, z, from_level=None, from_y=None, fluid_type=None):
    """Check if fluid can flow into (x, y, z).
    Must be in world vertical bounds and in a currently loaded chunk.
    Non-solid blocks (air, non-source fluid of the same type, or fluid_breakable) can be flowed into.
    Dry bottom slabs can be flowed into if from above or from level > 4 (over half height).
    Dry top slabs can be flowed into underneath.
    """
    if y < 0 or y >= WORLD_HEIGHT:
        return False
    if hasattr(world, "chunk_key") and hasattr(world, "chunks"):
        try:
            ck = world.chunk_key(x, y, z)
            if isinstance(ck, (tuple, list)) and len(ck) == 3:
                if ck not in world.chunks:
                    return False
        except Exception:
            pass
    block = world.get_block(x, y, z)
    if block == AIR:
        return True
    if is_fluid(block) and not is_fluid_source(block):
        if fluid_type is not None:
            cur_type = get_fluid_type(block)
            if cur_type and cur_type != fluid_type:
                return False
        return True
    b_obj = BLOCKS.get(block)
    if b_obj is not None and getattr(b_obj, "fluid_breakable", False):
        return True

    from server.blocks import get_slab_info

    s_info = get_slab_info(block)
    if s_info is not None:
        if s_info["double"] or is_fluid_source(block):
            return False
        if s_info["half"] == "bottom":
            if from_y is not None and from_y == y + 1:
                return True
            if from_level is not None:
                return from_level > 4
            above = world.get_block(x, y + 1, z) if (y + 1 < WORLD_HEIGHT) else AIR
            if is_fluid(above):
                return True
            for dx, dz in CARDINALS:
                nbr = world.get_block(x + dx, y, z + dz)
                if is_fluid(nbr) and get_fluid_level(nbr) > 4:
                    return True
            return False
        elif s_info["half"] == "top":
            if from_y is not None and from_y == y + 1:
                return False
            return True

    return False


def is_blocking(world, x, y, z, from_level=None, from_y=None):
    """Check if (x, y, z) blocks fluid flow (solid or unloaded)."""
    if y < 0 or y >= WORLD_HEIGHT:
        return True
    cx, cy, cz = world.chunk_key(x, y, z)
    if (cx, cy, cz) not in world.chunks:
        return True
    block = world.get_block(x, y, z)
    if block == AIR or is_fluid(block):
        return False
    b_obj = BLOCKS.get(block)
    if b_obj is not None and getattr(b_obj, "fluid_breakable", False):
        return False

    from server.blocks import get_slab_info

    s_info = get_slab_info(block)
    if s_info is not None:
        if s_info["double"]:
            return True
        if s_info["half"] == "bottom":
            if from_level is not None:
                return from_level <= 4
            if from_y is not None and from_y == y + 1:
                return False
            return True
        elif s_info["half"] == "top":
            if from_y is not None and from_y == y + 1:
                return True
            return False

    return world.is_solid(x, y, z)


def get_fluid_flow(world, x, y, z):
    """Calculate the 3D flow vector (fx, fy, fz) of fluid at integer block (x, y, z).
    Returns (0.0, 0.0, 0.0) if no fluid at (x, y, z).
    Horizontal components point in direction of downhill / spread gradient.
    Vertical component is negative if falling fluid or open space directly below.
    """
    block = world.get_block(x, y, z)
    if not is_fluid(block):
        return (0.0, 0.0, 0.0)

    cur_level = get_fluid_level(block)
    is_falling = is_falling_fluid(block)

    fy = 0.0
    if is_falling or (y + 1 < WORLD_HEIGHT and is_falling_fluid(world.get_block(x, y + 1, z))):
        fy = -1.0
    elif y > 0 and can_flow_into(world, x, y - 1, z):
        fy = -0.5

    vx = 0.0
    vz = 0.0

    for dx, dz in CARDINALS:
        nx, nz = x + dx, z + dz
        nbr = world.get_block(nx, y, nz)

        if is_fluid(nbr):
            nbr_level = get_fluid_level(nbr)
            if is_falling_fluid(nbr):
                nbr_level = 0
            elif y > 0 and (can_flow_into(world, nx, y - 1, nz) or is_falling_fluid(world.get_block(nx, y - 1, nz))):
                nbr_level = 0
            diff = cur_level - nbr_level
            vx += dx * diff
            vz += dz * diff
        elif can_flow_into(world, nx, y, nz):
            if y > 0 and (can_flow_into(world, nx, y - 1, nz) or is_fluid(world.get_block(nx, y - 1, nz))):
                diff = cur_level + 1
            else:
                diff = cur_level
            vx += dx * diff
            vz += dz * diff

    h_len = math.hypot(vx, vz)
    if h_len > 1e-4:
        fx = vx / h_len
        fz = vz / h_len
    else:
        fx = 0.0
        fz = 0.0

    return (fx, fy, fz)


def get_entity_fluid_flow(world, min_x, min_y, min_z, max_x, max_y, max_z):
    """Compute the averaged fluid flow vector across an entity bounding box."""
    if world is None:
        return (0.0, 0.0, 0.0)

    total_fx = 0.0
    total_fy = 0.0
    total_fz = 0.0
    count = 0

    for y in range(min_y, max_y + 1):
        for z in range(min_z, max_z + 1):
            for x in range(min_x, max_x + 1):
                block = world.get_block(x, y, z)
                if is_fluid(block):
                    fx, fy, fz = get_fluid_flow(world, x, y, z)
                    total_fx += fx
                    total_fy += fy
                    total_fz += fz
                    count += 1

    if count == 0:
        return (0.0, 0.0, 0.0)

    h_len = math.hypot(total_fx, total_fz)
    if h_len > 1e-4:
        out_fx = total_fx / h_len
        out_fz = total_fz / h_len
    else:
        out_fx = 0.0
        out_fz = 0.0

    out_fy = max(-1.0, min(1.0, total_fy / count))
    return (out_fx, out_fy, out_fz)


def find_dropoff_directions(world, x, y, z, max_dist=None):
    """Minecraft slope / dropoff pathfinding within dropoff_distance (default 4) blocks.
    Finds which cardinal directions lead to a dropoff (a place where fluid can
    fall down) in the fewest number of steps. If no dropoff exists within max_dist,
    returns all valid passable horizontal directions.
    """
    src_block = world.get_block(x, y, z)
    fluid_type = get_fluid_type(src_block)
    if max_dist is None:
        fluid_def = get_fluid_def(fluid_type) if fluid_type else None
        max_dist = fluid_def.dropoff_distance if fluid_def else DROPOFF_MAX_DIST

    valid_dirs = []
    for dx, dz in CARDINALS:
        nx = x + dx
        nz = z + dz
        if can_flow_into(world, nx, y, nz, fluid_type=fluid_type):
            valid_dirs.append((dx, dz))

    if not valid_dirs:
        return ()

    best_dist = max_dist + 1
    dir_distances = {}

    for start_dx, start_dz in valid_dirs:
        sx = x + start_dx
        sz = z + start_dz

        # If immediate dropoff directly under start block
        if y > 0 and (
            can_flow_into(world, sx, y - 1, sz, fluid_type=fluid_type)
            or is_fluid(world.get_block(sx, y - 1, sz))
        ):
            dir_distances[(start_dx, start_dz)] = 1
            if 1 < best_dist:
                best_dist = 1
            continue

        queue = collections.deque([(sx, sz, 1)])
        visited = {(x, z), (sx, sz)}
        found_dist = max_dist + 1

        while queue:
            cx, cz, dist = queue.popleft()
            if dist >= max_dist or dist >= best_dist:
                continue
            for ddx, ddz in CARDINALS:
                nnx = cx + ddx
                nnz = cz + ddz
                if (nnx, nnz) in visited or is_blocking(world, nnx, y, nnz):
                    continue
                visited.add((nnx, nnz))
                if y > 0 and (
                    can_flow_into(world, nnx, y - 1, nnz, fluid_type=fluid_type)
                    or is_fluid(world.get_block(nnx, y - 1, nnz))
                ):
                    found_dist = dist + 1
                    queue.clear()
                    break
                queue.append((nnx, nnz, dist + 1))

        dir_distances[(start_dx, start_dz)] = found_dist
        if found_dist < best_dist:
            best_dist = found_dist

    if best_dist <= max_dist:
        return tuple(d for d, dist in dir_distances.items() if dist == best_dist)
    return tuple(valid_dirs)


def compute_new_fluid_state(world, x, y, z):
    """Compute the new fluid block at (x, y, z) based on neighbors."""
    cx, cy, cz = world.chunk_key(x, y, z)
    if (cx, cy, cz) not in world.chunks:
        return world.get_block(x, y, z)

    cur = world.get_block(x, y, z)
    from server.blocks import get_slab_info

    s_info = get_slab_info(cur)
    if s_info is not None:
        if s_info["double"] or is_fluid_source(cur):
            return cur
        from server.blocks import get_waterlogged_slab

        max_level = 0
        if s_info["half"] == "bottom":
            above = world.get_block(x, y + 1, z) if (y + 1 < WORLD_HEIGHT) else AIR
            if is_fluid(above):
                max_level = 7
            for dx, dz in CARDINALS:
                nx = x + dx
                nz = z + dz
                nbr = world.get_block(nx, y, nz)
                if not is_fluid(nbr):
                    continue
                nbr_type = get_fluid_type(nbr) or "water"
                nbr_def = get_fluid_def(nbr_type)
                nbr_level = get_fluid_level(nbr)
                pot = _step_down_level(nbr, nbr_level, nbr_def)
                if pot <= 3:
                    continue
                if y > 0 and (can_flow_into(world, nx, y - 1, nz, fluid_type=nbr_type) or is_fluid(world.get_block(nx, y - 1, nz))):
                    continue
                dirs = find_dropoff_directions(world, nx, y, nz, max_dist=nbr_def.dropoff_distance)
                if (-dx, -dz) in dirs:
                    if pot > max_level:
                        max_level = pot
        elif s_info["half"] == "top":
            for dx, dz in CARDINALS:
                nx = x + dx
                nz = z + dz
                nbr = world.get_block(nx, y, nz)
                if not is_fluid(nbr):
                    continue
                nbr_type = get_fluid_type(nbr) or "water"
                nbr_def = get_fluid_def(nbr_type)
                nbr_level = get_fluid_level(nbr)
                pot = _step_down_level(nbr, nbr_level, nbr_def)
                if pot <= 0:
                    continue
                if y > 0 and (can_flow_into(world, nx, y - 1, nz, fluid_type=nbr_type) or is_fluid(world.get_block(nx, y - 1, nz))):
                    continue
                dirs = find_dropoff_directions(world, nx, y, nz, max_dist=nbr_def.dropoff_distance)
                if (-dx, -dz) in dirs:
                    if pot > max_level:
                        max_level = pot

        if max_level <= 0:
            if s_info["waterlogged"]:
                return f"{s_info['base_id']}_top" if s_info["half"] == "top" else s_info["base_id"]
            return cur
        return get_waterlogged_slab(s_info["base_id"], half=s_info["half"], level=max_level)

    if is_fluid_source(cur):
        return cur
    if not can_flow_into(world, x, y, z):
        return cur

    below = world.get_block(x, y - 1, z)
    below_solid_or_source = (y > 0) and (world.is_solid(x, y - 1, z) or is_fluid_source(below))

    # Infinite fluid source: 2 or more adjacent sources of the same fluid with infinite_source=True
    if below_solid_or_source:
        source_counts = {}
        for dx, dz in CARDINALS:
            nbr_blk = world.get_block(x + dx, y, z + dz)
            if is_fluid_source(nbr_blk):
                f_type = get_fluid_type(nbr_blk) or "water"
                f_def = get_fluid_def(f_type)
                if f_def.infinite_source:
                    source_counts[f_type] = source_counts.get(f_type, 0) + 1
                    if source_counts[f_type] >= 2:
                        return f_def.block

    # Falling fluid from above
    above = world.get_block(x, y + 1, z) if (y + 1 < WORLD_HEIGHT) else AIR
    if is_fluid(above):
        above_type = get_fluid_type(above) or "water"
        if can_flow_into(world, x, y, z, from_y=y + 1, fluid_type=above_type):
            return _fluid_falling_block(above_type)

    # Horizontal spread
    max_level = 0
    chosen_fluid = None
    cur_type = get_fluid_type(cur) if is_fluid(cur) else None

    for dx, dz in CARDINALS:
        nx = x + dx
        nz = z + dz
        nbr = world.get_block(nx, y, nz)
        if not is_fluid(nbr):
            continue
        nbr_type = get_fluid_type(nbr) or "water"
        nbr_def = get_fluid_def(nbr_type)
        nbr_level = get_fluid_level(nbr)
        pot = _step_down_level(nbr, nbr_level, nbr_def)
        if pot <= 0:
            continue
        if not can_flow_into(world, x, y, z, from_level=nbr_level, from_y=y, fluid_type=nbr_type):
            continue
        # If neighbor can fall down directly below it or is resting on fluid, it doesn't spread horizontally here
        if y > 0 and (can_flow_into(world, nx, y - 1, nz, fluid_type=nbr_type) or is_fluid(world.get_block(nx, y - 1, nz))):
            continue
        dirs = find_dropoff_directions(world, nx, y, nz, max_dist=nbr_def.dropoff_distance)
        if (-dx, -dz) in dirs:
            if pot > max_level or (pot == max_level and nbr_type == cur_type):
                max_level = pot
                chosen_fluid = nbr_type

    if max_level <= 0 or not chosen_fluid:
        return AIR if is_fluid(cur) else cur
    return _fluid_level_block(chosen_fluid, max_level)


class FluidSystem:
    def __init__(self, world):
        self.world = world
        self.queue = set()
        self.tick_timer = 0.0
        self.tick_interval = 0.15
        self.fluid_timers = {}

    def schedule(self, x, y, z):
        """Schedule a block and its 6 neighbors for fluid simulation if chunk is loaded."""
        if y < 0 or y >= WORLD_HEIGHT:
            return
        cx, cy, cz = self.world.chunk_key(x, y, z)
        if (cx, cy, cz) not in self.world.chunks:
            return
        self.queue.add((x, y, z))
        for dx, dz in CARDINALS:
            self.queue.add((x + dx, y, z + dz))
        if y > 0:
            self.queue.add((x, y - 1, z))
        if y + 1 < WORLD_HEIGHT:
            self.queue.add((x, y + 1, z))

    def _dominant_fluid_for_cell(self, x, y, z):
        blk = self.world.get_block(x, y, z)
        f_type = get_fluid_type(blk)
        if f_type:
            return f_type
        if y + 1 < WORLD_HEIGHT:
            above_type = get_fluid_type(self.world.get_block(x, y + 1, z))
            if above_type:
                return above_type
        for dx, dz in CARDINALS:
            nbr_type = get_fluid_type(self.world.get_block(x + dx, y, z + dz))
            if nbr_type:
                return nbr_type
        return "water"

    def update(self, dt):
        """Tick fluid simulation in player loaded chunks, respecting each fluid's speed."""
        if not self.queue:
            self.tick_timer = 0.0
            self.fluid_timers.clear()
            return []

        self.tick_timer += dt
        ready_fluids = set()
        for f_id, f_def in FLUIDS.items():
            t = self.fluid_timers.get(f_id, 0.0) + dt
            interval = f_def.tick_interval if self.tick_interval == 0.15 else (self.tick_interval / f_def.speed)
            if t >= interval:
                ready_fluids.add(f_id)
                self.fluid_timers[f_id] = 0.0
            else:
                self.fluid_timers[f_id] = t

        if self.tick_timer >= self.tick_interval:
            self.tick_timer = 0.0
            for f_id, f_def in FLUIDS.items():
                if f_def.speed >= 1.0:
                    ready_fluids.add(f_id)
                    self.fluid_timers[f_id] = 0.0

        if not ready_fluids:
            return []

        pending = list(self.queue)
        self.queue.clear()
        changed_blocks = []
        limit = 128
        count = 0

        for x, y, z in pending:
            if count >= limit:
                self.queue.add((x, y, z))
                continue
            cx, cy, cz = self.world.chunk_key(x, y, z)
            if (cx, cy, cz) not in self.world.chunks:
                continue

            dom_fluid = self._dominant_fluid_for_cell(x, y, z)
            if dom_fluid not in ready_fluids:
                self.queue.add((x, y, z))
                continue

            count += 1
            cur = self.world.get_block(x, y, z)
            new_block = compute_new_fluid_state(self.world, x, y, z)
            if new_block != cur:
                if self.world.set_block(x, y, z, new_block):
                    changed_blocks.append((x, y, z, new_block))
                    self.schedule(x, y, z)
            elif is_fluid(cur):
                from server.blocks import get_slab_info

                cur_type = get_fluid_type(cur) or "water"
                cur_def = get_fluid_def(cur_type)
                s_info = get_slab_info(cur)
                can_fall_below = True
                if s_info is not None and s_info["half"] == "bottom":
                    can_fall_below = False

                # If fluid can fall down, schedule below
                if can_fall_below and y > 0 and can_flow_into(self.world, x, y - 1, z, from_y=y, fluid_type=cur_type):
                    fall_blk = _fluid_falling_block(cur_type)
                    if self.world.get_block(x, y - 1, z) != fall_blk:
                        self.queue.add((x, y - 1, z))
                elif y > 0 and is_fluid(self.world.get_block(x, y - 1, z)):
                    # Fluid resting on fluid cannot spread horizontally
                    pass
                else:
                    cur_level = get_fluid_level(cur)
                    target_level = _step_down_level(cur, cur_level, cur_def)
                    if target_level > 0:
                        flow_dirs = find_dropoff_directions(self.world, x, y, z, max_dist=cur_def.dropoff_distance)
                        for dx, dz in flow_dirs:
                            nx = x + dx
                            nz = z + dz
                            if can_flow_into(self.world, nx, y, nz, from_level=cur_level, from_y=y, fluid_type=cur_type):
                                expected = _fluid_level_block(cur_type, target_level)
                                if self.world.get_block(nx, y, nz) != expected:
                                    self.queue.add((nx, y, nz))

            # Fluid collision transformation and fluid_meet interactions
            current_now = self.world.get_block(x, y, z)
            active_fluid = current_now if is_fluid(current_now) else None
            if active_fluid:
                active_type = get_fluid_type(active_fluid)
                active_is_source = is_fluid_source(active_fluid)
                for n_dx, n_dy, n_dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    nx, ny, nz = x + n_dx, y + n_dy, z + n_dz
                    if not (0 <= ny < WORLD_HEIGHT):
                        continue
                    nbr = self.world.get_block(nx, ny, nz)
                    if is_fluid(nbr):
                        nbr_type = get_fluid_type(nbr)
                        if nbr_type and active_type and nbr_type != active_type:
                            meet_recipe = find_fluid_meet_recipe(active_type, nbr_type)
                            if meet_recipe is not None:
                                nbr_is_source = is_fluid_source(nbr)
                                # If a fluid (flowing or source) hits a source block of the other fluid,
                                # convert the source block into source_result.
                                # If both are flowing, only place meet_recipe.result if explicitly defined.
                                target_pos = None
                                old_blk = None
                                res_blk = None
                                if not active_is_source and nbr_is_source:
                                    target_pos = (nx, ny, nz)
                                    old_blk = nbr
                                    res_blk = meet_recipe.source_result
                                elif active_is_source and not nbr_is_source:
                                    target_pos = (x, y, z)
                                    old_blk = active_fluid
                                    res_blk = meet_recipe.source_result
                                elif active_is_source and nbr_is_source:
                                    target_pos = (x, y, z)
                                    old_blk = active_fluid
                                    res_blk = meet_recipe.source_result or meet_recipe.result
                                elif meet_recipe.result:
                                    # Two flowing fluids meet: only place result if recipe defines a flow result
                                    target_pos = (x, y, z)
                                    old_blk = active_fluid
                                    res_blk = meet_recipe.result
                                if target_pos is not None and res_blk and res_blk != old_blk:
                                    tx, ty, tz = target_pos
                                    if self.world.set_block(tx, ty, tz, res_blk):
                                        changed_blocks.append((tx, ty, tz, res_blk))
                                        self.schedule(tx, ty, tz)
                                    if target_pos == (x, y, z):
                                        active_fluid = res_blk
                                        active_type = get_fluid_type(active_fluid)
                                        active_is_source = is_fluid_source(active_fluid)
                        continue

                    transformed = find_fluid_transform(active_fluid, nbr)
                    if transformed and transformed != nbr:
                        if self.world.set_block(nx, ny, nz, transformed):
                            changed_blocks.append((nx, ny, nz, transformed))
                            self.schedule(nx, ny, nz)
            else:
                for n_dx, n_dy, n_dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    nx, ny, nz = x + n_dx, y + n_dy, z + n_dz
                    if 0 <= ny < WORLD_HEIGHT:
                        adj = self.world.get_block(nx, ny, nz)
                        if is_fluid(adj):
                            transformed = find_fluid_transform(adj, current_now)
                            if transformed and transformed != current_now:
                                if self.world.set_block(x, y, z, transformed):
                                    changed_blocks.append((x, y, z, transformed))
                                    self.schedule(x, y, z)
                                break

        return changed_blocks
