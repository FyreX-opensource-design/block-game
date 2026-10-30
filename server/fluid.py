"""Fluid simulation system matching Minecraft fluid physics."""

import collections
import math

from server.blocks import AIR, BLOCKS, CHUNK_SIZE, WORLD_HEIGHT
from server.recipes import find_fluid_transform

WATER_SOURCE = "water"
WATER_FALLING = "water_falling"

FLUID_LEVELS = {
    "water": 8,
    "water_falling": 8,
    "water_7": 7,
    "water_6": 6,
    "water_5": 5,
    "water_4": 4,
    "water_3": 3,
    "water_2": 2,
    "water_1": 1,
}

WATER_LEVEL_BLOCKS = {
    8: "water_falling",
    7: "water_7",
    6: "water_6",
    5: "water_5",
    4: "water_4",
    3: "water_3",
    2: "water_2",
    1: "water_1",
}

EMPTY_BUCKETS = {
    "wood_bucket": "wood_water_bucket",
    "clay_bucket": "clay_water_bucket",
    "iron_bucket": "water_bucket",
    "bucket": "water_bucket",
}

FILLED_BUCKETS = {
    "wood_water_bucket": ("water", "wood_bucket"),
    "water_wood_bucket": ("water", "wood_bucket"),
    "clay_water_bucket": ("water", "clay_bucket"),
    "water_clay_bucket": ("water", "clay_bucket"),
    "iron_water_bucket": ("water", "iron_bucket"),
    "water_bucket": ("water", "iron_bucket"),
}


def is_empty_bucket(item_id):
    return item_id in EMPTY_BUCKETS


def is_filled_bucket(item_id):
    return item_id in FILLED_BUCKETS


def get_filled_bucket(bucket_id, fluid_id="water"):
    if fluid_id == "water":
        return EMPTY_BUCKETS.get(bucket_id)
    return None


def get_bucket_fluid(bucket_id):
    if bucket_id in FILLED_BUCKETS:
        return FILLED_BUCKETS[bucket_id][0]
    return None


def get_bucket_empty(bucket_id):
    if bucket_id in FILLED_BUCKETS:
        return FILLED_BUCKETS[bucket_id][1]
    return None

CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DROPOFF_MAX_DIST = 4


def is_fluid(block_id):
    if not block_id:
        return False
    if block_id in FLUID_LEVELS:
        return True
    from server.blocks import is_waterlogged_slab

    return is_waterlogged_slab(block_id)


def is_water(block_id):
    if not block_id:
        return False
    if block_id in FLUID_LEVELS:
        return True
    from server.blocks import is_waterlogged_slab

    return is_waterlogged_slab(block_id)


def is_water_source(block_id):
    if block_id == WATER_SOURCE:
        return True
    from server.blocks import get_slab_info

    s_info = get_slab_info(block_id)
    if s_info is not None and s_info["waterlogged"]:
        return s_info.get("level", 8) == 8
    return False


def get_fluid_level(block_id):
    if not block_id:
        return 0
    if block_id in FLUID_LEVELS:
        return FLUID_LEVELS[block_id]
    from server.blocks import get_slab_info

    s_info = get_slab_info(block_id)
    if s_info is not None and s_info["waterlogged"]:
        return s_info.get("level", 8)
    return 0


def can_flow_into(world, x, y, z, from_level=None, from_y=None):
    """Check if water can flow into (x, y, z).
    Must be in world vertical bounds and in a currently loaded chunk.
    Non-solid blocks (air, non-source water, or fluid_breakable) can be flowed into.
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
    if is_water(block) and not is_water_source(block):
        return True
    b_obj = BLOCKS.get(block)
    if b_obj is not None and getattr(b_obj, "fluid_breakable", False):
        return True

    from server.blocks import get_slab_info

    s_info = get_slab_info(block)
    if s_info is not None:
        if s_info["double"] or is_water_source(block):
            return False
        if s_info["half"] == "bottom":
            if from_y is not None and from_y == y + 1:
                return True
            if from_level is not None:
                return from_level > 4
            above = world.get_block(x, y + 1, z) if (y + 1 < WORLD_HEIGHT) else AIR
            if is_water(above):
                return True
            for dx, dz in CARDINALS:
                nbr = world.get_block(x + dx, y, z + dz)
                if is_water(nbr) and get_fluid_level(nbr) > 4:
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
    if block == AIR or is_water(block):
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
    Vertical component is negative if falling water or open space directly below.
    """
    block = world.get_block(x, y, z)
    if not is_fluid(block):
        return (0.0, 0.0, 0.0)

    cur_level = get_fluid_level(block)
    is_falling = block == WATER_FALLING

    fy = 0.0
    if is_falling or (y + 1 < WORLD_HEIGHT and world.get_block(x, y + 1, z) == WATER_FALLING):
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
            if nbr == WATER_FALLING:
                nbr_level = 0
            elif y > 0 and (can_flow_into(world, nx, y - 1, nz) or world.get_block(nx, y - 1, nz) == WATER_FALLING):
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


def find_dropoff_directions(world, x, y, z):
    """Minecraft slope / dropoff pathfinding within 4 blocks.
    Finds which cardinal directions lead to a dropoff (a place where water can
    fall down) in the fewest number of steps. If no dropoff exists within 4 blocks,
    returns all valid passable horizontal directions.
    """
    valid_dirs = []
    for dx, dz in CARDINALS:
        nx = x + dx
        nz = z + dz
        if can_flow_into(world, nx, y, nz):
            valid_dirs.append((dx, dz))

    if not valid_dirs:
        return ()

    best_dist = DROPOFF_MAX_DIST + 1
    dir_distances = {}

    for start_dx, start_dz in valid_dirs:
        sx = x + start_dx
        sz = z + start_dz

        # If immediate dropoff directly under start block
        if y > 0 and (can_flow_into(world, sx, y - 1, sz) or is_water(world.get_block(sx, y - 1, sz))):
            dir_distances[(start_dx, start_dz)] = 1
            if 1 < best_dist:
                best_dist = 1
            continue

        queue = collections.deque([(sx, sz, 1)])
        visited = {(x, z), (sx, sz)}
        found_dist = DROPOFF_MAX_DIST + 1

        while queue:
            cx, cz, dist = queue.popleft()
            if dist >= DROPOFF_MAX_DIST or dist >= best_dist:
                continue
            for ddx, ddz in CARDINALS:
                nnx = cx + ddx
                nnz = cz + ddz
                if (nnx, nnz) in visited or is_blocking(world, nnx, y, nnz):
                    continue
                visited.add((nnx, nnz))
                if y > 0 and (can_flow_into(world, nnx, y - 1, nnz) or is_water(world.get_block(nnx, y - 1, nnz))):
                    found_dist = dist + 1
                    queue.clear()
                    break
                queue.append((nnx, nnz, dist + 1))

        dir_distances[(start_dx, start_dz)] = found_dist
        if found_dist < best_dist:
            best_dist = found_dist

    if best_dist <= DROPOFF_MAX_DIST:
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
        if s_info["double"] or is_water_source(cur):
            return cur
        from server.blocks import get_waterlogged_slab

        max_level = 0
        if s_info["half"] == "bottom":
            above = world.get_block(x, y + 1, z) if (y + 1 < WORLD_HEIGHT) else AIR
            if is_water(above):
                max_level = 7
            for dx, dz in CARDINALS:
                nx = x + dx
                nz = z + dz
                nbr = world.get_block(nx, y, nz)
                if not is_water(nbr):
                    continue
                nbr_level = get_fluid_level(nbr)
                if nbr_level <= 4:
                    continue
                if y > 0 and (can_flow_into(world, nx, y - 1, nz) or is_water(world.get_block(nx, y - 1, nz))):
                    continue
                dirs = find_dropoff_directions(world, nx, y, nz)
                if (-dx, -dz) in dirs:
                    pot = nbr_level - 1
                    if pot > max_level:
                        max_level = pot
        elif s_info["half"] == "top":
            for dx, dz in CARDINALS:
                nx = x + dx
                nz = z + dz
                nbr = world.get_block(nx, y, nz)
                if not is_water(nbr):
                    continue
                nbr_level = get_fluid_level(nbr)
                if nbr_level <= 1:
                    continue
                if y > 0 and (can_flow_into(world, nx, y - 1, nz) or is_water(world.get_block(nx, y - 1, nz))):
                    continue
                dirs = find_dropoff_directions(world, nx, y, nz)
                if (-dx, -dz) in dirs:
                    pot = nbr_level - 1
                    if pot > max_level:
                        max_level = pot

        if max_level <= 0:
            if s_info["waterlogged"]:
                return f"{s_info['base_id']}_top" if s_info["half"] == "top" else s_info["base_id"]
            return cur
        return get_waterlogged_slab(s_info["base_id"], half=s_info["half"], level=max_level)

    if is_water_source(cur):
        return WATER_SOURCE
    if not can_flow_into(world, x, y, z):
        return cur

    below = world.get_block(x, y - 1, z)
    below_solid_or_source = (y > 0) and (world.is_solid(x, y - 1, z) or is_water_source(below))

    # Infinite water source: 2 or more adjacent sources on solid or source ground
    if below_solid_or_source:
        source_count = 0
        for dx, dz in CARDINALS:
            if is_water_source(world.get_block(x + dx, y, z + dz)):
                source_count += 1
        if source_count >= 2:
            return WATER_SOURCE

    # Falling water from above
    above = world.get_block(x, y + 1, z) if (y + 1 < WORLD_HEIGHT) else AIR
    if is_water(above):
        return WATER_FALLING

    # Horizontal spread
    max_level = 0
    for dx, dz in CARDINALS:
        nx = x + dx
        nz = z + dz
        nbr = world.get_block(nx, y, nz)
        if not is_water(nbr):
            continue
        nbr_level = get_fluid_level(nbr)
        if nbr_level <= 1:
            continue
        if not can_flow_into(world, x, y, z, from_level=nbr_level, from_y=y):
            continue
        # If neighbor can fall down directly below it or is resting on water, it doesn't spread horizontally here
        if y > 0 and (can_flow_into(world, nx, y - 1, nz) or is_water(world.get_block(nx, y - 1, nz))):
            continue
        dirs = find_dropoff_directions(world, nx, y, nz)
        if (-dx, -dz) in dirs:
            pot = nbr_level - 1
            if pot > max_level:
                max_level = pot

    if max_level <= 0:
        return AIR if is_water(cur) else cur
    return WATER_LEVEL_BLOCKS[max_level]


class FluidSystem:
    def __init__(self, world):
        self.world = world
        self.queue = set()
        self.tick_timer = 0.0
        self.tick_interval = 0.15

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

    def update(self, dt):
        """Tick fluid simulation in player loaded chunks."""
        self.tick_timer += dt
        if self.tick_timer < self.tick_interval:
            return []
        self.tick_timer = 0.0
        if not self.queue:
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
            count += 1
            cur = self.world.get_block(x, y, z)
            new_block = compute_new_fluid_state(self.world, x, y, z)
            if new_block != cur:
                if self.world.set_block(x, y, z, new_block):
                    changed_blocks.append((x, y, z, new_block))
                    self.schedule(x, y, z)
            elif is_water(cur):
                from server.blocks import get_slab_info

                s_info = get_slab_info(cur)
                can_fall_below = True
                if s_info is not None and s_info["half"] == "bottom":
                    can_fall_below = False

                # If water can fall down, schedule below
                if can_fall_below and y > 0 and can_flow_into(self.world, x, y - 1, z, from_y=y):
                    if self.world.get_block(x, y - 1, z) != WATER_FALLING:
                        self.queue.add((x, y - 1, z))
                elif y > 0 and is_water(self.world.get_block(x, y - 1, z)):
                    # Water resting on water cannot spread horizontally
                    pass
                else:
                    cur_level = get_fluid_level(cur)
                    if cur_level > 1:
                        flow_dirs = find_dropoff_directions(self.world, x, y, z)
                        for dx, dz in flow_dirs:
                            nx = x + dx
                            nz = z + dz
                            if can_flow_into(self.world, nx, y, nz, from_level=cur_level, from_y=y):
                                target_level = cur_level - 1
                                expected = WATER_LEVEL_BLOCKS[target_level]
                                if self.world.get_block(nx, y, nz) != expected:
                                    self.queue.add((nx, y, nz))

            # Fluid collision transformation
            active_fluid = new_block if is_water(new_block) else (cur if is_water(cur) else None)
            if active_fluid:
                for n_dx, n_dy, n_dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    nx, ny, nz = x + n_dx, y + n_dy, z + n_dz
                    if 0 <= ny < WORLD_HEIGHT:
                        nbr = self.world.get_block(nx, ny, nz)
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
                        if is_water(adj):
                            transformed = find_fluid_transform(adj, cur)
                            if transformed and transformed != cur:
                                if self.world.set_block(x, y, z, transformed):
                                    changed_blocks.append((x, y, z, transformed))
                                    self.schedule(x, y, z)
                                break

        return changed_blocks
