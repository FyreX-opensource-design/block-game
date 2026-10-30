import math

import numpy as np

from server.biomes import BIOMES, biome_at, climate_grid
from server.blocks import (
    AIR,
    BLOCKS,
    CHUNK_SIZE,
    CLAY,
    GRAVEL,
    GRASS,
    LOAD_RADIUS,
    SALT_WATER,
    SECTION_COUNT,
    STONE,
    UNLOAD_RADIUS,
    VERTICAL_RADIUS,
    VERTICAL_UNLOAD,
    WATER,
    WATER_LEVEL,
    WORLD_HEIGHT,
    block_from_index,
    block_index,
)
from server.caves import load_caves
from server.ores import load_ores
from server.noise import fbm3_grid, fbm_grid, hash2, terrain_grid, terrain_height

TREE_MARGIN = 4


def _intersect_ray_box(ox, oy, oz, dx, dy, dz, bx0, by0, bz0, bx1, by1, bz1):
    inv_dx = 1.0 / dx if dx != 0 else (1e30 if dx >= 0 else -1e30)
    inv_dy = 1.0 / dy if dy != 0 else (1e30 if dy >= 0 else -1e30)
    inv_dz = 1.0 / dz if dz != 0 else (1e30 if dz >= 0 else -1e30)

    tx0 = (bx0 - ox) * inv_dx
    tx1 = (bx1 - ox) * inv_dx
    if tx0 > tx1:
        tx0, tx1 = tx1, tx0

    ty0 = (by0 - oy) * inv_dy
    ty1 = (by1 - oy) * inv_dy
    if ty0 > ty1:
        ty0, ty1 = ty1, ty0

    tz0 = (bz0 - oz) * inv_dz
    tz1 = (bz1 - oz) * inv_dz
    if tz0 > tz1:
        tz0, tz1 = tz1, tz0

    tmin = max(tx0, ty0, tz0)
    tmax = min(tx1, ty1, tz1)

    if tmax < 0 or tmin > tmax:
        return None

    if tmin < 0:
        t_hit = 0.0
        hit_x, hit_y, hit_z = ox, oy, oz
        normal = (0, 1, 0)
    else:
        t_hit = tmin
        hit_x = ox + dx * t_hit
        hit_y = oy + dy * t_hit
        hit_z = oz + dz * t_hit
        if abs(tmin - tx0) < 1e-5:
            normal = (-1 if dx > 0 else 1, 0, 0)
        elif abs(tmin - ty0) < 1e-5:
            normal = (0, -1 if dy > 0 else 1, 0)
        else:
            normal = (0, 0, -1 if dz > 0 else 1)

    return t_hit, hit_x, hit_y, hit_z, normal


class Chunk:
    def __init__(self, cx, cy, cz):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.data = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint16)
        self.skylight = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
        self.blocklight = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, 5), dtype=np.uint8)
        self.light_valid = False


def tree_at(x, z, seed=0):
    height = terrain_height(x, z, seed)
    return _tree_site(x, z, height, seed, BIOMES, lambda px, pz: terrain_height(px, pz, seed)) is not None


def _tree_site(x, z, height, seed, biomes, height_at):
    biome = biome_at(x, z, seed, biomes)
    if not biome.trees or hash2(x, z, seed) > biome.tree_chance or height <= WATER_LEVEL:
        return None
    from server.tags import has_block_tag

    allowed = False
    for tree in biome.trees:
        valid_b = getattr(tree, "valid_blocks", ("grass", "dirt", "muddy_grass", "mud"))
        for vb in valid_b:
            if vb.startswith("#"):
                if has_block_tag(biome.surface, vb[1:]):
                    allowed = True
                    break
            elif biome.surface == vb:
                allowed = True
                break
        if allowed or biome.surface in ("grass", "dirt", "muddy_grass", "mud"):
            allowed = True
            break
    if not allowed:
        return None
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dz == 0:
                continue
            if abs(height_at(x + dx, z + dz) - height) > 1:
                return None
    choice = int(hash2(x + 3, z + 5, seed) * len(biome.trees))
    return biome.trees[min(choice, len(biome.trees) - 1)]


def _limit_vein_size(mask, noise, max_size):
    try:
        from scipy.ndimage import label

        labeled, num_features = label(mask)
    except ImportError:
        labeled, num_features = _label_mask(mask)

    if num_features == 0:
        return mask

    result = np.zeros_like(mask)
    shape = mask.shape
    for feat in range(1, num_features + 1):
        indices = np.where(labeled == feat)
        count = len(indices[0])
        if count <= max_size:
            result[indices] = True
        else:
            feat_noise = noise[indices]
            best_local_idx = int(np.argmax(feat_noise))
            seed = (int(indices[0][best_local_idx]), int(indices[1][best_local_idx]), int(indices[2][best_local_idx]))
            visited = {seed}
            queue = [seed]
            while queue and len(visited) < max_size:
                cy, cz, cx = queue.pop(0)
                for dy, dz, dx in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
                    ny, nz, nx = cy + dy, cz + dz, cx + dx
                    if 0 <= ny < shape[0] and 0 <= nz < shape[1] and 0 <= nx < shape[2]:
                        if labeled[ny, nz, nx] == feat and (ny, nz, nx) not in visited:
                            visited.add((ny, nz, nx))
                            queue.append((ny, nz, nx))
                            if len(visited) >= max_size:
                                break
            for vy, vz, vx in visited:
                result[vy, vz, vx] = True
    return result


def _label_mask(mask):
    labeled = np.zeros_like(mask, dtype=np.int32)
    cur_label = 0
    shape = mask.shape
    for y in range(shape[0]):
        for z in range(shape[1]):
            for x in range(shape[2]):
                if mask[y, z, x] and labeled[y, z, x] == 0:
                    cur_label += 1
                    queue = [(y, z, x)]
                    labeled[y, z, x] = cur_label
                    while queue:
                        cy, cz, cx = queue.pop(0)
                        for dy, dz, dx in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
                            ny, nz, nx = cy + dy, cz + dz, cx + dx
                            if 0 <= ny < shape[0] and 0 <= nz < shape[1] and 0 <= nx < shape[2]:
                                if mask[ny, nz, nx] and labeled[ny, nz, nx] == 0:
                                    labeled[ny, nz, nx] = cur_label
                                    queue.append((ny, nz, nx))
    return labeled, cur_label


class World:
    def __init__(self, seed=0, store=None, caves=None, biomes=None, ores=None, day_time=1000.0, features=None, spawn_point=None):
        self.seed = int(seed)
        self.store = store
        self.day_time = float(day_time)
        self.caves = load_caves() if caves is None else tuple(caves)
        self.biomes = BIOMES if biomes is None else tuple(biomes)
        self.ores = load_ores() if ores is None else tuple(ores)
        from server.features import FEATURES, load_features

        self.features = FEATURES if features is None else tuple(features)
        self.chunks = {}
        self.dirty = set()
        self.modified = set()
        self.columns = set()
        self.column_max = {}
        self.players = {}
        self.entities = []
        self.furnaces = {}
        self._spawn_point = None
        if spawn_point is not None:
            self.spawn_point = spawn_point
        from server.fluid import FluidSystem

        self.fluid = FluidSystem(self)
        from server.lighting import LightingSystem

        self.lighting = LightingSystem(self)
        from server.spawner import MobSpawner

        self.spawner = MobSpawner(self)
        from server.leaf_decay import LeafDecaySystem

        self.leaf_decay = LeafDecaySystem(self)
        from server.transformations import TransformationSystem

        self.transformations = TransformationSystem(self)
        self._singleplayer_player = None

    def advance_time(self, dt):
        """Advance world time by dt seconds (20 ticks per second, 24000 ticks per day cycle)."""
        self.day_time = (self.day_time + dt * 20.0) % 24000.0

    def get_sky_light_level(self):
        """Returns the current world-wide skylight level (7.0 to 15.0) based on day_time."""
        t = (self.day_time / 24000.0) % 1.0
        alpha = 2.0 * math.pi * (t - 0.25)
        sun_height = math.cos(alpha)
        transition = max(0.0, min(1.0, (sun_height + 0.15) / 0.30))
        s = transition * transition * (3.0 - 2.0 * transition)
        return 7.0 + 8.0 * s

    def get_sky_intensity(self):
        """Returns the current sky intensity multiplier (7/15 to 1.0)."""
        return self.get_sky_light_level() / 15.0

    def get_current_skylight(self, x, y, z):
        """Returns the effective skylight (0 to 15) at (x, y, z) scaled by the current day/night cycle."""
        raw = self.get_skylight(x, y, z)
        return int(round(raw * self.get_sky_intensity()))

    def get_skylight(self, x, y, z):
        if getattr(self, "lighting", None) is not None:
            return self.lighting.get_skylight(x, y, z)
        return 15

    def get_blocklight(self, x, y, z):
        if getattr(self, "lighting", None) is not None:
            return self.lighting.get_blocklight(x, y, z)
        return (0, 0, 0, 0, 0)

    def get_light(self, x, y, z):
        if getattr(self, "lighting", None) is not None:
            res = dict(self.lighting.get_light(x, y, z))
            res["current_sky"] = self.get_current_skylight(x, y, z)
            return res
        return {
            "sky": 15,
            "current_sky": int(round(15 * self.get_sky_intensity())),
            "block": (0, 0, 0, 0, 0),
            "r": 0,
            "g": 0,
            "b": 0,
            "uv": 0,
            "ir": 0,
        }

    def set_blocklight(self, x, y, z, r, g, b, uv=0, ir=0):
        if getattr(self, "lighting", None) is not None:
            return self.lighting.set_blocklight(x, y, z, r, g, b, uv, ir)

    def get_furnace(self, x, y, z):
        return self.furnaces.get((int(x), int(y), int(z)))

    def get_or_create_furnace(self, x, y, z, block_id=None):
        key = (int(x), int(y), int(z))
        if key not in self.furnaces:
            from server.furnace import Furnace

            b_id = block_id or self.get_block(*key)
            self.furnaces[key] = Furnace(key[0], key[1], key[2], block_id=b_id)
        elif block_id is not None:
            self.furnaces[key].block_id = str(block_id)
        return self.furnaces[key]

    def update_furnaces(self, dt):
        for furnace in list(self.furnaces.values()):
            was_lit = furnace.is_lit
            furnace.tick(dt)
            if furnace.is_lit != was_lit and getattr(self, "lighting", None) is not None:
                self.lighting.on_block_changed(furnace.x, furnace.y, furnace.z, was_emitter=True)

    def update_fluids(self, dt):
        if getattr(self, "fluid", None) is not None:
            return self.fluid.update(dt)
        return []

    def update_leaves(self, dt):
        if getattr(self, "leaf_decay", None) is not None:
            return self.leaf_decay.update(dt)
        return []

    def update_transformations(self, dt):
        if getattr(self, "transformations", None) is not None:
            return self.transformations.update(dt)
        return []

    def spawn_item(self, item, count=1, pos=None, vel=None, despawn_timer=300.0, pickup_delay=0.0):
        from server.entity import ItemEntity

        entity = ItemEntity(item, count=count, pos=pos, vel=vel, despawn_timer=despawn_timer, pickup_delay=pickup_delay)
        self.entities.append(entity)
        return entity

    def spawn_falling_block(self, block_id, pos=None, vel=None):
        from server.entity import FallingBlockEntity

        entity = FallingBlockEntity(block_id, pos=pos, vel=vel)
        self.entities.append(entity)
        return entity

    def spawn_mob(self, mob_or_type, pos=None, vel=None, yaw=0.0, pitch=0.0):
        from server.entity import Mob, Pig, Sheep

        if isinstance(mob_or_type, (Mob, Pig, Sheep)):
            mob = mob_or_type
            if pos is not None:
                mob.pos = list(pos)
        elif str(mob_or_type) == "pig":
            mob = Pig(pos=pos, vel=vel, yaw=yaw, pitch=pitch)
        elif str(mob_or_type) == "sheep":
            mob = Sheep(pos=pos, vel=vel, yaw=yaw, pitch=pitch)
        else:
            mob = Mob(mob_type=str(mob_or_type), pos=pos, vel=vel, yaw=yaw, pitch=pitch)
        mob.world = self
        self.entities.append(mob)
        return mob

    def hit_block_with_item(self, x, y, z, item_id, rng=None):
        import random
        from server.loot import load_loot_table, roll_loot_table
        from server.recipes import find_block_hit_recipe

        if rng is None:
            rng = random
        block_id = self.get_block(x, y, z)
        recipe = find_block_hit_recipe(item_id, block_id)
        if recipe is None:
            return False, [], False

        roll = rng.random()
        if roll <= recipe.chance:
            drops = []
            if recipe.loot_table:
                table = load_loot_table(recipe.loot_table)
                if table:
                    context = {"block_id": block_id, "held_item": item_id, "random": rng}
                    drops = roll_loot_table(table, context)
            elif recipe.result:
                drops = [(recipe.result, recipe.count)]
            return True, drops, recipe.consume_item
        return False, [], False

    def check_falling(self, x, y, z):
        if y <= 0 or y >= WORLD_HEIGHT:
            return None
        block_id = self.get_block(x, y, z)
        if not block_id or block_id == "air":
            return None
        from server.blocks import BLOCKS

        block = BLOCKS.get(block_id)
        if block is None or not getattr(block, "gravity", False):
            return None

        # Check if unsupported below
        below_id = self.get_block(x, y - 1, z)
        below_block = BLOCKS.get(below_id)
        # Air, empty, or non-solid (like water) cannot support falling blocks
        if below_id in ("", "air", None) or (below_block is not None and not below_block.solid):
            cx, cy, cz = self.chunk_key(x, y, z)
            chunk = self.chunks.get((cx, cy, cz))
            if chunk is not None:
                lx = x - cx * CHUNK_SIZE
                ly = y - cy * CHUNK_SIZE
                lz = z - cz * CHUNK_SIZE
                chunk.data[ly, lz, lx] = 0
                self.modified.add((cx, cy, cz))
                self.mark_dirty(cx, cy, cz)
                if lx == 0:
                    self.mark_dirty(cx - 1, cy, cz)
                if lx == CHUNK_SIZE - 1:
                    self.mark_dirty(cx + 1, cy, cz)
                if ly == 0:
                    self.mark_dirty(cx, cy - 1, cz)
                if ly == CHUNK_SIZE - 1:
                    self.mark_dirty(cx, cy + 1, cz)
                if lz == 0:
                    self.mark_dirty(cx, cy, cz - 1)
                if lz == CHUNK_SIZE - 1:
                    self.mark_dirty(cx, cy, cz + 1)
                if getattr(self, "lighting", None) is not None:
                    self.lighting.on_block_changed(x, y, z)
                if getattr(self, "fluid", None) is not None:
                    self.fluid.schedule(x, y, z)
            return self.spawn_falling_block(block_id, pos=[x + 0.5, float(y), z + 0.5])
        return None

    def notify_neighbors_falling(self, x, y, z):
        if getattr(self, "_in_falling_update", False):
            return
        self._in_falling_update = True
        try:
            queue = [(x, y, z), (x, y + 1, z), (x, y - 1, z), (x + 1, y, z), (x - 1, y, z), (x, y, z + 1), (x, y, z - 1)]
            seen = set()
            while queue:
                qx, qy, qz = queue.pop(0)
                if (qx, qy, qz) in seen:
                    continue
                seen.add((qx, qy, qz))
                fell = self.check_falling(qx, qy, qz)
                if fell is not None:
                    queue.append((qx, qy + 1, qz))
                    queue.append((qx + 1, qy, qz))
                    queue.append((qx - 1, qy, qz))
                    queue.append((qx, qy, qz + 1))
                    queue.append((qx, qy, qz - 1))
        finally:
            self._in_falling_update = False

    def remove_entity(self, entity_id):
        self.entities = [e for e in self.entities if e.id != int(entity_id)]

    def update_entities(self, dt):
        from server.entity import ItemEntity

        if getattr(self, "spawner", None) is not None:
            self.spawner.update(dt)

        alive = []
        for entity in self.entities:
            entity.update(dt, self)
            is_alive = getattr(entity, "alive", True)
            despawn_ok = getattr(entity, "despawn_timer", 1.0) > 0
            count_ok = getattr(entity, "count", 1) > 0
            if is_alive and despawn_ok and count_ok:
                alive.append(entity)
        # Try merging nearby identical items
        for i in range(len(alive)):
            if not isinstance(alive[i], ItemEntity) or alive[i].count <= 0:
                continue
            for j in range(i + 1, len(alive)):
                if not isinstance(alive[j], ItemEntity) or alive[i].count <= 0:
                    continue
                if alive[j].count > 0:
                    alive[i].try_merge(alive[j])
        self.entities = [e for e in alive if getattr(e, "count", 1) > 0 and getattr(e, "alive", True)]

    def register_player(self, player):
        if not player.uuid:
            raise ValueError("player needs a uuid")
        self.players[player.uuid] = player
        return player

    def unregister_player(self, player_id):
        return self.players.pop(str(player_id), None)

    def chunk_key(self, x, y, z):
        return (x // CHUNK_SIZE, y // CHUNK_SIZE, z // CHUNK_SIZE)

    def chunk_at(self, x, y, z):
        if y < 0 or y >= WORLD_HEIGHT:
            return None
        return self.chunks.get(self.chunk_key(x, y, z))

    def get_block(self, x, y, z):
        chunk = self.chunk_at(x, y, z)
        if chunk is None:
            return AIR
        lx = x - chunk.cx * CHUNK_SIZE
        ly = y - chunk.cy * CHUNK_SIZE
        lz = z - chunk.cz * CHUNK_SIZE
        block = block_from_index(chunk.data[ly, lz, lx])
        return block.id if block is not None else AIR

    def is_solid(self, x, y, z):
        block = BLOCKS.get(self.get_block(x, y, z))
        return block is not None and block.solid

    def blocks_movement(self, x, y, z):
        if y < 0:
            return True
        if y >= WORLD_HEIGHT:
            return False
        cx, cy, cz = self.chunk_key(x, y, z)
        if (cx, cy, cz) in self.chunks:
            return self.is_solid(x, y, z)
        column = (cx, cz)
        if column not in self.columns:
            return True
        if cy > self.column_max[column]:
            return False
        return True

    def get_collision_boxes(self, x, y, z):
        if y < 0:
            return [(float(x), float(y), float(z), float(x + 1), float(y + 1), float(z + 1))]
        if y >= WORLD_HEIGHT:
            return []
        cx, cy, cz = self.chunk_key(x, y, z)
        if (cx, cy, cz) not in self.chunks:
            return [(float(x), float(y), float(z), float(x + 1), float(y + 1), float(z + 1))] if self.blocks_movement(x, y, z) else []

        block_id = self.get_block(x, y, z)
        if not block_id or block_id == AIR:
            return []

        b_obj = BLOCKS.get(block_id)
        if b_obj is None or not b_obj.solid:
            return []

        from server.blocks import get_offset_info, get_offset_torch_info, get_slab_info

        t_info = get_offset_torch_info(b_obj)
        if t_info is not None:
            if t_info["capped"]:
                return [(float(x), float(y) + 0.5, float(z), float(x + 1), float(y + 1), float(z + 1))]
            return []

        o_info = get_offset_info(b_obj)
        if o_info is not None:
            if o_info["capped"]:
                return [(float(x), float(y) - 0.5, float(z), float(x + 1), float(y) + 1.0, float(z + 1))]
            return [(float(x), float(y) - 0.5, float(z), float(x + 1), float(y) + 0.5, float(z + 1))]

        info = get_slab_info(b_obj)
        if info is not None:
            if info["half"] == "bottom":
                return [(float(x), float(y), float(z), float(x + 1), float(y) + 0.5, float(z + 1))]
            elif info["half"] == "top":
                return [(float(x), float(y) + 0.5, float(z), float(x + 1), float(y + 1), float(z + 1))]
            else:
                return [(float(x), float(y), float(z), float(x + 1), float(y + 1), float(z + 1))]

        return [(float(x), float(y), float(z), float(x + 1), float(y + 1), float(z + 1))]

    def collides_box(self, min_x, min_y, min_z, max_x, max_y, max_z):
        ix0 = math.floor(min_x)
        ix1 = math.floor(max_x - 1e-6)
        iy0 = math.floor(min_y)
        iy1 = math.floor(max_y - 1e-6)
        iz0 = math.floor(min_z)
        iz1 = math.floor(max_z - 1e-6)

        for y in range(iy0, iy1 + 2):
            for z in range(iz0, iz1 + 1):
                for x in range(ix0, ix1 + 1):
                    boxes = self.get_collision_boxes(x, y, z)
                    for bx0, by0, bz0, bx1, by1, bz1 in boxes:
                        if max_x > bx0 and min_x < bx1 and max_y > by0 and min_y < by1 and max_z > bz0 and min_z < bz1:
                            return True
        return False

    def is_opaque(self, x, y, z):
        return self.get_block(x, y, z) != AIR

    def column_top(self, x, z):
        cx = x // CHUNK_SIZE
        cz = z // CHUNK_SIZE
        lx = x - cx * CHUNK_SIZE
        lz = z - cz * CHUNK_SIZE
        found = False
        for cy in range(SECTION_COUNT - 1, -1, -1):
            chunk = self.chunks.get((cx, cy, cz))
            if chunk is None:
                continue
            found = True
            column = chunk.data[:, lz, lx]
            for ly in range(CHUNK_SIZE - 1, -1, -1):
                block = block_from_index(column[ly])
                if block is not None and block.solid:
                    return cy * CHUNK_SIZE + ly
        if not found:
            return terrain_height(x, z, self.seed, self.biomes)
        return 0

    def set_block(self, x, y, z, block_id):
        if y <= 0 or y >= WORLD_HEIGHT:
            return False
        if block_id in ("", AIR, None):
            stored = 0
        elif block_id not in BLOCKS:
            return False
        else:
            stored = block_index(block_id)
        cx, cy, cz = self.chunk_key(x, y, z)
        chunk = self.chunks.get((cx, cy, cz))
        if chunk is None:
            return False
        lx = x - cx * CHUNK_SIZE
        ly = y - cy * CHUNK_SIZE
        lz = z - cz * CHUNK_SIZE
        if int(chunk.data[ly, lz, lx]) == stored:
            return False
        old_idx = int(chunk.data[ly, lz, lx])
        old_block = block_from_index(old_idx)
        if (old_block and (old_block.id == "furnace" or getattr(old_block, "base_id", "") == "furnace" or getattr(old_block, "station", None) == "furnace")) or (x, y, z) in self.furnaces:
            furnace = self.furnaces.pop((x, y, z), None)
            if furnace:
                from server.stacks import stack_count, stack_id

                for stack in (furnace.input, furnace.fuel, furnace.output):
                    sid = stack_id(stack)
                    scnt = stack_count(stack)
                    if sid and scnt > 0:
                        self.spawn_item(sid, count=scnt, pos=(x + 0.5, y + 0.5, z + 0.5))
        if old_block and getattr(old_block, "fluid_breakable", False):
            from server.fluid import is_fluid, is_water

            if is_water(block_id) or is_fluid(block_id):
                from server.loot import get_block_drops

                drops = get_block_drops(old_block)
                for drop_id, count in drops:
                    if count > 0:
                        self.spawn_item(drop_id, count=count, pos=(x + 0.5, y + 0.2, z + 0.5))
        chunk.data[ly, lz, lx] = stored
        is_furnace = block_id == "furnace" or (block_id in BLOCKS and (getattr(BLOCKS[block_id], "base_id", "") == "furnace" or getattr(BLOCKS[block_id], "station", None) == "furnace"))
        if is_furnace:
            self.get_or_create_furnace(x, y, z)
        self.modified.add((cx, cy, cz))
        self.mark_dirty(cx, cy, cz)
        if lx == 0:
            self.mark_dirty(cx - 1, cy, cz)
        if lx == CHUNK_SIZE - 1:
            self.mark_dirty(cx + 1, cy, cz)
        if ly == 0:
            self.mark_dirty(cx, cy - 1, cz)
        if ly == CHUNK_SIZE - 1:
            self.mark_dirty(cx, cy + 1, cz)
        if lz == 0:
            self.mark_dirty(cx, cy, cz - 1)
        if lz == CHUNK_SIZE - 1:
            self.mark_dirty(cx, cy, cz + 1)
        if getattr(self, "fluid", None) is not None:
            self.fluid.schedule(x, y, z)
        if getattr(self, "leaf_decay", None) is not None and old_block:
            from server.tags import has_block_tag
            if has_block_tag(old_block.id, "logs") or "log" in old_block.id:
                self.leaf_decay.schedule_neighbors(x, y, z, radius=5)
            elif has_block_tag(old_block.id, "leaves") or "leaves" in old_block.id:
                self.leaf_decay.schedule_neighbors(x, y, z, radius=2)
        if getattr(self, "lighting", None) is not None:
            from server.lighting import _get_luts

            op_lut, em_lut = _get_luts()
            old_em = em_lut[old_idx]
            new_em = em_lut[stored]
            if op_lut[old_idx] != op_lut[stored] or np.any(old_em != new_em):
                was_emitter = bool(np.any(old_em > 0) or np.any(new_em > 0))
                self.lighting.on_block_changed(x, y, z, was_emitter=was_emitter)
        if getattr(self, "on_block_set", None) is not None:
            self.on_block_set(x, y, z, block_id if block_id not in ("", None) else AIR)
        self.notify_neighbors_falling(x, y, z)
        self.check_plant_support(x, y + 1, z)
        for tx, ty, tz in ((x, y + 1, z), (x + 1, y, z), (x - 1, y, z), (x, y, z + 1), (x, y, z - 1)):
            self.check_torch_support(tx, ty, tz)
        return True

    def check_torch_support(self, x, y, z):
        if y <= 0 or y >= WORLD_HEIGHT:
            return
        t_id = self.get_block(x, y, z)
        if not t_id or t_id == AIR:
            return
        from server.blocks import BLOCKS, can_support_torch, is_torch

        b_obj = BLOCKS.get(t_id)
        if b_obj is None or not is_torch(b_obj):
            return
        mount = b_obj.properties.get("mount", "up")
        if mount in ("up", "up_offset"):
            support_id = self.get_block(x, y - 1, z)
        else:
            offsets = {
                "east": (-1, 0),
                "west": (1, 0),
                "south": (0, -1),
                "north": (0, 1),
            }
            dx, dz = offsets.get(mount, (0, 0))
            support_id = self.get_block(x + dx, y, z + dz)
        if not can_support_torch(support_id, mount):
            from server.blocks import get_offset_torch_info
            from server.loot import get_block_drops

            t_info = get_offset_torch_info(b_obj)
            if t_info is not None and t_info["capped"]:
                torch_obj = BLOCKS.get(t_info["base_id"], b_obj)
                drops = get_block_drops(torch_obj)
                for drop_id, count in drops:
                    if count > 0:
                        self.spawn_item(drop_id, count=count, pos=(x + 0.5, y + 0.0, z + 0.5))
                self.set_block(x, y, z, f"{t_info['top_slab']}_top")
            else:
                drops = get_block_drops(b_obj)
                for drop_id, count in drops:
                    if count > 0:
                        self.spawn_item(drop_id, count=count, pos=(x + 0.5, y + 0.2, z + 0.5))
                self.set_block(x, y, z, AIR)

    def check_plant_support(self, x, y, z):
        if y <= 0 or y >= WORLD_HEIGHT:
            return
        above_id = self.get_block(x, y, z)
        if not above_id or above_id == AIR:
            return
        from server.blocks import BLOCKS, can_sustain_plant

        b_obj = BLOCKS.get(above_id)
        if b_obj is None:
            return
        if (
            getattr(b_obj, "requires_soil", False)
            or getattr(b_obj, "plant", False)
            or b_obj.valid_ground is not None
            or above_id in ("tall_grass", "fern", "rose", "dandelion")
        ):
            ground_id = self.get_block(x, y - 1, z)
            if not can_sustain_plant(ground_id, b_obj):
                from server.loot import get_block_drops

                drops = get_block_drops(b_obj)
                for drop_id, count in drops:
                    if count > 0:
                        self.spawn_item(drop_id, count=count, pos=(x + 0.5, y + 0.2, z + 0.5))
                self.set_block(x, y, z, AIR)

    def mark_dirty(self, cx, cy, cz):
        key = (cx, cy, cz)
        if key in self.chunks:
            self.dirty.add(key)

    def take_dirty(self, limit=None, player_pos=None):
        if limit is None:
            keys = list(self.dirty)
            self.dirty.clear()
            return keys
        if not self.dirty:
            return []
        keys = list(self.dirty)
        if player_pos is not None:
            pcx = int(player_pos[0] // CHUNK_SIZE)
            pcy = int(player_pos[1] // CHUNK_SIZE)
            pcz = int(player_pos[2] // CHUNK_SIZE)
            keys.sort(key=lambda k: (k[0] - pcx) ** 2 + (k[1] - pcy) ** 2 + (k[2] - pcz) ** 2)
        taken = keys[:limit]
        for key in taken:
            self.dirty.discard(key)
        return taken

    def load_span(self, cx, cy, cz):
        cy = min(SECTION_COUNT - 1, max(0, int(cy)))
        layers = 0
        for dy in range(-VERTICAL_RADIUS, VERTICAL_RADIUS + 1):
            if 0 <= cy + dy < SECTION_COUNT:
                layers += 1
        return layers * (LOAD_RADIUS * 2 + 1) ** 2

    def sync_players(self, centers):
        """Sync chunks around multiple player chunk centers [(cx, cy, cz), ...].

        Evicts chunks that are outside UNLOAD_RADIUS from ALL player centers.
        Loads the closest missing chunk within LOAD_RADIUS of ANY player center.
        """
        parsed = [
            (int(c[0]), min(SECTION_COUNT - 1, max(0, int(c[1]))), int(c[2]))
            for c in centers
            if c is not None
        ]
        if not parsed:
            return None, []

        removed = []
        for key, chunk in list(self.chunks.items()):
            retained = any(
                abs(chunk.cx - cx) <= UNLOAD_RADIUS
                and abs(chunk.cz - cz) <= UNLOAD_RADIUS
                and abs(chunk.cy - cy) <= VERTICAL_UNLOAD
                for cx, cy, cz in parsed
            )
            if not retained:
                self.evict_chunk(key)
                removed.append(key)

        built = None
        best = (math.inf, 0)
        candidates = set()
        for cx, cy, cz in parsed:
            for dy in range(-VERTICAL_RADIUS, VERTICAL_RADIUS + 1):
                ty = cy + dy
                if ty < 0 or ty >= SECTION_COUNT:
                    continue
                for dz in range(-LOAD_RADIUS, LOAD_RADIUS + 1):
                    for dx in range(-LOAD_RADIUS, LOAD_RADIUS + 1):
                        candidates.add((cx + dx, ty, cz + dz))

        for cand in candidates:
            if cand in self.chunks:
                continue
            cand_dist = min(
                (cand[0] - cx) ** 2 + (cand[2] - cz) ** 2
                for cx, cy, cz in parsed
            )
            metric = (cand_dist, -cand[1])
            if metric < best:
                best = metric
                built = cand

        if built is not None:
            self.load_chunk(*built)
        return built, removed

    def sync_all_players(self):
        centers = [self.chunk_key(*p.pos) for p in self.players.values() if getattr(p, "pos", None)]
        return self.sync_players(centers)

    def sync(self, cx, cy, cz):
        return self.sync_players([(cx, cy, cz)])

    def evict_chunk(self, key):
        chunk = self.chunks.get(key)
        if chunk is None:
            return
        if self.store is not None and key in self.modified:
            self.store.write_chunk(chunk)
            self.modified.discard(key)
        del self.chunks[key]
        self.dirty.discard(key)
        self._note_evicted(chunk.cx, chunk.cy, chunk.cz)

    def flush(self):
        if self.store is None:
            return
        for key in list(self.modified):
            chunk = self.chunks.get(key)
            if chunk is not None:
                self.store.write_chunk(chunk)
            self.modified.discard(key)

    def load_chunk(self, cx, cy, cz):
        if cy < 0 or cy >= SECTION_COUNT:
            return None
        key = (cx, cy, cz)
        if key in self.chunks:
            return self.chunks[key]
        if self.store is not None:
            data = self.store.read_chunk(cx, cy, cz)
            if data is not None:
                chunk = Chunk(cx, cy, cz)
                chunk.data[:, :, :] = data
                self._remember(chunk)
                return chunk
        return self.generate_chunk(cx, cy, cz)

    def generate_column(self, cx, cz):
        return [self.generate_chunk(cx, cy, cz) for cy in range(SECTION_COUNT)]

    def generate_chunk(self, cx, cy, cz):
        chunk = Chunk(cx, cy, cz)
        x0 = cx * CHUNK_SIZE
        y0 = cy * CHUNK_SIZE
        z0 = cz * CHUNK_SIZE
        margin = TREE_MARGIN
        span = np.arange(-margin, CHUNK_SIZE + margin)
        zz, xx = np.meshgrid(z0 + span, x0 + span, indexing="ij")
        heights = terrain_grid(xx, zz, self.seed, self.biomes)

        def sample(lx, lz):
            return int(heights[lz + margin, lx + margin])

        data = chunk.data
        stone = block_index(STONE)
        levels = np.arange(y0, y0 + CHUNK_SIZE, dtype=np.int16).reshape(CHUNK_SIZE, 1, 1)
        surface = heights[margin : margin + CHUNK_SIZE, margin : margin + CHUNK_SIZE]
        tops = surface.reshape(1, CHUNK_SIZE, CHUNK_SIZE)
        climate = climate_grid(xx, zz, self.seed, self.biomes)
        chosen = np.full(climate.shape, len(self.biomes) - 1, dtype=np.int16)
        for index, biome in enumerate(self.biomes):
            chosen[(climate <= biome.ceiling) & (chosen == len(self.biomes) - 1)] = index
        here = chosen[margin : margin + CHUNK_SIZE, margin : margin + CHUNK_SIZE]
        depth = np.array([biome.soil_depth for biome in self.biomes], dtype=np.int16)[here]
        filled = levels <= tops
        stone_mask = filled & ((levels == 0) | (levels < tops - depth.reshape(1, CHUNK_SIZE, CHUNK_SIZE)))
        top_mask = filled & (levels == tops) & (levels != 0)
        cover_mask = filled & ~stone_mask & ~top_mask
        data[stone_mask] = stone
        for index, biome in enumerate(self.biomes):
            columns = here == index
            if not np.any(columns):
                continue
            data[cover_mask & columns] = block_index(biome.soil)
            data[top_mask & columns] = block_index(biome.surface)

        chunk_xx = xx[margin : margin + CHUNK_SIZE, margin : margin + CHUNK_SIZE]
        chunk_zz = zz[margin : margin + CHUNK_SIZE, margin : margin + CHUNK_SIZE]
        patch = fbm_grid(chunk_xx * 0.08 + 15.0, chunk_zz * 0.08 - 25.0, 2, self.seed + 313)
        patch_3d = patch.reshape(1, CHUNK_SIZE, CHUNK_SIZE)
        submerged = tops < WATER_LEVEL
        clay_mask = (top_mask | cover_mask) & submerged & (patch_3d > 0.62)
        gravel_mask = (top_mask | cover_mask) & submerged & (patch_3d < 0.38)
        data[clay_mask] = block_index(CLAY)
        data[gravel_mask] = block_index(GRAVEL)

        ocean_water = block_index(SALT_WATER)
        water_mask = (levels > tops) & (levels <= WATER_LEVEL)
        data[water_mask] = ocean_water

        self._place_ores(data, x0, y0, z0)
        self._carve_caves(data, x0, y0, z0, heights)
        self._place_features(data, x0, y0, z0, heights, sample)

        for lz in range(-2, CHUNK_SIZE + 2):
            for lx in range(-2, CHUNK_SIZE + 2):
                wx = x0 + lx
                wz = z0 + lz
                decorator = self._cached_tree(wx, wz, sample, lx, lz)
                if decorator is None:
                    continue
                self._place_tree(data, lx, lz, sample(lx, lz), wx, wz, y0, decorator)

        self._place_decorations(data, x0, y0, z0, sample)

        self._remember(chunk)
        return chunk

    def _place_ores(self, data, x0, y0, z0):
        if not self.ores:
            return
        y1 = y0 + CHUNK_SIZE - 1
        active = [ore for ore in self.ores if ore.enabled and ore.max_y >= y0 and ore.min_y <= y1]
        if not active:
            return
        ys = np.arange(y0, y0 + CHUNK_SIZE, dtype=np.float64).reshape(CHUNK_SIZE, 1, 1)
        zs = np.arange(z0, z0 + CHUNK_SIZE, dtype=np.float64).reshape(1, CHUNK_SIZE, 1)
        xs = np.arange(x0, x0 + CHUNK_SIZE, dtype=np.float64).reshape(1, 1, CHUNK_SIZE)
        for ore in active:
            targets = ore.target if isinstance(ore.target, (list, tuple)) else (ore.target,)
            target_indices = [block_index(t) for t in targets if block_index(t) is not None]
            if not target_indices:
                continue
            ore_index = block_index(ore.block)
            if ore_index is None:
                continue
            target_mask = np.isin(data, target_indices)
            if not np.any(target_mask):
                continue
            in_range = target_mask & (ys > 0) & (ys >= ore.min_y) & (ys <= ore.max_y)
            if not np.any(in_range):
                continue
            noise = fbm3_grid(
                xs * ore.scale,
                ys * ore.scale,
                zs * ore.scale,
                ore.octaves,
                self.seed + ore.seed_offset,
            )
            mask = in_range & (noise > ore.threshold)
            if ore.max_vein_size > 0 and np.any(mask):
                mask = _limit_vein_size(mask, noise, ore.max_vein_size)
            data[mask] = ore_index

    def _carve_caves(self, data, x0, y0, z0, heights):
        if not self.caves:
            return
        y1 = y0 + CHUNK_SIZE - 1
        active = [cave for cave in self.caves if cave.enabled and cave.max_y >= y0 and cave.min_y <= y1]
        if not active:
            return
        ys = np.arange(y0, y0 + CHUNK_SIZE, dtype=np.float64).reshape(CHUNK_SIZE, 1, 1)
        zs = np.arange(z0, z0 + CHUNK_SIZE, dtype=np.float64).reshape(1, CHUNK_SIZE, 1)
        xs = np.arange(x0, x0 + CHUNK_SIZE, dtype=np.float64).reshape(1, 1, CHUNK_SIZE)
        surface = heights[TREE_MARGIN : TREE_MARGIN + CHUNK_SIZE, TREE_MARGIN : TREE_MARGIN + CHUNK_SIZE]
        is_water_col = surface.reshape(1, CHUNK_SIZE, CHUNK_SIZE) <= WATER_LEVEL
        water = block_index(WATER)
        salt_water = block_index(SALT_WATER)
        for cave in active:
            noise = fbm3_grid(
                xs * cave.horizontal_scale,
                ys * cave.vertical_scale,
                zs * cave.horizontal_scale,
                cave.octaves,
                self.seed + cave.seed_offset,
            )
            if cave.style == "tunnel":
                opened = np.abs(noise - 0.5) < cave.thickness
            else:
                opened = noise > cave.threshold
            opened &= (ys > 0) & (ys >= cave.min_y) & (ys <= cave.max_y)
            max_y = np.where(
                is_water_col,
                surface.reshape(1, CHUNK_SIZE, CHUNK_SIZE) - np.maximum(cave.surface_margin, 4),
                surface.reshape(1, CHUNK_SIZE, CHUNK_SIZE) - cave.surface_margin,
            )
            opened &= ys <= max_y
            if cave.mask_scale > 0:
                mask = fbm_grid(xs * cave.mask_scale, zs * cave.mask_scale, 2, self.seed + cave.seed_offset + 4096)
                opened &= mask > cave.mask_threshold
            if cave.breach_threshold > 0:
                near_surface = ys > (surface.reshape(1, CHUNK_SIZE, CHUNK_SIZE) - cave.breach_depth)
                breach = fbm3_grid(
                    xs * cave.breach_scale,
                    ys * cave.breach_scale,
                    zs * cave.breach_scale,
                    1,
                    self.seed + cave.seed_offset + 77,
                )
                opened &= ~near_surface | ((breach > cave.breach_threshold) & ~is_water_col)
            data[opened & (data != 0) & (data != water) & (data != salt_water)] = 0

    def _place_features(self, data, x0, y0, z0, heights, sample):
        if not self.features:
            return
        water_idx = block_index(WATER)
        for feat in self.features:
            if feat.type == "lake":
                cx = x0 // CHUNK_SIZE
                cz = z0 // CHUNK_SIZE
                for ncz in (cz - 1, cz, cz + 1):
                    for ncx in (cx - 1, cx, cx + 1):
                        orig_x = ncx * CHUNK_SIZE
                        orig_z = ncz * CHUNK_SIZE
                        roll = hash2(orig_x + 17, orig_z - 29, self.seed + 8888)
                        if roll > feat.chance:
                            continue
                        rel_cx = 4 + int(hash2(orig_x + 3, orig_z + 7, self.seed + 9999) * 8)
                        rel_cz = 4 + int(hash2(orig_x + 11, orig_z - 13, self.seed + 9999) * 8)
                        center_wx = orig_x + rel_cx
                        center_wz = orig_z + rel_cz
                        center_y = terrain_height(center_wx, center_wz, self.seed, self.biomes)
                        if center_y <= WATER_LEVEL + 1 or center_y >= WORLD_HEIGHT - 10:
                            continue
                        biome = biome_at(center_wx, center_wz, self.seed, self.biomes)
                        if feat.biomes and biome.id not in feat.biomes:
                            continue

                        r = feat.radius_min + int(
                            hash2(center_wx + 5, center_wz - 5, self.seed + 1234)
                            * (feat.radius_max - feat.radius_min + 1)
                        )
                        # Check flatness and suitability across lake footprint (lakes only generate exposed on surface)
                        diag = max(1, int(r * 0.7))
                        samples = [
                            terrain_height(center_wx + dx, center_wz + dz, self.seed, self.biomes)
                            for dx, dz in (
                                (0, 0), (r, 0), (-r, 0), (0, r), (0, -r),
                                (diag, diag), (-diag, -diag), (diag, -diag), (-diag, diag),
                            )
                        ]
                        min_h = min(samples)
                        max_h = max(samples)
                        if max_h - min_h > 4:
                            continue
                        water_y = min_h
                        if water_y <= WATER_LEVEL or water_y >= WORLD_HEIGHT - 10:
                            continue
                        if any(cave.opens(center_wx, water_y, center_wz, center_y, self.seed) for cave in self.caves):
                            continue

                        if center_wx + r + 3 < x0 or center_wx - r - 3 >= x0 + CHUNK_SIZE:
                            continue
                        if center_wz + r + 3 < z0 or center_wz - r - 3 >= z0 + CHUNK_SIZE:
                            continue
                        if y0 > max_h + 2 or y0 + CHUNK_SIZE <= water_y - feat.depth - 1:
                            continue

                        depth = feat.depth
                        for lz in range(CHUNK_SIZE):
                            wz = z0 + lz
                            dz = wz - center_wz
                            for lx in range(CHUNK_SIZE):
                                wx = x0 + lx
                                dx = wx - center_wx
                                dist = math.sqrt(dx * dx + dz * dz) / r
                                local_g = sample(lx, lz)
                                if dist <= 1.0:
                                    bowl_depth = max(1, round(depth * (1.0 - dist * dist)))
                                    floor_y = water_y - bowl_depth
                                    top_fill = min(water_y, local_g)

                                    for y in range(floor_y, top_fill + 1):
                                        if y0 <= y < y0 + CHUNK_SIZE:
                                            data[y - y0, lz, lx] = water_idx

                                    for y in range(top_fill + 1, local_g + 2):
                                        if y0 <= y < y0 + CHUNK_SIZE:
                                            data[y - y0, lz, lx] = 0

                                    bed_y = floor_y - 1
                                    if y0 <= bed_y < y0 + CHUNK_SIZE:
                                        bed_roll = hash2(wx, wz, self.seed + 5555)
                                        if bed_roll < 0.25:
                                            bed = CLAY
                                        elif bed_roll < 0.55:
                                            bed = GRAVEL
                                        else:
                                            bed = "dirt"
                                        data[bed_y - y0, lz, lx] = block_index(bed)

                                elif dist <= 1.45:
                                    shore_y = local_g
                                    if y0 <= shore_y < y0 + CHUNK_SIZE:
                                        shore_roll = hash2(wx, wz, self.seed + 7777)
                                        if dist <= 1.2 or shore_roll < 0.5:
                                            shore = "mud"
                                        else:
                                            shore = "muddy_grass"
                                        data[shore_y - y0, lz, lx] = block_index(shore)
                                        if y0 <= shore_y - 1 < y0 + CHUNK_SIZE:
                                            data[shore_y - 1 - y0, lz, lx] = block_index("mud")

    def _remember(self, chunk):
        key = (chunk.cx, chunk.cy, chunk.cz)
        self.chunks[key] = chunk
        self.dirty.add(key)
        column = (chunk.cx, chunk.cz)
        self.columns.add(column)
        if chunk.cy > self.column_max.get(column, -1):
            self.column_max[column] = chunk.cy
        for dcx, dcy, dcz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            n_key = (chunk.cx + dcx, chunk.cy + dcy, chunk.cz + dcz)
            nch = self.chunks.get(n_key)
            if nch is not None and getattr(nch, "light_valid", False):
                nch.light_valid = False
            self.mark_dirty(*n_key)
        for ty in range(chunk.cy - 1, -1, -1):
            b_key = (chunk.cx, ty, chunk.cz)
            bch = self.chunks.get(b_key)
            if bch is not None and getattr(bch, "light_valid", False):
                bch.light_valid = False
                self.mark_dirty(*b_key)

    def _note_evicted(self, cx, cy, cz):
        remaining = [key[1] for key in self.chunks if key[0] == cx and key[2] == cz]
        column = (cx, cz)
        if not remaining:
            self.columns.discard(column)
            self.column_max.pop(column, None)
            return
        self.column_max[column] = max(remaining)

    def _cached_tree(self, x, z, sample, lx, lz):
        height = sample(lx, lz)
        decorator = _tree_site(
            x,
            z,
            height,
            self.seed,
            self.biomes,
            lambda px, pz: sample(lx + (px - x), lz + (pz - z)),
        )
        if decorator is None:
            return None
        if any(cave.opens(x, height, z, height, self.seed) for cave in self.caves):
            return None
        return decorator

    def _place_tree(self, data, lx, lz, ground, wx, wz, y0, tree):
        if 0 <= lx < CHUNK_SIZE and 0 <= lz < CHUNK_SIZE and y0 <= ground < y0 + CHUNK_SIZE:
            under = int(data[ground - y0, lz, lx])
            under_b = block_from_index(under)
            under_id = under_b.id if under_b else ""
            valid_b = getattr(tree, "valid_blocks", ("grass", "dirt", "muddy_grass", "mud"))
            from server.tags import has_block_tag

            is_valid = False
            for vb in valid_b:
                if vb.startswith("#"):
                    if has_block_tag(under_id, vb[1:]):
                        is_valid = True
                        break
                elif under_id == vb:
                    is_valid = True
                    break
            if not is_valid and under_id not in ("grass", "dirt", "muddy_grass", "mud"):
                return

        extra = tree.trunk_extra if hash2(wx + 19, wz - 7, self.seed) > tree.extra_threshold else 0
        trunk = max(1, tree.trunk_min + extra)

        def set_air(x, y, z, block_id):
            if x < 0 or z < 0 or x >= CHUNK_SIZE or z >= CHUNK_SIZE:
                return
            if y <= 0 or y < y0 or y >= y0 + CHUNK_SIZE or y >= WORLD_HEIGHT:
                return
            if data[y - y0, z, x] == 0:
                data[y - y0, z, x] = block_index(block_id)

        def set_trunk(x, y, z, block_id):
            if x < 0 or z < 0 or x >= CHUNK_SIZE or z >= CHUNK_SIZE:
                return
            if y <= 0 or y < y0 or y >= y0 + CHUNK_SIZE or y >= WORLD_HEIGHT:
                return
            cur = int(data[y - y0, z, x])
            if cur == 0 or cur == block_index(tree.leaves):
                data[y - y0, z, x] = block_index(block_id)

        shape = getattr(tree, "shape", "blob").lower()

        if shape in ("conifer", "pine", "spruce", "cone"):
            # Conifer / Pine shape: tall trunk with tiered tapering foliage
            for i in range(1, trunk + 1):
                set_trunk(lx, ground + i, lz, tree.trunk)
            leaf_start = ground + max(2, trunk - tree.leaf_down)
            # Tip block on top of trunk
            set_air(lx, ground + trunk + 1, lz, tree.leaves)
            for y in range(leaf_start, ground + trunk + 1):
                dy_from_top = (ground + trunk) - y
                if dy_from_top == 0:
                    cur_r = 1
                elif dy_from_top % 2 == 1:
                    cur_r = min(tree.radius, 1)
                else:
                    cur_r = min(tree.radius, 2 + (dy_from_top // 4))
                for dz in range(-cur_r, cur_r + 1):
                    for dx in range(-cur_r, cur_r + 1):
                        if dx == 0 and dz == 0:
                            continue
                        if cur_r > 1 and abs(dx) == cur_r and abs(dz) == cur_r:
                            continue
                        set_air(lx + dx, y, lz + dz, tree.leaves)

        elif shape in ("tall", "column", "birch"):
            # Tall slender tree with compact cylindrical canopy
            for i in range(1, trunk + 1):
                set_trunk(lx, ground + i, lz, tree.trunk)
            top = ground + trunk
            for dy in range(-tree.leaf_down, tree.leaf_up + 1):
                cur_r = tree.radius
                for dz in range(-cur_r, cur_r + 1):
                    for dx in range(-cur_r, cur_r + 1):
                        if dx == 0 and dz == 0 and dy <= 0:
                            continue
                        if abs(dx) == cur_r and abs(dz) == cur_r and (dy == -tree.leaf_down or dy == tree.leaf_up):
                            continue
                        set_air(lx + dx, top + dy, lz + dz, tree.leaves)

        elif shape in ("shrub", "bush"):
            # Low bush resting on ground
            set_trunk(lx, ground + 1, lz, tree.trunk)
            for dy in range(0, tree.leaf_up + 1):
                cur_r = tree.radius if dy == 0 else max(0, tree.radius - 1)
                for dz in range(-cur_r, cur_r + 1):
                    for dx in range(-cur_r, cur_r + 1):
                        if dx == 0 and dz == 0 and dy == 0:
                            continue
                        if cur_r > 1 and abs(dx) == cur_r and abs(dz) == cur_r:
                            continue
                        set_air(lx + dx, ground + 1 + dy, lz + dz, tree.leaves)

        elif shape in ("branching", "fancy", "large_oak"):
            # Majestic branching tree with angled branches and leafy clusters
            from server.blocks import rotation_variant

            for i in range(1, trunk + 1):
                set_trunk(lx, ground + i, lz, tree.trunk)
            top = ground + trunk
            branch_dirs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))
            branch_cnt = min(getattr(tree, "branch_count", 3), len(branch_dirs))
            for b_idx in range(branch_cnt):
                dir_idx = int(hash2(wx + b_idx * 7, wz - b_idx * 13, self.seed) * len(branch_dirs))
                bdx, bdz = branch_dirs[dir_idx]
                if abs(bdx) > abs(bdz):
                    branch_axis = "x"
                elif abs(bdz) > abs(bdx):
                    branch_axis = "z"
                else:
                    branch_axis = "x" if (bdx * bdz > 0) else "z"
                branch_log = rotation_variant(tree.trunk, branch_axis)
                blen = getattr(tree, "branch_length", 2)
                by = ground + max(2, trunk - 1)
                bx, bz = lx, lz
                for step in range(1, blen + 1):
                    bx += bdx
                    bz += bdz
                    by += (1 if step == blen else 0)
                    set_trunk(bx, by, bz, branch_log)
                # Crown around branch tip
                for c_dy in range(-1, 2):
                    for c_dz in range(-1, 2):
                        for c_dx in range(-1, 2):
                            if abs(c_dx) == 1 and abs(c_dz) == 1 and abs(c_dy) == 1:
                                continue
                            set_air(bx + c_dx, by + c_dy, bz + c_dz, tree.leaves)
            # Central crown
            for dy in range(-1, tree.leaf_up + 1):
                for dz in range(-tree.radius, tree.radius + 1):
                    for dx in range(-tree.radius, tree.radius + 1):
                        if dx == 0 and dz == 0 and dy <= 0:
                            continue
                        if abs(dx) == tree.radius and abs(dz) == tree.radius:
                            continue
                        set_air(lx + dx, top + dy, lz + dz, tree.leaves)

        elif shape in ("canopy", "flat", "acacia"):
            # Umbrella canopy tree
            for i in range(1, trunk + 1):
                set_trunk(lx, ground + i, lz, tree.trunk)
            top = ground + trunk
            # Top fork
            set_trunk(lx + 1, top, lz, tree.trunk)
            set_trunk(lx - 1, top, lz, tree.trunk)
            for dy in (0, 1):
                cur_r = tree.radius if dy == 0 else max(1, tree.radius - 1)
                for dz in range(-cur_r, cur_r + 1):
                    for dx in range(-cur_r, cur_r + 1):
                        if cur_r > 1 and abs(dx) == cur_r and abs(dz) == cur_r:
                            continue
                        set_air(lx + dx, top + dy, lz + dz, tree.leaves)

        else:
            # Default "blob" / standard oak shape
            for i in range(1, trunk + 1):
                set_trunk(lx, ground + i, lz, tree.trunk)
            top = ground + trunk
            for dy in range(-tree.leaf_down, tree.leaf_up + 1):
                for dz in range(-tree.radius, tree.radius + 1):
                    for dx in range(-tree.radius, tree.radius + 1):
                        if dx == 0 and dz == 0 and dy <= 0:
                            continue
                        manhattan = abs(dx) + abs(dz) + abs(dy)
                        if dy == tree.leaf_up and (abs(dx) > tree.top_radius or abs(dz) > tree.top_radius):
                            continue
                        if tree.skip_corners and abs(dx) == tree.radius and abs(dz) == tree.radius:
                            continue
                        if manhattan > tree.max_manhattan:
                            continue
                        if manhattan > tree.thin_above and hash2(wx + dx + 3, wz + dz + dy * 9, self.seed) > tree.thin_threshold:
                            continue
                        set_air(lx + dx, top + dy, lz + dz, tree.leaves)

    def _place_decorations(self, data, x0, y0, z0, sample):
        for lz in range(CHUNK_SIZE):
            wz = z0 + lz
            for lx in range(CHUNK_SIZE):
                wx = x0 + lx
                gy = sample(lx, lz)
                dec_y = gy + 1
                if dec_y < y0 or dec_y >= y0 + CHUNK_SIZE:
                    continue
                if gy <= WATER_LEVEL:
                    continue
                if data[dec_y - y0, lz, lx] != 0:
                    continue
                if gy >= y0:
                    surface_idx = data[gy - y0, lz, lx]
                    surface_block = block_from_index(surface_idx)
                    surface_id = surface_block.id if surface_block else ""
                else:
                    lower_chunk = self.chunks.get((x0 // CHUNK_SIZE, (y0 // CHUNK_SIZE) - 1, z0 // CHUNK_SIZE))
                    if lower_chunk is not None:
                        surface_idx = lower_chunk.data[gy - (y0 - CHUNK_SIZE), lz, lx]
                        surface_block = block_from_index(surface_idx)
                        surface_id = surface_block.id if surface_block else ""
                    else:
                        if any(cave.opens(wx, gy, wz, gy, self.seed) for cave in self.caves):
                            continue
                        biome = biome_at(wx, wz, self.seed, self.biomes)
                        surface_id = biome.surface

                if not surface_id:
                    continue

                biome = biome_at(wx, wz, self.seed, self.biomes)
                decorations = getattr(biome, "decorations", ())
                if not decorations:
                    continue

                roll = hash2(wx, wz, self.seed + 9876)
                accum = 0.0
                for dec in decorations:
                    accum += dec.chance
                    if roll < accum:
                        if surface_id in dec.surfaces:
                            chosen = dec.pick_block(wx, wz, self.seed)
                            b_idx = block_index(chosen)
                            if b_idx is not None:
                                data[dec_y - y0, lz, lx] = b_idx
                        break

    @property
    def spawn_point(self):
        return self.get_spawn_point()

    @spawn_point.setter
    def spawn_point(self, value):
        if value is None:
            self._spawn_point = None
        elif isinstance(value, dict):
            x = float(value.get("x", 0.5))
            z = float(value.get("z", 0.5))
            y = value.get("y")
            if y is None:
                y = float(self.column_top(int(math.floor(x)), int(math.floor(z))) + 1)
            self._spawn_point = (float(x), float(y), float(z))
        else:
            vals = list(value)
            if len(vals) == 2:
                x, z = float(vals[0]), float(vals[1])
                y = float(self.column_top(int(math.floor(x)), int(math.floor(z))) + 1)
                self._spawn_point = (x, y, z)
            else:
                self._spawn_point = (float(vals[0]), float(vals[1]), float(vals[2]))

    def get_spawn_point(self):
        if self._spawn_point is None:
            sx, sz, sy = self.find_spawn()
            self._spawn_point = (float(sx) + 0.5, float(sy), float(sz) + 0.5)
        return self._spawn_point

    def set_spawn_point(self, x, y=None, z=None):
        if y is not None and z is None:
            z = y
            y = None
        fx = float(x)
        fz = float(z if z is not None else 0.0)
        if y is None:
            fy = float(self.column_top(int(math.floor(fx)), int(math.floor(fz))) + 1)
        else:
            fy = float(y)
        self._spawn_point = (fx, fy, fz)
        return self._spawn_point

    def find_spawn(self, origin_x=0, origin_z=0):
        ox, oz = int(origin_x), int(origin_z)
        for radius in range(0, 33):
            for dz in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if radius != 0 and max(abs(dx), abs(dz)) != radius:
                        continue
                    x, z = ox + dx, oz + dz
                    if self._clear_spawn(x, z):
                        return x, z, terrain_height(x, z, self.seed, self.biomes) + 1
        fallback = None
        for radius in range(40, 2049, 8):
            for step in range(-radius, radius + 1, 8):
                for dx, dz in ((step, -radius), (step, radius), (-radius, step), (radius, step)):
                    x, z = ox + dx, oz + dz
                    if self._clear_spawn(x, z):
                        return x, z, terrain_height(x, z, self.seed, self.biomes) + 1
                    if fallback is None and self._valid_spawn_biome_and_height(x, z):
                        fallback = (x, z, terrain_height(x, z, self.seed, self.biomes) + 1)
        if fallback is not None:
            return fallback
        return ox, oz, terrain_height(ox, oz, self.seed, self.biomes) + 1

    def _valid_spawn_biome_and_height(self, x, z):
        height = terrain_height(x, z, self.seed, self.biomes)
        if height <= WATER_LEVEL:
            return False
        biome = biome_at(x, z, self.seed, self.biomes)
        if not getattr(biome, "valid_spawn", True):
            return False
        return True

    def _clear_spawn(self, x, z):
        if not self._valid_spawn_biome_and_height(x, z):
            return False
        height = terrain_height(x, z, self.seed, self.biomes)
        if _tree_site(x, z, height, self.seed, self.biomes, lambda px, pz: terrain_height(px, pz, self.seed, self.biomes)):
            return False
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if _tree_site(
                    x + dx,
                    z + dz,
                    terrain_height(x + dx, z + dz, self.seed, self.biomes),
                    self.seed,
                    self.biomes,
                    lambda px, pz: terrain_height(px, pz, self.seed, self.biomes),
                ):
                    return False
                if abs(terrain_height(x + dx, z + dz, self.seed, self.biomes) - height) > 1:
                    return False
        return True

    def raycast(self, ox, oy, oz, dx, dy, dz, max_dist, fluids=False, solid_only=False):
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length == 0:
            return None
        dx /= length
        dy /= length
        dz /= length
        x = math.floor(ox)
        y = math.floor(oy)
        z = math.floor(oz)
        step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
        step_z = 0 if dz == 0 else (1 if dz > 0 else -1)
        t_delta_x = math.inf if step_x == 0 else abs(1 / dx)
        t_delta_y = math.inf if step_y == 0 else abs(1 / dy)
        t_delta_z = math.inf if step_z == 0 else abs(1 / dz)
        t_max_x = math.inf if step_x == 0 else ((x + 1 - ox) / dx if step_x > 0 else (x - ox) / dx)
        t_max_y = math.inf if step_y == 0 else ((y + 1 - oy) / dy if step_y > 0 else (y - oy) / dy)
        t_max_z = math.inf if step_z == 0 else ((z + 1 - oz) / dz if step_z > 0 else (z - oz) / dz)
        nx = ny = nz = 0
        dist = 0.0
        limit = int(max_dist * 4) + 2
        for _ in range(limit):
            from server.fluid import is_fluid
            from server.blocks import get_offset_info, get_offset_torch_info, get_slab_info, is_offset_block, is_offset_torch, is_slab

            blk = self.get_block(x, y, z)
            above_blk = self.get_block(x, y + 1, z) if y + 1 < WORLD_HEIGHT else AIR
            above_is_offset = is_offset_block(above_blk)
            above_is_offset_torch = not solid_only and is_offset_torch(above_blk)

            if blk not in ("", AIR) or above_is_offset or above_is_offset_torch:
                boxes = []
                if blk not in ("", AIR):
                    is_f = is_fluid(blk)
                    is_s = self.is_solid(x, y, z)
                    s_info = get_slab_info(blk)
                    o_info = get_offset_info(blk)
                    t_info = get_offset_torch_info(blk)

                    if is_s or (fluids and is_f) or (not solid_only and not is_f):
                        b_obj = BLOCKS.get(blk)
                        if t_info is not None:
                            if not solid_only and b_obj is not None and b_obj.elements:
                                el = b_obj.elements[0]
                                fx0, fy0, fz0 = el["from"]
                                fx1, fy1, fz1 = el["to"]
                                boxes.append(
                                    (
                                        x,
                                        y,
                                        z,
                                        "offset_torch",
                                        (
                                            float(x) + max(0.0, (fx0 - 1.0) / 16.0),
                                            float(y) + (fy0 / 16.0),
                                            float(z) + max(0.0, (fz0 - 1.0) / 16.0),
                                            float(x) + min(1.0, (fx1 + 1.0) / 16.0),
                                            float(y) + min(1.0, (fy1 + 1.0) / 16.0),
                                            float(z) + min(1.0, (fz1 + 1.0) / 16.0),
                                        ),
                                    )
                                )
                            if t_info["capped"]:
                                boxes.append(
                                    (
                                        x,
                                        y,
                                        z,
                                        "top",
                                        (float(x), float(y) + 0.5, float(z), float(x + 1), float(y + 1), float(z + 1)),
                                    )
                                )
                        elif o_info is not None:
                            boxes.append(
                                (
                                    x,
                                    y,
                                    z,
                                    "offset",
                                    (float(x), float(y) - 0.5, float(z), float(x + 1), float(y) + 0.5, float(z + 1)),
                                )
                            )
                            if o_info["capped"]:
                                boxes.append(
                                    (
                                        x,
                                        y,
                                        z,
                                        "top",
                                        (float(x), float(y) + 0.5, float(z), float(x + 1), float(y + 1), float(z + 1)),
                                    )
                                )
                        elif s_info is not None:
                            if s_info["double"]:
                                boxes.append(
                                    (
                                        x,
                                        y,
                                        z,
                                        "bottom",
                                        (float(x), float(y), float(z), float(x + 1), float(y) + 0.5, float(z + 1)),
                                    )
                                )
                                boxes.append(
                                    (
                                        x,
                                        y,
                                        z,
                                        "top",
                                        (float(x), float(y) + 0.5, float(z), float(x + 1), float(y + 1), float(z + 1)),
                                    )
                                )
                            elif s_info["half"] == "bottom":
                                boxes.append(
                                    (
                                        x,
                                        y,
                                        z,
                                        "bottom",
                                        (float(x), float(y), float(z), float(x + 1), float(y) + 0.5, float(z + 1)),
                                    )
                                )
                            elif s_info["half"] == "top":
                                boxes.append(
                                    (
                                        x,
                                        y,
                                        z,
                                        "top",
                                        (float(x), float(y) + 0.5, float(z), float(x + 1), float(y + 1), float(z + 1)),
                                    )
                                )
                        elif b_obj is not None and b_obj.properties.get("torch") and b_obj.elements:
                            el = b_obj.elements[0]
                            fx0, fy0, fz0 = el["from"]
                            fx1, fy1, fz1 = el["to"]
                            boxes.append(
                                (
                                    x,
                                    y,
                                    z,
                                    "full",
                                    (
                                        float(x) + max(0.0, (fx0 - 1.0) / 16.0),
                                        float(y) + (fy0 / 16.0 if fy0 < 0.0 else max(0.0, fy0 / 16.0)),
                                        float(z) + max(0.0, (fz0 - 1.0) / 16.0),
                                        float(x) + min(1.0, (fx1 + 1.0) / 16.0),
                                        float(y) + min(1.0, (fy1 + 1.0) / 16.0),
                                        float(z) + min(1.0, (fz1 + 1.0) / 16.0),
                                    ),
                                )
                            )
                        else:
                            boxes.append(
                                (
                                    x,
                                    y,
                                    z,
                                    "full",
                                    (float(x), float(y), float(z), float(x + 1), float(y + 1), float(z + 1)),
                                )
                            )

                if above_is_offset:
                    boxes.append(
                        (
                            x,
                            y + 1,
                            z,
                            "offset",
                            (float(x), float(y) + 0.5, float(z), float(x + 1), float(y) + 1.5, float(z + 1)),
                        )
                    )
                elif above_is_offset_torch:
                    above_obj = BLOCKS.get(above_blk)
                    if above_obj is not None and above_obj.elements:
                        el = above_obj.elements[0]
                        fx0, fy0, fz0 = el["from"]
                        fx1, fy1, fz1 = el["to"]
                        boxes.append(
                            (
                                x,
                                y + 1,
                                z,
                                "offset_torch",
                                (
                                    float(x) + max(0.0, (fx0 - 1.0) / 16.0),
                                    float(y + 1) + (fy0 / 16.0),
                                    float(z) + max(0.0, (fz0 - 1.0) / 16.0),
                                    float(x) + min(1.0, (fx1 + 1.0) / 16.0),
                                    float(y + 1) + min(1.0, (fy1 + 1.0) / 16.0),
                                    float(z) + min(1.0, (fz1 + 1.0) / 16.0),
                                ),
                            )
                        )

                if boxes:
                    closest_hit = None
                    for tx, ty, tz, box_part, (bx0, by0, bz0, bx1, by1, bz1) in boxes:
                        hit_res = _intersect_ray_box(ox, oy, oz, dx, dy, dz, bx0, by0, bz0, bx1, by1, bz1)
                        if hit_res is not None:
                            t_hit, hx, hy, hz, b_norm = hit_res
                            if t_hit <= max_dist:
                                if closest_hit is None or t_hit < closest_hit[0]:
                                    closest_hit = (
                                        t_hit,
                                        hx,
                                        hy,
                                        hz,
                                        b_norm,
                                        tx,
                                        ty,
                                        tz,
                                        box_part,
                                        (bx0, by0, bz0, bx1, by1, bz1),
                                    )

                    if closest_hit is not None:
                        t_hit, hx, hy, hz, b_norm, tx, ty, tz, box_part, part_box = closest_hit
                        loc_y = hy - ty
                        if box_part in ("bottom", "top", "offset", "offset_torch"):
                            slab_half = box_part
                        else:
                            slab_half = "bottom" if loc_y < 0.5 else "top"
                        return {
                            "x": tx,
                            "y": ty,
                            "z": tz,
                            "nx": b_norm[0],
                            "ny": b_norm[1],
                            "nz": b_norm[2],
                            "dist": t_hit,
                            "hit_x": hx,
                            "hit_y": hy,
                            "hit_z": hz,
                            "local_y": loc_y,
                            "slab_half": slab_half,
                            "box": part_box,
                        }
            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                if t_max_x > max_dist:
                    return None
                x += step_x
                dist = t_max_x
                t_max_x += t_delta_x
                nx, ny, nz = -step_x, 0, 0
            elif t_max_y <= t_max_z:
                if t_max_y > max_dist:
                    return None
                y += step_y
                dist = t_max_y
                t_max_y += t_delta_y
                nx, ny, nz = 0, -step_y, 0
            else:
                if t_max_z > max_dist:
                    return None
                z += step_z
                dist = t_max_z
                t_max_z += t_delta_z
                nx, ny, nz = 0, 0, -step_z
        return None

    def raycast_entity(self, ox, oy, oz, dx, dy, dz, max_dist=4.0):
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length == 0:
            return None
        ndx, ndy, ndz = dx / length, dy / length, dz / length

        # Check if a solid block is closer than entity hit
        block_hit = self.raycast(ox, oy, oz, ndx, ndy, ndz, max_dist, solid_only=True)
        best_t = float(block_hit["dist"]) if block_hit is not None else float(max_dist)
        best_entity = None

        for entity in self.entities:
            if not getattr(entity, "is_mob", False) or not getattr(entity, "alive", True):
                continue

            r = float(getattr(entity, "radius", 0.45))
            h = float(getattr(entity, "height", 0.9))
            min_x, max_x = entity.pos[0] - r, entity.pos[0] + r
            min_y, max_y = entity.pos[1], entity.pos[1] + h
            min_z, max_z = entity.pos[2] - r, entity.pos[2] + r

            tmin = 0.0
            tmax = best_t

            # X slab
            if abs(ndx) > 1e-8:
                tx1 = (min_x - ox) / ndx
                tx2 = (max_x - ox) / ndx
                tmin = max(tmin, min(tx1, tx2))
                tmax = min(tmax, max(tx1, tx2))
            elif ox < min_x or ox > max_x:
                continue

            # Y slab
            if abs(ndy) > 1e-8:
                ty1 = (min_y - oy) / ndy
                ty2 = (max_y - oy) / ndy
                tmin = max(tmin, min(ty1, ty2))
                tmax = min(tmax, max(ty1, ty2))
            elif oy < min_y or oy > max_y:
                continue

            # Z slab
            if abs(ndz) > 1e-8:
                tz1 = (min_z - oz) / ndz
                tz2 = (max_z - oz) / ndz
                tmin = max(tmin, min(tz1, tz2))
                tmax = min(tmax, max(tz1, tz2))
            elif oz < min_z or oz > max_z:
                continue

            if tmax >= tmin and tmin < best_t:
                best_t = tmin
                best_entity = entity

        return best_entity

