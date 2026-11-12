import math
import random

from server.blocks import BLOCKS, CHUNK_SIZE


class MobSpawner:
    """Spawning and despawning system for world mobs."""

    def __init__(self, world):
        self.world = world
        self.spawn_timer = 0.0
        self.max_mobs = 20
        self.despawn_dist_immediate = 96.0
        self.despawn_dist_chance = 64.0
        self.min_spawn_dist = 24.0
        self.max_spawn_dist = 54.0

    def _get_players(self):
        players = list(self.world.players.values())
        if not players and getattr(self.world, "_singleplayer_player", None) is not None:
            players = [self.world._singleplayer_player]
        return players

    def update(self, dt):
        players = self._get_players()

        # 1. Despawn checks
        if players:
            for entity in self.world.entities:
                if not getattr(entity, "is_mob", False) or not getattr(entity, "alive", True):
                    continue
                min_dist = min(
                    math.hypot(entity.pos[0] - p.pos[0], entity.pos[2] - p.pos[2])
                    for p in players
                )
                if min_dist > self.despawn_dist_immediate:
                    entity.alive = False
                elif min_dist > self.despawn_dist_chance:
                    entity.idle_time += dt
                    if entity.idle_time > 30.0 and random.random() < 0.05:
                        entity.alive = False

        # 2. Spawning cycle
        self.spawn_timer += dt
        if self.spawn_timer >= 2.0:
            self.spawn_timer = 0.0
            self.try_spawn(players)

    def try_spawn(self, players=None):
        if players is None:
            players = self._get_players()
        if not players:
            return

        current_mobs = sum(
            1 for e in self.world.entities if getattr(e, "is_mob", False) and getattr(e, "alive", True)
        )
        if current_mobs >= self.max_mobs:
            return

        is_night = False
        if hasattr(self.world, "is_night"):
            is_night = self.world.is_night()
        elif hasattr(self.world, "day_time"):
            t = self.world.day_time % 24000.0
            is_night = 12000.0 <= t <= 24000.0

        for p in players:
            if not getattr(p, "pos", None):
                continue
            p_y = p.pos[1]
            for _ in range(8):
                if current_mobs >= self.max_mobs:
                    return

                dist = random.uniform(self.min_spawn_dist, self.max_spawn_dist)
                angle = random.uniform(0, 2 * math.pi)
                x = math.floor(p.pos[0] + dist * math.cos(angle))
                z = math.floor(p.pos[2] + dist * math.sin(angle))

                ground_ys = self._find_ground_y_candidates(x, z, player_y=p_y)
                if not ground_ys:
                    continue

                spawned = False
                for y in ground_ys:
                    # 1. Zombie spawn: valid in dark caves (day or night) or surface at night
                    if self.can_spawn_zombie_at(x, y, z):
                        group_size = random.randint(1, 2)
                        for _ in range(group_size):
                            if current_mobs >= self.max_mobs:
                                break
                            hx = x + random.randint(-2, 2)
                            hz = z + random.randint(-2, 2)
                            hy = self._find_ground_near(hx, hz, near_y=y)
                            if hy is not None and self.can_spawn_zombie_at(hx, hy, hz):
                                self.world.spawn_mob("zombie", pos=[hx + 0.5, hy + 1.0, hz + 0.5])
                                current_mobs += 1
                                spawned = True
                        if spawned:
                            break

                    # 2. Passive mob spawn: valid on grassy surface in daylight
                    elif not is_night and self._can_spawn_at(x, y, z):
                        herd_size = random.randint(1, 3)
                        mob_type = random.choice(("pig", "sheep"))
                        for _ in range(herd_size):
                            if current_mobs >= self.max_mobs:
                                break
                            hx = x + random.randint(-2, 2)
                            hz = z + random.randint(-2, 2)
                            hy = self._find_ground_near(hx, hz, near_y=y)
                            if hy is not None and self._can_spawn_at(hx, hy, hz):
                                self.world.spawn_mob(mob_type, pos=[hx + 0.5, hy + 1.0, hz + 0.5])
                                current_mobs += 1
                                spawned = True
                        if spawned:
                            break

                if spawned:
                    return

    def _find_ground_near(self, x, z, near_y, max_dy=4):
        """Find a valid ground y near near_y with 2 headroom blocks."""
        from server.fluid import is_fluid

        for y in range(near_y + max_dy, near_y - max_dy - 1, -1):
            block = self.world.get_block(x, y, z)
            if not block or block == "air" or is_fluid(block):
                continue
            b_obj = BLOCKS.get(block)
            if not b_obj or not getattr(b_obj, "solid", False):
                continue

            b1 = self.world.get_block(x, y + 1, z)
            b2 = self.world.get_block(x, y + 2, z)
            if is_fluid(b1) or is_fluid(b2):
                continue
            b1_obj = BLOCKS.get(b1)
            b2_obj = BLOCKS.get(b2)
            if b1_obj and getattr(b1_obj, "solid", False):
                continue
            if b2_obj and getattr(b2_obj, "solid", False):
                continue
            return y
        return None

    def _find_ground_y_candidates(self, x, z, player_y=None):
        """Find candidate ground Y levels in column (x, z) with 2 blocks headroom.
        Includes cave floors and surface levels, prioritizing proximity to player_y.
        """
        cx = x // CHUNK_SIZE
        cz = z // CHUNK_SIZE

        loaded_sections = [
            cy for (scx, cy, scz) in self.world.chunks.keys() if scx == cx and scz == cz
        ]
        if not loaded_sections:
            return []

        max_sec_y = max(loaded_sections)
        min_sec_y = min(loaded_sections)
        top_y = (max_sec_y + 1) * CHUNK_SIZE - 1
        bottom_y = min_sec_y * CHUNK_SIZE

        from server.fluid import is_fluid

        if player_y is not None:
            scan_min = max(bottom_y, int(player_y - 36))
            scan_max = min(top_y, int(player_y + 36))
        else:
            scan_min = bottom_y
            scan_max = top_y

        candidates = []
        for y in range(scan_max, scan_min - 1, -1):
            block = self.world.get_block(x, y, z)
            if not block or block == "air" or is_fluid(block):
                continue
            b_obj = BLOCKS.get(block)
            if not b_obj or not getattr(b_obj, "solid", False):
                continue

            b1 = self.world.get_block(x, y + 1, z)
            b2 = self.world.get_block(x, y + 2, z)
            if is_fluid(b1) or is_fluid(b2):
                continue
            b1_obj = BLOCKS.get(b1)
            b2_obj = BLOCKS.get(b2)
            if b1_obj and getattr(b1_obj, "solid", False):
                continue
            if b2_obj and getattr(b2_obj, "solid", False):
                continue

            candidates.append(y)

        # Fall back to full column if none found in focused player vertical range
        if player_y is not None and not candidates:
            for y in range(top_y, bottom_y, -1):
                block = self.world.get_block(x, y, z)
                if not block or block == "air" or is_fluid(block):
                    continue
                b_obj = BLOCKS.get(block)
                if not b_obj or not getattr(b_obj, "solid", False):
                    continue
                b1 = self.world.get_block(x, y + 1, z)
                b2 = self.world.get_block(x, y + 2, z)
                if is_fluid(b1) or is_fluid(b2):
                    continue
                b1_obj = BLOCKS.get(b1)
                b2_obj = BLOCKS.get(b2)
                if b1_obj and getattr(b1_obj, "solid", False):
                    continue
                if b2_obj and getattr(b2_obj, "solid", False):
                    continue
                candidates.append(y)

        if player_y is not None and candidates:
            candidates.sort(key=lambda y: abs(y - player_y))

        return candidates

    def _find_ground_y(self, x, z, near_y=None):
        if near_y is not None:
            gy = self._find_ground_near(x, z, near_y)
            if gy is not None:
                return gy

        # Look for top solid block in loaded chunks
        cx = x // CHUNK_SIZE
        cz = z // CHUNK_SIZE

        # Check if column is loaded
        loaded_sections = [
            cy for (scx, cy, scz) in self.world.chunks.keys() if scx == cx and scz == cz
        ]
        if not loaded_sections:
            return None

        max_sec_y = max(loaded_sections)
        min_sec_y = min(loaded_sections)
        top_y = (max_sec_y + 1) * CHUNK_SIZE - 1
        bottom_y = min_sec_y * CHUNK_SIZE

        for y in range(top_y, bottom_y, -1):
            block = self.world.get_block(x, y, z)
            if block and block != "air":
                b_obj = BLOCKS.get(block)
                if b_obj and b_obj.solid:
                    return y
        return None

    def _can_spawn_at(self, x, y, z, mob_type=None):
        if mob_type == "zombie":
            return self.can_spawn_zombie_at(x, y, z)

        ground = self.world.get_block(x, y, z)
        if ground not in ("grass", "dirt", "muddy_grass"):
            return False

        # Above must be clear
        b1 = self.world.get_block(x, y + 1, z)
        b2 = self.world.get_block(x, y + 2, z)

        from server.fluid import is_water

        if is_water(b1) or is_water(b2):
            return False

        b1_obj = BLOCKS.get(b1)
        b2_obj = BLOCKS.get(b2)
        if b1_obj and b1_obj.solid:
            return False
        if b2_obj and b2_obj.solid:
            return False

        # Lighting check
        if getattr(self.world, "lighting", None) is not None:
            skylight = self.world.lighting.get_skylight(x, y + 1, z)
            if skylight < 7:
                return False

        return True

    def can_spawn_zombie_at(self, x, y, z, skylight=None, blocklight=None):
        """Check if a zombie can spawn at (x, y, z) ground block.

        Conditions:
        - Solid ground (not air, not fluid, solid block).
        - Clear headroom at y+1 and y+2 (not solid, not water).
        - Spawning is prevented if skylight > 7.
        - Spawning is prevented if block light blue > 7, green > 7, or uv > 7.
        - Red light (and infrared) does NOT prevent zombie spawning.
        """
        ground = self.world.get_block(x, y, z)
        if not ground or ground == "air":
            return False

        from server.fluid import is_fluid

        if is_fluid(ground):
            return False

        b_obj = BLOCKS.get(ground)
        if not b_obj or not getattr(b_obj, "solid", False):
            return False

        # Above must be clear (2 blocks headroom)
        b1 = self.world.get_block(x, y + 1, z)
        b2 = self.world.get_block(x, y + 2, z)

        if is_fluid(b1) or is_fluid(b2):
            return False

        b1_obj = BLOCKS.get(b1)
        b2_obj = BLOCKS.get(b2)
        if b1_obj and getattr(b1_obj, "solid", False):
            return False
        if b2_obj and getattr(b2_obj, "solid", False):
            return False

        # Lighting check:
        # Spawning is prevented by skylight > 7 or blue, green, or UV light > 7.
        # Red light won't work to prevent zombie spawning.
        if skylight is None:
            if hasattr(self.world, "get_current_skylight"):
                skylight = self.world.get_current_skylight(x, y + 1, z)
            elif getattr(self.world, "lighting", None) is not None:
                skylight = self.world.lighting.get_skylight(x, y + 1, z)
            elif hasattr(self.world, "get_skylight"):
                skylight = self.world.get_skylight(x, y + 1, z)
            else:
                skylight = 0

        if skylight > 7:
            return False

        if blocklight is None:
            if hasattr(self.world, "get_blocklight"):
                blocklight = self.world.get_blocklight(x, y + 1, z)
            elif getattr(self.world, "lighting", None) is not None:
                blocklight = self.world.lighting.get_blocklight(x, y + 1, z)
            else:
                blocklight = (0, 0, 0, 0, 0)

        if isinstance(blocklight, dict):
            g = blocklight.get("g", blocklight.get("green", 0))
            b = blocklight.get("b", blocklight.get("blue", 0))
            uv = blocklight.get("uv", 0)
        elif isinstance(blocklight, (list, tuple)):
            g = blocklight[1] if len(blocklight) > 1 else 0
            b = blocklight[2] if len(blocklight) > 2 else 0
            uv = blocklight[3] if len(blocklight) > 3 else 0
        else:
            g = b = uv = 0

        if b > 7 or g > 7 or uv > 7:
            return False

        return True

    can_spawn_at = _can_spawn_at

