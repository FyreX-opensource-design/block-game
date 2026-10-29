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

        for p in players:
            if not getattr(p, "pos", None):
                continue
            for _ in range(4):
                dist = random.uniform(self.min_spawn_dist, self.max_spawn_dist)
                angle = random.uniform(0, 2 * math.pi)
                x = math.floor(p.pos[0] + dist * math.cos(angle))
                z = math.floor(p.pos[2] + dist * math.sin(angle))

                # Find ground height
                y = self._find_ground_y(x, z)
                if y is None:
                    continue

                if self._can_spawn_at(x, y, z):
                    herd_size = random.randint(1, 3)
                    for _ in range(herd_size):
                        hx = x + random.randint(-2, 2)
                        hz = z + random.randint(-2, 2)
                        hy = self._find_ground_y(hx, hz)
                        if hy is not None and self._can_spawn_at(hx, hy, hz):
                            self.world.spawn_mob("pig", pos=[hx + 0.5, hy + 1.0, hz + 0.5])
                    return

    def _find_ground_y(self, x, z):
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

    def _can_spawn_at(self, x, y, z):
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

