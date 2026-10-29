"""Leaf decay system matching Minecraft mechanics.

Leaves generated with trees slowly and randomly decay when disconnected from logs,
dropping their loot table. Leaves placed by players do not decay.
"""

import collections
import random

from server.blocks import AIR, BLOCKS
from server.loot import get_block_drops
from server.tags import has_block_tag

MAX_LEAF_DISTANCE = 4


def is_leaf(block_id):
    if not block_id or block_id == AIR:
        return False
    b = BLOCKS.get(block_id)
    if b is None:
        return False
    return has_block_tag(block_id, "leaves") or "leaves" in block_id or b.properties.get("leaf", False)


def is_decayable_leaf(block_id):
    if not block_id or block_id == AIR:
        return False
    b = BLOCKS.get(block_id)
    if b is None:
        return False
    if not is_leaf(block_id):
        return False
    if b.properties.get("persistent", False) or block_id.endswith("_persistent"):
        return False
    return True


def is_log(block_id):
    if not block_id or block_id == AIR:
        return False
    return has_block_tag(block_id, "logs") or "log" in block_id


def is_leaf_supported(world, x, y, z, max_dist=MAX_LEAF_DISTANCE):
    """BFS search from leaf (x, y, z) through leaves to find a log within max_dist steps."""
    queue = collections.deque([(x, y, z, 0)])
    visited = {(x, y, z)}

    while queue:
        cx, cy, cz, dist = queue.popleft()
        if dist >= max_dist:
            continue
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if (nx, ny, nz) in visited:
                continue
            nbr = world.get_block(nx, ny, nz)
            if not nbr or nbr == AIR:
                continue
            if is_log(nbr):
                return True
            if is_leaf(nbr):
                visited.add((nx, ny, nz))
                queue.append((nx, ny, nz, dist + 1))

    return False


class LeafDecaySystem:
    def __init__(self, world):
        self.world = world
        self.pending = set()
        self.decay_timers = {}  # (x, y, z) -> remaining_time
        self.tick_timer = 0.0
        self.tick_interval = 0.1  # tick every 100ms

    def schedule_check(self, x, y, z):
        if is_decayable_leaf(self.world.get_block(x, y, z)):
            self.pending.add((x, y, z))

    def schedule_neighbors(self, x, y, z, radius=4):
        """Schedule all decayable leaves in range around (x, y, z)."""
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    pos = (x + dx, y + dy, z + dz)
                    block = self.world.get_block(*pos)
                    if is_decayable_leaf(block):
                        self.pending.add(pos)

    def update(self, dt):
        self.tick_timer += dt
        if self.tick_timer < self.tick_interval:
            return []

        step_dt = self.tick_timer
        self.tick_timer = 0.0
        decayed = []

        # 1. Evaluate up to 32 pending leaf checks per tick
        if self.pending:
            batch = list(self.pending)[:32]
            for pos in batch:
                self.pending.discard(pos)
                block = self.world.get_block(*pos)
                if not is_decayable_leaf(block):
                    continue
                if not is_leaf_supported(self.world, *pos):
                    if pos not in self.decay_timers:
                        # Random delay between 0.3s and 2.5s for natural staggered decay
                        self.decay_timers[pos] = random.uniform(0.3, 2.5)

        # 2. Advance decay timers
        ready = []
        for pos, remaining in list(self.decay_timers.items()):
            remaining -= step_dt
            if remaining <= 0:
                ready.append(pos)
                del self.decay_timers[pos]
            else:
                self.decay_timers[pos] = remaining

        # 3. Decay ready leaves
        for x, y, z in ready:
            block = self.world.get_block(x, y, z)
            if not is_decayable_leaf(block):
                continue
            # Re-check support before breaking
            if is_leaf_supported(self.world, x, y, z):
                continue

            b_obj = BLOCKS.get(block)
            if self.world.set_block(x, y, z, AIR):
                decayed.append((x, y, z))
                # Spawn loot drops
                if b_obj is not None:
                    drops = get_block_drops(b_obj)
                    for drop_id, count in drops:
                        if count > 0:
                            self.world.spawn_item(
                                drop_id, count=count, pos=(x + 0.5, y + 0.3, z + 0.5)
                            )
                # Schedule immediate 6 neighbors to cascade decay
                for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    npos = (x + dx, y + dy, z + dz)
                    nblock = self.world.get_block(*npos)
                    if is_decayable_leaf(nblock):
                        self.pending.add(npos)

        return decayed

