"""Fire lit by lava. Flames spread a short way, then go out on their own."""

import random

import numpy as np

from server.blocks import AIR, BLOCKS, BY_INDEX
from server.tags import has_block_tag

_NEIGHBORS = (
    (0, 1, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 0, 1),
    (0, 0, -1),
    (0, -1, 0),
)
_WOOD_TAGS = ("logs", "planks", "leaves")
_WOOD_PIECES = ("fence", "door", "trapdoor", "sapling", "planks", "log", "wood", "leaves")


def is_flammable(block_id):
    """Wood, leaves, and plants can catch fire. Stone and fluids cannot."""
    if not block_id or block_id == AIR:
        return False
    block = BLOCKS.get(block_id)
    if block is None:
        return False
    if getattr(block, "plant", False):
        return True
    if any(has_block_tag(block_id, tag) for tag in _WOOD_TAGS):
        return True
    name = getattr(block, "base_id", "") or block_id
    return any(piece in name for piece in _WOOD_PIECES)


class FireSystem:
    """Tracks lava and the flames it starts. Every flame has a lifetime."""

    def __init__(self, world):
        self.world = world
        self.lava = {}
        self.flames = {}
        self._acc = 0.0
        self._cursor = 0
        self._lava_index = {}
        self._fire_index = {}
        self._indexed = False

    def _index(self):
        if self._indexed:
            return
        lava = {}
        fire = {}
        for index, block in enumerate(BY_INDEX):
            if block is None:
                continue
            props = getattr(block, "properties", None) or {}
            kind = props.get("lava")
            if kind:
                lava[int(index)] = str(kind)
            flame = props.get("fire")
            if flame:
                fire[int(index)] = str(flame)
        self._lava_index = lava
        self._fire_index = fire
        self._indexed = True

    def scan_chunk(self, chunk):
        """Remember lava and fire already stored in a generated or loaded chunk."""
        self._index()
        if not self._lava_index and not self._fire_index:
            return
        data = chunk.data
        wanted = np.fromiter(set(self._lava_index) | set(self._fire_index), dtype=np.int32)
        if wanted.size == 0:
            return
        hits = np.isin(data, wanted)
        if not np.any(hits):
            return
        x0 = chunk.cx * data.shape[2]
        y0 = chunk.cy * data.shape[0]
        z0 = chunk.cz * data.shape[1]
        for ly, lz, lx in zip(*np.nonzero(hits)):
            index = int(data[ly, lz, lx])
            pos = (x0 + int(lx), y0 + int(ly), z0 + int(lz))
            kind = self._lava_index.get(index)
            if kind:
                self.lava[pos] = kind
                continue
            flame = self._fire_index.get(index)
            if flame and pos not in self.flames:
                self.flames[pos] = (random.uniform(2.0, 5.0), flame)

    def on_block_set(self, x, y, z, block_id):
        self._index()
        pos = (int(x), int(y), int(z))
        block = BLOCKS.get(block_id)
        props = getattr(block, "properties", None) or {}
        kind = props.get("lava")
        if kind:
            self.lava[pos] = str(kind)
        else:
            self.lava.pop(pos, None)
        flame = props.get("fire")
        if flame:
            if pos not in self.flames:
                self.flames[pos] = (random.uniform(3.0, 6.5), str(flame))
        else:
            self.flames.pop(pos, None)

    def update(self, dt):
        if not getattr(self.world, "simulate_blocks", True):
            return []
        self._acc += max(0.0, float(dt))
        if self._acc < 0.4:
            return []
        step = self._acc
        self._acc = 0.0
        changed = []
        for pos in list(self.flames):
            life, kind = self.flames[pos]
            if self.world.get_block(*pos) != kind:
                self.flames.pop(pos, None)
                continue
            life -= step
            if life <= 0.0:
                self.flames.pop(pos, None)
                if self.world.set_block(*pos, AIR):
                    changed.append((*pos, AIR))
                continue
            self.flames[pos] = (life, kind)
            if life > 1.4 and random.random() < 0.28:
                self._spread(pos, kind, changed)
        if self.lava:
            keys = list(self.lava)
            count = min(16, len(keys))
            start = self._cursor % len(keys)
            self._cursor += count
            for offset in range(count):
                pos = keys[(start + offset) % len(keys)]
                kind = self.lava.get(pos)
                if kind and self.world.get_block(*pos) not in ("", AIR):
                    self._ignite_around(pos, kind, changed)
        return changed

    def _spread(self, pos, kind, changed):
        order = list(_NEIGHBORS)
        random.shuffle(order)
        for dx, dy, dz in order:
            spot = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
            if self._catch(spot, kind, changed, life=random.uniform(1.2, 2.6)):
                return

    def _ignite_around(self, pos, kind, changed):
        for dx, dy, dz in _NEIGHBORS:
            self._catch((pos[0] + dx, pos[1] + dy, pos[2] + dz), kind, changed, life=random.uniform(2.5, 5.5))

    def _catch(self, pos, kind, changed, life):
        block_id = self.world.get_block(*pos)
        if not is_flammable(block_id):
            return False
        block = BLOCKS.get(block_id)
        if block is not None and not block.solid:
            return self._light(pos, kind, life, changed)
        x, y, z = pos
        for dx, dy, dz in ((0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)):
            spot = (x + dx, y + dy, z + dz)
            if self.world.get_block(*spot) == AIR:
                return self._light(spot, kind, life, changed)
        return False

    def _light(self, pos, kind, life, changed):
        if kind not in BLOCKS:
            return False
        if pos in self.flames and self.world.get_block(*pos) == kind:
            return False
        self.flames[pos] = (float(life), kind)
        if self.world.set_block(*pos, kind):
            changed.append((*pos, kind))
            return True
        self.flames.pop(pos, None)
        return False
