"""Paired crafting tables and their catalyst slots."""

from server.blocks import BLOCKS

_BESIDE = ((1, 0), (-1, 0), (0, 1), (0, -1))


def paired_catalyst_count(world, x, y, z):
    """Catalyst slots for a crafting table that has a twin beside it.

    A lone table keeps the ordinary grid. Two of the same table, sharing a
    side, open that grid plus the number of catalyst slots named on the block.
    """
    if world is None:
        return 0
    x, y, z = int(x), int(y), int(z)
    block_id = world.get_block(x, y, z)
    block = BLOCKS.get(block_id)
    if block is None or getattr(block, "station", None) != "crafting_table":
        return 0
    count = int(getattr(block, "catalysts", 0) or 0)
    if count <= 0:
        return 0
    base = block.base_id or block_id
    for dx, dz in _BESIDE:
        other_id = world.get_block(x + dx, y, z + dz)
        other = BLOCKS.get(other_id)
        if other is None or getattr(other, "station", None) != "crafting_table":
            continue
        if (other.base_id or other_id) == base:
            return count
    return 0
