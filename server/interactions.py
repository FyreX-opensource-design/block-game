"""Data-driven tool right-clicks on blocks.

Every file in ``assets/interactions`` (``.json`` or ``.lua``) describes one
interaction. Files are tried in filename order and the first match wins::

    {
      "id": "till_soil",
      "tool": "hoe",
      "blocks": ["dirt", "grass", "mud", "muddy_grass"],
      "result": "farmland",
      "damage": 1,
      "require_clear_above": true
    }

``tool`` is the held item's tool type. ``item`` may be used instead of, or as
well as, ``tool``, and accepts an item id or a ``#tag``. ``blocks`` entries
may be block ids or ``#tags``.

When ``require_clear_above`` is true, the block above the target must be air
or a plant. Right-clicking that plant targets the block underneath.
"""

import json
import re
from pathlib import Path

from server.blocks import AIR, ASSETS, BLOCKS, WORLD_HEIGHT
from server.lua import load_table

_ID = re.compile(r"[a-z0-9_]+")


class Interaction:
    def __init__(
        self,
        interaction_id,
        result,
        blocks,
        tool=None,
        item=None,
        damage=1,
        require_clear_above=True,
        source=None,
    ):
        self.id = interaction_id
        self.result = result
        self.blocks = tuple(blocks)
        self.tool = tool
        self.item = item
        self.damage = int(damage)
        self.require_clear_above = bool(require_clear_above)
        self.file = str(source) if source else interaction_id

    def matches_block(self, block_id):
        if not block_id or block_id == AIR:
            return False
        from server.tags import matches_ingredient

        for entry in self.blocks:
            if matches_ingredient(entry, block_id):
                return True
        return False

    def matches_held(self, item):
        if self.tool:
            from server.loot import get_tool_type

            if get_tool_type(item) != self.tool:
                return False
        if self.item:
            from server.tags import matches_ingredient

            item_id = item if isinstance(item, str) else getattr(item, "id", "")
            if not matches_ingredient(self.item, item_id):
                return False
        return True


def load_interactions(folder=None):
    root = Path(folder) if folder is not None else ASSETS / "interactions"
    if not root.is_dir():
        return ()
    rules = []
    seen = set()
    for path in sorted(root.iterdir()):
        if path.suffix == ".json":
            fields = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".lua":
            fields = load_table(path)
        else:
            continue
        rule = interaction_from_mapping(fields, path)
        if rule.id in seen:
            raise ValueError(f"{path}: duplicate interaction id {rule.id}")
        seen.add(rule.id)
        rules.append(rule)
    return tuple(rules)


def interaction_from_mapping(fields, source):
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: interaction file must be an object")
    interaction_id = fields.get("id")
    if not isinstance(interaction_id, str) or _ID.fullmatch(interaction_id) is None:
        raise ValueError(f"{source}: interaction id must be a lowercase name")
    result = fields.get("result")
    if not isinstance(result, str) or result not in BLOCKS:
        raise ValueError(f"{source}: result must be a known block id, got {result!r}")
    raw_blocks = fields.get("blocks", fields.get("block"))
    if isinstance(raw_blocks, str):
        raw_blocks = [raw_blocks]
    if not isinstance(raw_blocks, (list, tuple)) or not raw_blocks:
        raise ValueError(f"{source}: blocks must be a non-empty list")
    blocks = [str(entry) for entry in raw_blocks]
    for entry in blocks:
        if entry.startswith("#"):
            if _ID.fullmatch(entry[1:]) is None:
                raise ValueError(f"{source}: invalid block tag {entry}")
        elif entry not in BLOCKS:
            raise ValueError(f"{source}: unknown block {entry}")

    tool = fields.get("tool")
    item = fields.get("item")
    if tool is not None:
        tool = str(tool)
        if _ID.fullmatch(tool) is None:
            raise ValueError(f"{source}: tool must be a lowercase name")
    if item is not None:
        item = str(item)
        if item.startswith("#"):
            if _ID.fullmatch(item[1:]) is None:
                raise ValueError(f"{source}: invalid item tag {item}")
        else:
            from server.items import ITEMS

            if item not in ITEMS:
                raise ValueError(f"{source}: unknown item {item}")
    if not tool and not item:
        raise ValueError(f"{source}: an interaction needs a tool or an item")

    damage = fields.get("damage", 1)
    if isinstance(damage, bool) or not isinstance(damage, int) or damage < 0:
        raise ValueError(f"{source}: damage must be a whole number of zero or more")
    return Interaction(
        interaction_id,
        result,
        blocks,
        tool=tool,
        item=item,
        damage=damage,
        require_clear_above=bool(fields.get("require_clear_above", True)),
        source=source,
    )


def _replaceable_cover(block_id):
    """Air and non-solid plants a tool can act through."""
    if not block_id or block_id == AIR:
        return True
    block = BLOCKS.get(block_id)
    if block is None or block.solid:
        return False
    return bool(
        block.plant
        or block.fluid_breakable
        or block.requires_soil
        or block.valid_ground is not None
    )


def _target_cell(world, interaction, x, y, z, block_id):
    if interaction.matches_block(block_id):
        if interaction.require_clear_above:
            if y + 1 >= WORLD_HEIGHT:
                return None
            if not _replaceable_cover(world.get_block(x, y + 1, z)):
                return None
        return (x, y, z)
    if (
        interaction.require_clear_above
        and y > 0
        and _replaceable_cover(block_id)
        and block_id not in (None, "", AIR)
        and interaction.matches_block(world.get_block(x, y - 1, z))
    ):
        return (x, y - 1, z)
    return None


def tool_interaction_at(world, item, x, y, z, interactions=None):
    """Return ``(x, y, z, interaction)`` for this right-click, or None."""
    if item is None:
        return None
    x, y, z = int(x), int(y), int(z)
    block_id = world.get_block(x, y, z)
    rules = INTERACTIONS if interactions is None else interactions
    for interaction in rules:
        if not interaction.matches_held(item):
            continue
        target = _target_cell(world, interaction, x, y, z, block_id)
        if target is not None:
            return (target[0], target[1], target[2], interaction)
    return None


INTERACTIONS = load_interactions()
