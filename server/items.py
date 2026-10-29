import json
import re

import numpy as np

from server.blocks import ASSETS, BLOCKS, BY_INDEX, TABS, _block_name, _load_lang
from server.icons import ICON_SIZE, render_block_icon, render_flat_icon
from server.images import read_png

_ID = re.compile(r"[a-z0-9_]+")
ACTIONS = ("none", "break", "place", "use", "mine", "eat", "attack")


class Item:
    def __init__(
        self,
        item_id,
        name,
        model,
        block,
        texture,
        max_stack,
        left,
        right,
        durability,
        mining_level,
        mining_speed,
        attack,
        food,
        repair,
        tabs,
        tool=None,
    ):
        self.id = item_id
        self.name = name
        self.model = model
        self.block = block
        self.texture = texture
        self.max_stack = max_stack
        self.left = left
        self.right = right
        self.durability = durability
        self.mining_level = mining_level
        self.mining_speed = mining_speed
        self.attack = attack
        self.food = food
        self.repair = repair
        self.tabs = tabs
        self.tool = str(tool).lower() if tool and tool != "none" else None
        self.icon = (0.0, 0.0, 1.0, 1.0)


def canonical_item(value):
    from server.blocks import BLOCKS, canonical_id

    if value is None or value == "" or value == "air":
        return ""
    if isinstance(value, dict):
        value = value.get("id", "")
    if isinstance(value, str) and value in ITEMS:
        return value
    block_id = canonical_id(value)
    if block_id:
        block = BLOCKS.get(block_id)
        if block and getattr(block, "base_id", None) and block.base_id in ITEMS:
            return block.base_id
        if block_id in ITEMS:
            return block_id
    return ""


def stack_limit(item_id):
    item = ITEMS.get(item_id)
    if item is None:
        return 64
    return item.max_stack


def creative_items(tab_id):
    ordered = []
    for block in BY_INDEX:
        if block is None:
            continue
        item = ITEMS.get(block.id)
        if item is not None and tab_id in item.tabs:
            ordered.append(block.id)
    extras = sorted(item.id for item in ITEMS.values() if not item.block and tab_id in item.tabs)
    return tuple(ordered + extras)


def _load_items():
    folder = ASSETS / "items"
    paths = sorted(folder.glob("*.json"))
    if not paths:
        raise FileNotFoundError(folder)
    lang = _load_lang()
    tab_ids = {tab.id for tab in TABS if tab.screen == "blocks"}
    prepared = []
    seen = set()
    for path in paths:
        entry = json.loads(path.read_text())
        item_id = str(entry.get("id", ""))
        if path.stem != item_id or _ID.fullmatch(item_id) is None:
            raise ValueError(f"{path.name} must be named {item_id}.json")
        if item_id in seen:
            raise ValueError(f"duplicate item {item_id}")
        seen.add(item_id)
        prepared.append((entry, _block_name(entry, lang)))
    items = {}
    pictures = {}
    for entry, name in prepared:
        item = _item_from_entry(entry, name, tab_ids)
        items[item.id] = item
        pictures[item.id] = _picture(entry, item)
    missing = [block_id for block_id in BLOCKS if block_id not in items and not getattr(BLOCKS[block_id], "is_variant", False)]
    if missing:
        raise ValueError(f"{missing[0]} needs an item definition")
    for entry, _name in prepared:
        repair = str(entry.get("repair", ""))
        if repair and repair not in items:
            raise ValueError(f"{entry['id']} repairs with unknown item {repair}")
    for tab in TABS:
        if tab.icon and tab.icon not in items:
            raise ValueError(f"{tab.id} icon is not an item")
    atlas, rects = _pack(pictures, [str(entry["id"]) for entry, _name in prepared])
    for item_id, rect in rects.items():
        items[item_id].icon = rect
    return items, atlas


def _item_from_entry(entry, name, tab_ids):
    item_id = str(entry["id"])
    model = str(entry.get("model", "flat"))
    if model not in ("block", "flat", "axes"):
        raise ValueError(f"{item_id} model must be block, flat, or axes")
    block = str(entry.get("block", ""))
    if block and block not in BLOCKS:
        raise ValueError(f"{item_id} block {block} does not exist")
    if model == "block" and not block:
        raise ValueError(f"{item_id} needs a block")
    texture = str(entry.get("texture", ""))
    if model in ("flat", "axes") and not texture:
        raise ValueError(f"{item_id} needs a texture")
    if texture and not (ASSETS / "textures" / f"{texture}.png").is_file():
        raise ValueError(f"{item_id} texture {texture} was not found")
    left = str(entry.get("left", "none"))
    right = str(entry.get("right", "none"))
    if left not in ACTIONS or right not in ACTIONS:
        raise ValueError(f"{item_id} click action must be one of {', '.join(ACTIONS)}")
    if right == "place" and not block:
        raise ValueError(f"{item_id} places a block")
    max_stack = _whole(entry.get("max_stack", 64), item_id, "max_stack")
    durability = _whole(entry.get("durability", 0), item_id, "durability")
    mining_level = _whole(entry.get("mining_level", 0), item_id, "mining_level")
    if max_stack < 1 or durability < 0 or mining_level < 0:
        raise ValueError(f"{item_id} stack, durability, and mining level must stay positive")
    mining_speed = _number(entry.get("mining_speed", 1), item_id, "mining_speed")
    attack = _number(entry.get("attack", 1), item_id, "attack")
    food = _whole(entry.get("food", 0), item_id, "food")
    if mining_speed <= 0 or attack < 0 or food < 0:
        raise ValueError(f"{item_id} speed, attack, and food must stay usable")
    raw_tabs = entry.get("tab", [])
    if isinstance(raw_tabs, str):
        raw_tabs = [raw_tabs]
    tabs = []
    for tab_id in raw_tabs:
        if tab_id not in tab_ids:
            raise ValueError(f"{item_id} tab {tab_id} is not a creative tab")
        if tab_id not in tabs:
            tabs.append(str(tab_id))
    tool = entry.get("tool")
    if not tool:
        if "pickaxe" in item_id:
            tool = "pickaxe"
        elif "axe" in item_id:
            tool = "axe"
        elif "shovel" in item_id:
            tool = "shovel"
    return Item(
        item_id,
        name,
        model,
        block,
        texture,
        max_stack,
        left,
        right,
        durability,
        mining_level,
        mining_speed,
        attack,
        food,
        str(entry.get("repair", "")),
        tuple(tabs),
        tool=tool,
    )


def _picture(entry, item):
    if item.model in ("flat", "axes"):
        return render_flat_icon(read_png(ASSETS / "textures" / f"{item.texture}.png"))
    block_path = ASSETS / "blocks" / f"{item.block}.json"
    if not block_path.is_file():
        block = BLOCKS.get(item.block)
        if block and getattr(block, "base_id", None):
            block_path = ASSETS / "blocks" / f"{block.base_id}.json"
    block_entry = json.loads(block_path.read_text())
    textures = block_entry["textures"]
    top_name = textures.get("top") or textures.get("side") or item.texture
    side_name = textures.get("side") or top_name
    is_slab = (
        block_entry.get("type") == "slab"
        or "slab" in block_entry.get("model", "")
        or "slab" in getattr(item, "id", "")
    )
    height_scale = 0.5 if is_slab else 1.0
    return render_block_icon(
        read_png(ASSETS / "textures" / f"{top_name}.png"),
        read_png(ASSETS / "textures" / f"{side_name}.png"),
        height_scale=height_scale,
    )


def _pack(pictures, order):
    tile = ICON_SIZE
    columns = min(16, len(order))
    rows = (len(order) + columns - 1) // columns
    atlas = np.zeros((rows * tile, columns * tile, 4), dtype=np.uint8)
    rects = {}
    for index, item_id in enumerate(order):
        column = index % columns
        row = index // columns
        x0 = column * tile
        y0 = row * tile
        atlas[y0 : y0 + tile, x0 : x0 + tile] = np.flipud(pictures[item_id])
        rects[item_id] = (
            (x0 + 0.5) / atlas.shape[1],
            (y0 + 0.5) / atlas.shape[0],
            (x0 + tile - 0.5) / atlas.shape[1],
            (y0 + tile - 0.5) / atlas.shape[0],
        )
    return atlas, rects


def _whole(value, item_id, name):
    number = _number(value, item_id, name)
    if int(number) != number:
        raise ValueError(f"{item_id} {name} must be a whole number")
    return int(number)


def _number(value, item_id, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{item_id} {name} must be a number")
    return float(value)


ITEMS, ITEM_ATLAS = _load_items()


def item_atlas_bytes():
    image = np.ascontiguousarray(ITEM_ATLAS)
    return image.tobytes(), (image.shape[1], image.shape[0])
