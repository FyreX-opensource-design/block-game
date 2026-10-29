"""Tag system for block and item tagging in recipes and world generation."""

from pathlib import Path
import json
import re

from server.blocks import ASSETS, BLOCKS
from server.lua import load_table

_TAG_ID = re.compile(r"[a-z0-9_/]+")

BLOCK_TAGS = {}
ITEM_TAGS = {}


def _read_tag_file(path):
    if path.suffix == ".json":
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    elif path.suffix == ".lua":
        try:
            return load_table(path)
        except Exception:
            return None
    return None


def _parse_tag_entries(data):
    if isinstance(data, list):
        return [str(v) for v in data]
    if isinstance(data, dict):
        vals = data.get("values", data.get("entries", ()))
        if isinstance(vals, (list, tuple)):
            return [str(v) for v in vals]
    return []


def load_tags(root_folder=None):
    global BLOCK_TAGS, ITEM_TAGS
    root = Path(root_folder) if root_folder is not None else ASSETS / "tags"
    block_tags = {}
    item_tags = {}

    # Load block tags from assets/tags/blocks
    blocks_dir = root / "blocks"
    if blocks_dir.is_dir():
        for path in sorted(blocks_dir.iterdir()):
            data = _read_tag_file(path)
            if data is not None:
                tag_name = path.stem
                entries = _parse_tag_entries(data)
                block_tags.setdefault(tag_name, set()).update(entries)

    # Load item tags from assets/tags/items
    items_dir = root / "items"
    if items_dir.is_dir():
        for path in sorted(items_dir.iterdir()):
            data = _read_tag_file(path)
            if data is not None:
                tag_name = path.stem
                entries = _parse_tag_entries(data)
                item_tags.setdefault(tag_name, set()).update(entries)

    # Collect tags declared directly on BLOCKS
    for b_id, b_obj in BLOCKS.items():
        if hasattr(b_obj, "properties"):
            for t in b_obj.properties.get("tags", ()):
                block_tags.setdefault(str(t), set()).add(b_id)
                base = getattr(b_obj, "base_id", None)
                if base:
                    block_tags.setdefault(str(t), set()).add(base)

    # Resolve nested tags (e.g. "#small_flowers" inside "#flowers")
    for tag_map in (block_tags, item_tags):
        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            iterations += 1
            for tag_name, values in list(tag_map.items()):
                to_add = set()
                to_remove = set()
                for val in values:
                    if val.startswith("#"):
                        subtag = val[1:]
                        to_remove.add(val)
                        if subtag in tag_map:
                            to_add.update(tag_map[subtag])
                if to_remove or to_add:
                    values.difference_update(to_remove)
                    if not to_add.issubset(values):
                        values.update(to_add)
                        changed = True

    BLOCK_TAGS = {k: frozenset(v) for k, v in block_tags.items()}
    ITEM_TAGS = {k: frozenset(v) for k, v in item_tags.items()}
    return BLOCK_TAGS, ITEM_TAGS


def get_block_tag(tag_name):
    return BLOCK_TAGS.get(tag_name, frozenset())


def get_item_tag(tag_name):
    return ITEM_TAGS.get(tag_name, frozenset())


def has_block_tag(block_id, tag_name):
    if not block_id:
        return False
    block_set = BLOCK_TAGS.get(tag_name)
    if block_set and block_id in block_set:
        return True
    from server.blocks import canonical_id

    cid = canonical_id(block_id)
    if block_set and cid in block_set:
        return True
    b_obj = BLOCKS.get(block_id) or BLOCKS.get(cid)
    if b_obj is not None and hasattr(b_obj, "properties"):
        if tag_name in b_obj.properties.get("tags", ()):
            return True
    return False


def has_item_tag(item_id, tag_name):
    if not item_id:
        return False
    item_set = ITEM_TAGS.get(tag_name)
    if item_set and item_id in item_set:
        return True
    from server.items import ITEMS

    item_obj = ITEMS.get(item_id)
    if item_obj is not None and hasattr(item_obj, "tags"):
        if tag_name in item_obj.tags:
            return True
    return False


def matches_tag(id_val, tag_name):
    """Check if id_val matches a tag in blocks or items."""
    return has_item_tag(id_val, tag_name) or has_block_tag(id_val, tag_name)


def matches_ingredient(pattern_str, actual_id):
    """Matches an ingredient requirement against an item/block ID."""
    if not pattern_str:
        return not actual_id
    if not actual_id:
        return False
    if pattern_str == actual_id:
        return True
    if pattern_str.startswith("#"):
        return matches_tag(actual_id, pattern_str[1:])
    from server.blocks import BLOCKS

    p_base = (
        BLOCKS[pattern_str].base_id
        if pattern_str in BLOCKS and getattr(BLOCKS[pattern_str], "base_id", None)
        else pattern_str
    )
    a_base = (
        BLOCKS[actual_id].base_id
        if actual_id in BLOCKS and getattr(BLOCKS[actual_id], "base_id", None)
        else actual_id
    )
    return p_base == a_base


# Initialize tags on import
load_tags()
