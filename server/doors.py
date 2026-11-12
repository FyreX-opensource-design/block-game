"""Data-driven door definitions loaded from assets/doors."""

import json
from pathlib import Path
import re

from server.blocks import ASSETS

_ID = re.compile(r"[a-z0-9_]+")


class Door:
    """Represents a data-driven door type defined in assets/doors."""

    def __init__(
        self,
        door_id,
        name=None,
        sound="wood",
        textures=None,
        hardness=3.0,
        tool="axe",
        openable_by_hand=True,
        item=None,
        loot_table=None,
        open_sound=None,
        close_sound=None,
    ):
        self.id = str(door_id)
        self.name = str(name or door_id.replace("_", " ").title())
        self.sound = str(sound or "wood")
        self.textures = dict(textures or {})
        self.hardness = float(hardness)
        self.tool = str(tool or "axe")
        self.openable_by_hand = bool(openable_by_hand)
        self.item = str(item or door_id)
        self.loot_table = str(loot_table or f"blocks/{door_id}")
        self.open_sound = open_sound
        self.close_sound = close_sound


def load_doors(folder=None):
    """Load all door definitions from assets/doors/*.json."""
    root = Path(folder) if folder is not None else ASSETS / "doors"
    if not root.is_dir():
        return {}
    doors = {}
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        door_id = str(data.get("id") or path.stem)
        if _ID.fullmatch(door_id) is None:
            continue
        doors[door_id] = Door(
            door_id,
            name=data.get("name") or data.get("display"),
            sound=data.get("sound", "wood"),
            textures=data.get("textures"),
            hardness=data.get("hardness", 3.0),
            tool=data.get("tool", "axe"),
            openable_by_hand=data.get("openable_by_hand", True),
            item=data.get("item", door_id),
            loot_table=data.get("loot_table"),
            open_sound=data.get("open_sound"),
            close_sound=data.get("close_sound"),
        )
    return doors


DOORS = load_doors()


def get_door(door_or_block_id):
    """Return the Door instance for a door ID or block variant ID, or None."""
    if not door_or_block_id:
        return None
    from server.blocks import get_door_info
    info = get_door_info(door_or_block_id)
    base_id = info["base_id"] if info else str(door_or_block_id)
    return DOORS.get(base_id)


class Trapdoor:
    """Represents a data-driven trapdoor type defined in assets/trapdoors."""

    def __init__(
        self,
        trapdoor_id,
        name=None,
        sound="wood",
        textures=None,
        hardness=3.0,
        tool="axe",
        openable_by_hand=True,
        item=None,
        loot_table=None,
    ):
        self.id = str(trapdoor_id)
        self.name = str(name or trapdoor_id.replace("_", " ").title())
        self.sound = str(sound or "wood")
        self.textures = dict(textures or {})
        self.hardness = float(hardness)
        self.tool = str(tool or "axe")
        self.openable_by_hand = bool(openable_by_hand)
        self.item = str(item or trapdoor_id)
        self.loot_table = str(loot_table or f"blocks/{trapdoor_id}")


def load_trapdoors(folder=None):
    """Load all trapdoor definitions from assets/trapdoors/*.json."""
    root = Path(folder) if folder is not None else ASSETS / "trapdoors"
    if not root.is_dir():
        return {}
    trapdoors = {}
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        td_id = str(data.get("id") or path.stem)
        if _ID.fullmatch(td_id) is None:
            continue
        trapdoors[td_id] = Trapdoor(
            td_id,
            name=data.get("name") or data.get("display"),
            sound=data.get("sound", "wood"),
            textures=data.get("textures"),
            hardness=data.get("hardness", 3.0),
            tool=data.get("tool", "axe"),
            openable_by_hand=data.get("openable_by_hand", True),
            item=data.get("item", td_id),
            loot_table=data.get("loot_table"),
        )
    return trapdoors


TRAPDOORS = load_trapdoors()


def get_trapdoor(trapdoor_or_block_id):
    """Return the Trapdoor instance for a trapdoor ID or block variant ID, or None."""
    if not trapdoor_or_block_id:
        return None
    from server.blocks import get_trapdoor_info
    info = get_trapdoor_info(trapdoor_or_block_id)
    base_id = info["base_id"] if info else str(trapdoor_or_block_id)
    return TRAPDOORS.get(base_id)


