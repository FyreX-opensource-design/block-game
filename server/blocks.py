import json
import math
import re
from pathlib import Path

import numpy as np

from server.images import read_png

AIR = "air"
GRASS = "grass"
DIRT = "dirt"
STONE = "stone"
SAND = "sand"
SANDSTONE = "sandstone"
SMOOTH_SANDSTONE = "smooth_sandstone"
LOG = "log"
LEAVES = "leaves"
WATER = "water"
SALT_WATER = "salt_water"
BRACKISH_WATER = "brackish_water"
GRAVEL = "gravel"
CLAY = "clay"
CRAFTING_TABLE = "crafting_table"
FURNACE = "furnace"
COAL_ORE = "coal_ore"
COPPER_ORE = "copper_ore"
IRON_ORE = "iron_ore"
CHUNK_SIZE = 16
WORLD_HEIGHT = 128
SECTION_COUNT = WORLD_HEIGHT // CHUNK_SIZE
LOAD_RADIUS = 3
UNLOAD_RADIUS = 5
VERTICAL_RADIUS = SECTION_COUNT
VERTICAL_UNLOAD = SECTION_COUNT
STONE_BASE = 70
WATER_LEVEL = 64
SAND_LEVEL = STONE_BASE + 8
HOTBAR_SLOTS = 9
_ID = re.compile(r"[a-z0-9_]+")

FACE_INDEX = {
    "east": 0,
    "west": 1,
    "up": 2,
    "down": 3,
    "south": 4,
    "north": 5,
    "cross_a": 6,
    "cross_b": 7,
}
ASSETS = Path(__file__).resolve().parents[1] / "assets"


class Tab:
    def __init__(self, tab_id, name, icon, screen="blocks"):
        self.id = tab_id
        self.name = name
        self.icon = icon
        self.screen = screen


class Block:
    def __init__(
        self,
        block_id,
        name,
        solid,
        full_cube,
        face_uvs,
        elements,
        cutout=False,
        opacity=1.0,
        tabs=(),
        properties=None,
        base_id=None,
        is_variant=False,
        hardness=1.0,
        tool=None,
        mining_level=0,
        loot_table=None,
        emission=(0, 0, 0, 0, 0),
        light=None,
        gravity=False,
        fluid_breakable=False,
        soil=False,
        plant=False,
        requires_soil=False,
        valid_ground=None,
        tint=None,
        station=None,
        recipe_types=None,
        cook_speed=1.0,
        slab=False,
        slab_half=None,
        waterlogged=False,
        double_slab=False,
    ):
        from server.lighting import parse_light_levels

        self.id = block_id
        self.name = name
        self.solid = solid
        self.full_cube = full_cube
        self.face_uvs = face_uvs
        self.elements = elements
        self.cutout = cutout
        self.opacity = opacity
        self.tabs = tuple(tabs)
        self.properties = dict(properties or {})
        self.base_id = base_id or block_id
        self.is_variant = is_variant
        self.hardness = float(hardness)
        self.tool = str(tool).lower() if tool and tool != "none" else None
        self.mining_level = int(mining_level)
        self.loot_table = loot_table
        self.gravity = bool(gravity)
        self.fluid_breakable = bool(fluid_breakable)
        self.soil = bool(soil)
        self.plant = bool(plant)
        self.requires_soil = bool(requires_soil)
        self.valid_ground = tuple(valid_ground) if valid_ground is not None else None
        actual_tint = tint if tint is not None else self.properties.get("tint")
        if actual_tint is None and block_id in ("grass", "muddy_grass"):
            actual_tint = "grass"
        self.tint = str(actual_tint).lower() if actual_tint else None
        if self.tint:
            self.properties["tint"] = self.tint
        self.slab = bool(slab or self.properties.get("slab", False) or self.properties.get("double_slab", False))
        self.slab_half = slab_half or self.properties.get("half", ("bottom" if self.slab else None))
        self.waterlogged = bool(waterlogged or self.properties.get("waterlogged", False))
        self.double_slab = bool(double_slab or self.properties.get("double_slab", False))
        if self.slab:
            self.properties["slab"] = True
            if self.slab_half:
                self.properties["half"] = self.slab_half
        if self.waterlogged:
            self.properties["waterlogged"] = True
        if self.double_slab:
            self.properties["double_slab"] = True

        actual_station = station if station is not None else (self.properties.get("station") or self.properties.get("container"))
        if actual_station is None:
            if block_id == "crafting_table" or self.base_id == "crafting_table":
                actual_station = "crafting_table"
            elif block_id == "furnace" or self.base_id == "furnace":
                actual_station = "furnace"
        self.station = str(actual_station).lower() if actual_station else None
        if self.station:
            self.properties["station"] = self.station

        r_types = recipe_types if recipe_types is not None else (self.properties.get("recipe_types") or self.properties.get("recipe_type"))
        if r_types is not None:
            self.recipe_types = (str(r_types),) if isinstance(r_types, str) else tuple(str(t) for t in r_types)
        elif self.station == "crafting_table":
            self.recipe_types = ("crafting_shaped", "crafting_shapeless")
        elif self.station == "furnace":
            self.recipe_types = ("smelting",)
        else:
            self.recipe_types = None
        if self.recipe_types:
            self.properties["recipe_types"] = self.recipe_types

        c_speed = cook_speed if cook_speed != 1.0 else (self.properties.get("cook_speed") or 1.0)
        self.cook_speed = float(c_speed)
        if self.cook_speed != 1.0:
            self.properties["cook_speed"] = self.cook_speed
        self.emission = parse_light_levels(light if light is not None else emission)
        self.light = self.emission
        source = face_uvs[FACE_INDEX["up"]] or next(uv for uv in face_uvs if uv is not None)
        self.icon = (
            min(point[0] for point in source),
            min(point[1] for point in source),
            max(point[0] for point in source),
            max(point[1] for point in source),
        )


def _rotate_uv(uv, turns=1):
    if uv is None:
        return None
    turns = turns % 4
    return tuple(uv[(i + turns) % 4] for i in range(4))


OPPOSITE_FACING = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}


def is_slab(block_or_id):
    if not block_or_id:
        return False
    if isinstance(block_or_id, str):
        block = BLOCKS.get(block_or_id)
    elif isinstance(block_or_id, Block):
        block = block_or_id
    else:
        return False
    if block is None:
        return False
    return bool(block.properties.get("slab") or block.properties.get("double_slab") or getattr(block, "slab", False))


def is_waterlogged_slab(block_or_id):
    if not block_or_id:
        return False
    if isinstance(block_or_id, str):
        block = BLOCKS.get(block_or_id)
    elif isinstance(block_or_id, Block):
        block = block_or_id
    else:
        return False
    if block is None:
        return False
    return bool(block.properties.get("waterlogged") or getattr(block, "waterlogged", False))


def get_slab_info(block_or_id):
    """Returns slab metadata dictionary or None if not a slab:
    {
        "is_slab": True,
        "double": bool,
        "base_id": str,
        "half": 'bottom' | 'top' | 'double',
        "waterlogged": bool,
        "bottom": str,
        "top": str,
    }
    """
    if not block_or_id:
        return None
    if isinstance(block_or_id, str):
        block = BLOCKS.get(block_or_id)
    elif isinstance(block_or_id, Block):
        block = block_or_id
    else:
        return None
    if block is None:
        return None
    props = block.properties
    if not (props.get("slab") or props.get("double_slab") or getattr(block, "slab", False)):
        return None

    is_double = bool(props.get("double_slab") or getattr(block, "double_slab", False))
    half = "double" if is_double else props.get("half", getattr(block, "slab_half", "bottom"))
    waterlogged = bool(props.get("waterlogged") or getattr(block, "waterlogged", False))
    level = int(props.get("level", 8 if waterlogged else 0))
    base_id = getattr(block, "base_id", block.id)
    for sfx in ("_top_waterlogged", "_waterlogged"):
        if sfx in base_id:
            idx = base_id.find(sfx)
            base_id = base_id[:idx]
            break
    if base_id.endswith("_top"):
        base_id = base_id[:-4]

    bottom = props.get("bottom", base_id)
    top = props.get("top", base_id)

    return {
        "is_slab": True,
        "double": is_double,
        "base_id": base_id,
        "half": half,
        "waterlogged": waterlogged,
        "bottom": bottom,
        "top": top,
        "level": level,
    }


def get_waterlogged_slab(base_id, half="bottom", level=8):
    s_info = get_slab_info(base_id)
    b_id = s_info["base_id"] if s_info else base_id
    if level >= 8:
        target = f"{b_id}_top_waterlogged" if half == "top" else f"{b_id}_waterlogged"
    elif level > 0:
        target = f"{b_id}_top_waterlogged_{level}" if half == "top" else f"{b_id}_waterlogged_{level}"
    else:
        target = f"{b_id}_top" if half == "top" else b_id
    return target if target in BLOCKS else (f"{b_id}_top" if half == "top" else b_id)


def get_double_slab(bottom_id, top_id):
    b_info = get_slab_info(bottom_id)
    b_base = b_info["base_id"] if b_info else bottom_id
    t_info = get_slab_info(top_id)
    t_base = t_info["base_id"] if t_info else top_id

    candidate1 = f"double_slab_{b_base}_{t_base}"
    if candidate1 in BLOCKS:
        return candidate1
    if b_base == t_base:
        candidate2 = f"double_slab_{b_base}"
        if candidate2 in BLOCKS:
            return candidate2
    return candidate1


def is_offset_block(block_or_id):
    if not block_or_id:
        return False
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None or not isinstance(block, Block):
        return False
    return bool(block.properties.get("offset_block"))


def get_offset_info(block_or_id):
    """Returns offset block metadata dict or None if not an offset block:
    {
        "is_offset": True,
        "base_id": str,
        "canonical_id": str,
        "capped": bool,
        "top_slab": str | None,
    }
    """
    if not block_or_id:
        return None
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None or not isinstance(block, Block):
        return None
    props = block.properties
    if not props.get("offset_block"):
        return None
    base_id = str(props.get("offset_base", getattr(block, "base_id", block.id)))
    capped = bool(props.get("capped", False))
    top_slab = props.get("top_slab") if capped else None
    return {
        "is_offset": True,
        "base_id": base_id,
        "canonical_id": getattr(block, "base_id", base_id),
        "capped": capped,
        "top_slab": top_slab,
    }


def get_offset_block(block_id):
    if not block_id:
        return None
    if is_offset_block(block_id):
        info = get_offset_info(block_id)
        cand = f"{info['base_id']}_offset"
        return cand if cand in BLOCKS else block_id
    cand = f"{block_id}_offset"
    return cand if cand in BLOCKS and is_offset_block(cand) else None


def get_capped_offset_block(offset_block_or_id, slab_block_or_id):
    o_info = get_offset_info(offset_block_or_id)
    base_id = o_info["base_id"] if o_info is not None else str(offset_block_or_id)
    s_info = get_slab_info(slab_block_or_id)
    slab_base = s_info["base_id"] if s_info is not None else str(slab_block_or_id)
    cand = f"{base_id}_offset_{slab_base}"
    return cand if cand in BLOCKS else None


def has_half_offset_top(block_or_id):
    """Returns True if `block_or_id` has its top solid surface at y + 0.5 (bottom single slab or uncapped offset block)."""
    if not block_or_id:
        return False
    s_info = get_slab_info(block_or_id)
    if s_info is not None:
        return not s_info["double"] and s_info["half"] == "bottom"
    o_info = get_offset_info(block_or_id)
    if o_info is not None:
        return not o_info["capped"]
    return False


def player_facing_from_yaw(yaw):
    """Returns the cardinal direction the player is looking towards ('north', 'south', 'east', 'west')."""
    if yaw is None:
        return "north"
    yaw = float(yaw) % (2 * math.pi)
    fx = -math.sin(yaw)
    fz = -math.cos(yaw)
    if abs(fx) > abs(fz):
        return "east" if fx > 0 else "west"
    return "south" if fz > 0 else "north"


def block_facing_from_yaw(yaw):
    """Returns the horizontal facing for a block facing the player."""
    looking = player_facing_from_yaw(yaw)
    return OPPOSITE_FACING[looking]


def rotation_variant(block_id, axis_or_facing):
    block = BLOCKS.get(block_id)
    if block is None:
        return block_id
    base = getattr(block, "base_id", block_id)
    val = str(axis_or_facing).lower()
    if val in ("y", "south", "none", "", "default"):
        return base
    variant = f"{base}_{val}"
    return variant if variant in BLOCKS else block_id


def is_torch(block_or_id):
    if not block_or_id:
        return False
    if isinstance(block_or_id, str):
        block = BLOCKS.get(block_or_id)
    elif isinstance(block_or_id, Block):
        block = block_or_id
    else:
        return False
    if block is None:
        return False
    return bool(block.properties.get("torch") or block.id.endswith("torch") or getattr(block, "base_id", "").endswith("torch"))


def is_offset_torch(block_or_id):
    if not block_or_id:
        return False
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None or not is_torch(block):
        return False
    return block.properties.get("mount") == "up_offset"


def get_offset_torch_info(block_or_id):
    if not block_or_id:
        return None
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None or not isinstance(block, Block):
        return None
    props = block.properties
    if props.get("mount") != "up_offset":
        return None
    capped = bool(props.get("capped_torch", False))
    base_id = str(props.get("torch_base", getattr(block, "base_id", block.id)))
    top_slab = props.get("top_slab") if capped else None
    return {
        "is_offset_torch": True,
        "base_id": base_id,
        "capped": capped,
        "top_slab": top_slab,
    }


def get_capped_offset_torch(torch_block_or_id, slab_block_or_id):
    t_info = get_offset_torch_info(torch_block_or_id)
    if t_info is not None:
        base_id = t_info["base_id"]
    else:
        b = BLOCKS.get(torch_block_or_id) if isinstance(torch_block_or_id, str) else torch_block_or_id
        base_id = getattr(b, "base_id", str(torch_block_or_id)) if b is not None else str(torch_block_or_id)
    s_info = get_slab_info(slab_block_or_id)
    slab_base = s_info["base_id"] if s_info is not None else str(slab_block_or_id)
    cand = f"{base_id}_offset_{slab_base}"
    return cand if cand in BLOCKS else None


def get_offset_torch(block_or_id):
    if not block_or_id:
        return None
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None or not is_torch(block):
        return None
    base = getattr(block, "base_id", block.id)
    cand = f"{base}_offset"
    return cand if cand in BLOCKS else None


def can_support_torch(block_or_id, mount="up"):
    """Check if `block_or_id` can support a torch with mount direction ('up', 'up_offset', 'north', 'south', 'east', 'west').
    - mount == 'up' (standing on top of support block below): full solid blocks, capped offset blocks, or top/double slabs (whose top is at y=16).
    - mount == 'up_offset' (standing at half-height on top of support block below): bottom single slabs or uncapped offset blocks (whose top is at y=8).
    - mount in ('north', 'south', 'east', 'west') (attached to side of support block): full solid blocks, offset blocks, double slabs, or bottom slabs.
    """
    if not block_or_id:
        return False
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None or not block.solid:
        return False
    if mount == "up_offset":
        return has_half_offset_top(block)
    t_info = get_offset_torch_info(block)
    if t_info is not None:
        return mount == "up" and bool(t_info["capped"])
    o_info = get_offset_info(block)
    if o_info is not None:
        if mount == "up":
            return bool(o_info["capped"])
        return True
    s_info = get_slab_info(block)
    if s_info is not None:
        if s_info["double"]:
            return True
        if mount == "up":
            return s_info["half"] == "top"
        return s_info["half"] == "bottom"
    return bool(block.full_cube)


def placement_block(block_id, normal, yaw=None, local_y=None):
    block = BLOCKS.get(block_id)
    if block is None:
        return block_id
    if is_torch(block):
        nx, ny, nz = normal
        base = getattr(block, "base_id", block_id)
        if ny > 0:
            return base
        if nx > 0:
            cand = f"{base}_wall_east"
            return cand if cand in BLOCKS else base
        if nx < 0:
            cand = f"{base}_wall_west"
            return cand if cand in BLOCKS else base
        if nz > 0:
            cand = f"{base}_wall_south"
            return cand if cand in BLOCKS else base
        if nz < 0:
            cand = f"{base}_wall_north"
            return cand if cand in BLOCKS else base
        return base
    if is_slab(block):
        nx, ny, nz = normal
        if ny < 0:
            top_var = f"{block.base_id}_top"
            return top_var if top_var in BLOCKS else block_id
        if ny == 0 and local_y is not None and local_y >= 0.5:
            top_var = f"{block.base_id}_top"
            return top_var if top_var in BLOCKS else block_id
        return block.base_id
    if block.properties.get("decayable", False) or "leaves" in block_id:
        pers_id = f"{block.base_id}_persistent"
        if pers_id in BLOCKS:
            return pers_id
    rot = block.properties.get("rotation")
    if rot in ("axis", "xyz") or "axis" in block.properties:
        nx, ny, nz = normal
        if abs(nx) > 0:
            return rotation_variant(block.base_id, "x")
        if abs(nz) > 0:
            return rotation_variant(block.base_id, "z")
        return rotation_variant(block.base_id, "y")
    if rot in ("horizontal", "cardinal", "facing") or "facing" in block.properties:
        if yaw is not None:
            facing = block_facing_from_yaw(yaw)
            return rotation_variant(block.base_id, facing)
        else:
            nx, ny, nz = normal
            if abs(nx) > 0:
                facing = "west" if nx > 0 else "east"
            elif abs(nz) > 0:
                facing = "north" if nz > 0 else "south"
            else:
                facing = "south"
            return rotation_variant(block.base_id, facing)
    return block_id


def block_from_index(index):
    index = int(index)
    if index <= 0 or index >= len(BY_INDEX):
        return None
    return BY_INDEX[index]


def block_index(block_id):
    if not block_id or block_id == AIR:
        return 0
    return INDEX[block_id]


def canonical_id(value):
    if value is None or value == "" or value == AIR:
        return ""
    if isinstance(value, str):
        if value in BLOCKS:
            return value
        norm = (
            value.lower()
            .replace("[axis=", "_")
            .replace("[facing=", "_")
            .rstrip("]")
            .replace(":", "_")
        )
        if norm in BLOCKS:
            return norm
        if norm.endswith("_y") and norm[:-2] in BLOCKS:
            return norm[:-2]
        if norm.endswith("_south") and norm[:-6] in BLOCKS:
            return norm[:-6]
        if norm.endswith("_persistent") and norm[:-11] in BLOCKS:
            return norm[:-11]
        if norm.endswith("_top_waterlogged") and norm[:-16] in BLOCKS:
            return norm[:-16]
        if norm.endswith("_waterlogged") and norm[:-12] in BLOCKS:
            return norm[:-12]
        for sfx in ("_top_waterlogged_", "_waterlogged_"):
            if sfx in norm:
                base = norm[:norm.find(sfx)]
                if base in BLOCKS:
                    return base
        if norm.endswith("_top") and norm[:-4] in BLOCKS:
            return norm[:-4]
        for wall_sfx in ("_wall_north", "_wall_south", "_wall_east", "_wall_west", "_offset"):
            if norm.endswith(wall_sfx) and norm[: -len(wall_sfx)] in BLOCKS:
                return norm[: -len(wall_sfx)]
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        block = block_from_index(int(value))
        return block.id if block else ""
    return ""


def normalize_hotbar(values):
    from server.stacks import normalize_slots

    return normalize_slots(values, HOTBAR_SLOTS)


def install(block, index):
    while len(BY_INDEX) <= index:
        BY_INDEX.append(None)
    BY_INDEX[index] = block
    BLOCKS[block.id] = block
    INDEX[block.id] = index


def uninstall(block_id):
    block = BLOCKS.pop(block_id, None)
    INDEX.pop(block_id, None)
    for slot, item in enumerate(BY_INDEX):
        if item is block:
            BY_INDEX[slot] = None


def _resolve_texture(token, mapping, block_name):
    if str(token).startswith("#"):
        key = token[1:]
        if key not in mapping:
            raise KeyError(f"{block_name} model uses #{key}, which is not in its textures")
        return mapping[key]
    return token


def _map_uv(rect, model_uv):
    u0, v0, u1, v1 = rect
    mu0, mv0, mu1, mv1 = model_uv
    left = u0 + (float(mu0) / 16.0) * (u1 - u0)
    right = u0 + (float(mu1) / 16.0) * (u1 - u0)
    top = v1 + (float(mv0) / 16.0) * (v0 - v1)
    bottom = v1 + (float(mv1) / 16.0) * (v0 - v1)
    return ((left, bottom), (right, bottom), (right, top), (left, top))


def _load_model(model):
    if isinstance(model, dict):
        document = model
        label = "inline model"
    else:
        path = ASSETS / "models" / f"{model}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        document = json.loads(path.read_text())
        label = str(path)
    elements = document.get("elements") or []
    if not elements:
        raise ValueError(f"{label} has no elements")
    return elements


def _tab_ids(entry):
    raw = entry.get("tab", [])
    if isinstance(raw, str):
        names = [raw]
    else:
        names = [str(name) for name in raw]
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def _load_tabs(lang):
    folder = ASSETS / "tabs"
    paths = sorted(folder.glob("*.json"))
    if not paths:
        raise FileNotFoundError(folder)
    pending = []
    seen = set()
    for path in paths:
        entry = json.loads(path.read_text())
        tab_id = str(entry["id"])
        if path.stem != tab_id:
            raise ValueError(f"{path.name} must be named {tab_id}.json")
        if tab_id in seen or _ID.fullmatch(tab_id) is None:
            raise ValueError(f"tab id {tab_id!r} must be a unique lowercase name")
        seen.add(tab_id)
        screen = str(entry.get("screen", "blocks"))
        if screen not in ("blocks", "player"):
            raise ValueError(f"{tab_id} screen must be blocks or player")
        icon = str(entry.get("icon", ""))
        if icon and _ID.fullmatch(icon) is None:
            raise ValueError(f"{tab_id} icon must be a block id")
        if screen == "blocks" and not icon:
            raise ValueError(f"{tab_id} needs a block id for its icon")
        pending.append((int(entry.get("order", 0)), Tab(tab_id, _block_name(entry, lang), icon, screen)))
    pending.sort(key=lambda item: (item[0], item[1].id))
    return tuple(tab for _order, tab in pending)


def _load_lang():
    path = ASSETS / "lang" / "en.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _block_name(entry, lang):
    if entry.get("display"):
        return str(entry["display"])
    key = entry.get("lang")
    if key:
        return str(lang.get(key, key))
    return str(entry["id"])


def _block_entries():
    folder = ASSETS / "blocks"
    paths = sorted(folder.glob("*.json"))
    if not paths:
        raise FileNotFoundError(folder)
    by_id = {}
    for path in paths:
        entry = json.loads(path.read_text())
        block_id = str(entry["id"])
        if path.stem != block_id:
            raise ValueError(f"{path.name} must be named {block_id}.json")
        by_id[block_id] = entry
    pinned = (GRASS, DIRT, STONE, SAND, LOG, LEAVES, "planks", "cobble")
    ordered = []
    for block_id in pinned:
        entry = by_id.pop(block_id, None)
        if entry is not None:
            ordered.append(entry)
    for block_id in sorted(by_id):
        ordered.append(by_id[block_id])
    return ordered


def _pack_atlas(images):
    names = list(images)
    tile = int(images[names[0]].shape[0])
    for name, image in images.items():
        if image.shape[0] != tile or image.shape[1] != tile:
            raise ValueError(f"{name} must be a {tile}x{tile} texture")
    columns = min(16, len(names))
    rows = (len(names) + columns - 1) // columns
    atlas = np.zeros((rows * tile, columns * tile, 4), dtype=np.uint8)
    rects = {}
    for index, name in enumerate(names):
        column = index % columns
        row = index // columns
        x0 = column * tile
        y0 = row * tile
        atlas[y0 : y0 + tile, x0 : x0 + tile] = np.flipud(images[name])
        rects[name] = (
            (x0 + 0.5) / atlas.shape[1],
            (y0 + 0.5) / atlas.shape[0],
            (x0 + tile - 0.5) / atlas.shape[1],
            (y0 + tile - 0.5) / atlas.shape[0],
        )
    return atlas, rects


def _load_registry():
    lang = _load_lang()
    tabs = _load_tabs(lang)
    block_tabs = {tab.id for tab in tabs if tab.screen == "blocks"}
    texture_names = []
    prepared = []
    seen = set()
    for entry in _block_entries():
        block_id = str(entry["id"])
        if block_id == AIR or block_id in seen or _ID.fullmatch(block_id) is None:
            raise ValueError(f"block id {block_id!r} must be a unique lowercase name")
        seen.add(block_id)
        textures = dict(entry["textures"])
        for texture_name in textures.values():
            if texture_name not in texture_names:
                texture_names.append(texture_name)
        prepared.append((entry, _load_model(entry["model"]), textures, _block_name(entry, lang)))

    images = {}
    for texture_name in texture_names:
        path = ASSETS / "textures" / f"{texture_name}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        images[texture_name] = read_png(path)
    atlas, rects = _pack_atlas(images)

    blocks = {}
    by_index = [None]
    index_of = {}
    creative = []
    for entry, elements, textures, name in prepared:
        resolved = []
        face_uvs = [None] * len(FACE_INDEX)
        for element in elements:
            origin = [float(value) for value in element["from"]]
            extent = [float(value) for value in element["to"]]
            faces = {}
            for face_name, spec in element["faces"].items():
                if face_name not in FACE_INDEX:
                    raise ValueError(f"{entry['id']} has an unknown face {face_name}")
                texture_name = _resolve_texture(spec["texture"], textures, entry["id"])
                if texture_name not in rects:
                    raise KeyError(f"{entry['id']} texture {texture_name} was not loaded")
                uv = _map_uv(rects[texture_name], spec.get("uv", [0, 0, 16, 16]))
                index = FACE_INDEX[face_name]
                faces[index] = uv
                if face_uvs[index] is None:
                    face_uvs[index] = uv
            resolved.append({"from": origin, "to": extent, "faces": faces})
        full_cube = (
            len(resolved) == 1
            and resolved[0]["from"] == [0.0, 0.0, 0.0]
            and resolved[0]["to"] == [16.0, 16.0, 16.0]
            and all(face_uvs[i] is not None for i in range(6))
        )
        if full_cube is False and any(uv is None for uv in face_uvs):
            fallback = next((uv for uv in face_uvs if uv is not None), None)
            if fallback is not None:
                face_uvs = [uv if uv is not None else fallback for uv in face_uvs]
        render = entry.get("render", "solid")
        if render not in ("solid", "cutout"):
            raise ValueError(f"{entry['id']} render must be solid or cutout")
        cutout = render == "cutout"
        opacity = float(entry.get("opacity", 1.0))
        if not 0.0 < opacity <= 1.0:
            raise ValueError(f"{entry['id']} opacity must be between 0 and 1")
        creative_tab = bool(entry.get("creative", True))
        membership = _tab_ids(entry) if creative_tab else ()
        if creative_tab and not membership:
            raise ValueError(f"{entry['id']} needs a creative tab")
        unknown = [tab_id for tab_id in membership if tab_id not in block_tabs]
        if unknown:
            raise ValueError(f"{entry['id']} tab {unknown[0]} is not a creative tab")
        properties = dict(entry.get("properties") or {})
        is_slab_type = entry.get("type") == "slab" or properties.get("slab", False) or entry.get("model") == "slab"
        rotation_type = properties.get("rotation") or entry.get("rotation")
        if not rotation_type and entry.get("model") == "orientable":
            rotation_type = "horizontal"
        base_props = dict(properties)
        if is_slab_type:
            base_props["slab"] = True
            base_props["half"] = "bottom"
            base_props["waterlogged"] = False
            full_cube = False
        tint_val = entry.get("tint") or properties.get("tint")
        if tint_val is None and entry["id"] in ("grass", "muddy_grass"):
            tint_val = "grass"
        if tint_val:
            base_props["tint"] = tint_val
        if rotation_type in ("axis", "xyz") or "axis" in properties:
            base_props["rotation"] = "axis"
            base_props["axis"] = "y"
        elif rotation_type in ("horizontal", "cardinal", "facing") or "facing" in properties:
            base_props["rotation"] = "horizontal"
            base_props["facing"] = properties.get("facing", "south")
        hardness = float(entry.get("hardness", 1.0))
        tool = entry.get("tool") or entry.get("tool_type")
        mining_level = int(entry.get("mining_level", 0))
        loot_table = entry.get("loot_table")
        light_val = entry.get("light") or entry.get("emission") or properties.get("light") or properties.get("emission")
        gravity_val = bool(entry.get("gravity", False) or properties.get("gravity", False))
        traits = entry.get("traits", [])
        if isinstance(traits, str):
            traits = [traits]
        fluid_breakable_val = bool(
            entry.get("fluid_breakable", False)
            or entry.get("breakable_by_fluid", False)
            or properties.get("fluid_breakable", False)
            or properties.get("breakable_by_fluid", False)
            or "fluid_breakable" in traits
            or "breakable_by_fluid" in traits
        )
        soil_val = bool(
            entry.get("soil", False)
            or properties.get("soil", False)
            or "soil" in traits
            or entry["id"] in ("dirt", "grass", "mud", "muddy_grass")
        )
        plant_val = bool(
            entry.get("plant", False)
            or properties.get("plant", False)
            or "plant" in traits
            or entry["id"] in ("tall_grass", "fern", "rose", "dandelion")
        )
        requires_soil_val = bool(
            entry.get("requires_soil", False)
            or properties.get("requires_soil", False)
            or "requires_soil" in traits
            or plant_val
        )
        valid_ground_val = entry.get("valid_ground") or properties.get("valid_ground")
        if valid_ground_val is not None:
            if isinstance(valid_ground_val, str):
                valid_ground_val = (valid_ground_val,)
            else:
                valid_ground_val = tuple(valid_ground_val)

        station_val = entry.get("station") or properties.get("station") or entry.get("container") or properties.get("container")
        recipe_types_val = entry.get("recipe_types") or properties.get("recipe_types") or entry.get("recipe_type") or properties.get("recipe_type")
        cook_speed_val = float(entry.get("cook_speed") or properties.get("cook_speed") or 1.0)

        block = Block(
            str(entry["id"]),
            name,
            bool(entry.get("solid", True)),
            full_cube,
            face_uvs,
            resolved,
            cutout,
            opacity,
            membership,
            properties=base_props,
            base_id=str(entry["id"]),
            is_variant=False,
            hardness=hardness,
            tool=tool,
            mining_level=mining_level,
            loot_table=loot_table,
            light=light_val,
            gravity=gravity_val,
            fluid_breakable=fluid_breakable_val,
            soil=soil_val,
            plant=plant_val,
            requires_soil=requires_soil_val,
            valid_ground=valid_ground_val,
            station=station_val,
            recipe_types=recipe_types_val,
            cook_speed=cook_speed_val,
            slab=is_slab_type,
            slab_half=("bottom" if is_slab_type else None),
        )
        def _register_block(b_id, b_obj):
            if len(by_index) == 99:
                by_index.append(None)
            blocks[b_id] = b_obj
            by_index.append(b_obj)
            index_of[b_id] = len(by_index) - 1

        _register_block(block.id, block)
        if creative_tab:
            creative.append(block.id)

        if rotation_type in ("axis", "xyz") or "axis" in properties:
            top_uv = face_uvs[FACE_INDEX["up"]]
            side_uv = face_uvs[FACE_INDEX["east"]] or face_uvs[FACE_INDEX["south"]]
            for axis in ("x", "z"):
                variant_id = f"{block.id}_{axis}"
                var_props = dict(properties)
                var_props["rotation"] = "axis"
                var_props["axis"] = axis
                if axis == "x":
                    var_face_uvs = [
                        top_uv,
                        top_uv,
                        _rotate_uv(side_uv, 1),
                        _rotate_uv(side_uv, 1),
                        _rotate_uv(side_uv, 1),
                        _rotate_uv(side_uv, 1),
                    ]
                else:
                    var_face_uvs = [
                        _rotate_uv(side_uv, 1),
                        _rotate_uv(side_uv, 1),
                        side_uv,
                        side_uv,
                        top_uv,
                        top_uv,
                    ]
                var_resolved = [
                    {
                        "from": [0.0, 0.0, 0.0],
                        "to": [16.0, 16.0, 16.0],
                        "faces": {i: var_face_uvs[i] for i in range(6)},
                    }
                ]
                var_block = Block(
                    variant_id,
                    f"{name} ({axis.upper()})",
                    block.solid,
                    True,
                    var_face_uvs,
                    var_resolved,
                    cutout,
                    opacity,
                    membership,
                    properties=var_props,
                    base_id=block.id,
                    is_variant=True,
                    hardness=block.hardness,
                    tool=block.tool,
                    mining_level=block.mining_level,
                    loot_table=block.loot_table,
                    emission=block.emission,
                    gravity=block.gravity,
                    fluid_breakable=block.fluid_breakable,
                    soil=block.soil,
                    plant=block.plant,
                    requires_soil=block.requires_soil,
                    valid_ground=block.valid_ground,
                    tint=block.tint,
                    station=block.station,
                    recipe_types=block.recipe_types,
                    cook_speed=block.cook_speed,
                )
                _register_block(variant_id, var_block)

        elif rotation_type in ("horizontal", "cardinal", "facing") or "facing" in properties:
            top_uv = face_uvs[FACE_INDEX["up"]]
            bottom_uv = face_uvs[FACE_INDEX["down"]]
            front_uv = face_uvs[FACE_INDEX["south"]]
            side_uv = face_uvs[FACE_INDEX["east"]] or face_uvs[FACE_INDEX["north"]] or face_uvs[FACE_INDEX["west"]]
            back_uv = face_uvs[FACE_INDEX["north"]] or side_uv

            for facing in ("north", "east", "west"):
                variant_id = f"{block.id}_{facing}"
                var_props = dict(properties)
                var_props["rotation"] = "horizontal"
                var_props["facing"] = facing
                if facing == "north":
                    var_face_uvs = [side_uv, side_uv, top_uv, bottom_uv, back_uv, front_uv]
                elif facing == "east":
                    var_face_uvs = [front_uv, back_uv, top_uv, bottom_uv, side_uv, side_uv]
                elif facing == "west":
                    var_face_uvs = [back_uv, front_uv, top_uv, bottom_uv, side_uv, side_uv]

                var_resolved = [
                    {
                        "from": [0.0, 0.0, 0.0],
                        "to": [16.0, 16.0, 16.0],
                        "faces": {i: var_face_uvs[i] for i in range(6)},
                    }
                ]
                var_block = Block(
                    variant_id,
                    f"{name} ({facing.capitalize()})",
                    block.solid,
                    True,
                    var_face_uvs,
                    var_resolved,
                    cutout,
                    opacity,
                    membership,
                    properties=var_props,
                    base_id=block.id,
                    is_variant=True,
                    hardness=block.hardness,
                    tool=block.tool,
                    mining_level=block.mining_level,
                    loot_table=block.loot_table,
                    emission=block.emission,
                    gravity=block.gravity,
                    fluid_breakable=block.fluid_breakable,
                    soil=block.soil,
                    plant=block.plant,
                    requires_soil=block.requires_soil,
                    valid_ground=block.valid_ground,
                    tint=block.tint,
                    station=block.station,
                    recipe_types=block.recipe_types,
                    cook_speed=block.cook_speed,
                )
                _register_block(variant_id, var_block)

        fluid_type = properties.get("fluid") if isinstance(properties.get("fluid"), str) else ("water" if entry.get("id") == "water" else None)
        if fluid_type:
            fall_id = f"{block.id}_falling"
            fall_props = dict(properties)
            fall_props["fluid"] = fluid_type
            fall_props["level"] = 8
            fall_props["falling"] = True
            fall_block = Block(
                fall_id,
                f"{name} (Falling)",
                False,
                True,
                face_uvs,
                resolved,
                cutout,
                opacity,
                membership,
                properties=fall_props,
                base_id=block.id,
                is_variant=True,
                hardness=block.hardness,
                tool=block.tool,
                mining_level=block.mining_level,
                loot_table=block.loot_table,
                emission=block.emission,
            )
            _register_block(fall_id, fall_block)

            for level in range(7, 0, -1):
                flow_id = f"{block.id}_{level}"
                flow_props = dict(properties)
                flow_props["fluid"] = fluid_type
                flow_props["level"] = level
                flow_props["falling"] = False
                h = float(level) * 2.0
                flow_resolved = [
                    {
                        "from": [0.0, 0.0, 0.0],
                        "to": [16.0, h, 16.0],
                        "faces": {i: face_uvs[i] for i in range(6)},
                    }
                ]
                flow_block = Block(
                    flow_id,
                    f"{name} ({level})",
                    False,
                    False,
                    face_uvs,
                    flow_resolved,
                    cutout,
                    opacity,
                    membership,
                    properties=flow_props,
                    base_id=block.id,
                    is_variant=True,
                    hardness=block.hardness,
                    tool=block.tool,
                    mining_level=block.mining_level,
                    loot_table=block.loot_table,
                    emission=block.emission,
                    gravity=block.gravity,
                    fluid_breakable=block.fluid_breakable,
                    soil=block.soil,
                    plant=block.plant,
                    requires_soil=block.requires_soil,
                    valid_ground=block.valid_ground,
                )
                _register_block(flow_id, flow_block)

        if entry.get("id") == "leaves" or properties.get("decayable") or properties.get("leaf"):
            pers_props = dict(base_props)
            pers_props["persistent"] = True
            pers_props["decayable"] = False
            pers_id = f"{block.id}_persistent"
            pers_block = Block(
                pers_id,
                block.name,
                block.solid,
                block.full_cube,
                face_uvs,
                resolved,
                cutout,
                opacity,
                membership,
                properties=pers_props,
                base_id=block.id,
                is_variant=True,
                hardness=block.hardness,
                tool=block.tool,
                mining_level=block.mining_level,
                loot_table=block.loot_table,
                emission=block.emission,
                gravity=block.gravity,
                fluid_breakable=block.fluid_breakable,
                soil=block.soil,
                plant=block.plant,
                requires_soil=block.requires_soil,
                valid_ground=block.valid_ground,
            )
            _register_block(pers_id, pers_block)

        if properties.get("torch") or entry.get("model") == "torch":
            block.properties["torch"] = True
            block.properties["mount"] = "up"
            wall_boxes = {
                "east": ([0.0, 2.0, 7.0], [4.0, 12.0, 9.0]),
                "west": ([12.0, 2.0, 7.0], [16.0, 12.0, 9.0]),
                "south": ([7.0, 2.0, 0.0], [9.0, 12.0, 4.0]),
                "north": ([7.0, 2.0, 12.0], [9.0, 12.0, 16.0]),
            }
            base_el_faces = dict(resolved[0]["faces"]) if resolved else {i: face_uvs[i] for i in range(6)}
            for wall_dir, (w_from, w_to) in wall_boxes.items():
                wall_id = f"{block.id}_wall_{wall_dir}"
                wall_props = dict(base_props)
                wall_props["torch"] = True
                wall_props["mount"] = wall_dir
                wall_resolved = [
                    {
                        "from": list(w_from),
                        "to": list(w_to),
                        "faces": dict(base_el_faces),
                    }
                ]
                wall_block = Block(
                    wall_id,
                    f"{name} (Wall {wall_dir.capitalize()})",
                    False,
                    False,
                    face_uvs,
                    wall_resolved,
                    cutout,
                    opacity,
                    membership,
                    properties=wall_props,
                    base_id=block.id,
                    is_variant=True,
                    hardness=block.hardness,
                    tool=block.tool,
                    mining_level=block.mining_level,
                    loot_table=block.loot_table,
                    light=light_val,
                    emission=block.emission,
                    gravity=block.gravity,
                    fluid_breakable=block.fluid_breakable,
                )
                _register_block(wall_id, wall_block)

            offset_torch_id = f"{block.id}_offset"
            offset_torch_props = dict(base_props)
            offset_torch_props["torch"] = True
            offset_torch_props["mount"] = "up_offset"
            offset_torch_resolved = [
                {
                    "from": [7.0, -8.0, 7.0],
                    "to": [9.0, 2.0, 9.0],
                    "faces": dict(base_el_faces),
                }
            ]
            offset_torch_block = Block(
                offset_torch_id,
                f"{name} (Offset)",
                False,
                False,
                face_uvs,
                offset_torch_resolved,
                cutout,
                opacity,
                membership,
                properties=offset_torch_props,
                base_id=block.id,
                is_variant=True,
                hardness=block.hardness,
                tool=block.tool,
                mining_level=block.mining_level,
                loot_table=block.loot_table,
                light=light_val,
                emission=block.emission,
                gravity=block.gravity,
                fluid_breakable=block.fluid_breakable,
            )
            _register_block(offset_torch_id, offset_torch_block)

        if is_slab_type:
            top_model_elements = _load_model("slab_top")
            top_resolved = []
            for element in top_model_elements:
                origin = [float(value) for value in element["from"]]
                extent = [float(value) for value in element["to"]]
                faces = {}
                for face_name, spec in element["faces"].items():
                    texture_name = _resolve_texture(spec["texture"], textures, entry["id"])
                    uv = _map_uv(rects[texture_name], spec.get("uv", [0, 0, 16, 16]))
                    faces[FACE_INDEX[face_name]] = uv
                top_resolved.append({"from": origin, "to": extent, "faces": faces})

            top_id = f"{block.id}_top"
            top_props = dict(base_props)
            top_props["half"] = "top"
            top_block = Block(
                top_id,
                f"{name} (Top)",
                True,
                False,
                face_uvs,
                top_resolved,
                cutout,
                opacity,
                membership,
                properties=top_props,
                base_id=block.id,
                is_variant=True,
                hardness=block.hardness,
                tool=block.tool,
                mining_level=block.mining_level,
                loot_table=block.loot_table,
                emission=block.emission,
                slab=True,
                slab_half="top",
            )
            _register_block(top_id, top_block)

            wl_id = f"{block.id}_waterlogged"
            wl_props = dict(base_props)
            wl_props["waterlogged"] = True
            wl_props["fluid"] = "water"
            wl_props["level"] = 8
            wl_block = Block(
                wl_id,
                f"{name} (Waterlogged)",
                True,
                False,
                face_uvs,
                resolved,
                cutout,
                opacity,
                membership,
                properties=wl_props,
                base_id=block.id,
                is_variant=True,
                hardness=block.hardness,
                tool=block.tool,
                mining_level=block.mining_level,
                loot_table=block.loot_table,
                emission=block.emission,
                slab=True,
                slab_half="bottom",
                waterlogged=True,
            )
            _register_block(wl_id, wl_block)

            wlt_id = f"{block.id}_top_waterlogged"
            wlt_props = dict(top_props)
            wlt_props["waterlogged"] = True
            wlt_props["fluid"] = "water"
            wlt_props["level"] = 8
            wlt_block = Block(
                wlt_id,
                f"{name} (Top Waterlogged)",
                True,
                False,
                face_uvs,
                top_resolved,
                cutout,
                opacity,
                membership,
                properties=wlt_props,
                base_id=block.id,
                is_variant=True,
                hardness=block.hardness,
                tool=block.tool,
                mining_level=block.mining_level,
                loot_table=block.loot_table,
                emission=block.emission,
                slab=True,
                slab_half="top",
                waterlogged=True,
            )
            _register_block(wlt_id, wlt_block)

            for lvl in range(7, 0, -1):
                wl_flow_id = f"{block.id}_waterlogged_{lvl}"
                wl_flow_props = dict(base_props)
                wl_flow_props["waterlogged"] = True
                wl_flow_props["fluid"] = "water"
                wl_flow_props["level"] = lvl
                wl_flow_block = Block(
                    wl_flow_id,
                    f"{name} (Waterlogged {lvl})",
                    True,
                    False,
                    face_uvs,
                    resolved,
                    cutout,
                    opacity,
                    membership,
                    properties=wl_flow_props,
                    base_id=block.id,
                    is_variant=True,
                    hardness=block.hardness,
                    tool=block.tool,
                    mining_level=block.mining_level,
                    loot_table=block.loot_table,
                    emission=block.emission,
                    slab=True,
                    slab_half="bottom",
                    waterlogged=True,
                )
                _register_block(wl_flow_id, wl_flow_block)

                wlt_flow_id = f"{block.id}_top_waterlogged_{lvl}"
                wlt_flow_props = dict(top_props)
                wlt_flow_props["waterlogged"] = True
                wlt_flow_props["fluid"] = "water"
                wlt_flow_props["level"] = lvl
                wlt_flow_block = Block(
                    wlt_flow_id,
                    f"{name} (Top Waterlogged {lvl})",
                    True,
                    False,
                    face_uvs,
                    top_resolved,
                    cutout,
                    opacity,
                    membership,
                    properties=wlt_flow_props,
                    base_id=block.id,
                    is_variant=True,
                    hardness=block.hardness,
                    tool=block.tool,
                    mining_level=block.mining_level,
                    loot_table=block.loot_table,
                    emission=block.emission,
                    slab=True,
                    slab_half="top",
                    waterlogged=True,
                )
                _register_block(wlt_flow_id, wlt_flow_block)

    import itertools

    slab_blocks = [
        b
        for b in list(blocks.values())
        if b.properties.get("slab") and not b.is_variant and not b.properties.get("double_slab")
    ]
    for slab_a, slab_b in itertools.product(slab_blocks, slab_blocks):
        double_id = f"double_slab_{slab_a.id}_{slab_b.id}"
        slab_b_top = blocks[f"{slab_b.id}_top"]
        bottom_elements = []
        for el in slab_a.elements:
            faces = {k: v for k, v in el["faces"].items() if k != 2}
            bottom_elements.append({"from": list(el["from"]), "to": list(el["to"]), "faces": faces})
        top_elements = []
        for el in slab_b_top.elements:
            faces = {k: v for k, v in el["faces"].items() if k != 3}
            top_elements.append({"from": list(el["from"]), "to": list(el["to"]), "faces": faces})
        double_elements = bottom_elements + top_elements
        double_props = {
            "double_slab": True,
            "slab": True,
            "half": "double",
            "bottom": slab_a.id,
            "top": slab_b.id,
        }
        d_name = (
            f"{slab_a.name} / {slab_b.name} Double Slab"
            if slab_a.id != slab_b.id
            else f"{slab_a.name} Double Slab"
        )
        double_face_uvs = list(slab_a.face_uvs)
        if len(slab_b.face_uvs) > 2 and slab_b.face_uvs[2] is not None:
            double_face_uvs[2] = slab_b.face_uvs[2]
        d_block = Block(
            double_id,
            d_name,
            True,
            True,
            double_face_uvs,
            double_elements,
            cutout=False,
            opacity=1.0,
            properties=double_props,
            base_id=slab_a.id,
            is_variant=True,
            hardness=max(slab_a.hardness, slab_b.hardness),
            tool=slab_a.tool or slab_b.tool,
            mining_level=max(slab_a.mining_level, slab_b.mining_level),
            slab=True,
            slab_half="double",
            double_slab=True,
        )
        if len(by_index) == 99:
            by_index.append(None)
        blocks[double_id] = d_block
        by_index.append(d_block)
        index_of[double_id] = len(by_index) - 1

        if slab_a.id == slab_b.id:
            alias_id = f"double_slab_{slab_a.id}"
            blocks[alias_id] = d_block
            index_of[alias_id] = index_of[double_id]

    full_solid_blocks = [
        b
        for b in list(blocks.values())
        if b.solid
        and b.full_cube
        and not b.properties.get("double_slab")
        and not b.properties.get("slab")
        and not b.properties.get("fluid")
        and not b.properties.get("offset_block")
    ]
    for fb in full_solid_blocks:
        offset_id = f"{fb.id}_offset"
        offset_faces = {i: fb.face_uvs[i] for i in range(6) if fb.face_uvs[i] is not None}
        offset_elements = [
            {
                "from": [0.0, -8.0, 0.0],
                "to": [16.0, 8.0, 16.0],
                "faces": offset_faces,
                "tint": fb.tint,
            }
        ]
        offset_props = dict(fb.properties)
        offset_props["offset_block"] = True
        offset_props["offset_base"] = fb.id
        offset_props["capped"] = False
        offset_props["top_slab"] = None
        off_block = Block(
            offset_id,
            f"{fb.name} (Offset)",
            True,
            False,
            list(fb.face_uvs),
            offset_elements,
            cutout=fb.cutout,
            opacity=fb.opacity,
            properties=offset_props,
            base_id=fb.base_id,
            is_variant=True,
            hardness=fb.hardness,
            tool=fb.tool,
            mining_level=fb.mining_level,
            loot_table=fb.loot_table,
            emission=fb.emission,
            gravity=fb.gravity,
            fluid_breakable=fb.fluid_breakable,
            soil=fb.soil,
            plant=fb.plant,
            requires_soil=fb.requires_soil,
            valid_ground=fb.valid_ground,
            tint=fb.tint,
            station=fb.station,
            recipe_types=fb.recipe_types,
            cook_speed=fb.cook_speed,
        )
        if len(by_index) == 99:
            by_index.append(None)
        blocks[offset_id] = off_block
        by_index.append(off_block)
        index_of[offset_id] = len(by_index) - 1

        for slab_b in slab_blocks:
            capped_id = f"{fb.id}_offset_{slab_b.id}"
            slab_b_top = blocks[f"{slab_b.id}_top"]
            bot_faces = {i: fb.face_uvs[i] for i in range(6) if i != 2 and fb.face_uvs[i] is not None}
            bot_el = {
                "from": [0.0, -8.0, 0.0],
                "to": [16.0, 8.0, 16.0],
                "faces": bot_faces,
                "tint": fb.tint,
                "cutout": fb.cutout,
            }
            top_els = []
            for el in slab_b_top.elements:
                t_faces = {k: v for k, v in el["faces"].items() if k != 3 or fb.cutout}
                top_els.append(
                    {
                        "from": list(el["from"]),
                        "to": list(el["to"]),
                        "faces": t_faces,
                        "tint": slab_b.tint,
                        "cutout": slab_b.cutout,
                    }
                )
            capped_elements = [bot_el] + top_els
            capped_props = dict(fb.properties)
            capped_props["offset_block"] = True
            capped_props["offset_base"] = fb.id
            capped_props["capped"] = True
            capped_props["top_slab"] = slab_b.id
            capped_face_uvs = list(fb.face_uvs)
            if len(slab_b.face_uvs) > 2 and slab_b.face_uvs[2] is not None:
                capped_face_uvs[2] = slab_b.face_uvs[2]
            cap_block = Block(
                capped_id,
                f"{fb.name} (Offset) / {slab_b.name}",
                True,
                False,
                capped_face_uvs,
                capped_elements,
                cutout=fb.cutout,
                opacity=1.0,
                properties=capped_props,
                base_id=fb.base_id,
                is_variant=True,
                hardness=max(fb.hardness, slab_b.hardness),
                tool=fb.tool or slab_b.tool,
                mining_level=max(fb.mining_level, slab_b.mining_level),
                loot_table=fb.loot_table,
                emission=fb.emission,
                gravity=fb.gravity,
                fluid_breakable=fb.fluid_breakable,
                soil=slab_b.soil,
                tint=fb.tint,
                station=fb.station,
                recipe_types=fb.recipe_types,
                cook_speed=fb.cook_speed,
            )
            if len(by_index) == 99:
                by_index.append(None)
            blocks[capped_id] = cap_block
            by_index.append(cap_block)
            index_of[capped_id] = len(by_index) - 1

    base_torch_blocks = [
        b
        for b in list(blocks.values())
        if b.properties.get("torch") and not b.is_variant
    ]
    for tb in base_torch_blocks:
        tb_off = blocks.get(f"{tb.id}_offset")
        if tb_off is None:
            continue
        for slab_b in slab_blocks:
            capped_t_id = f"{tb.id}_offset_{slab_b.id}"
            slab_b_top = blocks[f"{slab_b.id}_top"]
            top_els = [
                {
                    "from": list(el["from"]),
                    "to": list(el["to"]),
                    "faces": dict(el["faces"]),
                    "tint": slab_b.tint,
                }
                for el in slab_b_top.elements
            ]
            capped_t_elements = [dict(tb_off.elements[0])] + top_els
            capped_t_props = dict(tb_off.properties)
            capped_t_props["torch"] = True
            capped_t_props["mount"] = "up_offset"
            capped_t_props["capped_torch"] = True
            capped_t_props["torch_base"] = tb.id
            capped_t_props["top_slab"] = slab_b.id
            cap_t_block = Block(
                capped_t_id,
                f"{tb.name} (Offset) / {slab_b.name}",
                True,
                False,
                list(slab_b_top.face_uvs),
                capped_t_elements,
                cutout=True,
                opacity=0.0,
                properties=capped_t_props,
                base_id=tb.id,
                is_variant=True,
                hardness=slab_b.hardness,
                tool=slab_b.tool,
                mining_level=slab_b.mining_level,
                loot_table=tb.loot_table,
                light=tb.light,
                emission=tb.emission,
                gravity=False,
                fluid_breakable=False,
                soil=slab_b.soil,
                tint=slab_b.tint,
            )
            if len(by_index) == 99:
                by_index.append(None)
            blocks[capped_t_id] = cap_t_block
            by_index.append(cap_t_block)
            index_of[capped_t_id] = len(by_index) - 1

    if len(by_index) > 65536:
        raise ValueError("there are more than 65535 blocks")
    grouped = {
        tab.id: tuple(block_id for block_id in creative if tab.id in blocks[block_id].tabs)
        for tab in tabs
        if tab.screen == "blocks"
    }
    return blocks, by_index, index_of, tuple(creative), tabs, grouped, atlas


BLOCKS, BY_INDEX, INDEX, CREATIVE, TABS, TAB_ITEMS, ATLAS = _load_registry()
BY_NAME = BLOCKS
DEFAULT_HOTBAR = ("",) * HOTBAR_SLOTS
for _required in (GRASS, DIRT, STONE, SAND, LOG, LEAVES):
    if _required not in BLOCKS:
        raise KeyError(f"missing block {_required}")
if WORLD_HEIGHT % CHUNK_SIZE:
    raise ValueError("world height must be a whole number of chunks")
SECTION_COUNT = WORLD_HEIGHT // CHUNK_SIZE


def atlas_bytes():
    image = np.ascontiguousarray(ATLAS)
    return image.tobytes(), (image.shape[1], image.shape[0])


SOIL_BLOCKS = ("dirt", "grass", "mud", "muddy_grass")


def is_soil(block):
    if not block:
        return False
    if isinstance(block, str):
        b_obj = BLOCKS.get(block)
        if b_obj is not None:
            return getattr(b_obj, "soil", False) or block in SOIL_BLOCKS
        return block in SOIL_BLOCKS
    return getattr(block, "soil", False) or getattr(block, "id", "") in SOIL_BLOCKS


def can_sustain_plant(ground, plant=None):
    if not ground:
        return False
    ground_id = ground if isinstance(ground, str) else getattr(ground, "id", "")
    ground_obj = BLOCKS.get(ground_id) if isinstance(ground, str) else ground
    if ground_obj is None or not ground_obj.solid:
        return False

    plant_id = plant if isinstance(plant, str) else (getattr(plant, "id", "") if plant else "")
    plant_obj = BLOCKS.get(plant_id) if isinstance(plant, str) else plant

    if plant_obj is not None and plant_obj.valid_ground is not None:
        if ground_id in plant_obj.valid_ground:
            return True
        if "soil" in plant_obj.valid_ground and is_soil(ground_obj):
            return True
        return False

    return is_soil(ground_obj)


def can_place_block(world, x, y, z, block_id, hit=None):
    if y <= 0 or y >= WORLD_HEIGHT:
        return False
    if not block_id or block_id not in BLOCKS:
        return False
    b_obj = BLOCKS[block_id]

    cur_blk = world.get_block(x, y, z)
    cur_t_info = get_offset_torch_info(cur_blk)
    new_t_info = get_offset_torch_info(b_obj)

    # Allow capping an uncapped offset torch with a top slab, or breaking one part of a capped offset torch.
    if cur_t_info is not None:
        if not cur_t_info["capped"] and new_t_info is not None and new_t_info["capped"]:
            return True
        if cur_t_info["capped"] and new_t_info is not None and not new_t_info["capped"]:
            return True
        new_s = get_slab_info(b_obj)
        if cur_t_info["capped"] and new_s is not None and not new_s["double"] and new_s["half"] == "top":
            return True
        return False

    # Cannot place any block directly into a voxel already occupied by a torch.
    if is_torch(cur_blk):
        return False

    # If the voxel above holds an offset block or offset torch, its bottom half occupies [y + 0.5, y + 1.0] in this voxel.
    if y + 1 < WORLD_HEIGHT:
        above_blk = world.get_block(x, y + 1, z)
        if is_offset_block(above_blk):
            new_s = get_slab_info(b_obj)
            new_o = get_offset_info(b_obj)
            allowed_under_offset = (
                (new_s is not None and not new_s["double"] and new_s["half"] == "bottom")
                or (new_o is not None and not new_o["capped"])
                or (new_t_info is not None and not new_t_info["capped"])
            )
            if not allowed_under_offset:
                return False
        elif is_offset_torch(above_blk):
            new_s = get_slab_info(b_obj)
            new_o = get_offset_info(b_obj)
            allowed_under_offset_torch = (
                (new_s is not None and not new_s["double"] and new_s["half"] == "bottom")
                or (new_o is not None and not new_o["capped"])
            )
            if not allowed_under_offset_torch:
                return False

    # If placing an offset block, its bottom half occupies [y - 0.5, y] in the voxel below.
    if is_offset_block(b_obj):
        if y - 1 < 0:
            return False
        below_blk = world.get_block(x, y - 1, z)
        below_obj = BLOCKS.get(below_blk)
        below_t_info = get_offset_torch_info(below_blk)
        if not (
            below_blk in ("", AIR, None)
            or (below_obj is not None and bool(below_obj.properties.get("fluid")))
            or has_half_offset_top(below_blk)
            or (below_t_info is not None and not below_t_info["capped"])
        ):
            return False

    # Current position must not be solid (unless combining slabs into a double slab, capping an offset block/torch, or partial breaking)
    if world.is_solid(x, y, z):
        cur_off = get_offset_info(cur_blk)
        new_off = get_offset_info(b_obj)
        if cur_off is not None:
            # Capping an uncapped offset block with a top slab
            if not cur_off["capped"] and new_off is not None and new_off["capped"]:
                return True
            # Breaking the top slab of a capped offset block back to uncapped offset block
            if cur_off["capped"] and new_off is not None and not new_off["capped"]:
                return True
            # Breaking the offset full block of a capped offset block, leaving the top slab
            new_s = get_slab_info(b_obj)
            if cur_off["capped"] and new_s is not None and not new_s["double"] and new_s["half"] == "top":
                return True
        if new_t_info is not None and new_t_info["capped"] and is_slab(cur_blk):
            cur_s_info = get_slab_info(cur_blk)
            if cur_s_info is not None and not cur_s_info["double"] and cur_s_info["half"] == "top":
                return can_support_torch(world.get_block(x, y - 1, z), "up_offset")
        if is_slab(b_obj) and is_slab(cur_blk):
            cur_info = get_slab_info(cur_blk)
            new_info = get_slab_info(b_obj)
            if cur_info and new_info:
                # Combining single slabs into double slab:
                if not cur_info["double"] and new_info["double"]:
                    return True
                # Breaking half a double slab into single slab:
                if cur_info["double"] and not new_info["double"]:
                    return True
                # Replacing single slab with single slab (e.g. state transition or logging)
                if not cur_info["double"] and not new_info["double"]:
                    return True
        return False

    # If placing against another block via hit
    if hit is not None:
        hx, hy, hz = hit.get("x", x), hit.get("y", y), hit.get("z", z)
        hit_block_id = world.get_block(hx, hy, hz)
        hit_obj = BLOCKS.get(hit_block_id)
        # Cannot place on top of or against non-solid blocks (e.g. flowers, tall grass)
        if hit_obj is not None and not hit_obj.solid:
            return False

    # Check torch support requirements
    if is_torch(b_obj):
        cur_block = world.get_block(x, y, z)
        if cur_block not in ("", AIR, None):
            return False
        mount = b_obj.properties.get("mount", "up")
        if hit is not None:
            ny = hit.get("ny", 0)
            if ny < 0:
                return False
            if ny > 0 and mount not in ("up", "up_offset"):
                return False
        if mount in ("up", "up_offset"):
            support_id = world.get_block(x, y - 1, z)
            return can_support_torch(support_id, mount)
        support_offsets = {
            "east": (-1, 0),
            "west": (1, 0),
            "south": (0, -1),
            "north": (0, 1),
        }
        dx, dz = support_offsets.get(mount, (0, 0))
        support_id = world.get_block(x + dx, y, z + dz)
        return can_support_torch(support_id, mount)

    # Check plant / soil requirements
    if (
        getattr(b_obj, "requires_soil", False)
        or getattr(b_obj, "plant", False)
        or b_obj.valid_ground is not None
        or block_id in ("tall_grass", "fern", "rose", "dandelion")
    ):
        # Must be placed on the top face
        if hit is not None and hit.get("ny", 0) != 1:
            return False

        # Current block space cannot be occupied by fluid or existing block
        cur_block = world.get_block(x, y, z)
        if cur_block not in ("", AIR, None):
            return False

        # Ground directly below (x, y - 1, z) must sustain the plant
        below_id = world.get_block(x, y - 1, z)
        if not can_sustain_plant(below_id, b_obj):
            return False

    return True
