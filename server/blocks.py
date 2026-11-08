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
FACE_UV_VECTORS = (
    ((0, 0, -1), (0, 1, 0)),   # 0: +X (East): right=-Z, up=+Y
    ((0, 0, 1), (0, 1, 0)),    # 1: -X (West): right=+Z, up=+Y
    ((1, 0, 0), (0, 0, -1)),   # 2: +Y (Up): right=+X, up=-Z
    ((1, 0, 0), (0, 0, 1)),    # 3: -Y (Down): right=+X, up=+Z
    ((1, 0, 0), (0, 1, 0)),    # 4: +Z (South): right=+X, up=+Y
    ((-1, 0, 0), (0, 1, 0)),   # 5: -Z (North): right=-X, up=+Y
)
GLASS_CTM_UVS = ()
GLASS_CTM_RECTS = {}


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
        sound=None,
        sounds=None,
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
        sound_val = sound if sound is not None else (self.properties.get("sound") or (properties.get("sound") if properties else None))
        sounds_val = sounds if sounds is not None else (self.properties.get("sounds") or (properties.get("sounds") if properties else None))
        self.sound = str(sound_val).lower() if sound_val else None
        self.sounds = dict(sounds_val) if isinstance(sounds_val, dict) else None
        if self.sound:
            self.properties["sound"] = self.sound
        if self.sounds:
            self.properties["sounds"] = self.sounds
        self.slab = bool(slab or self.properties.get("slab", False) or self.properties.get("double_slab", False))
        self.slab_half = slab_half or self.properties.get("half", ("bottom" if self.slab else None))
        self.vertical_slab = bool(self.properties.get("vertical_slab", False))
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
        source = (face_uvs[FACE_INDEX["up"]] if face_uvs else None) or next((uv for uv in (face_uvs or ()) if uv is not None), None)
        if source:
            self.icon = (
                min(point[0] for point in source),
                min(point[1] for point in source),
                max(point[0] for point in source),
                max(point[1] for point in source),
            )
        else:
            self.icon = (0.0, 0.0, 1.0, 1.0)

    def get_walk_sound(self, rng=None):
        from server.sound import get_block_walk_sound

        return get_block_walk_sound(self, rng=rng)

    def get_breaking_sound(self, rng=None):
        from server.sound import get_block_breaking_sound

        return get_block_breaking_sound(self, rng=rng)

    def get_break_sound(self, rng=None):
        from server.sound import get_block_break_sound

        return get_block_break_sound(self, rng=rng)


def _rotate_uv(uv, turns=1):
    if uv is None:
        return None
    turns = turns % 4
    return tuple(uv[(i + turns) % 4] for i in range(4))


# New face index -> old face index when a vertical slab is rotated around Y.
_VERTICAL_SLAB_FACE_MAPS = {
    "north": (1, 0, 2, 3, 5, 4),
    "east": (4, 5, 2, 3, 1, 0),
    "west": (5, 4, 2, 3, 0, 1),
}


def _rotate_vertical_slab_elements(elements, face_uvs, facing):
    """Return model elements and face UVs rotated to the requested facing."""
    if facing == "south":
        return elements, list(face_uvs)
    face_map = _VERTICAL_SLAB_FACE_MAPS[facing]
    rotated_uvs = [face_uvs[old] for old in face_map]
    rotated = []
    for element in elements:
        x0, y0, z0 = (float(value) for value in element["from"])
        x1, y1, z1 = (float(value) for value in element["to"])
        if facing == "north":
            bounds = ((x0, y0, 16.0 - z1), (x1, y1, 16.0 - z0))
        elif facing == "east":
            bounds = ((z0, y0, 16.0 - x1), (z1, y1, 16.0 - x0))
        else:  # west
            bounds = ((16.0 - z1, y0, x0), (16.0 - z0, y1, x1))
        faces = {}
        face_uv_specs = {}
        for new_face, old_face in enumerate(face_map):
            uv = element.get("faces", {}).get(old_face)
            if uv is not None:
                faces[new_face] = uv
            if "face_uv_specs" in element and old_face in element["face_uv_specs"]:
                face_uv_specs[new_face] = element["face_uv_specs"][old_face]
        # Recompute top (+Y) and bottom (-Y) UV specs and textures for the rotated bounding box
        spec_2 = [int(round(bounds[0][0])), int(round(bounds[0][2])), int(round(bounds[1][0])), int(round(bounds[1][2]))]
        spec_3 = [int(round(bounds[0][0])), int(round(16.0 - bounds[1][2])), int(round(bounds[1][0])), int(round(16.0 - bounds[0][2]))]
        if 2 in face_uv_specs:
            face_uv_specs[2] = spec_2
        if 3 in face_uv_specs:
            face_uv_specs[3] = spec_3
        rects_map = element.get("texture_rects", {})
        if 2 in faces and 2 in rects_map:
            faces[2] = _map_uv(rects_map[2], spec_2)
            rotated_uvs[2] = faces[2]
        if 3 in faces and 3 in rects_map:
            faces[3] = _map_uv(rects_map[3], spec_3)
            rotated_uvs[3] = faces[3]
        el_dict = {
            "from": list(bounds[0]),
            "to": list(bounds[1]),
            "faces": faces,
            "face_uv_specs": face_uv_specs,
        }
        if "texture_rects" in element:
            el_dict["texture_rects"] = dict(element["texture_rects"])
        if "glass" in element:
            el_dict["glass"] = element["glass"]
        if "cutout" in element:
            el_dict["cutout"] = element["cutout"]
        rotated.append(el_dict)
    return rotated, rotated_uvs


OPPOSITE_FACING = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}

FACING_NORMALS = {
    "north": (0, 0, -1),
    "south": (0, 0, 1),
    "east": (1, 0, 0),
    "west": (-1, 0, 0),
}


def facing_from_normal(normal):
    """Return the cardinal facing for an axis-aligned normal, or None."""
    nx, ny, nz = normal
    if nx > 0:
        return "east"
    if nx < 0:
        return "west"
    if nz > 0:
        return "south"
    if nz < 0:
        return "north"
    return None


SUB_CELL_BLOCK = "sub_cell"


def is_sub_cell(block_or_id):
    if not block_or_id:
        return False
    if block_or_id == SUB_CELL_BLOCK:
        return True
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return False
    return bool(block.properties.get("sub_cell"))


def is_glass(block_or_id):
    if not block_or_id:
        return False
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return isinstance(block_or_id, str) and (
            block_or_id in ("glass", "glass_slab", "glass_vertical_slab")
            or "glass" in block_or_id
        )
    props = getattr(block, "properties", {})
    if props.get("double_slab"):
        bottom_id = props.get("bottom", "")
        top_id = props.get("top", "")
        return is_glass(bottom_id) and is_glass(top_id)
    if props.get("double_vertical_slab"):
        pos_id = props.get("positive_block") or props.get("positive", "")
        neg_id = props.get("negative_block") or props.get("negative", "")
        return is_glass(pos_id) and is_glass(neg_id)
    if props.get("capped") and props.get("offset_block"):
        base_id = props.get("offset_base", getattr(block, "base_id", ""))
        top_id = props.get("top_slab", "")
        return is_glass(base_id) and is_glass(top_id)
    return bool(
        getattr(block, "glass", False)
        or props.get("glass", False)
        or getattr(block, "id", "") in ("glass", "glass_slab", "glass_vertical_slab")
        or (getattr(block, "base_id", "") in ("glass", "glass_slab", "glass_vertical_slab"))
    )


def is_face_glass(block_or_id, normal_i):
    if not block_or_id:
        return False
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return False
    if getattr(block, "full_cube", False) and is_glass(block):
        return True
    if hasattr(block, "elements") and block.elements:
        for el in block.elements:
            el_glass = el.get("glass")
            if el_glass is None:
                el_glass = is_glass(block)
            if el_glass and normal_i in el.get("faces", {}):
                return True
    return bool(getattr(block, "glass", False) or block.properties.get("glass", False))


def get_glass_ctm_uv(mask, model_uv=(0, 0, 16, 16)):
    rect = GLASS_CTM_RECTS.get(mask)
    if rect is not None:
        return _map_uv(rect, model_uv)
    if GLASS_CTM_UVS and 0 <= mask < len(GLASS_CTM_UVS):
        return GLASS_CTM_UVS[mask]
    return None


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


def is_vertical_slab(block_or_id):
    """Return True if block_or_id is a half-cell vertical slab."""
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return False
    props = getattr(block, "properties", {})
    if props.get("double_vertical_slab") or getattr(block, "double_vertical_slab", False):
        return False
    if props.get("offset_block") or getattr(block, "offset_block", False):
        return False
    if getattr(block, "vertical_slab", False) or props.get("vertical_slab"):
        return True
    base_id = getattr(block, "base_id", str(block.id))
    return base_id == "vertical_slab" or "_vertical_slab" in base_id


FACING_DIRECTIONS = ("north", "south", "east", "west")


def get_vertical_slab_info(block_or_id):
    """Return vertical slab metadata, including its cardinal facing."""
    if not block_or_id:
        return None
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None or not is_vertical_slab(block):
        return None
    props = block.properties
    facing = str(props.get("facing", "south")).lower()
    if facing not in FACING_DIRECTIONS:
        facing = "south"
    waterlogged = bool(props.get("waterlogged", False))
    level = int(props.get("level", 8 if waterlogged else 0))
    return {
        "id": block.id,
        "vertical_slab": True,
        "facing": facing,
        "base_id": getattr(block, "base_id", block.id),
        "waterlogged": waterlogged,
        "level": level,
    }


def get_waterlogged_vertical_slab(block_or_id, level=8):
    """Return the waterlogged vertical slab block ID for the given level (1..8), or None."""
    info = get_vertical_slab_info(block_or_id)
    if info is None:
        return None
    base = info["base_id"]
    facing = info["facing"]
    if level >= 8:
        cand = f"{base}_waterlogged" if facing == "south" else f"{base}_{facing}_waterlogged"
    elif level > 0:
        cand = f"{base}_waterlogged_{level}" if facing == "south" else f"{base}_{facing}_waterlogged_{level}"
    else:
        cand = base if facing == "south" else f"{base}_{facing}"
    return cand if cand in BLOCKS else None


def get_unwaterlogged_vertical_slab(block_or_id):
    """Return the dry vertical slab block ID, or the block ID itself."""
    info = get_vertical_slab_info(block_or_id)
    if info is None:
        return block_or_id if isinstance(block_or_id, str) else getattr(block_or_id, "id", "")
    base = info["base_id"]
    facing = info["facing"]
    cand = base if facing == "south" else f"{base}_{facing}"
    return cand if cand in BLOCKS else base


def is_waterlogged_vertical_slab(block_or_id):
    info = get_vertical_slab_info(block_or_id)
    return bool(info and info.get("waterlogged"))


def get_double_vertical_slab_info(block_or_id):
    """Return metadata for a full-cell double vertical slab, or None."""
    if not block_or_id:
        return None
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None or not isinstance(block, Block):
        return None
    props = block.properties
    if not props.get("double_vertical_slab"):
        return None
    axis = str(props.get("axis", "z")).lower()
    if axis not in ("x", "z"):
        axis = "z"
    positive_facing = props.get("positive_facing", "south" if axis == "z" else "east")
    negative_facing = props.get("negative_facing", "north" if axis == "z" else "west")
    return {
        "id": block.id,
        "double_vertical_slab": True,
        "axis": axis,
        "positive": props.get("positive", ""),
        "negative": props.get("negative", ""),
        "positive_block": props.get("positive_block", ""),
        "negative_block": props.get("negative_block", ""),
        "positive_facing": positive_facing,
        "negative_facing": negative_facing,
    }


def get_double_vertical_slab(block_a, block_b):
    """Return the full-cell block made by two complementary vertical slabs."""
    a_info = get_vertical_slab_info(block_a) or get_vertical_slab_offset_info(block_a)
    b_info = get_vertical_slab_info(block_b) or get_vertical_slab_offset_info(block_b)
    if a_info is None or b_info is None:
        return None
    facing_a = a_info["facing"]
    facing_b = b_info["facing"]
    if facing_a == facing_b:
        return None
    if {facing_a, facing_b} == {"south", "north"}:
        axis, positive_facing, negative_facing = "z", "south", "north"
    elif {facing_a, facing_b} == {"east", "west"}:
        axis, positive_facing, negative_facing = "x", "east", "west"
    else:
        return None

    if facing_a == positive_facing:
        positive_base, negative_base = a_info["base_id"], b_info["base_id"]
    else:
        positive_base, negative_base = b_info["base_id"], a_info["base_id"]

    is_offset = is_vertical_slab_offset(block_a) or is_vertical_slab_offset(block_b)
    if is_offset:
        candidate = f"double_vertical_slab_{axis}_{positive_base}_{negative_base}_offset"
        top_slab = a_info.get("top_slab") or b_info.get("top_slab")
        if candidate in BLOCKS:
            if top_slab:
                capped_cand = f"{candidate}_{top_slab}"
                if capped_cand in BLOCKS:
                    return capped_cand
            return candidate
        if positive_base == negative_base:
            alias = f"double_vertical_slab_{axis}_{positive_base}_offset"
            if alias in BLOCKS:
                if top_slab:
                    capped_alias = f"{alias}_{top_slab}"
                    if capped_alias in BLOCKS:
                        return capped_alias
                return alias
        return None

    candidate = f"double_vertical_slab_{axis}_{positive_base}_{negative_base}"
    if candidate in BLOCKS:
        return candidate
    if positive_base == negative_base:
        alias = f"double_vertical_slab_{axis}_{positive_base}"
        if alias in BLOCKS:
            return alias
    return None


def get_vertical_slab_torch_info(block_or_id):
    """Return metadata for a vertical slab with a torch on its back face."""
    if not block_or_id:
        return None
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None or not isinstance(block, Block):
        return None
    props = block.properties
    torch_block = props.get("back_torch")
    if not torch_block:
        return None
    return {
        "id": block.id,
        "back_torch": True,
        "slab_block": props.get("slab_block", block.id),
        "torch_block": torch_block,
        "torch_mount": props.get("torch_mount"),
        "facing": props.get("facing", "south"),
    }


def get_vertical_slab_torch(slab_block_or_id, torch_block_or_id):
    """Return the combined slab+torch block for a torch on the slab's back face."""
    slab_info = get_vertical_slab_info(slab_block_or_id)
    if slab_info is None:
        return None
    slab_block = BLOCKS.get(slab_info["id"])
    torch_block = (
        BLOCKS.get(torch_block_or_id)
        if isinstance(torch_block_or_id, str)
        else torch_block_or_id
    )
    if slab_block is None or torch_block is None or not is_torch(torch_block):
        return None
    torch_base = getattr(torch_block, "base_id", torch_block.id)
    candidate = f"{slab_block.id}_back_{torch_base}"
    return candidate if candidate in BLOCKS else None


def vertical_slab_bounds(x, y, z, facing):
    """World-space AABB for the solid half of a vertical slab."""
    x = float(x)
    y = float(y)
    z = float(z)
    if facing == "north":
        return (x, y, z, x + 1.0, y + 1.0, z + 0.5)
    if facing == "east":
        return (x + 0.5, y, z, x + 1.0, y + 1.0, z + 1.0)
    if facing == "west":
        return (x, y, z, x + 0.5, y + 1.0, z + 1.0)
    # South-facing (the default model) occupies the positive-Z half.
    return (x, y, z + 0.5, x + 1.0, y + 1.0, z + 1.0)


def vertical_slab_offset_bounds(x, y, z, facing):
    """World-space AABB for the solid half of an uncapped vertical slab offset block."""
    vx0, _, vz0, vx1, _, vz1 = vertical_slab_bounds(x, y, z, facing)
    return (vx0, float(y) - 0.5, vz0, vx1, float(y) + 0.5, vz1)


def door_bounds(x, y, z, facing, is_open=False, hinge="left"):
    """World-space AABB for a door block (lower or upper half)."""
    x = float(x)
    y = float(y)
    z = float(z)
    t = 3.0 / 16.0
    if not is_open:
        if facing == "north":
            return (x, y, z, x + 1.0, y + 1.0, z + t)
        elif facing == "south":
            return (x, y, z + 1.0 - t, x + 1.0, y + 1.0, z + 1.0)
        elif facing == "west":
            return (x, y, z, x + t, y + 1.0, z + 1.0)
        else:  # east
            return (x + 1.0 - t, y, z, x + 1.0, y + 1.0, z + 1.0)
    else:
        if facing == "north":
            return (
                (x, y, z, x + t, y + 1.0, z + 1.0)
                if hinge == "left"
                else (x + 1.0 - t, y, z, x + 1.0, y + 1.0, z + 1.0)
            )
        elif facing == "south":
            return (
                (x + 1.0 - t, y, z, x + 1.0, y + 1.0, z + 1.0)
                if hinge == "left"
                else (x, y, z, x + t, y + 1.0, z + 1.0)
            )
        elif facing == "west":
            return (
                (x, y, z + 1.0 - t, x + 1.0, y + 1.0, z + 1.0)
                if hinge == "left"
                else (x, y, z, x + 1.0, y + 1.0, z + t)
            )
        else:  # east
            return (
                (x, y, z, x + 1.0, y + 1.0, z + t)
                if hinge == "left"
                else (x, y, z + 1.0 - t, x + 1.0, y + 1.0, z + 1.0)
            )


def get_door_relative_offsets(facing):
    """Return ((left_dx, left_dz), (right_dx, right_dz)) relative to the given facing."""
    if facing == "north":
        return (-1, 0), (1, 0)
    elif facing == "south":
        return (1, 0), (-1, 0)
    elif facing == "west":
        return (0, 1), (0, -1)
    else:  # east
        return (0, -1), (0, 1)


def is_door(block_or_id):
    """Return True if block_or_id is a door (any half, facing, or open state)."""
    if not block_or_id:
        return False
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return False
    props = getattr(block, "properties", {})
    b_id = getattr(block, "id", str(block_or_id))
    if props.get("trapdoor") or getattr(block, "trapdoor", False) or "trapdoor" in b_id:
        return False
    if props.get("door") or getattr(block, "door", False):
        return True
    return "door" in b_id


def get_door_info(block_or_id):
    """Return door metadata or None if not a door:
    {
        "base_id": str,
        "half": "lower" | "upper",
        "facing": "north" | "south" | "east" | "west",
        "hinge": "left" | "right",
        "open": bool,
    }
    """
    if not block_or_id:
        return None
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return None
    props = getattr(block, "properties", {})
    b_id = getattr(block, "id", str(block_or_id))
    if props.get("trapdoor") or getattr(block, "trapdoor", False) or "trapdoor" in b_id:
        return None
    if not (props.get("door") or getattr(block, "door", False) or "door" in b_id):
        return None

    base_id = getattr(block, "base_id", None)
    if not base_id or base_id == b_id:
        parts = b_id.split("_")
        door_idx = next((i for i, p in enumerate(parts) if "door" in p), -1)
        if door_idx != -1:
            base_id = "_".join(parts[: door_idx + 1])
        else:
            base_id = "wooden_door"

    half = props.get("half")
    if half is None:
        half = "upper" if "_upper" in b_id else "lower"

    facing = props.get("facing")
    if facing is None:
        for f in ("north", "south", "east", "west"):
            if f"_{f}" in b_id:
                facing = f
                break
        if facing is None:
            facing = "south"

    hinge = props.get("hinge")
    if hinge is None:
        hinge = "right" if "_right" in b_id else "left"

    is_open = props.get("open")
    if is_open is None:
        is_open = b_id.endswith("_open")

    return {
        "base_id": base_id,
        "half": half,
        "facing": facing,
        "hinge": hinge,
        "open": bool(is_open),
    }


def get_door_block(base_id, half="lower", facing="south", open=False, hinge="left"):
    """Return the block ID string for a door variant."""
    hinge_str = "_right" if hinge == "right" else ""
    open_suffix = "_open" if open else ""
    return f"{base_id}_{half}_{facing}{hinge_str}{open_suffix}"


def _get_door_elements(
    uv_main_front,
    uv_main_back,
    uv_edge_h,
    uv_edge_h_z,
    uv_edge_v0,
    uv_edge_v1,
    facing,
    is_open=False,
    hinge="left",
):
    t = 3.0
    faces = {}
    if not is_open:
        if facing == "north":
            origin, extent = [0.0, 0.0, 0.0], [16.0, 16.0, t]
            if hinge == "left":
                faces[FACE_INDEX["north"]] = uv_main_front
                faces[FACE_INDEX["south"]] = uv_main_back
                faces[FACE_INDEX["west"]] = uv_edge_v0
                faces[FACE_INDEX["east"]] = uv_edge_v1
            else:
                faces[FACE_INDEX["north"]] = uv_main_back
                faces[FACE_INDEX["south"]] = uv_main_front
                faces[FACE_INDEX["west"]] = uv_edge_v1
                faces[FACE_INDEX["east"]] = uv_edge_v0
            faces[FACE_INDEX["up"]] = uv_edge_h
            faces[FACE_INDEX["down"]] = uv_edge_h
        elif facing == "south":
            origin, extent = [0.0, 0.0, 16.0 - t], [16.0, 16.0, 16.0]
            if hinge == "left":
                faces[FACE_INDEX["south"]] = uv_main_front
                faces[FACE_INDEX["north"]] = uv_main_back
                faces[FACE_INDEX["west"]] = uv_edge_v1
                faces[FACE_INDEX["east"]] = uv_edge_v0
            else:
                faces[FACE_INDEX["south"]] = uv_main_back
                faces[FACE_INDEX["north"]] = uv_main_front
                faces[FACE_INDEX["west"]] = uv_edge_v0
                faces[FACE_INDEX["east"]] = uv_edge_v1
            faces[FACE_INDEX["up"]] = uv_edge_h
            faces[FACE_INDEX["down"]] = uv_edge_h
        elif facing == "west":
            origin, extent = [0.0, 0.0, 0.0], [t, 16.0, 16.0]
            if hinge == "left":
                faces[FACE_INDEX["west"]] = uv_main_front
                faces[FACE_INDEX["east"]] = uv_main_back
                faces[FACE_INDEX["south"]] = uv_edge_v0
                faces[FACE_INDEX["north"]] = uv_edge_v1
            else:
                faces[FACE_INDEX["west"]] = uv_main_back
                faces[FACE_INDEX["east"]] = uv_main_front
                faces[FACE_INDEX["south"]] = uv_edge_v1
                faces[FACE_INDEX["north"]] = uv_edge_v0
            faces[FACE_INDEX["up"]] = uv_edge_h_z
            faces[FACE_INDEX["down"]] = uv_edge_h_z
        else:  # east
            origin, extent = [16.0 - t, 0.0, 0.0], [16.0, 16.0, 16.0]
            if hinge == "left":
                faces[FACE_INDEX["east"]] = uv_main_front
                faces[FACE_INDEX["west"]] = uv_main_back
                faces[FACE_INDEX["north"]] = uv_edge_v0
                faces[FACE_INDEX["south"]] = uv_edge_v1
            else:
                faces[FACE_INDEX["east"]] = uv_main_back
                faces[FACE_INDEX["west"]] = uv_main_front
                faces[FACE_INDEX["north"]] = uv_edge_v1
                faces[FACE_INDEX["south"]] = uv_edge_v0
            faces[FACE_INDEX["up"]] = uv_edge_h_z
            faces[FACE_INDEX["down"]] = uv_edge_h_z
    else:
        if facing == "north":
            if hinge == "left":
                origin, extent = [0.0, 0.0, 0.0], [t, 16.0, 16.0]
                faces[FACE_INDEX["west"]] = uv_main_front
                faces[FACE_INDEX["east"]] = uv_main_back
            else:
                origin, extent = [16.0 - t, 0.0, 0.0], [16.0, 16.0, 16.0]
                faces[FACE_INDEX["east"]] = uv_main_back
                faces[FACE_INDEX["west"]] = uv_main_front
            faces[FACE_INDEX["up"]] = uv_edge_h_z
            faces[FACE_INDEX["down"]] = uv_edge_h_z
            faces[FACE_INDEX["north"]] = uv_edge_v0
            faces[FACE_INDEX["south"]] = uv_edge_v1
        elif facing == "south":
            if hinge == "left":
                origin, extent = [16.0 - t, 0.0, 0.0], [16.0, 16.0, 16.0]
                faces[FACE_INDEX["east"]] = uv_main_front
                faces[FACE_INDEX["west"]] = uv_main_back
            else:
                origin, extent = [0.0, 0.0, 0.0], [t, 16.0, 16.0]
                faces[FACE_INDEX["west"]] = uv_main_back
                faces[FACE_INDEX["east"]] = uv_main_front
            faces[FACE_INDEX["up"]] = uv_edge_h_z
            faces[FACE_INDEX["down"]] = uv_edge_h_z
            faces[FACE_INDEX["south"]] = uv_edge_v0
            faces[FACE_INDEX["north"]] = uv_edge_v1
        elif facing == "west":
            if hinge == "left":
                origin, extent = [0.0, 0.0, 16.0 - t], [16.0, 16.0, 16.0]
                faces[FACE_INDEX["south"]] = uv_main_front
                faces[FACE_INDEX["north"]] = uv_main_back
            else:
                origin, extent = [0.0, 0.0, 0.0], [16.0, 16.0, t]
                faces[FACE_INDEX["north"]] = uv_main_back
                faces[FACE_INDEX["south"]] = uv_main_front
            faces[FACE_INDEX["up"]] = uv_edge_h
            faces[FACE_INDEX["down"]] = uv_edge_h
            faces[FACE_INDEX["west"]] = uv_edge_v0
            faces[FACE_INDEX["east"]] = uv_edge_v1
        else:  # east
            if hinge == "left":
                origin, extent = [0.0, 0.0, 0.0], [16.0, 16.0, t]
                faces[FACE_INDEX["north"]] = uv_main_front
                faces[FACE_INDEX["south"]] = uv_main_back
            else:
                origin, extent = [0.0, 0.0, 16.0 - t], [16.0, 16.0, 16.0]
                faces[FACE_INDEX["south"]] = uv_main_back
                faces[FACE_INDEX["north"]] = uv_main_front
            faces[FACE_INDEX["up"]] = uv_edge_h
            faces[FACE_INDEX["down"]] = uv_edge_h
            faces[FACE_INDEX["east"]] = uv_edge_v0
            faces[FACE_INDEX["west"]] = uv_edge_v1

    return [{"from": origin, "to": extent, "faces": faces}]

    return [{"from": origin, "to": extent, "faces": faces}]


def trapdoor_bounds(x, y, z, half="bottom", facing="north", is_open=False):
    """World-space AABB for a trapdoor block."""
    x = float(x)
    y = float(y)
    z = float(z)
    t = 3.0 / 16.0
    if not is_open:
        if half == "bottom":
            return (x, y, z, x + 1.0, y + t, z + 1.0)
        else:
            return (x, y + 1.0 - t, z, x + 1.0, y + 1.0, z + 1.0)
    else:
        if facing == "north":
            return (x, y, z, x + 1.0, y + 1.0, z + t)
        elif facing == "south":
            return (x, y, z + 1.0 - t, x + 1.0, y + 1.0, z + 1.0)
        elif facing == "west":
            return (x, y, z, x + t, y + 1.0, z + 1.0)
        else:  # east
            return (x + 1.0 - t, y, z, x + 1.0, y + 1.0, z + 1.0)


def is_trapdoor(block_or_id):
    """Return True if block_or_id is a trapdoor."""
    if not block_or_id:
        return False
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return False
    props = getattr(block, "properties", {})
    if props.get("trapdoor") or getattr(block, "trapdoor", False):
        return True
    b_id = getattr(block, "id", str(block_or_id))
    return "trapdoor" in b_id


def get_trapdoor_info(block_or_id):
    """Return trapdoor metadata or None if not a trapdoor:
    {
        "base_id": str,
        "half": "bottom" | "top",
        "facing": "north" | "south" | "east" | "west",
        "open": bool,
    }
    """
    if not block_or_id:
        return None
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return None
    props = getattr(block, "properties", {})
    b_id = getattr(block, "id", str(block_or_id))
    if not (props.get("trapdoor") or getattr(block, "trapdoor", False) or "trapdoor" in b_id):
        return None

    base_id = getattr(block, "base_id", None)
    if not base_id or base_id == b_id:
        parts = b_id.split("_")
        td_idx = next((i for i, p in enumerate(parts) if "trapdoor" in p), -1)
        if td_idx != -1:
            base_id = "_".join(parts[: td_idx + 1])
        else:
            base_id = "wooden_trapdoor"

    half = props.get("half")
    if half is None:
        half = "top" if "_top" in b_id else "bottom"

    facing = props.get("facing")
    if facing is None:
        for f in ("north", "south", "east", "west"):
            if f"_{f}" in b_id:
                facing = f
                break
        if facing is None:
            facing = "north"

    is_open = props.get("open")
    if is_open is None:
        is_open = b_id.endswith("_open")

    return {
        "base_id": base_id,
        "half": half,
        "facing": facing,
        "open": bool(is_open),
    }


def get_trapdoor_block(base_id, half="bottom", facing="north", open=False):
    """Return the block ID string for a trapdoor variant."""
    open_suffix = "_open" if open else ""
    return f"{base_id}_{half}_{facing}{open_suffix}"


def _get_trapdoor_elements(
    uv_main,
    uv_edge_h,
    uv_edge_h_z,
    uv_edge_v0,
    uv_edge_v1,
    half="bottom",
    facing="north",
    is_open=False,
):
    t = 3.0
    faces = {}
    if not is_open:
        if half == "bottom":
            origin, extent = [0.0, 0.0, 0.0], [16.0, t, 16.0]
        else:
            origin, extent = [0.0, 16.0 - t, 0.0], [16.0, 16.0, 16.0]
        faces[FACE_INDEX["up"]] = uv_main
        faces[FACE_INDEX["down"]] = uv_main
        faces[FACE_INDEX["north"]] = uv_edge_h
        faces[FACE_INDEX["south"]] = uv_edge_h
        faces[FACE_INDEX["west"]] = uv_edge_h
        faces[FACE_INDEX["east"]] = uv_edge_h
    else:
        if facing == "north":
            origin, extent = [0.0, 0.0, 0.0], [16.0, 16.0, t]
            faces[FACE_INDEX["north"]] = uv_main
            faces[FACE_INDEX["south"]] = uv_main
            faces[FACE_INDEX["up"]] = uv_edge_h
            faces[FACE_INDEX["down"]] = uv_edge_h
            faces[FACE_INDEX["west"]] = uv_edge_v0
            faces[FACE_INDEX["east"]] = uv_edge_v1
        elif facing == "south":
            origin, extent = [0.0, 0.0, 16.0 - t], [16.0, 16.0, 16.0]
            faces[FACE_INDEX["south"]] = uv_main
            faces[FACE_INDEX["north"]] = uv_main
            faces[FACE_INDEX["up"]] = uv_edge_h
            faces[FACE_INDEX["down"]] = uv_edge_h
            faces[FACE_INDEX["west"]] = uv_edge_v1
            faces[FACE_INDEX["east"]] = uv_edge_v0
        elif facing == "west":
            origin, extent = [0.0, 0.0, 0.0], [t, 16.0, 16.0]
            faces[FACE_INDEX["west"]] = uv_main
            faces[FACE_INDEX["east"]] = uv_main
            faces[FACE_INDEX["up"]] = uv_edge_h_z
            faces[FACE_INDEX["down"]] = uv_edge_h_z
            faces[FACE_INDEX["south"]] = uv_edge_v0
            faces[FACE_INDEX["north"]] = uv_edge_v1
        else:  # east
            origin, extent = [16.0 - t, 0.0, 0.0], [16.0, 16.0, 16.0]
            faces[FACE_INDEX["east"]] = uv_main
            faces[FACE_INDEX["west"]] = uv_main
            faces[FACE_INDEX["up"]] = uv_edge_h_z
            faces[FACE_INDEX["down"]] = uv_edge_h_z
            faces[FACE_INDEX["north"]] = uv_edge_v0
            faces[FACE_INDEX["south"]] = uv_edge_v1

    return [{"from": origin, "to": extent, "faces": faces}]


def _make_fence_gate_box(from_pos, to_pos, rect):
    x0, y0, z0 = [float(v) for v in from_pos]
    x1, y1, z1 = [float(v) for v in to_pos]
    faces = {
        FACE_INDEX["down"]: _map_uv(rect, [x0, z0, x1, z1]),
        FACE_INDEX["up"]: _map_uv(rect, [x0, z0, x1, z1]),
        FACE_INDEX["north"]: _map_uv(rect, [x0, 16.0 - y1, x1, 16.0 - y0]),
        FACE_INDEX["south"]: _map_uv(rect, [x0, 16.0 - y1, x1, 16.0 - y0]),
        FACE_INDEX["west"]: _map_uv(rect, [z0, 16.0 - y1, z1, 16.0 - y0]),
        FACE_INDEX["east"]: _map_uv(rect, [z0, 16.0 - y1, z1, 16.0 - y0]),
    }
    return {"from": [x0, y0, z0], "to": [x1, y1, z1], "faces": faces}


def _get_fence_gate_elements(rect, facing="north", is_open=False):
    elements = []
    if not is_open:
        if facing in ("north", "south"):
            boxes = [
                ([0.0, 5.0, 7.0], [2.0, 16.0, 9.0]),
                ([14.0, 5.0, 7.0], [16.0, 16.0, 9.0]),
                ([2.0, 12.0, 7.0], [6.0, 15.0, 9.0]),
                ([2.0, 6.0, 7.0], [6.0, 9.0, 9.0]),
                ([6.0, 5.0, 7.0], [8.0, 15.0, 9.0]),
                ([8.0, 5.0, 7.0], [10.0, 15.0, 9.0]),
                ([10.0, 12.0, 7.0], [14.0, 15.0, 9.0]),
                ([10.0, 6.0, 7.0], [14.0, 9.0, 9.0]),
            ]
        else:  # east or west
            boxes = [
                ([7.0, 5.0, 0.0], [9.0, 16.0, 2.0]),
                ([7.0, 5.0, 14.0], [9.0, 16.0, 16.0]),
                ([7.0, 12.0, 2.0], [9.0, 15.0, 6.0]),
                ([7.0, 6.0, 2.0], [9.0, 9.0, 6.0]),
                ([7.0, 5.0, 6.0], [9.0, 15.0, 8.0]),
                ([7.0, 5.0, 8.0], [9.0, 15.0, 10.0]),
                ([7.0, 12.0, 10.0], [9.0, 15.0, 14.0]),
                ([7.0, 6.0, 10.0], [9.0, 9.0, 14.0]),
            ]
    else:
        if facing == "north":
            boxes = [
                ([0.0, 5.0, 7.0], [2.0, 16.0, 9.0]),
                ([14.0, 5.0, 7.0], [16.0, 16.0, 9.0]),
                ([0.0, 5.0, 1.0], [2.0, 15.0, 3.0]),
                ([0.0, 12.0, 3.0], [2.0, 15.0, 7.0]),
                ([0.0, 6.0, 3.0], [2.0, 9.0, 7.0]),
                ([14.0, 5.0, 1.0], [16.0, 15.0, 3.0]),
                ([14.0, 12.0, 3.0], [16.0, 15.0, 7.0]),
                ([14.0, 6.0, 3.0], [16.0, 9.0, 7.0]),
            ]
        elif facing == "south":
            boxes = [
                ([0.0, 5.0, 7.0], [2.0, 16.0, 9.0]),
                ([14.0, 5.0, 7.0], [16.0, 16.0, 9.0]),
                ([0.0, 12.0, 9.0], [2.0, 15.0, 13.0]),
                ([0.0, 6.0, 9.0], [2.0, 9.0, 13.0]),
                ([0.0, 5.0, 13.0], [2.0, 15.0, 15.0]),
                ([14.0, 12.0, 9.0], [16.0, 15.0, 13.0]),
                ([14.0, 6.0, 9.0], [16.0, 9.0, 13.0]),
                ([14.0, 5.0, 13.0], [16.0, 15.0, 15.0]),
            ]
        elif facing == "west":
            boxes = [
                ([7.0, 5.0, 0.0], [9.0, 16.0, 2.0]),
                ([7.0, 5.0, 14.0], [9.0, 16.0, 16.0]),
                ([1.0, 5.0, 0.0], [3.0, 15.0, 2.0]),
                ([3.0, 12.0, 0.0], [7.0, 15.0, 2.0]),
                ([3.0, 6.0, 0.0], [7.0, 9.0, 2.0]),
                ([1.0, 5.0, 14.0], [3.0, 15.0, 16.0]),
                ([3.0, 12.0, 14.0], [7.0, 15.0, 16.0]),
                ([3.0, 6.0, 14.0], [7.0, 9.0, 16.0]),
            ]
        else:  # east
            boxes = [
                ([7.0, 5.0, 0.0], [9.0, 16.0, 2.0]),
                ([7.0, 5.0, 14.0], [9.0, 16.0, 16.0]),
                ([9.0, 12.0, 0.0], [13.0, 15.0, 2.0]),
                ([9.0, 6.0, 0.0], [13.0, 9.0, 2.0]),
                ([13.0, 5.0, 0.0], [15.0, 15.0, 2.0]),
                ([9.0, 12.0, 14.0], [13.0, 15.0, 16.0]),
                ([9.0, 6.0, 14.0], [13.0, 9.0, 16.0]),
                ([13.0, 5.0, 14.0], [15.0, 15.0, 16.0]),
            ]
    for b_from, b_to in boxes:
        elements.append(_make_fence_gate_box(b_from, b_to, rect))
    return elements


def is_fence_gate(block_or_id):
    """Return True if block_or_id is a fence gate block."""
    if not block_or_id:
        return False
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return False
    props = getattr(block, "properties", {})
    if props.get("fence_gate") or getattr(block, "fence_gate", False):
        return True
    b_id = getattr(block, "id", str(block_or_id))
    return "fence_gate" in b_id or ("fence" in b_id and "gate" in b_id)


def get_fence_gate_info(block_or_id):
    """Return fence gate metadata or None if not a fence gate:
    {
        "base_id": str,
        "facing": "north" | "south" | "east" | "west",
        "open": bool,
    }
    """
    if not block_or_id:
        return None
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None or not is_fence_gate(block):
        return None
    props = getattr(block, "properties", {})
    b_id = getattr(block, "id", str(block_or_id))

    base_id = getattr(block, "base_id", None)
    if not base_id or base_id == b_id:
        parts = b_id.split("_")
        gate_idx = next((i for i, p in enumerate(parts) if "gate" in p), -1)
        if gate_idx != -1:
            base_id = "_".join(parts[: gate_idx + 1])
        else:
            base_id = "wooden_fence_gate"

    facing = props.get("facing")
    if facing is None:
        for f in ("north", "south", "east", "west"):
            if f"_{f}" in b_id:
                facing = f
                break
        if facing is None:
            facing = "north"

    is_open = props.get("open")
    if is_open is None:
        is_open = b_id.endswith("_open")

    return {
        "base_id": base_id,
        "facing": facing,
        "open": bool(is_open),
    }


def get_fence_gate_block(base_id, facing="north", open=False):
    """Return the block ID string for a fence gate variant."""
    open_suffix = "_open" if open else ""
    return f"{base_id}_{facing}{open_suffix}"


def fence_gate_bounds(x, y, z, facing, is_open=False):
    """Collision / selection box for a fence gate."""
    x = float(x)
    y = float(y)
    z = float(z)
    if not is_open:
        h = 1.5
        if facing in ("north", "south"):
            return (x, y, z + 6.0 / 16.0, x + 1.0, y + h, z + 10.0 / 16.0)
        else:
            return (x + 6.0 / 16.0, y, z, x + 10.0 / 16.0, y + h, z + 1.0)
    else:
        return None


def fence_gate_collision_boxes(world, x, y, z):
    """World-space collision boxes for a fence gate at (x, y, z).
    1.5 blocks tall when closed, narrow side posts when open.
    """
    block_id = world.get_block(x, y, z) if world is not None else None
    info = get_fence_gate_info(block_id)
    facing = info["facing"] if info else "north"
    is_open = info["open"] if info else False
    x, y, z = float(x), float(y), float(z)
    h = 1.5

    if not is_open:
        if facing in ("north", "south"):
            return [(x, y, z + 6.0 / 16.0, x + 1.0, y + h, z + 10.0 / 16.0)]
        else:
            return [(x + 6.0 / 16.0, y, z, x + 10.0 / 16.0, y + h, z + 1.0)]
    else:
        if facing in ("north", "south"):
            return [
                (x, y, z + 6.0 / 16.0, x + 2.0 / 16.0, y + h, z + 10.0 / 16.0),
                (x + 14.0 / 16.0, y, z + 6.0 / 16.0, x + 1.0, y + h, z + 10.0 / 16.0),
            ]
        else:
            return [
                (x + 6.0 / 16.0, y, z, x + 10.0 / 16.0, y + h, z + 2.0 / 16.0),
                (x + 6.0 / 16.0, y, z + 14.0 / 16.0, x + 10.0 / 16.0, y + h, z + 1.0),
            ]


def fence_gate_raycast_boxes(world, x, y, z):
    """World-space raycast/selection boxes for a fence gate at (x, y, z)."""
    block_id = world.get_block(x, y, z) if world is not None else None
    info = get_fence_gate_info(block_id)
    facing = info["facing"] if info else "north"
    is_open = info["open"] if info else False
    ix, iy, iz = int(x), int(y), int(z)
    fx, fy, fz = float(x), float(y), float(z)

    if not is_open:
        if facing in ("north", "south"):
            box = (fx, fy + 4.0 / 16.0, fz + 6.0 / 16.0, fx + 1.0, fy + 1.0, fz + 10.0 / 16.0)
        else:
            box = (fx + 6.0 / 16.0, fy + 4.0 / 16.0, fz, fx + 10.0 / 16.0, fy + 1.0, fz + 1.0)
        return [(ix, iy, iz, "fence_gate", box)]
    else:
        if facing == "north":
            return [
                (ix, iy, iz, "fence_gate_w", (fx, fy + 4.0 / 16.0, fz, fx + 2.0 / 16.0, fy + 1.0, fz + 10.0 / 16.0)),
                (ix, iy, iz, "fence_gate_e", (fx + 14.0 / 16.0, fy + 4.0 / 16.0, fz, fx + 1.0, fy + 1.0, fz + 10.0 / 16.0)),
            ]
        elif facing == "south":
            return [
                (ix, iy, iz, "fence_gate_w", (fx, fy + 4.0 / 16.0, fz + 6.0 / 16.0, fx + 2.0 / 16.0, fy + 1.0, fz + 1.0)),
                (ix, iy, iz, "fence_gate_e", (fx + 14.0 / 16.0, fy + 4.0 / 16.0, fz + 6.0 / 16.0, fx + 1.0, fy + 1.0, fz + 1.0)),
            ]
        elif facing == "west":
            return [
                (ix, iy, iz, "fence_gate_n", (fx, fy + 4.0 / 16.0, fz, fx + 10.0 / 16.0, fy + 1.0, fz + 2.0 / 16.0)),
                (ix, iy, iz, "fence_gate_s", (fx, fy + 4.0 / 16.0, fz + 14.0 / 16.0, fx + 10.0 / 16.0, fy + 1.0, fz + 1.0)),
            ]
        else:
            return [
                (ix, iy, iz, "fence_gate_n", (fx + 6.0 / 16.0, fy + 4.0 / 16.0, fz, fx + 1.0, fy + 1.0, fz + 2.0 / 16.0)),
                (ix, iy, iz, "fence_gate_s", (fx + 6.0 / 16.0, fy + 4.0 / 16.0, fz + 14.0 / 16.0, fx + 1.0, fy + 1.0, fz + 1.0)),
            ]


def is_fence(block_or_id):
    """Return True if block_or_id is a fence block."""
    if not block_or_id:
        return False
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return False
    props = getattr(block, "properties", {})
    if props.get("fence") or getattr(block, "fence", False):
        return True
    b_id = getattr(block, "id", str(block_or_id))
    return "fence" in b_id and "gate" not in b_id


def can_fence_connect(block_or_id, direction=None, world=None, pos=None):
    """Return True if a fence should connect to the block at this position."""
    if not block_or_id:
        return False
    if is_sub_cell(block_or_id) and world is not None and pos is not None:
        sub = world.get_sub_cell(*pos)
        if sub is not None:
            if all(bool(o) for o in sub.octants):
                return True
            if direction == "north":
                return all(bool(sub.get_octant(ox, oy, 1)) for ox in (0, 1) for oy in (0, 1))
            elif direction == "south":
                return all(bool(sub.get_octant(ox, oy, 0)) for ox in (0, 1) for oy in (0, 1))
            elif direction == "west":
                return all(bool(sub.get_octant(1, oy, oz)) for oy in (0, 1) for oz in (0, 1))
            elif direction == "east":
                return all(bool(sub.get_octant(0, oy, oz)) for oy in (0, 1) for oz in (0, 1))
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return False
    if is_fence(block):
        return True
    if is_fence_gate(block):
        fg_info = get_fence_gate_info(block)
        if fg_info is not None:
            fg_facing = fg_info["facing"]
            if direction in ("west", "east"):
                return fg_facing in ("north", "south")
            elif direction in ("north", "south"):
                return fg_facing in ("east", "west")
        return True
    props = getattr(block, "properties", {})

    # Check double slab blocks (both horizontal double slabs and double vertical slabs)
    is_double_slab = bool(
        props.get("double_slab")
        or getattr(block, "double_slab", False)
        or props.get("double_vertical_slab")
        or getattr(block, "double_vertical_slab", False)
    )
    if is_double_slab:
        if not (props.get("is_vertical_slab_offset") and not props.get("capped")):
            return bool(block.solid)

    # Check vertical slab that has its back to the fence
    if direction is not None and is_vertical_slab(block):
        v_info = get_vertical_slab_info(block)
        if v_info is not None:
            if v_info["facing"] == OPPOSITE_FACING.get(direction):
                return bool(block.solid)
        return False

    # Connects to solid, full-cube blocks (e.g. wood planks, stone, etc.)
    return bool(block.solid and block.full_cube and not props.get("slab") and not props.get("vertical_slab"))


def fence_collision_boxes(world, x, y, z):
    """World-space collision boxes for a fence block at (x, y, z).
    Acts as 1.5 blocks tall!
    """
    x = float(x)
    y = float(y)
    z = float(z)
    h = 1.5

    # Center post (4x4 pixels centered)
    boxes = [(x + 6.0 / 16.0, y, z + 6.0 / 16.0, x + 10.0 / 16.0, y + h, z + 10.0 / 16.0)]

    if world is not None:
        ix, iy, iz = int(x), int(y), int(z)
        if can_fence_connect(world.get_block(ix, iy, iz - 1), "north", world, (ix, iy, iz - 1)):
            boxes.append((x + 6.0 / 16.0, y, z, x + 10.0 / 16.0, y + h, z + 6.0 / 16.0))
        if can_fence_connect(world.get_block(ix, iy, iz + 1), "south", world, (ix, iy, iz + 1)):
            boxes.append((x + 6.0 / 16.0, y, z + 10.0 / 16.0, x + 10.0 / 16.0, y + h, z + 1.0))
        if can_fence_connect(world.get_block(ix - 1, iy, iz), "west", world, (ix - 1, iy, iz)):
            boxes.append((x, y, z + 6.0 / 16.0, x + 6.0 / 16.0, y + h, z + 10.0 / 16.0))
        if can_fence_connect(world.get_block(ix + 1, iy, iz), "east", world, (ix + 1, iy, iz)):
            boxes.append((x + 10.0 / 16.0, y, z + 6.0 / 16.0, x + 1.0, y + h, z + 10.0 / 16.0))

    return boxes


def fence_raycast_boxes(world, x, y, z):
    """World-space selection/hit boxes for raycasting a fence block at (x, y, z).
    Height is 1.0 block tall so that placing a block above targets (x, y + 1, z).
    """
    ix, iy, iz = int(x), int(y), int(z)
    fx, fy, fz = float(x), float(y), float(z)

    boxes = [(ix, iy, iz, "fence_post", (fx + 6.0 / 16.0, fy, fz + 6.0 / 16.0, fx + 10.0 / 16.0, fy + 1.0, fz + 10.0 / 16.0))]

    if world is not None:
        if can_fence_connect(world.get_block(ix, iy, iz - 1), "north", world, (ix, iy, iz - 1)):
            boxes.append((ix, iy, iz, "fence_north", (fx + 6.0 / 16.0, fy, fz, fx + 10.0 / 16.0, fy + 1.0, fz + 6.0 / 16.0)))
        if can_fence_connect(world.get_block(ix, iy, iz + 1), "south", world, (ix, iy, iz + 1)):
            boxes.append((ix, iy, iz, "fence_south", (fx + 6.0 / 16.0, fy, fz + 10.0 / 16.0, fx + 10.0 / 16.0, fy + 1.0, fz + 1.0)))
        if can_fence_connect(world.get_block(ix - 1, iy, iz), "west", world, (ix - 1, iy, iz)):
            boxes.append((ix, iy, iz, "fence_west", (fx, fy, fz + 6.0 / 16.0, fx + 6.0 / 16.0, fy + 1.0, fz + 10.0 / 16.0)))
        if can_fence_connect(world.get_block(ix + 1, iy, iz), "east", world, (ix + 1, iy, iz)):
            boxes.append((ix, iy, iz, "fence_east", (fx + 10.0 / 16.0, fy, fz + 6.0 / 16.0, fx + 1.0, fy + 1.0, fz + 10.0 / 16.0)))

    return boxes


def get_fence_elements(world, wx, wy, wz, block):
    """Return model elements for a fence at (wx, wy, wz), including connecting rails."""
    elements = list(block.elements)
    if world is not None and hasattr(block, "fence_rails"):
        if can_fence_connect(world.get_block(wx, wy, wz - 1), "north", world, (wx, wy, wz - 1)):
            elements.extend(block.fence_rails["north"])
        if can_fence_connect(world.get_block(wx, wy, wz + 1), "south", world, (wx, wy, wz + 1)):
            elements.extend(block.fence_rails["south"])
        if can_fence_connect(world.get_block(wx - 1, wy, wz), "west", world, (wx - 1, wy, wz)):
            elements.extend(block.fence_rails["west"])
        if can_fence_connect(world.get_block(wx + 1, wy, wz), "east", world, (wx + 1, wy, wz)):
            elements.extend(block.fence_rails["east"])
    return elements


def is_wall(block_or_id):
    """Return True if block_or_id is a wall block."""
    if not block_or_id:
        return False
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return False
    props = getattr(block, "properties", {})
    if props.get("wall") or getattr(block, "wall", False):
        return True
    b_id = getattr(block, "id", str(block_or_id))
    return "wall" in b_id and "torch" not in b_id


def can_wall_connect(block_or_id, direction=None, world=None, pos=None):
    """Return True if a wall should connect to the block at this position."""
    if not block_or_id:
        return False
    if is_sub_cell(block_or_id) and world is not None and pos is not None:
        sub = world.get_sub_cell(*pos)
        if sub is not None:
            if all(bool(o) for o in sub.octants):
                return True
            if direction == "north":
                return all(bool(sub.get_octant(ox, oy, 1)) for ox in (0, 1) for oy in (0, 1))
            elif direction == "south":
                return all(bool(sub.get_octant(ox, oy, 0)) for ox in (0, 1) for oy in (0, 1))
            elif direction == "west":
                return all(bool(sub.get_octant(1, oy, oz)) for oy in (0, 1) for oz in (0, 1))
            elif direction == "east":
                return all(bool(sub.get_octant(0, oy, oz)) for oy in (0, 1) for oz in (0, 1))
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return False
    if is_wall(block):
        return True
    if is_fence_gate(block):
        fg_info = get_fence_gate_info(block)
        if fg_info is not None:
            fg_facing = fg_info["facing"]
            if direction in ("west", "east"):
                return fg_facing in ("north", "south")
            elif direction in ("north", "south"):
                return fg_facing in ("east", "west")
        return True
    props = getattr(block, "properties", {})

    # Check double slab blocks (both horizontal double slabs and double vertical slabs)
    is_double_slab = bool(
        props.get("double_slab")
        or getattr(block, "double_slab", False)
        or props.get("double_vertical_slab")
        or getattr(block, "double_vertical_slab", False)
    )
    if is_double_slab:
        if not (props.get("is_vertical_slab_offset") and not props.get("capped")):
            return bool(block.solid)

    # Check vertical slab that has its back to the wall
    if direction is not None and is_vertical_slab(block):
        v_info = get_vertical_slab_info(block)
        if v_info is not None:
            if v_info["facing"] == OPPOSITE_FACING.get(direction):
                return bool(block.solid)
        return False

    # Connects to solid, full-cube blocks (e.g. stone, planks, etc.)
    return bool(block.solid and block.full_cube and not props.get("slab") and not props.get("vertical_slab"))


def wall_collision_boxes(world, x, y, z):
    """World-space collision boxes for a wall block at (x, y, z).
    Acts as 1.5 blocks tall!
    """
    x = float(x)
    y = float(y)
    z = float(z)
    h = 1.5

    ix, iy, iz = int(x), int(y), int(z)
    n = can_wall_connect(world.get_block(ix, iy, iz - 1), "north", world, (ix, iy, iz - 1)) if world is not None else False
    s = can_wall_connect(world.get_block(ix, iy, iz + 1), "south", world, (ix, iy, iz + 1)) if world is not None else False
    w = can_wall_connect(world.get_block(ix - 1, iy, iz), "west", world, (ix - 1, iy, iz)) if world is not None else False
    e = can_wall_connect(world.get_block(ix + 1, iy, iz), "east", world, (ix + 1, iy, iz)) if world is not None else False

    above_b = world.get_block(ix, iy + 1, iz) if world is not None else None
    has_torch_above = False
    has_above_wall = False
    if above_b:
        if is_wall(above_b):
            has_above_wall = True
        elif above_b in BLOCKS:
            if BLOCKS[above_b].properties.get("torch"):
                has_torch_above = True
            elif getattr(BLOCKS[above_b], "solid", False):
                has_above_wall = True

    tall_n = has_above_wall or (is_wall(world.get_block(ix, iy + 1, iz - 1)) if world is not None else False)
    tall_s = has_above_wall or (is_wall(world.get_block(ix, iy + 1, iz + 1)) if world is not None else False)
    tall_w = has_above_wall or (is_wall(world.get_block(ix - 1, iy + 1, iz)) if world is not None else False)
    tall_e = has_above_wall or (is_wall(world.get_block(ix + 1, iy + 1, iz)) if world is not None else False)

    is_straight_ns = n and s and not w and not e
    is_straight_we = w and e and not n and not s
    is_straight = is_straight_ns or is_straight_we

    straight_height_diff = False
    if is_straight_we and (tall_w != tall_e):
        straight_height_diff = True
    elif is_straight_ns and (tall_n != tall_s):
        straight_height_diff = True

    has_post = (not is_straight) or has_torch_above or straight_height_diff or (not n and not s and not w and not e)

    if not has_post:
        if is_straight_we:
            return [(x, y, z + 5.0 / 16.0, x + 1.0, y + h, z + 11.0 / 16.0)]
        if is_straight_ns:
            return [(x + 5.0 / 16.0, y, z, x + 11.0 / 16.0, y + h, z + 1.0)]

    # Center post (8x8 pixels centered)
    boxes = [(x + 4.0 / 16.0, y, z + 4.0 / 16.0, x + 12.0 / 16.0, y + h, z + 12.0 / 16.0)]

    if n:
        boxes.append((x + 5.0 / 16.0, y, z, x + 11.0 / 16.0, y + h, z + 4.0 / 16.0))
    if s:
        boxes.append((x + 5.0 / 16.0, y, z + 12.0 / 16.0, x + 11.0 / 16.0, y + h, z + 1.0))
    if w:
        boxes.append((x, y, z + 5.0 / 16.0, x + 4.0 / 16.0, y + h, z + 11.0 / 16.0))
    if e:
        boxes.append((x + 12.0 / 16.0, y, z + 5.0 / 16.0, x + 1.0, y + h, z + 11.0 / 16.0))

    return boxes


def wall_raycast_boxes(world, x, y, z):
    """World-space selection/hit boxes for raycasting a wall block at (x, y, z).
    Height is 1.0 block tall so that placing a block above targets (x, y + 1, z).
    """
    ix, iy, iz = int(x), int(y), int(z)
    fx, fy, fz = float(x), float(y), float(z)

    n = can_wall_connect(world.get_block(ix, iy, iz - 1), "north", world, (ix, iy, iz - 1)) if world is not None else False
    s = can_wall_connect(world.get_block(ix, iy, iz + 1), "south", world, (ix, iy, iz + 1)) if world is not None else False
    w = can_wall_connect(world.get_block(ix - 1, iy, iz), "west", world, (ix - 1, iy, iz)) if world is not None else False
    e = can_wall_connect(world.get_block(ix + 1, iy, iz), "east", world, (ix + 1, iy, iz)) if world is not None else False

    above_b = world.get_block(ix, iy + 1, iz) if world is not None else None
    has_torch_above = False
    has_above_wall = False
    if above_b:
        if is_wall(above_b):
            has_above_wall = True
        elif above_b in BLOCKS:
            if BLOCKS[above_b].properties.get("torch"):
                has_torch_above = True
            elif getattr(BLOCKS[above_b], "solid", False):
                has_above_wall = True

    tall_n = has_above_wall or (is_wall(world.get_block(ix, iy + 1, iz - 1)) if world is not None else False)
    tall_s = has_above_wall or (is_wall(world.get_block(ix, iy + 1, iz + 1)) if world is not None else False)
    tall_w = has_above_wall or (is_wall(world.get_block(ix - 1, iy + 1, iz)) if world is not None else False)
    tall_e = has_above_wall or (is_wall(world.get_block(ix + 1, iy + 1, iz)) if world is not None else False)

    is_straight_ns = n and s and not w and not e
    is_straight_we = w and e and not n and not s
    is_straight = is_straight_ns or is_straight_we

    straight_height_diff = False
    if is_straight_we and (tall_w != tall_e):
        straight_height_diff = True
    elif is_straight_ns and (tall_n != tall_s):
        straight_height_diff = True

    has_post = (not is_straight) or has_torch_above or straight_height_diff or (not n and not s and not w and not e)

    if not has_post:
        if is_straight_we:
            top_y = fy + 1.0 if tall_w else fy + 14.0 / 16.0
            return [(ix, iy, iz, "wall_straight", (fx, fy, fz + 5.0 / 16.0, fx + 1.0, top_y, fz + 11.0 / 16.0))]
        if is_straight_ns:
            top_y = fy + 1.0 if tall_n else fy + 14.0 / 16.0
            return [(ix, iy, iz, "wall_straight", (fx + 5.0 / 16.0, fy, fz, fx + 11.0 / 16.0, top_y, fz + 1.0))]

    boxes = [(ix, iy, iz, "wall_post", (fx + 4.0 / 16.0, fy, fz + 4.0 / 16.0, fx + 12.0 / 16.0, fy + 1.0, fz + 12.0 / 16.0))]

    if n:
        top_y = fy + 1.0 if tall_n else fy + 14.0 / 16.0
        boxes.append((ix, iy, iz, "wall_north", (fx + 5.0 / 16.0, fy, fz, fx + 11.0 / 16.0, top_y, fz + 4.0 / 16.0)))
    if s:
        top_y = fy + 1.0 if tall_s else fy + 14.0 / 16.0
        boxes.append((ix, iy, iz, "wall_south", (fx + 5.0 / 16.0, fy, fz + 12.0 / 16.0, fx + 11.0 / 16.0, top_y, fz + 1.0)))
    if w:
        top_y = fy + 1.0 if tall_w else fy + 14.0 / 16.0
        boxes.append((ix, iy, iz, "wall_west", (fx, fy, fz + 5.0 / 16.0, fx + 4.0 / 16.0, top_y, fz + 11.0 / 16.0)))
    if e:
        top_y = fy + 1.0 if tall_e else fy + 14.0 / 16.0
        boxes.append((ix, iy, iz, "wall_east", (fx + 12.0 / 16.0, fy, fz + 5.0 / 16.0, fx + 1.0, top_y, fz + 11.0 / 16.0)))

    return boxes


def get_wall_elements(world, wx, wy, wz, block):
    """Return model elements for a wall at (wx, wy, wz), including connecting sides.
    Implements modern non-post wall sections on straight stretches without blocks above,
    and tall connections when walls/blocks are stacked vertically.
    """
    if world is None or not hasattr(block, "wall_sides"):
        return list(block.elements)

    n = can_wall_connect(world.get_block(wx, wy, wz - 1), "north", world, (wx, wy, wz - 1))
    s = can_wall_connect(world.get_block(wx, wy, wz + 1), "south", world, (wx, wy, wz + 1))
    w = can_wall_connect(world.get_block(wx - 1, wy, wz), "west", world, (wx - 1, wy, wz))
    e = can_wall_connect(world.get_block(wx + 1, wy, wz), "east", world, (wx + 1, wy, wz))

    above_b = world.get_block(wx, wy + 1, wz)
    has_torch_above = False
    has_above_wall = False
    if above_b:
        if is_wall(above_b):
            has_above_wall = True
        elif above_b in BLOCKS:
            if BLOCKS[above_b].properties.get("torch"):
                has_torch_above = True
            elif getattr(BLOCKS[above_b], "solid", False):
                has_above_wall = True

    tall_n = has_above_wall or is_wall(world.get_block(wx, wy + 1, wz - 1))
    tall_s = has_above_wall or is_wall(world.get_block(wx, wy + 1, wz + 1))
    tall_w = has_above_wall or is_wall(world.get_block(wx - 1, wy + 1, wz))
    tall_e = has_above_wall or is_wall(world.get_block(wx + 1, wy + 1, wz))

    is_straight_ns = n and s and not w and not e
    is_straight_we = w and e and not n and not s
    is_straight = is_straight_ns or is_straight_we

    straight_height_diff = False
    if is_straight_we and (tall_w != tall_e):
        straight_height_diff = True
    elif is_straight_ns and (tall_n != tall_s):
        straight_height_diff = True

    has_post = (not is_straight) or has_torch_above or straight_height_diff or (not n and not s and not w and not e)

    if not has_post:
        # Non-wall post section along straight stretch
        if is_straight_we:
            key = "straight_we_tall" if (tall_w and "straight_we_tall" in block.wall_sides) else "straight_we"
            if key in block.wall_sides:
                return list(block.wall_sides[key])
        if is_straight_ns:
            key = "straight_ns_tall" if (tall_n and "straight_ns_tall" in block.wall_sides) else "straight_ns"
            if key in block.wall_sides:
                return list(block.wall_sides[key])

    # Wall has a post
    elements = list(block.elements)

    if n:
        key = "north_tall" if (tall_n and "north_tall" in block.wall_sides) else "north"
        elements.extend(block.wall_sides[key])
    if s:
        key = "south_tall" if (tall_s and "south_tall" in block.wall_sides) else "south"
        elements.extend(block.wall_sides[key])
    if w:
        key = "west_tall" if (tall_w and "west_tall" in block.wall_sides) else "west"
        elements.extend(block.wall_sides[key])
    if e:
        key = "east_tall" if (tall_e and "east_tall" in block.wall_sides) else "east"
        elements.extend(block.wall_sides[key])

    return elements


_STAIR_STEP_QUADRANTS = {
    ("north", "straight"): ("NW", "NE"),
    ("north", "outer_left"): ("NW",),
    ("north", "outer_right"): ("NE",),
    ("north", "inner_left"): ("NW", "NE", "SW"),
    ("north", "inner_right"): ("NW", "NE", "SE"),

    ("south", "straight"): ("SW", "SE"),
    ("south", "outer_left"): ("SE",),
    ("south", "outer_right"): ("SW",),
    ("south", "inner_left"): ("SW", "SE", "NE"),
    ("south", "inner_right"): ("SW", "SE", "NW"),

    ("west", "straight"): ("NW", "SW"),
    ("west", "outer_left"): ("SW",),
    ("west", "outer_right"): ("NW",),
    ("west", "inner_left"): ("NW", "SW", "SE"),
    ("west", "inner_right"): ("NW", "SW", "NE"),

    ("east", "straight"): ("NE", "SE"),
    ("east", "outer_left"): ("NE",),
    ("east", "outer_right"): ("SE",),
    ("east", "inner_left"): ("NE", "SE", "NW"),
    ("east", "inner_right"): ("NE", "SE", "SW"),
}


def is_stairs(block_or_id):
    """Return True if block_or_id is a stairs block."""
    if not block_or_id:
        return False
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None:
        return False
    props = getattr(block, "properties", {})
    if props.get("stairs") or getattr(block, "stairs", False):
        return True
    b_id = getattr(block, "id", str(block_or_id))
    return "stairs" in b_id


def get_stairs_info(block_or_id):
    """Return stairs metadata dict or None if not stairs:
    {
        "is_stairs": True,
        "base_id": str,
        "facing": 'north' | 'south' | 'east' | 'west',
        "half": 'bottom' | 'top',
        "waterlogged": bool,
    }
    """
    if not is_stairs(block_or_id):
        return None
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    props = getattr(block, "properties", {})
    facing = props.get("facing")
    if not facing:
        b_id = getattr(block, "id", "")
        for f in ("north", "south", "east", "west"):
            if f"_{f}" in b_id:
                facing = f
                break
        if not facing:
            facing = "north"
    half = props.get("half")
    if not half:
        b_id = getattr(block, "id", "")
        half = "top" if "_top" in b_id else "bottom"

    return {
        "is_stairs": True,
        "base_id": getattr(block, "base_id", getattr(block, "id", str(block_or_id))),
        "facing": facing,
        "half": half,
        "waterlogged": bool(props.get("waterlogged", False)),
    }


def get_stairs_block(base_id, facing="north", half="bottom"):
    """Return variant ID for stairs given base ID, facing, and half."""
    if facing == "north" and half == "bottom":
        return base_id
    if half == "bottom":
        return f"{base_id}_{facing}"
    if facing == "north":
        return f"{base_id}_north_top"
    return f"{base_id}_{facing}_top"


def _can_take_shape(world, wx, wy, wz, check_dir, facing, half):
    dx, dz = {
        "north": (0, -1),
        "south": (0, 1),
        "west": (-1, 0),
        "east": (1, 0),
    }[check_dir]
    info = get_stairs_info(world.get_block(wx + dx, wy, wz + dz))
    if not info:
        return True
    return not (info["facing"] == facing and info["half"] == half)


def get_stair_shape(world, wx, wy, wz, facing, half):
    """Return stair shape: 'straight', 'inner_left', 'inner_right', 'outer_left', or 'outer_right'."""
    if world is None:
        return "straight"

    behind_pos = {
        "north": (wx, wy, wz - 1),
        "south": (wx, wy, wz + 1),
        "west": (wx - 1, wy, wz),
        "east": (wx + 1, wy, wz),
    }[facing]

    front_pos = {
        "north": (wx, wy, wz + 1),
        "south": (wx, wy, wz - 1),
        "west": (wx + 1, wy, wz),
        "east": (wx - 1, wy, wz),
    }[facing]

    behind_info = get_stairs_info(world.get_block(*behind_pos))
    front_info = get_stairs_info(world.get_block(*front_pos))

    # Outer corner: connected to perpendicular stair behind (high side)
    if behind_info and behind_info["half"] == half:
        b_facing = behind_info["facing"]
        if facing in ("north", "south") and b_facing in ("west", "east"):
            opp_dir = "east" if b_facing == "west" else "west"
            if _can_take_shape(world, wx, wy, wz, opp_dir, facing, half):
                if facing == "north":
                    return "outer_left" if b_facing == "west" else "outer_right"
                else:
                    return "outer_left" if b_facing == "east" else "outer_right"
        elif facing in ("west", "east") and b_facing in ("north", "south"):
            opp_dir = "south" if b_facing == "north" else "north"
            if _can_take_shape(world, wx, wy, wz, opp_dir, facing, half):
                if facing == "west":
                    return "outer_left" if b_facing == "south" else "outer_right"
                else:
                    return "outer_left" if b_facing == "north" else "outer_right"

    # Inner corner: connected to perpendicular stair in front (low side)
    if front_info and front_info["half"] == half:
        f_facing = front_info["facing"]
        if facing in ("north", "south") and f_facing in ("west", "east"):
            if _can_take_shape(world, wx, wy, wz, f_facing, facing, half):
                if facing == "north":
                    return "inner_left" if f_facing == "west" else "inner_right"
                else:
                    return "inner_left" if f_facing == "east" else "inner_right"
        elif facing in ("west", "east") and f_facing in ("north", "south"):
            if _can_take_shape(world, wx, wy, wz, f_facing, facing, half):
                if facing == "west":
                    return "inner_left" if f_facing == "south" else "inner_right"
                else:
                    return "inner_left" if f_facing == "north" else "inner_right"

    return "straight"


def _stair_boxes_local(facing, half, shape="straight"):
    """Return local 0..16 bounding boxes [(x0, y0, z0, x1, y1, z1), ...] for stair."""
    base_box = (0.0, 0.0, 0.0, 16.0, 8.0, 16.0) if half == "bottom" else (0.0, 8.0, 0.0, 16.0, 16.0, 16.0)
    y0, y1 = (8.0, 16.0) if half == "bottom" else (0.0, 8.0)
    quads = _STAIR_STEP_QUADRANTS.get((facing, shape), ("NW", "NE"))

    quad_boxes = {
        "NW": (0.0, y0, 0.0, 8.0, y1, 8.0),
        "NE": (8.0, y0, 0.0, 16.0, y1, 8.0),
        "SW": (0.0, y0, 8.0, 8.0, y1, 16.0),
        "SE": (8.0, y0, 8.0, 16.0, y1, 16.0),
    }

    if set(quads) == {"NW", "NE"}:
        step_boxes = [(0.0, y0, 0.0, 16.0, y1, 8.0)]
    elif set(quads) == {"SW", "SE"}:
        step_boxes = [(0.0, y0, 8.0, 16.0, y1, 16.0)]
    elif set(quads) == {"NW", "SW"}:
        step_boxes = [(0.0, y0, 0.0, 8.0, y1, 16.0)]
    elif set(quads) == {"NE", "SE"}:
        step_boxes = [(8.0, y0, 0.0, 16.0, y1, 16.0)]
    elif len(quads) == 1:
        step_boxes = [quad_boxes[quads[0]]]
    elif len(quads) == 3:
        if "NW" in quads and "NE" in quads:
            rem = (set(quads) - {"NW", "NE"}).pop()
            step_boxes = [(0.0, y0, 0.0, 16.0, y1, 8.0), quad_boxes[rem]]
        elif "SW" in quads and "SE" in quads:
            rem = (set(quads) - {"SW", "SE"}).pop()
            step_boxes = [(0.0, y0, 8.0, 16.0, y1, 16.0), quad_boxes[rem]]
        elif "NW" in quads and "SW" in quads:
            rem = (set(quads) - {"NW", "SW"}).pop()
            step_boxes = [(0.0, y0, 0.0, 8.0, y1, 16.0), quad_boxes[rem]]
        else:
            rem = (set(quads) - {"NE", "SE"}).pop()
            step_boxes = [(8.0, y0, 0.0, 16.0, y1, 16.0), quad_boxes[rem]]
    else:
        step_boxes = [quad_boxes[q] for q in quads]

    return [base_box] + step_boxes


def _make_stair_element(x0, y0, z0, x1, y1, z1, top_rect, bottom_rect, side_rect, omit_faces=()):
    faces = {}
    if "down" not in omit_faces:
        faces[FACE_INDEX["down"]] = _map_uv(bottom_rect, [x0, z0, x1, z1])
    if "up" not in omit_faces:
        faces[FACE_INDEX["up"]] = _map_uv(top_rect, [x0, z0, x1, z1])
    if "north" not in omit_faces:
        faces[FACE_INDEX["north"]] = _map_uv(side_rect, [16 - x1, 16 - y1, 16 - x0, 16 - y0])
    if "south" not in omit_faces:
        faces[FACE_INDEX["south"]] = _map_uv(side_rect, [x0, 16 - y1, x1, 16 - y0])
    if "west" not in omit_faces:
        faces[FACE_INDEX["west"]] = _map_uv(side_rect, [z0, 16 - y1, z1, 16 - y0])
    if "east" not in omit_faces:
        faces[FACE_INDEX["east"]] = _map_uv(side_rect, [16 - z1, 16 - y1, 16 - z0, 16 - y0])
    return {
        "from": [float(x0), float(y0), float(z0)],
        "to": [float(x1), float(y1), float(z1)],
        "faces": faces,
    }


def _make_stair_elements_for_state(facing, half, shape, top_rect, bottom_rect, side_rect):
    """Generate model elements for a stair state, omitting internal duplicate/touching faces."""
    base_box = (0.0, 0.0, 0.0, 16.0, 8.0, 16.0) if half == "bottom" else (0.0, 8.0, 0.0, 16.0, 16.0, 16.0)
    base_el = _make_stair_element(*base_box, top_rect, bottom_rect, side_rect)

    y0, y1 = (8.0, 16.0) if half == "bottom" else (0.0, 8.0)
    quads = _STAIR_STEP_QUADRANTS.get((facing, shape), ("NW", "NE"))

    quad_coords = {
        "NW": (0.0, 0.0, 8.0, 8.0),
        "NE": (8.0, 0.0, 16.0, 8.0),
        "SW": (0.0, 8.0, 8.0, 16.0),
        "SE": (8.0, 8.0, 16.0, 16.0),
    }

    step_elements = []
    if shape == "straight":
        if set(quads) == {"NW", "NE"}:
            s_box = (0.0, y0, 0.0, 16.0, y1, 8.0)
        elif set(quads) == {"SW", "SE"}:
            s_box = (0.0, y0, 8.0, 16.0, y1, 16.0)
        elif set(quads) == {"NW", "SW"}:
            s_box = (0.0, y0, 0.0, 8.0, y1, 16.0)
        else:
            s_box = (8.0, y0, 0.0, 16.0, y1, 16.0)
        omit = ("down",) if half == "bottom" else ("up",)
        step_elements.append(_make_stair_element(*s_box, top_rect, bottom_rect, side_rect, omit_faces=omit))
    else:
        for q in quads:
            x0, z0, x1, z1 = quad_coords[q]
            omit = set()
            if half == "bottom":
                omit.add("down")
            else:
                omit.add("up")
            if q == "NW":
                if "NE" in quads:
                    omit.add("east")
                if "SW" in quads:
                    omit.add("south")
            elif q == "NE":
                if "NW" in quads:
                    omit.add("west")
                if "SE" in quads:
                    omit.add("south")
            elif q == "SW":
                if "SE" in quads:
                    omit.add("east")
                if "NW" in quads:
                    omit.add("north")
            elif q == "SE":
                if "SW" in quads:
                    omit.add("west")
                if "NE" in quads:
                    omit.add("north")
            step_elements.append(_make_stair_element(x0, y0, z0, x1, y1, z1, top_rect, bottom_rect, side_rect, omit_faces=omit))

    return [base_el] + step_elements


def get_stair_elements(world, wx, wy, wz, block):
    """Return model elements for a stair at (wx, wy, wz), adapting to corner connections."""
    info = get_stairs_info(block)
    facing = info["facing"] if info else "north"
    half = info["half"] if info else "bottom"
    shape = get_stair_shape(world, wx, wy, wz, facing, half)
    if shape == "straight" and hasattr(block, "elements") and block.elements:
        return list(block.elements)

    rects = getattr(block, "stair_rects", None)
    if not rects:
        return list(block.elements) if (hasattr(block, "elements") and block.elements) else []
    top_rect, bottom_rect, side_rect = rects

    return _make_stair_elements_for_state(facing, half, shape, top_rect, bottom_rect, side_rect)


def stair_collision_boxes(world, x, y, z, block):
    """World-space collision boxes for a stair block at (x, y, z)."""
    info = get_stairs_info(block)
    facing = info["facing"] if info else "north"
    half = info["half"] if info else "bottom"
    shape = get_stair_shape(world, int(x), int(y), int(z), facing, half)
    local_boxes = _stair_boxes_local(facing, half, shape)
    fx, fy, fz = float(x), float(y), float(z)
    return [
        (fx + b[0] / 16.0, fy + b[1] / 16.0, fz + b[2] / 16.0, fx + b[3] / 16.0, fy + b[4] / 16.0, fz + b[5] / 16.0)
        for b in local_boxes
    ]


def stair_raycast_boxes(world, x, y, z, block):
    """World-space selection/hit boxes for raycasting a stair block at (x, y, z)."""
    c_boxes = stair_collision_boxes(world, x, y, z, block)
    ix, iy, iz = int(x), int(y), int(z)
    boxes = []
    for idx, c_box in enumerate(c_boxes):
        part = "base" if idx == 0 else f"step_{idx}"
        boxes.append((ix, iy, iz, part, c_box))
    return boxes


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
    vo = get_vertical_slab_offset_block(block_id)
    if vo is not None:
        return vo
    if is_offset_block(block_id):
        info = get_offset_info(block_id)
        cand = f"{info['base_id']}_offset"
        return cand if cand in BLOCKS else block_id
    cand = f"{block_id}_offset"
    return cand if cand in BLOCKS and is_offset_block(cand) else None


def get_capped_offset_block(offset_block_or_id, slab_block_or_id):
    if is_vertical_slab_offset(offset_block_or_id):
        return get_capped_vertical_slab_offset(offset_block_or_id, slab_block_or_id)
    o_info = get_offset_info(offset_block_or_id)
    base_id = o_info["base_id"] if o_info is not None else str(offset_block_or_id)
    s_info = get_slab_info(slab_block_or_id)
    slab_base = s_info["base_id"] if s_info is not None else str(slab_block_or_id)
    cand = f"{base_id}_offset_{slab_base}"
    if cand in BLOCKS:
        return cand
    cand2 = f"{offset_block_or_id}_{slab_base}"
    return cand2 if cand2 in BLOCKS else None


def is_vertical_slab_offset(block_or_id):
    """Return True if block_or_id is a vertical slab offset block (uncapped or capped)."""
    if not block_or_id:
        return False
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None or not isinstance(block, Block):
        return False
    props = block.properties
    if not props.get("offset_block"):
        return False
    if props.get("double_vertical_slab"):
        return False
    return bool(props.get("is_vertical_slab_offset") or props.get("vertical_slab"))


def get_vertical_slab_offset_info(block_or_id):
    """Return vertical slab offset metadata dict or None if not a vertical slab offset:
    {
        "is_vertical_slab_offset": True,
        "base_id": str,
        "canonical_id": str,
        "capped": bool,
        "top_slab": str | None,
        "facing": str,
    }
    """
    if not block_or_id:
        return None
    block = BLOCKS.get(block_or_id) if isinstance(block_or_id, str) else block_or_id
    if block is None or not isinstance(block, Block):
        return None
    if not is_vertical_slab_offset(block):
        return None
    props = block.properties
    if props.get("double_vertical_slab"):
        return None
    base_id = str(props.get("offset_base", getattr(block, "base_id", block.id)))
    capped = bool(props.get("capped", False))
    top_slab = props.get("top_slab") if capped else None
    facing = str(props.get("facing", "south")).lower()
    if facing not in FACING_DIRECTIONS:
        facing = "south"
    return {
        "is_vertical_slab_offset": True,
        "base_id": base_id,
        "canonical_id": getattr(block, "base_id", base_id),
        "capped": capped,
        "top_slab": top_slab,
        "facing": facing,
    }


def get_vertical_slab_offset_block(v_slab_block_or_id):
    """Return the uncapped vertical slab offset block ID for a vertical slab, or None."""
    if not v_slab_block_or_id:
        return None
    vo_info = get_vertical_slab_offset_info(v_slab_block_or_id)
    if vo_info is not None:
        base = vo_info["base_id"]
        facing = vo_info["facing"]
        if facing == "south":
            cand = f"{base}_offset"
            if cand in BLOCKS:
                return cand
        cand = f"{base}_{facing}_offset"
        return cand if cand in BLOCKS else getattr(BLOCKS.get(v_slab_block_or_id), "id", str(v_slab_block_or_id))
    v_info = get_vertical_slab_info(v_slab_block_or_id)
    if v_info is None:
        return None
    facing = v_info["facing"]
    base = v_info["base_id"]
    if facing == "south":
        cand = f"{base}_offset"
        if cand in BLOCKS:
            return cand
    cand = f"{v_info['id']}_offset"
    if cand in BLOCKS:
        return cand
    cand2 = f"{base}_{facing}_offset"
    return cand2 if cand2 in BLOCKS else None


def get_capped_vertical_slab_offset(offset_block_or_id, slab_block_or_id):
    """Return the capped vertical slab offset block ID, or None.
    
    Caps an offset vertical slab with a horizontal slab block.
    """
    o_info = get_vertical_slab_offset_info(offset_block_or_id)
    if o_info is None:
        return None
    base_id = o_info["base_id"]
    facing = o_info["facing"]
    s_info = get_slab_info(slab_block_or_id)
    if s_info is None:
        return None
    slab_base = s_info["base_id"]
    if facing == "south":
        cand = f"{base_id}_offset_{slab_base}"
        if cand in BLOCKS:
            return cand
        cand_s = f"{base_id}_south_offset_{slab_base}"
        if cand_s in BLOCKS:
            return cand_s
    cand = f"{base_id}_{facing}_offset_{slab_base}"
    if cand in BLOCKS:
        return cand
    return None


def has_half_offset_top(block_or_id):
    """Returns True if `block_or_id` has its top solid surface at y + 0.5 (bottom single slab or uncapped offset block)."""
    if not block_or_id or is_vertical_slab_offset(block_or_id):
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
    if is_stairs(block):
        st_info = get_stairs_info(block)
        half = st_info["half"] if st_info else "bottom"
        return get_stairs_block(base, facing=val, half=half)
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
    if block.properties.get("back_torch"):
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
    if cand in BLOCKS:
        return cand
    cand2 = f"{offset_block_or_id}_{slab_base}"
    return cand2 if cand2 in BLOCKS else None


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
    v_off_info = get_vertical_slab_offset_info(block)
    if v_off_info is not None:
        if mount == "up":
            return bool(v_off_info["capped"])
        return True
    s_info = get_slab_info(block)
    if s_info is not None:
        if s_info["double"]:
            return True
        if mount == "up":
            return s_info["half"] == "top"
        return s_info["half"] == "bottom"
    v_info = get_vertical_slab_info(block)
    if v_info is not None:
        # A single vertical slab only offers half a cell of support. Torches
        # can hang on the front long face and the two short sides, but the
        # back long face is inside the cell and uses a combined block instead.
        if mount in ("up", "up_offset"):
            return False
        return mount != OPPOSITE_FACING[v_info["facing"]]
    if is_fence(block) or is_wall(block):
        return mount == "up"
    if is_stairs(block):
        st_info = get_stairs_info(block)
        return mount == "up" and bool(st_info and st_info["half"] == "top")
    return bool(block.full_cube)


def placement_block(block_id, normal, yaw=None, local_y=None):
    block = BLOCKS.get(block_id)
    if block is None:
        return block_id
    if is_stairs(block):
        if yaw is not None:
            facing = player_facing_from_yaw(yaw)
        else:
            nx, ny, nz = normal
            if abs(nx) > 0:
                facing = "east" if nx > 0 else "west"
            elif abs(nz) > 0:
                facing = "south" if nz > 0 else "north"
            else:
                facing = "north"
        nx, ny, nz = normal
        is_top = (ny < 0) or (ny == 0 and local_y is not None and local_y >= 0.5)
        half = "top" if is_top else "bottom"
        base = getattr(block, "base_id", block.id)
        return get_stairs_block(base, facing=facing, half=half)
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
    if is_fence_gate(block):
        if yaw is not None:
            facing = player_facing_from_yaw(yaw)
        else:
            nx, ny, nz = normal
            if abs(nx) > 0:
                facing = "east" if nx > 0 else "west"
            elif abs(nz) > 0:
                facing = "south" if nz > 0 else "north"
            else:
                facing = "north"
        base = getattr(block, "base_id", block.id)
        return get_fence_gate_block(base, facing=facing, open=False)
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
    if is_vertical_slab(block) or is_vertical_slab_offset(block):
        if yaw is not None:
            facing = player_facing_from_yaw(yaw)
            return rotation_variant(block.base_id, facing)
        else:
            nx, ny, nz = normal
            if abs(nx) > 0:
                facing = "east" if nx > 0 else "west"
            elif abs(nz) > 0:
                facing = "south" if nz > 0 else "north"
            else:
                facing = "north"
            return rotation_variant(block.base_id, facing)
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


_FACE_ALIASES = {
    "up": ("up", "top"),
    "down": ("down", "bottom"),
    "north": ("north", "back"),
    "south": ("south", "front"),
    "east": ("east", "right"),
    "west": ("west", "left"),
}


def _resolve_texture(token, mapping, block_name, face_name=None):
    if face_name in _FACE_ALIASES:
        for alias in _FACE_ALIASES[face_name]:
            if alias in mapping:
                return mapping[alias]

    if str(token).startswith("#"):
        key = token[1:]
        if key in mapping:
            return mapping[key]

        # Contextual face fallbacks if the model's token was not in mapping
        if face_name in ("north", "south", "east", "west") and "side" in mapping:
            return mapping["side"]
        if face_name == "down":
            if "bottom" in mapping:
                return mapping["bottom"]
            if "top" in mapping:
                return mapping["top"]
        if face_name == "up" and "top" in mapping:
            return mapping["top"]

        # Universal fallback for any face
        if "texture" in mapping:
            return mapping["texture"]
        if "all" in mapping:
            return mapping["all"]

        raise KeyError(f"{block_name} model uses #{key}, which is not in its textures")

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

    for m in range(16):
        ctm_name = f"glass_{m}"
        if ctm_name not in texture_names and (ASSETS / "textures" / f"{ctm_name}.png").is_file():
            texture_names.append(ctm_name)

    images = {}
    for texture_name in texture_names:
        path = ASSETS / "textures" / f"{texture_name}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        images[texture_name] = read_png(path)
    atlas, rects = _pack_atlas(images)

    global GLASS_CTM_UVS, GLASS_CTM_RECTS
    glass_ctm_uvs = []
    glass_ctm_rects = {}
    for m in range(16):
        c_name = f"glass_{m}"
        if c_name in rects:
            glass_ctm_rects[m] = rects[c_name]
            glass_ctm_uvs.append(_map_uv(rects[c_name], [0, 0, 16, 16]))
        elif "glass" in rects:
            glass_ctm_rects[m] = rects["glass"]
            glass_ctm_uvs.append(_map_uv(rects["glass"], [0, 0, 16, 16]))
    GLASS_CTM_UVS = tuple(glass_ctm_uvs)
    GLASS_CTM_RECTS = glass_ctm_rects

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
            face_uv_specs = {}
            texture_rects = {}
            for face_name, spec in element["faces"].items():
                if face_name not in FACE_INDEX:
                    raise ValueError(f"{entry['id']} has an unknown face {face_name}")
                texture_name = _resolve_texture(spec["texture"], textures, entry["id"], face_name=face_name)
                if texture_name not in rects:
                    raise KeyError(f"{entry['id']} texture {texture_name} was not loaded")
                spec_uv = spec.get("uv", [0, 0, 16, 16])
                uv = _map_uv(rects[texture_name], spec_uv)
                index = FACE_INDEX[face_name]
                faces[index] = uv
                face_uv_specs[index] = spec_uv
                texture_rects[index] = rects[texture_name]
                if face_uvs[index] is None:
                    face_uvs[index] = uv
            resolved.append({"from": origin, "to": extent, "faces": faces, "face_uv_specs": face_uv_specs, "texture_rects": texture_rects})
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
        is_vertical_slab_type = entry.get("type") == "vertical_slab" or entry.get("model") == "vertical_slab"
        is_door_type = entry.get("type") == "door" or properties.get("door", False)
        is_trapdoor_type = entry.get("type") == "trapdoor" or properties.get("trapdoor", False)
        is_fence_gate_type = entry.get("type") == "fence_gate" or properties.get("fence_gate", False) or entry.get("model") == "fence_gate"
        is_fence_type = (entry.get("type") == "fence" or properties.get("fence", False) or entry.get("model") == "fence_post") and not is_fence_gate_type
        is_wall_type = entry.get("type") == "wall" or properties.get("wall", False) or entry.get("model") == "wall_post"
        is_stairs_type = entry.get("type") == "stairs" or properties.get("stairs", False) or entry.get("model") == "stairs"
        rotation_type = properties.get("rotation") or entry.get("rotation")
        if not rotation_type and entry.get("model") == "orientable":
            rotation_type = "horizontal"
        if not rotation_type and is_vertical_slab_type:
            rotation_type = "horizontal"
        base_props = dict(properties)
        if is_slab_type:
            base_props["slab"] = True
            base_props["half"] = "bottom"
            base_props["waterlogged"] = False
            full_cube = False
        if is_vertical_slab_type:
            base_props["vertical_slab"] = True
            base_props["rotation"] = "horizontal"
            base_props["facing"] = properties.get("facing", "south")
            full_cube = False
        if is_door_type:
            base_props["door"] = True
            base_props["half"] = properties.get("half", "lower")
            base_props["facing"] = properties.get("facing", "south")
            base_props["open"] = properties.get("open", False)
            full_cube = False
        if is_trapdoor_type:
            base_props["trapdoor"] = True
            base_props["half"] = properties.get("half", "bottom")
            base_props["facing"] = properties.get("facing", "north")
            base_props["open"] = properties.get("open", False)
            full_cube = False
        if is_fence_gate_type:
            base_props["fence_gate"] = True
            base_props["facing"] = properties.get("facing", "north")
            base_props["open"] = properties.get("open", False)
            full_cube = False
        if is_fence_type:
            base_props["fence"] = True
            full_cube = False
        if is_wall_type:
            base_props["wall"] = True
            full_cube = False
        if is_stairs_type:
            base_props["stairs"] = True
            base_props["facing"] = properties.get("facing", "north")
            base_props["half"] = properties.get("half", "bottom")
            base_props["rotation"] = "horizontal"
            full_cube = False
        tint_val = entry.get("tint") or properties.get("tint")
        if tint_val is None and entry["id"] in ("grass", "muddy_grass"):
            tint_val = "grass"
        if tint_val:
            base_props["tint"] = tint_val
        if rotation_type in ("axis", "xyz") or "axis" in properties:
            base_props["rotation"] = "axis"
            base_props["axis"] = "y"
        elif (rotation_type in ("horizontal", "cardinal", "facing") or "facing" in properties) and not is_stairs_type:
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
        sound_val = entry.get("sound") or properties.get("sound")
        sounds_val = entry.get("sounds") or properties.get("sounds")
        if sound_val:
            base_props["sound"] = sound_val
        if sounds_val:
            base_props["sounds"] = sounds_val

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
            sound=sound_val,
            sounds=sounds_val,
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

        elif (rotation_type in ("horizontal", "cardinal", "facing") or "facing" in properties) and not is_stairs_type:
            for facing in ("north", "east", "west"):
                variant_id = f"{block.id}_{facing}"
                if is_vertical_slab_type:
                    var_resolved, var_face_uvs = _rotate_vertical_slab_elements(resolved, face_uvs, facing)
                    var_props = dict(base_props)
                    var_props["rotation"] = "horizontal"
                    var_props["facing"] = facing
                    var_block = Block(
                        variant_id,
                        f"{name} ({facing.capitalize()})",
                        block.solid,
                        False,
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
                    continue
                var_props = dict(properties)
                var_props["rotation"] = "horizontal"
                var_props["facing"] = facing
                face_map = _VERTICAL_SLAB_FACE_MAPS[facing]
                var_face_uvs = [face_uvs[old] for old in face_map]

                var_element = {
                    "from": [0.0, 0.0, 0.0],
                    "to": [16.0, 16.0, 16.0],
                    "faces": {i: var_face_uvs[i] for i in range(6)},
                }
                if resolved and "texture_rects" in resolved[0]:
                    var_element["texture_rects"] = {
                        i: resolved[0]["texture_rects"][face_map[i]]
                        for i in range(6)
                        if face_map[i] in resolved[0]["texture_rects"]
                    }
                if resolved and "face_uv_specs" in resolved[0]:
                    var_element["face_uv_specs"] = {
                        i: resolved[0]["face_uv_specs"][face_map[i]]
                        for i in range(6)
                        if face_map[i] in resolved[0]["face_uv_specs"]
                    }
                var_resolved = [var_element]
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

        if is_door_type:
            lower_tex = textures.get("lower", entry["id"])
            upper_tex = textures.get("upper", entry["id"])
            edge_tex = textures.get("edge") or textures.get("side")
            if not edge_tex or edge_tex not in rects:
                if "planks" in rects and "wood" in entry["id"]:
                    edge_tex = "planks"
                elif lower_tex in rects:
                    edge_tex = lower_tex
                else:
                    edge_tex = next(iter(rects))

            uv_edge_h = _map_uv(rects[edge_tex], [0, 0, 16, 3])
            uv_edge_h_z = _map_uv(rects[edge_tex], [0, 0, 3, 16])
            uv_edge_v0 = _map_uv(rects[edge_tex], [0, 0, 3, 16])
            uv_edge_v1 = _map_uv(rects[edge_tex], [13, 0, 16, 16])

            tex_map = {
                "lower": lower_tex,
                "upper": upper_tex,
            }
            for half in ("lower", "upper"):
                tex_name = tex_map.get(half, lower_tex)
                if tex_name not in rects:
                    tex_name = lower_tex if lower_tex in rects else next(iter(rects))
                uv_main_front = _map_uv(rects[tex_name], [0, 0, 16, 16])
                uv_main_back = _map_uv(rects[tex_name], [16, 0, 0, 16])

                for facing in ("north", "south", "east", "west"):
                    for hinge in ("left", "right"):
                        for is_open in (False, True):
                            var_id = get_door_block(block.id, half, facing, is_open, hinge)
                            if var_id == block.id:
                                continue
                            var_elements = _get_door_elements(
                                uv_main_front,
                                uv_main_back,
                                uv_edge_h,
                                uv_edge_h_z,
                                uv_edge_v0,
                                uv_edge_v1,
                                facing,
                                is_open,
                                hinge,
                            )
                            var_face_uvs = [
                                var_elements[0]["faces"].get(i, uv_main_front)
                                for i in range(len(FACE_INDEX))
                            ]
                            var_props = dict(base_props)
                            var_props["door"] = True
                            var_props["half"] = half
                            var_props["facing"] = facing
                            var_props["hinge"] = hinge
                            var_props["open"] = is_open
                            open_str = " (Open)" if is_open else ""
                            hinge_str = " Right" if hinge == "right" else ""
                            var_name = f"{name} {half.capitalize()} ({facing.capitalize()}){hinge_str}{open_str}"
                            var_block = Block(
                                var_id,
                                var_name,
                                True,
                                False,
                                var_face_uvs,
                                var_elements,
                                cutout=True,
                                opacity=1.0,
                                tabs=(),
                                properties=var_props,
                                base_id=block.id,
                                is_variant=True,
                                hardness=block.hardness,
                                tool=block.tool,
                                mining_level=block.mining_level,
                                loot_table=block.loot_table,
                                sound=sound_val,
                                sounds=sounds_val,
                            )
                            _register_block(var_id, var_block)
                            if hinge == "left":
                                left_alias = f"{block.id}_{half}_{facing}_left{'_open' if is_open else ''}"
                                _register_block(left_alias, var_block)

        if is_trapdoor_type:
            tex_name = textures.get("texture") or textures.get("top") or entry["id"]
            edge_tex = textures.get("edge") or textures.get("side") or textures.get("planks")
            if not edge_tex or edge_tex not in rects:
                if "planks" in rects and "wood" in entry["id"]:
                    edge_tex = "planks"
                elif tex_name in rects:
                    edge_tex = tex_name
                else:
                    edge_tex = next(iter(rects))

            if tex_name not in rects:
                tex_name = entry["id"] if entry["id"] in rects else next(iter(rects))

            uv_main = _map_uv(rects[tex_name], [0, 0, 16, 16])
            uv_edge_h = _map_uv(rects[edge_tex], [0, 0, 16, 3])
            uv_edge_h_z = _map_uv(rects[edge_tex], [0, 0, 3, 16])
            uv_edge_v0 = _map_uv(rects[edge_tex], [0, 0, 3, 16])
            uv_edge_v1 = _map_uv(rects[edge_tex], [13, 0, 16, 16])

            for half in ("bottom", "top"):
                for facing in ("north", "south", "east", "west"):
                    for is_open in (False, True):
                        var_id = get_trapdoor_block(block.id, half, facing, is_open)
                        if var_id == block.id:
                            continue
                        var_elements = _get_trapdoor_elements(
                            uv_main,
                            uv_edge_h,
                            uv_edge_h_z,
                            uv_edge_v0,
                            uv_edge_v1,
                            half,
                            facing,
                            is_open,
                        )
                        var_face_uvs = [
                            var_elements[0]["faces"].get(i, uv_main)
                            for i in range(len(FACE_INDEX))
                        ]
                        var_props = dict(base_props)
                        var_props["trapdoor"] = True
                        var_props["half"] = half
                        var_props["facing"] = facing
                        var_props["open"] = is_open
                        open_str = " (Open)" if is_open else ""
                        var_name = f"{name} {half.capitalize()} ({facing.capitalize()}){open_str}"
                        var_block = Block(
                            var_id,
                            var_name,
                            True,
                            False,
                            var_face_uvs,
                            var_elements,
                            cutout=True,
                            opacity=1.0,
                            tabs=(),
                            properties=var_props,
                            base_id=block.id,
                            is_variant=True,
                            hardness=block.hardness,
                            tool=block.tool,
                            mining_level=block.mining_level,
                            loot_table=block.loot_table,
                            sound=sound_val,
                            sounds=sounds_val,
                        )
                        _register_block(var_id, var_block)

        if is_fence_gate_type:
            tex_name = textures.get("texture") or textures.get("side") or textures.get("top") or entry["id"]
            if tex_name not in rects:
                tex_name = "planks" if "planks" in rects else next(iter(rects))
            tex_rect = rects[tex_name]

            for facing in ("north", "south", "east", "west"):
                for is_open in (False, True):
                    var_id = get_fence_gate_block(block.id, facing, is_open)
                    var_elements = _get_fence_gate_elements(tex_rect, facing, is_open)
                    var_face_uvs = [
                        var_elements[0]["faces"].get(i, _map_uv(tex_rect, [0, 0, 16, 16]))
                        for i in range(len(FACE_INDEX))
                    ]
                    var_props = dict(base_props)
                    var_props["fence_gate"] = True
                    var_props["facing"] = facing
                    var_props["open"] = is_open
                    open_str = " (Open)" if is_open else ""
                    var_name = f"{name} ({facing.capitalize()}){open_str}"
                    var_block = Block(
                        var_id,
                        var_name,
                        True,
                        False,
                        var_face_uvs,
                        var_elements,
                        cutout=True,
                        opacity=1.0,
                        tabs=(),
                        properties=var_props,
                        base_id=block.id,
                        is_variant=True,
                        hardness=block.hardness,
                        tool=block.tool,
                        mining_level=block.mining_level,
                        loot_table=block.loot_table,
                        sound=sound_val,
                        sounds=sounds_val,
                    )
                    _register_block(var_id, var_block)
                    if facing == "north" and not is_open:
                        base_gate = Block(
                            block.id,
                            name,
                            True,
                            False,
                            var_face_uvs,
                            var_elements,
                            cutout=True,
                            opacity=1.0,
                            tabs=block.tabs,
                            properties=var_props,
                            base_id=block.id,
                            is_variant=False,
                            hardness=block.hardness,
                            tool=block.tool,
                            mining_level=block.mining_level,
                            loot_table=block.loot_table,
                            sound=sound_val,
                            sounds=sounds_val,
                        )
                        _register_block(block.id, base_gate)

        if is_stairs_type:
            top_tex = textures.get("top") or textures.get("up") or textures.get("texture") or textures.get("side") or entry["id"]
            bottom_tex = textures.get("bottom") or textures.get("down") or top_tex
            side_tex = textures.get("side") or top_tex
            if top_tex not in rects:
                top_tex = next(iter(rects))
            if bottom_tex not in rects:
                bottom_tex = top_tex
            if side_tex not in rects:
                side_tex = top_tex

            top_rect = rects[top_tex]
            bottom_rect = rects[bottom_tex]
            side_rect = rects[side_tex]
            stair_rects = (top_rect, bottom_rect, side_rect)
            block.stair_rects = stair_rects

            for half in ("bottom", "top"):
                for facing in ("north", "south", "east", "west"):
                    var_id = get_stairs_block(block.id, facing, half)
                    if var_id == block.id:
                        continue
                    var_elements = _make_stair_elements_for_state(
                        facing, half, "straight", top_rect, bottom_rect, side_rect
                    )
                    var_props = dict(base_props)
                    var_props["facing"] = facing
                    var_props["half"] = half
                    var_props["rotation"] = "horizontal"
                    half_str = " (Top)" if half == "top" else ""
                    var_name = f"{name} ({facing.capitalize()}){half_str}"

                    var_block = Block(
                        var_id,
                        var_name,
                        block.solid,
                        False,
                        face_uvs,
                        var_elements,
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
                        sound=sound_val,
                        sounds=sounds_val,
                    )
                    var_block.stair_rects = stair_rects
                    _register_block(var_id, var_block)

            top_alias = f"{block.id}_top"
            north_top = f"{block.id}_north_top"
            if north_top in blocks and top_alias not in blocks:
                blocks[top_alias] = blocks[north_top]

        if is_fence_type:
            tex_name = textures.get("texture") or textures.get("side") or textures.get("top") or entry["id"]
            if tex_name not in rects:
                tex_name = "planks" if "planks" in rects else next(iter(rects))

            uv_side = _map_uv(rects[tex_name], [0, 0, 6, 3])
            uv_top = _map_uv(rects[tex_name], [7, 0, 9, 6])
            uv_end = _map_uv(rects[tex_name], [7, 0, 9, 3])
            uv_top_we = _map_uv(rects[tex_name], [0, 7, 6, 9])

            block.fence_rails = {
                "north": [
                    {
                        "from": [7.0, 12.0, 0.0],
                        "to": [9.0, 15.0, 6.0],
                        "faces": {
                            FACE_INDEX["north"]: uv_end,
                            FACE_INDEX["south"]: uv_end,
                            FACE_INDEX["west"]: uv_side,
                            FACE_INDEX["east"]: uv_side,
                            FACE_INDEX["up"]: uv_top,
                            FACE_INDEX["down"]: uv_top,
                        },
                    },
                    {
                        "from": [7.0, 6.0, 0.0],
                        "to": [9.0, 9.0, 6.0],
                        "faces": {
                            FACE_INDEX["north"]: uv_end,
                            FACE_INDEX["south"]: uv_end,
                            FACE_INDEX["west"]: uv_side,
                            FACE_INDEX["east"]: uv_side,
                            FACE_INDEX["up"]: uv_top,
                            FACE_INDEX["down"]: uv_top,
                        },
                    },
                ],
                "south": [
                    {
                        "from": [7.0, 12.0, 10.0],
                        "to": [9.0, 15.0, 16.0],
                        "faces": {
                            FACE_INDEX["north"]: uv_end,
                            FACE_INDEX["south"]: uv_end,
                            FACE_INDEX["west"]: uv_side,
                            FACE_INDEX["east"]: uv_side,
                            FACE_INDEX["up"]: uv_top,
                            FACE_INDEX["down"]: uv_top,
                        },
                    },
                    {
                        "from": [7.0, 6.0, 10.0],
                        "to": [9.0, 9.0, 16.0],
                        "faces": {
                            FACE_INDEX["north"]: uv_end,
                            FACE_INDEX["south"]: uv_end,
                            FACE_INDEX["west"]: uv_side,
                            FACE_INDEX["east"]: uv_side,
                            FACE_INDEX["up"]: uv_top,
                            FACE_INDEX["down"]: uv_top,
                        },
                    },
                ],
                "west": [
                    {
                        "from": [0.0, 12.0, 7.0],
                        "to": [6.0, 15.0, 9.0],
                        "faces": {
                            FACE_INDEX["west"]: uv_end,
                            FACE_INDEX["east"]: uv_end,
                            FACE_INDEX["north"]: uv_side,
                            FACE_INDEX["south"]: uv_side,
                            FACE_INDEX["up"]: uv_top_we,
                            FACE_INDEX["down"]: uv_top_we,
                        },
                    },
                    {
                        "from": [0.0, 6.0, 7.0],
                        "to": [6.0, 9.0, 9.0],
                        "faces": {
                            FACE_INDEX["west"]: uv_end,
                            FACE_INDEX["east"]: uv_end,
                            FACE_INDEX["north"]: uv_side,
                            FACE_INDEX["south"]: uv_side,
                            FACE_INDEX["up"]: uv_top_we,
                            FACE_INDEX["down"]: uv_top_we,
                        },
                    },
                ],
                "east": [
                    {
                        "from": [10.0, 12.0, 7.0],
                        "to": [16.0, 15.0, 9.0],
                        "faces": {
                            FACE_INDEX["west"]: uv_end,
                            FACE_INDEX["east"]: uv_end,
                            FACE_INDEX["north"]: uv_side,
                            FACE_INDEX["south"]: uv_side,
                            FACE_INDEX["up"]: uv_top_we,
                            FACE_INDEX["down"]: uv_top_we,
                        },
                    },
                    {
                        "from": [10.0, 6.0, 7.0],
                        "to": [16.0, 9.0, 9.0],
                        "faces": {
                            FACE_INDEX["west"]: uv_end,
                            FACE_INDEX["east"]: uv_end,
                            FACE_INDEX["north"]: uv_side,
                            FACE_INDEX["south"]: uv_side,
                            FACE_INDEX["up"]: uv_top_we,
                            FACE_INDEX["down"]: uv_top_we,
                        },
                    },
                ],
            }
            block.inventory_elements = [
                {
                    "from": [0.0, 0.0, 6.0],
                    "to": [4.0, 16.0, 10.0],
                    "faces": {
                        FACE_INDEX["down"]: _map_uv(rects[tex_name], [0.0, 6.0, 4.0, 10.0]),
                        FACE_INDEX["up"]: _map_uv(rects[tex_name], [0.0, 6.0, 4.0, 10.0]),
                        FACE_INDEX["north"]: _map_uv(rects[tex_name], [0.0, 0.0, 4.0, 16.0]),
                        FACE_INDEX["south"]: _map_uv(rects[tex_name], [0.0, 0.0, 4.0, 16.0]),
                        FACE_INDEX["west"]: _map_uv(rects[tex_name], [6.0, 0.0, 10.0, 16.0]),
                        FACE_INDEX["east"]: _map_uv(rects[tex_name], [6.0, 0.0, 10.0, 16.0]),
                    },
                },
                {
                    "from": [12.0, 0.0, 6.0],
                    "to": [16.0, 16.0, 10.0],
                    "faces": {
                        FACE_INDEX["down"]: _map_uv(rects[tex_name], [12.0, 6.0, 16.0, 10.0]),
                        FACE_INDEX["up"]: _map_uv(rects[tex_name], [12.0, 6.0, 16.0, 10.0]),
                        FACE_INDEX["north"]: _map_uv(rects[tex_name], [12.0, 0.0, 16.0, 16.0]),
                        FACE_INDEX["south"]: _map_uv(rects[tex_name], [12.0, 0.0, 16.0, 16.0]),
                        FACE_INDEX["west"]: _map_uv(rects[tex_name], [6.0, 0.0, 10.0, 16.0]),
                        FACE_INDEX["east"]: _map_uv(rects[tex_name], [6.0, 0.0, 10.0, 16.0]),
                    },
                },
                {
                    "from": [4.0, 12.0, 7.0],
                    "to": [12.0, 15.0, 9.0],
                    "faces": {
                        FACE_INDEX["down"]: _map_uv(rects[tex_name], [4.0, 7.0, 12.0, 9.0]),
                        FACE_INDEX["up"]: _map_uv(rects[tex_name], [4.0, 7.0, 12.0, 9.0]),
                        FACE_INDEX["north"]: _map_uv(rects[tex_name], [4.0, 1.0, 12.0, 4.0]),
                        FACE_INDEX["south"]: _map_uv(rects[tex_name], [4.0, 1.0, 12.0, 4.0]),
                        FACE_INDEX["west"]: _map_uv(rects[tex_name], [7.0, 1.0, 9.0, 4.0]),
                        FACE_INDEX["east"]: _map_uv(rects[tex_name], [7.0, 1.0, 9.0, 4.0]),
                    },
                },
                {
                    "from": [4.0, 6.0, 7.0],
                    "to": [12.0, 9.0, 9.0],
                    "faces": {
                        FACE_INDEX["down"]: _map_uv(rects[tex_name], [4.0, 7.0, 12.0, 9.0]),
                        FACE_INDEX["up"]: _map_uv(rects[tex_name], [4.0, 7.0, 12.0, 9.0]),
                        FACE_INDEX["north"]: _map_uv(rects[tex_name], [4.0, 7.0, 12.0, 10.0]),
                        FACE_INDEX["south"]: _map_uv(rects[tex_name], [4.0, 7.0, 12.0, 10.0]),
                        FACE_INDEX["west"]: _map_uv(rects[tex_name], [7.0, 7.0, 9.0, 10.0]),
                        FACE_INDEX["east"]: _map_uv(rects[tex_name], [7.0, 7.0, 9.0, 10.0]),
                    },
                },
            ]

        if is_wall_type:
            tex_name = textures.get("texture") or textures.get("side") or textures.get("top") or entry["id"]
            if tex_name not in rects:
                tex_name = "cobble" if "cobble" in rects else next(iter(rects))

            # Low arm UVs (height 14)
            uv_end = _map_uv(rects[tex_name], [5, 2, 11, 16])
            uv_side = _map_uv(rects[tex_name], [0, 2, 4, 16])
            uv_side_alt = _map_uv(rects[tex_name], [12, 2, 16, 16])
            uv_top_ns = _map_uv(rects[tex_name], [5, 0, 11, 4])
            uv_top_we = _map_uv(rects[tex_name], [0, 5, 4, 11])

            # Tall arm UVs (height 16)
            uv_end_tall = _map_uv(rects[tex_name], [5, 0, 11, 16])
            uv_side_tall = _map_uv(rects[tex_name], [0, 0, 4, 16])
            uv_side_alt_tall = _map_uv(rects[tex_name], [12, 0, 16, 16])

            # Straight wall UVs (length 16)
            uv_straight_side = _map_uv(rects[tex_name], [0, 2, 16, 16])
            uv_straight_side_tall = _map_uv(rects[tex_name], [0, 0, 16, 16])
            uv_straight_top_we = _map_uv(rects[tex_name], [0, 5, 16, 11])
            uv_straight_top_ns = _map_uv(rects[tex_name], [5, 0, 11, 16])

            block.wall_sides = {
                "north": [
                    {
                        "from": [5.0, 0.0, 0.0],
                        "to": [11.0, 14.0, 4.0],
                        "faces": {
                            FACE_INDEX["north"]: uv_end,
                            FACE_INDEX["south"]: uv_end,
                            FACE_INDEX["west"]: uv_side,
                            FACE_INDEX["east"]: uv_side_alt,
                            FACE_INDEX["up"]: uv_top_ns,
                            FACE_INDEX["down"]: uv_top_ns,
                        },
                    }
                ],
                "north_tall": [
                    {
                        "from": [5.0, 0.0, 0.0],
                        "to": [11.0, 16.0, 4.0],
                        "faces": {
                            FACE_INDEX["north"]: uv_end_tall,
                            FACE_INDEX["south"]: uv_end_tall,
                            FACE_INDEX["west"]: uv_side_tall,
                            FACE_INDEX["east"]: uv_side_alt_tall,
                            FACE_INDEX["up"]: uv_top_ns,
                            FACE_INDEX["down"]: uv_top_ns,
                        },
                    }
                ],
                "south": [
                    {
                        "from": [5.0, 0.0, 12.0],
                        "to": [11.0, 14.0, 16.0],
                        "faces": {
                            FACE_INDEX["north"]: uv_end,
                            FACE_INDEX["south"]: uv_end,
                            FACE_INDEX["west"]: uv_side_alt,
                            FACE_INDEX["east"]: uv_side,
                            FACE_INDEX["up"]: uv_top_ns,
                            FACE_INDEX["down"]: uv_top_ns,
                        },
                    }
                ],
                "south_tall": [
                    {
                        "from": [5.0, 0.0, 12.0],
                        "to": [11.0, 16.0, 16.0],
                        "faces": {
                            FACE_INDEX["north"]: uv_end_tall,
                            FACE_INDEX["south"]: uv_end_tall,
                            FACE_INDEX["west"]: uv_side_alt_tall,
                            FACE_INDEX["east"]: uv_side_tall,
                            FACE_INDEX["up"]: uv_top_ns,
                            FACE_INDEX["down"]: uv_top_ns,
                        },
                    }
                ],
                "west": [
                    {
                        "from": [0.0, 0.0, 5.0],
                        "to": [4.0, 14.0, 11.0],
                        "faces": {
                            FACE_INDEX["west"]: uv_end,
                            FACE_INDEX["east"]: uv_end,
                            FACE_INDEX["north"]: uv_side,
                            FACE_INDEX["south"]: uv_side_alt,
                            FACE_INDEX["up"]: uv_top_we,
                            FACE_INDEX["down"]: uv_top_we,
                        },
                    }
                ],
                "west_tall": [
                    {
                        "from": [0.0, 0.0, 5.0],
                        "to": [4.0, 16.0, 11.0],
                        "faces": {
                            FACE_INDEX["west"]: uv_end_tall,
                            FACE_INDEX["east"]: uv_end_tall,
                            FACE_INDEX["north"]: uv_side_tall,
                            FACE_INDEX["south"]: uv_side_alt_tall,
                            FACE_INDEX["up"]: uv_top_we,
                            FACE_INDEX["down"]: uv_top_we,
                        },
                    }
                ],
                "east": [
                    {
                        "from": [12.0, 0.0, 5.0],
                        "to": [16.0, 14.0, 11.0],
                        "faces": {
                            FACE_INDEX["west"]: uv_end,
                            FACE_INDEX["east"]: uv_end,
                            FACE_INDEX["north"]: uv_side_alt,
                            FACE_INDEX["south"]: uv_side,
                            FACE_INDEX["up"]: uv_top_we,
                            FACE_INDEX["down"]: uv_top_we,
                        },
                    }
                ],
                "east_tall": [
                    {
                        "from": [12.0, 0.0, 5.0],
                        "to": [16.0, 16.0, 11.0],
                        "faces": {
                            FACE_INDEX["west"]: uv_end_tall,
                            FACE_INDEX["east"]: uv_end_tall,
                            FACE_INDEX["north"]: uv_side_alt_tall,
                            FACE_INDEX["south"]: uv_side_tall,
                            FACE_INDEX["up"]: uv_top_we,
                            FACE_INDEX["down"]: uv_top_we,
                        },
                    }
                ],
                "straight_we": [
                    {
                        "from": [0.0, 0.0, 5.0],
                        "to": [16.0, 14.0, 11.0],
                        "faces": {
                            FACE_INDEX["west"]: uv_end,
                            FACE_INDEX["east"]: uv_end,
                            FACE_INDEX["north"]: uv_straight_side,
                            FACE_INDEX["south"]: uv_straight_side,
                            FACE_INDEX["up"]: uv_straight_top_we,
                            FACE_INDEX["down"]: uv_straight_top_we,
                        },
                    }
                ],
                "straight_we_tall": [
                    {
                        "from": [0.0, 0.0, 5.0],
                        "to": [16.0, 16.0, 11.0],
                        "faces": {
                            FACE_INDEX["west"]: uv_end_tall,
                            FACE_INDEX["east"]: uv_end_tall,
                            FACE_INDEX["north"]: uv_straight_side_tall,
                            FACE_INDEX["south"]: uv_straight_side_tall,
                            FACE_INDEX["up"]: uv_straight_top_we,
                            FACE_INDEX["down"]: uv_straight_top_we,
                        },
                    }
                ],
                "straight_ns": [
                    {
                        "from": [5.0, 0.0, 0.0],
                        "to": [11.0, 14.0, 16.0],
                        "faces": {
                            FACE_INDEX["north"]: uv_end,
                            FACE_INDEX["south"]: uv_end,
                            FACE_INDEX["west"]: uv_straight_side,
                            FACE_INDEX["east"]: uv_straight_side,
                            FACE_INDEX["up"]: uv_straight_top_ns,
                            FACE_INDEX["down"]: uv_straight_top_ns,
                        },
                    }
                ],
                "straight_ns_tall": [
                    {
                        "from": [5.0, 0.0, 0.0],
                        "to": [11.0, 16.0, 16.0],
                        "faces": {
                            FACE_INDEX["north"]: uv_end_tall,
                            FACE_INDEX["south"]: uv_end_tall,
                            FACE_INDEX["west"]: uv_straight_side_tall,
                            FACE_INDEX["east"]: uv_straight_side_tall,
                            FACE_INDEX["up"]: uv_straight_top_ns,
                            FACE_INDEX["down"]: uv_straight_top_ns,
                        },
                    }
                ],
            }
            block.inventory_elements = [
                {
                    "from": [4.0, 0.0, 4.0],
                    "to": [12.0, 16.0, 12.0],
                    "faces": {
                        FACE_INDEX["down"]: _map_uv(rects[tex_name], [4.0, 4.0, 12.0, 12.0]),
                        FACE_INDEX["up"]: _map_uv(rects[tex_name], [4.0, 4.0, 12.0, 12.0]),
                        FACE_INDEX["north"]: _map_uv(rects[tex_name], [4.0, 0.0, 12.0, 16.0]),
                        FACE_INDEX["south"]: _map_uv(rects[tex_name], [4.0, 0.0, 12.0, 16.0]),
                        FACE_INDEX["west"]: _map_uv(rects[tex_name], [4.0, 0.0, 12.0, 16.0]),
                        FACE_INDEX["east"]: _map_uv(rects[tex_name], [4.0, 0.0, 12.0, 16.0]),
                    },
                },
                {
                    "from": [5.0, 0.0, 0.0],
                    "to": [11.0, 13.0, 16.0],
                    "faces": {
                        FACE_INDEX["down"]: _map_uv(rects[tex_name], [5.0, 0.0, 11.0, 16.0]),
                        FACE_INDEX["up"]: _map_uv(rects[tex_name], [5.0, 0.0, 11.0, 16.0]),
                        FACE_INDEX["north"]: _map_uv(rects[tex_name], [5.0, 3.0, 11.0, 16.0]),
                        FACE_INDEX["south"]: _map_uv(rects[tex_name], [5.0, 3.0, 11.0, 16.0]),
                        FACE_INDEX["west"]: _map_uv(rects[tex_name], [0.0, 3.0, 16.0, 16.0]),
                        FACE_INDEX["east"]: _map_uv(rects[tex_name], [0.0, 3.0, 16.0, 16.0]),
                    },
                },
            ]

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
                face_uv_specs = {}
                for face_name, spec in element["faces"].items():
                    texture_name = _resolve_texture(spec["texture"], textures, entry["id"], face_name=face_name)
                    spec_uv = spec.get("uv", [0, 0, 16, 16])
                    uv = _map_uv(rects[texture_name], spec_uv)
                    index = FACE_INDEX[face_name]
                    faces[index] = uv
                    face_uv_specs[index] = spec_uv
                top_resolved.append({
                    "from": origin,
                    "to": extent,
                    "faces": faces,
                    "face_uv_specs": face_uv_specs,
                    "glass": bool(entry.get("properties", {}).get("glass") or entry.get("id") == "glass_slab"),
                    "cutout": entry.get("render") == "cutout",
                })

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
        a_glass = bool(slab_a.properties.get("glass") or getattr(slab_a, "glass", False) or slab_a.id == "glass_slab")
        b_glass = bool(slab_b.properties.get("glass") or getattr(slab_b, "glass", False) or slab_b.id == "glass_slab")
        bottom_elements = []
        for el in slab_a.elements:
            faces = {k: v for k, v in el["faces"].items() if k != 2 or (not a_glass and b_glass)}
            el_dict = dict(el)
            el_dict["from"] = list(el["from"])
            el_dict["to"] = list(el["to"])
            el_dict["faces"] = faces
            el_dict["glass"] = a_glass
            el_dict["cutout"] = bool(slab_a.cutout)
            if "face_uv_specs" in el:
                el_dict["face_uv_specs"] = {k: v for k, v in el["face_uv_specs"].items() if k in faces}
            bottom_elements.append(el_dict)
        top_elements = []
        for el in slab_b_top.elements:
            faces = {k: v for k, v in el["faces"].items() if k != 3 or (not b_glass and a_glass)}
            el_dict = dict(el)
            el_dict["from"] = list(el["from"])
            el_dict["to"] = list(el["to"])
            el_dict["faces"] = faces
            el_dict["glass"] = b_glass
            el_dict["cutout"] = bool(slab_b.cutout)
            if "face_uv_specs" in el:
                el_dict["face_uv_specs"] = {k: v for k, v in el["face_uv_specs"].items() if k in faces}
            top_elements.append(el_dict)
        double_elements = bottom_elements + top_elements
        double_props = {
            "double_slab": True,
            "slab": True,
            "half": "double",
            "bottom": slab_a.id,
            "top": slab_b.id,
        }
        if a_glass and b_glass:
            double_props["glass"] = True
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
            cutout=slab_a.cutout or slab_b.cutout,
            opacity=1.0 if not (slab_a.cutout and slab_b.cutout) else 0.0,
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
        fb_glass = bool(fb.properties.get("glass") or getattr(fb, "glass", False) or "glass" in fb.id)
        offset_elements = [
            {
                "from": [0.0, -8.0, 0.0],
                "to": [16.0, 8.0, 16.0],
                "faces": offset_faces,
                "tint": fb.tint,
                "glass": fb_glass,
                "cutout": fb.cutout,
                "face_uv_specs": {i: [0, 0, 16, 16] for i in offset_faces},
            }
        ]
        offset_props = dict(fb.properties)
        offset_props["offset_block"] = True
        offset_props["offset_base"] = fb.id
        offset_props["capped"] = False
        offset_props["top_slab"] = None
        if fb_glass:
            offset_props["glass"] = True
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
            slab_b_glass = bool(slab_b.properties.get("glass") or getattr(slab_b, "glass", False) or "glass" in slab_b.id)
            bot_faces = {i: fb.face_uvs[i] for i in range(6) if (i != 2 or (not fb_glass and slab_b_glass)) and fb.face_uvs[i] is not None}
            bot_el = {
                "from": [0.0, -8.0, 0.0],
                "to": [16.0, 8.0, 16.0],
                "faces": bot_faces,
                "tint": fb.tint,
                "glass": fb_glass,
                "cutout": fb.cutout,
                "face_uv_specs": {i: [0, 0, 16, 16] for i in bot_faces},
            }
            top_els = []
            for el in slab_b_top.elements:
                t_faces = {k: v for k, v in el["faces"].items() if k != 3 or (not slab_b_glass and fb_glass)}
                t_specs = {}
                if "face_uv_specs" in el:
                    t_specs = {k: v for k, v in el["face_uv_specs"].items() if k in t_faces}
                else:
                    t_specs = {k: [0, 0, 16, 8] if k in (0, 1, 4, 5) else [0, 0, 16, 16] for k in t_faces}
                top_els.append(
                    {
                        "from": list(el["from"]),
                        "to": list(el["to"]),
                        "faces": t_faces,
                        "tint": slab_b.tint,
                        "glass": slab_b_glass,
                        "cutout": slab_b.cutout,
                        "face_uv_specs": t_specs,
                    }
                )
            capped_elements = [bot_el] + top_els
            capped_props = dict(fb.properties)
            capped_props["offset_block"] = True
            capped_props["offset_base"] = fb.id
            capped_props["capped"] = True
            capped_props["top_slab"] = slab_b.id
            if fb_glass and slab_b_glass:
                capped_props["glass"] = True
            else:
                capped_props.pop("glass", None)
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

    vertical_base_slabs = [
        b
        for b in blocks.values()
        if b.properties.get("vertical_slab")
        and not b.is_variant
        and not b.properties.get("double_vertical_slab")
        and not b.properties.get("back_torch")
    ]

    # Register south-facing aliases for all vertical slabs (e.g., stone_vertical_slab_south → stone_vertical_slab)
    for vs_base in vertical_base_slabs:
        south_alias_id = f"{vs_base.id}_south"
        if south_alias_id not in blocks:
            south_alias_props = dict(vs_base.properties)
            south_alias_props["vertical_slab"] = True
            south_alias_block = Block(
                south_alias_id,
                vs_base.name + " (South)",
                vs_base.solid,
                vs_base.properties.get("can_place_as_wall"),
                list(vs_base.face_uvs),
                [dict(el) for el in vs_base.elements],
                cutout=vs_base.cutout,
                opacity=vs_base.opacity,
                properties=south_alias_props,
                base_id=vs_base.id,
                is_variant=True,
                hardness=vs_base.hardness,
                tool=vs_base.tool,
                mining_level=vs_base.mining_level,
                loot_table=vs_base.loot_table,
                light=vs_base.light,
                emission=vs_base.emission,
                gravity=vs_base.gravity,
                fluid_breakable=vs_base.fluid_breakable,
                soil=vs_base.soil,
                plant=vs_base.plant,
                requires_soil=vs_base.requires_soil,
                valid_ground=vs_base.valid_ground,
                tint=vs_base.tint,
                station=vs_base.station,
                recipe_types=vs_base.recipe_types,
                cook_speed=vs_base.cook_speed,
            )
            blocks[south_alias_id] = south_alias_block
            by_index.append(south_alias_block)
            index_of[south_alias_id] = len(by_index) - 1

    def _vertical_facing_block(base_block, facing):
        if facing == "south":
            return base_block
        return blocks.get(f"{base_block.id}_{facing}", base_block)

    # Register waterlogged vertical slabs (source lvl=8 and flowing lvl=1..7) for all facings
    for vs_base in vertical_base_slabs:
        for v_facing in ("south", "north", "east", "west"):
            v_block = _vertical_facing_block(vs_base, v_facing)
            v_facing_name = v_facing.capitalize()
            for lvl in range(8, 0, -1):
                if v_facing == "south":
                    wl_id = f"{vs_base.id}_waterlogged" if lvl == 8 else f"{vs_base.id}_waterlogged_{lvl}"
                else:
                    wl_id = f"{vs_base.id}_{v_facing}_waterlogged" if lvl == 8 else f"{vs_base.id}_{v_facing}_waterlogged_{lvl}"

                wl_name = (
                    f"{vs_base.name} ({v_facing_name}, Waterlogged)"
                    if lvl == 8
                    else f"{vs_base.name} ({v_facing_name}, Waterlogged {lvl})"
                )
                wl_props = dict(v_block.properties)
                wl_props["vertical_slab"] = True
                wl_props["facing"] = v_facing
                wl_props["waterlogged"] = True
                wl_props["fluid"] = "water"
                wl_props["level"] = lvl

                wl_block = Block(
                    wl_id,
                    wl_name,
                    True,
                    False,
                    list(v_block.face_uvs),
                    [dict(el) for el in v_block.elements],
                    cutout=v_block.cutout,
                    opacity=v_block.opacity,
                    properties=wl_props,
                    base_id=vs_base.id,
                    is_variant=True,
                    hardness=vs_base.hardness,
                    tool=vs_base.tool,
                    mining_level=vs_base.mining_level,
                    loot_table=vs_base.loot_table,
                    emission=vs_base.emission,
                    gravity=vs_base.gravity,
                    fluid_breakable=False,
                    tint=vs_base.tint,
                    waterlogged=True,
                )
                if len(by_index) == 99:
                    by_index.append(None)
                blocks[wl_id] = wl_block
                by_index.append(wl_block)
                index_of[wl_id] = len(by_index) - 1

                if v_facing == "south":
                    alias_id = (
                        f"{vs_base.id}_south_waterlogged"
                        if lvl == 8
                        else f"{vs_base.id}_south_waterlogged_{lvl}"
                    )
                    blocks[alias_id] = wl_block
                    index_of[alias_id] = index_of[wl_id]

    horizontal_base_slabs = [
        b for b in blocks.values()
        if b.properties.get("slab")
        and not b.is_variant
        and not b.properties.get("double_slab")
        and not getattr(b, "vertical_slab", False)
    ]

    for axis, positive_facing, negative_facing, positive_face, negative_face in (
        ("z", "south", "north", FACE_INDEX["north"], FACE_INDEX["south"]),
        ("x", "east", "west", FACE_INDEX["west"], FACE_INDEX["east"]),
    ):
        for positive_base in vertical_base_slabs:
            positive_block = _vertical_facing_block(positive_base, positive_facing)
            for negative_base in vertical_base_slabs:
                negative_block = _vertical_facing_block(negative_base, negative_facing)
                double_id = (
                    f"double_vertical_slab_{axis}_"
                    f"{positive_block.base_id}_{negative_block.base_id}"
                )
                pos_glass = bool(positive_block.properties.get("glass") or getattr(positive_block, "glass", False) or "glass" in positive_block.base_id)
                neg_glass = bool(negative_block.properties.get("glass") or getattr(negative_block, "glass", False) or "glass" in negative_block.base_id)
                double_elements = []
                for source, other_glass, hidden_face in (
                    (positive_block, neg_glass, positive_face),
                    (negative_block, pos_glass, negative_face),
                ):
                    src_glass = bool(source.properties.get("glass") or getattr(source, "glass", False) or "glass" in source.base_id)
                    for element in source.elements:
                        element_copy = dict(element)
                        element_copy["from"] = list(element["from"])
                        element_copy["to"] = list(element["to"])
                        element_copy["faces"] = {
                            face: uv
                            for face, uv in element.get("faces", {}).items()
                            if face != hidden_face or (not src_glass and other_glass)
                        }
                        element_copy["glass"] = src_glass
                        element_copy["cutout"] = bool(source.cutout)
                        if "face_uv_specs" in element:
                            element_copy["face_uv_specs"] = {
                                k: v for k, v in element["face_uv_specs"].items() if k in element_copy["faces"]
                            }
                        double_elements.append(element_copy)
                double_props = {
                    "vertical_slab": True,
                    "double_vertical_slab": True,
                    "axis": axis,
                    "positive": positive_block.base_id,
                    "negative": negative_block.base_id,
                    "positive_block": positive_block.id,
                    "negative_block": negative_block.id,
                    "positive_facing": positive_facing,
                    "negative_facing": negative_facing,
                }
                if pos_glass and neg_glass:
                    double_props["glass"] = True
                same_material = positive_block.base_id == negative_block.base_id
                double_name = (
                    f"{positive_block.name} Double Vertical Slab"
                    if same_material
                    else f"{positive_block.name} / {negative_block.name} Double Vertical Slab"
                )
                double_block = Block(
                    double_id,
                    double_name,
                    True,
                    True,
                    list(positive_block.face_uvs),
                    double_elements,
                    cutout=bool(positive_block.cutout or negative_block.cutout),
                    opacity=1.0 if not (positive_block.cutout and negative_block.cutout) else 0.0,
                    properties=double_props,
                    base_id=positive_block.base_id,
                    is_variant=True,
                    hardness=max(positive_block.hardness, negative_block.hardness),
                    tool=positive_block.tool or negative_block.tool,
                    mining_level=max(positive_block.mining_level, negative_block.mining_level),
                    gravity=False,
                    fluid_breakable=False,
                    tint=positive_block.tint,
                )
                if len(by_index) == 99:
                    by_index.append(None)
                blocks[double_id] = double_block
                by_index.append(double_block)
                index_of[double_id] = len(by_index) - 1
                if same_material:
                    alias_id = f"double_vertical_slab_{axis}_{positive_block.base_id}"
                    blocks[alias_id] = double_block
                    index_of[alias_id] = index_of[double_id]

                # Create offset double vertical slab (occupies y in [-8, 8])
                off_double_id = f"{double_id}_offset"
                off_double_elements = []
                for element in double_elements:
                    el_c = dict(element)
                    el_c["from"] = [float(element["from"][0]), float(element["from"][1]) - 8.0, float(element["from"][2])]
                    el_c["to"] = [float(element["to"][0]), float(element["to"][1]) - 8.0, float(element["to"][2])]
                    el_c["faces"] = dict(element["faces"])
                    if "face_uv_specs" in element:
                        el_c["face_uv_specs"] = dict(element["face_uv_specs"])
                    off_double_elements.append(el_c)

                off_double_props = dict(double_props)
                off_double_props["offset_block"] = True
                off_double_props["offset_base"] = double_id
                off_double_props["capped"] = False
                off_double_props["top_slab"] = None

                off_double_name = f"{double_name} (Offset)"
                off_double_block = Block(
                    off_double_id,
                    off_double_name,
                    True,
                    False,
                    list(positive_block.face_uvs),
                    off_double_elements,
                    cutout=bool(positive_block.cutout or negative_block.cutout),
                    opacity=1.0 if not (positive_block.cutout and negative_block.cutout) else 0.0,
                    properties=off_double_props,
                    base_id=positive_block.base_id,
                    is_variant=True,
                    hardness=max(positive_block.hardness, negative_block.hardness),
                    tool=positive_block.tool or negative_block.tool,
                    mining_level=max(positive_block.mining_level, negative_block.mining_level),
                    gravity=False,
                    fluid_breakable=False,
                    tint=positive_block.tint,
                )
                if len(by_index) == 99:
                    by_index.append(None)
                blocks[off_double_id] = off_double_block
                by_index.append(off_double_block)
                index_of[off_double_id] = len(by_index) - 1

                if same_material:
                    alias_off_id = f"double_vertical_slab_{axis}_{positive_block.base_id}_offset"
                    blocks[alias_off_id] = off_double_block
                    index_of[alias_off_id] = index_of[off_double_id]

                # Capped variants with horizontal slabs
                for h_slab in horizontal_base_slabs:
                    capped_double_id = f"{off_double_id}_{h_slab.id}"
                    capped_double_name = f"{off_double_name} / {h_slab.name}"
                    h_top_id = f"{h_slab.id}_top"
                    h_top_block = blocks.get(h_top_id)
                    if h_top_block is None:
                        continue

                    bot_els = []
                    h_is_glass = bool(h_slab.properties.get("glass") or getattr(h_slab, "glass", False) or "glass" in h_slab.id)
                    for element in off_double_elements:
                        b_el = dict(element)
                        b_glass = b_el.get("glass", False)
                        b_el["faces"] = {k: v for k, v in element["faces"].items() if k != 2 or (not b_glass and h_is_glass)}
                        if "face_uv_specs" in element:
                            b_el["face_uv_specs"] = {k: v for k, v in element["face_uv_specs"].items() if k in b_el["faces"]}
                        bot_els.append(b_el)

                    top_els = []
                    for element in h_top_block.elements:
                        t_faces = {k: v for k, v in element["faces"].items() if k != 3 or (not h_is_glass and any(e.get("glass") for e in off_double_elements))}
                        t_specs = {}
                        if "face_uv_specs" in element:
                            t_specs = {k: v for k, v in element["face_uv_specs"].items() if k in t_faces}
                        else:
                            t_specs = {k: [0, 0, 16, 8] if k in (0, 1, 4, 5) else [0, 0, 16, 16] for k in t_faces}
                        top_els.append({
                            "from": list(element["from"]),
                            "to": list(element["to"]),
                            "faces": t_faces,
                            "tint": h_slab.tint,
                            "glass": h_is_glass,
                            "cutout": bool(h_slab.cutout),
                            "face_uv_specs": t_specs,
                        })

                    capped_double_elements = bot_els + top_els
                    capped_face_uvs = list(positive_block.face_uvs)
                    if len(h_top_block.face_uvs) > 2 and h_top_block.face_uvs[2] is not None:
                        capped_face_uvs[2] = h_top_block.face_uvs[2]

                    capped_props = dict(off_double_props)
                    capped_props["capped"] = True
                    capped_props["top_slab"] = h_slab.id
                    if all(e.get("glass") for e in bot_els) and h_is_glass:
                        capped_props["glass"] = True
                    else:
                        capped_props.pop("glass", None)

                    capped_double_block = Block(
                        capped_double_id,
                        capped_double_name,
                        True,
                        False,
                        capped_face_uvs,
                        capped_double_elements,
                        cutout=False,
                        opacity=1.0,
                        properties=capped_props,
                        base_id=positive_block.base_id,
                        is_variant=True,
                        hardness=max(positive_block.hardness, negative_block.hardness),
                        tool=positive_block.tool or negative_block.tool,
                        mining_level=max(positive_block.mining_level, negative_block.mining_level),
                        gravity=False,
                        fluid_breakable=False,
                        tint=positive_block.tint,
                    )
                    if len(by_index) == 99:
                        by_index.append(None)
                    blocks[capped_double_id] = capped_double_block
                    by_index.append(capped_double_block)
                    index_of[capped_double_id] = len(by_index) - 1

                    if same_material:
                        alias_cap_id = f"double_vertical_slab_{axis}_{positive_block.base_id}_offset_{h_slab.id}"
                        blocks[alias_cap_id] = capped_double_block
                        index_of[alias_cap_id] = index_of[capped_double_id]

    # Create vertical slab offset blocks for each vertical slab base and facing.
    # The offset occupies y in [-8, 8] (world space [y - 0.5, y + 0.5]).
    for vs in vertical_base_slabs:
        vs_name = vs.name
        for v_facing in ("north", "south", "east", "west"):
            v_block = _vertical_facing_block(vs, v_facing) if v_facing != "south" else vs
            v_facing_name = v_facing.capitalize()
            if v_facing == "south":
                off_id = f"{vs.id}_offset"
            else:
                off_id = f"{vs.id}_{v_facing}_offset"
            off_name = f"{vs_name} ({v_facing_name}, offset)"

            # Shift the vertical slab elements by -8 in Y
            vs_glass = bool(vs.properties.get("glass") or getattr(vs, "glass", False) or "glass" in vs.id)
            off_elements = []
            for element in v_block.elements:
                el_copy = dict(element)
                el_copy["from"] = [element["from"][0], element["from"][1] - 8.0, element["from"][2]]
                el_copy["to"] = [element["to"][0], element["to"][1] - 8.0, element["to"][2]]
                el_copy["faces"] = dict(element.get("faces", {}))
                el_copy["tint"] = vs.tint
                el_copy["glass"] = vs_glass
                el_copy["cutout"] = vs.cutout
                if "face_uv_specs" in element:
                    el_copy["face_uv_specs"] = dict(element["face_uv_specs"])
                off_elements.append(el_copy)

            off_props = dict(vs.properties)
            off_props["offset_block"] = True
            off_props["offset_base"] = vs.id
            off_props["capped"] = False
            off_props["top_slab"] = None
            off_props["vertical_slab"] = True
            off_props["is_vertical_slab_offset"] = True
            off_props["facing"] = v_facing
            if vs_glass:
                off_props["glass"] = True

            off_block = Block(
                off_id,
                off_name,
                True,
                False,
                list(v_block.face_uvs),
                off_elements,
                cutout=vs.cutout,
                opacity=1.0,
                properties=off_props,
                base_id=vs.id,
                is_variant=True,
                hardness=vs.hardness,
                tool=vs.tool,
                mining_level=vs.mining_level,
                loot_table=vs.loot_table,
                emission=vs.emission,
                gravity=vs.gravity,
                fluid_breakable=vs.fluid_breakable,
                soil=vs.soil,
                plant=vs.plant,
                requires_soil=vs.requires_soil,
                valid_ground=vs.valid_ground,
                tint=vs.tint,
                station=vs.station,
                recipe_types=vs.recipe_types,
                cook_speed=vs.cook_speed,
            )
            if len(by_index) == 99:
                by_index.append(None)
            blocks[off_id] = off_block
            by_index.append(off_block)
            index_of[off_id] = len(by_index) - 1

            if v_facing == "south":
                # Alias stone_vertical_slab_south_offset -> stone_vertical_slab_offset
                alias_id = f"{vs.id}_south_offset"
                blocks[alias_id] = off_block
                index_of[alias_id] = index_of[off_id]

            # Create capped variants for each vertical slab offset with each horizontal slab
            for h_slab in horizontal_base_slabs:
                if v_facing == "south":
                    capped_id = f"{vs.id}_offset_{h_slab.id}"
                else:
                    capped_id = f"{vs.id}_{v_facing}_offset_{h_slab.id}"
                capped_name = f"{off_name} / {h_slab.name}"
                h_top_id = f"{h_slab.id}_top"
                h_top_block = blocks.get(h_top_id)
                if h_top_block is None:
                    continue

                # Bottom part: offset vertical slab elements
                vs_is_glass = bool(vs.properties.get("glass") or getattr(vs, "glass", False) or "glass" in vs.id)
                h_is_glass = bool(h_slab.properties.get("glass") or getattr(h_slab, "glass", False) or "glass" in h_slab.id)
                bot_elements = []
                for el in off_elements:
                    bot_el = dict(el)
                    bot_el["faces"] = {k: v for k, v in el["faces"].items() if k != 2 or (not vs_is_glass and h_is_glass)}
                    bot_el["glass"] = vs_is_glass
                    bot_el["cutout"] = bool(vs.cutout)
                    if "face_uv_specs" in el:
                        bot_el["face_uv_specs"] = {k: v for k, v in el["face_uv_specs"].items() if k in bot_el["faces"]}
                    bot_elements.append(bot_el)

                # Top part: horizontal top slab elements at [0, 8, 0] to [16, 16, 16]
                top_els = []
                for el in h_top_block.elements:
                    t_faces = {k: v for k, v in el["faces"].items() if k != 3 or (not h_is_glass and vs_is_glass)}
                    t_specs = {}
                    if "face_uv_specs" in el:
                        t_specs = {k: v for k, v in el["face_uv_specs"].items() if k in t_faces}
                    else:
                        t_specs = {k: [0, 0, 16, 8] if k in (0, 1, 4, 5) else [0, 0, 16, 16] for k in t_faces}
                    top_els.append({
                        "from": list(el["from"]),
                        "to": list(el["to"]),
                        "faces": t_faces,
                        "tint": h_slab.tint,
                        "glass": h_is_glass,
                        "cutout": bool(h_slab.cutout),
                        "face_uv_specs": t_specs,
                    })

                capped_elements = bot_elements + top_els
                capped_face_uvs = list(v_block.face_uvs)
                if len(h_top_block.face_uvs) > 2 and h_top_block.face_uvs[2] is not None:
                    capped_face_uvs[2] = h_top_block.face_uvs[2]

                capped_props = dict(off_props)
                capped_props["capped"] = True
                capped_props["top_slab"] = h_slab.id
                if vs_is_glass and h_is_glass:
                    capped_props["glass"] = True
                else:
                    capped_props.pop("glass", None)

                capped_block = Block(
                    capped_id,
                    capped_name,
                    True,
                    False,
                    capped_face_uvs,
                    capped_elements,
                    cutout=vs.cutout or h_slab.cutout,
                    opacity=1.0,
                    properties=capped_props,
                    base_id=vs.id,
                    is_variant=True,
                    hardness=max(vs.hardness, h_slab.hardness),
                    tool=vs.tool or h_slab.tool,
                    mining_level=max(vs.mining_level, h_slab.mining_level),
                    loot_table=vs.loot_table,
                    emission=vs.emission,
                    gravity=vs.gravity,
                    fluid_breakable=vs.fluid_breakable,
                    soil=h_slab.soil,
                    tint=vs.tint,
                    station=vs.station,
                    recipe_types=vs.recipe_types,
                    cook_speed=vs.cook_speed,
                )
                if len(by_index) == 99:
                    by_index.append(None)
                blocks[capped_id] = capped_block
                by_index.append(capped_block)
                index_of[capped_id] = len(by_index) - 1

                if v_facing == "south":
                    alias_capped_id = f"{vs.id}_south_offset_{h_slab.id}"
                    blocks[alias_capped_id] = capped_block
                    index_of[alias_capped_id] = index_of[capped_id]

    # A torch on the back (inner) long face of a vertical slab has to live in
    # the empty half of the same cell, so it needs a combined block.
    for base_slab in vertical_base_slabs:
        for facing in FACING_DIRECTIONS:
            slab_block = _vertical_facing_block(base_slab, facing)
            mount = OPPOSITE_FACING[facing]
            shift = 8.0 if mount in ("south", "east") else -8.0
            for tb in base_torch_blocks:
                wall_block = blocks.get(f"{tb.id}_wall_{mount}")
                if wall_block is None or not wall_block.elements:
                    continue
                combined_id = f"{slab_block.id}_back_{tb.id}"
                torch_element = dict(wall_block.elements[0])
                torch_element["from"] = list(torch_element["from"])
                torch_element["to"] = list(torch_element["to"])
                if mount in ("north", "south"):
                    torch_element["from"][2] += shift
                    torch_element["to"][2] += shift
                else:
                    torch_element["from"][0] += shift
                    torch_element["to"][0] += shift
                torch_element["cutout"] = True
                torch_element["part"] = "torch"
                combined_elements = []
                for element in slab_block.elements:
                    element_copy = dict(element)
                    element_copy["from"] = list(element["from"])
                    element_copy["to"] = list(element["to"])
                    element_copy["faces"] = dict(element.get("faces", {}))
                    combined_elements.append(element_copy)
                combined_elements.append(torch_element)
                combined_props = dict(slab_block.properties)
                combined_props["back_torch"] = tb.id
                combined_props["slab_block"] = slab_block.id
                combined_props["torch_block"] = tb.id
                combined_props["torch_mount"] = mount
                combined_block = Block(
                    combined_id,
                    f"{slab_block.name} with {tb.name}",
                    True,
                    False,
                    list(slab_block.face_uvs),
                    combined_elements,
                    cutout=False,
                    opacity=1.0,
                    properties=combined_props,
                    base_id=slab_block.base_id,
                    is_variant=True,
                    hardness=slab_block.hardness,
                    tool=slab_block.tool,
                    mining_level=slab_block.mining_level,
                    light=tb.light,
                    emission=tb.emission,
                    gravity=False,
                    fluid_breakable=False,
                    tint=slab_block.tint,
                )
                if len(by_index) == 99:
                    by_index.append(None)
                blocks[combined_id] = combined_block
                by_index.append(combined_block)
                index_of[combined_id] = len(by_index) - 1

    # Register sub_cell sentinel block for cells containing sub-voxels/octants
    sub_cell_block = Block(
        "sub_cell",
        "Sub Cell",
        solid=True,
        full_cube=False,
        face_uvs=list(blocks["stone"].face_uvs) if "stone" in blocks else [None, None, None, None, None, None],
        elements=[],
        properties={"sub_cell": True},
        base_id="sub_cell",
        is_variant=True,
    )
    if len(by_index) == 99:
        by_index.append(None)
    blocks["sub_cell"] = sub_cell_block
    by_index.append(sub_cell_block)
    index_of["sub_cell"] = len(by_index) - 1

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

    # Allow toggling or updating door states (open/closed, mirroring hinge)
    if is_door(cur_blk) and is_door(b_obj):
        cur_d = get_door_info(cur_blk)
        new_d = get_door_info(b_obj)
        if cur_d and new_d and cur_d["base_id"] == new_d["base_id"] and cur_d["half"] == new_d["half"]:
            return True

    # Allow toggling trapdoors (open/closed)
    if is_trapdoor(cur_blk) and is_trapdoor(b_obj):
        cur_t = get_trapdoor_info(cur_blk)
        new_t = get_trapdoor_info(b_obj)
        if cur_t and new_t and cur_t["base_id"] == new_t["base_id"] and cur_t["half"] == new_t["half"]:
            return True

    # Allow toggling fence gates (open/closed)
    if is_fence_gate(cur_blk) and is_fence_gate(b_obj):
        cur_g = get_fence_gate_info(cur_blk)
        new_g = get_fence_gate_info(b_obj)
        if cur_g and new_g and cur_g["base_id"] == new_g["base_id"]:
            return True

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
        below_v_info = get_vertical_slab_offset_info(below_blk)
        if not (
            below_blk in ("", AIR, None)
            or (below_obj is not None and bool(below_obj.properties.get("fluid")))
            or has_half_offset_top(below_blk)
            or (below_v_info is not None and not below_v_info["capped"])
            or (below_t_info is not None and not below_t_info["capped"])
        ):
            return False

    # Current position must not be solid (unless combining slabs into a double slab, capping an offset block/torch, or partial breaking)
    if world.is_solid(x, y, z):
        cur_off = get_offset_info(cur_blk)
        new_off = get_offset_info(b_obj)
        cur_double_v = get_double_vertical_slab_info(cur_blk)
        new_double_v = get_double_vertical_slab_info(b_obj)
        cur_back_torch = get_vertical_slab_torch_info(cur_blk)
        new_back_torch = get_vertical_slab_torch_info(b_obj)
        cur_single_v = get_vertical_slab_info(cur_blk)
        new_single_v = get_vertical_slab_info(b_obj)
        cur_v_off = get_vertical_slab_offset_info(cur_blk)
        new_v_off = get_vertical_slab_offset_info(b_obj)
        cur_v_any = cur_single_v or cur_v_off
        new_v_any = new_single_v or new_v_off

        # Combine a single vertical slab with a complementary one in the
        # other half of the same cell.
        if new_double_v is not None and cur_v_any is not None and cur_back_torch is None:
            cur_base = cur_v_any["base_id"]
            if cur_base in (new_double_v["positive"], new_double_v["negative"]):
                return True
        # Attach a torch to the back face of a vertical slab in the same cell.
        if new_back_torch is not None and cur_blk == new_back_torch["slab_block"]:
            return True
        # Break one half of a double vertical slab, leaving the other half.
        if cur_double_v is not None and new_v_any is not None:
            new_base = new_v_any["base_id"]
            if new_base in (
                cur_double_v["positive"],
                cur_double_v["negative"],
            ):
                return True
        # Remove the torch from a combined back-torch block.
        if cur_back_torch is not None and b_obj.id == cur_back_torch["slab_block"]:
            return True
        
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
        if cur_v_off is not None:
            new_s = get_slab_info(b_obj)
            if cur_v_off["capped"] and new_s is not None and not new_s["double"] and new_s["half"] == "top":
                return True
        if new_t_info is not None and new_t_info["capped"] and is_slab(cur_blk):
            cur_s_info = get_slab_info(cur_blk)
            if cur_s_info is not None and not cur_s_info["double"] and cur_s_info["half"] == "top":
                return can_support_torch(world.get_block(x, y - 1, z), "up_offset")
        # Capping an existing top slab with an offset block or offset vertical slab
        if is_slab(cur_blk):
            cur_s_info = get_slab_info(cur_blk)
            if cur_s_info is not None and not cur_s_info["double"] and cur_s_info["half"] == "top":
                if (new_off is not None and new_off["capped"]) or (new_v_off is not None and new_v_off["capped"]):
                    if y - 1 >= 0:
                        below_blk = world.get_block(x, y - 1, z)
                        below_obj = BLOCKS.get(below_blk)
                        below_t_info = get_offset_torch_info(below_blk)
                        below_v_info = get_vertical_slab_offset_info(below_blk)
                        if (
                            below_blk in ("", AIR, None)
                            or (below_obj is not None and bool(below_obj.properties.get("fluid")))
                            or has_half_offset_top(below_blk)
                            or (below_v_info is not None and not below_v_info["capped"])
                            or (below_t_info is not None and not below_t_info["capped"])
                        ):
                            return True
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
