import json
import logging
from pathlib import Path
import random

from server.blocks import ASSETS

logger = logging.getLogger(__name__)

# Fallback sound set mappings: category -> {"walk": tuple(...), "breaking": tuple(...), "break": tuple(...)}
# Filenames match audio files in assets/sounds/ without the extension.
DEFAULT_SOUND_SETS = {
    "grass": {
        "walk": ("grass_walking_1", "grass_walking_2"),
        "breaking": ("soil_breaking",),
    },
    "dirt": {
        "walk": ("dirt_walking_1",),
        "breaking": ("soil_breaking",),
    },
    "gravel": {
        "walk": ("dirt_walking_1",),
        "breaking": ("soil_breaking",),
    },
    "soil": {
        "walk": ("dirt_walking_1",),
        "breaking": ("soil_breaking",),
    },
    "wood": {
        "walk": ("walking_wood_1", "wood_walking_2"),
        "breaking": ("wood_breaking",),
    },
}

SOUND_SETS = {}


def _normalize_sound_entry(raw_dict):
    """Normalize a sound set dictionary into tuple values with canonical keys."""
    if not isinstance(raw_dict, dict):
        return {}
    entry = {}
    for key, val in raw_dict.items():
        k = str(key).lower()
        if k in ("walk", "step"):
            norm_key = "walk"
        elif k in ("breaking", "hit", "punch", "dig"):
            norm_key = "breaking"
        elif k in ("break", "destroyed"):
            norm_key = "break"
        else:
            norm_key = k

        if isinstance(val, (list, tuple)):
            entry[norm_key] = tuple(str(x) for x in val if x)
        elif isinstance(val, str) and val:
            entry[norm_key] = (val,)
    return entry


def load_sound_sets():
    """Load sound sets from assets/sound_sets/*.json, assets/sounds/*.json, or assets/sounds.json."""
    global SOUND_SETS
    sound_sets = {}

    # 1. Start with copy of defaults
    for k, v in DEFAULT_SOUND_SETS.items():
        sound_sets[k] = dict(v)

    # 2. Check assets/sound_sets and assets/sounds for json definitions
    search_dirs = [
        ASSETS / "sound_sets",
        ASSETS / "sounds",
    ]
    for folder in search_dirs:
        if not folder.is_dir():
            continue
        for json_file in folder.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if "id" in data:
                        set_id = str(data["id"]).lower()
                        normalized = _normalize_sound_entry(data)
                        if set_id in sound_sets:
                            sound_sets[set_id].update(normalized)
                        else:
                            sound_sets[set_id] = normalized
                    else:
                        for set_id, val in data.items():
                            if isinstance(val, dict):
                                s_id = str(set_id).lower()
                                normalized = _normalize_sound_entry(val)
                                if s_id in sound_sets:
                                    sound_sets[s_id].update(normalized)
                                else:
                                    sound_sets[s_id] = normalized
            except Exception as e:
                logger.warning("Failed to load sound set from %s: %s", json_file, e)

    # 3. Check assets/sounds.json
    sounds_json = ASSETS / "sounds.json"
    if sounds_json.is_file():
        try:
            with open(sounds_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for set_id, val in data.items():
                    if isinstance(val, dict):
                        s_id = str(set_id).lower()
                        normalized = _normalize_sound_entry(val)
                        if s_id in sound_sets:
                            sound_sets[s_id].update(normalized)
                        else:
                            sound_sets[s_id] = normalized
        except Exception as e:
            logger.warning("Failed to load sounds.json: %s", e)

    SOUND_SETS = sound_sets
    return SOUND_SETS


# Initialize sound sets on module load
load_sound_sets()


# Block ID / property heuristics for blocks without explicit sound property
def infer_sound_category(block_obj, block_id: str):
    b_id = str(block_id).lower()
    base_id = getattr(block_obj, "base_id", b_id).lower()

    if getattr(block_obj, "plant", False) or any(k in b_id or k in base_id for k in ("grass", "leaf", "leaves", "sapling", "fern", "rose", "dandelion")):
        return "grass"
    if getattr(block_obj, "soil", False) or any(k in b_id or k in base_id for k in ("dirt", "mud", "gravel", "sand", "clay")):
        return "dirt"
    if getattr(block_obj, "tool", None) == "axe" or any(k in b_id or k in base_id for k in ("wood", "plank", "log", "crafting_table", "carpenters_table")):
        return "wood"
    return None


def get_block_sound_group(block_or_id):
    """Retrieve the sound mapping dict for a block or block ID: {'walk': (...), 'breaking': (...)}."""
    from server.blocks import BLOCKS

    if hasattr(block_or_id, "id"):
        block_obj = block_or_id
        block_id = getattr(block_obj, "id", "")
    else:
        block_id = str(block_or_id or "")
        block_obj = BLOCKS.get(block_id)

    if not block_id or block_id == "air":
        return None

    # 1. Explicit sounds dict on block: {"walk": [...], "breaking": [...]}
    sounds_dict = getattr(block_obj, "sounds", None)
    if isinstance(sounds_dict, dict):
        return _normalize_sound_entry(sounds_dict)
    props = getattr(block_obj, "properties", None) or {}
    if isinstance(props.get("sounds"), dict):
        return _normalize_sound_entry(props["sounds"])

    # 2. Explicit sound category name on block
    category = getattr(block_obj, "sound", None) or props.get("sound")
    if category is not None:
        cat_str = str(category).lower()
        if cat_str in SOUND_SETS:
            return SOUND_SETS[cat_str]
        if cat_str in DEFAULT_SOUND_SETS:
            return DEFAULT_SOUND_SETS[cat_str]

    # 3. Inferred sound category from block traits
    inferred = infer_sound_category(block_obj, block_id)
    if inferred:
        if inferred in SOUND_SETS:
            return SOUND_SETS[inferred]
        if inferred in DEFAULT_SOUND_SETS:
            return DEFAULT_SOUND_SETS[inferred]

    return None


def get_block_walk_sound(block_or_id, rng=None):
    """Return a single walk sound name (without extension) for the given block, or None."""
    group = get_block_sound_group(block_or_id)
    if not group:
        return None
    walk_sounds = group.get("walk")
    if not walk_sounds:
        return None
    if isinstance(walk_sounds, str):
        return walk_sounds
    r = rng or random
    return r.choice(walk_sounds)


def get_block_breaking_sound(block_or_id, rng=None):
    """Return a single sound name played WHILE breaking the block, or None."""
    group = get_block_sound_group(block_or_id)
    if not group:
        return None
    breaking_sounds = group.get("breaking")
    if not breaking_sounds:
        return None
    if isinstance(breaking_sounds, str):
        return breaking_sounds
    r = rng or random
    return r.choice(breaking_sounds)


def get_block_break_sound(block_or_id, rng=None):
    """Return a single break sound name (without extension) for when a block is destroyed, or None."""
    group = get_block_sound_group(block_or_id)
    if not group:
        return None
    break_sounds = group.get("break")
    if not break_sounds:
        return None
    if isinstance(break_sounds, str):
        return break_sounds
    r = rng or random
    return r.choice(break_sounds)
