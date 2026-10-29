import json
import random
from pathlib import Path

from server.blocks import ASSETS, BLOCKS, canonical_id
from server.items import canonical_item

_LOOT_CACHE = {}


def load_loot_table(table_id):
    if not table_id:
        return None
    if table_id in _LOOT_CACHE:
        return _LOOT_CACHE[table_id]

    # Try json
    json_path = ASSETS / "loot" / f"{table_id}.json"
    if json_path.is_file():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            _LOOT_CACHE[table_id] = data
            return data
        except (OSError, json.JSONDecodeError):
            pass

    # Try lua
    lua_path = ASSETS / "loot" / f"{table_id}.lua"
    if lua_path.is_file():
        try:
            from server.lua import load_table

            data = load_table(lua_path)
            _LOOT_CACHE[table_id] = data
            return data
        except Exception:
            pass

    return None


def get_tool_type(item):
    if item is None:
        return None
    tool = getattr(item, "tool", None)
    if tool:
        return str(tool).lower()
    item_id = str(getattr(item, "id", "")).lower()
    if "pickaxe" in item_id:
        return "pickaxe"
    if "axe" in item_id:
        return "axe"
    if "shovel" in item_id:
        return "shovel"
    return None


def can_harvest(block, item):
    """Check if block yields loot drops when broken by item."""
    if block is None:
        return False
    min_level = getattr(block, "mining_level", 0)
    if min_level <= 0:
        return True
    if item is None:
        return False
    required_tool = getattr(block, "tool", None)
    tool_type = get_tool_type(item)
    if required_tool and tool_type != required_tool:
        return False
    item_level = getattr(item, "mining_level", 0) or 0
    return item_level >= min_level


def calculate_break_time(block, item):
    """Calculate the breaking time in seconds for survival mode."""
    if block is None:
        return 0.0
    hardness = getattr(block, "hardness", 1.0)
    if hardness <= 0:
        return 0.0

    harvest = can_harvest(block, item)
    speed = 1.0
    if item is not None:
        tool_type = get_tool_type(item)
        required_tool = getattr(block, "tool", None)
        item_speed = getattr(item, "mining_speed", 1.0) or 1.0
        # If tool matches the block or block has no specific tool requirement
        if (required_tool and tool_type == required_tool) or (not required_tool and tool_type):
            speed = max(1.0, float(item_speed))

    multiplier = 1.5 if harvest else 5.0
    return (hardness * multiplier) / speed


def evaluate_condition(condition, context):
    cond_type = condition.get("condition")
    if cond_type == "match_tool":
        item = context.get("held_item")
        req_tool = condition.get("tool")
        req_level = int(condition.get("min_level", 0))
        if item is None:
            return False
        if req_tool:
            tool_type = get_tool_type(item)
            if tool_type != req_tool:
                return False
        item_level = int(getattr(item, "mining_level", 0) or 0)
        return item_level >= req_level
    return True


def roll_loot_table(table, context=None):
    """Roll a loot table and return list of (item_id, count) drops."""
    if not table:
        return []
    context = context or {}
    rng = context.get("random", random)
    drops = []

    pools = table.get("pools", [])
    for pool in pools:
        rolls = int(pool.get("rolls", 1))
        for _ in range(rolls):
            entries = pool.get("entries", [])
            has_weight = any("weight" in e for e in entries)
            if has_weight:
                valid_entries = []
                weights = []
                for entry in entries:
                    conditions = entry.get("conditions", [])
                    passed = all(evaluate_condition(cond, context) for cond in conditions)
                    if passed:
                        valid_entries.append(entry)
                        weights.append(float(entry.get("weight", 1.0)))
                if valid_entries:
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    acc = 0.0
                    chosen_entry = valid_entries[-1]
                    for ent, w in zip(valid_entries, weights):
                        acc += w
                        if r <= acc:
                            chosen_entry = ent
                            break
                    item_name = chosen_entry.get("name") or chosen_entry.get("item")
                    if item_name:
                        count_val = chosen_entry.get("count", 1)
                        if isinstance(count_val, (list, tuple)) and len(count_val) >= 2:
                            count = rng.randint(int(count_val[0]), int(count_val[1]))
                        else:
                            count = int(count_val)
                        if count > 0:
                            drops.append((canonical_item(item_name), count))
                continue

            for entry in entries:
                # Check chance
                chance = float(entry.get("chance", 1.0))
                if chance < 1.0 and rng.random() > chance:
                    continue

                # Check conditions
                conditions = entry.get("conditions", [])
                passed = True
                for cond in conditions:
                    if not evaluate_condition(cond, context):
                        passed = False
                        break
                if not passed:
                    continue

                item_name = entry.get("name") or entry.get("item")
                if not item_name:
                    continue

                count_val = entry.get("count", 1)
                if isinstance(count_val, (list, tuple)) and len(count_val) >= 2:
                    count = rng.randint(int(count_val[0]), int(count_val[1]))
                else:
                    count = int(count_val)

                if count > 0:
                    drops.append((canonical_item(item_name), count))

    return drops


def get_block_drops(block_or_id, held_item=None, rng=None):
    """Determine the item drops when a block is broken."""
    if isinstance(block_or_id, str):
        block = BLOCKS.get(canonical_id(block_or_id))
    else:
        block = block_or_id

    if block is None:
        return []

    context = {
        "block": block,
        "held_item": held_item,
        "random": rng or random,
    }

    # If block has a loot_table specified
    loot_table_id = getattr(block, "loot_table", None)
    if loot_table_id:
        table = load_loot_table(loot_table_id)
        if table:
            return roll_loot_table(table, context)

    # Fallback default: if block can be harvested, drop 1 of base item
    if can_harvest(block, held_item):
        base_id = getattr(block, "base_id", block.id)
        item_id = canonical_item(base_id)
        if item_id:
            return [(item_id, 1)]

    return []


def get_mob_drops(mob, killer=None, held_item=None, rng=None):
    """Determine the item drops when a mob is killed."""
    if mob is None:
        return []

    context = {
        "mob": mob,
        "killer": killer,
        "held_item": held_item,
        "random": rng or random,
    }

    loot_table_id = getattr(mob, "loot_table", None)
    if loot_table_id:
        table = load_loot_table(loot_table_id)
        if table:
            return roll_loot_table(table, context)

    return []


