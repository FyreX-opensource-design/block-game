import json
import math
from pathlib import Path

from server.blocks import ASSETS, BLOCKS

_ID_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_")


CRAFTING_SHAPED = "crafting_shaped"
CRAFTING_SHAPELESS = "crafting_shapeless"
SMELTING = "smelting"
BLOCK_HIT = "block_hit"
FLUID_TRANSFORM = "fluid_transform"
FLUID_MEET = "fluid_meet"
CRAFTING_TYPES = (CRAFTING_SHAPED, CRAFTING_SHAPELESS)


class Recipe:
    def __init__(
        self,
        recipe_id,
        shaped,
        pattern,
        ingredients,
        result,
        count,
        recipe_type=None,
        cook_time=10.0,
        experience=0.1,
        item=None,
        blocks=(),
        chance=1.0,
        loot_table=None,
        consume_item=True,
        fluid=None,
        block=None,
        fluids=(),
        source_result=None,
        flowing_fluid=None,
        source_fluid=None,
    ):
        self.id = recipe_id
        self.shaped = shaped
        self.pattern = pattern
        self.ingredients = ingredients
        self.result = result
        self.count = count
        self.type = recipe_type or (CRAFTING_SHAPED if shaped else CRAFTING_SHAPELESS)
        self.cook_time = float(cook_time)
        self.experience = float(experience)
        self.item = item
        self.blocks = tuple(blocks)
        self.chance = float(chance)
        self.loot_table = loot_table
        self.consume_item = bool(consume_item)
        self.fluid = fluid
        self.block = block
        self.fluids = tuple(fluids)
        self.source_result = source_result if source_result is not None else result
        self.flowing_fluid = flowing_fluid
        self.source_fluid = source_fluid


def _trim(rows):
    keep_rows = [index for index, row in enumerate(rows) if any(row)]
    width = max(len(row) for row in rows)
    keep_cols = [index for index in range(width) if any(row[index] for row in rows if index < len(row))]
    if not keep_rows or not keep_cols:
        raise ValueError("recipe pattern is empty")
    return tuple(tuple(rows[row][col] for col in keep_cols) for row in keep_rows)


def _window(ids):
    total = len(ids)
    if total == 4:
        grid_size = 2
    elif total == 9:
        grid_size = 3
    else:
        grid_size = int(math.isqrt(total))
        if grid_size * grid_size != total:
            grid_size = 2

    cells = tuple(ids[r * grid_size : (r + 1) * grid_size] for r in range(grid_size))
    rows = [index for index in range(grid_size) if any(cells[index])]
    cols = [index for index in range(grid_size) if any(cells[row][index] for row in range(grid_size))]
    if not rows or not cols:
        return None
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    pattern = tuple(tuple(cells[r][c] for c in range(min_c, max_c + 1)) for r in range(min_r, max_r + 1))
    forward = [r * grid_size + c for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1) if ids[r * grid_size + c]]
    mirrored = [r * grid_size + c for r in range(min_r, max_r + 1) for c in reversed(range(min_c, max_c + 1)) if ids[r * grid_size + c]]
    return pattern, forward, mirrored


def _mirror(pattern):
    return tuple(tuple(reversed(row)) for row in pattern)


def _pattern_matches(grid_pattern, recipe_pattern):
    if len(grid_pattern) != len(recipe_pattern):
        return False
    from server.tags import matches_ingredient

    for r_row, g_row in zip(recipe_pattern, grid_pattern):
        if len(r_row) != len(g_row):
            return False
        for r_cell, g_cell in zip(r_row, g_row):
            if not matches_ingredient(r_cell, g_cell):
                return False
    return True


def _shapeless_matches(actual_items, recipe_ingredients):
    if len(actual_items) != len(recipe_ingredients):
        return False
    if not any(ing.startswith("#") for ing in recipe_ingredients):
        return sorted(actual_items) == sorted(recipe_ingredients)
    from server.tags import matches_ingredient

    remaining_actual = list(actual_items)
    for expected in recipe_ingredients:
        matched = False
        for i, actual in enumerate(remaining_actual):
            if matches_ingredient(expected, actual):
                remaining_actual.pop(i)
                matched = True
                break
        if not matched:
            return False
    return len(remaining_actual) == 0


def match_grid(ids, recipes, recipe_types=None):
    norm_ids = tuple(BLOCKS[i].base_id if i in BLOCKS and getattr(BLOCKS[i], "base_id", None) else i for i in ids)
    window = _window(norm_ids)
    filled = [index for index, block_id in enumerate(norm_ids) if block_id]
    allowed_types = set(recipe_types) if recipe_types else set(CRAFTING_TYPES)
    for recipe in recipes:
        if recipe.type not in allowed_types:
            continue
        if recipe.shaped:
            if window is None:
                continue
            pattern, forward, mirrored = window
            if _pattern_matches(pattern, recipe.pattern):
                return recipe.result, recipe.count, forward
            if _pattern_matches(pattern, _mirror(recipe.pattern)):
                return recipe.result, recipe.count, mirrored
            continue
        else:
            actual_items = [norm_ids[index] for index in filled]
            if _shapeless_matches(actual_items, recipe.ingredients):
                return recipe.result, recipe.count, filled
    return None


def craft_match(ids, recipe_types=None):
    return match_grid(ids, RECIPES, recipe_types=recipe_types)


def _block_name(token, recipe_id):
    from server.items import ITEMS

    block_id = str(token)
    if block_id.startswith("#"):
        return block_id
    if block_id not in BLOCKS and block_id not in ITEMS:
        raise ValueError(f"{recipe_id} uses unknown block {block_id}")
    return block_id


def recipe_from_mapping(entry, source):
    if not isinstance(entry, dict):
        raise ValueError(f"{source}: recipe file must be an object")
    recipe_id = str(entry.get("id", Path(source).stem if source else ""))
    if any(char not in _ID_CHARS for char in recipe_id) or not recipe_id:
        raise ValueError(f"{source}: recipe id must be a lowercase name")
    recipe_type = str(entry.get("type", ""))
    if recipe_type and any(char not in _ID_CHARS for char in recipe_type):
        raise ValueError(f"{recipe_id} type must be a lowercase name")
    if not recipe_type:
        if "pattern" in entry:
            recipe_type = CRAFTING_SHAPED
        elif "ingredients" in entry:
            recipe_type = CRAFTING_SHAPELESS
        else:
            raise ValueError(f"{recipe_id} needs a type")

    if recipe_type in (BLOCK_HIT, "hit_block", "item_hit_block"):
        item = str(entry.get("item", entry.get("ingredient", "")))
        blocks_raw = entry.get("blocks", entry.get("block", []))
        if isinstance(blocks_raw, str):
            blocks = (blocks_raw,)
        else:
            blocks = tuple(str(b) for b in blocks_raw)
        chance = float(entry.get("chance", 1.0))
        loot_table = entry.get("loot_table")
        consume_item = bool(entry.get("consume_item", entry.get("consume", True)))
        res = entry.get("result", "")
        res_id = res if isinstance(res, str) else (res.get("item", "") if isinstance(res, dict) else "")
        return Recipe(
            recipe_id,
            False,
            (),
            (),
            res_id,
            1,
            BLOCK_HIT,
            item=item,
            blocks=blocks,
            chance=chance,
            loot_table=loot_table,
            consume_item=consume_item,
        )

    if recipe_type in (FLUID_TRANSFORM, "fluid_collision"):
        fluid = str(entry.get("fluid", "water"))
        block = str(entry.get("block", entry.get("ingredient", "")))
        res = entry.get("result", "")
        res_id = res if isinstance(res, str) else (res.get("block", res.get("item", "")) if isinstance(res, dict) else "")
        return Recipe(
            recipe_id,
            False,
            (),
            (),
            res_id,
            1,
            FLUID_TRANSFORM,
            fluid=fluid,
            block=block,
        )

    if recipe_type in (FLUID_MEET, "fluid_mix", "fluid_interaction"):
        raw_fluids = entry.get("fluids", [])
        if isinstance(raw_fluids, str):
            fluids = (raw_fluids,)
        else:
            fluids = tuple(str(f) for f in raw_fluids)
        flowing_fluid = entry.get("flowing_fluid", entry.get("fluid_a"))
        source_fluid = entry.get("source_fluid", entry.get("fluid_b"))
        if not fluids and flowing_fluid and source_fluid:
            fluids = (str(flowing_fluid), str(source_fluid))
        if len(fluids) < 2:
            raise ValueError(f"{recipe_id} fluid_meet recipe needs two fluids")
        res = entry.get("result", entry.get("flow_result", ""))
        res_id = (
            res
            if isinstance(res, str)
            else (res.get("block", res.get("item", "")) if isinstance(res, dict) else "")
        )
        src_res = entry.get("source_result", entry.get("source_block", res_id))
        src_res_id = (
            src_res
            if isinstance(src_res, str)
            else (src_res.get("block", src_res.get("item", res_id)) if isinstance(src_res, dict) else res_id)
        )
        return Recipe(
            recipe_id,
            False,
            (),
            (),
            res_id,
            1,
            FLUID_MEET,
            fluids=fluids,
            source_result=src_res_id,
            flowing_fluid=str(flowing_fluid) if flowing_fluid else None,
            source_fluid=str(source_fluid) if source_fluid else None,
        )

    result = entry.get("result")
    if isinstance(result, str):
        result = {"item": result}
    if not isinstance(result, dict) or "item" not in result:
        raise ValueError(f"{recipe_id} needs a result item")
    result_id = _block_name(result["item"], recipe_id)
    count = int(result.get("count", 1))
    if not 1 <= count <= 64:
        raise ValueError(f"{recipe_id} result count must be from 1 to 64")

    if "pattern" in entry:
        if "ingredients" in entry:
            raise ValueError(f"{recipe_id} must be shaped or shapeless")
        rows = [str(row) for row in entry.get("pattern", [])]
        if not rows or len(rows) > 3 or any(len(row) > 3 for row in rows):
            raise ValueError(f"{recipe_id} pattern must fit in a 3x3 grid")
        key = {str(symbol): _block_name(block_id, recipe_id) for symbol, block_id in dict(entry.get("key", {})).items()}
        width = max(len(row) for row in rows)
        mapped = []
        for row in rows:
            cells = []
            for symbol in row.ljust(width):
                if symbol == " ":
                    cells.append("")
                    continue
                if symbol not in key:
                    raise ValueError(f"{recipe_id} pattern uses {symbol}, which is not in its key")
                cells.append(key[symbol])
            mapped.append(cells)
        return Recipe(recipe_id, True, _trim(mapped), (), result_id, count, recipe_type)
    if "ingredients" in entry:
        if "pattern" in entry:
            raise ValueError(f"{recipe_id} must be shaped or shapeless")
        ingredients = [_block_name(item, recipe_id) for item in entry.get("ingredients", [])]
        if not ingredients or len(ingredients) > 9:
            raise ValueError(f"{recipe_id} needs 1 to 9 ingredients")
        return Recipe(recipe_id, False, (), tuple(sorted(ingredients)), result_id, count, recipe_type)
    if "ingredient" in entry:
        raw_ing = entry.get("ingredient")
        if isinstance(raw_ing, dict):
            raw_ing = raw_ing.get("item", "")
        ingredient = _block_name(raw_ing, recipe_id) if raw_ing else ""
        ingredients = (ingredient,) if ingredient else ()
        cook_time = float(entry.get("cookingtime", entry.get("cook_time", 10.0)))
        experience = float(entry.get("experience", 0.1))
        return Recipe(recipe_id, False, (), ingredients, result_id, count, recipe_type, cook_time=cook_time, experience=experience)
    return Recipe(recipe_id, False, (), (), result_id, count, recipe_type)


def recipes_of(recipe_type):
    return tuple(recipe for recipe in RECIPES if recipe.type == recipe_type)


def smelt_match(item_id, recipe_types=None):
    if not item_id:
        return None
    from server.tags import matches_ingredient

    allowed_types = set(recipe_types) if recipe_types else {SMELTING}
    norm = BLOCKS[item_id].base_id if item_id in BLOCKS and getattr(BLOCKS[item_id], "base_id", None) else item_id
    for recipe in RECIPES:
        if recipe.type in allowed_types and not recipe.shaped and len(recipe.ingredients) == 1:
            ing = recipe.ingredients[0] if recipe.ingredients else ""
            if matches_ingredient(ing, item_id) or matches_ingredient(ing, norm):
                return recipe.result, recipe.count, recipe.cook_time, recipe.experience
    return None


def find_block_hit_recipe(item_id, block_id):
    if not item_id or not block_id:
        return None
    from server.tags import matches_ingredient

    norm = BLOCKS[block_id].base_id if block_id in BLOCKS and getattr(BLOCKS[block_id], "base_id", None) else block_id
    for recipe in RECIPES:
        if recipe.type == BLOCK_HIT:
            if not matches_ingredient(recipe.item, item_id):
                continue
            for b in recipe.blocks:
                if matches_ingredient(b, block_id) or matches_ingredient(b, norm):
                    return recipe
    return None


def _normalize_fluid_id(fluid_id):
    if not fluid_id:
        return ""
    if fluid_id in BLOCKS:
        b_obj = BLOCKS[fluid_id]
        f_prop = b_obj.properties.get("fluid")
        if isinstance(f_prop, str) and f_prop:
            return f_prop
        if getattr(b_obj, "base_id", None):
            return b_obj.base_id
    if fluid_id.endswith("_falling"):
        return fluid_id[: -len("_falling")]
    for level in range(1, 8):
        suffix = f"_{level}"
        if fluid_id.endswith(suffix):
            return fluid_id[: -len(suffix)]
    return str(fluid_id)


def find_fluid_transform(fluid_id, block_id):
    if not fluid_id or not block_id:
        return None
    norm_fluid = _normalize_fluid_id(fluid_id)
    norm = BLOCKS[block_id].base_id if block_id in BLOCKS and getattr(BLOCKS[block_id], "base_id", None) else block_id
    for recipe in RECIPES:
        if recipe.type == FLUID_TRANSFORM:
            if (
                recipe.fluid == fluid_id
                or recipe.fluid == norm_fluid
                or (norm_fluid in ("water", "salt_water", "brackish_water") and recipe.fluid == "water")
            ):
                if recipe.block == block_id or recipe.block == norm:
                    return recipe.result
    return None


def find_fluid_meet_recipe(fluid_a, fluid_b):
    """Find a fluid_meet recipe matching two meeting fluid blocks or fluid IDs."""
    norm_a = _normalize_fluid_id(fluid_a)
    norm_b = _normalize_fluid_id(fluid_b)
    if not norm_a or not norm_b or norm_a == norm_b:
        return None
    pair = {norm_a, norm_b}
    for recipe in RECIPES:
        if recipe.type == FLUID_MEET:
            if set(recipe.fluids[:2]) == pair:
                return recipe
    return None


def find_fluid_meet(fluid_a, fluid_b, target_is_source=False):
    """Return the resulting block id when fluid_a and fluid_b meet.
    If target_is_source is True (a flowing fluid hits a source block), returns recipe.source_result.
    Otherwise returns recipe.result.
    """
    recipe = find_fluid_meet_recipe(fluid_a, fluid_b)
    if recipe is None:
        return None
    if target_is_source:
        return recipe.source_result or recipe.result
    return recipe.result


class Fuel:
    def __init__(self, item_id, burn_time):
        self.item = item_id
        self.burn_time = float(burn_time)


def fuel_from_mapping(entry, source):
    if not isinstance(entry, dict):
        raise ValueError(f"{source}: fuel file must be an object")
    item = entry.get("item", entry.get("id"))
    if not item:
        raise ValueError(f"{source}: fuel must specify an item")
    burn_time = float(entry.get("burn_time", entry.get("burn_duration", entry.get("burn_ticks", 0.0))))
    if burn_time <= 0:
        raise ValueError(f"{source}: burn_time must be positive")
    return Fuel(str(item), burn_time)


def _load_fuels():
    folder = Path(ASSETS) / "fuels"
    fuels = {}
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            fuel = fuel_from_mapping(json.loads(path.read_text()), path.name)
            fuels[fuel.item] = fuel.burn_time

    items_folder = Path(ASSETS) / "items"
    if items_folder.is_dir():
        for path in sorted(items_folder.glob("*.json")):
            try:
                entry = json.loads(path.read_text())
                if isinstance(entry, dict):
                    item_id = str(entry.get("id", path.stem))
                    burn = entry.get("burn_time") if "burn_time" in entry else entry.get("fuel")
                    if burn is not None:
                        burn_val = float(burn)
                        if burn_val > 0 and item_id not in fuels:
                            fuels[item_id] = burn_val
            except Exception:
                pass

    defaults = {
        "coal": 80.0,
        "charcoal": 80.0,
        "log": 15.0,
        "pine_log": 15.0,
        "planks": 15.0,
        "pine_planks": 15.0,
        "planks_slab": 7.5,
        "pine_planks_slab": 7.5,
        "crafting_table": 15.0,
        "furnace": 15.0,
        "wood_pickaxe": 10.0,
        "stick": 5.0,
    }
    for k, v in defaults.items():
        if k not in fuels:
            fuels[k] = v
    return fuels


FUELS = _load_fuels()


def fuel_burn_time(item_id):
    if not item_id:
        return 0.0
    norm = BLOCKS[item_id].base_id if item_id in BLOCKS and getattr(BLOCKS[item_id], "base_id", None) else item_id
    if norm in FUELS:
        return FUELS[norm]
    if item_id in FUELS:
        return FUELS[item_id]
    return 0.0


def _load_one(path):
    recipe = recipe_from_mapping(json.loads(path.read_text()), path.name)
    if path.stem != recipe.id:
        raise ValueError(f"{path.name} must be named {recipe.id}.json")
    return recipe


def _load_recipes():
    folder = Path(ASSETS) / "recipes"
    if not folder.is_dir():
        return ()
    recipes = []
    seen = set()
    for path in sorted(folder.glob("*.json")):
        recipe = _load_one(path)
        if recipe.id in seen:
            raise ValueError(f"duplicate recipe {recipe.id}")
        seen.add(recipe.id)
        recipes.append(recipe)
    return tuple(recipes)


RECIPES = _load_recipes()
