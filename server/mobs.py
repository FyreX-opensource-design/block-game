"""Asset mobs.

Each file in ``assets/mobs`` (``.lua`` or ``.json``) is one mob. The model is a
builtin mesh, a lua box model, or a GeckoLib/AzureLib geometry. Goals are a
lua list of builtin goal types. Spawn rules are lua: biomes, a light range
for each channel, the blocks underfoot, and a y range.

    return {
      id = "zombie",
      name = "Zombie",
      health = 20,
      radius = 0.35,
      height = 1.95,
      model = { type = "builtin", id = "zombie" },
      goals = {
        { priority = 1, type = "melee", damage = 3, reach = 1.5, range = 16 },
        { priority = 2, type = "wander" },
      },
      spawn = {
        group = "hostile",
        order = 0,
        exclude_biomes = { "desert", "desert_hills" },
        headroom = 2,
        group_size = { 1, 2 },
        light = { sky = { max = 7 }, green = { max = 7 }, blue = { max = 7 }, uv = { max = 7 } },
      },
    }

Livestock can also declare ``breed``. ``food`` is item ids or ``#tags``.
``babies`` is one number or ``{min, max}``. ``grow_time`` is seconds until
the baby is grown; the default is one in-game day.
"""

import json
import re
from pathlib import Path

from server.blocks import ASSETS
from server.lua import load_table

_ID = re.compile(r"[a-z0-9_]+")
_LIGHT_CHANNELS = ("sky", "red", "green", "blue", "uv", "infrared")
_GOAL_TYPES = ("panic", "wander", "melee", "spider", "shoot", "eat_grass", "retaliate")


class LightBound:
    def __init__(self, minimum=None, maximum=None):
        self.minimum = None if minimum is None else float(minimum)
        self.maximum = None if maximum is None else float(maximum)

    def allows(self, value):
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True


class SpawnRule:
    def __init__(
        self,
        group,
        order,
        weight,
        biome_weights,
        biomes,
        exclude_biomes,
        blocks,
        min_y,
        max_y,
        headroom,
        group_size,
        time_of_day,
        light,
    ):
        self.group = group
        self.order = int(order)
        self.weight = float(weight)
        self.biome_weights = dict(biome_weights)
        self.biomes = frozenset(biomes)
        self.exclude_biomes = frozenset(exclude_biomes)
        self.blocks = tuple(blocks)
        self.min_y = min_y
        self.max_y = max_y
        self.headroom = int(headroom)
        self.group_size = (int(group_size[0]), int(group_size[1]))
        self.time_of_day = time_of_day
        self.light = light

    def allows_biome(self, biome_id):
        if self.biomes and biome_id not in self.biomes:
            return False
        if biome_id in self.exclude_biomes:
            return False
        return True

    def weight_for(self, biome_id):
        if not self.allows_biome(biome_id):
            return 0.0
        if biome_id in self.biome_weights:
            return float(self.biome_weights[biome_id])
        return self.weight


class ModelSpec:
    def __init__(self, kind, builtin_id, texture, texture_size, parts, geometry, animation, idle, walk, source):
        self.kind = kind
        self.builtin_id = builtin_id
        self.texture = texture
        self.texture_size = texture_size
        self.parts = parts
        self.geometry = geometry
        self.animation = animation
        self.idle = idle
        self.walk = walk
        self.source = source
        self.compiled = None

    def compile(self):
        if self.compiled is not None or self.kind == "builtin":
            return self.compiled
        from server.mob_model import build_lua_model, load_gecko_files

        if self.kind == "lua":
            self.compiled = build_lua_model(self.parts, self.texture_size)
        else:
            self.compiled = load_gecko_files(
                self.geometry,
                self.animation,
                self.texture_size,
                self.idle,
                self.walk,
            )
        return self.compiled


class BreedRule:
    def __init__(self, food, babies, grow_time, love_time, cooldown):
        self.food = tuple(food)
        self.babies = (int(babies[0]), int(babies[1]))
        self.grow_time = float(grow_time)
        self.love_time = float(love_time)
        self.cooldown = float(cooldown)


class MobDef:
    def __init__(self, mob_id, name, health, radius, height, eye, walk_speed, run_speed, climb_speed, loot, model, goals, spawn, breed, source, can_fly=False, fly_height=1.6):
        self.id = mob_id
        self.name = name
        self.health = float(health)
        self.radius = float(radius)
        self.height = float(height)
        self.eye = None if eye is None else float(eye)
        self.walk_speed = float(walk_speed)
        self.run_speed = float(run_speed)
        self.climb_speed = None if climb_speed is None else float(climb_speed)
        self.loot = loot
        self.model = model
        self.goals = tuple(goals)
        self.spawn = spawn
        self.breed = breed
        self.can_fly = bool(can_fly)
        self.fly_height = float(fly_height)
        self.source = source


def _number(value, name, source, default=None):
    if value is None and default is not None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{source}: {name} must be a number") from None


def _names(value, source, label):
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{source}: {label} must be a list of names")
    names = []
    for entry in value:
        text = str(entry)
        if text.startswith("#"):
            if _ID.fullmatch(text[1:]) is None:
                raise ValueError(f"{source}: invalid tag {text}")
        elif _ID.fullmatch(text) is None:
            raise ValueError(f"{source}: invalid {label} entry {text}")
        names.append(text)
    return names


def _resolve(path, roots):
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    for root in roots:
        found = Path(root) / path
        if found.is_file():
            return found
    raise ValueError(f"missing asset file {path}")


def _load_mapping(path):
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = load_table(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: file must return an object")
    return data


def _light_bounds(fields, source):
    raw = fields or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: light must be an object of channels")
    bounds = {}
    for channel, value in raw.items():
        name = str(channel)
        if name not in _LIGHT_CHANNELS:
            raise ValueError(f"{source}: unknown light channel {name}")
        if not isinstance(value, dict):
            raise ValueError(f"{source}: light.{name} must be an object with min and max")
        bounds[name] = LightBound(value.get("min"), value.get("max"))
    return bounds


def _parse_parts(raw, source, roots):
    if isinstance(raw, str):
        raw = _load_mapping(_resolve(raw, roots))
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{source}: lua model parts must be a list")
    parts = []
    for index, part in enumerate(raw):
        if not isinstance(part, dict):
            raise ValueError(f"{source}: part {index} must be an object")
        name = str(part.get("name") or f"part_{index}")
        pivot = part.get("pivot") or (0, 0, 0)
        box = part.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 6:
            raise ValueError(f"{source}: part {name} box must be six numbers")
        uv = part.get("uv") or {}
        if not isinstance(uv, dict):
            raise ValueError(f"{source}: part {name} uv must be an object")
        faces = {}
        for face, rect in uv.items():
            if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                raise ValueError(f"{source}: part {name} uv.{face} must be four numbers")
            faces[str(face)] = tuple(float(v) for v in rect)
        pose = str(part.get("pose") or "still")
        if pose not in ("still", "swing", "-swing", "look"):
            raise ValueError(f"{source}: part {name} pose must be still, swing, -swing, or look")
        parts.append({"name": name, "pivot": tuple(float(v) for v in pivot[:3]), "box": tuple(float(v) for v in box), "uv": faces, "pose": pose})
    return parts


def _parse_model(fields, source, roots):
    if isinstance(fields, str):
        fields = {"type": "lua", "file": fields}
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: model must be an object")
    kind = str(fields.get("type") or fields.get("kind") or "builtin")
    if kind == "azurelib":
        kind = "geckolib"
    if kind not in ("builtin", "lua", "geckolib"):
        raise ValueError(f"{source}: model type must be builtin, lua, geckolib, or azurelib")
    texture = str(fields.get("texture") or "")
    size = fields.get("texture_size") or (64, 64)
    texture_size = (float(size[0]), float(size[1]))
    parts = []
    geometry = ""
    animation = ""
    if kind == "lua":
        payload = fields
        if fields.get("file"):
            payload = _load_mapping(_resolve(fields["file"], roots))
            texture = texture or str(payload.get("texture") or "")
            if payload.get("texture_size"):
                size = payload["texture_size"]
                texture_size = (float(size[0]), float(size[1]))
        parts = _parse_parts(payload.get("parts") or fields.get("parts") or [], source, roots)
    elif kind == "geckolib":
        geometry = str(_resolve(fields.get("geometry") or fields.get("geo"), roots))
        if fields.get("animation") or fields.get("animations"):
            animation = str(_resolve(fields.get("animation") or fields.get("animations"), roots))
        texture = texture or str(fields.get("texture") or "")
    builtin_id = str(fields.get("id") or fields.get("builtin") or "")
    return ModelSpec(
        kind,
        builtin_id,
        texture,
        texture_size,
        parts,
        geometry,
        animation,
        str(fields.get("idle") or ""),
        str(fields.get("walk") or ""),
        source,
    )


def _parse_goals(fields, source, roots):
    if isinstance(fields, str):
        fields = _load_mapping(_resolve(fields, roots))
        if isinstance(fields, dict):
            fields = fields.get("goals", [])
    if not isinstance(fields, (list, tuple)):
        raise ValueError(f"{source}: goals must be a list")
    goals = []
    for index, goal in enumerate(fields):
        if not isinstance(goal, dict):
            raise ValueError(f"{source}: goal {index} must be an object")
        kind = str(goal.get("type") or goal.get("id") or "")
        if kind not in _GOAL_TYPES:
            raise ValueError(f"{source}: unknown goal type {kind}")
        goals.append(
            {
                "type": kind,
                "priority": int(_number(goal.get("priority", index + 1), "priority", source)),
                "speed": float(goal.get("speed", 2.2) or 2.2),
                "damage": float(goal.get("damage", 3) or 3),
                "reach": float(goal.get("reach", 1.5) or 1.5),
                "range": float(goal.get("range", 16) or 16),
                "preferred": float(goal.get("preferred", 14) or 14),
                "bow": str(goal.get("bow") or "wood_bow"),
                "arrow": str(goal.get("arrow") or "arrow"),
            }
        )
    return goals


def _parse_spawn(fields, source, roots):
    if isinstance(fields, str):
        fields = _load_mapping(_resolve(fields, roots))
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: spawn must be an object")
    group = str(fields.get("group") or "hostile")
    if group not in ("hostile", "passive"):
        raise ValueError(f"{source}: spawn group must be hostile or passive")
    size = fields.get("group_size") or (1, 1)
    if isinstance(size, (int, float)):
        size = (size, size)
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"{source}: group_size must be two numbers")
    time_of_day = str(fields.get("time") or "any")
    if time_of_day not in ("any", "day", "night"):
        raise ValueError(f"{source}: spawn time must be any, day, or night")
    weights = fields.get("biome_weights") or {}
    if not isinstance(weights, dict):
        raise ValueError(f"{source}: biome_weights must be an object")
    min_y = fields.get("min_y", fields.get("y_min"))
    max_y = fields.get("max_y", fields.get("y_max"))
    return SpawnRule(
        group,
        _number(fields.get("order", 0), "order", source, 0),
        _number(fields.get("weight", 1), "weight", source, 1),
        {str(key): float(value) for key, value in weights.items()},
        _names(fields.get("biomes"), source, "biomes"),
        _names(fields.get("exclude_biomes"), source, "exclude_biomes"),
        _names(fields.get("blocks"), source, "blocks"),
        None if min_y is None else float(min_y),
        None if max_y is None else float(max_y),
        _number(fields.get("headroom", 2), "headroom", source, 2),
        size,
        time_of_day,
        _light_bounds(fields.get("light"), source),
    )


def _parse_breed(fields, source):
    if not fields:
        return None
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: breed must be an object")
    food = _names(fields.get("food"), source, "food")
    if not food:
        raise ValueError(f"{source}: breed needs at least one food")
    babies = fields.get("babies", 1)
    if isinstance(babies, (int, float)):
        babies = (babies, babies)
    if not isinstance(babies, (list, tuple)) or len(babies) != 2:
        raise ValueError(f"{source}: babies must be a number or two numbers")
    lo = int(_number(babies[0], "babies", source))
    hi = int(_number(babies[1], "babies", source))
    if lo > hi:
        lo, hi = hi, lo
    if lo < 1:
        raise ValueError(f"{source}: babies must be at least 1")
    return BreedRule(
        food,
        (lo, hi),
        _number(fields.get("grow_time", 1200), "grow_time", source, 1200),
        _number(fields.get("love_time", 30), "love_time", source, 30),
        _number(fields.get("cooldown", 300), "cooldown", source, 300),
    )


def mob_from_mapping(fields, source, roots):
    if not isinstance(fields, dict):
        raise ValueError(f"{source}: mob file must be an object")
    mob_id = fields.get("id")
    if not isinstance(mob_id, str) or _ID.fullmatch(mob_id) is None:
        raise ValueError(f"{source}: mob id must be a lowercase name")
    return MobDef(
        mob_id,
        str(fields.get("name") or mob_id),
        _number(fields.get("health", 10), "health", source, 10),
        _number(fields.get("radius", 0.45), "radius", source, 0.45),
        _number(fields.get("height", 1.0), "height", source, 1),
        fields.get("eye"),
        _number(fields.get("walk_speed", 1.2), "walk_speed", source, 1.2),
        _number(fields.get("run_speed", 2.4), "run_speed", source, 2.4),
        fields.get("climb_speed"),
        str(fields.get("loot") or f"entities/{mob_id}"),
        _parse_model(fields.get("model") or {"type": "builtin", "id": mob_id}, source, roots),
        _parse_goals(fields.get("goals") or [], source, roots),
        _parse_spawn(fields.get("spawn") or {"group": "hostile"}, source, roots),
        _parse_breed(fields.get("breed"), source),
        str(source),
        bool(fields.get("fly", False)),
        _number(fields.get("fly_height", 1.6), "fly_height", source, 1.6),
    )


def load_mobs(folder=None):
    root = Path(folder) if folder is not None else ASSETS / "mobs"
    if not root.is_dir():
        return {}
    assets_root = root.parent if root.name == "mobs" else root
    roots = (root, assets_root, ASSETS)
    found = {}
    for path in sorted(root.iterdir()):
        if path.suffix not in (".json", ".lua"):
            continue
        mob = mob_from_mapping(_load_mapping(path), path, roots)
        if mob.id in found:
            raise ValueError(f"{path}: duplicate mob id {mob.id}")
        found[mob.id] = mob
    return found


def mobs_in_group(group, biome_id, catalog=None):
    catalog = MOBS if catalog is None else catalog
    chosen = [mob for mob in catalog.values() if mob.spawn.group == group and mob.spawn.weight_for(biome_id) > 0]
    chosen.sort(key=lambda mob: (mob.spawn.order, mob.id))
    return chosen


def _channel_levels(skylight, blocklight):
    levels = {"sky": float(skylight or 0)}
    if isinstance(blocklight, dict):
        levels["red"] = float(blocklight.get("r", blocklight.get("red", 0)) or 0)
        levels["green"] = float(blocklight.get("g", blocklight.get("green", 0)) or 0)
        levels["blue"] = float(blocklight.get("b", blocklight.get("blue", 0)) or 0)
        levels["uv"] = float(blocklight.get("uv", 0) or 0)
        levels["infrared"] = float(blocklight.get("ir", blocklight.get("infrared", 0)) or 0)
    elif isinstance(blocklight, (list, tuple)):
        values = list(blocklight) + [0, 0, 0, 0, 0]
        levels["red"], levels["green"], levels["blue"], levels["uv"], levels["infrared"] = (
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
            float(values[4]),
        )
    else:
        levels.update(red=0.0, green=0.0, blue=0.0, uv=0.0, infrared=0.0)
    return levels


def _read_light(world, x, y, z):
    skylight = 0
    if hasattr(world, "get_current_skylight"):
        skylight = world.get_current_skylight(x, y + 1, z)
    elif getattr(world, "lighting", None) is not None:
        skylight = world.lighting.get_skylight(x, y + 1, z)
    elif hasattr(world, "get_skylight"):
        skylight = world.get_skylight(x, y + 1, z)
    blocklight = (0, 0, 0, 0, 0)
    if hasattr(world, "get_blocklight"):
        blocklight = world.get_blocklight(x, y + 1, z)
    elif getattr(world, "lighting", None) is not None:
        blocklight = world.lighting.get_blocklight(x, y + 1, z)
    return skylight, blocklight


def _is_night(world):
    if hasattr(world, "is_night"):
        return bool(world.is_night())
    day_time = getattr(world, "day_time", 0) or 0
    return 12000.0 <= (day_time % 24000.0) <= 24000.0


def spawn_allowed(mob, world, x, y, z, skylight=None, blocklight=None):
    """True when this mob's spawn rule accepts the ground block at (x, y, z)."""
    rule = mob.spawn
    if rule.time_of_day == "day" and _is_night(world):
        return False
    if rule.time_of_day == "night" and not _is_night(world):
        return False
    if rule.min_y is not None and y < rule.min_y:
        return False
    if rule.max_y is not None and y > rule.max_y:
        return False
    ground = world.get_block(x, y, z)
    if not ground or ground == "air":
        return False
    from server.fluid import is_fluid

    if is_fluid(ground):
        return False
    if rule.blocks:
        from server.tags import matches_ingredient

        if not any(matches_ingredient(entry, ground) for entry in rule.blocks):
            return False
    else:
        from server.blocks import BLOCKS

        block = BLOCKS.get(ground)
        if not block or not getattr(block, "solid", False):
            return False
    for step in range(1, rule.headroom + 1):
        above = world.get_block(x, y + step, z)
        if is_fluid(above):
            return False
        from server.blocks import BLOCKS

        above_block = BLOCKS.get(above)
        if above_block and getattr(above_block, "solid", False):
            return False
    if skylight is None or blocklight is None:
        read_sky, read_block = _read_light(world, x, y, z)
        if skylight is None:
            skylight = read_sky
        if blocklight is None:
            blocklight = read_block
    levels = _channel_levels(skylight, blocklight)
    for channel, bound in rule.light.items():
        if not bound.allows(levels.get(channel, 0.0)):
            return False
    return True


def build_goals(mob, definition):
    from server.ai import (
        BreedGoal,
        EatGrassGoal,
        TemptGoal,
        GoalSelector,
        PanicGoal,
        SkeletonShootGoal,
        SpiderAttackGoal,
        RetaliateGoal,
        WanderGoal,
        ZombieAttackGoal,
    )

    selector = GoalSelector()
    for goal in definition.goals:
        kind = goal["type"]
        if kind == "panic":
            made = PanicGoal(mob, speed_mult=goal["speed"])
        elif kind == "wander":
            made = WanderGoal(mob)
        elif kind == "melee":
            made = ZombieAttackGoal(mob, target_range=goal["range"], reach=goal["reach"], damage=goal["damage"])
        elif kind == "retaliate":
            made = RetaliateGoal(mob, target_range=goal["range"], reach=goal["reach"], damage=goal["damage"])
        elif kind == "spider":
            made = SpiderAttackGoal(mob, target_range=goal["range"], reach=goal["reach"], damage=goal["damage"])
        elif kind == "shoot":
            made = SkeletonShootGoal(
                mob,
                target_range=goal["range"],
                preferred=goal["preferred"],
                bow_id=goal["bow"],
                arrow_id=goal["arrow"],
            )
        elif kind == "eat_grass":
            made = EatGrassGoal(mob)
        else:
            continue
        selector.add_goal(goal["priority"], made)
    if definition.breed is not None:
        selector.add_goal(1.5, BreedGoal(mob))
        # Breeding still wins when a mate is close. Following food beats wandering.
        selector.add_goal(1.6, TemptGoal(mob))
    return selector


def apply_mob_asset(mob, catalog=None):
    """Copy size, speeds, loot, and goals from the mob's asset. Returns False if it has none."""
    catalog = MOBS if catalog is None else catalog
    definition = catalog.get(getattr(mob, "mob_type", ""))
    if definition is None:
        return False
    mob.name = definition.name
    mob.walk_speed = definition.walk_speed
    mob.run_speed = definition.run_speed
    mob.radius = definition.radius
    mob.height = definition.height
    if definition.eye is not None:
        mob.eye = definition.eye
    if definition.climb_speed is not None:
        mob.can_climb = True
        mob.climb_speed = definition.climb_speed
    mob.can_fly = definition.can_fly
    mob.fly_height = definition.fly_height
    mob.loot_table = definition.loot
    mob.model = definition.model
    mob.adult_radius = mob.radius
    mob.adult_height = mob.height
    if definition.breed is not None:
        mob.breed_food = definition.breed.food
        mob.baby_count = definition.breed.babies
        mob.grow_time = definition.breed.grow_time
        mob.love_duration = definition.breed.love_time
        mob.breed_cooldown = definition.breed.cooldown
    mob.goals = build_goals(mob, definition)
    return True


MOBS = load_mobs()
