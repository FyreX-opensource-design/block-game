"""A one-block pit packed with fiber, logs, and one item stack.

Right-click the hole to fill it. Lighting the top with a fire starter cooks
the stack. The log and the fire each pick the recipe in ``assets/pit_kiln``.
How many of each ingredient, and which items light which fire, live in
``assets/pit_kiln.json``.
"""

import json
from pathlib import Path

from server.blocks import AIR, ASSETS, BLOCKS
from server.fire import FIRE_TYPES, flame_kind
from server.stacks import empty_stack, make_stack, stack_count, stack_damage, stack_id
from server.tags import has_block_tag, has_item_tag, matches_ingredient

SAVE_FILE = "pit_kilns.json"


class PitRecipe:
    def __init__(self, recipe_id, input_pattern, logs, fire, result, count, time):
        self.id = recipe_id
        self.input = input_pattern
        self.logs = logs
        self.fire = fire
        self.result = result
        self.count = int(count)
        self.time = float(time)

    def score(self, input_id, log_id, fire_kind):
        """Higher means a closer match. Negative means this recipe does not fit."""
        incoming = _rank(self.input, input_id)
        wood = _rank(self.logs, log_id)
        flame = _rank(self.fire, fire_kind)
        if incoming < 0 or wood < 0 or flame < 0:
            return -1
        return incoming * 10 + wood * 3 + flame


class PitKilnRules:
    def __init__(self, block, fiber, fiber_count, log_count, time, igniters, walls):
        self.block = block
        self.fiber = frozenset(fiber)
        self.fiber_count = int(fiber_count)
        self.log_count = int(log_count)
        self.time = float(time)
        self.igniters = dict(igniters)
        self.walls = str(walls)


def _rank(pattern, actual):
    if not actual:
        return -1
    if pattern in ("", "*"):
        return 1
    if not matches_ingredient(pattern, actual):
        return -1
    if str(pattern).startswith("#"):
        return 2
    return 4


def _number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if number <= 0:
        raise ValueError(f"{label} must be above zero")
    return number


def _whole(value, label):
    number = _number(value, label)
    if int(number) != number:
        raise ValueError(f"{label} must be a whole number")
    return int(number)


def load_pit_rules(path=None):
    file = Path(path) if path is not None else ASSETS / "pit_kiln.json"
    data = {}
    if file.is_file():
        loaded = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{file.name} must be an object")
        data = loaded
    block = str(data.get("block", "pit_kiln"))
    if block not in BLOCKS:
        raise ValueError(f"pit kiln block {block} does not exist")
    fiber = data.get("fiber", ["wheat", "plant_fiber"])
    if isinstance(fiber, str):
        fiber = [fiber]
    fiber = tuple(str(item) for item in fiber)
    if not fiber:
        raise ValueError("pit kiln needs a fiber item")
    from server.items import ITEMS

    for item_id in fiber:
        if item_id not in ITEMS:
            raise ValueError(f"pit kiln fiber {item_id} does not exist")
    igniters = {}
    raw_igniters = data.get("igniters", {})
    if not isinstance(raw_igniters, dict):
        raise ValueError("pit kiln igniters must be an object")
    for item_id, spec in raw_igniters.items():
        item_id = str(item_id)
        if item_id not in ITEMS:
            raise ValueError(f"pit kiln igniter {item_id} does not exist")
        if not isinstance(spec, dict):
            raise ValueError(f"{item_id} igniter must be an object")
        kind = str(spec.get("fire", "fire"))
        if kind not in FIRE_TYPES:
            raise ValueError(f"{item_id} lights unknown fire {kind}")
        igniters[item_id] = {
            "fire": kind,
            "consume": bool(spec.get("consume", False)),
            "damage": int(spec.get("damage", 0) or 0),
        }
    walls = str(data.get("walls", "#soil"))
    if walls.startswith("#"):
        walls = walls[1:]
    if not walls:
        raise ValueError("pit kiln needs a wall tag")
    return PitKilnRules(
        block,
        fiber,
        _whole(data.get("fiber_count", 8), "fiber_count"),
        _whole(data.get("log_count", 3), "log_count"),
        _number(data.get("time", 20), "time"),
        igniters,
        walls,
    )


def load_pit_recipes(folder=None, default_time=20.0):
    directory = Path(folder) if folder is not None else ASSETS / "pit_kiln"
    recipes = []
    seen = set()
    if not directory.is_dir():
        return tuple()
    from server.items import ITEMS

    for path in sorted(directory.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(entry, dict):
            raise ValueError(f"{path.name} must be an object")
        recipe_id = str(entry.get("id", path.stem))
        if recipe_id in seen:
            raise ValueError(f"duplicate pit kiln recipe {recipe_id}")
        seen.add(recipe_id)
        raw_input = entry.get("input", "")
        if isinstance(raw_input, dict):
            raw_input = raw_input.get("item", "")
        pattern = str(raw_input)
        if not pattern:
            raise ValueError(f"{recipe_id} needs an input")
        logs = str(entry.get("logs", "#logs"))
        fire = str(entry.get("fire", "*"))
        if fire not in ("", "*") and fire not in FIRE_TYPES:
            raise ValueError(f"{recipe_id} uses unknown fire {fire}")
        result = entry.get("result", "")
        count = 1
        if isinstance(result, dict):
            count = int(result.get("count", 1))
            result = result.get("item", "")
        result = str(result)
        if result not in ITEMS and result not in BLOCKS:
            raise ValueError(f"{recipe_id} result {result} does not exist")
        if count < 1:
            raise ValueError(f"{recipe_id} result count must be at least 1")
        time = _number(entry.get("time", default_time), f"{recipe_id} time")
        recipes.append(PitRecipe(recipe_id, pattern, logs, fire, result, count, time))
    return tuple(recipes)


PIT_RULES = load_pit_rules()
PIT_RECIPES = load_pit_recipes(default_time=PIT_RULES.time)


def igniter_spec(item_id):
    return PIT_RULES.igniters.get(item_id)


def is_fiber(item_id):
    if not item_id:
        return False
    if item_id in PIT_RULES.fiber or has_item_tag(item_id, "fiber"):
        return True
    return False


def is_kiln_log(item_id):
    return bool(item_id) and (has_block_tag(item_id, "logs") or has_item_tag(item_id, "logs"))


def find_pit_recipe(input_id, log_id, fire_kind):
    best = None
    best_score = -1
    for recipe in PIT_RECIPES:
        score = recipe.score(input_id, log_id, fire_kind)
        if score > best_score:
            best = recipe
            best_score = score
    return best


def split_stacks(item_id, count):
    from server.items import stack_limit

    left = int(count)
    if not item_id or left <= 0:
        return []
    limit = max(1, stack_limit(item_id))
    stacks = []
    while left > 0:
        taken = min(limit, left)
        stacks.append(make_stack(item_id, taken))
        left -= taken
    return stacks


def light_face(world, x, y, z, nx, ny, nz, item_id):
    """Place this starter's fire against the open face of a solid block.

    Returns ``(x, y, z, fire id)`` when a flame is placed.
    """
    spec = igniter_spec(item_id)
    if world is None or spec is None:
        return None
    kind = str(spec["fire"])
    if kind not in BLOCKS:
        return None
    support = BLOCKS.get(world.get_block(int(x), int(y), int(z)))
    if support is None or not getattr(support, "solid", False):
        return None
    px, py, pz = int(x) + int(nx), int(y) + int(ny), int(z) + int(nz)
    from server.blocks import WORLD_HEIGHT

    if py <= 0 or py >= WORLD_HEIGHT:
        return None
    if world.get_block(px, py, pz) not in ("", AIR, None):
        return None
    if not world.set_block(px, py, pz, kind):
        return None
    return (px, py, pz, kind)


def spend_igniter(slot):
    """Use one fire starter. Sticks are used up. Steel takes durability."""
    from server.stacks import damage_tool

    item_id = stack_id(slot)
    spec = igniter_spec(item_id)
    if spec is None:
        return slot
    count = stack_count(slot)
    damage = stack_damage(slot)
    if spec["consume"]:
        count -= 1
        if count <= 0:
            return empty_stack()
    slot = make_stack(item_id, count, damage=damage)
    if spec["damage"] > 0:
        slot = damage_tool(slot, spec["damage"])
    return slot


def _item_name(item_id):
    from server.items import ITEMS

    item = ITEMS.get(item_id)
    if item is not None and getattr(item, "name", ""):
        return item.name
    block = BLOCKS.get(item_id)
    if block is not None and getattr(block, "name", ""):
        return block.name
    return str(item_id).replace("_", " ").title()


def pit_contents_text(pit):
    """The line the player reads while looking at a packed pit."""
    if pit is None:
        return f"Fiber 0/{PIT_RULES.fiber_count}\nLogs 0/{PIT_RULES.log_count}"
    if pit.output:
        return "\n".join(
            f"{_item_name(stack_id(stack))} ×{stack_count(stack)}"
            for stack in pit.output
            if stack_id(stack)
        )
    if pit.fiber:
        bits = [f"{_item_name(item_id)} {count}" for item_id, count in sorted(pit.fiber.items())]
        fiber_line = f"{', '.join(bits)} ({pit.fiber_total}/{PIT_RULES.fiber_count})"
    else:
        fiber_line = f"Fiber 0/{PIT_RULES.fiber_count}"
    if pit.log_id:
        log_line = f"{_item_name(pit.log_id)} {pit.logs}/{PIT_RULES.log_count}"
    else:
        log_line = f"Logs 0/{PIT_RULES.log_count}"
    lines = [fiber_line, log_line]
    if stack_id(pit.input):
        lines.append(f"{_item_name(stack_id(pit.input))} ×{stack_count(pit.input)}")
    if pit.fire_kind:
        lines.append("Firing")
    elif (
        pit.fiber_total >= PIT_RULES.fiber_count
        and pit.logs >= PIT_RULES.log_count
        and stack_count(pit.input) > 0
    ):
        lines.append("Ready to light")
    return "\n".join(lines)


def pit_status(world, x, y, z, nx=0, ny=0, nz=0):
    """Counts for the pit under the crosshair, or empty when it is not a pit."""
    cell = pit_cell(world, x, y, z, nx, ny, nz)
    if cell is None:
        return ""
    kilns = world.pit_kilns
    pit = kilns.pit_at(*cell)
    if pit is None and not kilns.is_pit(*cell):
        return ""
    return pit_contents_text(pit)


def _any_uv(element):
    faces = element.get("faces") or {}
    if not faces:
        return None
    return next(iter(faces.values()))


def _box(origin, extent, uv):
    faces = {index: uv for index in range(6)}
    return {
        "from": [float(value) for value in origin],
        "to": [float(value) for value in extent],
        "faces": faces,
        "cutout": True,
    }


def _log_box(origin, extent, side_uv, end_uv):
    return {
        "from": [float(value) for value in origin],
        "to": [float(value) for value in extent],
        "faces": {
            0: end_uv,
            1: end_uv,
            2: side_uv,
            3: side_uv,
            4: side_uv,
            5: side_uv,
        },
        "cutout": True,
    }


def fiber_bed_height(count):
    count = max(0, int(count))
    if count <= 0:
        return 0.0
    return 1.5 + 4.5 * (count / PIT_RULES.fiber_count)


def pit_is_packed(pit):
    """The hole is full once every ingredient is in, and it stays full until emptied."""
    if pit is None:
        return False
    if pit.output:
        return True
    return (
        pit.fiber_total >= PIT_RULES.fiber_count
        and pit.logs >= PIT_RULES.log_count
        and stack_count(pit.input) > 0
    )


def pit_elements(world, x, y, z, block):
    """Straw fills the hole as fiber goes in. Each log is its own piece."""
    template = list(getattr(block, "elements", None) or [])
    if len(template) < 2 or _any_uv(template[0]) is None:
        return template
    kilns = getattr(world, "pit_kilns", None)
    pit = kilns.pit_at(x, y, z) if kilns is not None else None
    fiber_uv = _any_uv(template[0])
    log_faces = template[1].get("faces") or {}
    log_side = log_faces.get(0) or _any_uv(template[1])
    log_end = log_faces.get(2) or log_side
    marker_uv = _any_uv(template[2]) if len(template) > 2 else fiber_uv
    fiber_count = pit.fiber_total if pit is not None else 0
    log_count = pit.logs if pit is not None else 0
    if pit_is_packed(pit) and marker_uv is not None:
        return [_box((0.0, 0.0, 0.0), (16.0, 16.0, 16.0), marker_uv)]
    elements = []
    bed = fiber_bed_height(fiber_count)
    if bed > 0:
        elements.append(_box((1.0, 0.0, 1.0), (15.0, bed, 15.0), fiber_uv))
    base = bed if bed > 0 else 0.4
    for index in range(log_count):
        z0 = 1.2 + index * 4.7
        elements.append(
            _log_box(
                (1.2, base, z0),
                (14.8, base + 3.2, z0 + 3.8),
                log_side,
                log_end,
            )
        )
    top = base + (3.2 if log_count else 0.0)
    if pit is not None and stack_id(pit.input) and marker_uv is not None:
        elements.append(_box((6.0, top, 6.0), (10.0, min(15.0, top + 3.0), 10.0), marker_uv))
    elif pit is not None and pit.output and marker_uv is not None:
        elements.append(_box((5.0, 0.4, 5.0), (11.0, 4.4, 11.0), marker_uv))
    return elements or template


def pit_cell(world, x, y, z, nx=0, ny=0, nz=0):
    """The pit a click is aimed at, or None when the face is not a pit."""
    kilns = getattr(world, "pit_kilns", None)
    if kilns is None:
        return None
    x, y, z = int(x), int(y), int(z)
    block = world.get_block(x, y, z)
    if block == kilns.block_id:
        return (x, y, z)
    if flame_kind(block) and y > 0 and world.get_block(x, y - 1, z) == kilns.block_id:
        return (x, y - 1, z)
    ax, ay, az = x + int(nx), y + int(ny), z + int(nz)
    if kilns.is_pit(ax, ay, az):
        return (ax, ay, az)
    return None


class Pit:
    def __init__(self, x, y, z):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        self.fiber = {}
        self.log_id = ""
        self.logs = 0
        self.input = empty_stack()
        self.fire_kind = ""
        self.cook = 0.0
        self.duration = 0.0
        self.result_item = ""
        self.result_count = 1
        self.output = []
        self.recipe_id = ""

    @property
    def fiber_total(self):
        return sum(int(count) for count in self.fiber.values())

    @property
    def pos(self):
        return (self.x, self.y, self.z)

    def contents(self):
        """Items to give back when the pit is broken."""
        piles = []
        for item_id, count in sorted(self.fiber.items()):
            if item_id and count > 0:
                piles.append((item_id, int(count)))
        if self.log_id and self.logs > 0:
            piles.append((self.log_id, int(self.logs)))
        if stack_id(self.input) and stack_count(self.input) > 0:
            piles.append((stack_id(self.input), stack_count(self.input)))
        for stack in self.output:
            if stack_id(stack) and stack_count(stack) > 0:
                piles.append((stack_id(stack), stack_count(stack)))
        return piles

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "fiber": {str(k): int(v) for k, v in self.fiber.items() if v},
            "log": self.log_id,
            "logs": int(self.logs),
            "input": dict(self.input) if stack_id(self.input) else "",
            "fire": self.fire_kind,
            "cook": float(self.cook),
            "duration": float(self.duration),
            "result": self.result_item,
            "result_count": int(self.result_count),
            "output": [dict(stack) for stack in self.output if stack_id(stack)],
            "recipe": self.recipe_id,
        }

    @classmethod
    def from_dict(cls, data):
        pit = cls(data.get("x", 0), data.get("y", 0), data.get("z", 0))
        fiber = data.get("fiber", {})
        if isinstance(fiber, dict):
            pit.fiber = {str(k): int(v) for k, v in fiber.items() if int(v) > 0}
        pit.log_id = str(data.get("log", "") or "")
        pit.logs = max(0, int(data.get("logs", 0) or 0))
        raw_input = data.get("input", "")
        pit.input = make_stack(stack_id(raw_input), stack_count(raw_input)) if stack_id(raw_input) else empty_stack()
        pit.fire_kind = str(data.get("fire", "") or "")
        pit.cook = float(data.get("cook", 0) or 0)
        pit.duration = float(data.get("duration", 0) or 0)
        pit.result_item = str(data.get("result", "") or "")
        pit.result_count = max(1, int(data.get("result_count", 1) or 1))
        pit.output = []
        for stack in data.get("output", []) or []:
            if stack_id(stack):
                pit.output.append(make_stack(stack_id(stack), stack_count(stack)))
        pit.recipe_id = str(data.get("recipe", "") or "")
        return pit


class PitKilnSystem:
    def __init__(self, world):
        self.world = world
        self.block_id = PIT_RULES.block
        self.pits = {}
        self._load()

    def is_pit(self, x, y, z):
        """An open hole, or one already packed: solid floor and four soil sides."""
        x, y, z = int(x), int(y), int(z)
        if y <= 1 or y + 1 >= self._height():
            return False
        block = self.world.get_block(x, y, z)
        if block not in ("", AIR, None, self.block_id):
            return False
        if not self.world.is_solid(x, y - 1, z):
            return False
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if not has_block_tag(self.world.get_block(x + dx, y, z + dz), PIT_RULES.walls):
                return False
        above = self.world.get_block(x, y + 1, z)
        return above in ("", AIR, None) or bool(flame_kind(above))

    def pit_at(self, x, y, z):
        return self.pits.get((int(x), int(y), int(z)))

    def has_output(self, x, y, z):
        pit = self.pit_at(x, y, z)
        return pit is not None and bool(pit.output)

    def add(self, x, y, z, item_id, count):
        """Put items into the hole. Returns how many were accepted."""
        from server.items import canonical_item, stack_limit

        count = int(count)
        if count <= 0:
            return 0
        item_id = canonical_item(item_id)
        if not item_id or not self.is_pit(x, y, z):
            return 0
        pos = (int(x), int(y), int(z))
        pit = self.pits.get(pos)
        if pit is not None and (pit.fire_kind or pit.output):
            return 0
        if pit is None:
            pit = Pit(*pos)
        taken = self._accept(pit, item_id, count, stack_limit(item_id))
        if taken <= 0:
            return 0
        self.pits[pos] = pit
        if self.world.get_block(*pos) != self.block_id:
            if not self.world.set_block(*pos, self.block_id):
                self._refund(pit, item_id, taken)
                if not pit.contents():
                    self.pits.pop(pos, None)
                return 0
        self._remesh(*pos)
        return taken

    def can_light(self, x, y, z, item_id):
        spec = igniter_spec(item_id)
        pit = self.pit_at(x, y, z)
        if spec is None or pit is None or pit.output:
            return False
        if not self._top_clear(pit, spec["fire"]):
            return False
        if pit.fire_kind:
            return pit.fire_kind == spec["fire"] and self.world.get_block(*self._top(pit)) != pit.fire_kind
        return self._packed(pit) and find_pit_recipe(stack_id(pit.input), pit.log_id, spec["fire"]) is not None

    def ignite(self, x, y, z, item_id):
        """Light the cell above the pit. The starter decides the fire."""
        if not self.can_light(x, y, z, item_id):
            return False
        spec = igniter_spec(item_id)
        pit = self.pit_at(x, y, z)
        kind = spec["fire"]
        if not pit.fire_kind:
            recipe = find_pit_recipe(stack_id(pit.input), pit.log_id, kind)
            if recipe is None:
                return False
            pit.recipe_id = recipe.id
            pit.duration = recipe.time
            pit.result_item = recipe.result
            pit.result_count = recipe.count
            pit.fire_kind = kind
        top = self._top(pit)
        block = self.world.get_block(*top)
        if block in ("", AIR, None):
            if not self.world.set_block(*top, kind):
                if pit.cook <= 0:
                    pit.fire_kind = ""
                return False
        elif block != kind:
            return False
        self.world.fire.hold(*top, kind)
        return True

    def take(self, x, y, z):
        """Remove the finished stacks and clear the hole."""
        pos = (int(x), int(y), int(z))
        pit = self.pits.get(pos)
        if pit is None or not pit.output:
            return []
        stacks = list(pit.output)
        pit.output = []
        pit.fiber = {}
        pit.logs = 0
        pit.log_id = ""
        pit.input = empty_stack()
        pit.fire_kind = ""
        self._extinguish(pit)
        if not self.world.set_block(*pos, AIR):
            pit.output = stacks
            return []
        return stacks

    def break_at(self, x, y, z, spawn=True):
        pos = (int(x), int(y), int(z))
        pit = self.pits.pop(pos, None)
        if pit is None:
            return
        self._extinguish(pit)
        if not spawn:
            return
        for item_id, count in pit.contents():
            self.world.spawn_item(item_id, count=count, pos=(pos[0] + 0.5, pos[1] + 0.5, pos[2] + 0.5))

    def update(self, dt):
        dt = max(0.0, float(dt))
        if dt <= 0:
            return
        for pos, pit in list(self.pits.items()):
            if not self._resident(*pos):
                continue
            if self.world.get_block(*pos) != self.block_id:
                self.break_at(*pos, spawn=True)
                continue
            if not pit.fire_kind or pit.output:
                continue
            top = self._top(pit)
            if self.world.get_block(*top) != pit.fire_kind:
                self.world.fire.release(*top)
                continue
            self.world.fire.hold(*top, pit.fire_kind)
            pit.cook += dt
            if (
                pit.duration <= 0
                or pit.cook < pit.duration
                or not pit.result_item
                or stack_count(pit.input) <= 0
            ):
                continue
            amount = stack_count(pit.input) * pit.result_count
            pit.output = split_stacks(pit.result_item, amount)
            pit.input = empty_stack()
            pit.fiber = {}
            pit.logs = 0
            pit.log_id = ""
            pit.fire_kind = ""
            pit.cook = 0.0
            self._extinguish(pit)
            self._remesh(*pos)

    def export(self):
        return [pit.to_dict() for pit in self.pits.values()]

    def import_data(self, data):
        self.pits = {}
        if not data:
            return
        entries = data.get("pits", data) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            pit = Pit.from_dict(entry)
            self.pits[pit.pos] = pit

    def save(self):
        store = getattr(self.world, "store", None)
        if store is None or not hasattr(store, "write_meta"):
            return
        store.write_meta(SAVE_FILE, {"version": 1, "pits": self.export()})

    def _load(self):
        store = getattr(self.world, "store", None)
        if store is None or not hasattr(store, "read_meta"):
            return
        self.import_data(store.read_meta(SAVE_FILE))

    def _accept(self, pit, item_id, count, limit):
        # The hole stays empty until fiber goes in. Logs and the item stack follow.
        if pit.fiber_total <= 0 and not is_fiber(item_id):
            return 0
        if is_fiber(item_id) and pit.fiber_total < PIT_RULES.fiber_count:
            taken = min(count, PIT_RULES.fiber_count - pit.fiber_total)
            pit.fiber[item_id] = pit.fiber.get(item_id, 0) + taken
            return taken
        if is_kiln_log(item_id) and pit.logs < PIT_RULES.log_count:
            if pit.log_id and pit.log_id != item_id and not matches_ingredient(pit.log_id, item_id):
                return 0
            taken = min(count, PIT_RULES.log_count - pit.logs)
            pit.log_id = pit.log_id or item_id
            pit.logs += taken
            return taken
        if igniter_spec(item_id) is not None:
            return 0
        current = stack_id(pit.input)
        if current and current != item_id:
            return 0
        room = max(0, limit - stack_count(pit.input))
        taken = min(count, room)
        if taken <= 0:
            return 0
        pit.input = make_stack(item_id, stack_count(pit.input) + taken)
        return taken

    def _refund(self, pit, item_id, count):
        if item_id in pit.fiber:
            pit.fiber[item_id] = max(0, pit.fiber[item_id] - count)
            if pit.fiber[item_id] <= 0:
                pit.fiber.pop(item_id, None)
            return
        if pit.log_id == item_id and pit.logs:
            pit.logs = max(0, pit.logs - count)
            if pit.logs <= 0:
                pit.log_id = ""
            return
        if stack_id(pit.input) == item_id:
            left = stack_count(pit.input) - count
            pit.input = make_stack(item_id, left) if left > 0 else empty_stack()

    def _packed(self, pit):
        return (
            pit.fiber_total >= PIT_RULES.fiber_count
            and pit.logs >= PIT_RULES.log_count
            and stack_count(pit.input) > 0
        )

    def _top(self, pit):
        return (pit.x, pit.y + 1, pit.z)

    def _top_clear(self, pit, kind):
        block = self.world.get_block(*self._top(pit))
        return block in ("", AIR, None) or block == kind

    def _extinguish(self, pit):
        x, y, z = self._top(pit)
        if y >= self._height():
            return
        fire = getattr(self.world, "fire", None)
        if fire is not None:
            fire.release(x, y, z)
        if flame_kind(self.world.get_block(x, y, z)):
            self.world.set_block(x, y, z, AIR)

    def _remesh(self, x, y, z):
        cx, cy, cz = self.world.chunk_key(x, y, z)
        self.world.mark_dirty(cx, cy, cz)

    def _resident(self, x, y, z):
        return self.world.chunk_key(x, y, z) in self.world.chunks

    def _height(self):
        from server.blocks import WORLD_HEIGHT

        return WORLD_HEIGHT
