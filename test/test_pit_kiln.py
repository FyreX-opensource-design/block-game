import unittest

from server.blocks import BLOCKS
from server.loot import get_block_drops
from server.items import ITEMS
from server.pit_kiln import (
    PIT_RECIPES,
    PIT_RULES,
    fiber_bed_height,
    find_pit_recipe,
    igniter_spec,
    pit_cell,
    pit_contents_text,
    pit_elements,
    pit_is_packed,
    pit_status,
    spend_igniter,
)
from server.recipes import craft_match
from server.stacks import make_stack, stack_count, stack_damage, stack_id
from server.world import World


def _world():
    return World(seed=1, caves=[], ores=[], features=[], underground_biomes=[])


def _dig(world, x=8, y=40, z=8):
    world.load_chunk(0, y // 16, 0)
    for ix in range(x - 2, x + 3):
        for iz in range(z - 2, z + 3):
            world.set_block(ix, y - 1, iz, "stone")
            world.set_block(ix, y, iz, "stone")
            world.set_block(ix, y + 1, iz, "air")
            world.set_block(ix, y + 2, iz, "air")
    world.set_block(x, y, z, "air")
    return x, y, z


def _fill(world, x, y, z, fuel="log", item="clay", amount=4, fiber="wheat"):
    kilns = world.pit_kilns
    kilns.add(x, y, z, fiber, 3)
    kilns.add(x, y, z, "plant_fiber", 8)
    kilns.add(x, y, z, fuel, 3)
    kilns.add(x, y, z, item, amount)
    return kilns


def _piles(world):
    found = {}
    for entity in world.entities:
        item = getattr(entity, "item", "")
        if item:
            found[item] = found.get(item, 0) + entity.count
    return found


class PitKilnTests(unittest.TestCase):
    def test_igniters_and_their_recipes(self):
        self.assertEqual(PIT_RULES.fiber_count, 8)
        self.assertEqual(PIT_RULES.log_count, 3)
        self.assertEqual(igniter_spec("fire_stick")["fire"], "fire")
        self.assertTrue(igniter_spec("fire_stick")["consume"])
        self.assertEqual(igniter_spec("flint_and_steel")["fire"], "fire")
        self.assertGreater(igniter_spec("flint_and_steel")["damage"], 0)
        self.assertEqual(igniter_spec("sulfuric_steel")["fire"], "blue_fire")
        self.assertIn("fire_stick", ITEMS)
        self.assertIn("pit_kiln", BLOCKS)
        self.assertEqual(get_block_drops("pit_kiln"), [])

        self.assertEqual(craft_match(["plant_fiber", "", "stick", ""])[:2], ("fire_stick", 4))
        self.assertEqual(craft_match(["wheat", "", "stick", ""])[:2], ("fire_stick", 4))
        self.assertEqual(
            craft_match(["knapped_flint", "", "", "iron_ingot"])[:2],
            ("flint_and_steel", 1),
        )
        self.assertEqual(
            craft_match(["sulfur", "", "", "iron_ingot"])[:2],
            ("sulfuric_steel", 1),
        )
        spent = spend_igniter(make_stack("fire_stick", 4))
        self.assertEqual(stack_count(spent), 3)
        steel = spend_igniter(make_stack("flint_and_steel", 1))
        self.assertEqual(stack_id(steel), "flint_and_steel")
        self.assertEqual(stack_damage(steel), 1)
        blue = spend_igniter(make_stack("sulfuric_steel", 1))
        self.assertEqual(stack_damage(blue), 1)

    def test_the_pit_shows_how_full_it_is(self):
        world = _world()
        x, y, z = _dig(world)
        self.assertEqual(
            pit_status(world, x, y - 1, z, 0, 1, 0),
            f"Fiber 0/{PIT_RULES.fiber_count}\nLogs 0/{PIT_RULES.log_count}",
        )
        kilns = world.pit_kilns
        kilns.add(x, y, z, "wheat", 3)
        kilns.add(x, y, z, "plant_fiber", 2)
        kilns.add(x, y, z, "birch_log", 2)
        pit = kilns.pit_at(x, y, z)
        text = pit_contents_text(pit)
        self.assertIn("Wheat 3", text)
        self.assertIn("Plant Fiber 2", text)
        self.assertIn(f"(5/{PIT_RULES.fiber_count})", text)
        self.assertIn(f"Birch Log 2/{PIT_RULES.log_count}", text)
        self.assertEqual(pit_status(world, x, y, z), text)

        self.assertFalse(pit_is_packed(pit))
        short = pit_elements(world, x, y, z, BLOCKS["pit_kiln"])
        kilns.add(x, y, z, "wheat", 8)
        kilns.add(x, y, z, "birch_log", 1)
        kilns.add(x, y, z, "clay", 4)
        full = pit_elements(world, x, y, z, BLOCKS["pit_kiln"])
        logs = [element for element in short if element["from"][0] == 1.2]
        self.assertEqual(short[0]["to"][1], fiber_bed_height(5))
        self.assertEqual(len(logs), 2)
        packed = kilns.pit_at(x, y, z)
        self.assertTrue(pit_is_packed(packed))
        self.assertEqual(len(full), 1)
        self.assertEqual(full[0]["from"], [0.0, 0.0, 0.0])
        self.assertEqual(full[0]["to"], [16.0, 16.0, 16.0])
        self.assertIn("Clay ×4", pit_contents_text(packed))
        self.assertIn("Ready to light", pit_contents_text(packed))
        packed.output = [make_stack("brick", 4)]
        packed.input = {"id": "", "count": 0}
        packed.fiber = {}
        packed.logs = 0
        done = pit_elements(world, x, y, z, BLOCKS["pit_kiln"])
        self.assertEqual(done[0]["to"], [16.0, 16.0, 16.0])
        from server.mesh import build_chunk_mesh

        self.assertFalse(BLOCKS["pit_kiln"].full_cube)
        self.assertIsNotNone(build_chunk_mesh(world, 0, y // 16, 0))

    def test_log_and_fire_choose_the_recipe(self):
        oak = find_pit_recipe("log", "log", "fire")
        birch = find_pit_recipe("log", "birch_log", "fire")
        blue = find_pit_recipe("log", "log", "blue_fire")
        both = find_pit_recipe("log", "birch_log", "blue_fire")
        self.assertEqual(oak.result, "charcoal")
        self.assertEqual(birch.result, "charcoal")
        self.assertGreater(birch.count, oak.count)
        self.assertGreater(blue.count, oak.count)
        self.assertGreater(both.count, birch.count)
        self.assertGreater(both.count, blue.count)
        self.assertLess(both.time, oak.time)

        bricks = find_pit_recipe("clay", "log", "fire")
        hot_bricks = find_pit_recipe("clay", "birch_log", "blue_fire")
        self.assertEqual(bricks.result, "brick")
        self.assertEqual(hot_bricks.result, "brick")
        self.assertGreater(hot_bricks.count, bricks.count)
        self.assertLess(hot_bricks.time, bricks.time)

        glass = find_pit_recipe("sand", "log", "fire")
        hot_glass = find_pit_recipe("sand", "log", "blue_fire")
        self.assertEqual(glass.result, "glass")
        self.assertEqual(hot_glass.result, "glass")
        self.assertLess(hot_glass.time, glass.time)
        self.assertGreater(len(PIT_RECIPES), 0)

    def test_a_hole_is_packed_then_fired(self):
        world = _world()
        x, y, z = _dig(world)
        self.assertIsNone(world.pit_kilns.add(x, y + 5, z, "wheat", 1) or None)
        self.assertEqual(world.pit_kilns.add(4, y, 4, "wheat", 1), 0)

        kilns = _fill(world, x, y, z)
        pit = kilns.pit_at(x, y, z)
        self.assertEqual(pit.fiber_total, 8)
        self.assertEqual(pit.logs, 3)
        self.assertEqual(pit.log_id, "log")
        self.assertEqual(stack_count(pit.input), 4)
        self.assertEqual(world.get_block(x, y, z), "pit_kiln")
        self.assertEqual(kilns.add(x, y, z, "birch_log", 1), 0)
        self.assertEqual(pit.logs, 3)

        other = _world()
        ox, oy, oz = _dig(other)
        self.assertEqual(other.pit_kilns.add(ox, oy, oz, "log", 1), 1)
        self.assertEqual(other.pit_kilns.add(ox, oy, oz, "birch_log", 1), 0)
        self.assertEqual(other.pit_kilns.add(ox, oy, oz, "log_x", 2), 2)
        self.assertEqual(other.pit_kilns.pit_at(ox, oy, oz).logs, 3)
        self.assertEqual(other.pit_kilns.pit_at(ox, oy, oz).log_id, "log")

        aimed = pit_cell(world, x, y - 1, z, 0, 1, 0)
        self.assertEqual(aimed, (x, y, z))
        self.assertTrue(kilns.ignite(x, y, z, "fire_stick"))
        self.assertEqual(world.get_block(x, y + 1, z), "fire")
        self.assertIn((x, y + 1, z), world.fire.held)

        world.set_block(x + 1, y + 1, z, "log")
        world.set_block(x, y + 1, z + 1, "tall_grass")
        recipe = find_pit_recipe("clay", "log", "fire")
        for _ in range(8):
            world.update_fire(0.5)
            world.update_pit_kilns(0.4)
        self.assertEqual(world.get_block(x + 1, y + 1, z), "log")
        self.assertEqual(world.get_block(x, y + 1, z + 1), "tall_grass")
        self.assertEqual(world.get_block(x, y + 1, z), "fire")
        self.assertFalse(pit.output)

        world.set_block(x, y + 1, z, "air")
        world.update_pit_kilns(recipe.time)
        self.assertFalse(pit.output)
        self.assertGreater(pit.cook, 0)
        self.assertTrue(kilns.ignite(x, y, z, "flint_and_steel"))
        self.assertEqual(world.get_block(x, y + 1, z), "fire")
        world.update_pit_kilns(recipe.time)
        self.assertEqual(world.get_block(x, y + 1, z), "air")
        self.assertNotIn((x, y + 1, z), world.fire.held)
        stacks = kilns.take(x, y, z)
        self.assertEqual(sum(stack_count(stack) for stack in stacks), recipe.count * 4)
        self.assertEqual(stack_id(stacks[0]), "brick")
        self.assertEqual(world.get_block(x, y, z), "air")
        self.assertIsNone(kilns.pit_at(x, y, z))

    def test_birch_fuel_and_blue_fire_change_the_yield(self):
        world = _world()
        x, y, z = _dig(world)
        kilns = _fill(world, x, y, z, fuel="birch_log", item="log", amount=5)
        self.assertTrue(kilns.ignite(x, y, z, "sulfuric_steel"))
        self.assertEqual(world.get_block(x, y + 1, z), "blue_fire")
        recipe = find_pit_recipe("log", "birch_log", "blue_fire")
        world.update_pit_kilns(recipe.time)
        stacks = kilns.take(x, y, z)
        self.assertEqual(stack_id(stacks[0]), "charcoal")
        self.assertEqual(sum(stack_count(stack) for stack in stacks), recipe.count * 5)
        self.assertEqual(recipe.count, 3)

    def test_breaking_the_pit_returns_what_was_packed(self):
        world = _world()
        x, y, z = _dig(world)
        _fill(world, x, y, z, amount=6)
        world.set_block(x, y, z, "air")
        piles = _piles(world)
        self.assertEqual(piles.get("wheat", 0), 3)
        self.assertEqual(piles.get("plant_fiber", 0), 5)
        self.assertEqual(piles.get("log", 0), 3)
        self.assertEqual(piles.get("clay", 0), 6)
        self.assertIsNone(world.pit_kilns.pit_at(x, y, z))

    def test_a_pit_remembers_its_contents(self):
        world = _world()
        x, y, z = _dig(world)
        kilns = _fill(world, x, y, z, fuel="birch_log", amount=2)
        kilns.ignite(x, y, z, "fire_stick")
        world.update_pit_kilns(1.5)
        saved = kilns.export()

        other = _world()
        ox, oy, oz = _dig(other)
        other.set_block(ox, oy, oz, "pit_kiln")
        other.pit_kilns.import_data(saved)
        pit = other.pit_kilns.pit_at(ox, oy, oz)
        self.assertEqual(pit.log_id, "birch_log")
        self.assertEqual(pit.logs, 3)
        self.assertEqual(stack_id(pit.input), "clay")
        self.assertEqual(pit.fire_kind, "fire")
        self.assertAlmostEqual(pit.cook, 1.5)
        other.update_pit_kilns(0.1)
        self.assertEqual(other.get_block(ox, oy + 1, oz), "air")
