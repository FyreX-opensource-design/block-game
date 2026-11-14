import unittest
from unittest.mock import MagicMock

from client.app import Game
from server.biomes import BIOMES
from server.blocks import AIR, ASSETS, can_place_block, can_sustain_plant
from server.crops import CROPS
from server.interactions import INTERACTIONS, Interaction, tool_interaction_at
from server.items import ITEMS, creative_items
from server.loot import get_block_drops, get_tool_type
from server.recipes import craft_match
from server.stacks import make_stack, stack_damage, stack_id
from server.tags import has_block_tag
from server.world import World


def _flat_world():
    world = World(seed=1, caves=[])
    world.load_chunk(0, 5, 0)
    world.chunks[(0, 5, 0)].data.fill(0)
    return world


def _age(world, seconds, step=0.5):
    for _ in range(int(seconds / step)):
        world.update_transformations(step)


class FarmingTests(unittest.TestCase):
    def test_hoe_is_a_tool_and_can_be_crafted(self):
        hoe = ITEMS["iron_hoe"]
        self.assertEqual(hoe.tool, "hoe")
        self.assertEqual(get_tool_type(hoe), "hoe")
        self.assertIn("wood_hoe", creative_items("tools"))
        self.assertIn("wheat_seeds", creative_items("natural"))
        grid = (
            "planks", "planks", "",
            "", "stick", "",
            "", "stick", "",
        )
        made = craft_match(grid)
        self.assertIsNotNone(made)
        self.assertEqual(made[0], "wood_hoe")
        stone = craft_match((
            "stone", "stone", "",
            "", "stick", "",
            "", "stick", "",
        ))
        self.assertEqual(stone[0], "stone_hoe")

    def test_hoe_tills_soil_from_the_interaction_asset(self):
        till = next(rule for rule in INTERACTIONS if rule.id == "till_soil")
        self.assertEqual(till.tool, "hoe")
        self.assertEqual(till.result, "farmland")
        self.assertEqual(till.blocks, ("dirt", "grass", "mud", "muddy_grass"))

        world = _flat_world()
        hoe = ITEMS["wood_hoe"]
        world.set_block(2, 80, 5, "dirt")
        world.set_block(3, 80, 5, "mud")
        world.set_block(4, 80, 5, "muddy_grass")
        world.set_block(6, 80, 5, "grass")
        world.set_block(8, 80, 5, "dirt")
        world.set_block(8, 81, 5, "stone")
        world.set_block(10, 80, 5, "stone")
        for x in (2, 3, 4, 6):
            found = tool_interaction_at(world, hoe, x, 80, 5)
            self.assertEqual(found[:3], (x, 80, 5))
            self.assertEqual(found[3].result, "farmland")
        self.assertIsNone(tool_interaction_at(world, hoe, 8, 80, 5))
        self.assertIsNone(tool_interaction_at(world, hoe, 10, 80, 5))
        self.assertIsNone(tool_interaction_at(world, ITEMS["stone_shovel"], 4, 80, 5))

        world.set_block(2, 81, 5, "tall_grass")
        self.assertEqual(tool_interaction_at(world, hoe, 2, 81, 5)[:3], (2, 80, 5))

        game = MagicMock()
        game.world = world
        game.player = MagicMock()
        game.selected = 0
        game.hotbar = [make_stack("stone_shovel", 1)]
        game.gamemode = "survival"
        game.network_client = None
        game._use_held_tool = lambda amount=1: Game._use_held_tool(game, amount)
        self.assertFalse(Game._handle_tool_interaction(game, {"x": 4, "y": 80, "z": 5}))
        self.assertEqual(world.get_block(4, 80, 5), "muddy_grass")

        game.hotbar = [make_stack("wood_hoe", 1)]
        self.assertTrue(Game._handle_tool_interaction(game, {"x": 2, "y": 81, "z": 5}))
        self.assertEqual(world.get_block(2, 80, 5), "farmland")
        self.assertEqual(world.get_block(2, 81, 5), AIR)
        self.assertIn("plant_fiber", {getattr(entity, "item", "") for entity in world.entities})
        self.assertEqual(stack_damage(game.hotbar[0]), 1)
        self.assertEqual(stack_id(game.hotbar[0]), "wood_hoe")

        self.assertTrue(Game._handle_tool_interaction(game, {"x": 6, "y": 80, "z": 5}))
        self.assertEqual(world.get_block(6, 80, 5), "farmland")
        self.assertEqual(stack_damage(game.hotbar[0]), 2)

        game.gamemode = "creative"
        self.assertTrue(Game._handle_tool_interaction(game, {"x": 3, "y": 80, "z": 5}))
        self.assertEqual(world.get_block(3, 80, 5), "farmland")
        self.assertEqual(stack_damage(game.hotbar[0]), 2)

    def test_tool_interactions_can_target_other_blocks(self):
        world = _flat_world()
        world.set_block(2, 80, 5, "stone")
        world.set_block(2, 81, 5, "stone")
        rule = Interaction(
            "flatten_stone",
            "cobble",
            ("stone",),
            tool="shovel",
            damage=0,
            require_clear_above=False,
        )
        found = tool_interaction_at(world, ITEMS["stone_shovel"], 2, 80, 5, interactions=(rule,))
        self.assertEqual(found[:3], (2, 80, 5))
        self.assertEqual(found[3].result, "cobble")
        tagged = Interaction("till_tagged_soil", "farmland", ("#soil",), tool="hoe")
        world.set_block(4, 80, 5, "grass")
        self.assertEqual(
            tool_interaction_at(world, ITEMS["wood_hoe"], 4, 80, 5, interactions=(tagged,))[:3],
            (4, 80, 5),
        )

    def test_dry_farmland_becomes_dirt_unless_fresh_water_is_close(self):
        self.assertTrue(has_block_tag("water", "fresh_water"))
        self.assertTrue(has_block_tag("water_7", "fresh_water"))
        self.assertFalse(has_block_tag("salt_water", "fresh_water"))
        self.assertFalse(has_block_tag("milk", "fresh_water"))

        dry = _flat_world()
        dry.set_block(2, 80, 5, "farmland")
        _age(dry, 20)
        self.assertEqual(dry.get_block(2, 80, 5), "farmland")
        _age(dry, 70)
        self.assertEqual(dry.get_block(2, 80, 5), "dirt")

        near = _flat_world()
        near.set_block(2, 80, 5, "farmland")
        near.set_block(6, 80, 5, "water")
        _age(near, 90)
        self.assertEqual(near.get_block(2, 80, 5), "farmland")

        far = _flat_world()
        far.set_block(2, 80, 5, "farmland")
        far.set_block(7, 80, 5, "water")
        _age(far, 90)
        self.assertEqual(far.get_block(2, 80, 5), "dirt")

        above = _flat_world()
        above.set_block(2, 80, 5, "farmland")
        above.set_block(2, 81, 5, "water_7")
        _age(above, 90)
        self.assertEqual(above.get_block(2, 80, 5), "farmland")

        too_high = _flat_world()
        too_high.set_block(2, 80, 5, "farmland")
        too_high.set_block(2, 82, 5, "water")
        _age(too_high, 90)
        self.assertEqual(too_high.get_block(2, 80, 5), "dirt")

        for other in ("salt_water", "milk"):
            world = _flat_world()
            world.set_block(2, 80, 5, "farmland")
            world.set_block(3, 80, 5, other)
            _age(world, 90)
            self.assertEqual(world.get_block(2, 80, 5), "dirt", other)

    def test_wheat_grows_eight_stages_on_watered_farmland(self):
        wheat = next(crop for crop in CROPS if crop.id == "wheat")
        self.assertEqual([f"wheat_{i}" for i in range(8)], list(wheat.stages))
        # Seven steps of 180–340s span about one to two day cycles (1200s each).
        self.assertGreaterEqual(wheat.min_time * 7, 1200)
        self.assertLessEqual(wheat.max_time * 7, 2400)
        world = _flat_world()
        world.set_block(2, 80, 5, "dirt")
        hit = {"x": 2, "y": 80, "z": 5, "ny": 1}
        self.assertFalse(can_place_block(world, 2, 81, 5, "wheat_0", hit=hit))
        world.set_block(2, 80, 5, "farmland")
        self.assertTrue(can_place_block(world, 2, 81, 5, "wheat_0", hit=hit))

        world.set_block(2, 81, 5, "wheat_0")
        world.crops.timers[(2, 81, 5)][1] = 0.01
        world.update_crops(1.0)
        self.assertEqual(world.get_block(2, 81, 5), "wheat_0")

        world.set_block(4, 81, 5, "water")
        for stage in range(7):
            world.crops.timers[(2, 81, 5)][1] = 0.01
            world.update_crops(1.0)
            self.assertEqual(world.get_block(2, 81, 5), f"wheat_{stage + 1}")
        self.assertNotIn((2, 81, 5), world.crops.timers)

        young = {item for item, _count in get_block_drops("wheat_3")}
        ripe = {item for item, _count in get_block_drops("wheat_7")}
        self.assertEqual(young, {"wheat_seeds"})
        self.assertIn("wheat", ripe)
        self.assertIn("wheat_seeds", ripe)
        self.assertEqual(ITEMS["wheat"].food, 5)
        self.assertEqual(ITEMS["wheat_seeds"].block, "wheat_0")

        dry = _flat_world()
        dry.set_block(2, 80, 5, "farmland")
        dry.set_block(2, 81, 5, "wheat_3")
        _age(dry, 90)
        self.assertEqual(dry.get_block(2, 80, 5), "dirt")
        self.assertEqual(dry.get_block(2, 81, 5), AIR)
        self.assertIn("wheat_seeds", {getattr(entity, "item", "") for entity in dry.entities})

    def test_carrot_and_potato_species_come_from_crop_files(self):
        species = {
            "orange_carrot": (4, ("plains", "lowland_plains")),
            "white_carrot": (4, ("birch_forest", "birch_forest_hills")),
            "purple_carrot": (4, ("highland_plains", "mountain_meadow")),
            "potato": (3, ("forest", "wooded_mountains")),
            "red_potato": (3, ("savanna", "savanna_hills")),
            "blue_potato": (3, ("pine_forest", "mountains")),
        }
        by_id = {crop.id: crop for crop in CROPS}
        self.assertFalse((ASSETS / "blocks" / "orange_carrot_0.json").is_file())
        self.assertFalse((ASSETS / "items" / "orange_carrot.json").is_file())

        for crop_id, (food, biomes) in species.items():
            crop = by_id[crop_id]
            self.assertEqual(list(crop.stages), [f"{crop_id}_{i}" for i in range(8)])
            self.assertGreaterEqual(crop.min_time * 7, 1200)
            self.assertLessEqual(crop.max_time * 7, 2400)
            seeds = f"{crop_id}_seeds"
            self.assertEqual(ITEMS[seeds].block, f"{crop_id}_0")
            self.assertEqual(ITEMS[seeds].right, "place")
            self.assertIn(seeds, creative_items("natural"))
            self.assertEqual(ITEMS[crop_id].food, food)
            self.assertEqual(ITEMS[crop_id].right, "eat")
            self.assertIn(crop_id, creative_items("materials"))
            young = {item for item, _count in get_block_drops(f"{crop_id}_2")}
            ripe = {item for item, _count in get_block_drops(f"{crop_id}_7", rng=_FixedRng(0))}
            self.assertEqual(young, {seeds})
            self.assertEqual(ripe, {crop_id, seeds})
            wild = f"wild_{crop_id}"
            self.assertTrue(can_sustain_plant("grass", wild))
            self.assertFalse(can_sustain_plant("farmland", wild))
            sure = {item for item, _count in get_block_drops(wild, rng=_FixedRng(0))}
            sparse = {item for item, _count in get_block_drops(wild, rng=_FixedRng(0.99))}
            self.assertEqual(sure, {seeds, crop_id})
            self.assertEqual(sparse, {seeds})
            for biome_id in biomes:
                biome = _biome(biome_id)
                names = [dec.id for dec in biome.decorations]
                self.assertIn(wild, names)
                self.assertLessEqual(sum(dec.chance for dec in biome.decorations), 1.0)

        for biome_id in ("desert", "ocean", "beach"):
            names = [dec.id for dec in _biome(biome_id).decorations]
            self.assertFalse(any(name.startswith("wild_") for name in names))

        world = _flat_world()
        world.set_block(2, 80, 5, "dirt")
        hit = {"x": 2, "y": 80, "z": 5, "ny": 1}
        self.assertFalse(can_place_block(world, 2, 81, 5, "orange_carrot_0", hit=hit))
        world.set_block(2, 80, 5, "farmland")
        self.assertTrue(can_place_block(world, 2, 81, 5, "orange_carrot_0", hit=hit))
        world.set_block(2, 81, 5, "orange_carrot_0")
        world.set_block(4, 81, 5, "water")
        world.crops.timers[(2, 81, 5)][1] = 0.01
        world.update_crops(1.0)
        self.assertEqual(world.get_block(2, 81, 5), "orange_carrot_1")


def _biome(biome_id):
    for biome in BIOMES:
        if biome.id == biome_id:
            return biome
        for variant in biome.variants:
            if variant.id == biome_id:
                return variant
    raise KeyError(biome_id)


class _FixedRng:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value

    def randint(self, low, high):
        return low
