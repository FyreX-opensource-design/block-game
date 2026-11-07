import unittest
import random
from server.items import ITEMS
from server.blocks import BLOCKS
from server.loot import get_block_drops, can_harvest, calculate_break_time
from server.recipes import craft_match


class TestToolProgression(unittest.TestCase):
    def test_plant_fiber_from_tall_grass(self):
        drops = get_block_drops("tall_grass", held_item=None)
        items = dict(drops)
        self.assertIn("plant_fiber", items)
        self.assertGreaterEqual(items["plant_fiber"], 1)

    def test_flint_tool_attributes(self):
        flint_pick = ITEMS["flint_pickaxe"]
        self.assertEqual(flint_pick.tool, "pickaxe")
        self.assertEqual(flint_pick.durability, 50)
        self.assertEqual(flint_pick.mining_level, 1)
        self.assertEqual(flint_pick.mining_speed, 2.0)
        self.assertEqual(flint_pick.attack, 2.0)

        flint_axe = ITEMS["flint_axe"]
        self.assertEqual(flint_axe.tool, "axe")
        self.assertEqual(flint_axe.durability, 50)
        self.assertEqual(flint_axe.mining_level, 1)
        self.assertEqual(flint_axe.mining_speed, 2.0)
        self.assertEqual(flint_axe.attack, 3.0)

        flint_shovel = ITEMS["flint_shovel"]
        self.assertEqual(flint_shovel.tool, "shovel")
        self.assertEqual(flint_shovel.durability, 50)
        self.assertEqual(flint_shovel.mining_level, 1)
        self.assertEqual(flint_shovel.mining_speed, 2.0)
        self.assertEqual(flint_shovel.attack, 2.0)

        flint_sword = ITEMS["flint_sword"]
        self.assertEqual(flint_sword.tool, "sword")
        self.assertEqual(flint_sword.durability, 50)
        self.assertEqual(flint_sword.mining_level, 0)
        self.assertEqual(flint_sword.attack, 4.0)

    def test_stone_tool_attributes(self):
        stone_pick = ITEMS["stone_pickaxe"]
        self.assertEqual(stone_pick.tool, "pickaxe")
        self.assertEqual(stone_pick.durability, 131)
        self.assertEqual(stone_pick.mining_level, 2)
        self.assertEqual(stone_pick.mining_speed, 4.0)
        self.assertEqual(stone_pick.attack, 3.0)

        stone_axe = ITEMS["stone_axe"]
        self.assertEqual(stone_axe.tool, "axe")
        self.assertEqual(stone_axe.durability, 131)
        self.assertEqual(stone_axe.mining_level, 2)
        self.assertEqual(stone_axe.mining_speed, 4.0)
        self.assertEqual(stone_axe.attack, 5.0)

        stone_shovel = ITEMS["stone_shovel"]
        self.assertEqual(stone_shovel.tool, "shovel")
        self.assertEqual(stone_shovel.durability, 131)
        self.assertEqual(stone_shovel.mining_level, 2)
        self.assertEqual(stone_shovel.mining_speed, 4.0)
        self.assertEqual(stone_shovel.attack, 3.0)

        stone_sword = ITEMS["stone_sword"]
        self.assertEqual(stone_sword.tool, "sword")
        self.assertEqual(stone_sword.durability, 131)
        self.assertEqual(stone_sword.mining_level, 0)
        self.assertEqual(stone_sword.attack, 5.0)

    def test_flint_tool_crafting(self):
        # Flint pickaxe
        pick_grid = (
            "knapped_flint", "plant_fiber", "knapped_flint",
            "",              "stick",       "",
            "",              "stick",       "",
        )
        res = craft_match(pick_grid)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "flint_pickaxe")

        # Flint axe
        axe_grid = (
            "knapped_flint", "knapped_flint", "",
            "knapped_flint", "plant_fiber",   "",
            "",              "stick",         "",
        )
        res = craft_match(axe_grid)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "flint_axe")

        # Flint shovel
        shovel_grid = (
            "", "knapped_flint", "",
            "", "plant_fiber",   "",
            "", "stick",         "",
        )
        res = craft_match(shovel_grid)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "flint_shovel")

        # Flint sword
        sword_grid = (
            "", "knapped_flint", "",
            "", "plant_fiber",   "",
            "", "stick",         "",
        )
        res = craft_match(sword_grid)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "flint_sword")

    def test_stone_tool_crafting_requires_smelted_stone(self):
        # Stone pickaxe with smelted stone succeeds
        pick_grid = (
            "stone", "stone", "stone",
            "",      "stick", "",
            "",      "stick", "",
        )
        res = craft_match(pick_grid)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "stone_pickaxe")

        # Stone pickaxe with cobblestone fails
        cobble_pick_grid = (
            "cobble", "cobble", "cobble",
            "",       "stick",  "",
            "",       "stick",  "",
        )
        self.assertIsNone(craft_match(cobble_pick_grid))

        # Stone axe with smelted stone succeeds
        axe_grid = (
            "stone", "stone", "",
            "stone", "stick", "",
            "",      "stick", "",
        )
        res = craft_match(axe_grid)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "stone_axe")

        # Stone shovel with smelted stone succeeds
        shovel_grid = (
            "", "stone", "",
            "", "stick", "",
            "", "stick", "",
        )
        res = craft_match(shovel_grid)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "stone_shovel")

        # Stone sword with smelted stone succeeds
        sword_grid = (
            "", "stone", "",
            "", "stone", "",
            "", "stick", "",
        )
        res = craft_match(sword_grid)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "stone_sword")

    def test_mining_progression_iron_ore(self):
        flint_pick = ITEMS["flint_pickaxe"]
        stone_pick = ITEMS["stone_pickaxe"]

        iron_block = BLOCKS["iron_ore"]
        stone_block = BLOCKS["stone"]

        # Flint pickaxe can harvest stone
        self.assertTrue(can_harvest(stone_block, flint_pick))
        drops = dict(get_block_drops("stone", held_item=flint_pick))
        self.assertIn("cobble", drops)

        # Flint pickaxe CANNOT harvest iron ore
        self.assertFalse(can_harvest(iron_block, flint_pick))
        flint_iron_drops = get_block_drops("iron_ore", held_item=flint_pick)
        self.assertEqual(flint_iron_drops, [])

        # Stone pickaxe CAN harvest iron ore
        self.assertTrue(can_harvest(iron_block, stone_pick))
        stone_iron_drops = dict(get_block_drops("iron_ore", held_item=stone_pick, rng=random.Random(0)))
        self.assertIn("raw_iron", stone_iron_drops)

        # Breaking time is faster with stone pickaxe (speed 4 vs speed 2)
        time_flint_stone = calculate_break_time(stone_block, flint_pick)
        time_stone_stone = calculate_break_time(stone_block, stone_pick)
        self.assertLess(time_stone_stone, time_flint_stone)


if __name__ == "__main__":
    unittest.main()

