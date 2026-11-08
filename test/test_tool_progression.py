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
            "", "knapped_flint", "",
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

    def test_bronze_and_iron_tool_attributes(self):
        # Bronze Tools (Mining level 3)
        b_pick = ITEMS["bronze_pickaxe"]
        self.assertEqual(b_pick.tool, "pickaxe")
        self.assertEqual(b_pick.mining_level, 3)
        self.assertEqual(b_pick.mining_speed, 5.5)
        self.assertEqual(b_pick.attack, 4.0)
        self.assertEqual(b_pick.durability, 220)

        b_axe = ITEMS["bronze_axe"]
        self.assertEqual(b_axe.tool, "axe")
        self.assertEqual(b_axe.mining_level, 3)
        self.assertEqual(b_axe.mining_speed, 5.5)
        self.assertEqual(b_axe.attack, 6.0)

        b_shovel = ITEMS["bronze_shovel"]
        self.assertEqual(b_shovel.tool, "shovel")
        self.assertEqual(b_shovel.mining_level, 3)
        self.assertEqual(b_shovel.mining_speed, 5.5)
        self.assertEqual(b_shovel.attack, 4.0)

        b_sword = ITEMS["bronze_sword"]
        self.assertEqual(b_sword.tool, "sword")
        self.assertEqual(b_sword.mining_level, 0)
        self.assertEqual(b_sword.attack, 6.0)

        # Iron Tools (Mining level 4)
        i_pick = ITEMS["iron_pickaxe"]
        self.assertEqual(i_pick.tool, "pickaxe")
        self.assertEqual(i_pick.mining_level, 4)
        self.assertEqual(i_pick.mining_speed, 7.0)
        self.assertEqual(i_pick.attack, 5.0)
        self.assertEqual(i_pick.durability, 250)

        i_axe = ITEMS["iron_axe"]
        self.assertEqual(i_axe.tool, "axe")
        self.assertEqual(i_axe.mining_level, 4)
        self.assertEqual(i_axe.mining_speed, 7.0)
        self.assertEqual(i_axe.attack, 7.0)

        i_shovel = ITEMS["iron_shovel"]
        self.assertEqual(i_shovel.tool, "shovel")
        self.assertEqual(i_shovel.mining_level, 4)
        self.assertEqual(i_shovel.mining_speed, 7.0)
        self.assertEqual(i_shovel.attack, 5.0)

        i_sword = ITEMS["iron_sword"]
        self.assertEqual(i_sword.tool, "sword")
        self.assertEqual(i_sword.mining_level, 0)
        self.assertEqual(i_sword.attack, 7.0)

    def test_bronze_mixing_and_smelting(self):
        from server.recipes import smelt_match

        # 3 raw copper + 1 raw tin -> 4 raw bronze mix
        mix_grid = (
            "raw_copper", "raw_copper", "",
            "raw_copper", "raw_tin",    "",
            "",           "",           "",
        )
        res_mix = craft_match(mix_grid)
        self.assertIsNotNone(res_mix)
        self.assertEqual(res_mix[0], "raw_bronze_mix")
        self.assertEqual(res_mix[1], 4)

        # 3 raw copper bit + 1 raw tin bit -> 4 raw bronze mix bit
        bit_grid = (
            "raw_copper_bit", "raw_copper_bit", "",
            "raw_copper_bit", "raw_tin_bit",    "",
            "",               "",               "",
        )
        res_bit = craft_match(bit_grid)
        self.assertIsNotNone(res_bit)
        self.assertEqual(res_bit[0], "raw_bronze_mix_bit")
        self.assertEqual(res_bit[1], 4)

        # Smelt raw bronze mix -> bronze ingot
        smelt_ingot = smelt_match("raw_bronze_mix")
        self.assertIsNotNone(smelt_ingot)
        self.assertEqual(smelt_ingot[0], "bronze_ingot")

        # Smelt raw bronze mix bit -> bronze nugget
        smelt_nugget = smelt_match("raw_bronze_mix_bit")
        self.assertIsNotNone(smelt_nugget)
        self.assertEqual(smelt_nugget[0], "bronze_nugget")

    def test_mining_progression_iron_ore(self):
        flint_pick = ITEMS["flint_pickaxe"]
        stone_pick = ITEMS["stone_pickaxe"]
        bronze_pick = ITEMS["bronze_pickaxe"]
        iron_pick = ITEMS["iron_pickaxe"]

        iron_block = BLOCKS["iron_ore"]
        stone_block = BLOCKS["stone"]

        # Flint pickaxe can harvest stone
        self.assertTrue(can_harvest(stone_block, flint_pick))
        drops = dict(get_block_drops("stone", held_item=flint_pick))
        self.assertIn("cobble", drops)

        # Flint pickaxe CANNOT harvest iron ore (mining_level 1 < 3)
        self.assertFalse(can_harvest(iron_block, flint_pick))
        flint_iron_drops = get_block_drops("iron_ore", held_item=flint_pick)
        self.assertEqual(flint_iron_drops, [])

        # Stone pickaxe CANNOT harvest iron ore (mining_level 2 < 3)
        self.assertFalse(can_harvest(iron_block, stone_pick))

        # Bronze pickaxe CAN harvest iron ore (mining_level 3 >= 3)
        self.assertTrue(can_harvest(iron_block, bronze_pick))
        bronze_iron_drops = dict(get_block_drops("iron_ore", held_item=bronze_pick, rng=random.Random(0)))
        self.assertIn("raw_iron", bronze_iron_drops)

        # Iron pickaxe CAN harvest iron ore (mining_level 4 >= 3)
        self.assertTrue(can_harvest(iron_block, iron_pick))

        # Breaking time speeds: flint < stone < bronze < iron
        time_flint = calculate_break_time(iron_block, flint_pick)
        time_stone = calculate_break_time(iron_block, stone_pick)
        time_bronze = calculate_break_time(iron_block, bronze_pick)
        time_iron = calculate_break_time(iron_block, iron_pick)
        self.assertLess(time_stone, time_flint)
        self.assertLess(time_bronze, time_stone)
        self.assertLess(time_iron, time_bronze)


if __name__ == "__main__":
    unittest.main()

