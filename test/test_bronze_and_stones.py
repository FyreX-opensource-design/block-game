import unittest
import numpy as np
from server.blocks import BLOCKS, block_from_index
from server.items import ITEMS
from server.loot import can_harvest, get_block_drops, calculate_break_time
from server.recipes import craft_match, smelt_match
from server.ores import load_ores
from server.world import World


class TestBronzeAndStones(unittest.TestCase):
    def test_bronze_and_iron_items_registered(self):
        expected_items = [
            "raw_bronze_mix",
            "raw_bronze_mix_bit",
            "bronze_ingot",
            "bronze_nugget",
            "bronze_pickaxe",
            "bronze_axe",
            "bronze_shovel",
            "bronze_sword",
            "iron_pickaxe",
            "iron_axe",
            "iron_shovel",
            "iron_sword",
        ]
        for item_id in expected_items:
            self.assertIn(item_id, ITEMS, f"{item_id} should be registered in ITEMS")

    def test_tool_mining_levels_and_progression(self):
        stone_pick = ITEMS["stone_pickaxe"]
        bronze_pick = ITEMS["bronze_pickaxe"]
        iron_pick = ITEMS["iron_pickaxe"]

        # Stone is level 2
        self.assertEqual(stone_pick.mining_level, 2)
        # Bronze is a level above stone (level 3)
        self.assertEqual(bronze_pick.mining_level, 3)
        # Iron is a level above bronze (level 4)
        self.assertEqual(iron_pick.mining_level, 4)

        # Durability progression
        self.assertLess(stone_pick.durability, bronze_pick.durability)
        self.assertLess(bronze_pick.durability, iron_pick.durability)

        # Mining speed progression
        self.assertLess(stone_pick.mining_speed, bronze_pick.mining_speed)
        self.assertLess(bronze_pick.mining_speed, iron_pick.mining_speed)

        # Attack damage progression
        self.assertLess(stone_pick.attack, bronze_pick.attack)
        self.assertLess(bronze_pick.attack, iron_pick.attack)

    def test_raw_bronze_mix_recipes(self):
        # 3 raw copper + 1 raw tin -> 4 raw bronze mix (shapeless)
        res_mix = craft_match(("raw_copper", "raw_copper", "raw_copper", "raw_tin"))
        self.assertIsNotNone(res_mix)
        self.assertEqual(res_mix[0], "raw_bronze_mix")
        self.assertEqual(res_mix[1], 4)

        # 3 raw copper bit + 1 raw tin bit -> 4 raw bronze mix bit (shapeless)
        res_bit = craft_match(("raw_copper_bit", "raw_copper_bit", "raw_copper_bit", "raw_tin_bit"))
        self.assertIsNotNone(res_bit)
        self.assertEqual(res_bit[0], "raw_bronze_mix_bit")
        self.assertEqual(res_bit[1], 4)

        # 9 bits -> 1 mix
        res_from_bits = craft_match(("raw_bronze_mix_bit",) * 9)
        self.assertIsNotNone(res_from_bits)
        self.assertEqual(res_from_bits[0], "raw_bronze_mix")
        self.assertEqual(res_from_bits[1], 1)

        # 1 mix -> 9 bits
        res_to_bits = craft_match(("raw_bronze_mix",))
        self.assertIsNotNone(res_to_bits)
        self.assertEqual(res_to_bits[0], "raw_bronze_mix_bit")
        self.assertEqual(res_to_bits[1], 9)

    def test_bronze_smelting_and_nuggets(self):
        # Smelt raw mix -> bronze ingot
        smelt_mix = smelt_match("raw_bronze_mix")
        self.assertIsNotNone(smelt_mix)
        self.assertEqual(smelt_mix[0], "bronze_ingot")

        # Smelt raw mix bit -> bronze nugget
        smelt_bit = smelt_match("raw_bronze_mix_bit")
        self.assertIsNotNone(smelt_bit)
        self.assertEqual(smelt_bit[0], "bronze_nugget")

        # 9 nuggets -> 1 ingot
        nuggets_to_ingot = craft_match(("bronze_nugget",) * 9)
        self.assertIsNotNone(nuggets_to_ingot)
        self.assertEqual(nuggets_to_ingot[0], "bronze_ingot")

        # 1 ingot -> 9 nuggets
        ingot_to_nuggets = craft_match(("bronze_ingot",))
        self.assertIsNotNone(ingot_to_nuggets)
        self.assertEqual(ingot_to_nuggets[0], "bronze_nugget")
        self.assertEqual(ingot_to_nuggets[1], 9)

    def test_tool_crafting_recipes(self):
        # Bronze tools
        bp = craft_match(("bronze_ingot", "bronze_ingot", "bronze_ingot", "", "stick", "", "", "stick", ""))
        self.assertIsNotNone(bp)
        self.assertEqual(bp[0], "bronze_pickaxe")

        ba = craft_match(("bronze_ingot", "bronze_ingot", "", "bronze_ingot", "stick", "", "", "stick", ""))
        self.assertIsNotNone(ba)
        self.assertEqual(ba[0], "bronze_axe")

        bs = craft_match(("", "bronze_ingot", "", "", "stick", "", "", "stick", ""))
        self.assertIsNotNone(bs)
        self.assertEqual(bs[0], "bronze_shovel")

        bsw = craft_match(("", "bronze_ingot", "", "", "bronze_ingot", "", "", "stick", ""))
        self.assertIsNotNone(bsw)
        self.assertEqual(bsw[0], "bronze_sword")

        # Iron tools
        ip = craft_match(("iron_ingot", "iron_ingot", "iron_ingot", "", "stick", "", "", "stick", ""))
        self.assertIsNotNone(ip)
        self.assertEqual(ip[0], "iron_pickaxe")

        ia = craft_match(("iron_ingot", "iron_ingot", "", "iron_ingot", "stick", "", "", "stick", ""))
        self.assertIsNotNone(ia)
        self.assertEqual(ia[0], "iron_axe")

        is_ = craft_match(("", "iron_ingot", "", "", "stick", "", "", "stick", ""))
        self.assertIsNotNone(is_)
        self.assertEqual(is_[0], "iron_shovel")

        isw = craft_match(("", "iron_ingot", "", "", "iron_ingot", "", "", "stick", ""))
        self.assertIsNotNone(isw)
        self.assertEqual(isw[0], "iron_sword")

    def test_stone_types_registered_and_loot(self):
        stone_types = ["granite", "diorite", "andesite", "marble", "limestone"]
        for st in stone_types:
            for variant in [st, f"smooth_{st}", f"gravelly_{st}"]:
                self.assertIn(variant, BLOCKS, f"{variant} should be in BLOCKS")
                self.assertIn(variant, ITEMS, f"{variant} should be in ITEMS")
                drops = get_block_drops(variant)
                self.assertEqual(drops, [(variant, 1)])

            # Check properties
            base_b = BLOCKS[st]
            self.assertEqual(base_b.tool, "pickaxe")
            self.assertEqual(base_b.mining_level, 1)

            smooth_b = BLOCKS[f"smooth_{st}"]
            self.assertEqual(smooth_b.tool, "pickaxe")
            self.assertEqual(smooth_b.mining_level, 1)

            gravel_b = BLOCKS[f"gravelly_{st}"]
            self.assertEqual(gravel_b.tool, "shovel")
            self.assertEqual(gravel_b.mining_level, 0)
            self.assertTrue(gravel_b.gravity)

    def test_stone_crafting_and_smelting(self):
        for st in ["granite", "diorite", "andesite", "marble", "limestone"]:
            sm_id = f"smooth_{st}"
            # 4 stone in 2x2 -> 4 smooth stone
            res_sm = craft_match((st, st, "", st, st, "", "", "", ""))
            self.assertIsNotNone(res_sm)
            self.assertEqual(res_sm[0], sm_id)
            self.assertEqual(res_sm[1], 4)

            # smelt stone -> smooth stone
            res_smelt = smelt_match(st)
            self.assertIsNotNone(res_smelt)
            self.assertEqual(res_smelt[0], sm_id)

            # stone + gravel -> 2 gravelly stone
            res_gr = craft_match((st, "gravel"))
            self.assertIsNotNone(res_gr)
            self.assertEqual(res_gr[0], f"gravelly_{st}")
            self.assertEqual(res_gr[1], 2)

    def test_massive_stone_patches_generation(self):
        ores = {ore.id: ore for ore in load_ores()}
        for st in ["granite", "diorite", "andesite", "marble", "limestone"]:
            # Massive patch ore exists
            self.assertIn(f"01_{st}" if st == "granite" else f"02_{st}" if st == "diorite" else f"03_{st}" if st == "andesite" else f"04_{st}" if st == "marble" else f"05_{st}", ores)

        # Generate world chunks and assert presence of stones
        world = World(seed=98765)
        found = set()
        for cx in range(-2, 3):
            for cz in range(-2, 3):
                for cy in range(2, 6):
                    world.load_chunk(cx, cy, cz)
                    chunk = world.chunks[(cx, cy, cz)]
                    for val in np.unique(chunk.data):
                        b = block_from_index(int(val))
                        if b and any(s in b.id for s in ["granite", "diorite", "andesite", "marble", "limestone"]):
                            found.add(b.id)

        self.assertGreaterEqual(len(found), 5, f"Expected multiple stone types in worldgen, got: {found}")


if __name__ == "__main__":
    unittest.main()

