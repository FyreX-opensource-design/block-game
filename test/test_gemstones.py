import unittest

import numpy as np

from server.blocks import BLOCKS, block_from_index
from server.items import ITEMS, creative_items
from server.loot import can_harvest, get_block_drops
from server.ores import load_ores
from server.recipes import craft_match, smelt_match
from server.world import World

GEMS = ("malachite", "emerald", "ruby", "sapphire", "diamond")
STONE_PREFIXES = ("", "granite_", "diorite_", "andesite_", "marble_", "limestone_")


class GemstoneTests(unittest.TestCase):
    def test_gemstone_ores_registered_and_mining_gates(self):
        bronze = ITEMS["bronze_pickaxe"]      # level 3
        iron = ITEMS["iron_pickaxe"]          # level 4
        malachite = ITEMS["malachite_pickaxe"]  # level 5
        emerald = ITEMS["emerald_pickaxe"]    # level 5
        ruby = ITEMS["ruby_pickaxe"]          # level 6
        sapphire = ITEMS["sapphire_pickaxe"]  # level 6
        diamond = ITEMS["diamond_pickaxe"]    # level 7

        expected_ore_level = {
            "malachite": 4,
            "emerald": 5,
            "ruby": 5,
            "sapphire": 5,
            "diamond": 6,
        }

        for gem in GEMS:
            self.assertIn(gem, ITEMS)
            for prefix in STONE_PREFIXES:
                ore_id = f"{prefix}{gem}_ore"
                self.assertIn(ore_id, BLOCKS, f"Missing ore block {ore_id}")
                self.assertIn(ore_id, ITEMS, f"Missing ore item {ore_id}")
                blk = BLOCKS[ore_id]
                self.assertEqual(blk.mining_level, expected_ore_level[gem])
                self.assertEqual(blk.tool, "pickaxe")
                self.assertEqual(len(blk.palette_face_uvs), 5)

                # Smelting & blasting fallback
                smelted = smelt_match(ore_id)
                self.assertIsNotNone(smelted)
                self.assertEqual(smelted[:2], (gem, 1))
                blasted = smelt_match(ore_id, recipe_types=("blasting",))
                self.assertIsNotNone(blasted)
                self.assertEqual(blasted[:2], (gem, 1))

        # Verify exact mining progression chain on stone + granite variants
        for prefix in ("", "granite_"):
            # Iron needed to mine malachite (bronze fails)
            self.assertFalse(can_harvest(BLOCKS[f"{prefix}malachite_ore"], bronze))
            self.assertEqual(get_block_drops(f"{prefix}malachite_ore", held_item=bronze), [])
            self.assertTrue(can_harvest(BLOCKS[f"{prefix}malachite_ore"], iron))
            self.assertEqual(get_block_drops(f"{prefix}malachite_ore", held_item=iron), [("malachite", 1)])

            # Malachite needed to mine ruby, sapphire, and emerald (iron fails)
            for target_gem in ("ruby", "sapphire", "emerald"):
                ore_id = f"{prefix}{target_gem}_ore"
                self.assertFalse(can_harvest(BLOCKS[ore_id], iron))
                self.assertEqual(get_block_drops(ore_id, held_item=iron), [])
                self.assertTrue(can_harvest(BLOCKS[ore_id], malachite))
                self.assertEqual(get_block_drops(ore_id, held_item=malachite), [(target_gem, 1)])

            # Either ruby or sapphire needed to mine diamond (malachite/emerald fail)
            d_ore = f"{prefix}diamond_ore"
            self.assertFalse(can_harvest(BLOCKS[d_ore], malachite))
            self.assertFalse(can_harvest(BLOCKS[d_ore], emerald))
            self.assertEqual(get_block_drops(d_ore, held_item=malachite), [])
            self.assertEqual(get_block_drops(d_ore, held_item=emerald), [])
            for pick in (ruby, sapphire, diamond):
                self.assertTrue(can_harvest(BLOCKS[d_ore], pick))
                self.assertEqual(get_block_drops(d_ore, held_item=pick), [("diamond", 1)])

    def test_gemstone_toolsets_stats_and_recipes(self):
        self.assertEqual(ITEMS["malachite_pickaxe"].mining_level, 5)
        self.assertEqual(ITEMS["emerald_pickaxe"].mining_level, 5)
        self.assertEqual(ITEMS["ruby_pickaxe"].mining_level, 6)
        self.assertEqual(ITEMS["sapphire_pickaxe"].mining_level, 6)
        self.assertEqual(ITEMS["diamond_pickaxe"].mining_level, 7)

        self.assertLess(ITEMS["iron_pickaxe"].durability, ITEMS["malachite_pickaxe"].durability)
        self.assertLess(ITEMS["malachite_pickaxe"].durability, ITEMS["ruby_pickaxe"].durability)
        self.assertLess(ITEMS["ruby_pickaxe"].durability, ITEMS["diamond_pickaxe"].durability)

        for gem in GEMS:
            for kind in ("pickaxe", "axe", "shovel", "sword"):
                tid = f"{gem}_{kind}"
                self.assertIn(tid, ITEMS, f"Missing tool {tid}")
                self.assertEqual(ITEMS[tid].tool, kind)
                self.assertEqual(ITEMS[tid].repair, gem)
                if kind == "sword":
                    self.assertEqual(ITEMS[tid].mining_level, 0)
                else:
                    self.assertEqual(ITEMS[tid].mining_level, ITEMS[f"{gem}_pickaxe"].mining_level)

            self.assertEqual(
                craft_match((gem, gem, gem, "", "stick", "", "", "stick", ""))[:2],
                (f"{gem}_pickaxe", 1),
            )
            self.assertEqual(
                craft_match((gem, gem, "", gem, "stick", "", "", "stick", ""))[:2],
                (f"{gem}_axe", 1),
            )
            self.assertEqual(
                craft_match((gem, gem, "", "stick", gem, "", "stick", "", ""))[:2],
                (f"{gem}_axe", 1),
            )
            self.assertEqual(
                craft_match(("", gem, "", "", "stick", "", "", "stick", ""))[:2],
                (f"{gem}_shovel", 1),
            )
            self.assertEqual(
                craft_match(("", gem, "", "", gem, "", "", "stick", ""))[:2],
                (f"{gem}_sword", 1),
            )

    def test_gemstone_worldgen_and_depth_bounds(self):
        ores = {o.id: o for o in load_ores()}
        for gem in GEMS:
            self.assertIn(gem, ores)
            self.assertTrue(ores[gem].enabled)
            self.assertIn("granite", ores[gem].target_blocks)
        self.assertLess(ores["diamond"].max_y, ores["malachite"].max_y)

        world = World(seed=12345, caves=[])
        spawned_y = {g: [] for g in GEMS}
        for cx in range(-2, 2):
            for cz in range(-2, 2):
                for cy in range(0, 4):
                    world.load_chunk(cx, cy, cz)
                    data = world.chunks[(cx, cy, cz)].data
                    for val in np.unique(data):
                        blk = block_from_index(int(val))
                        if not blk:
                            continue
                        for g in GEMS:
                            if f"{g}_ore" in blk.id:
                                ys = np.where(data == val)[0] + cy * 16
                                spawned_y[g].extend(int(y) for y in ys)

        for g in GEMS:
            self.assertTrue(spawned_y[g], f"{g}_ore did not generate in deep chunks")
            self.assertGreaterEqual(min(spawned_y[g]), ores[g].min_y)
            self.assertLessEqual(max(spawned_y[g]), ores[g].max_y)

    def test_gemstone_building_blocks_recipes_and_lamps(self):
        building = set(creative_items("building"))
        materials = set(creative_items("materials"))
        stone_pick = ITEMS["stone_pickaxe"]

        for gem in GEMS:
            block_id = f"{gem}_block"
            cut_id = f"cut_{gem}"
            bricks_id = f"{gem}_bricks"
            pillar_id = f"{gem}_pillar"
            chiseled_id = f"chiseled_{gem}"
            lamp_id = f"{gem}_lamp"

            expected = [
                block_id,
                cut_id,
                f"{cut_id}_slab",
                f"{cut_id}_vertical_slab",
                f"{cut_id}_stairs",
                f"{cut_id}_wall",
                bricks_id,
                f"{gem}_brick_slab",
                f"{gem}_brick_vertical_slab",
                f"{gem}_brick_stairs",
                f"{gem}_brick_wall",
                pillar_id,
                chiseled_id,
                lamp_id,
            ]
            for bid in expected:
                self.assertIn(bid, BLOCKS, f"Missing block {bid}")
                self.assertIn(bid, ITEMS, f"Missing item {bid}")
                self.assertIn(bid, building, f"Missing {bid} from Building tab")
                self.assertEqual(get_block_drops(bid, held_item=stone_pick), [(bid, 1)])

            self.assertIn(block_id, materials)
            self.assertIn(f"{pillar_id}_x", BLOCKS)
            self.assertIn(f"{pillar_id}_z", BLOCKS)
            self.assertGreater(sum(BLOCKS[lamp_id].light), 0)

            # Packing / unpacking and cut / brick / pillar / chiseled / lamp crafting
            self.assertEqual(craft_match((gem,) * 9)[:2], (block_id, 1))
            self.assertEqual(craft_match((block_id,))[:2], (gem, 9))
            self.assertEqual(craft_match((block_id, block_id, block_id, block_id))[:2], (cut_id, 4))
            self.assertEqual(craft_match((cut_id, cut_id, cut_id, cut_id))[:2], (bricks_id, 4))
            self.assertEqual(craft_match((cut_id, "", cut_id, ""))[:2], (pillar_id, 2))
            self.assertEqual(craft_match((f"{cut_id}_slab", "", f"{cut_id}_slab", ""))[:2], (chiseled_id, 1))
            self.assertEqual(
                craft_match(("", gem, "", gem, "torch", gem, "", gem, ""))[:2],
                (lamp_id, 2),
            )
            self.assertEqual(
                craft_match((bricks_id, bricks_id, bricks_id, "", "", "", "", "", ""))[:2],
                (f"{gem}_brick_slab", 6),
            )


if __name__ == "__main__":
    unittest.main()
