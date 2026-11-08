import unittest
from server.blocks import BLOCKS, can_place_block
from server.items import ITEMS, creative_items
from server.loot import get_block_drops
from server.recipes import craft_match, smelt_match
from server.world import World


class TestMetalBlocks(unittest.TestCase):
    def setUp(self):
        self.raw_blocks = ["raw_copper_block", "raw_tin_block", "raw_iron_block"]
        self.metal_blocks = ["copper_block", "tin_block", "iron_block", "bronze_block"]
        self.cut_blocks = ["cut_copper", "cut_tin", "cut_iron", "cut_bronze"]
        self.grates = ["copper_grate", "tin_grate", "iron_grate", "bronze_grate"]
        self.riveted = ["riveted_copper", "riveted_tin", "riveted_iron", "riveted_bronze"]
        self.all_blocks = (
            self.raw_blocks + self.metal_blocks + self.cut_blocks + self.grates + self.riveted
        )

    def test_blocks_and_items_registered(self):
        for b_id in self.all_blocks:
            self.assertIn(b_id, BLOCKS, f"{b_id} should be registered in BLOCKS")
            self.assertIn(b_id, ITEMS, f"{b_id} should be registered in ITEMS")
            b_obj = BLOCKS[b_id]
            self.assertTrue(b_obj.solid)
            self.assertEqual(ITEMS[b_id].block, b_id)

        # Check grates have cutout: True
        for g_id in self.grates:
            self.assertTrue(BLOCKS[g_id].cutout, f"{g_id} must have cutout True")

    def test_creative_inventory_presence(self):
        building_items = creative_items("building")
        for b_id in self.all_blocks:
            self.assertIn(b_id, building_items, f"{b_id} should appear in Building tab")

        materials_items = creative_items("materials")
        for b_id in self.raw_blocks + self.metal_blocks:
            self.assertIn(b_id, materials_items, f"{b_id} should appear in Materials tab")

    def test_loot_drops_require_pickaxe(self):
        flint_pick = ITEMS["flint_pickaxe"]
        stone_pick = ITEMS["stone_pickaxe"]

        # Level 1 blocks: copper and tin (mineable with flint pickaxe)
        for b_id in [
            "raw_copper_block", "raw_tin_block",
            "copper_block", "tin_block",
            "cut_copper", "cut_tin",
            "copper_grate", "tin_grate",
            "riveted_copper", "riveted_tin",
        ]:
            # No tool drops nothing
            self.assertEqual(get_block_drops(b_id, held_item=None), [])
            # Flint pickaxe drops block
            drops = get_block_drops(b_id, held_item=flint_pick)
            self.assertEqual(dict(drops), {b_id: 1})

        # Level 2 blocks: iron and bronze (require stone pickaxe)
        for b_id in [
            "raw_iron_block",
            "iron_block", "bronze_block",
            "cut_iron", "cut_bronze",
            "riveted_iron", "riveted_bronze",
        ]:
            self.assertEqual(get_block_drops(b_id, held_item=None), [])
            # Flint pickaxe (level 1) cannot harvest
            self.assertEqual(get_block_drops(b_id, held_item=flint_pick), [])
            # Stone pickaxe (level 2) harvests
            drops = get_block_drops(b_id, held_item=stone_pick)
            self.assertEqual(dict(drops), {b_id: 1})

    def test_raw_ore_block_crafting_and_smelting(self):
        for raw_item, raw_block, metal_block in [
            ("raw_copper", "raw_copper_block", "copper_block"),
            ("raw_tin", "raw_tin_block", "tin_block"),
            ("raw_iron", "raw_iron_block", "iron_block"),
        ]:
            # 3x3 packing
            grid = (raw_item,) * 9
            packed = craft_match(grid)
            self.assertIsNotNone(packed)
            self.assertEqual(packed[0], raw_block)
            self.assertEqual(packed[1], 1)

            # Unpacking
            unpacked = craft_match((raw_block,))
            self.assertIsNotNone(unpacked)
            self.assertEqual(unpacked[0], raw_item)
            self.assertEqual(unpacked[1], 9)

            # Smelting raw block -> metal block
            smelted = smelt_match(raw_block)
            self.assertIsNotNone(smelted)
            self.assertEqual(smelted[0], metal_block)
            self.assertEqual(smelted[1], 1)

    def test_metal_block_packing_and_unpacking(self):
        for ingot, block in [
            ("copper_ingot", "copper_block"),
            ("tin_ingot", "tin_block"),
            ("iron_ingot", "iron_block"),
            ("bronze_ingot", "bronze_block"),
        ]:
            # 3x3 packing
            grid = (ingot,) * 9
            packed = craft_match(grid)
            self.assertIsNotNone(packed)
            self.assertEqual(packed[0], block)
            self.assertEqual(packed[1], 1)

            # Unpacking
            unpacked = craft_match((block,))
            self.assertIsNotNone(unpacked)
            self.assertEqual(unpacked[0], ingot)
            self.assertEqual(unpacked[1], 9)

    def test_cut_metal_crafting(self):
        for block, cut_block in [
            ("copper_block", "cut_copper"),
            ("tin_block", "cut_tin"),
            ("iron_block", "cut_iron"),
            ("bronze_block", "cut_bronze"),
        ]:
            res = craft_match((block, block, block, block))
            self.assertIsNotNone(res)
            self.assertEqual(res[0], cut_block)
            self.assertEqual(res[1], 4)

    def test_metal_grate_crafting(self):
        for ingot, grate in [
            ("copper_ingot", "copper_grate"),
            ("tin_ingot", "tin_grate"),
            ("iron_ingot", "iron_grate"),
            ("bronze_ingot", "bronze_grate"),
        ]:
            res = craft_match((ingot, ingot, ingot, ingot))
            self.assertIsNotNone(res)
            self.assertEqual(res[0], grate)
            self.assertEqual(res[1], 4)

    def test_riveted_metal_crafting(self):
        for ingot, nugget, riveted in [
            ("copper_ingot", "copper_nugget", "riveted_copper"),
            ("tin_ingot", "tin_nugget", "riveted_tin"),
            ("iron_ingot", "iron_nugget", "riveted_iron"),
            ("bronze_ingot", "bronze_nugget", "riveted_bronze"),
        ]:
            # Standard diagonal
            res1 = craft_match((ingot, nugget, nugget, ingot))
            self.assertIsNotNone(res1)
            self.assertEqual(res1[0], riveted)
            self.assertEqual(res1[1], 2)

            # Mirrored diagonal
            res2 = craft_match((nugget, ingot, ingot, nugget))
            self.assertIsNotNone(res2)
            self.assertEqual(res2[0], riveted)
            self.assertEqual(res2[1], 2)

    def test_metal_grate_lighting_and_opacity(self):
        from server.lighting import get_block_opacity
        from server.world import Chunk, World
        from server.mesh import build_chunk_mesh
        from server.blocks import block_index

        for grate in ("copper_grate", "tin_grate", "iron_grate", "bronze_grate"):
            b = BLOCKS[grate]
            self.assertEqual(get_block_opacity(b), 0, f"{grate} must have light opacity 0")

        # Test skylight passes through grates and does not cast black shadow inside
        world = World()
        chunk = Chunk(0, 0, 0)
        world.chunks[(0, 0, 0)] = chunk
        chunk.data[9, 8, 8] = block_index("stone")
        chunk.data[10, 8, 8] = block_index("copper_grate")
        world.lighting.compute_chunk_lighting(0, 0, 0)

        # Light inside grate cell
        self.assertEqual(world.get_skylight(8, 10, 8), 15)

        # Mesh stone top face under grate is lit
        mesh = build_chunk_mesh(world, 0, 0, 0)
        found_stone_top = False
        for i in range(0, len(mesh.vertices), 4):
            normal = list(mesh.vertices[i, 3:6])
            if normal == [0.0, 1.0, 0.0]:
                self.assertAlmostEqual(mesh.lights[i, 0], 1.0, places=3)
                found_stone_top = True
                break
        self.assertTrue(found_stone_top)

    def test_placement_in_world(self):
        world = World(seed=1)
        world.generate_column(0, 0)
        for i, b_id in enumerate(self.all_blocks):
            y = 70 + i
            self.assertTrue(can_place_block(world, 5, y, 5, b_id))
            world.set_block(5, y, 5, b_id)
            self.assertEqual(world.get_block(5, y, 5), b_id)


if __name__ == "__main__":
    unittest.main()
