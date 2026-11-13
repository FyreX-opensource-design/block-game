import unittest

from server.biomes import BIOMES, all_biomes, biome_at, can_biomes_touch
from server.blocks import BLOCKS
from server.recipes import FUELS, craft_match, smelt_match
from server.tags import has_block_tag, has_item_tag
from server.trees import DECORATORS
from server.world import World


class BirchTests(unittest.TestCase):
    def test_birch_blocks_tags_and_crafting(self):
        for block_id in (
            "birch_log",
            "birch_wood",
            "birch_planks",
            "birch_planks_slab",
            "birch_planks_stairs",
            "birch_planks_vertical_slab",
            "birch_fence",
            "birch_fence_gate",
            "birch_door",
            "birch_trapdoor",
            "birch_leaves",
            "birch_sapling",
        ):
            self.assertIn(block_id, BLOCKS)

        self.assertIn("birch_log_x", BLOCKS)
        self.assertIn("birch_log_z", BLOCKS)
        self.assertIn("birch_wood_x", BLOCKS)
        self.assertIn("birch_wood_z", BLOCKS)
        self.assertTrue(BLOCKS["birch_leaves"].properties.get("decayable"))
        self.assertIn("birch_leaves_persistent", BLOCKS)

        self.assertTrue(has_block_tag("birch_log", "logs"))
        self.assertTrue(has_block_tag("birch_wood", "logs"))
        self.assertTrue(has_block_tag("birch_leaves", "leaves"))
        self.assertTrue(has_block_tag("birch_planks", "planks"))
        self.assertTrue(has_item_tag("birch_log", "logs"))
        self.assertTrue(has_item_tag("birch_wood", "logs"))
        self.assertTrue(has_item_tag("birch_leaves", "leaves"))
        self.assertTrue(has_item_tag("birch_planks", "planks"))

        # Bark faces all use the side texture, like the other wood blocks.
        uvs = BLOCKS["birch_wood"].face_uvs
        for index in range(1, 6):
            self.assertEqual(uvs[index], uvs[0])

        planks = craft_match(["birch_log", "", "", ""])
        self.assertEqual(planks[:2], ("birch_planks", 4))
        wood = craft_match(["birch_log", "birch_log", "birch_log", "birch_log"])
        self.assertEqual(wood[:2], ("birch_wood", 3))
        sticks = craft_match(["birch_planks", "", "birch_planks", ""])
        self.assertEqual(sticks[:2], ("stick", 4))

        smelt = smelt_match("birch_log")
        self.assertEqual(smelt[0], "charcoal")
        self.assertEqual(FUELS.get("birch_log"), 15.0)
        self.assertEqual(FUELS.get("birch_wood"), 15.0)
        self.assertEqual(FUELS.get("birch_planks"), 15.0)
        self.assertEqual(FUELS.get("birch_planks_slab"), 7.5)

    def test_birch_trees_and_biomes(self):
        trees = {tree.id: tree for tree in DECORATORS}
        self.assertEqual(trees["birch"].shape, "birch")
        self.assertEqual(trees["birch"].trunk, "birch_log")
        self.assertEqual(trees["birch"].leaves, "birch_leaves")
        self.assertEqual(trees["tall_birch"].trunk, "birch_log")
        self.assertGreater(trees["tall_birch"].trunk_min, trees["birch"].trunk_min)

        catalog = {biome.id: biome for biome in all_biomes(BIOMES)}
        forest = catalog["birch_forest"]
        hills = catalog["birch_forest_hills"]
        self.assertEqual(forest.variant_of, "forest")
        self.assertEqual(hills.variant_of, "pine_forest")
        self.assertEqual(forest.name, "Birch Forest")
        self.assertEqual(hills.name, "Birch Forest Hills")
        self.assertEqual(forest.temperature, (0.0, 0.5))
        self.assertEqual(hills.temperature, (0.0, 0.5))
        self.assertEqual([tree.id for tree in forest.trees], ["birch", "tall_birch"])
        self.assertEqual([tree.id for tree in hills.trees], ["tall_birch", "birch"])
        self.assertGreater(forest.tree_chance, catalog["forest"].tree_chance)
        self.assertGreater(hills.height_offset, forest.height_offset)

        plains = catalog["plains"]
        highland = catalog["highland_plains"]
        desert = catalog["desert"]
        self.assertTrue(can_biomes_touch(forest, plains))
        self.assertTrue(can_biomes_touch(forest, highland))
        self.assertTrue(can_biomes_touch(forest, hills))
        self.assertFalse(can_biomes_touch(forest, desert))
        self.assertFalse(can_biomes_touch(hills, desert))

    def test_birch_forests_generate_birch_trees(self):
        seed = 3
        spots = {"birch_forest": None, "birch_forest_hills": None}
        for z in range(-600, 601, 8):
            for x in range(-600, 601, 8):
                biome_id = biome_at(x, z, seed, BIOMES).id
                if biome_id in spots and spots[biome_id] is None:
                    spots[biome_id] = (x, z)
            if all(spots.values()):
                break
        self.assertIsNotNone(spots["birch_forest"], "cool forests should include birch forest")
        self.assertIsNotNone(spots["birch_forest_hills"], "cool high forests should include birch forest hills")

        world = World(seed=seed, caves=[])
        for biome_id, (x, z) in spots.items():
            cx, cz = x // 16, z // 16
            world.generate_column(cx, cz)
            found = False
            for lz in range(16):
                for lx in range(16):
                    wx, wz = cx * 16 + lx, cz * 16 + lz
                    for y in range(40, 140):
                        if world.get_block(wx, y, wz) == "birch_log":
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            self.assertTrue(found, f"{biome_id} should grow birch trees")


if __name__ == "__main__":
    unittest.main()
