import unittest

from server.biomes import BIOMES, all_biomes, biome_at, can_biomes_touch
from server.blocks import BLOCKS, is_soil, is_tall_plant
from server.recipes import FUELS, craft_match, smelt_match
from server.tags import has_block_tag, has_item_tag
from server.trees import DECORATORS
from server.world import World, tree_at, tree_blocks


class SpruceAndRedwoodTests(unittest.TestCase):
    def test_wood_sets_craft_and_burn(self):
        for wood in ("spruce", "redwood"):
            for block_id in (
                f"{wood}_log",
                f"{wood}_wood",
                f"{wood}_planks",
                f"{wood}_planks_slab",
                f"{wood}_planks_stairs",
                f"{wood}_planks_vertical_slab",
                f"{wood}_fence",
                f"{wood}_fence_gate",
                f"{wood}_door",
                f"{wood}_trapdoor",
                f"{wood}_leaves",
                f"{wood}_sapling",
            ):
                self.assertIn(block_id, BLOCKS)
            self.assertIn(f"{wood}_log_x", BLOCKS)
            self.assertIn(f"{wood}_leaves_persistent", BLOCKS)
            self.assertTrue(has_block_tag(f"{wood}_log", "logs"))
            self.assertTrue(has_block_tag(f"{wood}_planks", "planks"))
            self.assertTrue(has_block_tag(f"{wood}_leaves", "leaves"))
            self.assertTrue(has_item_tag(f"{wood}_log", "logs"))
            self.assertEqual(craft_match([f"{wood}_log", "", "", ""])[:2], (f"{wood}_planks", 4))
            self.assertEqual(craft_match([f"{wood}_log", f"{wood}_log", f"{wood}_log", f"{wood}_log"])[:2], (f"{wood}_wood", 3))
            self.assertEqual(smelt_match(f"{wood}_log")[0], "charcoal")
            self.assertEqual(FUELS.get(f"{wood}_log"), 15.0)
            self.assertEqual(FUELS.get(f"{wood}_planks_slab"), 7.5)

    def test_tree_shapes(self):
        trees = {tree.id: tree for tree in DECORATORS}
        self.assertEqual(trees["spruce"].shape, "spruce")
        self.assertEqual(trees["tall_spruce"].trunk, "spruce_log")
        self.assertGreater(trees["tall_spruce"].trunk_min, trees["spruce"].trunk_min)
        self.assertEqual(trees["redwood"].shape, "redwood")

        spruce = tree_blocks(trees["spruce"], 8, 80, 8, seed=1)
        trunks = {(x, z) for x, y, z, block, replace in spruce if block == "spruce_log"}
        self.assertEqual(trunks, {(8, 8)})

        redwood = tree_blocks(trees["redwood"], 8, 80, 8, seed=1)
        trunks = {(x, z) for x, y, z, block, replace in redwood if block == "redwood_log"}
        self.assertEqual(trunks, {(8, 8), (9, 8), (8, 9), (9, 9)})
        self.assertTrue(any(block == "redwood_leaves" for _x, _y, _z, block, _replace in redwood))

    def test_biomes_podzol_and_double_plants(self):
        catalog = {biome.id: biome for biome in all_biomes(BIOMES)}
        spruce = catalog["spruce_forest"]
        redwood = catalog["redwood_forest"]
        self.assertEqual(spruce.variant_of, "forest")
        self.assertEqual(redwood.variant_of, "forest")
        self.assertEqual(spruce.surface, "grass")
        self.assertEqual(redwood.surface, "podzol")
        self.assertEqual([tree.id for tree in spruce.trees], ["spruce", "tall_spruce"])
        self.assertEqual([tree.id for tree in redwood.trees], ["redwood"])
        for biome in (spruce, redwood):
            names = [dec.id for dec in biome.decorations]
            self.assertIn("double_tall_grass", names)
            self.assertIn("double_fern", names)
        self.assertTrue(can_biomes_touch(spruce, catalog["birch_forest"]))
        self.assertTrue(can_biomes_touch(redwood, spruce))
        self.assertFalse(can_biomes_touch(spruce, catalog["desert"]))
        self.assertFalse(can_biomes_touch(redwood, catalog["desert"]))

        self.assertTrue(is_soil("podzol"))
        self.assertTrue(has_block_tag("podzol", "soil"))
        self.assertTrue(is_tall_plant("double_tall_grass"))
        self.assertTrue(is_tall_plant("double_tall_grass_top"))
        self.assertTrue(is_tall_plant("double_fern"))
        self.assertTrue(is_tall_plant("double_fern_top"))

    def _tree_spot(self, seed, biome_id):
        anchor = None
        for z in range(-1000, 1001, 8):
            for x in range(-1000, 1001, 8):
                if biome_at(x, z, seed, BIOMES).id == biome_id:
                    anchor = (x, z)
                    break
            if anchor is not None:
                break
        if anchor is None:
            return None
        ax, az = anchor
        for z in range(az - 96, az + 97):
            for x in range(ax - 96, ax + 97):
                if biome_at(x, z, seed, BIOMES).id != biome_id:
                    continue
                if x % 16 == 15 or z % 16 == 15:
                    continue
                if tree_at(x, z, seed):
                    return x, z
        return None

    def test_forests_generate(self):
        seed = 3
        spots = {
            "spruce_forest": self._tree_spot(seed, "spruce_forest"),
            "redwood_forest": self._tree_spot(seed, "redwood_forest"),
        }
        self.assertIsNotNone(spots["spruce_forest"], "cool forests should include spruce forest")
        self.assertIsNotNone(spots["redwood_forest"], "cool forests should include redwood forest")

        world = World(seed=seed, caves=[])
        for biome_id, (x, z) in spots.items():
            world.generate_column(x // 16, z // 16)
            log = "spruce_log" if biome_id == "spruce_forest" else "redwood_log"
            surface = "grass" if biome_id == "spruce_forest" else "podzol"
            column = [world.get_block(x, y, z) for y in range(40, 140)]
            self.assertIn(log, column)
            ground = next(y for y in range(40, 140) if column[y - 40] == log) - 1
            self.assertEqual(world.get_block(x, ground, z), surface)

    def test_redwood_needs_four_saplings(self):
        world = World(seed=1)
        for cy in (5, 6):
            world.load_chunk(0, cy, 0)
            world.chunks[(0, cy, 0)].data.fill(0)
        world.set_block(5, 80, 5, "dirt")
        world.set_block(5, 81, 5, "redwood_sapling")
        world.saplings.timers[(5, 81, 5)][1] = 0.01
        self.assertEqual(world.update_saplings(1.0), [])
        self.assertEqual(world.get_block(5, 81, 5), "redwood_sapling")

        for dx in range(2):
            for dz in range(2):
                world.set_block(5 + dx, 80, 5 + dz, "podzol")
                world.set_block(5 + dx, 81, 5 + dz, "redwood_sapling")
        for pos in list(world.saplings.timers):
            world.saplings.timers[pos][1] = 0.01
        grown = world.update_saplings(1.0)
        self.assertTrue(any(tree_id == "redwood" for _x, _y, _z, tree_id in grown))
        for dx in range(2):
            for dz in range(2):
                self.assertEqual(world.get_block(5 + dx, 81, 5 + dz), "redwood_log")

    def test_one_spruce_sapling_grows(self):
        world = World(seed=1)
        world.load_chunk(0, 5, 0)
        world.chunks[(0, 5, 0)].data.fill(0)
        world.set_block(5, 80, 5, "grass")
        world.set_block(5, 81, 5, "spruce_sapling")
        world.saplings.timers[(5, 81, 5)][1] = 0.01
        grown = world.update_saplings(1.0)
        self.assertEqual(len(grown), 1)
        self.assertIn(grown[0][3], {"spruce", "tall_spruce"})
        self.assertEqual(world.get_block(5, 81, 5), "spruce_log")


if __name__ == "__main__":
    unittest.main()
