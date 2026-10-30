import unittest

from server.blocks import AIR, BLOCKS, placement_block
from server.recipes import craft_match
from server.tags import (
    get_block_tag,
    get_item_tag,
    has_block_tag,
    has_item_tag,
    matches_ingredient,
    matches_tag,
)
from server.trees import DECORATORS
from server.world import World


class TestTagsAndLeafDecay(unittest.TestCase):
    def test_tags_loading_and_matching(self):
        # Block tags
        self.assertTrue(has_block_tag("log", "logs"))
        self.assertTrue(has_block_tag("log_x", "logs"))
        self.assertTrue(has_block_tag("leaves", "leaves"))
        self.assertTrue(has_block_tag("leaves_persistent", "leaves"))
        self.assertTrue(has_block_tag("dirt", "soil"))
        self.assertTrue(has_block_tag("grass", "soil"))
        self.assertTrue(has_block_tag("mud", "soil"))
        self.assertTrue(has_block_tag("rose", "flowers"))
        self.assertTrue(has_block_tag("dandelion", "flowers"))

        # Item tags
        self.assertTrue(has_item_tag("log", "logs"))
        self.assertTrue(has_item_tag("leaves", "leaves"))
        self.assertTrue(has_item_tag("planks", "planks"))

        # Tag matching
        self.assertTrue(matches_tag("log", "logs"))
        self.assertTrue(matches_tag("leaves", "leaves"))
        self.assertTrue(matches_ingredient("#logs", "log"))
        self.assertTrue(matches_ingredient("#logs", "log_x"))
        self.assertTrue(matches_ingredient("#soil", "dirt"))
        self.assertFalse(matches_ingredient("#soil", "stone"))

    def test_recipe_matching_with_tags(self):
        # Planks recipe uses #logs ingredient
        res = craft_match(("log", "", "", ""))
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "planks")
        self.assertEqual(res[1], 4)

        # Rotation variant of log also matches #logs
        res_x = craft_match(("log_x", "", "", ""))
        self.assertIsNotNone(res_x)
        self.assertEqual(res_x[0], "planks")
        self.assertEqual(res_x[1], 4)

        # Pine log crafts into pine planks
        res_pine = craft_match(("pine_log", "", "", ""))
        self.assertIsNotNone(res_pine)
        self.assertEqual(res_pine[0], "pine_planks")
        self.assertEqual(res_pine[1], 4)

        # Pine planks match #planks recipes (e.g. stick)
        res_stick = craft_match(("pine_planks", "", "pine_planks", ""))
        self.assertIsNotNone(res_stick)
        self.assertEqual(res_stick[0], "stick")
        self.assertEqual(res_stick[1], 4)

    def test_player_placed_leaves_are_persistent(self):
        placed = placement_block("leaves", (0, 1, 0))
        self.assertEqual(placed, "leaves_persistent")
        block_obj = BLOCKS[placed]
        self.assertTrue(block_obj.properties.get("persistent"))
        self.assertFalse(block_obj.properties.get("decayable"))

    def test_leaf_decay_system(self):
        import random
        random.seed(42)
        world = World(seed=123)
        cx, cy, cz = 0, 5, 0
        world.load_chunk(cx, cy, cz)
        world.chunks[(cx, cy, cz)].data.fill(0)

        # Set up a tree: trunk at (5, 80, 5), leaves in 3x3 at y=81
        world.set_block(5, 80, 5, "log")
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                world.set_block(5 + dx, 81, 5 + dz, "leaves")

        # Set up player placed leaf at (9, 81, 9) without any logs nearby
        world.set_block(9, 81, 9, "leaves_persistent")

        # Update decay: connected leaves and persistent leaf must NOT decay
        for _ in range(25):
            world.update_leaves(0.1)

        self.assertEqual(world.get_block(5, 81, 5), "leaves")
        self.assertEqual(world.get_block(6, 81, 5), "leaves")
        self.assertEqual(world.get_block(9, 81, 9), "leaves_persistent")

        # Chop down the trunk
        initial_entities = len(world.entities)
        world.set_block(5, 80, 5, AIR)

        # Update decay: unsupported leaves must slowly decay to AIR and drop loot
        for _ in range(50):
            world.update_leaves(0.1)

        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                self.assertEqual(world.get_block(5 + dx, 81, 5 + dz), AIR)
        # Persistent leaf must STILL remain intact
        self.assertEqual(world.get_block(9, 81, 9), "leaves_persistent")

        # Loot dropped from decayed leaves (sticks)
        self.assertGreater(len(world.entities), initial_entities)
        self.assertTrue(any(e.item == "stick" for e in world.entities))

    def test_tree_shapes_diversity(self):
        tree_map = {t.id: t for t in DECORATORS}
        self.assertIn("oak", tree_map)
        self.assertIn("pine", tree_map)
        self.assertIn("fancy_oak", tree_map)
        self.assertIn("tall_oak", tree_map)
        self.assertIn("shrub", tree_map)

        self.assertEqual(tree_map["oak"].shape, "blob")
        self.assertEqual(tree_map["pine"].shape, "conifer")
        self.assertEqual(tree_map["fancy_oak"].shape, "branching")
        self.assertEqual(tree_map["tall_oak"].shape, "tall")
        self.assertEqual(tree_map["shrub"].shape, "shrub")

        world = World(seed=999)
        world.load_chunk(0, 5, 0)
        chunk = world.chunks[(0, 5, 0)]

        for t_id, tree in tree_map.items():
            chunk.data.fill(0)
            world.set_block(8, 80, 8, "dirt")
            world._place_tree(chunk.data, 8, 8, 80, 8, 8, 80, tree)

            has_trunk = any(world.get_block(8, y, 8) == tree.trunk for y in range(81, 96))
            has_leaves = any(
                world.get_block(x, y, z) == tree.leaves
                for x in range(5, 12)
                for y in range(81, 96)
                for z in range(5, 12)
            )
            self.assertTrue(has_trunk, f"{t_id} should place trunk")
            self.assertTrue(has_leaves, f"{t_id} should place leaves")
            if t_id == "fancy_oak":
                has_sideways_branch = any(
                    world.get_block(x, y, z) in ("log_x", "log_z")
                    for x in range(5, 12)
                    for y in range(81, 96)
                    for z in range(5, 12)
                    if not (x == 8 and z == 8)
                )
                self.assertTrue(has_sideways_branch, "fancy_oak branches should generate sideways logs (log_x / log_z)")


if __name__ == "__main__":
    unittest.main()
