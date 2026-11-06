import unittest
import numpy as np

import pyglet
pyglet.options["shadow_window"] = False

from server.blocks import (
    AIR,
    BLOCKS,
    can_wall_connect,
    can_support_torch,
    wall_collision_boxes,
    wall_raycast_boxes,
    get_wall_elements,
    is_wall,
    is_fence,
)
from server.items import ITEMS, ITEM_ATLAS
from server.lighting import get_block_opacity
from server.loot import get_block_drops
from server.recipes import RECIPES
from server.world import Chunk, World
from client.player_render import held_item_mesh


WALL_IDS = [
    "cobble_wall",
    "stone_wall",
    "stone_brick_wall",
    "mossy_stone_brick_wall",
    "cracked_stone_brick_wall",
    "chiseled_stone_brick_wall",
    "large_stone_brick_wall",
    "sandstone_wall",
    "smooth_sandstone_wall",
    "smooth_stone_wall",
]


class TestWalls(unittest.TestCase):
    def _create_world(self):
        world = World()
        for cy in range(8):
            c = Chunk(0, cy, 0)
            c.data.fill(0)
            world.chunks[(0, cy, 0)] = c
        return world

    def test_wall_registration(self):
        for wid in WALL_IDS:
            self.assertIn(wid, BLOCKS, f"{wid} missing from BLOCKS")
            self.assertIn(wid, ITEMS, f"{wid} missing from ITEMS")

            wall = BLOCKS[wid]
            self.assertTrue(wall.solid)
            self.assertFalse(wall.full_cube)
            self.assertTrue(is_wall(wall))
            self.assertTrue(is_wall(wid))
            self.assertFalse(is_fence(wall))

            # Check item places wall
            item = ITEMS[wid]
            self.assertEqual(item.block, wid)
            self.assertEqual(item.right, "place")

            # Check recipe (6 blocks -> 6 walls)
            recipe = next((r for r in RECIPES if getattr(r, "id", "") == wid), None)
            self.assertIsNotNone(recipe, f"recipe for {wid} missing")
            self.assertEqual(recipe.result, wid)
            self.assertEqual(recipe.count, 6)

            # Check drops
            drops = get_block_drops(wall)
            self.assertEqual(drops, [(wid, 1)])

    def test_wall_connection_rules(self):
        # Walls connect to same and other walls
        self.assertTrue(can_wall_connect("cobble_wall"))
        self.assertTrue(can_wall_connect("stone_wall"))
        self.assertTrue(can_wall_connect("stone_brick_wall"))

        # Solid full cubes connect
        self.assertTrue(can_wall_connect("stone"))
        self.assertTrue(can_wall_connect("cobble"))
        self.assertTrue(can_wall_connect("planks"))

        # Wooden fences do NOT connect to walls (like Minecraft)
        self.assertFalse(can_wall_connect("wooden_fence"))

        # Non-full cubes or non-solids do NOT connect
        self.assertFalse(can_wall_connect(AIR))
        self.assertFalse(can_wall_connect("water"))
        self.assertFalse(can_wall_connect("cobble_slab"))
        self.assertFalse(can_wall_connect("cobble_vertical_slab"))

    def test_wall_connect_to_double_slabs(self):
        # Walls connect to double horizontal slabs and double vertical slabs
        self.assertTrue(can_wall_connect("double_slab_cobble_slab"))
        self.assertTrue(can_wall_connect("double_slab_planks_slab"))
        self.assertTrue(can_wall_connect("double_vertical_slab_z_cobble_vertical_slab"))
        self.assertTrue(can_wall_connect("double_vertical_slab_x_planks_vertical_slab"))

        # Verify in-world connection to double slab
        world = self._create_world()
        world.set_block(5, 60, 5, "cobble_wall")
        world.set_block(6, 60, 5, "double_slab_cobble_slab")

        elements = get_wall_elements(world, 5, 60, 5, BLOCKS["cobble_wall"])
        # Post (1) + east arm (1) = 2 elements
        self.assertEqual(len(elements), 2)

        boxes = world.get_collision_boxes(5, 60, 5)
        # Collision extends towards east neighbor (> 5.7)
        self.assertTrue(any(bx1 > 5.7 for _, _, _, bx1, _, _ in boxes))

    def test_wall_connect_to_vertical_slabs_with_back_to_wall(self):
        # Vertical slab has solid half flush against wall (back to wall):
        # North neighbor (dir "north"): solid half on south side (facing "south")
        self.assertTrue(can_wall_connect("cobble_vertical_slab_south", "north"))
        # South neighbor (dir "south"): solid half on north side (facing "north")
        self.assertTrue(can_wall_connect("cobble_vertical_slab_north", "south"))
        # West neighbor (dir "west"): solid half on east side (facing "east")
        self.assertTrue(can_wall_connect("cobble_vertical_slab_east", "west"))
        # East neighbor (dir "east"): solid half on west side (facing "west")
        self.assertTrue(can_wall_connect("cobble_vertical_slab_west", "east"))

        # When front or side faces the wall, there is an air gap: must NOT connect
        self.assertFalse(can_wall_connect("cobble_vertical_slab_north", "north"))
        self.assertFalse(can_wall_connect("cobble_vertical_slab_south", "south"))
        self.assertFalse(can_wall_connect("cobble_vertical_slab_west", "west"))
        self.assertFalse(can_wall_connect("cobble_vertical_slab_east", "east"))
        self.assertFalse(can_wall_connect("cobble_vertical_slab_north", "east"))
        self.assertFalse(can_wall_connect("cobble_vertical_slab_east", "north"))

        # In-world verification:
        world = self._create_world()
        world.set_block(5, 60, 5, "cobble_wall")
        world.set_block(6, 60, 5, "cobble_vertical_slab_west")

        elements = get_wall_elements(world, 5, 60, 5, BLOCKS["cobble_wall"])
        self.assertEqual(len(elements), 2)

        boxes = world.get_collision_boxes(5, 60, 5)
        self.assertTrue(any(bx1 > 5.7 for _, _, _, bx1, _, _ in boxes))

        # If east neighbor faces away, wall should NOT connect to it
        world.set_block(6, 60, 5, "cobble_vertical_slab_east")
        elements_no_connect = get_wall_elements(world, 5, 60, 5, BLOCKS["cobble_wall"])
        self.assertEqual(len(elements_no_connect), 1)  # post only

    def test_wall_collision_1_5_blocks_tall(self):
        world = self._create_world()
        world.set_block(5, 59, 5, "stone")
        world.set_block(5, 60, 5, "cobble_wall")

        # Isolated wall collision boxes: top is y + 1.5 = 61.5
        boxes = world.get_collision_boxes(5, 60, 5)
        self.assertTrue(len(boxes) >= 1)
        for bx0, by0, bz0, bx1, by1, bz1 in boxes:
            self.assertAlmostEqual(by0, 60.0)
            self.assertAlmostEqual(by1, 61.5)

        # Player at ground y=60 trying to walk into the wall
        self.assertTrue(world.collides_box(5.1, 60.0, 5.1, 5.9, 61.8, 5.9))

        # A jumping player with feet at y=61.2 (jump height ~1.2 blocks above ground)
        # still collides with the 1.5 block high wall!
        self.assertTrue(world.collides_box(5.3, 61.2, 5.3, 5.7, 63.0, 5.7))

        # But a player who cleared 1.5 blocks (feet at y=61.6) clears the wall!
        self.assertFalse(world.collides_box(5.3, 61.6, 5.3, 5.7, 63.4, 5.7))

        # Connected wall collision: place another wall next to it at (6, 60, 5)
        world.set_block(6, 60, 5, "cobble_wall")
        boxes_5 = world.get_collision_boxes(5, 60, 5)
        # Should now have both post and east connecting side
        self.assertTrue(any(bx1 > 5.7 for _, _, _, bx1, _, _ in boxes_5))

    def test_full_block_in_space_above_wall(self):
        world = self._create_world()
        world.set_block(5, 60, 5, "cobble_wall")

        # The space directly above the wall at (5, 61, 5) is NOT solid
        self.assertFalse(world.is_solid(5, 61, 5))
        self.assertEqual(world.get_collision_boxes(5, 61, 5), [])

        # Can place a full block (e.g. stone) directly above the wall
        world.set_block(5, 61, 5, "stone")
        self.assertEqual(world.get_block(5, 60, 5), "cobble_wall")
        self.assertEqual(world.get_block(5, 61, 5), "stone")
        self.assertTrue(world.is_solid(5, 61, 5))

        # Torches can also be supported on top of walls
        self.assertTrue(can_support_torch(BLOCKS["cobble_wall"], mount="up"))
        self.assertFalse(can_support_torch(BLOCKS["cobble_wall"], mount="north"))

    def test_wall_raycasting(self):
        world = self._create_world()
        world.set_block(5, 60, 5, "cobble_wall")

        boxes = wall_raycast_boxes(world, 5, 60, 5)
        self.assertTrue(len(boxes) >= 1)
        # Verify box height is 1.0 (so clicking top targets y + 1)
        for _, _, _, _, (bx0, by0, bz0, bx1, by1, bz1) in boxes:
            self.assertAlmostEqual(by0, 60.0)
            self.assertAlmostEqual(by1, 61.0)

    def test_wall_elements_and_lighting(self):
        world = self._create_world()
        world.set_block(5, 60, 5, "cobble_wall")
        wall_block = BLOCKS["cobble_wall"]

        # Isolated wall has 1 element (the post, 8x8 pixels: 4 to 12)
        elements_isolated = get_wall_elements(world, 5, 60, 5, wall_block)
        self.assertEqual(len(elements_isolated), 1)
        self.assertEqual(elements_isolated[0]["from"], [4.0, 0.0, 4.0])
        self.assertEqual(elements_isolated[0]["to"], [12.0, 16.0, 12.0])

        # Connected wall has post + side elements (height 14, width 6)
        world.set_block(6, 60, 5, "cobble_wall")
        elements_connected = get_wall_elements(world, 5, 60, 5, wall_block)
        self.assertEqual(len(elements_connected), 2)
        east_arm = next(el for el in elements_connected if el["from"][0] >= 12.0)
        self.assertEqual(east_arm["to"][1], 14.0)

        # Lighting opacity is 0 (light passes through walls)
        self.assertEqual(get_block_opacity(wall_block), 0)

    def test_wall_inventory_icon(self):
        wall_item = ITEMS["cobble_wall"]
        cobble_item = ITEMS["cobble"]

        h, w = ITEM_ATLAS.shape[:2]
        wu0, wv0, wu1, wv1 = wall_item.icon
        cu0, cv0, cu1, cv1 = cobble_item.icon

        wall_crop = ITEM_ATLAS[int(wv0 * h) : int(wv1 * h), int(wu0 * w) : int(wu1 * w)]
        cobble_crop = ITEM_ATLAS[int(cv0 * h) : int(cv1 * h), int(cu0 * w) : int(cu1 * w)]

        # Wall item must use its wall model rather than a solid cobble cube
        self.assertFalse(np.array_equal(wall_crop, cobble_crop))
        self.assertLess(np.count_nonzero(wall_crop[:, :, 3]), np.count_nonzero(cobble_crop[:, :, 3]))

    def test_straight_wall_non_post_section(self):
        world = self._create_world()
        # Create a straight stretch of 3 walls along X: 4, 5, 6
        world.set_block(4, 60, 5, "cobble_wall")
        world.set_block(5, 60, 5, "cobble_wall")
        world.set_block(6, 60, 5, "cobble_wall")
        wall_block = BLOCKS["cobble_wall"]

        # Middle wall at (5, 60, 5) is on a straight stretch with no blocks above:
        # It must NOT have a post! It should have 1 continuous straight element (height 14).
        elements_mid = get_wall_elements(world, 5, 60, 5, wall_block)
        self.assertEqual(len(elements_mid), 1)
        self.assertEqual(elements_mid[0]["from"], [0.0, 0.0, 5.0])
        self.assertEqual(elements_mid[0]["to"], [16.0, 14.0, 11.0])

        # Raycast boxes for middle wall: 1 box spanning full block X, height 14/16
        r_boxes = wall_raycast_boxes(world, 5, 60, 5)
        self.assertEqual(len(r_boxes), 1)
        _, _, _, tag, (rx0, ry0, rz0, rx1, ry1, rz1) = r_boxes[0]
        self.assertEqual(tag, "wall_straight")
        self.assertAlmostEqual(rx0, 5.0)
        self.assertAlmostEqual(rx1, 6.0)
        self.assertAlmostEqual(ry1 - ry0, 14.0 / 16.0)

        # End wall at (4, 60, 5) only connects East -> has a post!
        elements_end = get_wall_elements(world, 4, 60, 5, wall_block)
        self.assertEqual(len(elements_end), 2)  # post + east arm

        # Adding a torch on top of the middle wall introduces a post
        world.set_block(5, 61, 5, "torch")
        elements_with_torch = get_wall_elements(world, 5, 60, 5, wall_block)
        # Should now have post + west arm + east arm
        self.assertEqual(len(elements_with_torch), 3)

    def test_stacked_walls_tall_connections(self):
        world = self._create_world()
        # Stack 2 layers of walls along X: 4, 5, 6 at y=60 and y=61
        for x in (4, 5, 6):
            world.set_block(x, 60, 5, "cobble_wall")
            world.set_block(x, 61, 5, "cobble_wall")
        wall_block = BLOCKS["cobble_wall"]

        # Middle bottom wall at (5, 60, 5):
        # Stacked under wall at y=61 -> straight_we_tall (height 16, no post)
        elements_bot_mid = get_wall_elements(world, 5, 60, 5, wall_block)
        self.assertEqual(len(elements_bot_mid), 1)
        self.assertEqual(elements_bot_mid[0]["from"], [0.0, 0.0, 5.0])
        self.assertEqual(elements_bot_mid[0]["to"], [16.0, 16.0, 11.0])

        # Middle top wall at (5, 61, 5):
        # Straight wall with nothing above -> straight_we (height 14, no post)
        elements_top_mid = get_wall_elements(world, 5, 61, 5, wall_block)
        self.assertEqual(len(elements_top_mid), 1)
        self.assertEqual(elements_top_mid[0]["from"], [0.0, 0.0, 5.0])
        self.assertEqual(elements_top_mid[0]["to"], [16.0, 14.0, 11.0])

        # End bottom wall at (4, 60, 5):
        # Has post + tall east arm (height 16)
        elements_bot_end = get_wall_elements(world, 4, 60, 5, wall_block)
        self.assertEqual(len(elements_bot_end), 2)
        east_arm = next(el for el in elements_bot_end if el["from"][0] >= 12.0)
        self.assertEqual(east_arm["to"][1], 16.0)

    def test_held_wall_model(self):
        matrix = np.eye(4, dtype=np.float32)
        mesh, kind = held_item_mesh("cobble_wall", matrix)
        self.assertEqual(kind, "block")
        verts, indices = mesh
        self.assertGreater(len(verts), 0)
        self.assertGreater(len(indices), 0)


if __name__ == "__main__":
    unittest.main()
