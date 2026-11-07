import unittest
from unittest.mock import MagicMock

import pyglet
pyglet.options["shadow_window"] = False

from server.blocks import (
    AIR,
    BLOCKS,
    can_fence_connect,
    can_support_torch,
    fence_collision_boxes,
    fence_raycast_boxes,
    get_fence_elements,
    is_fence,
)
from server.items import ITEMS
from server.lighting import get_block_opacity
from server.loot import get_block_drops
from server.recipes import RECIPES
from server.stacks import make_stack, stack_count
from server.world import Chunk, World


class TestFences(unittest.TestCase):
    def _create_world(self):
        world = World()
        for cy in range(8):
            c = Chunk(0, cy, 0)
            c.data.fill(0)
            world.chunks[(0, cy, 0)] = c
        return world

    def test_fence_registration(self):
        self.assertIn("wooden_fence", BLOCKS)
        self.assertIn("wooden_fence", ITEMS)

        fence = BLOCKS["wooden_fence"]
        self.assertTrue(fence.solid)
        self.assertFalse(fence.full_cube)
        self.assertTrue(is_fence(fence))
        self.assertTrue(is_fence("wooden_fence"))

        # Check item places wooden_fence
        item = ITEMS["wooden_fence"]
        self.assertEqual(item.block, "wooden_fence")
        self.assertEqual(item.right, "place")

        # Check recipe (planks and sticks -> 3 fences)
        recipe = next((r for r in RECIPES if getattr(r, "id", "") == "wooden_fence"), None)
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.result, "wooden_fence")
        self.assertEqual(recipe.count, 3)

        # Check drops
        drops = get_block_drops(fence)
        self.assertEqual(drops, [("wooden_fence", 1)])

    def test_pine_fence_registration(self):
        self.assertIn("pine_fence", BLOCKS)
        self.assertIn("pine_fence", ITEMS)

        fence = BLOCKS["pine_fence"]
        self.assertTrue(fence.solid)
        self.assertFalse(fence.full_cube)
        self.assertTrue(is_fence(fence))
        self.assertTrue(is_fence("pine_fence"))

        item = ITEMS["pine_fence"]
        self.assertEqual(item.block, "pine_fence")
        self.assertEqual(item.right, "place")

        recipe = next((r for r in RECIPES if getattr(r, "id", "") == "pine_fence"), None)
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.result, "pine_fence")
        self.assertEqual(recipe.count, 3)

        drops = get_block_drops(fence)
        self.assertEqual(drops, [("pine_fence", 1)])

    def test_fence_connection_rules(self):
        # Other fences connect
        self.assertTrue(can_fence_connect("wooden_fence"))

        # Solid full cubes connect
        self.assertTrue(can_fence_connect("stone"))
        self.assertTrue(can_fence_connect("planks"))

        # Non-full cubes or non-solids do NOT connect
        self.assertFalse(can_fence_connect(AIR))
        self.assertFalse(can_fence_connect("water"))
        self.assertFalse(can_fence_connect("planks_slab"))
        self.assertFalse(can_fence_connect("cobble_vertical_slab"))

    def test_fence_connect_to_double_slabs(self):
        # Fences connect to double horizontal slabs and double vertical slabs
        self.assertTrue(can_fence_connect("double_slab_planks_slab"))
        self.assertTrue(can_fence_connect("double_slab_cobble_slab"))
        self.assertTrue(can_fence_connect("double_vertical_slab_z_cobble_vertical_slab"))
        self.assertTrue(can_fence_connect("double_vertical_slab_x_planks_vertical_slab"))

        # Verify in-world connection to double slab
        world = self._create_world()
        world.set_block(5, 60, 5, "wooden_fence")
        world.set_block(6, 60, 5, "double_slab_planks_slab")

        elements = get_fence_elements(world, 5, 60, 5, BLOCKS["wooden_fence"])
        # Post (1) + east rails (2) = 3 elements
        self.assertEqual(len(elements), 3)

        boxes = world.get_collision_boxes(5, 60, 5)
        # Collision extends towards east neighbor (> 5.7)
        self.assertTrue(any(bx1 > 5.7 for _, _, _, bx1, _, _ in boxes))

    def test_fence_connect_to_vertical_slabs_with_back_to_fence(self):
        # Vertical slab has solid half flush against fence (back to fence):
        # North neighbor (dir "north"): solid half on south side (facing "south")
        self.assertTrue(can_fence_connect("cobble_vertical_slab_south", "north"))
        # South neighbor (dir "south"): solid half on north side (facing "north")
        self.assertTrue(can_fence_connect("cobble_vertical_slab_north", "south"))
        # West neighbor (dir "west"): solid half on east side (facing "east")
        self.assertTrue(can_fence_connect("cobble_vertical_slab_east", "west"))
        # East neighbor (dir "east"): solid half on west side (facing "west")
        self.assertTrue(can_fence_connect("cobble_vertical_slab_west", "east"))

        # When front or side faces the fence, there is an air gap: must NOT connect
        self.assertFalse(can_fence_connect("cobble_vertical_slab_north", "north"))
        self.assertFalse(can_fence_connect("cobble_vertical_slab_south", "south"))
        self.assertFalse(can_fence_connect("cobble_vertical_slab_west", "west"))
        self.assertFalse(can_fence_connect("cobble_vertical_slab_east", "east"))
        self.assertFalse(can_fence_connect("cobble_vertical_slab_north", "east"))
        self.assertFalse(can_fence_connect("cobble_vertical_slab_east", "north"))

        # In-world verification:
        world = self._create_world()
        world.set_block(5, 60, 5, "wooden_fence")
        # Place west-facing vertical slab to the East: solid side touches fence at x=6.0
        world.set_block(6, 60, 5, "cobble_vertical_slab_west")

        elements = get_fence_elements(world, 5, 60, 5, BLOCKS["wooden_fence"])
        self.assertEqual(len(elements), 3)

        boxes = world.get_collision_boxes(5, 60, 5)
        self.assertTrue(any(bx1 > 5.7 for _, _, _, bx1, _, _ in boxes))

        # If east neighbor faces away (facing "east"), fence should NOT connect to it
        world.set_block(6, 60, 5, "cobble_vertical_slab_east")
        elements_no_connect = get_fence_elements(world, 5, 60, 5, BLOCKS["wooden_fence"])
        self.assertEqual(len(elements_no_connect), 1)  # post only

    def test_fence_collision_1_5_blocks_tall(self):
        world = self._create_world()
        world.set_block(5, 59, 5, "stone")
        world.set_block(5, 60, 5, "wooden_fence")

        # Isolated fence collision boxes: top is y + 1.5 = 61.5
        boxes = world.get_collision_boxes(5, 60, 5)
        self.assertTrue(len(boxes) >= 1)
        # Verify box height is 1.5
        for bx0, by0, bz0, bx1, by1, bz1 in boxes:
            self.assertAlmostEqual(by0, 60.0)
            self.assertAlmostEqual(by1, 61.5)

        # Player at ground y=60 trying to walk into the fence
        self.assertTrue(world.collides_box(5.1, 60.0, 5.1, 5.9, 61.8, 5.9))

        # A jumping player with feet at y=61.2 (jump height ~1.2 blocks above ground)
        # still collides with the 1.5 block high fence!
        self.assertTrue(world.collides_box(5.3, 61.2, 5.3, 5.7, 63.0, 5.7))

        # But a player who cleared 1.5 blocks (feet at y=61.6) clears the fence!
        self.assertFalse(world.collides_box(5.3, 61.6, 5.3, 5.7, 63.4, 5.7))

        # Connected fence collision: place another fence next to it at (6, 60, 5)
        world.set_block(6, 60, 5, "wooden_fence")
        boxes_5 = world.get_collision_boxes(5, 60, 5)
        # Should now have both post and east connecting rail
        self.assertTrue(any(bx1 > 5.7 for _, _, _, bx1, _, _ in boxes_5))

    def test_full_block_in_space_above_fence(self):
        world = self._create_world()
        world.set_block(5, 59, 5, "stone")
        world.set_block(5, 60, 5, "wooden_fence")

        # The space directly above the fence at (5, 61, 5) is NOT solid
        self.assertEqual(world.get_block(5, 61, 5), AIR)
        self.assertFalse(world.is_solid(5, 61, 5))

        # Can place a full block (e.g. stone) directly above the fence
        world.set_block(5, 61, 5, "stone")
        self.assertEqual(world.get_block(5, 60, 5), "wooden_fence")
        self.assertEqual(world.get_block(5, 61, 5), "stone")
        self.assertTrue(world.is_solid(5, 61, 5))

        # Torches can also be supported on top of fences
        self.assertTrue(can_support_torch(BLOCKS["wooden_fence"], mount="up"))
        self.assertFalse(can_support_torch(BLOCKS["wooden_fence"], mount="north"))

    def test_fence_raycasting_and_placement_above(self):
        world = self._create_world()
        world.set_block(5, 59, 5, "stone")
        world.set_block(5, 60, 5, "wooden_fence")

        # Raycast pointing down towards top of fence from (5.5, 63.0, 5.5)
        hit = world.raycast(5.5, 63.0, 5.5, 0.0, -1.0, 0.0, max_dist=5.0, solid_only=True)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["x"], 5)
        self.assertEqual(hit["y"], 60)
        self.assertEqual(hit["z"], 5)
        self.assertEqual(hit["ny"], 1)

        # Using Game._place_block to place a stone block on top of the fence
        from client.app import Game
        game = Game.__new__(Game)
        game.world = world
        game.network_client = None
        game.gamemode = "survival"
        game.hotbar = [make_stack("stone", 1)]
        game.selected = 0
        game._upload_block_change = MagicMock()
        game.sound_manager = MagicMock()
        # Player is standing beside the fence
        game.player = MagicMock(pos=[3.0, 60.0, 5.0], radius=0.3, height=1.8, yaw=0.0, gamemode="survival")

        game.target = hit
        game._aim = MagicMock()
        game._place_block()

        # Stone block placed at (5, 61, 5) directly above fence!
        self.assertEqual(world.get_block(5, 61, 5), "stone")
        self.assertEqual(world.get_block(5, 60, 5), "wooden_fence")
        self.assertEqual(stack_count(game.hotbar[0]), 0)

    def test_fence_elements_and_lighting(self):
        world = self._create_world()
        world.set_block(5, 60, 5, "wooden_fence")
        fence_block = BLOCKS["wooden_fence"]

        # Isolated fence has 1 element (the post)
        elements_isolated = get_fence_elements(world, 5, 60, 5, fence_block)
        self.assertEqual(len(elements_isolated), 1)

        # Place neighbor to the east
        world.set_block(6, 60, 5, "wooden_fence")
        elements_connected = get_fence_elements(world, 5, 60, 5, fence_block)
        # Should now have post + 2 east rails (upper and lower)
        self.assertEqual(len(elements_connected), 3)

        # Lighting opacity is 0 (light passes through fences)
        self.assertEqual(get_block_opacity(fence_block), 0)

    def test_fence_inventory_icon(self):
        import numpy as np
        from server.items import ITEM_ATLAS

        fence_item = ITEMS["wooden_fence"]
        planks_item = ITEMS["planks"]

        h, w = ITEM_ATLAS.shape[:2]
        fu0, fv0, fu1, fv1 = fence_item.icon
        pu0, pv0, pu1, pv1 = planks_item.icon

        fence_crop = ITEM_ATLAS[int(fv0 * h) : int(fv1 * h), int(fu0 * w) : int(fu1 * w)]
        planks_crop = ITEM_ATLAS[int(pv0 * h) : int(pv1 * h), int(pu0 * w) : int(pu1 * w)]

        # Fence item must use its fence model with openings rather than a solid planks cube
        self.assertFalse(np.array_equal(fence_crop, planks_crop))
        self.assertLess(np.count_nonzero(fence_crop[:, :, 3]), np.count_nonzero(planks_crop[:, :, 3]))


if __name__ == "__main__":
    unittest.main()

