import math
import unittest

import pyglet
pyglet.options["shadow_window"] = False

from server.blocks import (
    AIR,
    BLOCKS,
    can_fence_connect,
    can_wall_connect,
    fence_gate_bounds,
    fence_gate_collision_boxes,
    fence_gate_raycast_boxes,
    get_fence_gate_block,
    get_fence_gate_info,
    is_fence_gate,
    placement_block,
)
from server.items import ITEMS
from server.loot import get_block_drops
from server.recipes import RECIPES
from server.world import Chunk, World


class TestFenceGates(unittest.TestCase):
    def _create_world(self):
        world = World()
        for cy in range(8):
            c = Chunk(0, cy, 0)
            c.data.fill(0)
            world.chunks[(0, cy, 0)] = c
        return world

    def test_fence_gate_registration(self):
        self.assertIn("wooden_fence_gate", BLOCKS)
        self.assertIn("wooden_fence_gate", ITEMS)

        gate = BLOCKS["wooden_fence_gate"]
        self.assertTrue(gate.solid)
        self.assertFalse(gate.full_cube)
        self.assertTrue(is_fence_gate(gate))
        self.assertTrue(is_fence_gate("wooden_fence_gate"))

        # Check variants
        for facing in ("north", "south", "east", "west"):
            for is_open in (False, True):
                var_id = get_fence_gate_block("wooden_fence_gate", facing, is_open)
                self.assertIn(var_id, BLOCKS)
                var_block = BLOCKS[var_id]
                self.assertTrue(is_fence_gate(var_block))
                info = get_fence_gate_info(var_block)
                self.assertIsNotNone(info)
                self.assertEqual(info["facing"], facing)
                self.assertEqual(info["open"], is_open)
                self.assertEqual(info["base_id"], "wooden_fence_gate")

        # Check item
        item = ITEMS["wooden_fence_gate"]
        self.assertEqual(item.block, "wooden_fence_gate")
        self.assertEqual(item.right, "place")

        # Check recipe (sticks and planks: S#S / S#S)
        recipe = next((r for r in RECIPES if getattr(r, "id", "") == "wooden_fence_gate"), None)
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.result, "wooden_fence_gate")
        self.assertEqual(recipe.count, 1)

        # Check drops
        drops = get_block_drops(gate)
        self.assertEqual(drops, [("wooden_fence_gate", 1)])

    def test_collision_boxes_closed_and_open(self):
        world = self._create_world()
        # Closed gate facing north: 1.5 blocks high
        world.set_block(5, 1, 5, "wooden_fence_gate_north")
        boxes_closed = world.get_collision_boxes(5, 1, 5)
        self.assertEqual(len(boxes_closed), 1)
        bx0, by0, bz0, bx1, by1, bz1 = boxes_closed[0]
        self.assertAlmostEqual(bx0, 5.0)
        self.assertAlmostEqual(bx1, 6.0)
        self.assertAlmostEqual(by0, 1.0)
        self.assertAlmostEqual(by1, 2.5)  # 1.5 blocks tall!
        self.assertAlmostEqual(bz0, 5.0 + 6.0 / 16.0)
        self.assertAlmostEqual(bz1, 5.0 + 10.0 / 16.0)

        # Open gate facing north: center passage is clear
        world.set_block(5, 1, 5, "wooden_fence_gate_north_open")
        boxes_open = world.get_collision_boxes(5, 1, 5)
        # Should only have side posts
        self.assertEqual(len(boxes_open), 2)
        # Left post: x in [5.0, 5.0 + 2/16]
        self.assertAlmostEqual(boxes_open[0][0], 5.0)
        self.assertAlmostEqual(boxes_open[0][3], 5.0 + 2.0 / 16.0)
        # Right post: x in [5.0 + 14/16, 6.0]
        self.assertAlmostEqual(boxes_open[1][0], 5.0 + 14.0 / 16.0)
        self.assertAlmostEqual(boxes_open[1][3], 6.0)
        # Center x in [5.0 + 2/16, 5.0 + 14/16] has no collision
        for box in boxes_open:
            self.assertFalse(box[0] <= 5.5 <= box[3] and box[2] <= 5.5 <= box[5])

    def test_fence_and_wall_connections(self):
        # Gate facing north/south runs along X (east-west)
        # Neighboring fences to west and east must connect
        self.assertTrue(can_fence_connect("wooden_fence_gate_north", "west"))
        self.assertTrue(can_fence_connect("wooden_fence_gate_north", "east"))
        self.assertFalse(can_fence_connect("wooden_fence_gate_north", "north"))
        self.assertFalse(can_fence_connect("wooden_fence_gate_north", "south"))

        self.assertTrue(can_fence_connect("wooden_fence_gate_south", "west"))
        self.assertTrue(can_fence_connect("wooden_fence_gate_south", "east"))
        self.assertFalse(can_fence_connect("wooden_fence_gate_south", "north"))
        self.assertFalse(can_fence_connect("wooden_fence_gate_south", "south"))

        # Gate facing east/west runs along Z (north-south)
        # Neighboring fences to north and south must connect
        self.assertTrue(can_fence_connect("wooden_fence_gate_east", "north"))
        self.assertTrue(can_fence_connect("wooden_fence_gate_east", "south"))
        self.assertFalse(can_fence_connect("wooden_fence_gate_east", "west"))
        self.assertFalse(can_fence_connect("wooden_fence_gate_east", "east"))

        self.assertTrue(can_fence_connect("wooden_fence_gate_west", "north"))
        self.assertTrue(can_fence_connect("wooden_fence_gate_west", "south"))
        self.assertFalse(can_fence_connect("wooden_fence_gate_west", "west"))
        self.assertFalse(can_fence_connect("wooden_fence_gate_west", "east"))

        # Same for walls
        self.assertTrue(can_wall_connect("wooden_fence_gate_north", "west"))
        self.assertTrue(can_wall_connect("wooden_fence_gate_north", "east"))
        self.assertFalse(can_wall_connect("wooden_fence_gate_north", "north"))
        self.assertFalse(can_wall_connect("wooden_fence_gate_north", "south"))

        self.assertTrue(can_wall_connect("wooden_fence_gate_east", "north"))
        self.assertTrue(can_wall_connect("wooden_fence_gate_east", "south"))
        self.assertFalse(can_wall_connect("wooden_fence_gate_east", "west"))
        self.assertFalse(can_wall_connect("wooden_fence_gate_east", "east"))

    def test_placement_block_orientation(self):
        # Placement based on yaw (in radians)
        # yaw = 0.0 -> north
        placed = placement_block("wooden_fence_gate", (0, 1, 0), yaw=0.0)
        self.assertEqual(placed, "wooden_fence_gate_north")

        # yaw = math.pi -> south
        placed = placement_block("wooden_fence_gate", (0, 1, 0), yaw=math.pi)
        self.assertEqual(placed, "wooden_fence_gate_south")

        # yaw = -math.pi / 2 -> east
        placed = placement_block("wooden_fence_gate", (0, 1, 0), yaw=-math.pi / 2)
        self.assertEqual(placed, "wooden_fence_gate_east")

        # yaw = math.pi / 2 -> west
        placed = placement_block("wooden_fence_gate", (0, 1, 0), yaw=math.pi / 2)
        self.assertEqual(placed, "wooden_fence_gate_west")


if __name__ == "__main__":
    unittest.main()
