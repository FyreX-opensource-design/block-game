import math
import unittest

from server.blocks import (
    AIR,
    BLOCKS,
    can_support_torch,
    get_stair_elements,
    get_stair_shape,
    get_stairs_block,
    get_stairs_info,
    is_stairs,
    placement_block,
    stair_collision_boxes,
    stair_raycast_boxes,
)
from server.items import ITEMS, creative_items
from server.loot import get_block_drops
from server.player import Player
from server.recipes import craft_match
from server.world import World


class TestStairs(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=42)
        for cx in (-1, 0, 1):
            for cz in (-1, 0, 1):
                self.world.load_chunk(cx, 6, cz)
                self.world.chunks[(cx, 6, cz)].data.fill(0)

    def test_stairs_registration_and_properties(self):
        stair_types = [
            "planks_stairs",
            "pine_planks_stairs",
            "cobble_stairs",
            "stone_stairs",
            "stone_brick_stairs",
            "sandstone_stairs",
            "smooth_sandstone_stairs",
            "smooth_stone_stairs",
        ]
        for s_id in stair_types:
            self.assertTrue(is_stairs(s_id), f"{s_id} should be recognized as stairs")
            b_obj = BLOCKS[s_id]
            self.assertFalse(b_obj.is_variant, f"{s_id} base block should not be a variant")
            self.assertTrue(b_obj.solid, f"{s_id} should be solid")
            self.assertFalse(b_obj.full_cube, f"{s_id} should not be full_cube")

            # Check 8 variants + top alias
            variants = [
                s_id,
                f"{s_id}_south",
                f"{s_id}_east",
                f"{s_id}_west",
                f"{s_id}_north_top",
                f"{s_id}_south_top",
                f"{s_id}_east_top",
                f"{s_id}_west_top",
            ]
            for v_id in variants:
                self.assertIn(v_id, BLOCKS, f"Variant {v_id} should be registered")
                self.assertTrue(is_stairs(v_id))

            self.assertIn(f"{s_id}_top", BLOCKS)

            # Metadata info
            info = get_stairs_info(s_id)
            self.assertIsNotNone(info)
            self.assertEqual(info["facing"], "north")
            self.assertEqual(info["half"], "bottom")

            top_info = get_stairs_info(f"{s_id}_south_top")
            self.assertIsNotNone(top_info)
            self.assertEqual(top_info["facing"], "south")
            self.assertEqual(top_info["half"], "top")

            # Creative inventory should contain only base block, not variants
            building_items = creative_items("building")
            self.assertIn(s_id, building_items)
            self.assertNotIn(f"{s_id}_south", building_items)
            self.assertNotIn(f"{s_id}_top", building_items)

    def test_stairs_placement(self):
        # Floor placement with yaw (looking north = 0.0)
        placed_n = placement_block("planks_stairs", (0, 1, 0), yaw=0.0)
        self.assertEqual(placed_n, "planks_stairs")

        # Floor placement with yaw (looking south = pi)
        placed_s = placement_block("planks_stairs", (0, 1, 0), yaw=math.pi)
        self.assertEqual(placed_s, "planks_stairs_south")

        # Floor placement with yaw (looking east = -pi/2)
        placed_e = placement_block("planks_stairs", (0, 1, 0), yaw=-math.pi / 2)
        self.assertEqual(placed_e, "planks_stairs_east")

        # Floor placement with yaw (looking west = pi/2)
        placed_w = placement_block("planks_stairs", (0, 1, 0), yaw=math.pi / 2)
        self.assertEqual(placed_w, "planks_stairs_west")

        # Ceiling placement (normal ny < 0) -> top half
        placed_top = placement_block("planks_stairs", (0, -1, 0), yaw=0.0)
        self.assertEqual(placed_top, "planks_stairs_north_top")

        # Side placement lower half (local_y < 0.5) -> bottom half
        placed_side_bot = placement_block("planks_stairs", (1, 0, 0), yaw=0.0, local_y=0.25)
        self.assertEqual(placed_side_bot, "planks_stairs")

        # Side placement upper half (local_y >= 0.5) -> top half
        placed_side_top = placement_block("planks_stairs", (1, 0, 0), yaw=0.0, local_y=0.75)
        self.assertEqual(placed_side_top, "planks_stairs_north_top")

    def test_stairs_shapes_and_connections(self):
        # Place isolated stair
        self.world.set_block(0, 100, 0, "planks_stairs")
        self.assertEqual(get_stair_shape(self.world, 0, 100, 0, "north", "bottom"), "straight")
        self.assertEqual(len(stair_collision_boxes(self.world, 0, 100, 0, BLOCKS["planks_stairs"])), 2)
        self.assertEqual(len(get_stair_elements(self.world, 0, 100, 0, BLOCKS["planks_stairs"])), 2)

        # Place perpendicular stair behind (at z = -1) facing east -> outer corner (hip roof / roof tip)
        self.world.set_block(0, 100, -1, "planks_stairs_east")
        self.assertEqual(get_stair_shape(self.world, 0, 100, 0, "north", "bottom"), "outer_right")
        self.assertEqual(len(stair_collision_boxes(self.world, 0, 100, 0, BLOCKS["planks_stairs"])), 2)
        self.assertEqual(len(get_stair_elements(self.world, 0, 100, 0, BLOCKS["planks_stairs"])), 2)

        # Place perpendicular stair behind facing west -> outer corner
        self.world.set_block(0, 100, -1, "planks_stairs_west")
        self.assertEqual(get_stair_shape(self.world, 0, 100, 0, "north", "bottom"), "outer_left")
        self.assertEqual(len(stair_collision_boxes(self.world, 0, 100, 0, BLOCKS["planks_stairs"])), 2)
        self.assertEqual(len(get_stair_elements(self.world, 0, 100, 0, BLOCKS["planks_stairs"])), 2)

        # Place perpendicular stair in front (at z = 1) facing east -> inner corner (valley)
        self.world.set_block(0, 100, -1, AIR)
        self.world.set_block(0, 100, 1, "planks_stairs_east")
        self.assertEqual(get_stair_shape(self.world, 0, 100, 0, "north", "bottom"), "inner_right")
        self.assertEqual(len(stair_collision_boxes(self.world, 0, 100, 0, BLOCKS["planks_stairs"])), 3)
        self.assertEqual(len(get_stair_elements(self.world, 0, 100, 0, BLOCKS["planks_stairs"])), 4)

        # Place perpendicular stair in front facing west -> inner corner
        self.world.set_block(0, 100, 1, "planks_stairs_west")
        self.assertEqual(get_stair_shape(self.world, 0, 100, 0, "north", "bottom"), "inner_left")

        # Different halves do not connect into corners
        self.world.set_block(0, 100, 1, "planks_stairs_west_top")
        self.assertEqual(get_stair_shape(self.world, 0, 100, 0, "north", "bottom"), "straight")

    def test_stairs_collision_and_raycasting(self):
        # Bottom stair boxes
        b_boxes = stair_collision_boxes(None, 0, 100, 0, BLOCKS["planks_stairs"])
        self.assertEqual(len(b_boxes), 2)
        # Base box: 0.0 to 0.5 height
        self.assertEqual(b_boxes[0], (0.0, 100.0, 0.0, 1.0, 100.5, 1.0))
        # High step box: 0.5 to 1.0 height on north side (z: 0.0..0.5)
        self.assertEqual(b_boxes[1], (0.0, 100.5, 0.0, 1.0, 101.0, 0.5))

        # Top stair boxes
        top_boxes = stair_collision_boxes(None, 0, 100, 0, BLOCKS["planks_stairs_north_top"])
        self.assertEqual(len(top_boxes), 2)
        self.assertEqual(top_boxes[0], (0.0, 100.5, 0.0, 1.0, 101.0, 1.0))
        self.assertEqual(top_boxes[1], (0.0, 100.0, 0.0, 1.0, 100.5, 0.5))

        # Raycast boxes include parts
        r_boxes = stair_raycast_boxes(None, 0, 100, 0, BLOCKS["planks_stairs"])
        self.assertEqual(len(r_boxes), 2)
        self.assertEqual(r_boxes[0][3], "base")
        self.assertEqual(r_boxes[1][3], "step_1")

    def test_player_stepping_up_stairs(self):
        # Platform at y=99
        for x in range(-3, 4):
            for z in range(-3, 4):
                self.world.set_block(x, 99, z, "stone")

        # North stair at (0, 100, 0)
        self.world.set_block(0, 100, 0, "planks_stairs")
        self.world.set_block(0, 100, -1, "stone")

        player = Player(self.world)
        player.pos = [0.5, 100.0, 1.8]

        # Walk forward toward negative z onto the stairs
        for _ in range(25):
            player.move_horizontal(0.0, -0.1)

        # Player should have ascended to y=101.0
        self.assertGreater(player.pos[1], 100.5)
        self.assertAlmostEqual(player.pos[1], 101.0, delta=0.01)

    def test_stairs_crafting_recipes(self):
        # Standard L-shape
        res_std = craft_match(("planks", "", "", "planks", "planks", "", "planks", "planks", "planks"))
        self.assertIsNotNone(res_std)
        self.assertEqual(res_std[0], "planks_stairs")
        self.assertEqual(res_std[1], 4)

        # Mirrored L-shape
        res_mir = craft_match(("", "", "planks", "", "planks", "planks", "planks", "planks", "planks"))
        self.assertIsNotNone(res_mir)
        self.assertEqual(res_mir[0], "planks_stairs")
        self.assertEqual(res_mir[1], 4)

        # Pine planks stairs
        res_pine = craft_match(("pine_planks", "", "", "pine_planks", "pine_planks", "", "pine_planks", "pine_planks", "pine_planks"))
        self.assertIsNotNone(res_pine)
        self.assertEqual(res_pine[0], "pine_planks_stairs")
        self.assertEqual(res_pine[1], 4)

        # Cobblestone stairs
        res_cobble = craft_match(("cobble", "", "", "cobble", "cobble", "", "cobble", "cobble", "cobble"))
        self.assertIsNotNone(res_cobble)
        self.assertEqual(res_cobble[0], "cobble_stairs")
        self.assertEqual(res_cobble[1], 4)

        # Stone stairs
        res_stone = craft_match(("stone", "", "", "stone", "stone", "", "stone", "stone", "stone"))
        self.assertIsNotNone(res_stone)
        self.assertEqual(res_stone[0], "stone_stairs")
        self.assertEqual(res_stone[1], 4)

    def test_stairs_loot_drops(self):
        drops = get_block_drops("planks_stairs")
        self.assertEqual(drops, [("planks_stairs", 1)])

        drops_top = get_block_drops("planks_stairs_north_top")
        self.assertEqual(drops_top, [("planks_stairs", 1)])

    def test_torch_on_stairs(self):
        # Bottom stair cannot support torch on top (it's not flat at y=16)
        self.assertFalse(can_support_torch("planks_stairs", mount="up"))
        # Top (upside-down) stair has full flat top surface at y=16
        self.assertTrue(can_support_torch("planks_stairs_north_top", mount="up"))


if __name__ == "__main__":
    unittest.main()

