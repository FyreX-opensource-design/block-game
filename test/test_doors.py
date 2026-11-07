import math
import unittest
from unittest.mock import MagicMock, patch

import pyglet
pyglet.options["shadow_window"] = False

from server.blocks import (
    AIR,
    BLOCKS,
    door_bounds,
    get_door_block,
    get_door_info,
    is_door,
)
from server.items import ITEMS
from server.loot import get_block_drops
from server.recipes import RECIPES
from server.stacks import make_stack, stack_count, stack_id
from server.world import Chunk, World


class TestDoors(unittest.TestCase):
    def _create_world(self):
        world = World()
        for cy in range(8):
            c = Chunk(0, cy, 0)
            c.data.fill(0)
            world.chunks[(0, cy, 0)] = c
        return world

    def test_door_blocks_and_item_registered(self):
        self.assertIn("wooden_door", BLOCKS)
        self.assertIn("wooden_door", ITEMS)

        # Check that item places wooden_door
        door_item = ITEMS["wooden_door"]
        self.assertEqual(door_item.right, "place")
        self.assertEqual(door_item.block, "wooden_door")

        # Check crafting recipe (6 planks -> 3 doors)
        recipe = next((r for r in RECIPES if getattr(r, "id", "") == "wooden_door"), None)
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.result, "wooden_door")
        self.assertEqual(recipe.count, 3)

        # Check drops
        drops = get_block_drops(BLOCKS["wooden_door"])
        self.assertEqual(drops, [("wooden_door", 1)])

    def test_pine_door_blocks_and_item_registered(self):
        self.assertIn("pine_door", BLOCKS)
        self.assertIn("pine_door", ITEMS)

        door_item = ITEMS["pine_door"]
        self.assertEqual(door_item.right, "place")
        self.assertEqual(door_item.block, "pine_door")

        recipe = next((r for r in RECIPES if getattr(r, "id", "") == "pine_door"), None)
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.result, "pine_door")
        self.assertEqual(recipe.count, 3)

        drops = get_block_drops(BLOCKS["pine_door"])
        self.assertEqual(drops, [("pine_door", 1)])

    def test_door_variants_and_info(self):
        for half in ("lower", "upper"):
            for facing in ("north", "south", "east", "west"):
                for is_open in (False, True):
                    bid = get_door_block("wooden_door", half, facing, is_open)
                    self.assertIn(bid, BLOCKS)
                    self.assertTrue(is_door(bid))
                    info = get_door_info(bid)
                    self.assertIsNotNone(info)
                    self.assertEqual(info["base_id"], "wooden_door")
                    self.assertEqual(info["half"], half)
                    self.assertEqual(info["facing"], facing)
                    self.assertEqual(info["open"], is_open)

    def test_door_bounds_and_collision(self):
        # When closed facing north: z is in [0, 3/16]
        b_closed = door_bounds(10, 5, 10, "north", is_open=False)
        self.assertAlmostEqual(b_closed[0], 10.0)
        self.assertAlmostEqual(b_closed[2], 10.0)
        self.assertAlmostEqual(b_closed[3], 11.0)
        self.assertAlmostEqual(b_closed[5], 10.0 + 3.0 / 16.0)

        # When open facing north: swings to west wall, x in [0, 3/16], z in [0, 1]
        b_open = door_bounds(10, 5, 10, "north", is_open=True)
        self.assertAlmostEqual(b_open[0], 10.0)
        self.assertAlmostEqual(b_open[1], 5.0)
        self.assertAlmostEqual(b_open[3], 10.0 + 3.0 / 16.0)
        self.assertAlmostEqual(b_open[5], 11.0)

        # In world collision test:
        world = self._create_world()
        # Floor
        world.set_block(10, 4, 10, "stone")
        # Lower and upper closed door
        world.set_block(10, 5, 10, "wooden_door_lower_north")
        world.set_block(10, 6, 10, "wooden_door_upper_north")

        # A box passing through the center of the doorway (x=10.5, z=10.1) should collide when closed
        self.assertTrue(world.collides_box(10.2, 5.0, 10.05, 10.8, 6.8, 10.15))

        # Open both halves
        world.set_block(10, 5, 10, "wooden_door_lower_north_open")
        world.set_block(10, 6, 10, "wooden_door_upper_north_open")

        # Center of doorway (x=10.5, z=10.1) is now OPEN and does NOT collide!
        self.assertFalse(world.collides_box(10.4, 5.0, 10.05, 10.8, 6.8, 10.15))
        # But walking into the west edge (where the open door panel rests) still collides:
        self.assertTrue(world.collides_box(10.0, 5.0, 10.4, 10.1, 6.8, 10.8))

    def test_door_placement(self):
        from client.app import Game
        game = Game.__new__(Game)
        game.world = self._create_world()
        game.network_client = None
        game.gamemode = "survival"
        game.hotbar = [make_stack("wooden_door", 2)]
        game.selected = 0
        game._upload_block_change = MagicMock()
        game.sound_manager = MagicMock()

        player = MagicMock()
        player.pos = [5.0, 60.0, 8.0]
        player.radius = 0.3
        player.height = 1.8
        player.yaw = 0.0  # looking North
        player.gamemode = "survival"
        game.player = player

        # Solid floor at (5, 59, 5)
        game.world.set_block(5, 59, 5, "stone")

        # Place door by clicking top face of floor
        game.target = {"x": 5, "y": 59, "z": 5, "nx": 0, "ny": 1, "nz": 0}
        game._aim = MagicMock()
        game._place_block()

        # Lower half placed at (5, 60, 5) and upper half at (5, 61, 5)
        low_block = game.world.get_block(5, 60, 5)
        high_block = game.world.get_block(5, 61, 5)
        self.assertTrue(is_door(low_block))
        self.assertTrue(is_door(high_block))
        low_info = get_door_info(low_block)
        high_info = get_door_info(high_block)
        self.assertEqual(low_info["half"], "lower")
        self.assertEqual(high_info["half"], "upper")
        self.assertFalse(low_info["open"])
        self.assertFalse(high_info["open"])

        # Count in inventory reduced from 2 to 1
        self.assertEqual(stack_count(game.hotbar[0]), 1)

    def test_door_placement_blocked_above_or_below(self):
        from client.app import Game
        game = Game.__new__(Game)
        game.world = self._create_world()
        game.network_client = None
        game.gamemode = "survival"
        game.hotbar = [make_stack("wooden_door", 1)]
        game.selected = 0
        game._upload_block_change = MagicMock()
        game.sound_manager = MagicMock()
        game.player = MagicMock(pos=[5.0, 60.0, 8.0], radius=0.3, height=1.8, yaw=0.0, gamemode="survival")

        # Attempt to place without solid ground
        game.target = {"x": 5, "y": 59, "z": 5, "nx": 0, "ny": 1, "nz": 0}
        game._aim = MagicMock()
        game._place_block()
        self.assertEqual(game.world.get_block(5, 60, 5), AIR)
        self.assertEqual(stack_count(game.hotbar[0]), 1)

        # With ground, but space above is blocked at (5, 61, 5)
        game.world.set_block(5, 59, 5, "stone")
        game.world.set_block(5, 61, 5, "stone")
        game._place_block()
        self.assertEqual(game.world.get_block(5, 60, 5), AIR)
        self.assertEqual(stack_count(game.hotbar[0]), 1)

    def test_door_toggle_open_and_close(self):
        from client.app import Game
        game = Game.__new__(Game)
        game.world = self._create_world()
        game.network_client = None
        game.gamemode = "survival"
        game.hotbar = []
        game.selected = 0
        game.player = MagicMock(pos=[5.0, 60.0, 5.0], eye=1.62, yaw=0.0, pitch=0.0, gamemode="survival")
        game._upload_block_change = MagicMock()
        game.sound_manager = MagicMock()

        # Place lower and upper door
        game.world.set_block(5, 59, 5, "stone")
        game.world.set_block(5, 60, 5, "wooden_door_lower_south")
        game.world.set_block(5, 61, 5, "wooden_door_upper_south")

        # Right click the lower half -> opens both halves
        game.target = {"x": 5, "y": 60, "z": 5}
        game._aim = MagicMock()
        game._on_right_click()

        low_info = get_door_info(game.world.get_block(5, 60, 5))
        high_info = get_door_info(game.world.get_block(5, 61, 5))
        self.assertTrue(low_info["open"])
        self.assertTrue(high_info["open"])

        # Right click the upper half -> closes both halves
        game.target = {"x": 5, "y": 61, "z": 5}
        game._on_right_click()

        low_info = get_door_info(game.world.get_block(5, 60, 5))
        high_info = get_door_info(game.world.get_block(5, 61, 5))
        self.assertFalse(low_info["open"])
        self.assertFalse(high_info["open"])

    def test_breaking_lower_half_breaks_upper_half(self):
        from client.app import Game
        game = Game.__new__(Game)
        game.world = self._create_world()
        game.network_client = None
        game.gamemode = "survival"
        game.hotbar = []
        game.selected = 0
        game.player = MagicMock(pos=[5.0, 60.0, 5.0], eye=1.62, yaw=0.0, pitch=0.0, gamemode="survival")
        game._upload_block_change = MagicMock()
        game.sound_manager = MagicMock()
        game._spawn_break_drops = MagicMock()
        game._reset_breaking = MagicMock()

        # Setup door
        game.world.set_block(5, 59, 5, "stone")
        game.world.set_block(5, 60, 5, "wooden_door_lower_south")
        game.world.set_block(5, 61, 5, "wooden_door_upper_south")

        # Break lower half at (5, 60, 5)
        game.target = {"x": 5, "y": 60, "z": 5}
        game._aim = MagicMock()
        game._break_block()

        # Both lower and upper halves must now be AIR
        self.assertEqual(game.world.get_block(5, 60, 5), AIR)
        self.assertEqual(game.world.get_block(5, 61, 5), AIR)

        # Door drop spawned exactly once
        game._spawn_break_drops.assert_called_once_with([("wooden_door", 1)], 5, 60.5, 5)

    def test_breaking_upper_half_breaks_lower_half(self):
        from client.app import Game
        game = Game.__new__(Game)
        game.world = self._create_world()
        game.network_client = None
        game.gamemode = "survival"
        game.hotbar = []
        game.selected = 0
        game.player = MagicMock(pos=[5.0, 60.0, 5.0], eye=1.62, yaw=0.0, pitch=0.0, gamemode="survival")
        game._upload_block_change = MagicMock()
        game.sound_manager = MagicMock()
        game._spawn_break_drops = MagicMock()
        game._reset_breaking = MagicMock()

        # Setup door
        game.world.set_block(5, 59, 5, "stone")
        game.world.set_block(5, 60, 5, "wooden_door_lower_south")
        game.world.set_block(5, 61, 5, "wooden_door_upper_south")

        # Break upper half at (5, 61, 5)
        game.target = {"x": 5, "y": 61, "z": 5}
        game._aim = MagicMock()
        game._break_block()

        # Both lower and upper halves must now be AIR
        self.assertEqual(game.world.get_block(5, 60, 5), AIR)
        self.assertEqual(game.world.get_block(5, 61, 5), AIR)

        # Door drop spawned exactly once
        game._spawn_break_drops.assert_called_once_with([("wooden_door", 1)], 5, 60.5, 5)

    def test_breaking_ground_beneath_door_breaks_both_halves(self):
        world = self._create_world()
        world.set_block(5, 59, 5, "stone")
        world.set_block(5, 60, 5, "wooden_door_lower_south")
        world.set_block(5, 61, 5, "wooden_door_upper_south")

        # Break the stone block underneath at (5, 59, 5)
        world.set_block(5, 59, 5, AIR)

        # Both door halves must have been broken
        self.assertEqual(world.get_block(5, 60, 5), AIR)
        self.assertEqual(world.get_block(5, 61, 5), AIR)

    def test_door_asset_definition(self):
        from server.doors import DOORS, get_door
        self.assertIn("wooden_door", DOORS)
        door = get_door("wooden_door")
        self.assertIsNotNone(door)
        self.assertEqual(door.id, "wooden_door")
        self.assertEqual(door.sound, "wood")
        self.assertTrue(door.openable_by_hand)
        self.assertEqual(door.textures.get("edge"), "planks")

        # Can also resolve Door instance from a variant ID
        var_door = get_door("wooden_door_lower_south")
        self.assertIsNotNone(var_door)
        self.assertEqual(var_door.id, "wooden_door")

    def test_door_thin_textures_and_lighting(self):
        from server.blocks import FACE_INDEX
        from server.lighting import get_block_opacity

        # Check door opacity is 0 for lighting transmission
        self.assertEqual(get_block_opacity(BLOCKS["wooden_door"]), 0)
        self.assertEqual(get_block_opacity(BLOCKS["wooden_door_lower_north"]), 0)
        self.assertEqual(get_block_opacity(BLOCKS["wooden_door_upper_south_open"]), 0)

        # Check that top face of upper door does not use the front face texture
        upper_door = BLOCKS["wooden_door_upper_south"]
        elements = upper_door.elements[0]
        faces = elements["faces"]

        uv_up = faces[FACE_INDEX["up"]]
        uv_front = faces[FACE_INDEX["south"]]
        uv_west = faces[FACE_INDEX["west"]]

        # Up face (top edge) must NOT be equal to front texture
        self.assertNotEqual(uv_up, uv_front)

        # Thin side (west) must NOT be equal to front texture
        self.assertNotEqual(uv_west, uv_front)

    def test_double_door_variants_and_hinges(self):
        for half in ("lower", "upper"):
            for facing in ("north", "south", "east", "west"):
                for hinge in ("left", "right"):
                    for is_open in (False, True):
                        bid = get_door_block("wooden_door", half, facing, is_open, hinge)
                        self.assertIn(bid, BLOCKS)
                        info = get_door_info(bid)
                        self.assertIsNotNone(info)
                        self.assertEqual(info["base_id"], "wooden_door")
                        self.assertEqual(info["half"], half)
                        self.assertEqual(info["facing"], facing)
                        self.assertEqual(info["hinge"], hinge)
                        self.assertEqual(info["open"], is_open)

    def test_double_door_bounds_and_collision(self):
        # Open bounds for left vs right hinge facing north
        t = 3.0 / 16.0
        b_left = door_bounds(10, 5, 10, "north", is_open=True, hinge="left")
        b_right = door_bounds(11, 5, 10, "north", is_open=True, hinge="right")

        # Left door swings to its west wall: x in [10.0, 10.0 + t]
        self.assertAlmostEqual(b_left[0], 10.0)
        self.assertAlmostEqual(b_left[3], 10.0 + t)

        # Right door swings to its east wall: x in [12.0 - t, 12.0]
        self.assertAlmostEqual(b_right[0], 12.0 - t)
        self.assertAlmostEqual(b_right[3], 12.0)

        world = self._create_world()
        world.set_block(10, 4, 10, "stone")
        world.set_block(11, 4, 10, "stone")

        # Closed double door:
        world.set_block(10, 5, 10, "wooden_door_lower_north")
        world.set_block(10, 6, 10, "wooden_door_upper_north")
        world.set_block(11, 5, 10, "wooden_door_lower_north_right")
        world.set_block(11, 6, 10, "wooden_door_upper_north_right")

        # Walking through middle of doorway (x=10.8 to 11.2) collides when closed
        self.assertTrue(world.collides_box(10.8, 5.0, 10.05, 11.2, 6.8, 10.15))

        # Open both doors
        world.set_block(10, 5, 10, "wooden_door_lower_north_open")
        world.set_block(10, 6, 10, "wooden_door_upper_north_open")
        world.set_block(11, 5, 10, "wooden_door_lower_north_right_open")
        world.set_block(11, 6, 10, "wooden_door_upper_north_right_open")

        # Middle of doorway is now completely open!
        self.assertFalse(world.collides_box(10.4, 5.0, 10.05, 11.6, 6.8, 10.15))
        # Outer edges collide with open door panels
        self.assertTrue(world.collides_box(10.0, 5.0, 10.4, 10.1, 6.8, 10.8))
        self.assertTrue(world.collides_box(11.9, 5.0, 10.4, 12.0, 6.8, 10.8))

    def test_double_door_placement_auto_mirroring(self):
        from client.app import Game
        game = Game.__new__(Game)
        game.world = self._create_world()
        game.network_client = None
        game.gamemode = "survival"
        game.hotbar = [make_stack("wooden_door", 10)]
        game.selected = 0
        game._upload_block_change = MagicMock()
        game.sound_manager = MagicMock()
        game.player = MagicMock(pos=[5.0, 60.0, 8.0], radius=0.3, height=1.8, yaw=0.0, gamemode="survival")

        # Floors at (5, 59, 5) and (6, 59, 5)
        game.world.set_block(5, 59, 5, "stone")
        game.world.set_block(6, 59, 5, "stone")

        # Place first door at (5, 60, 5) facing north
        game.target = {"x": 5, "y": 59, "z": 5, "nx": 0, "ny": 1, "nz": 0}
        game._aim = MagicMock()
        game._place_block()

        door1_info = get_door_info(game.world.get_block(5, 60, 5))
        self.assertEqual(door1_info["hinge"], "left")

        # Place second door adjacent on the right at (6, 60, 5) facing north
        game.target = {"x": 6, "y": 59, "z": 5, "nx": 0, "ny": 1, "nz": 0}
        game._place_block()

        door2_info = get_door_info(game.world.get_block(6, 60, 5))
        self.assertEqual(door2_info["hinge"], "right")

        # Test reverse placement order:
        # Clear doors
        game.world.set_block(5, 60, 5, AIR)
        game.world.set_block(5, 61, 5, AIR)
        game.world.set_block(6, 60, 5, AIR)
        game.world.set_block(6, 61, 5, AIR)

        # Place right door at (6, 60, 5) first
        game.target = {"x": 6, "y": 59, "z": 5, "nx": 0, "ny": 1, "nz": 0}
        game._place_block()
        # Single door placed without walls defaults to left hinge
        self.assertEqual(get_door_info(game.world.get_block(6, 60, 5))["hinge"], "left")

        # Place left door at (5, 60, 5)
        game.target = {"x": 5, "y": 59, "z": 5, "nx": 0, "ny": 1, "nz": 0}
        game._place_block()

        # The new left door is left-hinged, and the existing right door was auto-mirrored to right-hinged!
        self.assertEqual(get_door_info(game.world.get_block(5, 60, 5))["hinge"], "left")
        self.assertEqual(get_door_info(game.world.get_block(6, 60, 5))["hinge"], "right")

    def test_double_door_synchronized_toggle(self):
        from client.app import Game
        game = Game.__new__(Game)
        game.world = self._create_world()
        game.network_client = None
        game.gamemode = "survival"
        game.hotbar = []
        game.selected = 0
        game.player = MagicMock(pos=[5.5, 60.0, 5.0], eye=1.62, yaw=0.0, pitch=0.0, gamemode="survival")
        game._upload_block_change = MagicMock()
        game.sound_manager = MagicMock()

        # Set up closed double doors at (5, 60, 5) and (6, 60, 5) facing south
        # When facing south, +X is left and -X is right, so (5) is right-hinged and (6) is left-hinged
        game.world.set_block(5, 59, 5, "stone")
        game.world.set_block(6, 59, 5, "stone")
        game.world.set_block(5, 60, 5, "wooden_door_lower_south_right")
        game.world.set_block(5, 61, 5, "wooden_door_upper_south_right")
        game.world.set_block(6, 60, 5, "wooden_door_lower_south")
        game.world.set_block(6, 61, 5, "wooden_door_upper_south")

        # Right-click the left door
        game.target = {"x": 5, "y": 60, "z": 5}
        game._aim = MagicMock()
        game._on_right_click()

        # Both doors should be open!
        self.assertTrue(get_door_info(game.world.get_block(5, 60, 5))["open"])
        self.assertTrue(get_door_info(game.world.get_block(5, 61, 5))["open"])
        self.assertTrue(get_door_info(game.world.get_block(6, 60, 5))["open"])
        self.assertTrue(get_door_info(game.world.get_block(6, 61, 5))["open"])

        # Right-click the right door
        game.target = {"x": 6, "y": 60, "z": 5}
        game._on_right_click()

        # Both doors should now be closed!
        self.assertFalse(get_door_info(game.world.get_block(5, 60, 5))["open"])
        self.assertFalse(get_door_info(game.world.get_block(5, 61, 5))["open"])
        self.assertFalse(get_door_info(game.world.get_block(6, 60, 5))["open"])
        self.assertFalse(get_door_info(game.world.get_block(6, 61, 5))["open"])

        # Also test north-facing double doors
        game.world.set_block(5, 60, 5, "wooden_door_lower_north")
        game.world.set_block(5, 61, 5, "wooden_door_upper_north")
        game.world.set_block(6, 60, 5, "wooden_door_lower_north_right")
        game.world.set_block(6, 61, 5, "wooden_door_upper_north_right")

        game.target = {"x": 6, "y": 61, "z": 5}
        game._on_right_click()
        self.assertTrue(get_door_info(game.world.get_block(5, 60, 5))["open"])
        self.assertTrue(get_door_info(game.world.get_block(6, 60, 5))["open"])

        game.target = {"x": 5, "y": 60, "z": 5}
        game._on_right_click()
        self.assertFalse(get_door_info(game.world.get_block(5, 60, 5))["open"])
        self.assertFalse(get_door_info(game.world.get_block(6, 60, 5))["open"])


if __name__ == "__main__":
    unittest.main()
