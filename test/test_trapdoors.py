import unittest
from unittest.mock import MagicMock

import pyglet
pyglet.options["shadow_window"] = False

from client.app import Game
from server.blocks import (
    AIR,
    BLOCKS,
    FACE_INDEX,
    get_trapdoor_block,
    get_trapdoor_info,
    is_trapdoor,
    trapdoor_bounds,
)
from server.doors import TRAPDOORS, get_trapdoor
from server.items import ITEMS
from server.lighting import get_block_opacity
from server.loot import get_block_drops
from server.recipes import RECIPES
from server.stacks import make_stack, stack_count, stack_id
from server.world import Chunk, World


class TestTrapdoors(unittest.TestCase):
    def _create_world(self):
        world = World()
        for cy in range(8):
            c = Chunk(0, cy, 0)
            c.data.fill(0)
            world.chunks[(0, cy, 0)] = c
        return world

    def test_trapdoor_registered_and_variants(self):
        self.assertIn("wooden_trapdoor", BLOCKS)
        self.assertIn("wooden_trapdoor", ITEMS)
        self.assertIn("wooden_trapdoor", TRAPDOORS)

        # Asset definition
        td_asset = get_trapdoor("wooden_trapdoor")
        self.assertIsNotNone(td_asset)
        self.assertEqual(td_asset.sound, "wood")
        self.assertTrue(td_asset.openable_by_hand)

        # Item definition
        item = ITEMS["wooden_trapdoor"]
        self.assertEqual(item.block, "wooden_trapdoor")
        self.assertEqual(item.right, "place")

        # Loot table
        drops = get_block_drops(BLOCKS["wooden_trapdoor"])
        self.assertEqual(drops, [("wooden_trapdoor", 1)])

        # Recipe (6 planks -> 2 trapdoors)
        recipe = next((r for r in RECIPES if getattr(r, "id", "") == "wooden_trapdoor"), None)
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.result, "wooden_trapdoor")
        self.assertEqual(recipe.count, 2)

        # All 16 variants
        for half in ("bottom", "top"):
            for facing in ("north", "south", "east", "west"):
                for is_open in (False, True):
                    bid = get_trapdoor_block("wooden_trapdoor", half, facing, is_open)
                    self.assertIn(bid, BLOCKS)
                    self.assertTrue(is_trapdoor(bid))
                    info = get_trapdoor_info(bid)
                    self.assertIsNotNone(info)
                    self.assertEqual(info["base_id"], "wooden_trapdoor")
                    self.assertEqual(info["half"], half)
                    self.assertEqual(info["facing"], facing)
                    self.assertEqual(info["open"], is_open)

    def test_trapdoor_bounds_and_collision(self):
        t = 3.0 / 16.0
        # Closed bottom
        bb = trapdoor_bounds(10, 5, 10, "bottom", "north", is_open=False)
        self.assertAlmostEqual(bb[1], 5.0)
        self.assertAlmostEqual(bb[4], 5.0 + t)

        # Closed top
        bt = trapdoor_bounds(10, 5, 10, "top", "north", is_open=False)
        self.assertAlmostEqual(bt[1], 5.0 + 1.0 - t)
        self.assertAlmostEqual(bt[4], 6.0)

        # Open north (upright against north wall z in [0, t])
        bon = trapdoor_bounds(10, 5, 10, "bottom", "north", is_open=True)
        self.assertAlmostEqual(bon[2], 10.0)
        self.assertAlmostEqual(bon[5], 10.0 + t)
        self.assertAlmostEqual(bon[1], 5.0)
        self.assertAlmostEqual(bon[4], 6.0)

        # Open south (upright against south wall z in [1-t, 1])
        bos = trapdoor_bounds(10, 5, 10, "bottom", "south", is_open=True)
        self.assertAlmostEqual(bos[2], 10.0 + 1.0 - t)
        self.assertAlmostEqual(bos[5], 11.0)

        # Open west (upright against west wall x in [0, t])
        bow = trapdoor_bounds(10, 5, 10, "bottom", "west", is_open=True)
        self.assertAlmostEqual(bow[0], 10.0)
        self.assertAlmostEqual(bow[3], 10.0 + t)

        # Open east (upright against east wall x in [1-t, 1])
        boe = trapdoor_bounds(10, 5, 10, "bottom", "east", is_open=True)
        self.assertAlmostEqual(boe[0], 10.0 + 1.0 - t)
        self.assertAlmostEqual(boe[3], 11.0)

        # World collision
        world = self._create_world()
        world.set_block(5, 5, 5, "wooden_trapdoor_bottom_north")
        # Collides with player walking on the hatch
        self.assertTrue(world.collides_box(5.2, 5.0, 5.2, 5.8, 5.5, 5.8))

        # Open the trapdoor
        world.set_block(5, 5, 5, "wooden_trapdoor_bottom_north_open")
        # Player falling through the hole no longer collides in the center
        self.assertFalse(world.collides_box(5.3, 5.0, 5.3, 5.7, 5.5, 5.7))
        # But collides with the north wall where the trapdoor is resting upright
        self.assertTrue(world.collides_box(5.2, 5.0, 5.0, 5.8, 5.5, 5.1))

    def test_trapdoor_placement(self):
        game = Game.__new__(Game)
        game.world = self._create_world()
        game.network_client = None
        game.gamemode = "survival"
        game.hotbar = [make_stack("wooden_trapdoor", 4)]
        game.selected = 0
        game.player = MagicMock(pos=[5.0, 60.0, 8.0], radius=0.3, height=1.8, yaw=0.0, gamemode="survival")
        game._upload_block_change = MagicMock()
        game.sound_manager = MagicMock()

        # Place on top of a solid block (ny=1) -> bottom half
        game.world.set_block(5, 59, 5, "stone")
        game.target = {"x": 5, "y": 59, "z": 5, "nx": 0, "ny": 1, "nz": 0}
        game._aim = MagicMock()
        game._place_block()

        placed = game.world.get_block(5, 60, 5)
        self.assertTrue(is_trapdoor(placed))
        info = get_trapdoor_info(placed)
        self.assertEqual(info["half"], "bottom")
        self.assertFalse(info["open"])
        self.assertEqual(stack_count(game.hotbar[0]), 3)

        # Place under a solid ceiling (ny=-1) -> top half
        game.world.set_block(5, 63, 5, "stone")
        game.target = {"x": 5, "y": 63, "z": 5, "nx": 0, "ny": -1, "nz": 0}
        game._place_block()

        placed_top = game.world.get_block(5, 62, 5)
        self.assertTrue(is_trapdoor(placed_top))
        info_top = get_trapdoor_info(placed_top)
        self.assertEqual(info_top["half"], "top")
        self.assertFalse(info_top["open"])
        self.assertEqual(stack_count(game.hotbar[0]), 2)

        # Place against side of a block (nz=1, pointing north into the wall)
        game.world.set_block(6, 60, 4, "stone")
        game.target = {"x": 6, "y": 60, "z": 4, "nx": 0, "ny": 0, "nz": 1, "local_y": 0.2}
        game._place_block()

        placed_side = game.world.get_block(6, 60, 5)
        self.assertTrue(is_trapdoor(placed_side))
        info_side = get_trapdoor_info(placed_side)
        self.assertEqual(info_side["facing"], "north")
        self.assertEqual(info_side["half"], "bottom")

    def test_trapdoor_toggle_open_and_close(self):
        game = Game.__new__(Game)
        game.world = self._create_world()
        game.network_client = None
        game.gamemode = "survival"
        game.hotbar = []
        game.selected = 0
        game.player = MagicMock(pos=[5.0, 60.0, 8.0], eye=1.62, yaw=0.0, pitch=0.0, gamemode="survival")
        game._upload_block_change = MagicMock()
        game.sound_manager = MagicMock()

        game.world.set_block(5, 60, 5, "wooden_trapdoor_bottom_north")
        game.target = {"x": 5, "y": 60, "z": 5}
        game._aim = MagicMock()

        # Right-click closed trapdoor -> opens
        game._on_right_click()
        info_open = get_trapdoor_info(game.world.get_block(5, 60, 5))
        self.assertTrue(info_open["open"])

        # Right-click open trapdoor -> closes
        game._on_right_click()
        info_closed = get_trapdoor_info(game.world.get_block(5, 60, 5))
        self.assertFalse(info_closed["open"])

    def test_trapdoor_breaking_and_drops(self):
        game = Game.__new__(Game)
        game.world = self._create_world()
        game.network_client = None
        game.gamemode = "survival"
        game.hotbar = []
        game.selected = 0
        game.player = MagicMock(pos=[5.0, 60.0, 8.0], eye=1.62, yaw=0.0, pitch=0.0, gamemode="survival")
        game._upload_block_change = MagicMock()
        game.sound_manager = MagicMock()
        game._spawn_break_drops = MagicMock()
        game._reset_breaking = MagicMock()

        game.world.set_block(5, 60, 5, "wooden_trapdoor_bottom_north")
        game.target = {"x": 5, "y": 60, "z": 5}
        game._aim = MagicMock()
        game._break_block()

        self.assertEqual(game.world.get_block(5, 60, 5), AIR)
        game._spawn_break_drops.assert_called_once_with([("wooden_trapdoor", 1)], 5, 60.5, 5)

    def test_trapdoor_lighting_and_textures(self):
        # Opacity must be 0 so light freely transmits through trapdoor
        self.assertEqual(get_block_opacity(BLOCKS["wooden_trapdoor"]), 0)
        self.assertEqual(get_block_opacity(BLOCKS["wooden_trapdoor_bottom_north"]), 0)
        self.assertEqual(get_block_opacity(BLOCKS["wooden_trapdoor_top_south_open"]), 0)

        # Inspect UVs of faces
        closed_td = BLOCKS["wooden_trapdoor_bottom_north"]
        faces = closed_td.elements[0]["faces"]

        uv_up = faces[FACE_INDEX["up"]]
        uv_north = faces[FACE_INDEX["north"]]

        # Up face is the large face (16x16), north is thin 16x3 edge -> different UV mapping
        self.assertNotEqual(uv_up, uv_north)


if __name__ == "__main__":
    unittest.main()

