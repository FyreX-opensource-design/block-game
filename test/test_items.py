import unittest

import numpy as np

from client.inventory import CreativeInventory
from server.blocks import GRASS, HOTBAR_SLOTS
from server.icons import render_block_icon
from server.images import read_png
from server.items import ITEMS, creative_items
from server.stacks import empty_slots, make_stack, stack_count, stack_id


class ItemTests(unittest.TestCase):
    def test_block_items_place_and_tools_are_kept_for_later(self):
        grass = ITEMS[GRASS]
        self.assertEqual(grass.model, "block")
        self.assertEqual(grass.right, "place")
        self.assertEqual(grass.left, "break")
        self.assertEqual(grass.max_stack, 64)
        self.assertEqual(grass.block, GRASS)
        pick = ITEMS["wood_pickaxe"]
        self.assertEqual(pick.left, "mine")
        self.assertEqual(pick.right, "none")
        self.assertEqual(pick.durability, 59)
        self.assertEqual(pick.mining_level, 1)
        self.assertEqual(pick.repair, "stick")
        self.assertEqual(pick.max_stack, 1)
        self.assertEqual(stack_count(make_stack("wood_pickaxe", 8)), 1)
        self.assertEqual(stack_id(make_stack("stick", 3)), "stick")
        self.assertIn("wood_pickaxe", creative_items("tools"))
        self.assertNotIn("wood_pickaxe", creative_items("building"))

    def test_block_icons_are_isometric(self):
        top = read_png("assets/textures/grass_top.png")
        side = read_png("assets/textures/grass_side.png")
        icon = render_block_icon(top, side)
        self.assertEqual(int(icon[0, 0, 3]), 0)
        self.assertEqual(int(icon[0, -1, 3]), 0)
        opaque = icon[:, :, 3] > 200
        self.assertGreater(int(opaque.sum()), 80)
        self.assertLess(int(opaque.sum()), 32 * 32 - 8)
        self.assertTrue(bool((icon[:, :, 1] > icon[:, :, 0] + 15).any()))
        self.assertTrue(bool((icon[:, :, 0] > icon[:, :, 1] + 10).any()))
        # Solid cube: the three projected faces leave almost no empty pixels near the center.
        self.assertGreater(int(opaque.sum()), 600)
        self.assertEqual(int(icon[16, 16, 3]), 255)
        self.assertEqual(int(icon[10, 16, 3]), 255)
        self.assertEqual(int(icon[22, 16, 3]), 255)

    def test_tools_show_up_in_the_creative_inventory(self):
        inventory = CreativeInventory()
        inventory.apply("tab:tools", "left", empty_slots(HOTBAR_SLOTS), 0)
        frame = inventory.layout(1280, 720, empty_slots(HOTBAR_SLOTS), 0, (0, 0))
        actions = {hit.action for hit in frame.hits}
        self.assertIn("item:wood_pickaxe", actions)
        self.assertIn("item:stick", actions)
        self.assertIn("item:debug_axes", actions)
        self.assertEqual(frame.labels[0].text, "Tools")
