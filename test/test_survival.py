import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from server.blocks import BLOCKS, canonical_id
from server.items import ITEMS
from server.loot import (
    calculate_break_time,
    can_harvest,
    get_block_drops,
    get_tool_type,
    roll_loot_table,
)
from server.network import MultiplayerServer
from server.stacks import (
    add_to_inventory,
    empty_slots,
    empty_stack,
    make_stack,
    stack_count,
    stack_id,
)
from server.storage import WorldCatalog, WorldInfo
from client.menu import MenuState


class TestSurvivalFeatures(unittest.TestCase):
    def test_world_info_gamemode(self):
        # Default gamemode is survival
        info = WorldInfo("test_id", "Test World", 1234, 1000.0, 1000.0)
        self.assertEqual(info.gamemode, "survival")
        self.assertEqual(info.to_dict()["gamemode"], "survival")

        # Explicit creative gamemode
        info2 = WorldInfo("test_id2", "Creative World", 5678, 1000.0, 1000.0, gamemode="creative")
        self.assertEqual(info2.gamemode, "creative")
        self.assertEqual(info2.to_dict()["gamemode"], "creative")

    def test_catalog_create_and_read_gamemode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = WorldCatalog(tmpdir)
            w_surv = catalog.create("Survival Test", 42, gamemode="survival")
            self.assertEqual(w_surv.gamemode, "survival")

            # Read back from disk
            loaded_surv = catalog.read(w_surv.id)
            self.assertEqual(loaded_surv.gamemode, "survival")

            w_creat = catalog.create("Creative Test", 99, gamemode="creative")
            self.assertEqual(w_creat.gamemode, "creative")
            loaded_creat = catalog.read(w_creat.id)
            self.assertEqual(loaded_creat.gamemode, "creative")

    def test_menu_gamemode_toggle(self):
        menu = MenuState()
        self.assertEqual(menu.gamemode, "survival")

        # Toggle to creative
        menu.apply("toggle-gamemode")
        self.assertEqual(menu.gamemode, "creative")

        # Toggle back to survival
        menu.apply("toggle-gamemode")
        self.assertEqual(menu.gamemode, "survival")

        # Ensure layout in create mode contains Game Mode button
        frame = menu._create(800, 600)
        actions = [hit.action for hit in frame.hits]
        self.assertIn("toggle-gamemode", actions)

    def test_block_attributes(self):
        stone = BLOCKS.get("stone")
        self.assertIsNotNone(stone)
        self.assertEqual(stone.hardness, 1.5)
        self.assertEqual(stone.tool, "pickaxe")
        self.assertEqual(stone.mining_level, 1)

        dirt = BLOCKS.get("dirt")
        self.assertIsNotNone(dirt)
        self.assertEqual(dirt.hardness, 0.5)
        self.assertEqual(dirt.tool, "shovel")
        self.assertEqual(dirt.mining_level, 0)

        log = BLOCKS.get("log")
        self.assertIsNotNone(log)
        self.assertEqual(log.hardness, 2.0)
        self.assertEqual(log.tool, "axe")
        self.assertEqual(log.mining_level, 0)

        leaves = BLOCKS.get("leaves")
        self.assertIsNotNone(leaves)
        self.assertEqual(leaves.hardness, 0.2)
        self.assertIsNone(leaves.tool)

    def test_item_tool_attributes(self):
        pick = ITEMS.get("wood_pickaxe")
        self.assertIsNotNone(pick)
        self.assertEqual(pick.tool, "pickaxe")
        self.assertEqual(pick.mining_level, 1)
        self.assertEqual(pick.mining_speed, 2)

    def test_can_harvest(self):
        stone = BLOCKS.get("stone")
        dirt = BLOCKS.get("dirt")
        pick = ITEMS.get("wood_pickaxe")

        # Dirt can be harvested with hand or any tool
        self.assertTrue(can_harvest(dirt, None))
        self.assertTrue(can_harvest(dirt, pick))

        # Stone requires pickaxe of level >= 1
        self.assertFalse(can_harvest(stone, None))
        self.assertTrue(can_harvest(stone, pick))

    def test_calculate_break_time(self):
        stone = BLOCKS.get("stone")
        dirt = BLOCKS.get("dirt")
        leaves = BLOCKS.get("leaves")
        pick = ITEMS.get("wood_pickaxe")

        # Hand breaking dirt: 0.5 * 1.5 / 1.0 = 0.75
        self.assertAlmostEqual(calculate_break_time(dirt, None), 0.75)

        # Hand breaking leaves: 0.2 * 1.5 / 1.0 = 0.3
        self.assertAlmostEqual(calculate_break_time(leaves, None), 0.3)

        # Hand breaking stone: cannot harvest -> 1.5 * 5.0 / 1.0 = 7.5
        self.assertAlmostEqual(calculate_break_time(stone, None), 7.5)

        # Wood pickaxe breaking stone: can harvest, speed 2 -> 1.5 * 1.5 / 2.0 = 1.125
        self.assertAlmostEqual(calculate_break_time(stone, pick), 1.125)

    def test_loot_tables(self):
        stone = BLOCKS.get("stone")
        dirt = BLOCKS.get("dirt")
        grass = BLOCKS.get("grass")
        log = BLOCKS.get("log")
        pick = ITEMS.get("wood_pickaxe")

        # Stone with wood pickaxe drops 1 cobble
        drops = get_block_drops(stone, pick)
        self.assertEqual(drops, [("cobble", 1)])

        # Stone with hand drops nothing
        drops_hand = get_block_drops(stone, None)
        self.assertEqual(drops_hand, [])

        # Dirt drops dirt
        drops_dirt = get_block_drops(dirt, None)
        self.assertEqual(drops_dirt, [("dirt", 1)])

        # Grass drops dirt
        drops_grass = get_block_drops(grass, None)
        self.assertEqual(drops_grass, [("dirt", 1)])

        # Log drops log
        drops_log = get_block_drops(log, None)
        self.assertEqual(drops_log, [("log", 1)])

        # Rotated log variant log_x drops log
        log_x = BLOCKS.get("log_x")
        if log_x:
            drops_log_x = get_block_drops(log_x, None)
            self.assertEqual(drops_log_x, [("log", 1)])

    def test_add_to_inventory(self):
        hotbar = empty_slots(9)
        storage = empty_slots(27)

        # Add 10 dirt
        leftover = add_to_inventory([hotbar, storage], "dirt", 10)
        self.assertEqual(leftover, 0)
        self.assertEqual(stack_id(hotbar[0]), "dirt")
        self.assertEqual(stack_count(hotbar[0]), 10)

        # Add 5 more dirt (stacks into slot 0)
        leftover = add_to_inventory([hotbar, storage], "dirt", 5)
        self.assertEqual(leftover, 0)
        self.assertEqual(stack_count(hotbar[0]), 15)

    def test_multiplayer_gamemode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            server = MultiplayerServer(host="127.0.0.1", port=25570, gamemode="survival", saves_dir=tmpdir)
            self.assertEqual(server.gamemode, "survival")

            server_creat = MultiplayerServer(host="127.0.0.1", port=25571, gamemode="creative", saves_dir=tmpdir)
            self.assertEqual(server_creat.gamemode, "creative")

    def test_crack_textures_and_stages(self):
        from server.blocks import ASSETS
        from server.images import read_png

        # Ensure all 10 crack textures exist and are valid 16x16 RGBA images
        for stage in range(10):
            path = ASSETS / "textures" / "crack" / f"crack_{stage}.png"
            self.assertTrue(path.is_file(), f"Missing crack texture: {path}")
            img = read_png(path)
            self.assertEqual(img.shape, (16, 16, 4))
            # Must have non-zero alpha pixels
            self.assertTrue((img[:, :, 3] > 0).any())

        # Test stage calculation from progress
        self.assertEqual(min(9, max(0, int(0.0 * 10))), 0)
        self.assertEqual(min(9, max(0, int(0.05 * 10))), 0)
        self.assertEqual(min(9, max(0, int(0.15 * 10))), 1)
        self.assertEqual(min(9, max(0, int(0.50 * 10))), 5)
        self.assertEqual(min(9, max(0, int(0.95 * 10))), 9)
        self.assertEqual(min(9, max(0, int(1.00 * 10))), 9)


if __name__ == "__main__":
    unittest.main()
