import os
import shutil
import tempfile
import threading
import time
import unittest

from client.inventory import CreativeInventory
from client.settings import DEFAULT_SETTINGS, load_settings, save_settings
from client.web_ui import WebCompanion
from server.furnace import Furnace
from server.stacks import CRAFT_SLOTS, HOTBAR_SLOTS, INVENTORY_SLOTS, empty_slots, make_stack, stack_count, stack_id


class DummyGame:
    def __init__(self):
        self.lock = threading.RLock()
        self.world = object()
        self.gamemode = "survival"
        self.mode = "play"
        self.selected = 0
        self.hotbar = empty_slots(HOTBAR_SLOTS)
        self.storage = empty_slots(INVENTORY_SLOTS)
        self.craft = empty_slots(CRAFT_SLOTS)
        self.inventory = CreativeInventory()

    def _capture_furnace_slots(self):
        return None

    def _sync_furnace_changes(self, prev):
        pass


class WebCompanionTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.game = DummyGame()
        self.companion = WebCompanion(self.game, host="127.0.0.1", port=0)
        self.client = self.companion.app.test_client()

    def tearDown(self):
        self.companion.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_settings_load_and_save(self):
        settings = load_settings(self.tmp_dir)
        self.assertEqual(settings["web_ui_enabled"], False)
        self.assertEqual(settings["web_ui_port"], 5000)

        settings["web_ui_enabled"] = True
        settings["web_ui_port"] = 8080
        save_settings(settings, self.tmp_dir)

        loaded = load_settings(self.tmp_dir)
        self.assertEqual(loaded["web_ui_enabled"], True)
        self.assertEqual(loaded["web_ui_port"], 8080)

    def test_ptp_sync_and_delay_calculation(self):
        # 1. Sync
        t1_client = time.monotonic()
        res_sync = self.client.get("/api/ptp/sync")
        self.assertEqual(res_sync.status_code, 200)
        t2_client = time.monotonic()
        data_sync = res_sync.get_json()
        self.assertIn("t1", data_sync)
        t1_master = data_sync["t1"]

        # 2. Delay
        t3_client = time.monotonic()
        res_delay = self.client.post("/api/ptp/delay", json={"t3": t3_client})
        self.assertEqual(res_delay.status_code, 200)
        t4_client = time.monotonic()
        data_delay = res_delay.get_json()
        self.assertIn("t4", data_delay)
        t4_master = data_delay["t4"]

        # PTP IEEE 1588 math:
        delay = ((t4_master - t1_master) - (t3_client - t2_client)) / 2.0
        offset = ((t2_client - t1_master) - (t4_master - t3_client)) / 2.0

        # Since master and slave run in the same process, offset and delay should be close to 0
        self.assertAlmostEqual(offset, 0.0, delta=0.5)
        self.assertGreaterEqual(delay, -0.05)

    def test_get_state_default_inventory(self):
        self.game.hotbar[0] = make_stack("planks", 10)
        self.game.storage[1] = make_stack("cobble", 5)

        res = self.client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["gamemode"], "survival")
        self.assertEqual(data["container_type"], "player")
        self.assertEqual(data["hotbar"][0]["id"], "planks")
        self.assertEqual(data["hotbar"][0]["count"], 10)
        self.assertEqual(data["storage"][1]["id"], "cobble")
        self.assertEqual(data["storage"][1]["count"], 5)
        self.assertEqual(len(data["craft"]), 4)

    def test_container_switching_crafting_table(self):
        # In-game right click on crafting table
        self.game.inventory.container_type = "crafting_table"
        self.game.inventory.container_craft = empty_slots(9)
        self.game.inventory.container_craft[0] = make_stack("planks", 4)
        self.game.inventory.table_preview = make_stack("crafting_table", 1)

        res = self.client.get("/api/state")
        data = res.get_json()
        self.assertEqual(data["container_type"], "crafting_table")
        self.assertEqual(len(data["container_craft"]), 9)
        self.assertEqual(data["container_craft"][0]["id"], "planks")
        self.assertEqual(data["table_preview"]["id"], "crafting_table")

    def test_container_switching_furnace(self):
        # In-game right click on furnace
        furnace = Furnace(x=10, y=20, z=30, input_stack=make_stack("iron_ore", 3), fuel_stack=make_stack("coal", 2))
        furnace.burn_time = 5.0
        furnace.burn_duration = 10.0
        furnace.cook_time = 4.0
        furnace.cook_duration = 10.0
        self.game.inventory.container_type = "furnace"
        self.game.inventory.furnace = furnace

        res = self.client.get("/api/state")
        data = res.get_json()
        self.assertEqual(data["container_type"], "furnace")
        self.assertIsNotNone(data["furnace"])
        self.assertEqual(data["furnace"]["input"]["id"], "iron_ore")
        self.assertEqual(data["furnace"]["input"]["count"], 3)
        self.assertEqual(data["furnace"]["fuel"]["id"], "coal")
        self.assertAlmostEqual(data["furnace"]["burn_progress"], 0.5)
        self.assertAlmostEqual(data["furnace"]["cook_progress"], 0.4)
        self.assertTrue(data["furnace"]["is_burning"])

    def test_item_movement_via_action(self):
        self.game.hotbar[0] = make_stack("planks", 10)

        # 1. Click slot to pick up item
        res = self.client.post("/api/action", json={"action": "hotbar:0", "button": "left", "shift": False})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.game.inventory.held, "planks")
        self.assertEqual(self.game.inventory.held_count, 10)
        self.assertEqual(stack_count(self.game.hotbar[0]), 0)

        # 2. Place into storage slot 5
        res = self.client.post("/api/action", json={"action": "store:5", "button": "left", "shift": False})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.game.inventory.held, "")
        self.assertEqual(stack_id(self.game.storage[5]), "planks")
        self.assertEqual(stack_count(self.game.storage[5]), 10)

    def test_crafting_in_web_ui(self):
        # Place 1 log in craft slot 0
        self.game.craft[0] = make_stack("log", 1)
        self.game.inventory.preview = self.game.inventory._preview(self.game.craft)
        self.assertEqual(stack_id(self.game.inventory.preview), "planks")
        self.assertEqual(stack_count(self.game.inventory.preview), 4)

        # Take craft result via action
        res = self.client.post("/api/action", json={"action": "result:out", "button": "left", "shift": False})
        self.assertEqual(res.status_code, 200)
        # Log consumed, planks in held cursor
        self.assertEqual(self.game.inventory.held, "planks")
        self.assertEqual(self.game.inventory.held_count, 4)
        self.assertEqual(stack_count(self.game.craft[0]), 0)

    def test_icon_endpoint(self):
        # Check an item icon
        res = self.client.get("/api/icon/planks")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "image/png")
        self.assertTrue(res.data.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_index_serves_html(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Block Game Companion", res.data)
        self.assertIn(b"PTP", res.data)

    def test_menu_options_layout_and_actions(self):
        from client.menu import MenuState
        menu = MenuState()
        menu.screen = "title"

        action = menu.apply("options")
        self.assertEqual(action, "options")
        self.assertEqual(menu.screen, "options")

        frame = menu.layout(800, 600)
        self.assertGreater(len(frame.rects), 0)
        self.assertGreater(len(frame.labels), 0)
        label_texts = [lbl.text for lbl in frame.labels]
        self.assertIn("Options", label_texts)
        self.assertIn("Web UI: OFF", label_texts)

        # Toggle Web UI
        toggle_action = menu.apply("toggle-web-ui")
        self.assertEqual(toggle_action, "toggle-web-ui")
        self.assertTrue(menu.web_ui_enabled)

        # Test IP address display in options menu
        menu.local_ips = ["192.168.1.100", "10.0.0.5"]
        frame_ip = menu.layout(800, 600)
        label_ip_texts = [lbl.text for lbl in frame_ip.labels]
        self.assertIn("Web UI: ON", label_ip_texts)
        self.assertIn("Phone / Tablet URL:", label_ip_texts)
        self.assertIn("http://192.168.1.100:5000", label_ip_texts)

        # Back
        back_action = menu.apply("back-options")
        self.assertEqual(back_action, "back-options")
        self.assertEqual(menu.screen, "title")

    def test_touch_creative_item_and_slot_placement(self):
        self.game.gamemode = "creative"
        self.game.selected = 3

        # 1. Tap creative item with touch and selected_hotbar:
        # Equips full stack into hotbar[3] and does NOT leave item stuck in held
        res = self.client.post("/api/action", json={
            "action": "item:cobble_slab",
            "touch": True,
            "selected_hotbar": 3
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(stack_id(self.game.hotbar[3]), "cobble_slab")
        self.assertEqual(stack_count(self.game.hotbar[3]), 64)
        self.assertEqual(self.game.inventory.held, "")

        # 2. Select hotbar slot action
        res = self.client.post("/api/action", json={"action": "select_hotbar:5"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.game.selected, 5)

        # 3. Creative drag-and-drop to storage slot 8
        res = self.client.post("/api/action", json={
            "action": "drop_creative",
            "item_id": "stone",
            "target": "store:8"
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(stack_id(self.game.storage[8]), "stone")
        self.assertEqual(stack_count(self.game.storage[8]), 64)
        self.assertEqual(self.game.inventory.held, "")

        # 4. If an item was in held_infinite, placing it in a slot clears held
        self.game.inventory.held = "planks"
        self.game.inventory.held_count = 1
        self.game.inventory.held_infinite = True

        res = self.client.post("/api/action", json={"action": "hotbar:1", "button": "left"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(stack_id(self.game.hotbar[1]), "planks")
        self.assertEqual(stack_count(self.game.hotbar[1]), 64)
        self.assertEqual(self.game.inventory.held, "")
        self.assertFalse(self.game.inventory.held_infinite)

    def test_state_when_not_in_world(self):
        self.game.world = None
        self.game.mode = "title"

        res = self.client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertFalse(data["in_world"])
        self.assertEqual(data["game_status"], "In Main Menu")

        # Action is ignored when not in world
        self.game.hotbar[0] = make_stack("planks", 10)
        res_act = self.client.post("/api/action", json={"action": "hotbar:0", "button": "left"})
        self.assertEqual(res_act.status_code, 200)
        self.assertEqual(stack_count(self.game.hotbar[0]), 10)
        self.assertEqual(self.game.inventory.held, "")
