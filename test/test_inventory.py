import unittest

from client.inventory import CreativeInventory
from server.blocks import HOTBAR_SLOTS, LOG
from server.recipes import CRAFTING_SHAPELESS, Recipe, craft_match, match_grid, recipe_from_mapping, recipes_of
from server.stacks import (
    INVENTORY_SLOTS,
    empty_slots,
    make_stack,
    normalize_slots,
    pack_stack,
    stack_count,
    stack_id,
)


class InventoryTests(unittest.TestCase):
    def test_log_in_the_crafting_grid_makes_four_planks(self):
        craft = empty_slots(4)
        craft[0] = make_stack(LOG, 2)
        matched = craft_match(tuple(stack_id(slot) for slot in craft))
        self.assertEqual(matched[0], "planks")
        self.assertEqual(matched[1], 4)
        self.assertEqual(recipes_of(CRAFTING_SHAPELESS)[0].type, CRAFTING_SHAPELESS)
        later = recipe_from_mapping({"id": "cook_log", "type": "smelting", "result": "planks"}, "memory")
        self.assertEqual(later.type, "smelting")
        self.assertIsNone(match_grid(("log", "", "", ""), (later,)))

        inventory = CreativeInventory()
        storage = empty_slots(INVENTORY_SLOTS)
        hotbar = empty_slots(HOTBAR_SLOTS)
        inventory.apply("result:out", "left", hotbar, 0, storage, craft)
        self.assertEqual(stack_id(craft[0]), LOG)
        self.assertEqual(stack_count(craft[0]), 1)
        self.assertEqual(inventory.held, "planks")
        self.assertEqual(inventory.held_count, 4)

        inventory.apply("store:26", "left", hotbar, 0, storage, craft)
        self.assertEqual(stack_id(storage[26]), "planks")
        self.assertEqual(stack_count(storage[26]), 4)
        self.assertEqual(inventory.held, "")
        self.assertEqual(pack_stack(storage[26]), {"id": "planks", "count": 4})
        restored = normalize_slots([pack_stack(storage[26])], 1)
        self.assertEqual(stack_count(restored[0]), 4)

    def test_player_tab_has_storage_and_a_crafting_grid(self):
        inventory = CreativeInventory()
        inventory.tab = "inventory"
        storage = empty_slots(INVENTORY_SLOTS)
        craft = empty_slots(4)
        craft[3] = make_stack(LOG, 1)
        frame = inventory.layout(1280, 720, empty_slots(HOTBAR_SLOTS), 0, (0, 0), storage, craft)
        actions = {hit.action for hit in frame.hits}
        for index in range(INVENTORY_SLOTS):
            self.assertIn(f"store:{index}", actions)
        for index in range(4):
            self.assertIn(f"craft:{index}", actions)
        self.assertIn("result:out", actions)
        self.assertEqual(frame.labels[0].text, "Inventory")
        self.assertEqual(stack_id(inventory.preview), "planks")
        self.assertIn("4", [label.text for label in frame.labels])

    def test_creative_block_can_be_stored_past_the_hotbar(self):
        inventory = CreativeInventory()
        storage = empty_slots(INVENTORY_SLOTS)
        hotbar = empty_slots(HOTBAR_SLOTS)
        inventory.apply("item:stone", "left", hotbar, 0, storage)
        inventory.apply("store:0", "left", hotbar, 0, storage)
        self.assertEqual(stack_id(storage[0]), "stone")
        self.assertTrue(inventory.held_infinite)
        inventory.dismiss_held(hotbar, storage, empty_slots(4))
        self.assertEqual(inventory.held, "")
        inventory.apply("store:0", "left", hotbar, 0, storage)
        inventory.apply("hotbar:3", "left", hotbar, 0, storage)
        self.assertEqual(stack_id(hotbar[3]), "stone")
        self.assertEqual(stack_id(storage[0]), "")

    def test_shaped_recipe_matches_either_direction(self):
        recipe = Recipe("stones", True, (("stone", "cobble"),), (), "cobble", 1)
        across = match_grid(("", "", "stone", "cobble"), (recipe,))
        flipped = match_grid(("cobble", "stone", "", ""), (recipe,))
        self.assertEqual(across[2], [2, 3])
        self.assertEqual(flipped[2], [1, 0])
        self.assertIsNone(match_grid(("stone", "cobble", "dirt", ""), (recipe,)))

    def test_full_cursor_does_not_consume_the_log(self):
        inventory = CreativeInventory()
        inventory.held = "planks"
        inventory.held_count = 61
        inventory.held_infinite = False
        craft = empty_slots(4)
        craft[1] = make_stack(LOG, 1)
        inventory.apply("result:out", "left", empty_slots(HOTBAR_SLOTS), 0, empty_slots(INVENTORY_SLOTS), craft)
        self.assertEqual(stack_count(craft[1]), 1)
        self.assertEqual(inventory.held_count, 61)

    def test_shift_left_click_creative_item_takes_full_stack(self):
        inventory = CreativeInventory()
        storage = empty_slots(INVENTORY_SLOTS)
        hotbar = empty_slots(HOTBAR_SLOTS)
        inventory.apply("item:stone", "left", hotbar, 0, storage, shift=True)
        self.assertEqual(stack_id(hotbar[0]), "stone")
        self.assertEqual(stack_count(hotbar[0]), 64)
        self.assertEqual(inventory.held, "")

        # A second shift-click takes another stack of 64 into hotbar[1]
        inventory.apply("item:dirt", "left", hotbar, 0, storage, shift=True)
        self.assertEqual(stack_id(hotbar[1]), "dirt")
        self.assertEqual(stack_count(hotbar[1]), 64)

    def test_increment_held_stack_in_creative_menu(self):
        inventory = CreativeInventory()
        storage = empty_slots(INVENTORY_SLOTS)
        hotbar = empty_slots(HOTBAR_SLOTS)

        # Initial left-click picks up item with count 1
        inventory.apply("item:stone", "left", hotbar, 0, storage)
        self.assertEqual(inventory.held, "stone")
        self.assertEqual(inventory.held_count, 1)

        # Subsequent left-clicks on the same item increment count by 1
        inventory.apply("item:stone", "left", hotbar, 0, storage)
        self.assertEqual(inventory.held, "stone")
        self.assertEqual(inventory.held_count, 2)
        self.assertFalse(inventory.held_infinite)

        inventory.apply("item:stone", "left", hotbar, 0, storage)
        self.assertEqual(inventory.held_count, 3)

        # Right clicking the same item also increments count by 1
        inventory.apply("item:stone", "right", hotbar, 0, storage)
        self.assertEqual(inventory.held_count, 4)

    def test_shift_right_click_creative_item_holds_full_stack(self):
        inventory = CreativeInventory()
        storage = empty_slots(INVENTORY_SLOTS)
        hotbar = empty_slots(HOTBAR_SLOTS)

        # Shift + Right Click puts 64 directly onto the cursor
        inventory.apply("item:stone", "right", hotbar, 0, storage, shift=True)
        self.assertEqual(inventory.held, "stone")
        self.assertEqual(inventory.held_count, 64)
        self.assertFalse(inventory.held_infinite)
        # Hotbar should remain unaffected
        self.assertEqual(stack_id(hotbar[0]), "")

    def test_shift_click_hotbar_and_storage(self):
        inventory = CreativeInventory()
        storage = empty_slots(INVENTORY_SLOTS)
        hotbar = empty_slots(HOTBAR_SLOTS)
        hotbar[0] = make_stack("stone", 10)

        # Shift click from hotbar moves to storage
        inventory.apply("hotbar:0", "left", hotbar, 0, storage, shift=True)
        self.assertEqual(stack_id(hotbar[0]), "")
        self.assertEqual(stack_id(storage[0]), "stone")
        self.assertEqual(stack_count(storage[0]), 10)

        # Shift click from storage moves back to hotbar
        inventory.apply("store:0", "left", hotbar, 0, storage, shift=True)
        self.assertEqual(stack_id(storage[0]), "")
        self.assertEqual(stack_id(hotbar[0]), "stone")
        self.assertEqual(stack_count(hotbar[0]), 10)

    def test_shift_click_furnace_interactions(self):
        from server.furnace import Furnace

        inventory = CreativeInventory()
        storage = empty_slots(INVENTORY_SLOTS)
        hotbar = empty_slots(HOTBAR_SLOTS)
        furnace = Furnace()

        hotbar[0] = make_stack("cobble", 5)
        hotbar[1] = make_stack("planks", 3)

        # Shift-clicking cobble moves to furnace input
        inventory.apply("hotbar:0", "left", hotbar, 0, storage, furnace=furnace, shift=True)
        self.assertEqual(stack_id(hotbar[0]), "")
        self.assertEqual(stack_id(furnace.input), "cobble")
        self.assertEqual(stack_count(furnace.input), 5)

        # Shift-clicking planks moves to furnace fuel
        inventory.apply("hotbar:1", "left", hotbar, 0, storage, furnace=furnace, shift=True)
        self.assertEqual(stack_id(hotbar[1]), "")
        self.assertEqual(stack_id(furnace.fuel), "planks")
        self.assertEqual(stack_count(furnace.fuel), 3)

        # Shift-clicking furnace output transfers to hotbar
        furnace.output = make_stack("stone", 2)
        inventory.apply("furnace:output", "left", hotbar, 0, storage, furnace=furnace, shift=True)
        self.assertEqual(stack_id(furnace.output), "")
        self.assertEqual(stack_id(hotbar[0]), "stone")
        self.assertEqual(stack_count(hotbar[0]), 2)

    def test_gamemode_switch_to_creative_uses_creative_tab(self):
        inventory = CreativeInventory()
        storage = empty_slots(INVENTORY_SLOTS)
        craft = empty_slots(4)
        hotbar = empty_slots(HOTBAR_SLOTS)

        frame = inventory.layout(1280, 720, hotbar, 0, (0, 0), storage, craft, gamemode="survival")
        self.assertEqual(inventory.tab, "inventory")
        self.assertIn("store:0", {hit.action for hit in frame.hits})

        frame = inventory.layout(1280, 720, hotbar, 0, (0, 0), storage, craft, gamemode="creative")
        self.assertEqual(inventory.tab, "building")
        actions = {hit.action for hit in frame.hits}
        self.assertIn("tab:building", actions)
        self.assertIn("item:planks", actions)

    def test_gamemode_switch_back_to_survival_uses_player_tab(self):
        inventory = CreativeInventory()
        hotbar = empty_slots(HOTBAR_SLOTS)
        frame = inventory.layout(1280, 720, hotbar, 0, (0, 0), gamemode="survival")
        self.assertEqual(inventory.tab, "inventory")

        frame = inventory.layout(1280, 720, hotbar, 0, (0, 0), gamemode="creative")
        self.assertEqual(inventory.tab, "building")

        frame = inventory.layout(1280, 720, hotbar, 0, (0, 0), gamemode="survival")
        self.assertEqual(inventory.tab, "inventory")
        actions = {hit.action for hit in frame.hits}
        self.assertIn("store:0", actions)
        self.assertIn("craft:0", actions)
