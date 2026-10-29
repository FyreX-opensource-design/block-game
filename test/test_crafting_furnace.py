import unittest

from client.inventory import CreativeInventory
from server.blocks import CRAFTING_TABLE, FURNACE
from server.furnace import Furnace
from server.items import ITEMS
from server.recipes import (
    CRAFTING_SHAPED,
    CRAFTING_SHAPELESS,
    SMELTING,
    RECIPES,
    craft_match,
    fuel_burn_time,
    recipes_of,
    smelt_match,
)
from server.stacks import empty_slots, empty_stack, make_stack, stack_count, stack_id
from server.world import World


class CraftingAndFurnaceTests(unittest.TestCase):
    def test_blocks_and_items_registered(self):
        self.assertIn("crafting_table", ITEMS)
        self.assertIn("furnace", ITEMS)
        self.assertEqual(ITEMS["crafting_table"].block, "crafting_table")
        self.assertEqual(ITEMS["furnace"].block, "furnace")

    def test_recipe_types_and_definitions(self):
        shaped = recipes_of(CRAFTING_SHAPED)
        shapeless = recipes_of(CRAFTING_SHAPELESS)
        smelting = recipes_of(SMELTING)

        shaped_ids = {r.id for r in shaped}
        self.assertIn("crafting_table", shaped_ids)
        self.assertIn("furnace", shaped_ids)
        self.assertIn("stick", shaped_ids)
        self.assertIn("wood_pickaxe", shaped_ids)

        smelting_ids = {r.id for r in smelting}
        self.assertIn("stone", smelting_ids)

    def test_crafting_table_and_sticks_in_2x2_and_3x3(self):
        # 2x2 stick
        res = craft_match(("planks", "", "planks", ""))
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "stick")
        self.assertEqual(res[1], 4)

        # 2x2 crafting table
        res = craft_match(("planks", "planks", "planks", "planks"))
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "crafting_table")
        self.assertEqual(res[1], 1)

        # 3x3 sticks in corner
        grid_3x3 = [""] * 9
        grid_3x3[0] = "planks"
        grid_3x3[3] = "planks"
        res = craft_match(tuple(grid_3x3))
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "stick")
        self.assertEqual(res[1], 4)
        self.assertEqual(res[2], [0, 3])

    def test_furnace_3x3_crafting(self):
        grid = (
            "cobble", "cobble", "cobble",
            "cobble", "",       "cobble",
            "cobble", "cobble", "cobble",
        )
        res = craft_match(grid)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "furnace")
        self.assertEqual(res[1], 1)
        self.assertEqual(res[2], [0, 1, 2, 3, 5, 6, 7, 8])

    def test_wood_pickaxe_3x3_crafting(self):
        grid = (
            "planks", "planks", "planks",
            "",       "stick",  "",
            "",       "stick",  "",
        )
        res = craft_match(grid)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "wood_pickaxe")
        self.assertEqual(res[1], 1)
        self.assertEqual(res[2], [0, 1, 2, 4, 7])

    def test_smelting_cobblestone_into_stone(self):
        res = smelt_match("cobble")
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "stone")
        self.assertEqual(res[1], 1)
        self.assertEqual(res[2], 10.0)

    def test_fuel_burn_times(self):
        self.assertEqual(fuel_burn_time("planks"), 15.0)
        self.assertEqual(fuel_burn_time("log"), 15.0)
        self.assertEqual(fuel_burn_time("crafting_table"), 15.0)
        self.assertEqual(fuel_burn_time("wood_pickaxe"), 10.0)
        self.assertEqual(fuel_burn_time("stick"), 5.0)
        self.assertEqual(fuel_burn_time("cobble"), 0.0)

    def test_furnace_simulation_cooks_cobble_to_stone(self):
        furnace = Furnace(
            input_stack=make_stack("cobble", 2),
            fuel_stack=make_stack("planks", 1),
        )
        # 1s tick
        furnace.tick(1.0)
        self.assertTrue(furnace.is_lit)
        self.assertEqual(furnace.burn_duration, 15.0)
        self.assertEqual(furnace.cook_time, 1.0)
        self.assertEqual(stack_count(furnace.input), 2)
        self.assertEqual(stack_count(furnace.fuel), 0)  # 1 plank consumed to ignite

        # Advance another 9.0s -> cook_time reached 10.0s
        furnace.tick(9.0)
        self.assertEqual(stack_id(furnace.output), "stone")
        self.assertEqual(stack_count(furnace.output), 1)
        self.assertEqual(stack_count(furnace.input), 1)
        self.assertGreater(furnace.burn_time, 0.0)

    def test_crafting_table_ui_layout_and_crafting(self):
        inv = CreativeInventory()
        inv.container_type = "crafting_table"
        storage = empty_slots(27)
        hotbar = empty_slots(9)

        # Place 4 planks in 3x3 grid to make crafting table
        inv.container_craft[0] = make_stack("planks", 1)
        inv.container_craft[1] = make_stack("planks", 1)
        inv.container_craft[3] = make_stack("planks", 1)
        inv.container_craft[4] = make_stack("planks", 1)

        frame = inv.layout(1280, 720, hotbar, 0, (0, 0), storage, container_type="crafting_table")
        actions = {hit.action for hit in frame.hits}
        for i in range(9):
            self.assertIn(f"table_craft:{i}", actions)
        self.assertIn("table_result:out", actions)
        self.assertEqual(stack_id(inv.table_preview), "crafting_table")

        # Take result
        inv.apply("table_result:out", "left", hotbar, 0, storage)
        self.assertEqual(inv.held, "crafting_table")
        self.assertEqual(inv.held_count, 1)
        self.assertEqual(stack_count(inv.container_craft[0]), 0)

    def test_furnace_ui_layout_and_interaction(self):
        inv = CreativeInventory()
        inv.container_type = "furnace"
        furnace = Furnace(input_stack=make_stack("cobble", 5), fuel_stack=make_stack("stick", 3))
        inv.furnace = furnace
        storage = empty_slots(27)
        hotbar = empty_slots(9)

        frame = inv.layout(1280, 720, hotbar, 0, (0, 0), storage, container_type="furnace", furnace=furnace)
        actions = {hit.action for hit in frame.hits}
        self.assertIn("furnace:input", actions)
        self.assertIn("furnace:fuel", actions)
        self.assertIn("furnace:output", actions)

        # Test taking output when item is cooked
        furnace.output = make_stack("stone", 2)
        inv.apply("furnace:output", "left", hotbar, 0, storage, furnace=furnace)
        self.assertEqual(inv.held, "stone")
        self.assertEqual(inv.held_count, 2)
        self.assertEqual(stack_count(furnace.output), 0)

    def test_breaking_furnace_drops_stored_contents(self):
        world = World(seed=12345)
        world.generate_chunk(0, 4, 0)
        # Set furnace block at (1, 65, 1)
        world.set_block(1, 65, 1, FURNACE)
        furnace = world.get_furnace(1, 65, 1)
        self.assertIsNotNone(furnace)
        furnace.input = make_stack("cobble", 10)
        furnace.fuel = make_stack("planks", 5)
        furnace.output = make_stack("stone", 3)

        # Break furnace (set to air)
        world.set_block(1, 65, 1, "")
        self.assertIsNone(world.get_furnace(1, 65, 1))

        # Check dropped item entities
        dropped_items = {e.item: e.count for e in world.entities}
        self.assertEqual(dropped_items.get("cobble"), 10)
        self.assertEqual(dropped_items.get("planks"), 5)
        self.assertEqual(dropped_items.get("stone"), 3)

    def test_player_coordinates_and_inventory_dismissal(self):
        from server.player import Player

        world = World(seed=12345)
        player = Player(world)
        player.pos = [10.5, 65.0, 20.5]
        self.assertEqual(player.x, 10.5)
        self.assertEqual(player.y, 65.0)
        self.assertEqual(player.z, 20.5)
        player.x = 15.0
        self.assertEqual(player.pos[0], 15.0)

        inv = CreativeInventory()
        inv.container_type = "crafting_table"
        inv.container_craft[0] = make_stack("planks", 4)
        hotbar = empty_slots(9)
        storage = empty_slots(27)
        player_pos = tuple(player.pos)
        inv.dismiss_container(hotbar, storage, world, player_pos)
        self.assertEqual(inv.container_type, "player")
        self.assertEqual(stack_count(inv.container_craft[0]), 0)

    def test_creative_inventory_grid_expansion(self):
        inv = CreativeInventory()
        inv.tab = "building"
        hotbar = empty_slots(9)
        storage = empty_slots(27)
        frame = inv.layout(1280, 720, hotbar, 0, (0, 0), storage)
        # Should render 5 rows x 9 columns = 45 total slot cells
        slot_rects = [r for r in frame.rects if r[2] == 52 and r[3] == 52]
        # In addition to 9 hotbar slots, creative grid has 45 slots
        self.assertGreaterEqual(len(slot_rects), 45 + 9)

    def test_right_click_single_item_placement(self):
        inv = CreativeInventory()
        inv.held = "planks"
        inv.held_count = 10
        inv.held_infinite = False

        hotbar = empty_slots(9)
        storage = empty_slots(27)

        # Right-click on empty slot 0 -> places 1 item
        inv.apply("store:0", "right", hotbar, 0, storage)
        self.assertEqual(stack_id(storage[0]), "planks")
        self.assertEqual(stack_count(storage[0]), 1)
        self.assertEqual(inv.held_count, 9)

        # Right-click on same slot -> adds 1 more item
        inv.apply("store:0", "right", hotbar, 0, storage)
        self.assertEqual(stack_count(storage[0]), 2)
        self.assertEqual(inv.held_count, 8)

    def test_right_click_splits_stack_when_cursor_empty(self):
        inv = CreativeInventory()
        inv.clear_held()

        hotbar = empty_slots(9)
        storage = empty_slots(27)
        storage[0] = make_stack("cobble", 5)

        # Right-click on slot with 5 items -> takes 3 (ceil(5/2)), leaves 2
        inv.apply("store:0", "right", hotbar, 0, storage)
        self.assertEqual(inv.held, "cobble")
        self.assertEqual(inv.held_count, 3)
        self.assertEqual(stack_id(storage[0]), "cobble")
        self.assertEqual(stack_count(storage[0]), 2)

    def test_right_click_drag_distributes_single_items(self):
        inv = CreativeInventory()
        inv.container_type = "crafting_table"
        inv.held = "cobble"
        inv.held_count = 8
        inv.held_infinite = False

        hotbar = empty_slots(9)
        storage = empty_slots(27)

        # Drag over crafting table slots 0, 1, 2, 5, 8, 7, 6, 3 (a furnace ring)
        slots_to_drag = [0, 1, 2, 5, 8, 7, 6, 3]
        for slot_idx in slots_to_drag:
            action = f"table_craft:{slot_idx}"
            inv.apply(action, "right_drag", hotbar, 0, storage)

        self.assertEqual(inv.held_count, 0)
        self.assertEqual(inv.held, "")

        for slot_idx in slots_to_drag:
            self.assertEqual(stack_id(inv.container_craft[slot_idx]), "cobble")
            self.assertEqual(stack_count(inv.container_craft[slot_idx]), 1)
        self.assertEqual(stack_id(inv.container_craft[4]), "")

        # Verify preview matches furnace!
        preview = inv._table_preview(inv.container_craft)
        self.assertEqual(stack_id(preview), "furnace")
        self.assertEqual(stack_count(preview), 1)

    def test_fuels_loaded_from_assets(self):
        from server.recipes import FUELS, fuel_burn_time
        self.assertIn("planks", FUELS)
        self.assertIn("log", FUELS)
        self.assertIn("crafting_table", FUELS)
        self.assertIn("furnace", FUELS)
        self.assertIn("stick", FUELS)
        self.assertEqual(fuel_burn_time("stick"), 5.0)
        self.assertEqual(fuel_burn_time("planks"), 15.0)

    def test_label_pool_color_conversion(self):
        from client.menu import LabelSpec

        spec_float = LabelSpec("test", 10, 10, 14, "center", "center", (1.0, 0.5, 0.0, 1.0))
        spec_int = LabelSpec("test2", 20, 20, 14, "center", "center", (255, 128, 0, 255))
        # Ensure conversion logic handles float tuples properly
        color = spec_float.color
        if color and any(isinstance(v, float) for v in color):
            color = tuple(int(v * 255 if v <= 1.0 else v) for v in color)
        self.assertEqual(color, (255, 127, 0, 255))


    def test_taking_input_out_resets_cooking(self):
        furnace = Furnace(
            input_stack=make_stack("cobble", 1),
            fuel_stack=make_stack("planks", 1),
        )
        # Cook for 5 seconds
        furnace.tick(5.0)
        self.assertAlmostEqual(furnace.cook_time, 5.0)

        # Remove input
        furnace.input = empty_stack()
        furnace.tick(0.1)
        self.assertEqual(furnace.cook_time, 0.0)

    def test_furnace_layout_no_overlap(self):
        inv = CreativeInventory()
        furnace = Furnace()
        frame = inv.layout(1280, 720, empty_slots(9), 0, (0, 0), empty_slots(27), container_type="furnace", furnace=furnace)

        # Find input slot hit
        input_hit = next(h for h in frame.hits if h.action == "furnace:input")
        # Find title and subtitle labels
        subtitle_label = next(lbl for lbl in frame.labels if lbl.text in ("Smelting", "Furnace"))

        # Verify input slot is completely below the subtitle
        self.assertLess(input_hit.y + input_hit.h, subtitle_label.y)

    def test_station_custom_recipe_type_filtering(self):
        from server.blocks import BLOCKS

        # Carpenter's table should be registered as a crafting_table station
        self.assertIn("carpenters_table", BLOCKS)
        ct = BLOCKS["carpenters_table"]
        self.assertEqual(ct.station, "crafting_table")
        self.assertEqual(ct.recipe_types, ("carpenters_table",))

        # 1 log in carpenters table crafts 16 sticks
        log_grid = ("log", "", "", "", "", "", "", "", "")
        matched_carpenter = craft_match(log_grid, recipe_types=ct.recipe_types)
        self.assertIsNotNone(matched_carpenter)
        self.assertEqual(matched_carpenter[0], "stick")
        self.assertEqual(matched_carpenter[1], 16)

        # In standard crafting grid (or player 2x2), 1 log crafts 4 planks, NOT sticks
        matched_standard = craft_match(log_grid)
        self.assertIsNotNone(matched_standard)
        self.assertEqual(matched_standard[0], "planks")
        self.assertEqual(matched_standard[1], 4)

        # In carpenter's table, standard plank recipe does NOT match
        plank_grid = ("planks", "", "", "", "", "", "", "", "")
        self.assertIsNone(craft_match(plank_grid, recipe_types=ct.recipe_types))

        # Creative inventory preview & take with carpenter's table container_block
        inv = CreativeInventory()
        inv.container_type = "crafting_table"
        inv.container_block = "carpenters_table"
        inv.container_craft[0] = make_stack("log", 1)
        prev = inv._table_preview(inv.container_craft)
        self.assertEqual(stack_id(prev), "stick")
        self.assertEqual(stack_count(prev), 16)

        inv._take_table_result(inv.container_craft)
        self.assertEqual(inv.held, "stick")
        self.assertEqual(inv.held_count, 16)
        self.assertEqual(stack_count(inv.container_craft[0]), 0)

    def test_furnace_variant_custom_recipe_and_cook_speed(self):
        from server.blocks import BLOCKS

        self.assertIn("blast_furnace", BLOCKS)
        bf_block = BLOCKS["blast_furnace"]
        self.assertEqual(bf_block.station, "furnace")
        self.assertEqual(bf_block.recipe_types, ("blasting",))
        self.assertEqual(bf_block.cook_speed, 2.0)

        # Cobble has a smelting recipe, so blast furnace (blasting only) cannot smelt it
        self.assertIsNotNone(smelt_match("cobble", recipe_types=("smelting",)))
        self.assertIsNone(smelt_match("cobble", recipe_types=bf_block.recipe_types))

        # Blast furnace recipe types match iron_ingot from blasting
        matched = smelt_match("raw_iron", recipe_types=bf_block.recipe_types)
        self.assertIsNotNone(matched)
        self.assertEqual(matched[0], "iron_ingot")

        # Create blast furnace instance and test cook speed multiplier
        bf = Furnace(
            input_stack=make_stack("raw_iron", 1),
            fuel_stack=make_stack("coal", 1),
            block_id="blast_furnace",
            recipe_types=bf_block.recipe_types,
            cook_speed=bf_block.cook_speed,
        )
        self.assertEqual(bf.cook_speed, 2.0)
        # Tick for 2.0s -> cook_time should advance by 2.0 * 2.0 = 4.0s
        bf.tick(2.0)
        self.assertAlmostEqual(bf.cook_time, 4.0)

    def test_furnace_serialization_with_variant_properties(self):
        bf = Furnace(
            x=10,
            y=64,
            z=20,
            block_id="blast_furnace",
            recipe_types=("blasting",),
            cook_speed=2.5,
            input_stack=make_stack("raw_iron", 3),
        )
        data = bf.to_dict()
        self.assertEqual(data["block_id"], "blast_furnace")
        self.assertEqual(data["recipe_types"], ["blasting"])
        self.assertEqual(data["cook_speed"], 2.5)

        restored = Furnace.from_dict(data)
        self.assertEqual(restored.x, 10)
        self.assertEqual(restored.y, 64)
        self.assertEqual(restored.z, 20)
        self.assertEqual(restored.block_id, "blast_furnace")
        self.assertEqual(restored.recipe_types, ("blasting",))
        self.assertEqual(restored.cook_speed, 2.5)
        self.assertEqual(stack_id(restored.input), "raw_iron")
        self.assertEqual(stack_count(restored.input), 3)

    def test_world_variant_stations(self):
        world = World(seed=999)
        world.generate_chunk(0, 0, 0)
        world.set_block(5, 10, 5, "blast_furnace")
        self.assertIn((5, 10, 5), world.furnaces)
        furnace = world.get_or_create_furnace(5, 10, 5)
        self.assertEqual(furnace.block_id, "blast_furnace")
        self.assertEqual(furnace.cook_speed, 2.0)
        self.assertEqual(furnace.recipe_types, ("blasting",))

        # Put an item in and remove block
        furnace.input = make_stack("raw_iron", 2)
        world.set_block(5, 10, 5, "")
        self.assertNotIn((5, 10, 5), world.furnaces)
        # Drops should spawn in world entities
        dropped_items = {e.item: e.count for e in world.entities}
        self.assertEqual(dropped_items.get("raw_iron"), 2)



if __name__ == "__main__":
    unittest.main()
