import math

from client.menu import DIM, Frame, MUTED, TITLE
from server.blocks import BLOCKS, HOTBAR_SLOTS, TABS
from server.items import ITEMS, creative_items, stack_limit
from server.recipes import craft_match, fuel_burn_time, smelt_match
from server.stacks import (
    CRAFT_SLOTS,
    INVENTORY_SLOTS,
    add_to_inventory,
    empty_slots,
    empty_stack,
    make_stack,
    normalize_stack,
    stack_count,
    stack_id,
)

COLUMNS = 9
VISIBLE_ROWS = 5
CELL = 52
GAP = 8
TAB = 44
TAB_LIFT = 8
TAB_OVERLAP = 12
STORE_GAP = 16
CRAFT_GAP = 22
SLOT = (0.08, 0.09, 0.07, 1.0)
SLOT_HOVER = (0.22, 0.26, 0.18, 1.0)
PANEL = (0.11, 0.13, 0.10, 0.96)
TAB_IDLE = (0.18, 0.20, 0.16, 1.0)
TAB_HOVER = (0.26, 0.32, 0.20, 1.0)
SELECTED = (0.93, 0.95, 0.88, 1.0)


class CreativeInventory:
    def __init__(self):
        self.scroll = 0
        self.held = ""
        self.held_count = 0
        self.held_infinite = False
        self.hover = None
        self.tab = TABS[0].id
        self.preview = empty_stack()
        self.container_type = "player"
        self.container_block = None
        self.container_craft = empty_slots(9)
        self.furnace = None
        self.table_preview = empty_stack()
        self.drag_button = None
        self.drag_slots = set()

    def layout(
        self,
        width,
        height,
        hotbar,
        selected,
        mouse,
        storage=None,
        craft=None,
        gamemode="creative",
        container_type=None,
        furnace=None,
    ):
        if container_type is not None:
            self.container_type = container_type
        if furnace is not None:
            self.furnace = furnace

        storage = self._slots(storage, INVENTORY_SLOTS)
        craft = self._slots(craft, CRAFT_SLOTS)
        frame = Frame()
        frame.icons = []
        frame.rect(0, 0, width, height, DIM)
        frame.hit(0, 0, width, height, "clear")

        if self.container_type == "crafting_table":
            return self._layout_crafting_table(frame, width, height, hotbar, selected, mouse, storage)
        if self.container_type == "furnace":
            return self._layout_furnace(frame, width, height, hotbar, selected, mouse, storage, self.furnace)

        if gamemode == "survival" and self._current().screen != "player":
            player_tab = next((t for t in TABS if t.screen == "player"), TABS[-1])
            self.tab = player_tab.id

        current = self._current()
        player = current.screen == "player"
        self.preview = self._preview(craft) if player else empty_stack()
        if player:
            panel_h = self._player_height()
            grid_w = COLUMNS * CELL + (COLUMNS - 1) * GAP
        else:
            items = creative_items(self.tab)
            rows = max(1, math.ceil(len(items) / COLUMNS))
            visible = VISIBLE_ROWS
            grid_w = COLUMNS * CELL + (COLUMNS - 1) * GAP
            grid_h = visible * CELL + (visible - 1) * GAP
            panel_h = 96 + grid_h + 132
        tabs_list = [t for t in TABS if (gamemode == "creative" or t.screen == "player")]
        tabs_w = len(tabs_list) * TAB + max(0, len(tabs_list) - 1) * GAP
        panel_w = max(grid_w + 48, tabs_w + 48)
        x = max(16, (width - panel_w) / 2)
        y = max(16, (height - panel_h - TAB - TAB_LIFT) / 2)
        panel_top = y + panel_h
        tabs_x = x + (panel_w - tabs_w) / 2
        for index, tab in enumerate(tabs_list):
            if tab.id == self.tab:
                continue
            self._tab(frame, tab, tabs_x + index * (TAB + GAP), panel_top + TAB_LIFT, TAB, False)
        frame.rect(x, y, panel_w, panel_h, PANEL)
        for index, tab in enumerate(tabs_list):
            if tab.id != self.tab:
                continue
            self._tab(
                frame,
                tab,
                tabs_x + index * (TAB + GAP),
                panel_top - TAB_OVERLAP,
                TAB + TAB_LIFT + TAB_OVERLAP,
                True,
            )
        frame.label(current.name, x + panel_w / 2, y + panel_h - 36, 28, "center", "center", TITLE)
        frame.label(self._detail(hotbar, storage, craft, current), x + panel_w / 2, y + panel_h - 68, 16, "center", "center", MUTED)
        frame.label("E or Esc to close", x + panel_w / 2, y + 26, 14, "center", "center", MUTED)

        origin_x = x + (panel_w - grid_w) / 2
        if player:
            self._draw_player(frame, origin_x, y, storage, craft)
        else:
            self.scroll = max(0, min(self.scroll, max(0, rows - visible)))
            origin_y = y + panel_h - 96 - CELL
            for slot in range(visible * COLUMNS):
                index = self.scroll * COLUMNS + slot
                column = slot % COLUMNS
                row = slot // COLUMNS
                cell_x = origin_x + column * (CELL + GAP)
                cell_y = origin_y - row * (CELL + GAP)
                if index >= len(items):
                    frame.rect(cell_x, cell_y, CELL, CELL, SLOT)
                    continue
                self._slot(frame, cell_x, cell_y, f"item:{items[index]}", make_stack(items[index], 1))

        self._draw_hotbar(frame, x, y, panel_w, hotbar, selected)
        if self.held:
            self._icon(frame, mouse[0] - 18, mouse[1] - 18, 40, self.held)
            if self.held_count > 1:
                frame.label(str(self.held_count), mouse[0] + 16, mouse[1] - 4, 13, "right", "bottom", TITLE)
        return frame

    def _layout_crafting_table(self, frame, width, height, hotbar, selected, mouse, storage):
        self.table_preview = self._table_preview(self.container_craft)
        grid_w = COLUMNS * CELL + (COLUMNS - 1) * GAP
        panel_w = grid_w + 48
        panel_h = 560
        x = max(16, (width - panel_w) / 2)
        y = max(16, (height - panel_h) / 2)
        frame.rect(x, y, panel_w, panel_h, PANEL)
        title = "Crafting Table"
        if self.container_block and self.container_block in BLOCKS:
            title = BLOCKS[self.container_block].name
        frame.label(title, x + panel_w / 2, y + panel_h - 26, 24, "center", "center", TITLE)
        frame.label(self._detail_container(hotbar, storage), x + panel_w / 2, y + panel_h - 52, 16, "center", "center", MUTED)
        frame.label("E or Esc to close", x + panel_w / 2, y + 26, 14, "center", "center", MUTED)

        origin_x = x + (panel_w - grid_w) / 2
        top_storage = 58 + CELL + STORE_GAP + 2 * (CELL + GAP)
        craft_top = top_storage + CELL + CRAFT_GAP + 2 * (CELL + GAP)

        for row in range(3):
            for col in range(3):
                index = row * 3 + col
                self._slot(
                    frame,
                    origin_x + col * (CELL + GAP),
                    y + craft_top - row * (CELL + GAP),
                    f"table_craft:{index}",
                    self.container_craft[index],
                )
        result_x = origin_x + 3 * (CELL + GAP) + 36
        result_y = y + craft_top - (CELL + GAP)
        frame.label(">", result_x - 24, result_y + CELL / 2, 22, "center", "center", MUTED)
        self._slot(frame, result_x, result_y, "table_result:out", self.table_preview)

        for row in range(3):
            for col in range(COLUMNS):
                index = row * COLUMNS + col
                self._slot(
                    frame,
                    origin_x + col * (CELL + GAP),
                    y + top_storage - row * (CELL + GAP),
                    f"store:{index}",
                    storage[index],
                )

        self._draw_hotbar(frame, x, y, panel_w, hotbar, selected)

        if self.held:
            self._icon(frame, mouse[0] - 18, mouse[1] - 18, 40, self.held)
            if self.held_count > 1:
                frame.label(str(self.held_count), mouse[0] + 16, mouse[1] - 4, 13, "right", "bottom", TITLE)
        return frame

    def _layout_furnace(self, frame, width, height, hotbar, selected, mouse, storage, furnace):
        grid_w = COLUMNS * CELL + (COLUMNS - 1) * GAP
        panel_w = grid_w + 48
        panel_h = 510
        x = max(16, (width - panel_w) / 2)
        y = max(16, (height - panel_h) / 2)
        frame.rect(x, y, panel_w, panel_h, PANEL)
        title = "Furnace"
        furnace_block = getattr(furnace, "block_id", None) or self.container_block
        if furnace_block and furnace_block in BLOCKS:
            title = BLOCKS[furnace_block].name
        frame.label(title, x + panel_w / 2, y + panel_h - 26, 24, "center", "center", TITLE)
        frame.label(self._detail_container(hotbar, storage), x + panel_w / 2, y + panel_h - 52, 16, "center", "center", MUTED)
        frame.label("E or Esc to close", x + panel_w / 2, y + 26, 14, "center", "center", MUTED)

        origin_x = x + (panel_w - grid_w) / 2
        top_storage = 58 + CELL + STORE_GAP + 2 * (CELL + GAP)
        cx = origin_x + grid_w / 2

        in_x = cx - 60
        in_y = y + 386
        self._slot(frame, in_x, in_y, "furnace:input", furnace.input if furnace else empty_stack())

        fuel_x = cx - 60
        fuel_y = y + 312
        self._slot(frame, fuel_x, fuel_y, "furnace:fuel", furnace.fuel if furnace else empty_stack())

        flame_x = cx - 60 + (CELL - 18) / 2
        flame_y = y + 368
        burn_pct = (furnace.burn_time / furnace.burn_duration) if (furnace and furnace.burn_duration > 0) else 0.0
        frame.rect(flame_x, flame_y, 18, 14, (0.18, 0.12, 0.08, 1.0))
        if burn_pct > 0:
            h = max(2, int(14 * burn_pct))
            frame.rect(flame_x, flame_y, 18, h, (0.95, 0.5, 0.1, 1.0))
            frame.label("^", flame_x + 9, flame_y + 7, 12, "center", "center", (255, 204, 51, 255))

        arrow_x = cx - 4
        arrow_y = y + 349
        cook_pct = (furnace.cook_time / furnace.cook_duration) if (furnace and furnace.cook_duration > 0) else 0.0
        frame.rect(arrow_x, arrow_y, 30, 12, (0.16, 0.16, 0.16, 1.0))
        if cook_pct > 0:
            w = max(2, int(30 * cook_pct))
            frame.rect(arrow_x, arrow_y, w, 12, (0.85, 0.85, 0.85, 1.0))
        frame.label(">", arrow_x + 15, arrow_y + 6, 16, "center", "center", (25, 25, 25, 255) if cook_pct > 0.5 else (180, 180, 180, 255))

        out_x = cx + 36
        out_y = y + 328
        self._slot(frame, out_x, out_y, "furnace:output", furnace.output if furnace else empty_stack())

        for row in range(3):
            for col in range(COLUMNS):
                index = row * COLUMNS + col
                self._slot(
                    frame,
                    origin_x + col * (CELL + GAP),
                    y + top_storage - row * (CELL + GAP),
                    f"store:{index}",
                    storage[index],
                )

        self._draw_hotbar(frame, x, y, panel_w, hotbar, selected)

        if self.held:
            self._icon(frame, mouse[0] - 18, mouse[1] - 18, 40, self.held)
            if self.held_count > 1:
                frame.label(str(self.held_count), mouse[0] + 16, mouse[1] - 4, 13, "right", "bottom", TITLE)
        return frame

    def scroll_by(self, delta):
        if self._current().screen != "blocks":
            return
        rows = max(1, math.ceil(len(creative_items(self.tab)) / COLUMNS))
        self.scroll = max(0, min(max(0, rows - VISIBLE_ROWS), self.scroll + delta))

    def apply(
        self,
        action,
        button,
        hotbar,
        selected,
        storage=None,
        craft=None,
        gamemode="creative",
        furnace=None,
        shift=False,
    ):
        storage = storage if storage is not None else []
        craft = craft if craft is not None else []
        furnace = furnace if furnace is not None else self.furnace
        if action in (None, "clear"):
            if button == "left":
                self.dismiss_held(hotbar, storage, craft)
            return None
        kind, value = action.split(":", 1)
        if kind == "tab":
            tab_obj = next((t for t in TABS if t.id == value), None)
            if tab_obj is not None and value != self.tab:
                if gamemode == "survival" and tab_obj.screen != "player":
                    return None
                self.tab = value
                self.scroll = 0
            return None
        if kind == "item":
            if gamemode == "survival":
                return None
            limit = stack_limit(value)
            if shift and button == "left":
                leftover = add_to_inventory([hotbar, storage], value, limit)
                if leftover > 0 and not self.held:
                    self.held = value
                    self.held_count = leftover
                    self.held_infinite = False
                return None
            if shift and button == "right":
                self.held = value
                self.held_count = limit
                self.held_infinite = False
                return None
            if self.held == value:
                if button in ("left", "right"):
                    if self.held_count < limit:
                        self.held_count += 1
                    self.held_infinite = False
                    return None
            if self.held:
                if button == "left":
                    self.held = value
                    self.held_count = 1
                    self.held_infinite = True
                return None
            if button == "right":
                hotbar[selected] = make_stack(value, 1)
                return selected
            self.held = value
            self.held_count = 1
            self.held_infinite = True
            return None
        if kind == "hotbar":
            index = int(value)
            if shift:
                if furnace is not None:
                    self._shift_click_into_furnace_or_storage(hotbar, index, furnace, storage)
                else:
                    self._shift_click_slot(hotbar, index, [storage])
                return index
            self._click_slot(hotbar, index, button)
            return index
        if kind == "store":
            index = int(value)
            if shift:
                if furnace is not None:
                    self._shift_click_into_furnace_or_storage(storage, index, furnace, hotbar)
                else:
                    self._shift_click_slot(storage, index, [hotbar])
                return None
            self._click_slot(storage, index, button)
            return None
        if kind == "craft":
            index = int(value)
            if shift:
                self._shift_click_slot(craft, index, [hotbar, storage])
                return None
            self._click_slot(craft, index, button)
            return None
        if kind == "result":
            if shift:
                self._shift_take_result(craft, [hotbar, storage])
                return None
            self._take_result(craft)
            return None
        if kind == "table_craft":
            index = int(value)
            if shift:
                self._shift_click_slot(self.container_craft, index, [hotbar, storage])
                return None
            self._click_slot(self.container_craft, index, button)
            return None
        if kind == "table_result":
            if shift:
                self._shift_take_table_result(self.container_craft, [hotbar, storage])
                return None
            self._take_table_result(self.container_craft)
            return None
        if kind == "furnace":
            if furnace is not None:
                if value == "input":
                    slots = [furnace.input]
                    if shift:
                        self._shift_click_slot(slots, 0, [hotbar, storage])
                        furnace.input = slots[0]
                        return None
                    self._click_slot(slots, 0, button)
                    furnace.input = slots[0]
                elif value == "fuel":
                    slots = [furnace.fuel]
                    if shift:
                        self._shift_click_slot(slots, 0, [hotbar, storage])
                        furnace.fuel = slots[0]
                        return None
                    self._click_slot(slots, 0, button)
                    furnace.fuel = slots[0]
                elif value == "output":
                    if shift:
                        self._shift_take_furnace_output(furnace, [hotbar, storage])
                        return None
                    self._take_furnace_output(furnace)
            return None
        return None

    def _shift_click_slot(self, source_slots, index, target_groups):
        if index < 0 or index >= len(source_slots):
            return
        stack = normalize_stack(source_slots[index])
        item_id = stack_id(stack)
        count = stack_count(stack)
        if not item_id or count <= 0:
            return
        leftover = add_to_inventory(target_groups, item_id, count)
        if leftover > 0:
            source_slots[index] = make_stack(item_id, leftover)
        else:
            source_slots[index] = empty_stack()

    def _get_table_recipe_types(self):
        if not self.container_block:
            return None
        blk = BLOCKS.get(self.container_block)
        if blk is None:
            return None
        return getattr(blk, "recipe_types", None)

    def _shift_click_into_furnace_or_storage(self, source_slots, index, furnace, fallback_group):
        if index < 0 or index >= len(source_slots):
            return
        stack = normalize_stack(source_slots[index])
        item_id = stack_id(stack)
        count = stack_count(stack)
        if not item_id or count <= 0:
            return

        f_types = getattr(furnace, "recipe_types", None)
        if smelt_match(item_id, recipe_types=f_types) is not None:
            inp = normalize_stack(furnace.input)
            if not stack_id(inp) or (stack_id(inp) == item_id and stack_count(inp) < stack_limit(item_id)):
                slots = [furnace.input]
                leftover = add_to_inventory([slots], item_id, count)
                furnace.input = slots[0]
                if leftover <= 0:
                    source_slots[index] = empty_stack()
                    return
                source_slots[index] = make_stack(item_id, leftover)
                count = leftover

        if fuel_burn_time(item_id) > 0:
            fuel = normalize_stack(furnace.fuel)
            if not stack_id(fuel) or (stack_id(fuel) == item_id and stack_count(fuel) < stack_limit(item_id)):
                slots = [furnace.fuel]
                leftover = add_to_inventory([slots], item_id, count)
                furnace.fuel = slots[0]
                if leftover <= 0:
                    source_slots[index] = empty_stack()
                    return
                source_slots[index] = make_stack(item_id, leftover)
                count = leftover

        self._shift_click_slot(source_slots, index, [fallback_group])

    def _shift_take_furnace_output(self, furnace, target_groups):
        out = normalize_stack(furnace.output)
        out_id = stack_id(out)
        out_cnt = stack_count(out)
        if not out_id or out_cnt <= 0:
            return
        leftover = add_to_inventory(target_groups, out_id, out_cnt)
        if leftover > 0:
            furnace.output = make_stack(out_id, leftover)
        else:
            furnace.output = empty_stack()

    def _shift_take_result(self, craft, target_groups):
        while True:
            ids = tuple(stack_id(slot) if stack_count(slot) else "" for slot in craft[:CRAFT_SLOTS])
            matched = craft_match(ids)
            if matched is None:
                break
            res_id, res_count, indexes = matched
            room = 0
            limit = stack_limit(res_id)
            for group in target_groups:
                for slot in group:
                    s = normalize_stack(slot)
                    if not stack_id(s):
                        room += limit
                    elif stack_id(s) == res_id:
                        room += limit - stack_count(s)
            if room < res_count:
                break
            for idx in indexes:
                s = normalize_stack(craft[idx])
                left = stack_count(s) - 1
                craft[idx] = make_stack(stack_id(s), left) if left > 0 else empty_stack()
            add_to_inventory(target_groups, res_id, res_count)

    def _shift_take_table_result(self, craft, target_groups):
        recipe_types = self._get_table_recipe_types()
        while True:
            if len(craft) < 9:
                break
            ids = tuple(stack_id(slot) if stack_count(slot) else "" for slot in craft)
            matched = craft_match(ids, recipe_types=recipe_types)
            if matched is None:
                break
            res_id, res_count, indexes = matched
            room = 0
            limit = stack_limit(res_id)
            for group in target_groups:
                for slot in group:
                    s = normalize_stack(slot)
                    if not stack_id(s):
                        room += limit
                    elif stack_id(s) == res_id:
                        room += limit - stack_count(s)
            if room < res_count:
                break
            for idx in indexes:
                s = normalize_stack(craft[idx])
                left = stack_count(s) - 1
                craft[idx] = make_stack(stack_id(s), left) if left > 0 else empty_stack()
            add_to_inventory(target_groups, res_id, res_count)

    def _take_table_result(self, craft):
        if len(craft) < 9:
            return
        ids = tuple(stack_id(slot) if stack_count(slot) else "" for slot in craft)
        matched = craft_match(ids, recipe_types=self._get_table_recipe_types())
        if matched is None:
            return
        if self.held_infinite:
            self.clear_held()
        result_id, result_count, indexes = matched
        if self.held and self.held != result_id:
            return
        have = self.held_count if self.held else 0
        if have + result_count > stack_limit(result_id):
            return
        for index in indexes:
            slot = normalize_stack(craft[index])
            left = stack_count(slot) - 1
            craft[index] = make_stack(stack_id(slot), left) if left > 0 else empty_stack()
        self.held = result_id
        self.held_count = have + result_count
        self.held_infinite = False

    def _take_furnace_output(self, furnace):
        out = normalize_stack(furnace.output)
        out_id = stack_id(out)
        out_cnt = stack_count(out)
        if not out_id or out_cnt <= 0:
            return
        if self.held and self.held != out_id:
            return
        have = self.held_count if self.held else 0
        if have + out_cnt > stack_limit(out_id):
            take = stack_limit(out_id) - have
        else:
            take = out_cnt
        if take <= 0:
            return
        rem = out_cnt - take
        furnace.output = make_stack(out_id, rem) if rem > 0 else empty_stack()
        self.held = out_id
        self.held_count = have + take
        self.held_infinite = False

    def _table_preview(self, craft):
        ids = tuple(stack_id(slot) if stack_count(slot) else "" for slot in list(craft)[:9])
        if len(ids) < 9:
            ids = ids + ("",) * (9 - len(ids))
        matched = craft_match(ids, recipe_types=self._get_table_recipe_types())
        if matched is None:
            return empty_stack()
        return make_stack(matched[0], matched[1])

    def dismiss_container(self, hotbar, storage, world=None, player_pos=None):
        if self.container_type == "crafting_table":
            for index in range(len(self.container_craft)):
                slot = normalize_stack(self.container_craft[index])
                sid = stack_id(slot)
                scnt = stack_count(slot)
                if sid and scnt > 0:
                    self.container_craft[index] = empty_stack()
                    self.held = sid
                    self.held_count = scnt
                    self.stash_held(hotbar, storage, [])
                    if self.held and self.held_count > 0:
                        if world is not None and player_pos is not None:
                            world.spawn_item(self.held, count=self.held_count, pos=player_pos)
                        self.clear_held()
        self.container_type = "player"
        self.container_block = None
        self.furnace = None
        self.drag_slots.clear()
        self.drag_button = None

    def _detail_container(self, hotbar, storage):
        name = self._hovered_name(hotbar, storage, [])
        if name:
            return name
        if self.container_type == "crafting_table":
            if self.container_block and self.container_block in BLOCKS:
                return BLOCKS[self.container_block].name
            return "Crafting"
        if self.container_type == "furnace":
            furnace_block = getattr(self.furnace, "block_id", None) or self.container_block
            if furnace_block and furnace_block in BLOCKS:
                return BLOCKS[furnace_block].name
            return "Smelting"
        return ""

    def clear_held(self):
        self.held = ""
        self.held_count = 0
        self.held_infinite = False

    def dismiss_held(self, hotbar, storage, craft):
        if self.held and not self.held_infinite:
            self.stash_held(hotbar, storage, craft)
            return
        self.clear_held()

    def stash_held(self, hotbar, storage, craft):
        block_id = self.held
        remaining = self.held_count
        self.clear_held()
        if not block_id or remaining <= 0:
            return
        groups = (storage, hotbar, craft)
        for slots in groups:
            for index, raw in enumerate(slots):
                slot = normalize_stack(raw)
                if stack_id(slot) != block_id or stack_count(slot) >= stack_limit(block_id):
                    continue
                moved = min(stack_limit(block_id) - stack_count(slot), remaining)
                slots[index] = make_stack(block_id, stack_count(slot) + moved)
                remaining -= moved
                if remaining <= 0:
                    return
        for slots in groups:
            for index, raw in enumerate(slots):
                if stack_id(raw):
                    continue
                slots[index] = make_stack(block_id, remaining)
                return
        self.held = block_id
        self.held_count = remaining
        self.held_infinite = False

    def _click_slot(self, slots, index, button):
        if index < 0 or index >= len(slots):
            return
        current = normalize_stack(slots[index])

        if self.held_infinite:
            if button == "right":
                slots[index] = empty_stack()
            else:
                slots[index] = make_stack(self.held, 1)
            return

        if button in ("right", "right_drag"):
            if not self.held:
                if button == "right" and stack_id(current):
                    total = stack_count(current)
                    take = (total + 1) // 2
                    left = total - take
                    slots[index] = make_stack(stack_id(current), left) if left > 0 else empty_stack()
                    self.held = stack_id(current)
                    self.held_count = take
                    self.held_infinite = False
                return

            if not stack_id(current):
                slots[index] = make_stack(self.held, 1)
                self.held_count -= 1
                if self.held_count <= 0:
                    self.clear_held()
                return

            if stack_id(current) == self.held:
                if stack_count(current) < stack_limit(self.held):
                    slots[index] = make_stack(self.held, stack_count(current) + 1)
                    self.held_count -= 1
                    if self.held_count <= 0:
                        self.clear_held()
                return
            return

        if not self.held:
            if stack_id(current):
                self.held = stack_id(current)
                self.held_count = stack_count(current)
                self.held_infinite = False
                slots[index] = empty_stack()
            return
        if not stack_id(current):
            slots[index] = make_stack(self.held, self.held_count)
            self.clear_held()
            return
        if stack_id(current) == self.held:
            room = stack_limit(self.held) - stack_count(current)
            moved = min(room, self.held_count)
            slots[index] = make_stack(self.held, stack_count(current) + moved)
            self.held_count -= moved
            if self.held_count <= 0:
                self.clear_held()
            return
        slots[index] = make_stack(self.held, self.held_count)
        self.held = stack_id(current)
        self.held_count = stack_count(current)
        self.held_infinite = False

    def _take_result(self, craft):
        if len(craft) < CRAFT_SLOTS:
            return
        ids = tuple(stack_id(slot) if stack_count(slot) else "" for slot in craft)
        matched = craft_match(ids)
        if matched is None:
            return
        if self.held_infinite:
            self.clear_held()
        result_id, result_count, indexes = matched
        if self.held and self.held != result_id:
            return
        have = self.held_count if self.held else 0
        if have + result_count > stack_limit(result_id):
            return
        for index in indexes:
            slot = normalize_stack(craft[index])
            left = stack_count(slot) - 1
            craft[index] = make_stack(stack_id(slot), left) if left > 0 else empty_stack()
        self.held = result_id
        self.held_count = have + result_count
        self.held_infinite = False

    def _player_height(self):
        top_storage = 58 + CELL + STORE_GAP + 2 * (CELL + GAP)
        craft_top = top_storage + CELL + CRAFT_GAP + CELL + GAP
        return craft_top + CELL + 96

    def _draw_player(self, frame, origin_x, y, storage, craft):
        top_storage = 58 + CELL + STORE_GAP + 2 * (CELL + GAP)
        craft_top = top_storage + CELL + CRAFT_GAP + CELL + GAP
        for row in range(2):
            for col in range(2):
                index = row * 2 + col
                self._slot(
                    frame,
                    origin_x + col * (CELL + GAP),
                    y + craft_top - row * (CELL + GAP),
                    f"craft:{index}",
                    craft[index],
                )
        result_x = origin_x + 2 * (CELL + GAP) + 36
        result_y = y + craft_top - (CELL + GAP) / 2
        frame.label(">", result_x - 24, result_y + CELL / 2, 22, "center", "center", MUTED)
        self._slot(frame, result_x, result_y, "result:out", self.preview)
        for row in range(3):
            for col in range(COLUMNS):
                index = row * COLUMNS + col
                self._slot(
                    frame,
                    origin_x + col * (CELL + GAP),
                    y + top_storage - row * (CELL + GAP),
                    f"store:{index}",
                    storage[index],
                )

    def _draw_hotbar(self, frame, x, y, panel_w, hotbar, selected):
        hotbar_w = HOTBAR_SLOTS * CELL + (HOTBAR_SLOTS - 1) * GAP
        hotbar_x = x + (panel_w - hotbar_w) / 2
        hotbar_y = y + 58
        for index in range(HOTBAR_SLOTS):
            cell_x = hotbar_x + index * (CELL + GAP)
            slot = hotbar[index] if index < len(hotbar) else ""
            self._slot(frame, cell_x, hotbar_y, f"hotbar:{index}", slot, index == selected)

    def _slot(self, frame, x, y, action, stack, selected=False):
        if selected:
            frame.rect(x - 3, y - 3, CELL + 6, CELL + 6, SELECTED)
        color = SLOT_HOVER if self.hover == action else SLOT
        frame.rect(x, y, CELL, CELL, color)
        frame.hit(x, y, CELL, CELL, action)
        self._icon(frame, x, y, CELL, stack_id(stack))
        count = stack_count(stack)
        if count > 1:
            frame.label(str(count), x + CELL - 5, y + 5, 13, "right", "bottom", TITLE)

    def _tab(self, frame, tab, x, y, height, selected):
        action = f"tab:{tab.id}"
        if selected:
            color = PANEL
        elif self.hover == action:
            color = TAB_HOVER
        else:
            color = TAB_IDLE
        frame.rect(x, y, TAB, height, color)
        frame.hit(x, y, TAB, height, action)
        icon_y = y + height - TAB
        if tab.icon:
            self._icon(frame, x, icon_y, TAB, tab.icon)
        else:
            self._glyph(frame, x, icon_y)

    def _glyph(self, frame, x, y):
        pad = 11
        gap = 4
        size = (TAB - pad * 2 - gap) / 2
        for row in range(2):
            for col in range(2):
                frame.rect(
                    x + pad + col * (size + gap),
                    y + pad + (1 - row) * (size + gap),
                    size,
                    size,
                    SLOT_HOVER,
                )

    def _detail(self, hotbar, storage, craft, current):
        if self.hover and self.hover.startswith("tab:"):
            tab_id = self.hover.split(":", 1)[1]
            hovered = next((tab.name for tab in TABS if tab.id == tab_id), "")
            if hovered and tab_id != current.id:
                return hovered
        name = self._hovered_name(hotbar, storage, craft)
        if name:
            return name
        if current.screen == "player":
            return "Crafting"
        return "Creative Inventory"

    def _hovered_name(self, hotbar, storage, craft):
        slot = None
        if self.hover and self.hover.startswith("item:"):
            return self._name(make_stack(self.hover.split(":", 1)[1], 1))
        if self.hover and self.hover.startswith("hotbar:"):
            index = int(self.hover.split(":", 1)[1])
            slot = hotbar[index] if index < len(hotbar) else ""
        elif self.hover and self.hover.startswith("store:"):
            index = int(self.hover.split(":", 1)[1])
            slot = storage[index] if index < len(storage) else ""
        elif self.hover and self.hover.startswith("craft:"):
            index = int(self.hover.split(":", 1)[1])
            slot = craft[index] if index < len(craft) else ""
        elif self.hover and self.hover.startswith("result:"):
            slot = self.preview
        elif self.hover and self.hover.startswith("table_craft:"):
            index = int(self.hover.split(":", 1)[1])
            slot = self.container_craft[index] if index < len(self.container_craft) else ""
        elif self.hover and self.hover.startswith("table_result:"):
            slot = self.table_preview
        elif self.hover and self.hover == "furnace:input":
            slot = self.furnace.input if self.furnace else ""
        elif self.hover and self.hover == "furnace:fuel":
            slot = self.furnace.fuel if self.furnace else ""
        elif self.hover and self.hover == "furnace:output":
            slot = self.furnace.output if self.furnace else ""
        elif self.held:
            return self._name(make_stack(self.held, self.held_count))
        return self._name(slot) if slot is not None else ""

    def _name(self, slot):
        item = ITEMS.get(stack_id(slot))
        if item is None:
            return ""
        count = stack_count(slot)
        if count > 1:
            return f"{item.name} x{count}"
        return item.name

    def _preview(self, craft):
        ids = tuple(stack_id(slot) if stack_count(slot) else "" for slot in list(craft)[:CRAFT_SLOTS])
        if len(ids) < CRAFT_SLOTS:
            ids = ids + ("",) * (CRAFT_SLOTS - len(ids))
        matched = craft_match(ids)
        if matched is None:
            return empty_stack()
        return make_stack(matched[0], matched[1])

    def _current(self):
        return next(tab for tab in TABS if tab.id == self.tab)

    def _slots(self, slots, count):
        if slots is None:
            return empty_slots(count)
        return slots

    def _icon(self, frame, x, y, size, block_id):
        item = ITEMS.get(block_id)
        if item is None:
            return
        pad = max(4, size * 0.1)
        frame.icons.append((x + pad, y + pad, size - pad * 2, size - pad * 2, *item.icon))
