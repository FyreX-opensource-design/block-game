import time

VISIBLE_ROWS = 6

PANEL = (0.11, 0.13, 0.10, 0.94)
BUTTON = (0.27, 0.36, 0.20, 1.0)
BUTTON_HOVER = (0.36, 0.48, 0.24, 1.0)
BUTTON_DISABLED = (0.20, 0.22, 0.18, 0.72)
ROW = (0.17, 0.19, 0.14, 1.0)
ROW_SELECTED = (0.34, 0.46, 0.22, 1.0)
FIELD = (0.07, 0.08, 0.06, 1.0)
FIELD_FOCUS = (0.93, 0.95, 0.88, 1.0)
DIM = (0.04, 0.05, 0.03, 0.45)
INK = (244, 240, 230, 255)
MUTED = (186, 194, 166, 255)
TITLE = (244, 240, 230, 255)


class LabelSpec:
    def __init__(self, text, x, y, size=18, anchor_x="left", anchor_y="center", color=INK):
        self.text = text
        self.x = x
        self.y = y
        self.size = size
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.color = color


class HitBox:
    def __init__(self, x, y, w, h, action, enabled=True):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.action = action
        self.enabled = enabled

    def contains(self, x, y):
        return self.enabled and self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h


class Frame:
    def __init__(self):
        self.rects = []
        self.labels = []
        self.hits = []

    def rect(self, x, y, w, h, color):
        self.rects.append((x, y, w, h, color))

    def label(self, text, x, y, size=18, anchor_x="left", anchor_y="center", color=INK):
        self.labels.append(LabelSpec(text, x, y, size, anchor_x, anchor_y, color))

    def hit(self, x, y, w, h, action, enabled=True):
        self.hits.append(HitBox(x, y, w, h, action, enabled))

    def button(self, x, y, w, h, text, action, hover, enabled=True):
        if not enabled:
            color = BUTTON_DISABLED
        elif hover == action:
            color = BUTTON_HOVER
        else:
            color = BUTTON
        self.rect(x, y, w, h, color)
        self.label(text, x + w / 2, y + h / 2, 18, "center", "center")
        self.hit(x, y, w, h, action, enabled)


class MenuState:
    def __init__(self):
        self.screen = "title"
        self.worlds = []
        self.servers = []
        self.selected = -1
        self.scroll = 0
        self.name = ""
        self.seed_text = ""
        self.gamemode = "survival"
        self.username = "Player"
        self.server_name = ""
        self.server_address = ""
        self.focus = "name"
        self.hover = None
        self.message = ""
        self.saves_hint = ""
        self._row_click = 0.0
        self.web_ui_enabled = False
        self.web_ui_port = 5000
        self.local_ips = []
        self._prev_screen = "title"

    def layout(self, width, height):
        if self.screen == "title":
            return self._title(width, height)
        if self.screen == "worlds":
            return self._worlds(width, height)
        if self.screen == "servers":
            return self._servers(width, height)
        if self.screen == "add_server":
            return self._add_server(width, height)
        if self.screen == "direct_connect":
            return self._direct_connect(width, height)
        if self.screen == "create":
            return self._create(width, height)
        if self.screen == "confirm":
            return self._confirm(width, height)
        if self.screen == "options":
            return self._options(width, height)
        if self.screen == "pause":
            return self._pause(width, height)
        return Frame()

    def hit_test(self, frame, x, y):
        for hit in reversed(frame.hits):
            if hit.contains(x, y):
                return hit.action
        return None

    def apply(self, action):
        if action is None:
            return None
        if action.startswith("row:"):
            index = int(action[4:])
            now = time.monotonic()
            play = index == self.selected and now - self._row_click < 0.4
            self.selected = index
            self._row_click = now
            if play:
                return "join-server" if self.screen == "servers" else "play"
            return None
        if action.startswith("focus:"):
            self.focus = action.split(":", 1)[1]
            return None
        if action == "toggle-gamemode":
            self.gamemode = "creative" if self.gamemode == "survival" else "survival"
            return None
        if action == "options":
            self._prev_screen = self.screen
            self.screen = "options"
            return "options"
        if action == "toggle-web-ui":
            self.web_ui_enabled = not self.web_ui_enabled
            return "toggle-web-ui"
        if action == "back-options":
            self.screen = getattr(self, "_prev_screen", "title") or "title"
            return "back-options"
        return action

    def _active_items(self):
        return self.servers if self.screen == "servers" else self.worlds

    def move_selection(self, delta):
        items = self._active_items()
        if not items:
            self.selected = -1
            return
        if self.selected < 0:
            self.selected = 0
        else:
            self.selected = max(0, min(len(items) - 1, self.selected + delta))
        self.ensure_visible()

    def scroll_by(self, delta):
        items = self._active_items()
        if not items:
            return
        maximum = max(0, len(items) - VISIBLE_ROWS)
        self.scroll = max(0, min(maximum, self.scroll + delta))

    def ensure_visible(self):
        if self.selected < 0:
            return
        if self.selected < self.scroll:
            self.scroll = self.selected
        if self.selected >= self.scroll + VISIBLE_ROWS:
            self.scroll = self.selected - VISIBLE_ROWS + 1

    def selected_world(self):
        if self.selected < 0 or self.selected >= len(self.worlds):
            return None
        return self.worlds[self.selected]

    def selected_server(self):
        if self.selected < 0 or self.selected >= len(self.servers):
            return None
        return self.servers[self.selected]

    def _panel(self, frame, width, height, panel_w, panel_h):
        x = max(16, (width - panel_w) / 2)
        y = max(16, (height - panel_h) / 2)
        frame.rect(x, y, min(panel_w, width - 32), min(panel_h, height - 32), PANEL)
        return x, y, min(panel_w, width - 32), min(panel_h, height - 32)

    def _title(self, width, height):
        frame = Frame()
        x, y, w, h = self._panel(frame, width, height, 460, 490)
        frame.label("Block Game", x + w / 2, y + h - 58, 46, "center", "center", TITLE)
        frame.label("A small sandbox", x + w / 2, y + h - 96, 16, "center", "center", MUTED)
        field_w = 280
        field_x = x + (w - field_w) / 2
        frame.label("Player Name", field_x, y + h - 128, 14, "left", "center", MUTED)
        self._field(frame, field_x, y + h - 170, field_w, 38, "username", self.username or "", "Player")
        self._stack(
            frame,
            field_x,
            y + 36,
            [
                ("Singleplayer", "singleplayer", True),
                ("Multiplayer", "multiplayer", True),
                ("Options", "options", True),
                ("Quit", "quit", True),
            ],
            button_w=field_w,
        )
        if self.saves_hint:
            frame.label(self.saves_hint, x + w / 2, y + 16, 13, "center", "center", MUTED)
        return frame

    def _worlds(self, width, height):
        frame = Frame()
        x, y, w, h = self._panel(frame, width, height, 760, 560)
        frame.label("Singleplayer", x + 36, y + h - 42, 32, "left", "center")
        list_top = y + h - 78
        row_h = 58
        row_x = x + 24
        row_w = w - 48
        if not self.worlds:
            frame.label("No worlds yet. Create one to start.", x + w / 2, y + h / 2, 18, "center", "center", MUTED)
        else:
            maximum = max(0, len(self.worlds) - VISIBLE_ROWS)
            self.scroll = max(0, min(maximum, self.scroll))
            for slot in range(VISIBLE_ROWS):
                index = self.scroll + slot
                if index >= len(self.worlds):
                    break
                info = self.worlds[index]
                row_y = list_top - (slot + 1) * (row_h + 6)
                action = f"row:{index}"
                color = ROW_SELECTED if index == self.selected else ROW
                if self.hover == action and index != self.selected:
                    color = (0.24, 0.28, 0.18, 1.0)
                frame.rect(row_x, row_y, row_w, row_h, color)
                frame.hit(row_x, row_y, row_w, row_h, action)
                frame.label(info.name, row_x + 16, row_y + 36, 20, "left", "center")
                mode_str = getattr(info, "gamemode", "survival").capitalize()
                frame.label(
                    f"{mode_str}    Seed {info.seed}    {format_played(info.last_played)}",
                    row_x + 16,
                    row_y + 16,
                    13,
                    "left",
                    "center",
                    MUTED,
                )
        has_selection = self.selected_world() is not None
        self._row_buttons(
            frame,
            x + 24,
            y + 22,
            [
                ("Play", "play", has_selection),
                ("New World", "new", True),
                ("Delete", "delete", has_selection),
                ("Back", "back", True),
            ],
            w - 48,
        )
        return frame

    def _servers(self, width, height):
        frame = Frame()
        x, y, w, h = self._panel(frame, width, height, 760, 560)
        frame.label("Multiplayer", x + 36, y + h - 42, 32, "left", "center")
        list_top = y + h - 78
        row_h = 58
        row_x = x + 24
        row_w = w - 48
        if not self.servers:
            frame.label("No saved servers. Add one to start.", x + w / 2, y + h / 2, 18, "center", "center", MUTED)
        else:
            maximum = max(0, len(self.servers) - VISIBLE_ROWS)
            self.scroll = max(0, min(maximum, self.scroll))
            for slot in range(VISIBLE_ROWS):
                index = self.scroll + slot
                if index >= len(self.servers):
                    break
                info = self.servers[index]
                row_y = list_top - (slot + 1) * (row_h + 6)
                action = f"row:{index}"
                color = ROW_SELECTED if index == self.selected else ROW
                if self.hover == action and index != self.selected:
                    color = (0.24, 0.28, 0.18, 1.0)
                frame.rect(row_x, row_y, row_w, row_h, color)
                frame.hit(row_x, row_y, row_w, row_h, action)
                frame.label(info.get("name", "Server"), row_x + 16, row_y + 36, 20, "left", "center")
                frame.label(
                    info.get("address", ""),
                    row_x + 16,
                    row_y + 16,
                    13,
                    "left",
                    "center",
                    MUTED,
                )
        has_selection = self.selected_server() is not None
        if self.message:
            frame.label(self.message, x + w / 2, y + 80, 15, "center", "center", (232, 164, 120, 255))
        self._row_buttons(
            frame,
            x + 24,
            y + 22,
            [
                ("Join Server", "join-server", has_selection),
                ("Direct Connect", "direct-connect", True),
                ("Add Server", "add-server", True),
                ("Delete", "delete-server", has_selection),
                ("Back", "back", True),
            ],
            w - 48,
        )
        return frame

    def _add_server(self, width, height):
        frame = Frame()
        x, y, w, h = self._panel(frame, width, height, 520, 420)
        frame.label("Add Server", x + w / 2, y + h - 52, 32, "center", "center")
        field_x = x + 48
        field_w = w - 96
        frame.label("Server Name", field_x, y + h - 108, 15, "left", "center", MUTED)
        self._field(frame, field_x, y + h - 164, field_w, 44, "server_name", self.server_name or "", "Server Name")
        frame.label("Server Address", field_x, y + h - 196, 15, "left", "center", MUTED)
        self._field(frame, field_x, y + h - 252, field_w, 44, "server_address", self.server_address or "", "127.0.0.1:25565")
        if self.message:
            frame.label(self.message, x + w / 2, y + 118, 15, "center", "center", (232, 164, 120, 255))
        self._row_buttons(
            frame,
            x + 48,
            y + 36,
            [("Done", "save-server", True), ("Cancel", "cancel-server", True)],
            w - 96,
        )
        return frame

    def _direct_connect(self, width, height):
        frame = Frame()
        x, y, w, h = self._panel(frame, width, height, 520, 340)
        frame.label("Direct Connect", x + w / 2, y + h - 52, 32, "center", "center")
        field_x = x + 48
        field_w = w - 96
        frame.label("Server Address", field_x, y + h - 118, 15, "left", "center", MUTED)
        self._field(frame, field_x, y + h - 174, field_w, 44, "server_address", self.server_address or "", "127.0.0.1:25565")
        if self.message:
            frame.label(self.message, x + w / 2, y + 108, 15, "center", "center", (232, 164, 120, 255))
        self._row_buttons(
            frame,
            x + 48,
            y + 36,
            [("Join", "join-direct", True), ("Cancel", "cancel-direct", True)],
            w - 96,
        )
        return frame

    def _create(self, width, height):
        frame = Frame()
        x, y, w, h = self._panel(frame, width, height, 520, 480)
        frame.label("New World", x + w / 2, y + h - 52, 32, "center", "center")
        field_x = x + 48
        field_w = w - 96
        frame.label("Name", field_x, y + h - 108, 15, "left", "center", MUTED)
        self._field(frame, field_x, y + h - 164, field_w, 44, "name", self.name or "", "World name")
        frame.label("Seed", field_x, y + h - 196, 15, "left", "center", MUTED)
        self._field(frame, field_x, y + h - 252, field_w, 44, "seed", self.seed_text, "Random if empty")
        frame.label("Game Mode", field_x, y + h - 284, 15, "left", "center", MUTED)
        frame.button(field_x, y + h - 340, field_w, 44, f"Game Mode: {self.gamemode.capitalize()}", "toggle-gamemode", self.hover)
        if self.message:
            frame.label(self.message, x + w / 2, y + 104, 15, "center", "center", (232, 164, 120, 255))
        self._row_buttons(
            frame,
            x + 48,
            y + 36,
            [("Create", "create", True), ("Back", "back-worlds", True)],
            w - 96,
        )
        return frame

    def _confirm(self, width, height):
        frame = Frame()
        frame.rect(0, 0, width, height, DIM)
        x, y, w, h = self._panel(frame, width, height, 460, 220)
        world = self.selected_world()
        name = world.name if world else "this world"
        frame.label("Delete this world?", x + w / 2, y + h - 58, 26, "center", "center")
        frame.label(name, x + w / 2, y + h - 98, 16, "center", "center", MUTED)
        self._row_buttons(
            frame,
            x + 36,
            y + 32,
            [("Delete", "confirm-delete", True), ("Cancel", "cancel-delete", True)],
            w - 72,
        )
        return frame

    def _pause(self, width, height):
        frame = Frame()
        frame.rect(0, 0, width, height, DIM)
        x, y, w, h = self._panel(frame, width, height, 420, 360)
        frame.label("Paused", x + w / 2, y + h - 54, 34, "center", "center")
        self._stack(
            frame,
            x + (w - 260) / 2,
            y + 44,
            [("Resume", "resume", True), ("Options", "options", True), ("Save and Quit", "save-quit", True)],
        )
        return frame

    def _options(self, width, height):
        frame = Frame()
        x, y, w, h = self._panel(frame, width, height, 520, 420)
        frame.label("Options", x + w / 2, y + h - 46, 32, "center", "center")
        field_w = 340
        field_x = x + (w - field_w) / 2

        status_text = f"Web UI: {'ON' if self.web_ui_enabled else 'OFF'}"
        frame.button(field_x, y + h - 108, field_w, 44, status_text, "toggle-web-ui", self.hover)

        frame.label("Web UI Port", field_x, y + h - 144, 14, "left", "center", MUTED)
        self._field(frame, field_x, y + h - 192, field_w, 40, "web_ui_port", str(self.web_ui_port), "5000")

        if self.web_ui_enabled:
            port = self.web_ui_port
            ips = getattr(self, "local_ips", []) or []
            if ips:
                frame.label("Phone / Tablet URL:", x + w / 2, y + h - 230, 13, "center", "center", MUTED)
                frame.label(f"http://{ips[0]}:{port}", x + w / 2, y + h - 252, 16, "center", "center", TITLE)
                if len(ips) > 1:
                    frame.label(f"Alt: http://{ips[1]}:{port}   |   localhost:{port}", x + w / 2, y + h - 274, 12, "center", "center", MUTED)
                else:
                    frame.label(f"Local: http://localhost:{port}", x + w / 2, y + h - 274, 12, "center", "center", MUTED)
            else:
                frame.label(f"Running: http://localhost:{port}", x + w / 2, y + h - 234, 15, "center", "center", TITLE)
                frame.label("(Connect to Wi-Fi to access from your phone or tablet)", x + w / 2, y + h - 258, 12, "center", "center", MUTED)

        self._row_buttons(
            frame,
            field_x,
            y + 26,
            [("Done", "back-options", True)],
            field_w,
        )
        return frame

    def _field(self, frame, x, y, w, h, field, value, placeholder):
        focused = self.focus == field
        if focused:
            frame.rect(x - 2, y - 2, w + 4, h + 4, FIELD_FOCUS)
        frame.rect(x, y, w, h, FIELD)
        shown = value if value else placeholder
        color = INK if value else MUTED
        if focused and value:
            shown = value + "|"
        elif focused and not value:
            shown = "|"
            color = INK
        frame.label(shown, x + 12, y + h / 2, 18, "left", "center", color)
        frame.hit(x, y, w, h, f"focus:{field}")

    def _stack(self, frame, x, bottom, items, button_w=280):
        cursor = bottom
        for text, action, enabled in reversed(items):
            frame.button(x, cursor, button_w, 48, text, action, self.hover, enabled)
            cursor += 62

    def _row_buttons(self, frame, x, y, items, width):
        gap = 10
        count = len(items)
        button_w = (width - gap * (count - 1)) / count
        for index, (text, action, enabled) in enumerate(items):
            frame.button(x + index * (button_w + gap), y, button_w, 46, text, action, self.hover, enabled)


def format_played(timestamp):
    if not timestamp:
        return "Not played"
    return time.strftime("%b %d, %Y  %H:%M", time.localtime(timestamp))
