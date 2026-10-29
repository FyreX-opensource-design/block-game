import time

from client.menu import LabelSpec

CHAT_FADE_SECONDS = 9.0
MAX_DISPLAYED_MESSAGES = 10
CHAT_INPUT_MAX_LEN = 120

DEFAULT_TEXT_COLOR = (244, 240, 230, 255)
SYSTEM_COLOR = (255, 215, 0, 255)
ERROR_COLOR = (255, 100, 100, 255)
BACKGROUND_COLOR = (0.05, 0.06, 0.04, 0.55)
INPUT_BG_COLOR = (0.04, 0.05, 0.03, 0.85)


class ChatMessage:
    def __init__(self, text, sender="", color=None, timestamp=None):
        self.text = text
        self.sender = sender
        self.color = color or (DEFAULT_TEXT_COLOR if sender else SYSTEM_COLOR)
        self.timestamp = timestamp if timestamp is not None else time.monotonic()

    @property
    def formatted_text(self):
        if self.sender:
            return f"<{self.sender}> {self.text}"
        return self.text


class ChatState:
    def __init__(self):
        self.messages = []
        self.open = False
        self.input_text = ""
        self.cursor = 0
        self.history = []
        self.history_index = -1
        self._temp_input = ""

    def add_message(self, text, sender="", color=None):
        msg = ChatMessage(text, sender=sender, color=color)
        self.messages.append(msg)
        if len(self.messages) > 100:
            self.messages.pop(0)

    def open_chat(self, initial_text=""):
        self.open = True
        self.input_text = initial_text
        self.cursor = len(initial_text)
        self.history_index = -1
        self._temp_input = ""

    def close_chat(self):
        self.open = False
        self.input_text = ""
        self.cursor = 0
        self.history_index = -1

    def send_current(self):
        text = self.input_text.strip()
        self.close_chat()
        if not text:
            return None
        # Add to history
        if not self.history or self.history[-1] != text:
            self.history.append(text)
            if len(self.history) > 50:
                self.history.pop(0)
        return text

    def history_prev(self):
        if not self.history:
            return
        if self.history_index == -1:
            self._temp_input = self.input_text
            self.history_index = len(self.history) - 1
        elif self.history_index > 0:
            self.history_index -= 1
        self.input_text = self.history[self.history_index]
        self.cursor = len(self.input_text)

    def history_next(self):
        if self.history_index == -1:
            return
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.input_text = self.history[self.history_index]
        else:
            self.history_index = -1
            self.input_text = self._temp_input
        self.cursor = len(self.input_text)

    def cursor_left(self):
        self.cursor = max(0, self.cursor - 1)

    def cursor_right(self):
        self.cursor = min(len(self.input_text), self.cursor + 1)

    def cursor_home(self):
        self.cursor = 0

    def cursor_end(self):
        self.cursor = len(self.input_text)

    def backspace(self):
        if self.cursor > 0:
            self.input_text = self.input_text[: self.cursor - 1] + self.input_text[self.cursor :]
            self.cursor -= 1

    def delete(self):
        if self.cursor < len(self.input_text):
            self.input_text = self.input_text[: self.cursor] + self.input_text[self.cursor + 1 :]

    def add_char(self, char):
        if len(self.input_text) + len(char) <= CHAT_INPUT_MAX_LEN:
            self.input_text = self.input_text[: self.cursor] + char + self.input_text[self.cursor :]
            self.cursor += len(char)

    def layout(self, width, height, scale):
        """Build rects and labels to draw.
        
        Returns (rects, labels).
        """
        rects = []
        labels = []
        now = time.monotonic()

        # Messages positioning
        msg_x = 16
        base_y = 88  # Just above the hotbar (which is ~58px high)
        line_height = 20
        chat_w = min(560, width - 32)

        # Select messages to display
        if self.open:
            visible_msgs = self.messages[-MAX_DISPLAYED_MESSAGES:]
        else:
            # Only messages within fade time
            visible_msgs = [m for m in self.messages if now - m.timestamp < CHAT_FADE_SECONDS]
            visible_msgs = visible_msgs[-MAX_DISPLAYED_MESSAGES:]

        if visible_msgs and self.open:
            total_h = len(visible_msgs) * line_height + 8
            rects.append((msg_x - 4, base_y - 4, chat_w + 8, total_h, BACKGROUND_COLOR))

        # Render message lines from bottom to top
        for idx, msg in enumerate(reversed(visible_msgs)):
            y = base_y + idx * line_height + 4
            # If closed, apply alpha fadeout on older messages
            color = msg.color
            if not self.open:
                age = now - msg.timestamp
                if age > CHAT_FADE_SECONDS - 2.0:
                    alpha_factor = max(0.0, (CHAT_FADE_SECONDS - age) / 2.0)
                    color = (color[0], color[1], color[2], int(color[3] * alpha_factor))
            labels.append(LabelSpec(msg.formatted_text, msg_x, y, size=14, anchor_x="left", anchor_y="bottom", color=color))

        # If chat is open, draw input box below message area
        if self.open:
            input_y = 48
            input_h = 28
            rects.append((msg_x - 4, input_y, chat_w + 8, input_h, INPUT_BG_COLOR))
            # Blinking cursor at current insertion index
            cursor_str = "|" if int(now * 2) % 2 == 0 else ""
            pos = max(0, min(len(self.input_text), self.cursor))
            display_text = self.input_text[:pos] + cursor_str + self.input_text[pos:]
            labels.append(LabelSpec(display_text, msg_x, input_y + input_h / 2, size=15, anchor_x="left", anchor_y="center", color=(255, 255, 255, 255)))

        return rects, labels

