import tempfile
import time
import unittest

from client.chat import (
    ChatMessage,
    ChatState,
    CHAT_FADE_SECONDS,
    DEFAULT_TEXT_COLOR,
    SYSTEM_COLOR,
)
from client.network import NetworkClient
from server.commands import CommandContext, CommandHandler
from server.network import MultiplayerServer
from server.player import Player
from server.world import World


class ChatStateTests(unittest.TestCase):
    def test_chat_message_formatting(self):
        msg1 = ChatMessage("Hello world", sender="Alice")
        self.assertEqual(msg1.formatted_text, "<Alice> Hello world")
        self.assertEqual(msg1.color, DEFAULT_TEXT_COLOR)

        msg2 = ChatMessage("World Seed: 42", sender="")
        self.assertEqual(msg2.formatted_text, "World Seed: 42")
        self.assertEqual(msg2.color, SYSTEM_COLOR)

    def test_chat_state_add_message_and_pruning(self):
        state = ChatState()
        for i in range(120):
            state.add_message(f"Message {i}", sender="Bob")
        self.assertEqual(len(state.messages), 100)
        self.assertEqual(state.messages[-1].text, "Message 119")

    def test_chat_state_open_close_input(self):
        state = ChatState()
        self.assertFalse(state.open)
        state.open_chat("/")
        self.assertTrue(state.open)
        self.assertEqual(state.input_text, "/")

        state.add_char("h")
        state.add_char("e")
        state.add_char("l")
        state.add_char("p")
        self.assertEqual(state.input_text, "/help")

        state.backspace()
        self.assertEqual(state.input_text, "/hel")

        line = state.send_current()
        self.assertEqual(line, "/hel")
        self.assertFalse(state.open)
        self.assertEqual(state.input_text, "")

    def test_chat_cursor_navigation_and_middle_insertion(self):
        state = ChatState()
        state.open_chat("helo")
        self.assertEqual(state.cursor, 4)

        # Move left twice to between 'e' and 'l'
        state.cursor_left()
        self.assertEqual(state.cursor, 3)
        state.cursor_left()
        self.assertEqual(state.cursor, 2)

        # Insert 'l' -> "hello"
        state.add_char("l")
        self.assertEqual(state.input_text, "hello")
        self.assertEqual(state.cursor, 3)

        # Move left to beginning and test clamping
        state.cursor_home()
        self.assertEqual(state.cursor, 0)
        state.cursor_left()
        self.assertEqual(state.cursor, 0)

        # Insert at start
        state.add_char("/")
        self.assertEqual(state.input_text, "/hello")
        self.assertEqual(state.cursor, 1)

        # Move right and test delete in middle
        state.cursor_right()  # cursor is at 2 (after 'h')
        self.assertEqual(state.cursor, 2)
        state.delete()  # deletes 'e'
        self.assertEqual(state.input_text, "/hllo")
        self.assertEqual(state.cursor, 2)

        # Backspace at cursor 2 deletes 'h'
        state.backspace()
        self.assertEqual(state.input_text, "/llo")
        self.assertEqual(state.cursor, 1)

        # Cursor end and clamping
        state.cursor_end()
        self.assertEqual(state.cursor, 4)
        state.cursor_right()
        self.assertEqual(state.cursor, 4)

    def test_chat_history_navigation(self):
        state = ChatState()
        state.open_chat("/first")
        state.send_current()
        state.open_chat("/second")
        state.send_current()

        self.assertEqual(state.history, ["/first", "/second"])

        # Navigate history
        state.open_chat()
        state.history_prev()
        self.assertEqual(state.input_text, "/second")
        state.history_prev()
        self.assertEqual(state.input_text, "/first")
        state.history_next()
        self.assertEqual(state.input_text, "/second")
        state.history_next()
        self.assertEqual(state.input_text, "")

    def test_chat_layout(self):
        state = ChatState()
        state.add_message("Line 1", sender="Alice")
        state.add_message("Line 2", sender="")

        # Closed chat layout
        rects, labels = state.layout(800, 600, 1.0)
        self.assertEqual(len(labels), 2)
        # In closed mode with no open input box, rects is empty
        self.assertEqual(len(rects), 0)

        # Open chat layout
        state.open_chat("hi")
        rects, labels = state.layout(800, 600, 1.0)
        # Rects include message background + input box background
        self.assertGreaterEqual(len(rects), 1)
        # Labels include messages + input box label
        self.assertEqual(len(labels), 3)


class CommandHandlerTests(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=98765)
        self.player = Player(self.world)
        self.player.pos = [10.5, 64.0, -20.5]
        self.handler = CommandHandler()

    def test_help_command(self):
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/help")
        self.assertTrue(handled)
        texts = [r[0] for r in ctx.responses]
        self.assertTrue(any("Available Commands" in t for t in texts))
        self.assertTrue(any("/help" in t for t in texts))
        self.assertTrue(any("/tp" in t for t in texts))

    def test_help_specific_command(self):
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/help tp")
        self.assertTrue(handled)
        self.assertTrue(any("/tp <x> <y> <z>" in r[0] for r in ctx.responses))

    def test_seed_command(self):
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/seed")
        self.assertTrue(handled)
        self.assertTrue(any("98765" in r[0] for r in ctx.responses))

    def test_pos_command(self):
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/pos")
        self.assertTrue(handled)
        self.assertTrue(any("10.50" in r[0] and "64.00" in r[0] and "-20.50" in r[0] for r in ctx.responses))

    def test_tp_command(self):
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/tp 100 80 200")
        self.assertTrue(handled)
        self.assertEqual(self.player.pos, [100.0, 80.0, 200.0])
        self.assertTrue(any("Teleported to 100.0, 80.0, 200.0" in r[0] for r in ctx.responses))

    def test_unknown_command(self):
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/foobar")
        self.assertTrue(handled)
        self.assertTrue(any("Unknown command: /foobar" in r[0] for r in ctx.responses))

    def test_custom_command_registration(self):
        def cmd_heal(context, args):
            context.reply("Healed to full!")

        self.handler.register("heal", cmd_heal, "Restores health", usage="/heal")
        ctx = CommandContext(player=self.player, world=self.world)
        self.handler.execute(ctx, "/heal")
        self.assertTrue(any("Healed to full!" in r[0] for r in ctx.responses))

    def test_gamemode_command_sets_creative(self):
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/gamemode creative")
        self.assertTrue(handled)
        self.assertEqual(self.player.gamemode, "creative")
        self.assertTrue(any("Game mode set to creative" in r[0] for r in ctx.responses))

    def test_gamemode_command_sets_survival(self):
        self.player.gamemode = "creative"
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/gamemode survival")
        self.assertTrue(handled)
        self.assertEqual(self.player.gamemode, "survival")
        self.assertTrue(any("Game mode set to survival" in r[0] for r in ctx.responses))

    def test_gamemode_command_rejects_unknown_mode(self):
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/gamemode hardcore")
        self.assertTrue(handled)
        self.assertTrue(any("Game mode must be creative or survival" in r[0] for r in ctx.responses))
        self.assertIsNone(self.player.gamemode)

    def test_gamemode_command_requires_a_mode(self):
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/gamemode")
        self.assertTrue(handled)
        self.assertTrue(any("Usage: /gamemode" in r[0] for r in ctx.responses))

    def test_spawnpoint_command_sets_current_position(self):
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/spawnpoint")
        self.assertTrue(handled)
        self.assertEqual(self.player.get_spawn_point(), [10.5, 64.0, -20.5])
        self.assertTrue(any("Personal spawn point set to (10.5, 64.0, -20.5)" in r[0] for r in ctx.responses))

    def test_spawnpoint_command_accepts_coordinates(self):
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/spawnpoint 100 70 -50")
        self.assertTrue(handled)
        self.assertEqual(self.player.get_spawn_point(), [100.0, 70.0, -50.0])

    def test_sethome_alias(self):
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/sethome 12 34 56")
        self.assertTrue(handled)
        self.assertEqual(self.player.get_spawn_point(), [12.0, 34.0, 56.0])

    def test_spawnpoint_clear(self):
        self.player.set_spawn_point(1.0, 2.0, 3.0)
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/spawnpoint clear")
        self.assertTrue(handled)
        self.assertIsNone(self.player.get_spawn_point())
        self.assertTrue(any("cleared" in r[0].lower() for r in ctx.responses))

    def test_home_command_teleports_to_spawn_point(self):
        self.player.set_spawn_point(200.0, 80.0, 300.0)
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/home")
        self.assertTrue(handled)
        self.assertEqual(self.player.pos, [200.0, 80.0, 300.0])
        self.assertEqual(self.player.vel, [0.0, 0.0, 0.0])

    def test_home_command_requires_spawn_point(self):
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/home")
        self.assertTrue(handled)
        self.assertTrue(any("No personal spawn point set" in r[0] for r in ctx.responses))

    def test_delhome_clears_spawn_point(self):
        self.player.set_spawn_point(1.0, 2.0, 3.0)
        ctx = CommandContext(player=self.player, world=self.world)
        handled = self.handler.execute(ctx, "/delhome")
        self.assertTrue(handled)
        self.assertIsNone(self.player.get_spawn_point())

    def test_spawnpoint_rejects_non_numeric(self):
        ctx = CommandContext(player=self.player, world=self.world)
        self.handler.execute(ctx, "/spawnpoint a b c")
        self.assertIsNone(self.player.get_spawn_point())
        self.assertTrue(any("must be numbers" in r[0] for r in ctx.responses))

    def test_gamemode_command_targets_other_player(self):
        class FakeClient:
            def __init__(self, uuid, username):
                self.uuid = uuid
                self.username = username
                self.alive = True

            def send(self, packet):
                self.last_packet = packet

        class FakeServer:
            clients = {"u2": FakeClient("u2", "Bob")}

        other = Player(self.world)
        other.apply_identity("u2", None)
        self.world.players["u2"] = other
        ctx = CommandContext(sender="Alice", player=self.player, world=self.world, server=FakeServer())
        handled = self.handler.execute(ctx, "/gamemode creative Bob")
        self.assertTrue(handled)
        self.assertEqual(other.gamemode, "creative")
        self.assertTrue(any("Game mode set to creative for" in r[0] for r in ctx.responses))


class NetworkChatIntegrationTests(unittest.TestCase):
    def test_multiplayer_chat_and_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = MultiplayerServer(host="127.0.0.1", port=0, saves_dir=tmp, world_name="chattest", seed=42)
            server.start()
            port = server.actual_port

            try:
                # Client 1 connects
                client1 = NetworkClient("127.0.0.1", port, "uuid-1", "Alice")
                self.assertTrue(client1.connect(timeout=2.0))

                time.sleep(0.15)
                server.tick()
                ev1 = client1.poll_events()
                # Should have received login_ack and join announcement
                join_events = [e for e in ev1 if e.get("type") == "chat"]
                self.assertTrue(any("Alice joined the game" in e.get("text", "") for e in join_events))

                # Client 2 connects
                client2 = NetworkClient("127.0.0.1", port, "uuid-2", "Bob")
                self.assertTrue(client2.connect(timeout=2.0))

                time.sleep(0.15)
                server.tick()

                # Client 1 should receive Bob joined notice
                ev1 = client1.poll_events()
                join_bob = [e for e in ev1 if e.get("type") == "chat" and "Bob joined the game" in e.get("text", "")]
                self.assertTrue(len(join_bob) >= 1)

                # Client 1 sends a chat message
                client1.send_chat("Hello everyone!")
                time.sleep(0.1)
                server.tick()

                # Client 2 should receive the chat message from Alice
                ev2 = client2.poll_events()
                chat_msgs = [e for e in ev2 if e.get("type") == "chat" and e.get("sender") == "Alice"]
                self.assertEqual(len(chat_msgs), 1)
                self.assertEqual(chat_msgs[0]["text"], "Hello everyone!")

                # Client 2 runs a command: /seed
                client2.send_chat("/seed")
                time.sleep(0.1)
                server.tick()

                ev2 = client2.poll_events()
                seed_replies = [e for e in ev2 if e.get("type") == "chat" and "World Seed: 42" in e.get("text", "")]
                self.assertEqual(len(seed_replies), 1)

                # Client 1 runs /say Hello world!
                client1.send_chat("/say Hello world!")
                time.sleep(0.1)
                server.tick()

                ev1 = client1.poll_events()
                ev2 = client2.poll_events()
                say_msgs_c1 = [e for e in ev1 if e.get("type") == "chat" and "[Alice] Hello world!" in e.get("text", "")]
                say_msgs_c2 = [e for e in ev2 if e.get("type") == "chat" and "[Alice] Hello world!" in e.get("text", "")]
                self.assertEqual(len(say_msgs_c1), 1)
                self.assertEqual(len(say_msgs_c2), 1)

                # Disconnect client 2, client 1 receives leave notice
                client2.disconnect()
                time.sleep(0.1)
                server.tick()

                ev1 = client1.poll_events()
                leave_bob = [e for e in ev1 if e.get("type") == "chat" and "Bob left the game" in e.get("text", "")]
                self.assertTrue(len(leave_bob) >= 1)

                client1.disconnect()

            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()

