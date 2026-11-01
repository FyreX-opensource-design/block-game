import tempfile
import time
import unittest

import numpy as np

from client.menu import MenuState
from client.network import NetworkClient
from server.blocks import AIR, LOAD_RADIUS, STONE, UNLOAD_RADIUS
from server.network import (
    MultiplayerServer,
    decode_chunk,
    decode_skin,
    encode_chunk,
    encode_skin,
)
from server.profile import load_skin_rgba
from server.servers import load_servers, parse_address, save_servers
from server.world import Chunk, World


class ServerAddressTests(unittest.TestCase):
    def test_parse_address_defaults_and_ports(self):
        self.assertEqual(parse_address("localhost"), ("localhost", 25565))
        self.assertEqual(parse_address("127.0.0.1:25565"), ("127.0.0.1", 25565))
        self.assertEqual(parse_address("play.example.com:4242"), ("play.example.com", 4242))
        self.assertEqual(parse_address(""), ("127.0.0.1", 25565))
        self.assertEqual(parse_address("192.168.1.5:invalid"), ("192.168.1.5", 25565))

    def test_save_and_load_servers(self):
        with tempfile.TemporaryDirectory() as tmp:
            defaults = load_servers(tmp)
            self.assertEqual(len(defaults), 1)
            self.assertEqual(defaults[0]["name"], "Local Server")

            custom = [
                {"name": "Survival Hub", "address": "mc.example.com:25565"},
                {"name": "Creative LAN", "address": "192.168.1.100"},
            ]
            save_servers(custom, tmp)
            loaded = load_servers(tmp)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["name"], "Survival Hub")
            self.assertEqual(loaded[0]["address"], "mc.example.com:25565")
            self.assertEqual(loaded[1]["name"], "Creative LAN")
            self.assertEqual(loaded[1]["address"], "192.168.1.100")


class MenuMultiplayerTests(unittest.TestCase):
    def test_menu_multiplayer_navigation(self):
        menu = MenuState()
        frame = menu.layout(1280, 720)
        # Verify Multiplayer button exists on title
        actions = [hit.action for hit in frame.hits]
        self.assertIn("multiplayer", actions)

        menu.screen = "servers"
        menu.servers = [
            {"name": "Server A", "address": "1.1.1.1"},
            {"name": "Server B", "address": "2.2.2.2"},
        ]
        menu.selected = 0
        frame = menu.layout(1280, 720)
        actions = [hit.action for hit in frame.hits]
        self.assertIn("join-server", actions)
        self.assertIn("add-server", actions)
        self.assertIn("direct-connect", actions)

        # Selection
        menu.move_selection(1)
        self.assertEqual(menu.selected, 1)
        self.assertEqual(menu.selected_server()["name"], "Server B")

        # Add server screen
        menu.screen = "add_server"
        menu.server_name = "My Server"
        menu.server_address = "play.server.io"
        frame = menu.layout(1280, 720)
        actions = [hit.action for hit in frame.hits]
        self.assertIn("save-server", actions)
        self.assertIn("cancel-server", actions)


class SerializationTests(unittest.TestCase):
    def test_skin_roundtrip(self):
        skin = load_skin_rgba()
        encoded = encode_skin(skin)
        self.assertIsInstance(encoded, str)
        self.assertGreater(len(encoded), 50)
        decoded = decode_skin(encoded)
        self.assertIsNotNone(decoded)
        self.assertTrue(np.array_equal(skin, decoded))

    def test_chunk_roundtrip(self):
        chunk = Chunk(2, 4, 6)
        chunk.data[3, 5, 7] = 2  # dirt
        chunk.data[10, 10, 10] = 3  # stone
        encoded = encode_chunk(chunk.data)
        self.assertIsInstance(encoded, str)
        decoded = decode_chunk(encoded)
        self.assertTrue(np.array_equal(chunk.data, decoded))


class MultiPlayerChunkLoadingTests(unittest.TestCase):
    def test_chunks_load_and_evict_around_multiple_players(self):
        world = World(seed=999)
        # Player 1 at (0, 4, 0)
        # Player 2 at (40, 4, 40) (far apart)
        p1 = (0, 4, 0)
        p2 = (40, 4, 40)

        # Load around both players
        built_set = set()
        for _ in range(20):
            built, _ = world.sync_players([p1, p2])
            if built is not None:
                built_set.add(built)

        # Both areas should have chunks loaded
        near_p1 = [k for k in world.chunks if abs(k[0] - p1[0]) <= LOAD_RADIUS and abs(k[2] - p1[2]) <= LOAD_RADIUS]
        near_p2 = [k for k in world.chunks if abs(k[0] - p2[0]) <= LOAD_RADIUS and abs(k[2] - p2[2]) <= LOAD_RADIUS]
        self.assertGreater(len(near_p1), 0)
        self.assertGreater(len(near_p2), 0)

        # Move Player 2 away: Player 2 leaves (40, 4, 40) and goes near Player 1
        # Chunks near (40, 4, 40) should be evicted because NO player is near them!
        p2_new = (1, 4, 0)
        evicted = []
        for _ in range(50):
            _, removed = world.sync_players([p1, p2_new])
            evicted.extend(removed)

        # Chunks around (40, 4, 40) are now unloaded
        remaining_p2_old = [k for k in world.chunks if abs(k[0] - 40) <= LOAD_RADIUS and abs(k[2] - 40) <= LOAD_RADIUS]
        self.assertEqual(len(remaining_p2_old), 0)

        # But chunks around p1 remain loaded
        remaining_p1 = [k for k in world.chunks if abs(k[0] - 0) <= LOAD_RADIUS and abs(k[2] - 0) <= LOAD_RADIUS]
        self.assertGreater(len(remaining_p1), 0)


class NetworkIntegrationTests(unittest.TestCase):
    def test_multiplayer_clients_skin_and_world_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = MultiplayerServer(host="127.0.0.1", port=0, saves_dir=tmp, seed=12345)
            server.start()
            port = server.actual_port

            skin1 = load_skin_rgba()
            # Modify skin2 slightly to distinguish
            skin2 = skin1.copy()
            skin2[0, 0] = (255, 0, 0, 255)

            uuid1 = "11111111-1111-1111-1111-111111111111"
            uuid2 = "22222222-2222-2222-2222-222222222222"

            client1 = NetworkClient("127.0.0.1", port, uuid1, "Alice", skin1)
            client2 = NetworkClient("127.0.0.1", port, uuid2, "Bob", skin2)

            self.assertTrue(client1.connect())
            self.assertTrue(client2.connect())

            t0 = time.time()
            c1_received_c2_skin = None
            c2_received_c1_skin = None
            c1_received_chunks = []
            c2_received_chunks = []
            c1_received_move = False
            c2_received_block = False

            while time.time() - t0 < 4.0:
                for pkt in client1.poll_events():
                    if pkt.get("type") == "player_join" and pkt.get("uuid") == uuid2:
                        c1_received_c2_skin = decode_skin(pkt.get("skin"))
                    elif pkt.get("type") == "chunk":
                        c1_received_chunks.append((pkt["cx"], pkt["cy"], pkt["cz"]))
                    elif pkt.get("type") == "block_change" and pkt.get("block") == "planks":
                        c2_received_block = True

                for pkt in client2.poll_events():
                    if pkt.get("type") == "login_ack":
                        for p in pkt.get("players", []):
                            if p.get("uuid") == uuid1:
                                c2_received_c1_skin = decode_skin(p.get("skin"))
                    elif pkt.get("type") == "chunk":
                        c2_received_chunks.append((pkt["cx"], pkt["cy"], pkt["cz"]))
                    elif pkt.get("type") == "player_move" and pkt.get("uuid") == uuid1:
                        c1_received_move = True

                # Send test movement from Client 1
                client1.send_move([10.0, 32.0, 15.0], [0.0, 0.0, 0.0], 1.2, -0.4, "wood_pickaxe")

                # Send test block placement from Client 2
                client2.send_set_block(0, 10, 0, "planks")

                if (
                    c1_received_c2_skin is not None
                    and c2_received_c1_skin is not None
                    and len(c1_received_chunks) > 0
                    and len(c2_received_chunks) > 0
                    and c1_received_move
                    and c2_received_block
                ):
                    break
                time.sleep(0.05)

            # Verify skins were exchanged and match player UUIDs
            self.assertIsNotNone(c1_received_c2_skin, "Client 1 should receive Client 2 skin")
            self.assertTrue(np.array_equal(c1_received_c2_skin, skin2))
            self.assertIsNotNone(c2_received_c1_skin, "Client 2 should receive Client 1 skin")
            self.assertTrue(np.array_equal(c2_received_c1_skin, skin1))

            # Verify chunk streaming to each online player
            self.assertGreater(len(c1_received_chunks), 0)
            self.assertGreater(len(c2_received_chunks), 0)

            # Verify movement and block synchronization
            self.assertTrue(c1_received_move, "Client 2 should receive Client 1 movement")
            self.assertTrue(c2_received_block, "Client 1 should receive Client 2 block change")
            self.assertEqual(server.world.get_block(0, 10, 0), "planks")

            # Clean disconnect
            client2.disconnect()
            time.sleep(0.1)

            # Client 1 should observe player_leave for Client 2
            c1_saw_leave = False
            for pkt in client1.poll_events():
                if pkt.get("type") == "player_leave" and pkt.get("uuid") == uuid2:
                    c1_saw_leave = True
            self.assertTrue(c1_saw_leave)
            self.assertNotIn(uuid2, server.clients)

            client1.disconnect()
            server.stop()

    def test_bootstrap_step_safe_when_player_is_none(self):
        from unittest.mock import MagicMock
        from client.app import Game
        mock_game = MagicMock()
        mock_game.player = None
        mock_game.world = None
        mock_game.mode = "loading"
        mock_game.network_client = "mock_client"
        mock_game._network_tick = MagicMock()
        mock_game._bootstrap_step = lambda: Game._bootstrap_step(mock_game)

        # Calling tick should call _network_tick and skip _bootstrap_step
        Game.tick(mock_game, 0.016)
        mock_game._network_tick.assert_called_once()

        # Calling _bootstrap_step directly should safely return without AttributeError
        res = Game._bootstrap_step(mock_game)
        self.assertIsNone(res)

    def test_server_custom_seed_and_existing_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            s1 = MultiplayerServer(host="127.0.0.1", port=0, world_name="Alpha", seed=777888, saves_dir=tmp)
            s1.start()
            self.assertEqual(s1.world.seed, 777888)
            s1.stop()

            # Starting again with the same world name re-loads the existing world and preserves its seed
            s2 = MultiplayerServer(host="127.0.0.1", port=0, world_name="Alpha", seed=None, saves_dir=tmp)
            s2.start()
            self.assertEqual(s2.world.seed, 777888)
            s2.stop()

    def test_client_detects_server_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = MultiplayerServer(host="127.0.0.1", port=0, world_name="TestShutdown", saves_dir=tmp)
            server.start()
            port = server.actual_port

            client = NetworkClient("127.0.0.1", port, "uuid-test", "Player1")
            self.assertTrue(client.connect(timeout=2.0))
            self.assertTrue(client.connected)

            # Turn off server
            server.stop()

            # Client should detect disconnection within ~0.2s
            for _ in range(20):
                time.sleep(0.02)
                if not client.connected:
                    break

            self.assertFalse(client.connected)
            self.assertIsNotNone(client.error)

    def test_player_location_persisted_on_disconnect_and_reconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = MultiplayerServer(host="127.0.0.1", port=0, world_name="PersistTest", saves_dir=tmp)
            server.start()
            port = server.actual_port
            uuid = "test-player-uuid-1"

            # 1. Connect and move
            client = NetworkClient("127.0.0.1", port, uuid, "Explorer")
            self.assertTrue(client.connect(timeout=2.0))
            time.sleep(0.05)
            # Drain initial login_ack
            client.poll_events()

            target_pos = [42.5, 64.0, -18.5]
            client.send_move(target_pos, [0.0, 0.0, 0.0], 1.57, -0.3, "")
            time.sleep(0.08)

            # 2. Disconnect
            client.disconnect()
            time.sleep(0.05)

            # 3. Reconnect with same UUID
            client2 = NetworkClient("127.0.0.1", port, uuid, "Explorer")
            self.assertTrue(client2.connect(timeout=2.0))

            login_ack = None
            t0 = time.time()
            while time.time() - t0 < 2.0:
                for pkt in client2.poll_events():
                    if pkt.get("type") == "login_ack":
                        login_ack = pkt
                        break
                if login_ack:
                    break
                time.sleep(0.02)

            self.assertIsNotNone(login_ack)
            self.assertAlmostEqual(login_ack["pos"][0], 42.5)
            self.assertAlmostEqual(login_ack["pos"][1], 64.0)
            self.assertAlmostEqual(login_ack["pos"][2], -18.5)
            self.assertAlmostEqual(login_ack["yaw"], 1.57, delta=0.01)
            self.assertAlmostEqual(login_ack["pitch"], -0.3, delta=0.01)

            client2.disconnect()
            server.stop()

    def test_player_location_persisted_across_server_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            uuid = "test-player-uuid-2"
            target_pos = [100.5, 75.0, 200.5]

            # Session 1: connect, move, and stop server
            s1 = MultiplayerServer(host="127.0.0.1", port=0, world_name="RestartTest", saves_dir=tmp)
            s1.start()
            c1 = NetworkClient("127.0.0.1", s1.actual_port, uuid, "Traveler")
            self.assertTrue(c1.connect(timeout=2.0))
            time.sleep(0.05)
            c1.poll_events()

            c1.send_move(target_pos, [0.0, 0.0, 0.0], 2.1, 0.5, "")
            time.sleep(0.08)

            # Stop server (should save all player data to disk)
            s1.stop()
            c1.disconnect()

            # Session 2: start server again for same world
            s2 = MultiplayerServer(host="127.0.0.1", port=0, world_name="RestartTest", saves_dir=tmp)
            s2.start()

            c2 = NetworkClient("127.0.0.1", s2.actual_port, uuid, "Traveler")
            self.assertTrue(c2.connect(timeout=2.0))

            login_ack = None
            t0 = time.time()
            while time.time() - t0 < 2.0:
                for pkt in c2.poll_events():
                    if pkt.get("type") == "login_ack":
                        login_ack = pkt
                        break
                if login_ack:
                    break
                time.sleep(0.02)

            self.assertIsNotNone(login_ack)
            self.assertAlmostEqual(login_ack["pos"][0], 100.5)
            self.assertAlmostEqual(login_ack["pos"][1], 75.0)
            self.assertAlmostEqual(login_ack["pos"][2], 200.5)
            self.assertAlmostEqual(login_ack["yaw"], 2.1, delta=0.01)
            self.assertAlmostEqual(login_ack["pitch"], 0.5, delta=0.01)

            c2.disconnect()
            s2.stop()

    def test_personal_spawnpoint_persists_and_home_teleports(self):
        with tempfile.TemporaryDirectory() as tmp:
            uuid = "spawn-player-uuid"
            spawn = [12.5, 70.0, -34.5]

            server = MultiplayerServer(host="127.0.0.1", port=0, world_name="SpawnTest", saves_dir=tmp)
            server.start()
            port = server.actual_port

            def wait_for(client, wanted, timeout=2.0):
                found = []
                t0 = time.time()
                while time.time() - t0 < timeout:
                    server.tick()
                    for pkt in client.poll_events():
                        if pkt.get("type") == wanted:
                            found.append(pkt)
                    if found:
                        return found
                    time.sleep(0.02)
                return found

            client = NetworkClient("127.0.0.1", port, uuid, "Homebody")
            self.assertTrue(client.connect(timeout=2.0))
            self.assertTrue(wait_for(client, "login_ack"))

            # Set a personal spawn point via command
            client.send_chat(f"/spawnpoint {spawn[0]} {spawn[1]} {spawn[2]}")
            replies = wait_for(client, "chat")
            self.assertTrue(any("Personal spawn point set" in p.get("text", "") for p in replies))

            player = server.world.players.get(uuid)
            self.assertIsNotNone(player)
            self.assertEqual(player.get_spawn_point(), spawn)
            self.assertEqual(server.players_data[uuid]["spawn"], spawn)

            # Walk away, disconnect, and rejoin: personal spawn wins over last position
            client.send_move([50.5, 70.0, 50.5], [0.0, 0.0, 0.0], 0.0, 0.0, "")
            time.sleep(0.08)
            server.tick()
            client.disconnect()
            time.sleep(0.08)
            server.tick()

            client2 = NetworkClient("127.0.0.1", port, uuid, "Homebody")
            self.assertTrue(client2.connect(timeout=2.0))
            ack = wait_for(client2, "login_ack")
            self.assertTrue(ack)
            self.assertEqual([round(v, 1) for v in ack[0]["pos"]], spawn)

            # /home teleports the client back to the personal spawn point
            client2.send_move([80.5, 72.0, 80.5], [0.0, 0.0, 0.0], 0.0, 0.0, "")
            time.sleep(0.08)
            server.tick()
            client2.send_chat("/home")
            teleports = wait_for(client2, "teleport")
            self.assertTrue(teleports)
            self.assertEqual([round(v, 1) for v in teleports[0]["pos"]], spawn)

            client2.disconnect()
            server.stop()

    def test_game_returns_to_menu_on_server_shutdown_during_play(self):
        from unittest.mock import MagicMock
        from client.app import Game
        from client.menu import MenuState

        mock_game = MagicMock()
        mock_game.mode = "play"
        mock_game.ready = True
        mock_game.menu = MenuState()
        mock_game.other_players = {"other-uuid": "dummy"}
        mock_game.keys = {1, 2}
        mock_game._smoke = True

        mock_client = MagicMock()
        mock_client.poll_events.return_value = []
        mock_client.connected = False
        mock_client.error = "Connection reset by server"
        mock_game.network_client = mock_client

        # Bind methods from Game
        mock_game._handle_server_disconnect = lambda msg: Game._handle_server_disconnect(mock_game, msg)
        mock_game._drop_view = MagicMock()

        # Run network tick while in play mode
        Game._network_tick(mock_game, 0.016)

        self.assertIsNone(mock_game.network_client)
        self.assertEqual(mock_game.mode, "servers")
        self.assertEqual(mock_game.menu.screen, "servers")
        self.assertEqual(len(mock_game.other_players), 0)
        self.assertEqual(len(mock_game.keys), 0)
        self.assertTrue(mock_game.menu.message.startswith("Disconnected: Connection reset by server"))
        mock_game._drop_view.assert_called_once()

    def test_chunk_sync_priority_top_down_and_distance(self):
        from server.network import MultiplayerServer, RemoteClient
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmp:
            server = MultiplayerServer(host="127.0.0.1", port=0, saves_dir=tmp, world=World(seed=42))
            mock_client = MagicMock(spec=RemoteClient)
            mock_client.alive = True
            mock_client.uuid = "player-1"
            mock_client.pos = [0.0, 32.0, 0.0]  # Chunk (0, 2, 0)
            mock_client.sent_chunks = set()
            sent_packets = []
            mock_client.send = lambda pkt: sent_packets.append(pkt)

            server.clients["player-1"] = mock_client

            # Manually populate chunks at cx=0, cz=0 for ty=0..3 and cx=1, cz=0 for ty=0..3
            for cx in (0, 1):
                for ty in range(4):
                    server.world.chunks[(cx, ty, 0)] = Chunk(cx, ty, 0)

            server._sync_player_chunks()

            sent_keys = [(pkt["cx"], pkt["cy"], pkt["cz"]) for pkt in sent_packets if pkt.get("type") == "chunk"]
            self.assertGreater(len(sent_keys), 0)

            # Within column (0, 0), higher chunks should be sent before lower chunks
            col0_chunks = [k for k in sent_keys if k[0] == 0 and k[2] == 0]
            col0_ys = [k[1] for k in col0_chunks]
            self.assertEqual(col0_ys, sorted(col0_ys, reverse=True))

            # Column (0, 0) (distance 0) should be sent before column (1, 0) (distance 1)
            first_col1_idx = next(i for i, k in enumerate(sent_keys) if k[0] == 1)
            last_col0_idx = max(i for i, k in enumerate(sent_keys) if k[0] == 0)
            self.assertLess(last_col0_idx, first_col1_idx)

    def test_client_chunk_reception_remembers_and_marks_dirty(self):
        from unittest.mock import MagicMock
        from client.app import Game
        from server.world import World, Chunk
        from server.network import encode_chunk

        mock_game = MagicMock()
        mock_game.world = World(seed=123)
        mock_game.renderer = MagicMock()
        mock_game.mode = "play"
        mock_game.ready = True
        mock_game._upload = MagicMock()
        mock_game._net_timer = 0.0

        # Pre-populate lower chunk (0, 1, 0)
        lower_ch = Chunk(0, 1, 0)
        mock_game.world._remember(lower_ch)
        lower_ch.light_valid = True
        mock_game.world.dirty.clear()

        # Receive top chunk (0, 2, 0)
        top_data = np.zeros((16, 16, 16), dtype=np.uint8)
        pkt = {
            "type": "chunk",
            "cx": 0,
            "cy": 2,
            "cz": 0,
            "data": encode_chunk(top_data),
        }
        mock_client = MagicMock()
        mock_client.poll_events.return_value = [pkt]
        mock_client.connected = True
        mock_game.network_client = mock_client

        Game._network_tick(mock_game, 0.016)

        # Verify column registered
        self.assertIn((0, 0), mock_game.world.columns)
        self.assertEqual(mock_game.world.column_max[(0, 0)], 2)
        # Verify lower chunk had light invalidated and was marked dirty
        self.assertFalse(lower_ch.light_valid)
        self.assertIn((0, 1, 0), mock_game.world.dirty)


class MultiplayerFurnaceAndSlabSyncTests(unittest.TestCase):
    def test_mixed_double_slab_multiplayer_sync(self):
        from unittest.mock import MagicMock
        from server.blocks import can_place_block, get_double_slab
        from server.network import MultiplayerServer, RemoteClient
        from server.world import Chunk, World

        with tempfile.TemporaryDirectory() as tmp:
            world = World(seed=99)
            server = MultiplayerServer(host="127.0.0.1", port=0, saves_dir=tmp, world=world)
            chunk = Chunk(0, 0, 0)
            world.chunks[(0, 0, 0)] = chunk

            # Client 1
            client1 = MagicMock(spec=RemoteClient)
            client1.alive = True
            client1.uuid = "player-1"
            client1.pos = [0.0, 5.0, 0.0]
            client1_packets = []
            client1.send = lambda pkt: client1_packets.append(pkt)
            server.clients["player-1"] = client1

            # Client 2
            client2 = MagicMock(spec=RemoteClient)
            client2.alive = True
            client2.uuid = "player-2"
            client2.pos = [2.0, 5.0, 2.0]
            client2_packets = []
            client2.send = lambda pkt: client2_packets.append(pkt)
            server.clients["player-2"] = client2

            # Place single slab at (1, 1, 1)
            world.set_block(1, 1, 1, "stone_slab")
            self.assertEqual(world.get_block(1, 1, 1), "stone_slab")

            # 1. Combining slabs into mixed double slab via set_block packet
            mixed_double = get_double_slab("stone_slab", "cobble_slab")
            self.assertTrue(can_place_block(world, 1, 1, 1, mixed_double))

            client1_packets.clear()
            client2_packets.clear()
            server._handle_packet(client1, {"type": "set_block", "x": 1, "y": 1, "z": 1, "block": mixed_double})
            self.assertEqual(world.get_block(1, 1, 1), mixed_double)

            # Both clients receive block_change
            bcast1 = [p for p in client1_packets if p.get("type") == "block_change" and p.get("block") == mixed_double]
            bcast2 = [p for p in client2_packets if p.get("type") == "block_change" and p.get("block") == mixed_double]
            self.assertEqual(len(bcast1) + len(bcast2), 2)

            # 2. Breaking half a mixed double slab leaves remaining half slab
            remaining_slab = "stone_slab"  # or "cobble_slab_top"
            self.assertTrue(can_place_block(world, 1, 1, 1, remaining_slab))

            client1_packets.clear()
            client2_packets.clear()
            server._handle_packet(client1, {"type": "set_block", "x": 1, "y": 1, "z": 1, "block": remaining_slab})
            self.assertEqual(world.get_block(1, 1, 1), remaining_slab)

            bcast1 = [p for p in client1_packets if p.get("type") == "block_change" and p.get("block") == remaining_slab]
            bcast2 = [p for p in client2_packets if p.get("type") == "block_change" and p.get("block") == remaining_slab]
            self.assertEqual(len(bcast1) + len(bcast2), 2)

    def test_furnace_multiplayer_sync(self):
        from unittest.mock import MagicMock
        from server.network import MultiplayerServer, RemoteClient
        from server.stacks import make_stack
        from server.world import Chunk, World

        with tempfile.TemporaryDirectory() as tmp:
            world = World(seed=77)
            server = MultiplayerServer(host="127.0.0.1", port=0, saves_dir=tmp, world=world)
            chunk = Chunk(0, 0, 0)
            world.chunks[(0, 0, 0)] = chunk

            # Set furnace block in world
            world.set_block(2, 2, 2, "furnace")

            client1 = MagicMock(spec=RemoteClient)
            client1.alive = True
            client1.uuid = "p1"
            client1.pos = [2.0, 2.0, 2.0]
            client1.sent_chunks = set()
            c1_pkts = []
            client1.send = lambda pkt: c1_pkts.append(pkt)
            server.clients["p1"] = client1

            client2 = MagicMock(spec=RemoteClient)
            client2.alive = True
            client2.uuid = "p2"
            client2.pos = [2.5, 2.0, 2.5]
            client2.sent_chunks = set()
            c2_pkts = []
            client2.send = lambda pkt: c2_pkts.append(pkt)
            server.clients["p2"] = client2

            # 1. Client 1 opens furnace
            server._handle_packet(client1, {"type": "furnace_open", "x": 2, "y": 2, "z": 2})
            self.assertEqual(server.open_furnaces.get("p1"), (2, 2, 2))
            sync_c1 = [p for p in c1_pkts if p.get("type") == "furnace_sync"]
            self.assertEqual(len(sync_c1), 1)
            self.assertEqual(sync_c1[0]["furnace"]["x"], 2)

            # 2. Client 2 opens the same furnace
            server._handle_packet(client2, {"type": "furnace_open", "x": 2, "y": 2, "z": 2})
            self.assertEqual(server.open_furnaces.get("p2"), (2, 2, 2))

            # 3. Client 1 puts raw iron and coal in furnace
            c1_pkts.clear()
            c2_pkts.clear()
            in_stack = make_stack("raw_iron", 5)
            fuel_stack = make_stack("coal", 2)
            server._handle_packet(
                client1,
                {
                    "type": "furnace_update",
                    "x": 2,
                    "y": 2,
                    "z": 2,
                    "input": in_stack,
                    "fuel": fuel_stack,
                    "output": {},
                },
            )

            # Authoritative server furnace updated
            f_server = world.furnaces[(2, 2, 2)]
            self.assertEqual(f_server.input["id"], "raw_iron")
            self.assertEqual(f_server.input["count"], 5)
            self.assertEqual(f_server.fuel["id"], "coal")

            # Client 2 received furnace_sync with the new items
            sync_c2 = [p for p in c2_pkts if p.get("type") == "furnace_sync"]
            self.assertGreater(len(sync_c2), 0)
            self.assertEqual(sync_c2[-1]["furnace"]["input"]["id"], "raw_iron")

            # 4. Tick server: furnace starts cooking and consumes fuel
            c1_pkts.clear()
            c2_pkts.clear()
            server._furnace_sync_timer = 0.1  # Force periodic sync trigger
            server.tick()
            self.assertTrue(f_server.is_lit)

            # Both clients received periodic furnace_sync update
            sync_c1 = [p for p in c1_pkts if p.get("type") == "furnace_sync"]
            sync_c2 = [p for p in c2_pkts if p.get("type") == "furnace_sync"]
            self.assertGreater(len(sync_c1), 0)
            self.assertGreater(len(sync_c2), 0)

            # 5. Client closes furnace
            server._handle_packet(client1, {"type": "furnace_close", "x": 2, "y": 2, "z": 2})
            self.assertNotIn("p1", server.open_furnaces)

    def test_item_pickup_when_tabbed_out(self):
        from unittest.mock import MagicMock
        from server.network import MultiplayerServer, RemoteClient
        from server.world import Chunk, World

        with tempfile.TemporaryDirectory() as tmp:
            world = World(seed=12)
            server = MultiplayerServer(host="127.0.0.1", port=0, saves_dir=tmp, world=world)
            chunk = Chunk(0, 0, 0)
            world.chunks[(0, 0, 0)] = chunk

            # Connected client standing at (5.0, 10.0, 5.0)
            client = MagicMock(spec=RemoteClient)
            client.alive = True
            client.uuid = "tabbed-player"
            client.pos = [5.0, 10.0, 5.0]
            client.sent_chunks = set()
            client.out_queue = []
            pkts = []
            client.send = lambda pkt: pkts.append(pkt)
            server.clients["tabbed-player"] = client

            # Spawn item at (5.2, 10.5, 5.2) within pickup range (dist <= 1.6)
            item_entity = world.spawn_item("apple", count=3, pos=[5.2, 10.5, 5.2], pickup_delay=0.0)
            eid = item_entity.id
            self.assertIn(item_entity, world.entities)

            # Server tick runs without any client action (client is tabbed out / inactive)
            server.tick()

            # Item must be picked up server-side: removed from world, give_item sent to client
            self.assertNotIn(item_entity, world.entities)
            give_pkts = [p for p in pkts if p.get("type") == "give_item"]
            self.assertEqual(len(give_pkts), 1)
            self.assertEqual(give_pkts[0]["id"], eid)
            self.assertEqual(give_pkts[0]["item"], "apple")
            self.assertEqual(give_pkts[0]["count"], 3)

            # Verify remove_entity was also broadcast
            remove_pkts = [p for p in pkts if p.get("type") == "remove_entity" and p.get("id") == eid]
            self.assertEqual(len(remove_pkts), 1)

    def test_client_open_furnace_and_sync_no_crash(self):
        from unittest.mock import MagicMock
        from client.app import Game
        from server.furnace import Furnace
        from server.world import Chunk, World

        mock_game = MagicMock()
        mock_game.world = World(seed=1)
        c = Chunk(0, 0, 0)
        mock_game.world.chunks[(0, 0, 0)] = c
        mock_game.world.set_block(2, 2, 2, "furnace")

        mock_game.inventory = MagicMock()
        mock_game.inventory.container_type = "player"
        mock_game.inventory.furnace = None
        mock_game.chat = MagicMock()
        mock_game.chat.open = False
        mock_game.player = MagicMock()
        mock_game.player.pos = [0, 0, 0]
        mock_game.keys = set()
        mock_game._smoke = False
        mock_game.set_exclusive_mouse = MagicMock()
        mock_game._net_timer = 0.0
        mock_game.mode = "inventory"

        mock_net = MagicMock()
        mock_game.network_client = mock_net

        # Client opens furnace
        Game._set_inventory(mock_game, True, container_type="furnace", furnace_pos=(2, 2, 2), container_block="furnace")
        mock_net.send_furnace_open.assert_called_once_with(2, 2, 2)

        # Server sends furnace_sync packet; client receives and processes it
        pkt = {
            "type": "furnace_sync",
            "furnace": Furnace(2, 2, 2, block_id="furnace").to_dict(),
        }
        mock_net.poll_events.return_value = [pkt]
        Game._network_tick(mock_game, 0.016)

        # Successfully parsed without error and updated inventory furnace
        self.assertIsNotNone(mock_game.inventory.furnace)
        self.assertEqual(mock_game.inventory.furnace.x, 2)
        self.assertEqual(mock_game.inventory.furnace.y, 2)
        self.assertEqual(mock_game.inventory.furnace.z, 2)

    def test_concurrent_furnace_access_no_item_loss(self):
        """Ensure two clients interacting with the same furnace concurrently don't overwrite each other's items."""
        from unittest.mock import MagicMock
        from server.network import MultiplayerServer, RemoteClient
        from server.stacks import make_stack
        from server.world import Chunk, World

        with tempfile.TemporaryDirectory() as tmp:
            world = World(seed=12)
            server = MultiplayerServer(host="127.0.0.1", port=0, saves_dir=tmp, world=world)
            chunk = Chunk(0, 0, 0)
            world.chunks[(0, 0, 0)] = chunk
            world.set_block(2, 2, 2, "furnace")

            c1_pkts = []
            c2_pkts = []
            client1 = MagicMock(spec=RemoteClient)
            client1.alive = True
            client1.uuid = "p1"
            client1.pos = [2.0, 2.0, 2.0]
            client1.sent_chunks = set()
            client1.send = lambda pkt: c1_pkts.append(pkt)

            client2 = MagicMock(spec=RemoteClient)
            client2.alive = True
            client2.uuid = "p2"
            client2.pos = [2.5, 2.0, 2.5]
            client2.sent_chunks = set()
            client2.send = lambda pkt: c2_pkts.append(pkt)

            server.clients["p1"] = client1
            server.clients["p2"] = client2

            # Both players open furnace
            server._handle_packet(client1, {"type": "furnace_open", "x": 2, "y": 2, "z": 2})
            server._handle_packet(client2, {"type": "furnace_open", "x": 2, "y": 2, "z": 2})

            # Player 1 deposits raw iron into input slot (only sending "input" slot)
            c1_pkts.clear()
            c2_pkts.clear()
            server._handle_packet(
                client1,
                {
                    "type": "furnace_update",
                    "x": 2,
                    "y": 2,
                    "z": 2,
                    "slots": {"input": make_stack("raw_iron", 5)},
                },
            )

            # Player 2 simultaneously deposits coal into fuel slot (only sending "fuel" slot)
            server._handle_packet(
                client2,
                {
                    "type": "furnace_update",
                    "x": 2,
                    "y": 2,
                    "z": 2,
                    "slots": {"fuel": make_stack("coal", 3)},
                },
            )

            # Authoritative server furnace must contain BOTH items
            f_server = world.furnaces[(2, 2, 2)]
            self.assertEqual(f_server.input["id"], "raw_iron")
            self.assertEqual(f_server.input["count"], 5)
            self.assertEqual(f_server.fuel["id"], "coal")
            self.assertEqual(f_server.fuel["count"], 3)

            # Client 1 received update about coal, Client 2 received update about raw iron
            c1_sync = [p for p in c1_pkts if p.get("type") == "furnace_sync"]
            c2_sync = [p for p in c2_pkts if p.get("type") == "furnace_sync"]
            self.assertGreater(len(c1_sync), 0)
            self.assertGreater(len(c2_sync), 0)
            self.assertEqual(c1_sync[-1]["furnace"]["fuel"]["id"], "coal")
            self.assertEqual(c2_sync[0]["furnace"]["input"]["id"], "raw_iron")

            # Periodic progress ticks only send furnace_progress without resetting slots
            c1_pkts.clear()
            c2_pkts.clear()
            server._furnace_sync_timer = 0.1
            server.tick()
            c1_prog = [p for p in c1_pkts if p.get("type") == "furnace_progress"]
            c2_prog = [p for p in c2_pkts if p.get("type") == "furnace_progress"]
            self.assertGreater(len(c1_prog), 0)
            self.assertGreater(len(c2_prog), 0)
            self.assertNotIn("furnace", c1_prog[0])
            self.assertIn("cook_time", c1_prog[0])


class RemoteBlockChangeTests(unittest.TestCase):
    def _game(self):
        from client.app import Game

        world = World(seed=1)
        world.chunks[(0, 0, 0)] = Chunk(0, 0, 0)
        game = Game.__new__(Game)
        game.world = world
        uploaded = []
        game._upload = lambda key: uploaded.append(key)
        return game, world, uploaded

    def test_remote_block_change_updates_world_and_mesh(self):
        game, world, uploaded = self._game()
        self.assertEqual(world.get_block(1, 1, 1), AIR)

        self.assertTrue(game._apply_remote_block_change(1, 1, 1, "stone"))

        self.assertEqual(world.get_block(1, 1, 1), "stone")
        # The affected chunk is rebuilt immediately, not left to the dirty queue.
        self.assertIn((0, 0, 0), uploaded)

    def test_remote_block_change_for_unloaded_chunk_is_ignored(self):
        game, _world, uploaded = self._game()

        self.assertFalse(game._apply_remote_block_change(100, 50, 100, "stone"))
        self.assertEqual(uploaded, [])


class RemoteItemDropTests(unittest.TestCase):
    """Breaking blocks must not create local drops in multiplayer.

    The server spawns the item and broadcasts ``spawn_item``. If the client also
    spawns one locally the breaker ends up with two entities for one block.
    """

    def _game(self, network_client=None):
        from client.app import Game
        from server.stacks import make_stack

        world = World(seed=1)
        world.chunks[(0, 0, 0)] = Chunk(0, 0, 0)
        world.set_block(2, 2, 2, "stone")
        game = Game.__new__(Game)
        game.world = world
        game.gamemode = "survival"
        game.selected = 0
        game.hotbar = [make_stack("wood_pickaxe", 1)]
        game.network_client = network_client
        game.target = {"x": 2, "y": 2, "z": 2, "local_y": 0.5}
        game._aim = lambda: None
        game._upload_block_change = lambda x, y, z: None
        return game, world

    def _item_drops(self, world):
        from server.entity import ItemEntity

        return [e for e in world.entities if isinstance(e, ItemEntity)]

    def test_multiplayer_break_does_not_spawn_local_drop(self):
        from unittest.mock import MagicMock

        client = MagicMock()
        game, world = self._game(client)

        game._break_block()

        self.assertEqual(self._item_drops(world), [])
        client.send_drop_item.assert_called_once()
        client.send_set_block.assert_called_once_with(2, 2, 2, AIR)
        self.assertEqual(world.get_block(2, 2, 2), AIR)

    def test_singleplayer_break_spawns_local_drop(self):
        game, world = self._game(None)

        game._break_block()

        drops = self._item_drops(world)
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0].item, "cobble")
        self.assertEqual(world.get_block(2, 2, 2), AIR)

    def test_multiplayer_drop_held_item_does_not_spawn_local_drop(self):
        from unittest.mock import MagicMock

        from server.player import Player
        from server.stacks import make_stack, stack_count

        client = MagicMock()
        game, world = self._game(client)
        game.player = Player(world)
        game.player.pos = [2.5, 3.0, 2.5]
        game.forward = (1.0, 0.0, 0.0)
        game.hotbar = [make_stack("stone", 2)]

        game._drop_held_item()

        self.assertEqual(self._item_drops(world), [])
        client.send_drop_item.assert_called_once()
        self.assertEqual(stack_count(game.hotbar[0]), 1)

    def test_multiplayer_loose_item_is_spawned_by_server(self):
        from unittest.mock import MagicMock

        client = MagicMock()
        game, world = self._game(client)

        game._spawn_loose_item("sand", 2, [1.0, 2.0, 3.0], pickup_delay=0.5)

        self.assertEqual(self._item_drops(world), [])
        client.send_drop_item.assert_called_once_with(
            "sand", 2, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 0.5
        )

    def test_singleplayer_loose_item_spawns_locally(self):
        game, world = self._game(None)

        game._spawn_loose_item("sand", 2, [1.0, 2.0, 3.0], pickup_delay=0.5)

        drops = self._item_drops(world)
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0].item, "sand")
        self.assertEqual(drops[0].count, 2)
        self.assertEqual(drops[0].pickup_delay, 0.5)


class ItemPickupTests(unittest.TestCase):
    """Only one side may grant an item when a drop is at the player's feet."""

    def _game(self, network_client=None):
        from client.app import Game
        from server.player import Player

        world = World(seed=1)
        game = Game.__new__(Game)
        game.world = world
        game.player = Player(world)
        game.player.pos = [2.5, 3.0, 2.5]
        game.mode = "play"
        game.network_client = network_client
        game.hotbar = [None] * 9
        game.storage = [None] * 27
        return game, world

    def _has_item(self, game, item_id):
        from server.stacks import stack_id

        return any(stack_id(slot) == item_id for slot in game.hotbar + game.storage)

    def test_multiplayer_pickup_is_left_to_the_server(self):
        from unittest.mock import MagicMock

        client = MagicMock()
        game, world = self._game(client)
        entity = world.spawn_item("sand", count=1, pos=[2.5, 3.0, 2.5], pickup_delay=0.0)

        game._update_item_pickups(0.02)

        # The server removes the entity and sends give_item; the client must not
        # add its own copy as well.
        self.assertFalse(self._has_item(game, "sand"))
        self.assertIn(entity, world.entities)
        client.send_pickup_item.assert_not_called()

    def test_singleplayer_pickup_adds_to_inventory(self):
        game, world = self._game(None)
        entity = world.spawn_item("sand", count=1, pos=[2.5, 3.0, 2.5], pickup_delay=0.0)

        game._update_item_pickups(0.02)

        self.assertTrue(self._has_item(game, "sand"))
        self.assertNotIn(entity, world.entities)


if __name__ == "__main__":
    unittest.main()
