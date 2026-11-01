import tempfile
import time
import unittest
from pathlib import Path

from client.network import NetworkClient
from server.blocks import AIR, STONE
from server.damage import (
    DEFAULT_MAX_HEALTH,
    DamageType,
    damage_type_from_mapping,
    damage_types_for_kind,
    get_damage_type,
    load_damage_types,
)
from server.network import MultiplayerServer
from server.player import Player


class _HealthWorld:
    """Small world stub that lets health tests place blocks by coordinate."""

    def __init__(self, spawn=(0.5, 64.0, 0.5), ground=63):
        self.blocks = {}
        self.spawn = tuple(spawn)
        self.ground = int(ground)

    def get_block(self, x, y, z):
        return self.blocks.get((int(x), int(y), int(z)), AIR)

    def column_top(self, x, z):
        return self.ground

    def get_spawn_point(self):
        return self.spawn


class DamageAssetTests(unittest.TestCase):
    def test_default_assets_define_fall_and_suffocation(self):
        fall = get_damage_type("fall")
        suffocation = get_damage_type("suffocation")
        self.assertIsNotNone(fall)
        self.assertIsNotNone(suffocation)
        self.assertEqual(fall.kind, "fall")
        self.assertEqual(fall.min_distance, 3.0)
        self.assertTrue(fall.per_block)
        self.assertEqual(suffocation.kind, "suffocation")
        self.assertEqual(suffocation.interval, 0.5)
        self.assertIn(fall, damage_types_for_kind("fall"))
        self.assertIn(suffocation, damage_types_for_kind("suffocation"))

    def test_loads_json_and_lua_damage_types_from_a_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "fall.json").write_text(
                '{"id": "fall", "kind": "fall", "damage": 2, "min_distance": 4}',
                encoding="utf-8",
            )
            (folder / "freeze.lua").write_text(
                'return { id = "freeze", name = "Freeze", kind = "suffocation", '
                'damage = 3, interval = 1.25, message = "froze to death" }',
                encoding="utf-8",
            )
            types = load_damage_types(folder)
            self.assertEqual(set(types), {"fall", "freeze"})
            self.assertIsInstance(types["fall"], DamageType)
            self.assertEqual(types["fall"].damage, 2.0)
            self.assertEqual(types["fall"].min_distance, 4.0)
            self.assertEqual(types["freeze"].kind, "suffocation")
            self.assertEqual(types["freeze"].interval, 1.25)
            self.assertEqual(types["freeze"].message, "froze to death")

    def test_rejects_unknown_damage_kind(self):
        with self.assertRaises(ValueError):
            damage_type_from_mapping({"id": "void", "kind": "magic"})

    def test_builtin_fallback_when_folder_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            types = load_damage_types(Path(tmp) / "does_not_exist")
            self.assertIn("fall", types)
            self.assertIn("suffocation", types)


class PlayerHealthTests(unittest.TestCase):
    def setUp(self):
        self.world = _HealthWorld()
        self.player = Player(self.world)

    def test_new_player_starts_at_full_health(self):
        self.assertEqual(self.player.health, DEFAULT_MAX_HEALTH)
        self.assertEqual(self.player.max_health, DEFAULT_MAX_HEALTH)
        self.assertFalse(self.player.dead)

    def test_fall_damage_scales_with_blocks_beyond_threshold(self):
        self.player.pos = [0.5, 70.0, 0.5]
        self.assertEqual(self.player.update_health(0.05, grounded=False, in_water=False), [])
        self.player.pos = [0.5, 63.0, 0.5]
        events = self.player.update_health(0.05, grounded=True, in_water=False)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].damage_type.id, "fall")
        # 7 blocks fallen - 3 safe blocks = 4 HP
        self.assertEqual(events[0].amount, 4.0)
        self.assertEqual(self.player.health, 16.0)

    def test_short_falls_are_safe(self):
        self.player.pos = [0.5, 66.0, 0.5]
        self.player.update_health(0.05, grounded=False, in_water=False)
        self.player.pos = [0.5, 63.0, 0.5]
        self.assertEqual(self.player.update_health(0.05, grounded=True, in_water=False), [])
        self.assertEqual(self.player.health, DEFAULT_MAX_HEALTH)

    def test_landing_in_water_avoids_fall_damage(self):
        self.player.pos = [0.5, 90.0, 0.5]
        self.player.update_health(0.05, grounded=False, in_water=False)
        self.player.pos = [0.5, 63.0, 0.5]
        self.assertEqual(self.player.update_health(0.05, grounded=True, in_water=True), [])
        self.assertEqual(self.player.health, DEFAULT_MAX_HEALTH)

    def test_suffocation_ticks_every_half_second(self):
        self.player.pos = [0.5, 63.0, 0.5]
        self.world.blocks[(0, 64, 0)] = STONE
        for _ in range(9):
            self.assertEqual(self.player.update_health(0.05, grounded=True, in_water=False), [])
        events = self.player.update_health(0.05, grounded=True, in_water=False)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].damage_type.id, "suffocation")
        self.assertEqual(self.player.health, 19.0)

    def test_leaving_the_block_resets_the_suffocation_timer(self):
        self.player.pos = [0.5, 63.0, 0.5]
        self.world.blocks[(0, 64, 0)] = STONE
        for _ in range(5):
            self.player.update_health(0.05, grounded=True, in_water=False)
        del self.world.blocks[(0, 64, 0)]
        self.assertEqual(self.player.update_health(0.05, grounded=True, in_water=False), [])
        self.world.blocks[(0, 64, 0)] = STONE
        for _ in range(9):
            self.assertEqual(self.player.update_health(0.05, grounded=True, in_water=False), [])
        self.assertEqual(self.player.health, DEFAULT_MAX_HEALTH)

    def test_creative_players_are_immune(self):
        self.player.gamemode = "creative"
        self.player.pos = [0.5, 63.0, 0.5]
        self.world.blocks[(0, 64, 0)] = STONE
        self.player.pos = [0.5, 70.0, 0.5]
        self.player.update_health(0.05, grounded=False, in_water=False)
        self.player.pos = [0.5, 63.0, 0.5]
        self.assertEqual(self.player.update_health(0.05, grounded=True, in_water=False), [])
        self.assertEqual(self.player.take_damage(5.0), 0.0)
        self.assertEqual(self.player.health, DEFAULT_MAX_HEALTH)

    def test_death_respawns_at_personal_spawn_point(self):
        self.player.set_spawn_point(12.5, 70.0, -34.5)
        self.player.set_health(1.0)
        self.player.pos = [0.5, 63.0, 0.5]
        self.world.blocks[(0, 64, 0)] = STONE
        fatal = []
        for _ in range(20):
            fatal.extend(self.player.update_health(0.05, grounded=True, in_water=False))
            if self.player.dead:
                break
        self.assertTrue(self.player.dead)
        self.assertEqual(self.player.health, 0.0)
        self.assertTrue(any(event.fatal for event in fatal))

        pos = self.player.respawn()
        self.assertEqual(pos, [12.5, 70.0, -34.5])
        self.assertEqual(self.player.pos, [12.5, 70.0, -34.5])
        self.assertEqual(self.player.health, DEFAULT_MAX_HEALTH)
        self.assertFalse(self.player.dead)

    def test_respawn_falls_back_to_world_spawn(self):
        self.assertEqual(self.player.respawn_position(), [0.5, 64.0, 0.5])


class MultiplayerHealthTests(unittest.TestCase):
    def test_suffocation_death_respawns_client_at_spawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = MultiplayerServer(host="127.0.0.1", port=0, world_name="HealthTest", saves_dir=tmp)
            server.start()
            port = server.actual_port
            uuid = "health-player"

            def wait_for(client, wanted, timeout=4.0):
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

            def wait_for_types(client, wanted, timeout=4.0):
                found = {name: [] for name in wanted}
                t0 = time.time()
                while time.time() - t0 < timeout:
                    server.tick()
                    for pkt in client.poll_events():
                        name = pkt.get("type")
                        if name in found:
                            found[name].append(pkt)
                    if all(found[name] for name in wanted):
                        return found
                    time.sleep(0.02)
                return found

            client = NetworkClient("127.0.0.1", port, uuid, "Faller")
            try:
                self.assertTrue(client.connect(timeout=2.0))
                ack = wait_for(client, "login_ack")
                self.assertTrue(ack)
                self.assertEqual(ack[0]["health"], DEFAULT_MAX_HEALTH)
                self.assertEqual(ack[0]["max_health"], DEFAULT_MAX_HEALTH)

                player = server.world.players[uuid]
                spawn = [24.5, 70.0, -12.5]
                player.set_spawn_point(*spawn)

                # Kill the player; the server should respawn them on its next tick.
                player.set_health(1.0)
                self.assertEqual(player.take_damage(10.0, get_damage_type("suffocation")), 1.0)
                self.assertTrue(player.dead)

                received = wait_for_types(client, ("teleport", "health"))
                teleports = received["teleport"]
                self.assertTrue(teleports, "server never respawned the dead player")
                self.assertEqual([round(v, 1) for v in teleports[0]["pos"]], spawn)
                self.assertEqual(player.health, DEFAULT_MAX_HEALTH)
                self.assertFalse(player.dead)
                self.assertEqual(player.pos, spawn)

                health_packets = received["health"]
                self.assertTrue(health_packets)
                self.assertEqual(health_packets[0]["health"], DEFAULT_MAX_HEALTH)
                self.assertFalse(health_packets[0]["dead"])
            finally:
                client.disconnect()
                server.stop()

    def test_health_persists_across_disconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            uuid = "health-persist-player"
            server = MultiplayerServer(host="127.0.0.1", port=0, world_name="HealthPersist", saves_dir=tmp)
            server.start()
            port = server.actual_port

            def connect_and_wait():
                client = NetworkClient("127.0.0.1", port, uuid, "Wounded")
                self.assertTrue(client.connect(timeout=2.0))
                t0 = time.time()
                while time.time() - t0 < 4.0:
                    server.tick()
                    for pkt in client.poll_events():
                        if pkt.get("type") == "login_ack":
                            return client, pkt
                    time.sleep(0.02)
                self.fail("server never sent login_ack")

            client, ack = connect_and_wait()
            try:
                self.assertEqual(ack["health"], DEFAULT_MAX_HEALTH)
                player = server.world.players[uuid]
                player.set_health(9.5)

                client.disconnect()
                time.sleep(0.08)
                server.tick()
                self.assertEqual(server.players_data[uuid]["health"], 9.5)
                self.assertEqual(server.players_data[uuid]["max_health"], DEFAULT_MAX_HEALTH)

                client2, ack2 = connect_and_wait()
                try:
                    self.assertEqual(ack2["health"], 9.5)
                    self.assertEqual(ack2["max_health"], DEFAULT_MAX_HEALTH)
                    self.assertEqual(server.world.players[uuid].health, 9.5)
                finally:
                    client2.disconnect()
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
