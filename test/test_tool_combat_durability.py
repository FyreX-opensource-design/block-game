import unittest
from unittest.mock import MagicMock

from client.app import Game
from client.inventory import make_stack
from server.items import ITEMS
from server.network import MultiplayerServer, RemoteClient
from server.player import Player
from server.stacks import damage_tool, empty_stack, normalize_stack, pack_stack, stack_damage, stack_id
from server.world import World


class TestToolCombatAndDurability(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=42)
        for cy in range(16):
            self.world.load_chunk(0, cy, 0)

    def test_damage_tool_mechanics(self):
        # 1. Non-tool has no durability, damage_tool returns original stack
        dirt_stack = make_stack("dirt", 10)
        self.assertEqual(damage_tool(dirt_stack), dirt_stack)

        # 2. Wood pickaxe has durability 59
        pick = make_stack("wood_pickaxe", 1)
        self.assertEqual(stack_damage(pick), 0)

        # One use
        damaged = damage_tool(pick, amount=1)
        self.assertEqual(stack_id(damaged), "wood_pickaxe")
        self.assertEqual(stack_damage(damaged), 1)

        # 58 uses
        damaged58 = damage_tool(pick, amount=58)
        self.assertEqual(stack_damage(damaged58), 58)

        # 59th use breaks the pickaxe
        broken = damage_tool(damaged58, amount=1)
        self.assertEqual(broken, empty_stack())

        # Exceeding durability breaks the pickaxe
        broken2 = damage_tool(pick, amount=60)
        self.assertEqual(broken2, empty_stack())

    def test_pack_and_normalize_stack_preserves_damage(self):
        pick = make_stack("wood_pickaxe", 1, damage=15)
        self.assertEqual(stack_damage(pick), 15)

        packed = pack_stack(pick)
        self.assertEqual(packed, {"id": "wood_pickaxe", "count": 1, "damage": 15})

        restored = normalize_stack(packed)
        self.assertEqual(stack_id(restored), "wood_pickaxe")
        self.assertEqual(stack_damage(restored), 15)

        # Undamaged tool packs to clean string
        undamaged = make_stack("wood_pickaxe", 1, damage=0)
        self.assertEqual(pack_stack(undamaged), "wood_pickaxe")

    def test_breaking_blocks_in_survival_reduces_tool_durability(self):
        game = Game.__new__(Game)
        game.world = self.world
        game.gamemode = "survival"
        game.hotbar = [make_stack("wood_pickaxe", 1)]
        game.selected = 0
        game.network_client = None
        game.renderer = MagicMock()
        game._upload_block_change = MagicMock()
        game._spawn_break_drops = MagicMock()

        # Place a stone block
        self.world.set_block(5, 60, 5, "stone")

        # Break the block
        game.target = {"x": 5, "y": 60, "z": 5, "nx": 0, "ny": 1, "nz": 0}
        game._aim = MagicMock()
        game._break_block()

        # Tool durability should be reduced by 1
        held = game.hotbar[0]
        self.assertEqual(stack_id(held), "wood_pickaxe")
        self.assertEqual(stack_damage(held), 1)

    def test_breaking_blocks_in_creative_does_not_reduce_durability(self):
        game = Game.__new__(Game)
        game.world = self.world
        game.gamemode = "creative"
        game.hotbar = [make_stack("wood_pickaxe", 1)]
        game.selected = 0
        game.network_client = None
        game.renderer = MagicMock()
        game._upload_block_change = MagicMock()
        game._spawn_break_drops = MagicMock()

        self.world.set_block(5, 60, 5, "stone")
        game.target = {"x": 5, "y": 60, "z": 5, "nx": 0, "ny": 1, "nz": 0}
        game._aim = MagicMock()
        game._break_block()

        held = game.hotbar[0]
        self.assertEqual(stack_damage(held), 0)

    def test_tool_breaks_completely_after_max_uses(self):
        game = Game.__new__(Game)
        game.world = self.world
        game.gamemode = "survival"
        # Wood pickaxe has durability 59; start with 58 damage
        game.hotbar = [make_stack("wood_pickaxe", 1, damage=58)]
        game.selected = 0
        game.network_client = None
        game.renderer = MagicMock()
        game._upload_block_change = MagicMock()
        game._spawn_break_drops = MagicMock()

        self.world.set_block(5, 60, 5, "stone")
        game.target = {"x": 5, "y": 60, "z": 5, "nx": 0, "ny": 1, "nz": 0}
        game._aim = MagicMock()
        game._break_block()

        # 59th use: slot should now be empty_stack()
        held = game.hotbar[0]
        self.assertEqual(stack_id(held), "")
        self.assertEqual(held, empty_stack())

    def test_combat_damage_with_tool_vs_bare_hands_on_mob(self):
        from server.entity import Pig

        pig = Pig(pos=[10.0, 86.0, 10.0], health=10.0)
        self.world.entities.append(pig)

        game = Game.__new__(Game)
        game.world = self.world
        game.gamemode = "survival"
        game.player = Player(self.world)
        game.player.pos = [10.0, 86.0, 8.5]
        game.player.eye = 1.6
        game.player.yaw = 0.0  # looking towards +Z
        game.player.pitch = 0.0
        game.hotbar = [make_stack("wood_pickaxe", 1)]
        game.selected = 0
        game.network_client = None
        game.other_players = {}

        # 1. Attack mob with wood pickaxe (attack: 2)
        target = game._raycast_attack_target(10.0, 86.5, 8.5, 0.0, 0.0, 1.0, max_dist=3.5)
        self.assertIsNotNone(target)
        self.assertEqual(target[0], "mob")

        # Mock left click on target
        held_id = stack_id(game.hotbar[game.selected])
        attack_dmg = float(ITEMS[held_id].attack)
        self.assertEqual(attack_dmg, 2.0)

        target[1].take_damage(attack_dmg, attacker=game.player, world=game.world)
        game._use_held_tool()

        # Pig took 2.0 damage
        self.assertEqual(pig.health, 8.0)
        # Tool durability reduced by 1
        self.assertEqual(stack_damage(game.hotbar[0]), 1)

    def test_pvp_attack_other_player(self):
        server = MultiplayerServer(world=self.world)

        # Attacker client
        client_attacker = MagicMock(spec=RemoteClient)
        client_attacker.alive = True
        client_attacker.uuid = "p1"
        client_attacker.username = "Attacker"
        client_attacker.pos = [10.0, 64.0, 10.0]
        client_attacker.held_item = "wood_pickaxe"

        # Victim client & player
        client_victim = MagicMock(spec=RemoteClient)
        client_victim.alive = True
        client_victim.uuid = "p2"
        client_victim.username = "Victim"
        client_victim.pos = [10.0, 64.0, 11.5]
        client_victim.vel = [0.0, 0.0, 0.0]
        victim_sent = []
        client_victim.send = lambda pkt: victim_sent.append(pkt)

        player_victim = Player(self.world)
        player_victim.gamemode = "survival"
        player_victim.health = 20.0
        player_victim.max_health = 20.0

        server.clients["p1"] = client_attacker
        server.clients["p2"] = client_victim
        self.world.players["p2"] = player_victim

        broadcasted = []
        server.broadcast = lambda pkt: broadcasted.append(pkt)

        # Attacker hits victim with wood pickaxe (attack: 2)
        server._handle_packet(client_attacker, {
            "type": "attack_player",
            "uuid": "p2",
            "item": "wood_pickaxe",
        })

        # Victim health reduced by 2.0
        self.assertEqual(player_victim.health, 18.0)
        self.assertGreater(player_victim.hurt_time, 0.0)

        # Knockback sent to victim
        kb_pkts = [p for p in victim_sent if p.get("type") == "player_knockback"]
        self.assertTrue(len(kb_pkts) > 0)
        self.assertGreater(kb_pkts[0]["vel"][2], 0.0)  # knocked back along +Z

        # Health update sent to victim
        hp_pkts = [p for p in victim_sent if p.get("type") == "health"]
        self.assertTrue(len(hp_pkts) > 0)
        self.assertEqual(hp_pkts[0]["health"], 18.0)

        # Hurt broadcasted
        hurt_pkts = [p for p in broadcasted if p.get("type") == "player_hurt"]
        self.assertTrue(len(hurt_pkts) > 0)
        self.assertEqual(hurt_pkts[0]["uuid"], "p2")


if __name__ == "__main__":
    unittest.main()
