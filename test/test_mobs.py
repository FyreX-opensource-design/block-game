import math
import unittest
from unittest.mock import MagicMock

from server.ai import EatGrassGoal, Goal, GoalSelector, PanicGoal, WanderGoal, is_direction_safe, is_position_hazardous
from server.blocks import AIR, DIRT, GRASS, STONE
from server.entity import Mob, Pig, Sheep, entity_from_dict
from server.loot import get_mob_drops
from server.pig_model import PigMeshes, build_pig_meshes, pig_pose
from server.sheep_model import SheepMeshes, build_sheep_meshes, sheep_pose
from server.player import Player
from server.recipes import craft_match
from server.spawner import MobSpawner
from server.world import Chunk, World


class TestMobEntity(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=1)
        self.chunk = Chunk(0, 0, 0)
        self.chunk.data.fill(0)
        self.world.chunks[(0, 0, 0)] = self.chunk

    def test_pig_initialization(self):
        pig = Pig(pos=[10.0, 5.0, 10.0])
        self.assertEqual(pig.mob_type, "pig")
        self.assertEqual(pig.health, 10.0)
        self.assertEqual(pig.max_health, 10.0)
        self.assertEqual(pig.radius, 0.45)
        self.assertEqual(pig.height, 0.9)
        self.assertEqual(pig.walk_speed, 1.2)
        self.assertEqual(pig.run_speed, 2.8)
        self.assertTrue(pig.alive)
        self.assertFalse(pig.in_panic())
        self.assertIsNotNone(pig.goals)

    def test_pig_serialization(self):
        pig = Pig(pos=[12.5, 64.0, -8.2], yaw=1.57, health=8.0)
        data = pig.to_dict()
        self.assertEqual(data["type"], "pig")
        self.assertEqual(data["pos"], [12.5, 64.0, -8.2])
        self.assertAlmostEqual(data["yaw"], 1.57, places=2)
        self.assertEqual(data["health"], 8.0)

        restored = entity_from_dict(data)
        self.assertIsInstance(restored, Pig)
        self.assertEqual(restored.id, pig.id)
        self.assertEqual(restored.pos, [12.5, 64.0, -8.2])
        self.assertAlmostEqual(restored.yaw, 1.57, places=2)
        self.assertEqual(restored.health, 8.0)

    def test_goal_selector_priority(self):
        selector = GoalSelector()
        g1 = MagicMock(spec=Goal)
        g1.can_start.return_value = False
        g1.can_continue.return_value = False

        g2 = MagicMock(spec=Goal)
        g2.can_start.return_value = True
        g2.can_continue.return_value = True

        selector.add_goal(2, g2)
        selector.add_goal(1, g1)

        selector.tick(0.1)
        self.assertEqual(selector.active_goal, g2)
        g2.start.assert_called_once()
        g2.tick.assert_called_with(0.1)

        # Now high priority goal can start -> preempts g2
        g1.can_start.return_value = True
        g1.can_continue.return_value = True
        selector.tick(0.1)
        self.assertEqual(selector.active_goal, g1)
        g2.stop.assert_called_once()
        g1.start.assert_called_once()

    def test_pig_wander_goal(self):
        pig = Pig(pos=[5.0, 1.0, 5.0])
        wander = next(g for _, g in pig.goals.goals if isinstance(g, WanderGoal))

        # Initially idle
        wander.start()
        self.assertEqual(wander.state, "idle")
        self.assertEqual(pig.desired_speed, 0.0)

        # Fast forward past timer to trigger walking
        wander.timer = 0.01
        wander.tick(0.05)
        self.assertEqual(wander.state, "walk")
        self.assertEqual(pig.desired_speed, pig.walk_speed)

    def test_pig_hit_and_panic_goal(self):
        player = MagicMock()
        player.pos = [0.0, 1.0, 0.0]

        pig = Pig(pos=[2.0, 1.0, 0.0])
        self.world.spawn_mob(pig)

        # Wander is active initially
        pig.goals.tick(0.1)
        self.assertFalse(pig.in_panic())

        # Hit the pig
        hit = pig.take_damage(2.0, attacker=player, world=self.world)
        self.assertTrue(hit)
        self.assertEqual(pig.health, 8.0)
        self.assertTrue(pig.in_panic())
        self.assertGreater(pig.hurt_time, 0.0)
        self.assertGreater(pig.panic_timer, 4.0)

        # Knockback applied away from player (+X)
        self.assertGreater(pig.vel[0], 0.0)
        self.assertGreater(pig.vel[1], 0.0)

        # Invulnerability window: second hit within hurt_time is rejected
        second_hit = pig.take_damage(2.0, attacker=player, world=self.world)
        self.assertFalse(second_hit)
        self.assertEqual(pig.health, 8.0)

        # PanicGoal should now activate and set run_speed
        pig.goals.tick(0.1)
        active_goal = pig.goals.active_goal
        self.assertIsInstance(active_goal, PanicGoal)
        self.assertEqual(pig.desired_speed, pig.run_speed)

    def test_pig_death_and_drops_raw_porkchop(self):
        player = MagicMock()
        player.pos = [0.0, 1.0, 0.0]

        pig = Pig(pos=[5.0, 1.0, 5.0], health=2.0)
        self.world.spawn_mob(pig)

        initial_entities = len(self.world.entities)

        # Fatal blow
        pig.take_damage(2.0, attacker=player, world=self.world)
        self.assertFalse(pig.alive)
        self.assertEqual(pig.health, 0.0)

        # Verify porkchop drop entity was spawned in world
        new_items = [
            e for e in self.world.entities[initial_entities:] if getattr(e, "item", "") == "raw_porkchop"
        ]
        self.assertGreater(len(new_items), 0)
        self.assertGreaterEqual(new_items[0].count, 1)

    def test_pig_gravity_and_ground_collision(self):
        # Set solid stone ground at y=5
        for x in range(3):
            for z in range(3):
                self.world.set_block(x, 5, z, STONE)

        pig = Pig(pos=[1.5, 8.0, 1.5])
        self.world.spawn_mob(pig)

        for _ in range(30):
            pig.update(0.05, self.world)

        self.assertTrue(pig.on_ground)
        self.assertAlmostEqual(pig.pos[1], 6.0, delta=0.05)
        self.assertEqual(pig.vel[1], 0.0)

    def test_pig_step_up_one_block(self):
        # Floor at y=5, step up at x=2, y=6
        for x in range(4):
            for z in range(3):
                self.world.set_block(x, 5, z, STONE)
        self.world.set_block(2, 6, 1, STONE)

        pig = Pig(pos=[1.0, 6.0, 1.5])
        pig.on_ground = True
        self.world.spawn_mob(pig)

        # Move forward across +X over the 1-block obstacle
        pig.move_horizontal(1.5, 0.0, self.world)
        self.assertGreater(pig.pos[0], 1.5)
        self.assertAlmostEqual(pig.pos[1], 7.0, delta=0.1)

    def test_raycast_entity_targeting(self):
        pig = Pig(pos=[5.0, 1.0, 0.0])
        self.world.spawn_mob(pig)

        # Looking directly from origin towards pig (+X)
        hit = self.world.raycast_entity(0.0, 1.5, 0.0, 1.0, 0.0, 0.0, max_dist=10.0)
        self.assertEqual(hit, pig)

        # Looking away (-X)
        hit_away = self.world.raycast_entity(0.0, 1.5, 0.0, -1.0, 0.0, 0.0, max_dist=10.0)
        self.assertIsNone(hit_away)

        # Obstructed by solid block in front of pig
        self.world.set_block(2, 1, 0, STONE)
        hit_blocked = self.world.raycast_entity(0.0, 1.5, 0.0, 1.0, 0.0, 0.0, max_dist=10.0)
        self.assertIsNone(hit_blocked)


class TestMobSpawner(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=1)
        self.spawner = MobSpawner(self.world)

    def test_despawn_far_away(self):
        player = MagicMock()
        player.pos = [0.0, 64.0, 0.0]
        self.world.players["p1"] = player

        near_pig = Pig(pos=[10.0, 64.0, 0.0])
        far_pig = Pig(pos=[200.0, 64.0, 0.0])
        self.world.spawn_mob(near_pig)
        self.world.spawn_mob(far_pig)

        self.spawner.update(0.1)
        self.assertTrue(near_pig.alive)
        self.assertFalse(far_pig.alive)

        # Entity update removes dead mob
        self.world.update_entities(0.05)
        self.assertIn(near_pig, self.world.entities)
        self.assertNotIn(far_pig, self.world.entities)

    def test_can_spawn_at_with_lighting(self):
        self.world.load_chunk(0, 0, 0)
        self.world.set_block(0, 10, 0, GRASS)
        self.world.set_block(0, 11, 0, AIR)
        self.world.set_block(0, 12, 0, AIR)
        self.world.lighting.get_skylight = MagicMock(return_value=15)
        self.assertTrue(self.spawner._can_spawn_at(0, 10, 0))

        self.world.lighting.get_skylight = MagicMock(return_value=4)
        self.assertFalse(self.spawner._can_spawn_at(0, 10, 0))


class TestPigModel(unittest.TestCase):
    def test_build_meshes(self):
        meshes = build_pig_meshes()
        for part in ("body", "head", "snout", "leg_front_left", "leg_front_right", "leg_back_left", "leg_back_right"):
            self.assertIn(part, meshes)
            v, i = meshes[part]
            self.assertGreater(len(v), 0)
            self.assertGreater(len(i), 0)

    def test_posed_vertices(self):
        pig_meshes = PigMeshes()
        poses = pig_pose(time=1.0, speed=1.2, yaw=0.5, pitch=0.1)
        vertices, indices = pig_meshes.posed_vertices([10.0, 5.0, 10.0], 0.5, poses)
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(indices), 0)
        self.assertEqual(len(vertices) % 8, 0)
        self.assertEqual(len(indices) % 3, 0)

        # Check vertical height range: feet should touch ground (y=5.0) and top of head/body around y=5.8..6.0
        import numpy as np
        y_coords = vertices.reshape(-1, 8)[:, 1]
        self.assertAlmostEqual(float(np.min(y_coords)), 5.0, delta=0.05)
        self.assertAlmostEqual(float(np.max(y_coords)), 6.0, delta=0.1)


class TestMultiplayerMobs(unittest.TestCase):
    def test_remote_mob_interpolation(self):
        pig = Pig(pos=[10.0, 5.0, 10.0], yaw=0.0)
        pig.is_remote = True
        pig.target_pos = [12.0, 5.0, 10.0]
        pig.target_yaw = 1.0
        pig.vel = [1.0, 0.0, 0.0]

        # In remote mode, local AI goals should not be active
        wander = next(g for _, g in pig.goals.goals if isinstance(g, WanderGoal))
        wander.state = "idle"

        pig.update(0.05)
        # Should smoothly move towards target_pos (12.0) from 10.0
        self.assertGreater(pig.pos[0], 10.0)
        self.assertLess(pig.pos[0], 12.0)
        # Should smoothly rotate towards target_yaw (1.0) from 0.0
        self.assertGreater(pig.yaw, 0.0)
        self.assertLess(pig.yaw, 1.0)
        # Walk animation should advance smoothly
        self.assertGreater(pig.walk_time, 0.0)
        # AI wander state should NOT have changed on remote client
        self.assertEqual(wander.state, "idle")

    def test_network_client_attack_methods(self):
        from client.network import NetworkClient

        client = NetworkClient("127.0.0.1", 25565, "uuid", "player", None)
        client.connected = True

        # send_attack_mob
        client.send_attack_mob(42)
        events = client.out_queue
        self.assertEqual(len(events), 1)

        # generic send
        client.send({"type": "test_packet", "val": 123})
        self.assertEqual(len(events), 2)

    def test_server_attack_entity_handling(self):
        import tempfile
        from server.network import MultiplayerServer, RemoteClient

        with tempfile.TemporaryDirectory() as tmp:
            server = MultiplayerServer(host="127.0.0.1", port=0, saves_dir=tmp, world=World(seed=1))
            mock_client = MagicMock(spec=RemoteClient)
            mock_client.alive = True
            mock_client.uuid = "player-1"
            mock_client.pos = [10.0, 5.0, 10.0]
            server.clients["player-1"] = mock_client

            pig = Pig(pos=[10.5, 5.0, 10.0], health=10.0)
            server.world.entities.append(pig)

            broadcasted = []
            server.broadcast = lambda pkt: broadcasted.append(pkt)

            # Client attacks mob
            server._handle_packet(mock_client, {"type": "attack_entity", "id": pig.id})

            self.assertEqual(pig.health, 8.0)
            self.assertGreater(pig.hurt_time, 0.0)
            # Server broadcasted damage_mob packet
            self.assertTrue(any(p.get("type") == "damage_mob" and p.get("id") == pig.id for p in broadcasted))


class TestSheepAndWool(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=1)
        self.chunk = Chunk(0, 0, 0)
        self.chunk.data.fill(0)
        self.world.chunks[(0, 0, 0)] = self.chunk

    def test_sheep_initialization_and_serialization(self):
        sheep = Sheep(pos=[4.5, 6.0, 4.5], yaw=0.75, sheared=True, color="red", regrow_cooldown=8.5)
        self.assertEqual(sheep.mob_type, "sheep")
        self.assertTrue(sheep.sheared)
        self.assertEqual(sheep.color, "red")
        self.assertEqual(sheep.wool_block, "red_wool")

        goals_types = [type(g) for _, g in sheep.goals.goals]
        self.assertIn(PanicGoal, goals_types)
        self.assertIn(WanderGoal, goals_types)
        self.assertIn(EatGrassGoal, goals_types)

        data = sheep.to_dict()
        self.assertEqual(data["type"], "sheep")
        self.assertTrue(data["sheared"])
        self.assertEqual(data["color"], "red")

        restored = entity_from_dict(data)
        self.assertIsInstance(restored, Sheep)
        self.assertTrue(restored.sheared)
        self.assertEqual(restored.color, "red")
        self.assertAlmostEqual(restored.regrow_cooldown, 8.5, places=1)

    def test_shearing_drops_1_to_3_wool_and_marks_sheared(self):
        sheep = Sheep(pos=[5.5, 6.0, 5.5])
        self.world.spawn_mob(sheep)
        self.assertFalse(sheep.sheared)

        drops = sheep.shear(world=self.world)
        self.assertTrue(sheep.sheared)
        self.assertEqual(len(drops), 1)
        wool_id, count = drops[0]
        self.assertEqual(wool_id, "wool")
        self.assertGreaterEqual(count, 1)
        self.assertLessEqual(count, 3)

        # Verify ItemEntities spawned in world
        spawned_wool = [e for e in self.world.entities if getattr(e, "item", "") == "wool"]
        self.assertEqual(len(spawned_wool), count)

        # Shearing an already-sheared sheep yields nothing
        again = sheep.shear(world=self.world)
        self.assertEqual(again, [])

    def test_sheared_sheep_eats_grass_not_muddy_grass_and_regrows_wool(self):
        # Set up ground: muddy_grass at (5, 5, 5)
        for x in range(2, 9):
            for z in range(2, 9):
                self.world.set_block(x, 5, z, STONE)
        self.world.set_block(5, 5, 5, "muddy_grass")

        sheep = Sheep(pos=[5.5, 6.0, 5.5], sheared=True, regrow_cooldown=0.0)
        sheep.on_ground = True
        self.world.spawn_mob(sheep)

        eat_goal = next(g for _, g in sheep.goals.goals if isinstance(g, EatGrassGoal))
        # Standing on muddy_grass with no grass nearby: cannot start EatGrassGoal
        self.assertFalse(eat_goal.can_start())
        for _ in range(25):
            sheep.update(0.05, self.world)
        self.assertEqual(self.world.get_block(5, 5, 5), "muddy_grass")
        self.assertTrue(sheep.sheared)

        # Now place actual 'grass' under the sheep at (5, 5, 5)
        self.world.set_block(5, 5, 5, "grass")
        sheep.pos = [5.5, 6.0, 5.5]
        self.assertTrue(eat_goal.can_start())

        # Tick the sheep until it finishes eating grass (~0.8s)
        sheep.update(0.05, self.world)
        self.assertLess(sheep.pitch, 0.0)  # Head pitched downward while eating grass

        for _ in range(25):
            sheep.update(0.05, self.world)

        # Grass turned into dirt and sheep regrew its wool!
        self.assertEqual(self.world.get_block(5, 5, 5), "dirt")
        self.assertFalse(sheep.sheared)
        self.assertEqual(sheep.pitch, 0.0)

    def test_wool_dyeing_recipes(self):
        # Dye white wool with red_dye -> red_wool
        res = craft_match(["wool", "red_dye", "", ""])
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "red_wool")
        self.assertEqual(res[1], 1)

        # Dye red_wool with yellow_dye -> yellow_wool (via #wool tag)
        res2 = craft_match(["red_wool", "yellow_dye", "", ""])
        self.assertIsNotNone(res2)
        self.assertEqual(res2[0], "yellow_wool")
        self.assertEqual(res2[1], 1)

        # Craft red_dye from rose and yellow_dye from dandelion
        r_dye = craft_match(["rose", "", "", ""])
        self.assertIsNotNone(r_dye)
        self.assertEqual(r_dye[0], "red_dye")

        y_dye = craft_match(["dandelion", "", "", ""])
        self.assertIsNotNone(y_dye)
        self.assertEqual(y_dye[0], "yellow_dye")

        # Craft shears from 2 iron ingots
        shears_res = craft_match(["", "iron_ingot", "iron_ingot", ""])
        self.assertIsNotNone(shears_res)
        self.assertEqual(shears_res[0], "shears")

    def test_sheep_model_meshes(self):
        meshes = build_sheep_meshes()
        self.assertIn("wool_body", meshes)
        self.assertIn("body", meshes)
        sheep_meshes = SheepMeshes()
        poses = sheep_pose(1.0, 1.0, 0.0)
        v_full, _ = sheep_meshes.posed_vertices([0.0, 0.0, 0.0], 0.0, poses, sheared=False)
        v_sheared, _ = sheep_meshes.posed_vertices([0.0, 0.0, 0.0], 0.0, poses, sheared=True)
        self.assertGreater(len(v_full), len(v_sheared))


class TestPathfindingHazardAvoidance(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=1)
        self.chunk = Chunk(0, 0, 0)
        self.chunk.data.fill(0)
        self.world.chunks[(0, 0, 0)] = self.chunk

    def test_pig_and_sheep_avoid_falls_greater_than_4_blocks(self):
        # Create a ledge at y=10 for x in 2..5, z in 2..8
        # For x >= 6, ground is at y=4 (a 6-block drop, which is > 4 blocks!)
        for x in range(2, 6):
            for z in range(2, 8):
                self.world.set_block(x, 10, z, STONE)
        for x in range(6, 12):
            for z in range(2, 8):
                self.world.set_block(x, 4, z, STONE)

        for mob_cls in (Pig, Sheep):
            mob = mob_cls(pos=[5.2, 11.0, 5.0], yaw=-math.pi / 2)  # Facing +X toward the 6-block cliff
            mob.on_ground = True
            self.world.spawn_mob(mob)

            # Moving toward +X should be recognized as hazardous
            self.assertFalse(is_direction_safe(mob, -math.pi / 2, world=self.world))
            self.assertTrue(is_position_hazardous(self.world, 6.5, 11.0, 5.0, max_fall=4))

            # Attempting horizontal movement toward the cliff with avoid_hazards=True stops before falling
            mob.vel = [2.5, 0.0, 0.0]
            for _ in range(20):
                mob.update(0.05, self.world)
            self.assertLess(mob.pos[0], 6.0)
            self.assertAlmostEqual(mob.pos[1], 11.0, delta=0.1)

    def test_fall_of_4_blocks_or_less_is_allowed(self):
        # Ledge at y=10 for x in 2..5; ground at y=6 for x in 6..10 (exact 4-block drop: 11 -> 7)
        for x in range(2, 6):
            for z in range(2, 8):
                self.world.set_block(x, 10, z, STONE)
        for x in range(6, 10):
            for z in range(2, 8):
                self.world.set_block(x, 6, z, STONE)

        self.assertFalse(is_position_hazardous(self.world, 6.5, 11.0, 5.0, max_fall=4))

    def test_pig_and_sheep_avoid_fluids(self):
        # Solid ground at y=5 everywhere, except x >= 6 has water pool at y=5..6
        for x in range(2, 10):
            for z in range(2, 8):
                if x < 6:
                    self.world.set_block(x, 5, z, STONE)
                else:
                    self.world.set_block(x, 4, z, STONE)
                    self.world.set_block(x, 5, z, "water")

        for mob_cls in (Pig, Sheep):
            mob = mob_cls(pos=[5.2, 6.0, 5.0], yaw=-math.pi / 2)  # Facing +X toward water
            mob.on_ground = True
            self.world.spawn_mob(mob)

            self.assertFalse(is_direction_safe(mob, -math.pi / 2, world=self.world))
            self.assertTrue(is_position_hazardous(self.world, 6.5, 6.0, 5.0, max_fall=4))

            mob.vel = [2.5, 0.0, 0.0]
            for _ in range(20):
                mob.update(0.05, self.world)
            self.assertLess(mob.pos[0], 6.0)
            self.assertFalse(mob.in_water(self.world))


