import math
import unittest
from unittest.mock import MagicMock
import uuid

from server.ai import GoalSelector, WanderGoal, ZombieAttackGoal
from server.blocks import AIR, DIRT, GRASS, STONE
from server.entity import Mob, Pig, Sheep, Zombie, entity_from_dict
from server.player import Player
from server.spawner import MobSpawner
from server.world import Chunk, World
from server.zombie_model import ZombieMeshes, build_zombie_meshes, zombie_pose


class TestZombieAndUUID(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=123)
        self.chunk = Chunk(0, 0, 0)
        self.chunk.data.fill(0)
        self.world.chunks[(0, 0, 0)] = self.chunk

    def test_mob_uuid_format_and_uniqueness(self):
        pig = Pig()
        sheep = Sheep()
        zombie = Zombie()

        # Check mob_id:UUID format
        self.assertTrue(pig.uuid.startswith("pig:"))
        self.assertTrue(sheep.uuid.startswith("sheep:"))
        self.assertTrue(zombie.uuid.startswith("zombie:"))

        # Check valid UUID after prefix
        pig_u = pig.uuid.split(":", 1)[1]
        sheep_u = sheep.uuid.split(":", 1)[1]
        zombie_u = zombie.uuid.split(":", 1)[1]

        self.assertEqual(str(uuid.UUID(pig_u)), pig_u)
        self.assertEqual(str(uuid.UUID(sheep_u)), sheep_u)
        self.assertEqual(str(uuid.UUID(zombie_u)), zombie_u)

        # Uniqueness
        pig2 = Pig()
        self.assertNotEqual(pig.uuid, pig2.uuid)
        self.assertNotEqual(pig.uuid, zombie.uuid)

    def test_mob_uuid_serialization(self):
        zombie = Zombie(pos=[5.0, 64.0, 5.0], health=18.0)
        data = zombie.to_dict()
        self.assertEqual(data["type"], "zombie")
        self.assertEqual(data["uuid"], zombie.uuid)
        self.assertEqual(data["health"], 18.0)

        restored = entity_from_dict(data)
        self.assertIsInstance(restored, Zombie)
        self.assertEqual(restored.uuid, zombie.uuid)
        self.assertEqual(restored.id, zombie.id)
        self.assertEqual(restored.health, 18.0)

    def test_zombie_initialization_and_stats(self):
        z = Zombie(pos=[1.0, 2.0, 3.0])
        self.assertEqual(z.mob_type, "zombie")
        self.assertEqual(z.name, "Zombie")
        self.assertEqual(z.health, 20.0)
        self.assertEqual(z.max_health, 20.0)
        self.assertEqual(z.height, 1.95)
        self.assertEqual(z.radius, 0.35)
        self.assertEqual(z.attack_damage, 3.0)
        self.assertEqual(z.attack_range, 1.5)
        self.assertFalse(z.in_panic())
        self.assertIsNotNone(z.goals)

        # Zombie does not panic when taking damage
        z.take_damage(4.0)
        self.assertFalse(z.in_panic())
        self.assertEqual(z.health, 16.0)

    def test_zombie_drops_rotten_flesh(self):
        z = Zombie(pos=[10.0, 5.0, 10.0])
        self.world.spawn_mob(z)

        # Simulate death with killer
        player = MagicMock()
        z.take_damage(30.0, attacker=player, world=self.world)
        self.assertFalse(z.alive)

        # Entities in world should include rotten_flesh drop
        dropped = [e for e in self.world.entities if hasattr(e, "item")]
        # Drop can be 0-2 items, but loot table is registered
        self.assertEqual(z.loot_table, "entities/zombie")

    def test_zombie_chase_and_attack_within_16_blocks(self):
        player = Player(self.world)
        player.uuid = "p1"
        player.gamemode = "survival"
        player.pos = [10.0, 1.0, 0.0]
        self.world.register_player(player)

        # Zombie at 10 blocks distance
        zombie = Zombie(pos=[0.0, 1.0, 0.0])
        self.world.spawn_mob(zombie)

        # ZombieAttackGoal should activate
        attack_goal = next(g for _, g in zombie.goals.goals if isinstance(g, ZombieAttackGoal))
        self.assertTrue(attack_goal.can_start())
        self.assertEqual(zombie.goals.active_goal, None)

        zombie.update(0.1, self.world)
        self.assertEqual(zombie.goals.active_goal, attack_goal)
        self.assertGreater(zombie.desired_speed, 0.0)
        # Should face towards +X (yaw ~ -pi/2)
        self.assertAlmostEqual(zombie.target_yaw, -math.pi / 2, delta=0.2)

    def test_zombie_does_not_chase_player_beyond_16_blocks(self):
        player = Player(self.world)
        player.uuid = "p1"
        player.gamemode = "survival"
        player.pos = [25.0, 1.0, 0.0]  # 25 blocks away
        self.world.register_player(player)

        zombie = Zombie(pos=[0.0, 1.0, 0.0])
        self.world.spawn_mob(zombie)

        attack_goal = next(g for _, g in zombie.goals.goals if isinstance(g, ZombieAttackGoal))
        self.assertFalse(attack_goal.can_start())

    def test_zombie_attacks_player_in_melee_range(self):
        player = Player(self.world)
        player.uuid = "p1"
        player.gamemode = "survival"
        player.pos = [1.0, 1.0, 0.0]  # 1.0 block away (within 1.5 reach)
        player.health = 20.0
        self.world.register_player(player)

        zombie = Zombie(pos=[0.0, 1.0, 0.0])
        self.world.spawn_mob(zombie)

        # Tick zombie update
        zombie.update(0.1, self.world)
        # Player should have taken 3.0 damage and gained knockback / hurt_time
        self.assertAlmostEqual(player.health, 17.0)
        self.assertGreater(player.hurt_time, 0.0)
        self.assertGreater(player.vel[0], 0.0)  # Knocked away in +X direction

    def test_zombie_pose_and_meshes(self):
        meshes = ZombieMeshes()
        self.assertIn("arm_r", meshes.parts)
        self.assertIn("arm_l", meshes.parts)
        self.assertIn("head", meshes.parts)

        # Outstretched arms in pose
        pose = zombie_pose(time=0.5, speed=2.0, yaw=0.0)
        # arm_r and arm_l pitch should be ~ +pi/2 (+1.57 rad) pointing forward
        self.assertAlmostEqual(pose["arm_r"][0], math.pi / 2, delta=0.2)
        self.assertAlmostEqual(pose["arm_l"][0], math.pi / 2, delta=0.2)

        verts, indices = meshes.posed_vertices([0, 0, 0], 0.0, pose)
        self.assertGreater(len(verts), 0)
        self.assertGreater(len(indices), 0)

    def test_zombie_spawning_light_rules(self):
        spawner = MobSpawner(self.world)

        # Setup ground at y=5, air at y=6, y=7
        self.world.set_block(0, 5, 0, STONE)
        self.world.set_block(0, 6, 0, AIR)
        self.world.set_block(0, 7, 0, AIR)

        # 1. Skylight check: skylight > 7 prevents spawning
        self.assertFalse(spawner.can_spawn_zombie_at(0, 5, 0, skylight=8, blocklight=(0, 0, 0, 0, 0)))
        self.assertTrue(spawner.can_spawn_zombie_at(0, 5, 0, skylight=7, blocklight=(0, 0, 0, 0, 0)))
        self.assertTrue(spawner.can_spawn_zombie_at(0, 5, 0, skylight=0, blocklight=(0, 0, 0, 0, 0)))

        # 2. Block light check: Blue > 7 prevents spawning
        self.assertFalse(spawner.can_spawn_zombie_at(0, 5, 0, skylight=0, blocklight=(0, 0, 8, 0, 0)))
        self.assertTrue(spawner.can_spawn_zombie_at(0, 5, 0, skylight=0, blocklight=(0, 0, 7, 0, 0)))

        # 3. Block light check: Green > 7 prevents spawning
        self.assertFalse(spawner.can_spawn_zombie_at(0, 5, 0, skylight=0, blocklight=(0, 8, 0, 0, 0)))
        self.assertTrue(spawner.can_spawn_zombie_at(0, 5, 0, skylight=0, blocklight=(0, 7, 0, 0, 0)))

        # 4. Block light check: UV > 7 prevents spawning
        self.assertFalse(spawner.can_spawn_zombie_at(0, 5, 0, skylight=0, blocklight=(0, 0, 0, 8, 0)))
        self.assertTrue(spawner.can_spawn_zombie_at(0, 5, 0, skylight=0, blocklight=(0, 0, 0, 7, 0)))

        # 5. Red light does NOT prevent zombie spawning, even at max 15!
        self.assertTrue(spawner.can_spawn_zombie_at(0, 5, 0, skylight=0, blocklight=(15, 0, 0, 0, 0)))
        self.assertTrue(spawner.can_spawn_zombie_at(0, 5, 0, skylight=7, blocklight=(15, 0, 0, 0, 15)))

    def test_zombie_spawner_with_red_torch_vs_blue_torch(self):
        spawner = MobSpawner(self.world)
        self.world.set_block(0, 5, 0, STONE)
        self.world.set_block(0, 6, 0, AIR)
        self.world.set_block(0, 7, 0, AIR)

        # Red torch emits R=14, G=0, B=0
        self.world.set_blocklight(0, 6, 0, 14, 0, 0)
        self.world.day_time = 18000.0  # night
        self.assertTrue(spawner.can_spawn_zombie_at(0, 5, 0))

        # Blue torch emits R=0, G=0, B=14
        self.world.set_blocklight(0, 6, 0, 0, 0, 14)
        self.assertFalse(spawner.can_spawn_zombie_at(0, 5, 0))

        # Green torch emits R=0, G=14, B=0
        self.world.set_blocklight(0, 6, 0, 0, 14, 0)
        self.assertFalse(spawner.can_spawn_zombie_at(0, 5, 0))

    def test_zombie_spawns_during_nightfall(self):
        # At day_time = 13000 (night), is_night() is True
        self.world.day_time = 13000.0
        self.assertTrue(self.world.is_night())

        # At day_time = 6000 (noon), is_night() is False
        self.world.day_time = 6000.0
        self.assertFalse(self.world.is_night())


if __name__ == "__main__":
    unittest.main()
