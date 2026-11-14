import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock
import uuid

from server.ai import SpiderAttackGoal
from server.blocks import AIR, STONE
from server.entity import Mob, Spider, entity_from_dict
from server.items import ITEMS
from server.loot import load_loot_table, roll_loot_table
from server.player import Player
from server.spawner import MobSpawner
from server.spider_model import SpiderMeshes, build_spider_meshes, spider_pose
from server.storage import ChunkStore
from server.world import Chunk, World


class TestSpider(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=42)
        # Create chunk (0, 0, 0)
        self.chunk = Chunk(0, 0, 0)
        self.chunk.data.fill(0)
        self.world.chunks[(0, 0, 0)] = self.chunk

    def test_spider_uuid_and_serialization(self):
        spider = Spider(pos=[5.0, 10.0, 5.0], health=14.0)
        self.assertTrue(spider.uuid.startswith("spider:"))
        spider_u = spider.uuid.split(":", 1)[1]
        self.assertEqual(str(uuid.UUID(spider_u)), spider_u)

        data = spider.to_dict()
        self.assertEqual(data["type"], "spider")
        self.assertEqual(data["uuid"], spider.uuid)
        self.assertEqual(data["health"], 14.0)

        restored = entity_from_dict(data)
        self.assertIsInstance(restored, Spider)
        self.assertEqual(restored.uuid, spider.uuid)
        self.assertEqual(restored.id, spider.id)
        self.assertEqual(restored.health, 14.0)

    def test_spider_stats_and_properties(self):
        spider = Spider(pos=[1.0, 2.0, 3.0])
        self.assertEqual(spider.mob_type, "spider")
        self.assertEqual(spider.name, "Spider")
        self.assertEqual(spider.health, 16.0)
        self.assertEqual(spider.max_health, 16.0)
        self.assertEqual(spider.height, 0.8)
        self.assertEqual(spider.radius, 0.6)
        self.assertEqual(spider.attack_damage, 2.5)
        self.assertEqual(spider.attack_range, 1.8)
        self.assertEqual(spider.loot_table, "entities/spider")

    def test_spider_items_and_drops(self):
        # Items should be registered in assets
        string_item = ITEMS.get("string")
        self.assertIsNotNone(string_item)
        self.assertEqual(string_item.max_stack, 64)

        eye_item = ITEMS.get("spider_eye")
        self.assertIsNotNone(eye_item)
        self.assertEqual(eye_item.food, 2)

        fang_item = ITEMS.get("spider_fang")
        self.assertIsNotNone(fang_item)
        self.assertEqual(fang_item.attack, 2.0)

        # Loot table test
        table = load_loot_table("entities/spider")
        self.assertIsNotNone(table)
        rolls = [roll_loot_table(table) for _ in range(50)]
        all_item_ids = {item_id for roll in rolls for item_id, _ in roll}
        self.assertIn("string", all_item_ids)
        self.assertIn("spider_eye", all_item_ids)
        self.assertIn("spider_fang", all_item_ids)

    def test_spider_targeting_light_conditions(self):
        spider = Spider(pos=[5.0, 1.0, 5.0])
        self.world.spawn_mob(spider)

        player = Player(self.world)
        player.uuid = "player_1"
        player.gamemode = "survival"
        player.pos = [6.0, 1.0, 5.0]
        self.world.register_player(player)

        # Mock world get_light
        # Case 1: Low skylight (0) and low IR (0) -> Targets player
        self.world.get_light = MagicMock(return_value={
            "sky": 0, "current_sky": 0, "r": 0, "g": 0, "b": 0, "uv": 0, "ir": 0
        })
        self.assertTrue(spider.can_target_player(player, self.world))

        # Case 2: Skylight <= 10 (10) and IR <= 10 (10) -> Targets player
        self.world.get_light = MagicMock(return_value={
            "sky": 10, "current_sky": 10, "r": 0, "g": 0, "b": 0, "uv": 0, "ir": 10
        })
        self.assertTrue(spider.can_target_player(player, self.world))

        # Case 3: Skylight > 10 (11) and IR <= 10 (0) -> Does NOT target player
        self.world.get_light = MagicMock(return_value={
            "sky": 11, "current_sky": 11, "r": 0, "g": 0, "b": 0, "uv": 0, "ir": 0
        })
        self.assertFalse(spider.can_target_player(player, self.world))

        # Case 4: Skylight <= 10 (0) and IR > 10 (11) -> Does NOT target player
        self.world.get_light = MagicMock(return_value={
            "sky": 0, "current_sky": 0, "r": 0, "g": 0, "b": 0, "uv": 0, "ir": 11
        })
        self.assertFalse(spider.can_target_player(player, self.world))

        # Case 5: Both > 10 (15, 15) -> Does NOT target player
        self.world.get_light = MagicMock(return_value={
            "sky": 15, "current_sky": 15, "r": 15, "g": 15, "b": 15, "uv": 15, "ir": 15
        })
        self.assertFalse(spider.can_target_player(player, self.world))

    def test_spider_attack_goal_execution(self):
        player = Player(self.world)
        player.uuid = "player_1"
        player.gamemode = "survival"
        player.pos = [6.0, 1.0, 5.0]
        self.world.register_player(player)

        spider = Spider(pos=[5.0, 1.0, 5.0])
        self.world.spawn_mob(spider)

        goal = SpiderAttackGoal(spider, target_range=16.0, reach=1.8, damage=2.5)

        # In the dark: can start
        self.world.get_light = MagicMock(return_value={
            "sky": 0, "current_sky": 0, "r": 0, "g": 0, "b": 0, "uv": 0, "ir": 0
        })
        self.assertTrue(goal.can_start())
        goal.start()
        self.assertTrue(goal.can_continue())

        # Now high light is introduced (e.g. sunlight or high IR torch)
        self.world.get_light = MagicMock(return_value={
            "sky": 14, "current_sky": 14, "r": 0, "g": 0, "b": 0, "uv": 0, "ir": 0
        })
        # Goal should now stop
        self.assertFalse(goal.can_continue())
        goal.tick(0.05)
        self.assertIsNone(goal.target)

    def test_spider_climbing_wall(self):
        # Place a stone block next to the spider to form a wall
        # Spider at (5.0, 1.0, 5.0)
        # Floor at y=0
        for x in range(4, 7):
            for z in range(4, 7):
                self.world.set_block(x, 0, z, STONE)

        # Wall at (6, 1, 5) and (6, 2, 5)
        self.world.set_block(6, 1, 5, STONE)
        self.world.set_block(6, 2, 5, STONE)

        spider = Spider(pos=[5.4, 1.0, 5.0])
        self.world.spawn_mob(spider)

        self.assertTrue(spider.is_touching_wall(self.world))

        # When moving towards the wall, spider climbs up and faces into the wall
        spider.vel = [0.1, 0.0, 0.0]
        spider.update(0.05, self.world)
        self.assertTrue(spider.climbing)
        self.assertGreater(spider.vel[1], 0.0)
        # Wall is at +x (East) -> yaw should face East (-pi/2)
        self.assertAlmostEqual(spider.yaw, -math.pi / 2, places=4)

    def test_spider_ceiling_crawling(self):
        from server.ai import is_direction_safe

        # Create a ceiling strip along x directly above the spider
        # Ceiling at y=2 from x=4 to x=8
        for x in range(4, 9):
            self.world.set_block(x, 2, 5, STONE)

        spider = Spider(pos=[5.0, 1.1, 5.0])
        self.world.spawn_mob(spider)

        self.assertTrue(spider.is_touching_ceiling(self.world))

        # Update spider on ceiling
        spider.update(0.05, self.world)
        self.assertTrue(spider.on_ceiling)
        self.assertTrue(spider.on_ground)  # Can traverse horizontally without falling off
        self.assertGreaterEqual(spider.vel[1], 0.0)

        # Pathing: Moving along the ceiling (+x heading) must be safe (not rejected by fall check)
        # Heading towards +x: fx = -sin(yaw) -> yaw = -pi/2
        safe = is_direction_safe(spider, -math.pi / 2, world=self.world, lookahead=1.0)
        self.assertTrue(safe)

        # Move horizontally under the ceiling with avoid_hazards=True
        # Spider should advance and NOT have velocity zeroed out
        spider.vel[0] = 1.0
        spider.move_horizontal(0.1, 0.0, self.world, avoid_hazards=True)
        self.assertGreater(spider.pos[0], 5.0)

    def test_spawner_can_spawn_spider(self):
        spawner = MobSpawner(self.world)
        self.world.set_block(5, 1, 5, STONE)
        self.world.set_block(5, 2, 5, AIR)
        self.world.set_block(5, 3, 5, AIR)

        # Dark conditions
        self.assertTrue(spawner.can_spawn_spider_at(5, 1, 5, skylight=0, blocklight=(0, 0, 0, 0, 0)))

        # Bright daylight prevents spawning
        self.assertFalse(spawner.can_spawn_spider_at(5, 1, 5, skylight=10, blocklight=(0, 0, 0, 0, 0)))

        # Bright block light (blue/green/uv) prevents spawning
        self.assertFalse(spawner.can_spawn_spider_at(5, 1, 5, skylight=0, blocklight=(0, 10, 0, 0, 0)))

    def test_spider_model_and_poses(self):
        meshes = build_spider_meshes()
        self.assertIn("head", meshes)
        self.assertIn("thorax", meshes)
        self.assertIn("abdomen", meshes)
        for i in range(1, 5):
            self.assertIn(f"leg_r{i}", meshes)
            self.assertIn(f"leg_l{i}", meshes)

        # Normal walking pose
        poses = spider_pose(time=1.0, speed=0.8, yaw=0.0, pitch=0.0, climbing=False, on_ceiling=False)
        self.assertIn("leg_r1", poses)
        self.assertIn("leg_l1", poses)

        # Climbing pose has pitch tilt for whole spider surface alignment (head points up, belly on wall)
        climb_poses = spider_pose(time=1.0, speed=0.8, yaw=0.0, pitch=0.0, climbing=True, on_ceiling=False)
        self.assertGreater(climb_poses["__surface__"][0], 0.0)  # Tilted upward against the wall

        # Ceiling pose has 180 deg roll for upside down crawling
        ceiling_poses = spider_pose(time=1.0, speed=0.8, yaw=0.0, pitch=0.0, climbing=False, on_ceiling=True)
        self.assertAlmostEqual(ceiling_poses["__surface__"][1], math.pi, places=4)

        # Legs do not have longitudinal pitch jitter
        for name in ("leg_r1", "leg_l1", "leg_r2", "leg_l2"):
            self.assertEqual(poses[name][0], 0.0)

        # Posed vertices generate valid arrays
        spider_meshes = SpiderMeshes()
        verts, indices = spider_meshes.posed_vertices([5.0, 1.0, 5.0], 0.0, poses)
        self.assertEqual(len(verts), 2112)
        self.assertEqual(len(indices), 396)

    def test_spider_world_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ChunkStore(Path(folder) / "chunks")
            world = World(seed=999, store=store)

            spider = Spider(pos=[12.0, 30.0, 14.0], health=11.5)
            spider.climbing = True
            spider.on_ceiling = False
            world.spawn_mob(spider)

            world.flush()

            # Reopen world with same store
            new_world = World(seed=999, store=store)
            spiders = [e for e in new_world.entities if isinstance(e, Spider)]
            self.assertEqual(len(spiders), 1)
            loaded_spider = spiders[0]
            self.assertEqual(loaded_spider.uuid, spider.uuid)
            self.assertEqual(loaded_spider.health, 11.5)
            self.assertEqual(loaded_spider.pos, [12.0, 30.0, 14.0])
            self.assertTrue(loaded_spider.climbing)
            self.assertFalse(loaded_spider.on_ceiling)
            self.assertIs(loaded_spider.world, new_world)


if __name__ == "__main__":
    unittest.main()
