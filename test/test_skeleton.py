import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from server.entity import ArrowEntity, DesertSkeleton, DesertSpider, DesertZombie, ItemEntity, Skeleton, entity_from_dict
from server.loot import load_loot_table
from server.skeleton_model import SkeletonMeshes
from server.player import Player
from server.spawner import MobSpawner
from server.world import World


def _platform():
    world = World(seed=1, caves=[])
    world.load_chunk(0, 5, 0)
    chunk = world.chunks[(0, 5, 0)]
    chunk.data.fill(0)
    for x in range(16):
        for z in range(16):
            world.set_block(x, 80, z, "stone")
    return world


class SkeletonTests(unittest.TestCase):
    def test_desert_columns_spawn_desert_hostiles(self):
        spawner = MobSpawner(World(seed=1, caves=[]))
        with patch("server.biomes.biome_at", return_value=SimpleNamespace(id="desert")):
            desert = [name for name, _check in spawner.hostile_mob_types(3, 9)]
        with patch("server.biomes.biome_at", return_value=SimpleNamespace(id="desert_hills")):
            hills = [name for name, _check in spawner.hostile_mob_types(3, 9)]
        with patch("server.biomes.biome_at", return_value=SimpleNamespace(id="plains")):
            plains = [name for name, _check in spawner.hostile_mob_types(3, 9)]
        self.assertEqual(desert, ["desert_zombie", "desert_skeleton", "desert_spider"])
        self.assertEqual(hills, desert)
        self.assertEqual(plains, ["zombie", "skeleton", "spider"])

    def test_skeleton_draws_and_shoots_without_hitting_itself(self):
        world = _platform()
        skeleton = world.spawn_mob("skeleton", pos=[2.5, 81.0, 8.0])
        self.assertIsInstance(skeleton, Skeleton)
        self.assertEqual(skeleton.mob_type, "skeleton")
        player = Player(world)
        player.uuid = "target"
        player.gamemode = "survival"
        player.pos = [8.0, 81.0, 8.0]
        world.register_player(player)
        health = player.health
        for _ in range(50):
            world.update_entities(0.05)
        self.assertLess(player.health, health)
        self.assertEqual(skeleton.health, skeleton.max_health)

    def test_a_killing_arrow_respawns_the_player(self):
        from client.app import Game

        world = _platform()
        world.spawn_mob("skeleton", pos=[2.5, 81.0, 8.0])
        player = Player(world)
        player.uuid = "target"
        player.gamemode = "survival"
        player.set_health(1.0)
        player.pos = [8.0, 81.0, 8.0]
        world.register_player(player)
        for _ in range(50):
            if player.dead:
                break
            world.update_entities(0.05)
        self.assertTrue(player.dead)
        self.assertEqual(player.health, 0.0)
        self.assertEqual(player.death_cause, "shot by a Skeleton")

        game = MagicMock()
        game.player = player
        game.gamemode = "survival"
        game.chat = MagicMock()
        game._reset_breaking = lambda: None
        game._handle_local_death = lambda events: Game._handle_local_death(game, events)
        cause = player.death_cause
        Game._update_local_health(game, 0.05)
        self.assertFalse(player.dead)
        self.assertEqual(player.health, player.max_health)
        game.chat.add_message.assert_any_call(
            f"You died: {cause}",
            sender="",
            color=(255, 100, 100, 255),
        )

    def test_desert_kinds_round_trip_and_drop_their_loot(self):
        desert_zombie = DesertZombie(pos=[1.0, 2.0, 3.0])
        desert_spider = DesertSpider()
        desert_skeleton = DesertSkeleton()
        self.assertEqual(desert_zombie.mob_type, "desert_zombie")
        self.assertEqual(desert_spider.mob_type, "desert_spider")
        self.assertEqual(desert_skeleton.mob_type, "desert_skeleton")
        self.assertIsInstance(entity_from_dict(desert_skeleton.to_dict()), DesertSkeleton)
        self.assertIsInstance(entity_from_dict(desert_zombie.to_dict()), DesertZombie)
        self.assertIsInstance(entity_from_dict(desert_spider.to_dict()), DesertSpider)

        skeleton = Skeleton()
        skeleton.bow_pull = 0.4
        restored = entity_from_dict(skeleton.to_dict())
        self.assertIsInstance(restored, Skeleton)
        self.assertAlmostEqual(restored.bow_pull, 0.4)
        names = []
        for pool in load_loot_table("entities/skeleton")["pools"]:
            for entry in pool["entries"]:
                names.append(entry["name"])
        self.assertIn("bone", names)
        self.assertIn("arrow", names)
        flesh = [
            entry["name"]
            for pool in load_loot_table("entities/desert_zombie")["pools"]
            for entry in pool["entries"]
        ]
        self.assertEqual(flesh, ["rotten_flesh"])

    def test_skeleton_arrows_are_not_recoverable(self):
        world = _platform()
        world.set_block(6, 82, 8, "stone")
        shot = ArrowEntity(
            "arrow",
            pos=[4.2, 82.5, 8.2],
            vel=[30.0, 0.0, 0.0],
            damage=5,
            power=1,
            gravity=0,
            recoverable=False,
        )
        world.entities.append(shot)
        for _ in range(10):
            if not shot.alive:
                break
            shot.update(0.05, world)
        self.assertFalse(shot.alive)
        self.assertFalse(any(isinstance(entity, ItemEntity) for entity in world.entities))

        saved = ArrowEntity.from_dict(shot.to_dict())
        self.assertFalse(saved.recoverable)

    def test_skeleton_mesh_is_a_ribcage_with_thin_limbs(self):
        meshes = SkeletonMeshes()
        body, _indices = meshes.parts["body"]
        self.assertGreater(body.shape[0] // 8, 24 * 4)
        arm, _indices = meshes.parts["arm_r"]
        xs = arm.reshape(-1, 8)[:, 0]
        self.assertLess(float(xs.max() - xs.min()), 3.0 / 16.0)
        from server.zombie_model import skeleton_pose

        vertices, indices = meshes.posed_vertices([1.0, 2.0, 3.0], 0.4, skeleton_pose(0.0, 0.0, 0.4))
        self.assertGreater(len(vertices), 0)
        self.assertEqual(len(indices) % 3, 0)


if __name__ == "__main__":
    unittest.main()
