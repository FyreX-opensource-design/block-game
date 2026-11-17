import json
import tempfile
import unittest
from pathlib import Path

from server.ai import SkeletonShootGoal, WanderGoal, ZombieAttackGoal
from server.blocks import AIR, STONE
from server.entity import Pig, Skeleton, Zombie
from server.mob_model import _eval_molang
from server.mobs import MOBS, build_goals, load_mobs, spawn_allowed
from server.world import World


class MobAssetTests(unittest.TestCase):
    def test_shipped_mobs_take_their_goals_from_assets(self):
        zombie = Zombie()
        kinds = [type(goal).__name__ for _priority, goal in zombie.goals.goals]
        self.assertEqual(kinds, ["ZombieAttackGoal", "WanderGoal"])
        self.assertEqual(zombie.goals.goals[0][1].damage, 3.0)
        skeleton_goal = next(goal for _priority, goal in Skeleton().goals.goals if isinstance(goal, SkeletonShootGoal))
        self.assertEqual(skeleton_goal.preferred, 14.0)
        self.assertEqual(MOBS["cow"].model.kind, "builtin")
        self.assertEqual(MOBS["desert_zombie"].model.builtin_id, "zombie")

    def test_molang_trig_uses_degrees(self):
        self.assertAlmostEqual(_eval_molang("math.cos(query.anim_time * 180)", 1.0), -1.0)
        self.assertAlmostEqual(_eval_molang("math.sin(90)", 0.0), 1.0)

    def test_lua_model_gecko_model_and_spawn_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "mobs").mkdir()
            geo = {
                "format_version": "1.12.0",
                "minecraft:geometry": [
                    {
                        "description": {"identifier": "geometry.dummy", "texture_width": 16, "texture_height": 16},
                        "bones": [
                            {
                                "name": "body",
                                "pivot": [0, 8, 0],
                                "cubes": [{"origin": [-2, 8, -2], "size": [4, 4, 4], "uv": [0, 0]}],
                            }
                        ],
                    }
                ],
            }
            anim = {
                "format_version": "1.8.0",
                "animations": {
                    "animation.dummy.idle": {"loop": True, "animation_length": 1.0, "bones": {}},
                    "animation.dummy.walk": {
                        "loop": True,
                        "animation_length": 1.0,
                        "bones": {"body": {"rotation": {"0.0": [0, 0, 0], "0.5": [90, 0, 0], "1.0": [0, 0, 0]}}},
                    },
                },
            }
            (root / "dummy.geo.json").write_text(json.dumps(geo), encoding="utf-8")
            (root / "dummy.animation.json").write_text(json.dumps(anim), encoding="utf-8")
            (root / "mobs" / "dummy.lua").write_text(
                """
return {
  id = "dummy",
  name = "Dummy",
  health = 6,
  radius = 0.3,
  height = 1.2,
  model = {
    type = "geckolib",
    geometry = "dummy.geo.json",
    animation = "dummy.animation.json",
    texture = "entity/zombie.png",
    idle = "animation.dummy.idle",
    walk = "animation.dummy.walk",
  },
  goals = {
    { priority = 1, type = "melee", damage = 4, reach = 1.2, range = 8 },
    { priority = 2, type = "wander" },
  },
  spawn = {
    group = "hostile",
    biomes = { "plains" },
    blocks = { "stone" },
    min_y = 4,
    max_y = 10,
    headroom = 1,
    light = { sky = { max = 7 }, red = { max = 3 } },
  },
}
""",
                encoding="utf-8",
            )
            (root / "mobs" / "boxy.lua").write_text(
                """
return {
  id = "boxy",
  name = "Boxy",
  health = 4,
  model = {
    type = "lua",
    texture = "entity/pig.png",
    texture_size = { 16, 16 },
    parts = {
      { name = "body", pivot = { 0, 8, 0 }, box = { -4, 4, -4, 4, 12, 4 }, pose = "still",
        uv = { north = { 0, 0, 8, 8 }, south = { 0, 0, 8, 8 }, east = { 0, 0, 8, 8 }, west = { 0, 0, 8, 8 }, up = { 0, 0, 8, 8 }, down = { 0, 0, 8, 8 } } },
      { name = "leg", pivot = { 0, 4, 0 }, box = { -1, 0, -1, 1, 4, 1 }, pose = "swing",
        uv = { north = { 0, 0, 2, 4 }, south = { 0, 0, 2, 4 }, east = { 0, 0, 2, 4 }, west = { 0, 0, 2, 4 }, up = { 0, 0, 2, 2 }, down = { 0, 0, 2, 2 } } },
    },
  },
  goals = { { priority = 1, type = "wander" } },
  spawn = { group = "passive", time = "day", blocks = { "grass" }, light = { sky = { min = 7 } } },
}
""",
                encoding="utf-8",
            )
            catalog = load_mobs(root / "mobs")
            model = catalog["dummy"].model.compile()
            rest, _indices = model.posed_vertices([0.0, 0.0, 0.0], 0.0, 0.0, speed=1.0)
            turned, _indices = model.posed_vertices([0.0, 0.0, 0.0], 0.0, 0.5, speed=1.0)
            box = catalog["boxy"].model.compile()
            box_vertices, box_indices = box.posed_vertices([1.0, 0.0, 2.0], 0.2, 0.4, speed=1.0, pitch=0.3)

        self.assertEqual(set(catalog), {"dummy", "boxy"})
        world = World(seed=1, caves=[])
        world.load_chunk(0, 0, 0)
        world.set_block(0, 6, 0, STONE)
        world.set_block(0, 7, 0, AIR)
        self.assertTrue(spawn_allowed(catalog["dummy"], world, 0, 6, 0, skylight=0, blocklight=(0, 0, 0, 0, 0)))
        self.assertFalse(spawn_allowed(catalog["dummy"], world, 0, 6, 0, skylight=0, blocklight=(4, 0, 0, 0, 0)))
        self.assertFalse(spawn_allowed(catalog["dummy"], world, 0, 2, 0, skylight=0, blocklight=(0, 0, 0, 0, 0)))
        world.set_block(0, 6, 0, "dirt")
        self.assertFalse(spawn_allowed(catalog["dummy"], world, 0, 6, 0, skylight=0, blocklight=(0, 0, 0, 0, 0)))

        host = Pig()
        host.goals = build_goals(host, catalog["dummy"])
        self.assertIsInstance(host.goals.goals[0][1], ZombieAttackGoal)
        self.assertEqual(host.goals.goals[0][1].damage, 4.0)
        self.assertIsInstance(host.goals.goals[1][1], WanderGoal)

        self.assertGreater(abs(float(rest.reshape(-1, 8)[:, 1].max()) - float(turned.reshape(-1, 8)[:, 1].max())), 0.05)
        self.assertGreater(len(box_vertices), 0)
        self.assertEqual(len(box_indices) % 3, 0)

        mob = world.spawn_mob("cow", pos=[1.0, 2.0, 3.0])
        self.assertEqual(mob.mob_type, "cow")


if __name__ == "__main__":
    unittest.main()
