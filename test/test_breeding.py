import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from client.app import Game
from server.ai import BreedGoal, TemptGoal
from server.blocks import STONE
from server.cow_model import CowMeshes, cow_pose
from server.entity import Cow, Pig, Sheep, entity_from_dict
from server.mobs import MOBS
from server.particles import ParticleSystem
from server.pig_model import PigMeshes, pig_pose
from server.player import Player
from server.sheep_model import SheepMeshes, sheep_pose
from server.stacks import make_stack, stack_count, stack_id
from server.world import Chunk, World


def _height(vertices):
    return float(vertices.reshape(-1, 8)[:, 1].max())


class TestBreeding(unittest.TestCase):
    def test_food_comes_from_the_mob_asset(self):
        self.assertEqual(MOBS["cow"].breed.food, ("wheat",))
        self.assertEqual(MOBS["sheep"].breed.food, ("wheat",))
        self.assertEqual(MOBS["pig"].breed.food, ("wheat", "#carrots", "#potatoes"))
        self.assertEqual(MOBS["cow"].breed.babies, (1, 1))
        self.assertEqual(MOBS["sheep"].breed.babies, (1, 1))
        self.assertEqual(MOBS["pig"].breed.babies, (1, 3))
        self.assertEqual(MOBS["zombie"].breed, None)

        cow = Cow()
        pig = Pig()
        self.assertIsInstance(next(goal for _priority, goal in cow.goals.goals if isinstance(goal, BreedGoal)), BreedGoal)
        self.assertEqual(cow.feed("wheat"), "love")
        self.assertEqual(cow.feed("wheat"), "")
        self.assertEqual(cow.feed("orange_carrot"), "")
        self.assertEqual(pig.feed("stone"), "")
        self.assertEqual(pig.feed("potato_seeds"), "")
        self.assertEqual(pig.feed("orange_carrot"), "love")
        pig.love_time = 0.0
        self.assertEqual(pig.feed("blue_potato"), "love")
        pig.love_time = 0.0
        self.assertEqual(pig.feed("wheat"), "love")
        cow.age = 40.0
        cow.love_time = 0.0
        self.assertEqual(cow.feed("wheat"), "")

    def test_feeding_a_baby_grows_it_and_a_baby_cow_cannot_be_milked(self):
        cow = Cow()
        cow.make_baby()
        self.assertTrue(cow.is_baby)
        self.assertAlmostEqual(cow.radius, cow.adult_radius * 0.5)
        self.assertIsNone(cow.milk("iron_bucket"))
        before = cow.age
        self.assertEqual(cow.feed("wheat"), "grow")
        self.assertAlmostEqual(cow.age, before + (-before) * 0.1)
        cow.age = -0.02
        cow.update(0.05, None)
        self.assertFalse(cow.is_baby)
        self.assertAlmostEqual(cow.radius, cow.adult_radius)
        self.assertEqual(cow.milk("iron_bucket"), "milk_bucket")

    def test_two_adults_in_love_have_young(self):
        world = World(seed=1, caves=[])
        world.spawner = None
        cow_a = world.spawn_mob("cow", pos=[4.0, 8.0, 4.0])
        cow_b = world.spawn_mob("cow", pos=[4.6, 8.0, 4.0])
        sheep = world.spawn_mob("sheep", pos=[4.3, 8.0, 4.2])
        cow_a.love_time = 30.0
        cow_b.love_time = 30.0
        sheep.love_time = 30.0
        world.update_entities(0.05)
        babies = [entity for entity in world.entities if getattr(entity, "is_baby", False)]
        self.assertEqual([baby.mob_type for baby in babies], ["cow"])
        self.assertAlmostEqual(babies[0].age, -babies[0].grow_time, delta=0.1)
        self.assertGreater(cow_a.age, 0.0)
        self.assertGreater(cow_b.age, 0.0)
        self.assertEqual(cow_a.love_time, 0.0)
        self.assertGreater(sheep.love_time, 0.0)

        world.entities = [entity for entity in world.entities if entity.mob_type == "pig" or not getattr(entity, "is_mob", False)]
        pig_a = world.spawn_mob("pig", pos=[2.0, 8.0, 2.0])
        pig_b = world.spawn_mob("pig", pos=[2.5, 8.0, 2.0])
        pig_a.love_time = 30.0
        pig_b.love_time = 30.0
        with patch("server.ai.random.randint", return_value=3):
            world.update_entities(0.05)
        piglets = [entity for entity in world.entities if getattr(entity, "is_mob", False) and entity.is_baby]
        self.assertEqual(len(piglets), 3)
        self.assertTrue(all(piglet.mob_type == "pig" for piglet in piglets))

    def test_a_baby_is_saved_and_the_model_has_a_large_head(self):
        sheep = Sheep(color="red")
        sheep.make_baby()
        restored = entity_from_dict(sheep.to_dict())
        self.assertTrue(restored.is_baby)
        self.assertEqual(restored.color, "red")
        self.assertAlmostEqual(restored.radius, restored.adult_radius * 0.5)
        self.assertAlmostEqual(restored.age, sheep.age)

        for meshes, pose in (
            (CowMeshes(), cow_pose(0.0, 0.0, 0.0)),
            (PigMeshes(), pig_pose(0.0, 0.0, 0.0)),
            (SheepMeshes(), sheep_pose(0.0, 0.0, 0.0)),
        ):
            adult, _indices = meshes.posed_vertices([0.0, 0.0, 0.0], 0.0, pose, baby=False)
            baby, _indices = meshes.posed_vertices([0.0, 0.0, 0.0], 0.0, pose, baby=True)
            adult_h = _height(adult)
            baby_h = _height(baby)
            self.assertLess(baby_h, adult_h)
            self.assertGreater(baby_h, adult_h * 0.5)

    def test_hearts_rise_while_in_love_and_burst_once_for_a_newborn(self):
        cow = Cow(pos=[1.0, 2.0, 3.0])
        cow.love_time = 30.0
        hearts = ParticleSystem()
        hearts.emit_mob_effects([cow], 0.3)
        self.assertGreaterEqual(len(hearts.particles), 1)
        particle = hearts.particles[0]
        start_y = particle.y
        hearts.update(0.2)
        self.assertGreater(particle.y, start_y)
        hearts.update(5.0)
        self.assertEqual(hearts.particles, [])

        baby = Cow(pos=[1.0, 2.0, 3.0])
        baby.make_baby()
        burst = ParticleSystem()
        burst.emit_mob_effects([baby], 0.01)
        born = len(burst.particles)
        self.assertGreaterEqual(born, 5)
        burst.emit_mob_effects([baby], 0.01)
        self.assertEqual(len(burst.particles), born)

    def test_right_click_feeds_and_still_milks(self):
        cow = Cow(pos=[0.0, 0.0, 0.0])
        game = SimpleNamespace(
            player=SimpleNamespace(pos=[0.0, 0.0, 2.0], eye=1.6, yaw=0.0, pitch=0.0),
            world=SimpleNamespace(raycast_entity=lambda *_args, **_kwargs: cow),
            hotbar=[make_stack("wheat", 4)],
            selected=0,
            network_client=None,
            gamemode="survival",
        )
        game._feed_mob = lambda mob, item_id: Game._feed_mob(game, mob, item_id)
        game._swap_held_for = lambda filled: Game._swap_held_for(game, filled)
        self.assertTrue(Game._handle_mob_interaction(game))
        self.assertEqual(stack_count(game.hotbar[0]), 3)
        self.assertGreater(cow.love_time, 0.0)

        cow.love_time = 0.0
        cow.age = 20.0
        game.hotbar = [make_stack("wheat", 4)]
        self.assertTrue(Game._handle_mob_interaction(game))
        self.assertEqual(stack_count(game.hotbar[0]), 4)
        self.assertEqual(cow.love_time, 0.0)

        cow.age = 0.0
        game.gamemode = "creative"
        game.hotbar = [make_stack("wheat", 4)]
        self.assertTrue(Game._handle_mob_interaction(game))
        self.assertEqual(stack_count(game.hotbar[0]), 4)
        self.assertGreater(cow.love_time, 0.0)

        cow.love_time = 0.0
        game.gamemode = "survival"
        game.hotbar = [make_stack("iron_bucket", 1)]
        self.assertTrue(Game._handle_mob_interaction(game))
        self.assertEqual(stack_id(game.hotbar[0]), "milk_bucket")


def _blank_chunk(world, cx, cy, cz):
    chunk = Chunk(cx, cy, cz)
    chunk.data.fill(0)
    world.chunks[(cx, cy, cz)] = chunk
    return chunk


class TestTempt(unittest.TestCase):
    def test_breeding_food_is_a_goal_behind_breeding(self):
        cow = Cow()
        breed_priority = next(priority for priority, goal in cow.goals.goals if isinstance(goal, BreedGoal))
        tempt_priority = next(priority for priority, goal in cow.goals.goals if isinstance(goal, TemptGoal))
        self.assertLess(breed_priority, tempt_priority)
        self.assertEqual(tempt_priority, 1.6)

    def test_only_the_breeding_item_lures(self):
        world = World(seed=1)
        cow = world.spawn_mob("cow", pos=[0.0, 5.0, 0.0])
        pig = world.spawn_mob("pig", pos=[0.0, 5.0, 0.0])
        player = SimpleNamespace(pos=[4.0, 5.0, 0.0], held_item="wheat", dead=False)
        world.players["p"] = player

        tempt = next(goal for _priority, goal in cow.goals.goals if isinstance(goal, TemptGoal))
        self.assertIs(tempt._holder(), player)
        player.held_item = "orange_carrot"
        self.assertIsNone(tempt._holder())
        pig_tempt = next(goal for _priority, goal in pig.goals.goals if isinstance(goal, TemptGoal))
        self.assertIs(pig_tempt._holder(), player)
        player.held_item = ""
        self.assertIsNone(pig_tempt._holder())

    def test_cow_walks_toward_wheat_and_stops_short(self):
        world = World(seed=1)
        _blank_chunk(world, 0, 0, 0)
        for x in range(0, 16):
            for z in range(0, 16):
                world.set_block(x, 4, z, STONE)

        cow = world.spawn_mob("cow", pos=[2.5, 5.0, 8.0])
        player = Player(world)
        player.uuid = "farmer"
        player.pos = [10.5, 5.0, 8.0]
        player.held_item = "wheat"
        world.register_player(player)

        start = cow.pos[0]
        for _ in range(240):
            world.update_entities(0.05)

        gap = math.hypot(player.pos[0] - cow.pos[0], player.pos[2] - cow.pos[2])
        self.assertGreater(cow.pos[0], start + 2.0)
        self.assertGreater(gap, player.radius + cow.radius + 0.6)
        self.assertLess(gap, 3.2)
        self.assertEqual(cow.desired_speed, 0.0)


if __name__ == "__main__":
    unittest.main()
