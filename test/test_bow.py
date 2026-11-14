import math
import unittest
from unittest.mock import MagicMock

from client.app import Game
from server.entity import ArrowEntity, ItemEntity
from server.items import ITEMS
from server.player import Player
from server.player_model import first_person_held_matrix, held_item_matrix, nocked_arrow_matrix
from server.ranged import (
    ARROWS,
    BOWS,
    MIN_POWER,
    aim_direction,
    arrow_from_mapping,
    bow_from_mapping,
    fire_arrow,
    launch_speed,
)
from server.recipes import craft_match
from server.stacks import empty_stack, make_stack, stack_count, stack_damage, stack_id
from server.world import World


def _world():
    world = World(seed=1, caves=[])
    world.load_chunk(0, 5, 0)
    world.chunks[(0, 5, 0)].data.fill(0)
    return world


class _Spread:
    def uniform(self, low, high):
        return high


class BowTests(unittest.TestCase):
    def test_bow_and_arrow_assets_and_recipes(self):
        bow = BOWS["wood_bow"]
        arrow = ARROWS["arrow"]
        self.assertEqual(bow.item, "wood_bow")
        self.assertGreater(bow.distance, 0)
        self.assertGreater(bow.draw_time, 0)
        self.assertGreaterEqual(bow.accuracy, 0)
        self.assertLessEqual(bow.accuracy, 1)
        self.assertEqual(arrow.damage, 5)
        self.assertEqual(arrow.effects, ())
        self.assertIn("wood_bow", ITEMS)
        self.assertIn("arrow", ITEMS)

        made = craft_match((
            "", "stick", "string",
            "stick", "", "string",
            "", "stick", "string",
        ))
        self.assertIsNotNone(made)
        self.assertEqual(made[0], "wood_bow")
        arrows = craft_match((
            "flint", "", "",
            "stick", "", "",
            "plant_fiber", "", "",
        ))
        self.assertEqual(arrows[0], "arrow")
        self.assertEqual(arrows[1], 4)

        custom = arrow_from_mapping(
            {"id": "arrow", "damage": 2, "effects": [{"id": "poison", "seconds": 3}]},
            "test",
        )
        self.assertEqual(custom.effects[0]["id"], "poison")
        self.assertEqual(custom.effects[0]["seconds"], 3)

    def test_bow_distance_and_accuracy_shape_the_shot(self):
        arrow = ARROWS["arrow"]
        short = bow_from_mapping({"id": "wood_bow", "accuracy": 1, "distance": 12, "draw_time": 0.5}, "test")
        long = bow_from_mapping({"id": "wood_bow", "accuracy": 1, "distance": 36, "draw_time": 0.5}, "test")
        self.assertLess(launch_speed(short, arrow, 1), launch_speed(long, arrow, 1))
        self.assertAlmostEqual(launch_speed(long, arrow, 0.5), launch_speed(long, arrow, 1) * 0.5)

        straight = aim_direction(long, 1, (0.0, 0.0, -1.0))
        self.assertAlmostEqual(straight[0], 0.0, places=5)
        self.assertAlmostEqual(straight[1], 0.0, places=5)
        self.assertAlmostEqual(straight[2], -1.0, places=5)
        wild = bow_from_mapping({"id": "wood_bow", "accuracy": 0, "distance": 36, "draw_time": 0.5}, "test")
        bent = aim_direction(wild, 1, (0.0, 0.0, -1.0), _Spread())
        self.assertGreater(abs(bent[0]) + abs(bent[1]), 0.01)

        world = _world()
        fired = fire_arrow(world, long, arrow, [2.0, 84.0, 8.0], (1.0, 0.0, 0.0), 1)
        flight = math.sqrt((2.0 * 1.25) / arrow.gravity)
        steps = int(flight / 0.02)
        for _ in range(steps):
            fired.update(0.02, world)
        self.assertGreater(fired.pos[0] - 2.0, long.distance * 0.75)
        self.assertLess(84.0 - fired.pos[1], 2.0)

    def test_arrow_hits_a_mob_and_a_block(self):
        world = _world()
        pig = world.spawn_mob("pig", pos=[8.0, 82.0, 8.0])
        health = pig.health
        arrow = ArrowEntity("arrow", pos=[4.0, 82.45, 8.0], vel=[40.0, 0.0, 0.0], damage=5, power=1, gravity=0)
        world.entities.append(arrow)
        for _ in range(8):
            arrow.update(0.05, world)
        self.assertFalse(arrow.alive)
        self.assertAlmostEqual(pig.health, health - 5)
        self.assertFalse(any(isinstance(entity, ItemEntity) for entity in world.entities))

        missed = _world()
        missed.set_block(6, 82, 8, "stone")
        shot = ArrowEntity("arrow", pos=[4.2, 82.5, 8.2], vel=[30.0, 0.0, 0.0], damage=5, power=1, gravity=0)
        missed.entities.append(shot)
        for _ in range(10):
            if not shot.alive:
                break
            shot.update(0.05, missed)
        self.assertFalse(shot.alive)
        drops = [entity for entity in missed.entities if isinstance(entity, ItemEntity)]
        self.assertEqual([entity.item for entity in drops], ["arrow"])

    def test_arrow_effects_and_owner_protection(self):
        world = _world()
        pig = world.spawn_mob("pig", pos=[8.0, 82.0, 8.0])
        plain = ArrowEntity("arrow", pos=[4.0, 82.45, 8.0], vel=[40.0, 0.0, 0.0], damage=1, power=1, gravity=0)
        plain._strike_mob(pig, world)
        plain_speed = math.hypot(pig.vel[0], pig.vel[2])

        world = _world()
        pig = world.spawn_mob("pig", pos=[8.0, 82.0, 8.0])
        heavy = ArrowEntity(
            "arrow",
            pos=[4.0, 82.45, 8.0],
            vel=[40.0, 0.0, 0.0],
            damage=1,
            power=1,
            gravity=0,
            effects=({"id": "knockback", "amount": 3}, {"id": "poison", "seconds": 2}),
        )
        heavy._strike_mob(pig, world)
        self.assertGreater(math.hypot(pig.vel[0], pig.vel[2]), plain_speed * 2)

        world = _world()
        player = Player(world)
        player.uuid = "shooter"
        player.gamemode = "survival"
        player.pos = [5.0, 82.0, 8.0]
        world.register_player(player)
        own = ArrowEntity(
            "arrow",
            pos=[5.0, 83.2, 8.0],
            vel=[8.0, 0.0, 0.0],
            damage=5,
            power=1,
            gravity=0,
            owner="shooter",
        )
        for _ in range(6):
            own.update(0.05, world)
        self.assertEqual(player.health, player.max_health)
        self.assertTrue(own.alive)

    def test_drawing_a_bow_spends_an_arrow(self):
        world = _world()
        game = MagicMock()
        game.mode = "play"
        game.world = world
        game.player = MagicMock()
        game.player.gamemode = "survival"
        game.player.pos = [2.0, 82.0, 8.0]
        game.player.eye = 1.6
        game.player.yaw = 0.0
        game.player.pitch = 0.0
        game.player.uuid = "archer"
        game.selected = 0
        game.hotbar = [make_stack("wood_bow", 1), make_stack("arrow", 16)]
        game.storage = [empty_stack()]
        game.network_client = None
        game.gamemode = "survival"
        game._bow_drawing = True
        game._bow_draw = 0.01
        game._use_held_tool = lambda amount=1: Game._use_held_tool(game, amount)
        game._find_arrow = lambda: Game._find_arrow(game)
        game._consume_arrow = lambda arrow_id: Game._consume_arrow(game, arrow_id)
        self.assertFalse(Game._release_bow(game))
        self.assertEqual(stack_count(game.hotbar[1]), 16)

        game._bow_drawing = True
        game._bow_draw = 1.0
        self.assertTrue(Game._release_bow(game))
        self.assertEqual(stack_count(game.hotbar[1]), 15)
        self.assertEqual(stack_id(game.hotbar[0]), "wood_bow")
        self.assertEqual(stack_damage(game.hotbar[0]), 1)
        flying = [entity for entity in world.entities if isinstance(entity, ArrowEntity)]
        self.assertEqual(len(flying), 1)
        self.assertEqual(flying[0].damage, ARROWS["arrow"].damage)
        self.assertGreater(flying[0].power, MIN_POWER)

        game.gamemode = "creative"
        game.player.gamemode = "creative"
        game._bow_drawing = True
        game._bow_draw = 1.0
        self.assertTrue(Game._release_bow(game))
        self.assertEqual(stack_count(game.hotbar[1]), 15)
        self.assertEqual(stack_damage(game.hotbar[0]), 1)

    def test_drawing_pulls_the_bow_and_arrow_back(self):
        eye = (0.0, 1.6, 0.0)
        idle = first_person_held_matrix(eye, 0.0, 0.0, item_scale=0.68, flat=True, pull=0.0)
        drawn = first_person_held_matrix(eye, 0.0, 0.0, item_scale=0.68, flat=True, pull=1.0)
        # At yaw 0 the camera looks down -Z, so a larger Z is closer to the eye.
        self.assertGreater(float(drawn[2, 3]), float(idle[2, 3]) + 0.2)
        self.assertGreater(float(drawn[1, 3]), float(idle[1, 3]))
        slack = nocked_arrow_matrix(eye, 0.0, 0.0, 0.2)
        full = nocked_arrow_matrix(eye, 0.0, 0.0, 1.0)
        self.assertGreater(float(full[2, 3]), float(slack[2, 3]) + 0.2)

        rest = held_item_matrix((0.0, 0.0, 0.0), 0.0, 0.0, flat=True, pull=0.0)
        hauled = held_item_matrix((0.0, 0.0, 0.0), 0.0, 0.0, flat=True, pull=1.0)
        self.assertGreater(float(hauled[2, 3]), float(rest[2, 3]))
        self.assertGreater(float(hauled[1, 3]), float(rest[1, 3]))

        player = MagicMock()
        player.yaw = 0.0
        player.pitch = 0.0
        player.pos = [0.0, 0.0, 0.0]
        player.bow_pull = 1.0
        player.nocked_arrow = "arrow"
        from client.player_render import item_hold_matrix

        posed = item_hold_matrix(player, {"arm_r": (0.0, 0.0, 0.0)}, item_id="wood_bow")
        self.assertGreater(float(posed[2, 3]), float(rest[2, 3]) - 0.05)


if __name__ == "__main__":
    unittest.main()
