import math
import unittest

from server.entity import Cow, Pig
from server.player import Player
from server.world import World


class TestEntityCollision(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=1, caves=[])
        self.world.spawner = None

    def test_player_stops_at_a_mob_and_another_player(self):
        player = Player(self.world)
        player.uuid = "local"
        player.pos = [0.0, 200.0, 0.0]
        cow = Cow(pos=[1.0, 200.0, 0.0])
        self.world.entities.append(cow)

        player.sweep(0, 3.0)
        limit = cow.pos[0] - cow.radius - player.radius
        self.assertLess(player.pos[0], limit + 0.02)
        self.assertGreater(player.pos[0], 0.0)

        friend = Player(self.world)
        friend.uuid = "friend"
        friend.pos = [0.0, 200.0, 1.0]
        self.world.remote_players = {"friend": friend}
        player.pos = [0.0, 200.0, 0.0]
        player.sweep(2, 3.0)
        z_limit = friend.pos[2] - friend.radius - player.radius
        self.assertLess(player.pos[2], z_limit + 0.02)

    def test_mobs_stop_at_each_other_and_at_a_player(self):
        first = Pig(pos=[0.0, 200.0, 0.0])
        second = Pig(pos=[1.2, 200.0, 0.0])
        self.world.entities.extend((first, second))
        first.sweep(0, 3.0, self.world)
        limit = second.pos[0] - second.radius - first.radius
        self.assertLess(first.pos[0], limit + 0.02)
        self.assertGreater(first.pos[0], 0.2)

        player = Player(self.world)
        player.uuid = "local"
        player.pos = [0.0, 200.0, 1.2]
        self.world.players[player.uuid] = player
        walker = Pig(pos=[0.0, 200.0, 0.0])
        self.world.entities.append(walker)
        walker.sweep(2, 3.0, self.world)
        z_limit = player.pos[2] - player.radius - walker.radius
        self.assertLess(walker.pos[2], z_limit + 0.02)

    def test_items_do_not_block_and_an_overlap_can_be_left(self):
        player = Player(self.world)
        player.uuid = "local"
        player.pos = [0.0, 200.0, 0.0]
        self.world.spawn_item("wheat", pos=[0.4, 200.0, 0.0])
        player.sweep(0, 2.0)
        self.assertGreater(player.pos[0], 1.5)

        cow = Cow(pos=[0.2, 200.0, 0.0])
        self.world.entities.append(cow)
        player.pos = [0.2, 200.0, 0.0]
        self.assertTrue(player.collides_at(0.2, 200.0, 0.0))
        player.sweep(0, -1.0)
        self.assertLess(player.pos[0], 0.0)

    def test_overlapping_bodies_are_pushed_apart(self):
        from server.collision import overlaps_body, separate_from_bodies

        first = Pig(pos=[4.0, 80.0, 4.0])
        second = Pig(pos=[4.1, 80.0, 4.0])
        self.world.entities.extend((first, second))
        separate_from_bodies(first, self.world)
        self.assertFalse(overlaps_body(first.pos[0], first.pos[1], first.pos[2], first.radius, first.height, second))
        self.assertAlmostEqual(first.pos[1], 80.0, delta=0.01)
        self.assertGreater(math.hypot(first.pos[0] - second.pos[0], first.pos[2] - second.pos[2]), first.radius + second.radius - 0.02)


if __name__ == "__main__":
    unittest.main()
