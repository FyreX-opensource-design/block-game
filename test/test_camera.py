import math
import unittest
from unittest.mock import MagicMock

from client.player_render import VIEW_DISTANCE, VIEW_FIRST, VIEW_THIRD_BACK, VIEW_THIRD_FRONT
from server.player import Player
from server.world import World


class CameraObstructionTestApp:
    """Minimal harness emulating the App camera calculation."""
    def __init__(self, world, player):
        self.world = world
        self.player = player
        self.view_mode = VIEW_THIRD_BACK
        self.eye = (player.pos[0], player.pos[1] + player.eye, player.pos[2])
        self.forward = (0.0, 0.0, -1.0)  # Looking towards -Z

    def _camera_from_view(self):
        from client.app import Game
        return Game._camera_from_view(self)


class TestCameraObstruction(unittest.TestCase):
    def setUp(self):
        self.world = World(seed=1)
        # Clear a chunk area high up
        self.cx, self.cy, self.cz = 0, 5, 0  # y = 80..95
        self.world.load_chunk(self.cx, self.cy, self.cz)
        self.world.chunks[(self.cx, self.cy, self.cz)].data.fill(0)

        self.player = Player(self.world)
        self.player.pos = [8.5, 82.0, 8.5]
        self.app = CameraObstructionTestApp(self.world, self.player)

    def test_first_person_returns_eye(self):
        self.app.view_mode = VIEW_FIRST
        cam = self.app._camera_from_view()
        self.assertEqual(cam, self.app.eye)

    def test_open_space_reaches_full_distance(self):
        self.app.view_mode = VIEW_THIRD_BACK
        cam = self.app._camera_from_view()
        # In open space, camera should be behind player (+Z since looking -Z)
        eye = self.app.eye
        dist = math.sqrt((cam[0] - eye[0])**2 + (cam[1] - eye[1])**2 + (cam[2] - eye[2])**2)
        # Expected distance is sqrt(VIEW_DISTANCE^2 + 0.4^2)
        expected = math.sqrt(VIEW_DISTANCE**2 + 0.4**2)
        self.assertAlmostEqual(dist, expected, delta=0.01)
        # Camera is far along +Z
        self.assertGreater(cam[2], eye[2] + 3.5)

    def test_wall_behind_brings_camera_forward(self):
        # Place a stone wall 1.5 blocks behind the player (player at z=8.5, looking -Z, wall at z=10)
        self.world.set_block(8, 83, 10, "stone")
        self.world.set_block(8, 84, 10, "stone")

        self.app.view_mode = VIEW_THIRD_BACK
        cam = self.app._camera_from_view()

        # The camera must be in front of the wall (z < 10.0)
        self.assertLess(cam[2], 10.0)
        # Camera must not be inside any solid block
        self.assertFalse(self.world.is_solid(math.floor(cam[0]), math.floor(cam[1]), math.floor(cam[2])))

    def test_ceiling_obstruction_prevents_clipping(self):
        # Place ceiling right above player eye
        eye_y = math.floor(self.app.eye[1])
        self.world.set_block(8, eye_y + 1, 8, "stone")
        self.world.set_block(8, eye_y + 1, 9, "stone")

        self.app.view_mode = VIEW_THIRD_BACK
        cam = self.app._camera_from_view()

        # Camera y must not penetrate the ceiling block
        self.assertFalse(self.world.is_solid(math.floor(cam[0]), math.floor(cam[1]), math.floor(cam[2])))

    def test_front_view_obstruction(self):
        # Looking -Z in front third person: camera is in front (towards -Z)
        self.world.set_block(8, 83, 6, "stone")
        self.world.set_block(8, 84, 6, "stone")

        self.app.view_mode = VIEW_THIRD_FRONT
        cam = self.app._camera_from_view()

        # Camera should be stopped before the stone block at z=6 (i.e. z > 6.0)
        self.assertGreater(cam[2], 6.0)
        self.assertFalse(self.world.is_solid(math.floor(cam[0]), math.floor(cam[1]), math.floor(cam[2])))


if __name__ == "__main__":
    unittest.main()
