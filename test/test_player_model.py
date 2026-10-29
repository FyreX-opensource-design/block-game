import math
import tempfile
import unittest
from uuid import UUID

import numpy as np

from client.player_render import VIEW_FIRST, VIEW_THIRD_BACK, VIEW_THIRD_FRONT, held_item_mesh, item_hold_matrix
from server.player import Player
from server.player_model import (
    PART_ORDER,
    build_part_mesh,
    build_player_meshes,
    first_person_held_matrix,
    hand_anchor,
    held_item_matrix,
    part_names,
    part_pivot,
)
from server.profile import load_profile, load_skin_rgba, save_profile, skin_path
from server.world import World


class ProfileTests(unittest.TestCase):
    def test_profile_keeps_a_stable_uuid(self):
        with tempfile.TemporaryDirectory() as folder:
            first = load_profile(folder)
            UUID(first["uuid"])
            again = load_profile(folder)
            self.assertEqual(again["uuid"], first["uuid"])
            self.assertEqual(again["skin"], "default")
            save_profile({"uuid": first["uuid"], "skin": "default"}, folder)
            skin = load_skin_rgba("default")
            self.assertEqual(skin.shape, (64, 64, 4))
            self.assertTrue(skin_path("default").is_file())


class PlayerIdentityTests(unittest.TestCase):
    def test_apply_identity_registers_on_the_world(self):
        world = World(seed=1, caves=[])
        player = Player(world)
        skin = load_skin_rgba("default")
        player.apply_identity("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", skin)
        world.register_player(player)
        self.assertEqual(player.uuid, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        self.assertIs(player.skin, skin)
        self.assertIs(world.players[player.uuid], player)


class ModelTests(unittest.TestCase):
    def test_steve_parts_and_uvs(self):
        self.assertEqual(part_names(), PART_ORDER)
        meshes = build_player_meshes()
        self.assertEqual(set(meshes), set(PART_ORDER))
        head, indices = build_part_mesh("head")
        self.assertEqual(len(indices), 36)
        self.assertEqual(head.shape[0] % 8, 0)
        uvs = head.reshape(-1, 8)[:, 6:8]
        self.assertTrue(np.all(uvs >= 0))
        self.assertTrue(np.all(uvs <= 1))
        # Head front UVs live in the classic 8,8-16,16 region (after flip).
        self.assertTrue(np.any((uvs[:, 0] > 8 / 64) & (uvs[:, 0] < 16 / 64)))
        px, py, pz = part_pivot("arm_r")
        self.assertAlmostEqual(px, -6 / 16)
        self.assertAlmostEqual(py, 22 / 16)

    def test_third_person_flat_item_held_forward(self):
        pos = (0.0, 0.0, 0.0)
        yaw = 0.0
        arm_pitch = 0.0
        mat = held_item_matrix(pos, yaw, arm_pitch, item_scale=0.55, flat=True)
        (verts, _), kind = held_item_mesh("wood_pickaxe", mat)
        self.assertEqual(kind, "item")
        v = verts.reshape(-1, 8)[:, :3]

        hand_x, hand_y, hand_z = (pos[0] - 6.0 / 16.0, pos[1] + 12.0 / 16.0, pos[2] + hand_anchor()[2])
        # Pickaxe extends forward along -Z into the world
        self.assertLess(v[:, 2].min(), hand_z - 0.5)
        # Lateral span stays tightly centered around the hand rather than sticking away to the right
        self.assertAlmostEqual((v[:, 0].min() + v[:, 0].max()) / 2.0, hand_x, delta=0.05)

    def test_third_person_blocks_are_bigger(self):
        pos = (0.0, 0.0, 0.0)
        yaw = 0.0
        arm_pitch = 0.0
        mat = held_item_matrix(pos, yaw, arm_pitch, item_scale=0.75, flat=False)
        (verts, _), kind = held_item_mesh("stone", mat)
        self.assertEqual(kind, "block")
        v = verts.reshape(-1, 8)[:, :3]
        width = v[:, 0].max() - v[:, 0].min()
        # Scale 0.75 on a 0.5 unit cube yields ~0.375 units (significantly bigger than old 0.20 units)
        self.assertAlmostEqual(width, 0.375, delta=0.01)

    def test_first_person_flat_item_orientation(self):
        eye = (0.0, 1.6, 0.0)
        yaw = 0.0
        pitch = 0.0
        mat = first_person_held_matrix(eye, yaw, pitch, item_scale=0.68, flat=True)
        (verts, _), kind = held_item_mesh("wood_pickaxe", mat)
        self.assertEqual(kind, "item")
        v = verts.reshape(-1, 8)
        positions = v[:, :3]
        normals = v[:, 3:6]

        # Origin / handle is at bottom right
        handle_pos = mat @ [0.0, 0.0, 0.0, 1.0]
        # At yaw=0, pitch=0: camera right is +X, camera up is +Y, forward is -Z
        self.assertGreater(handle_pos[0], 0.5)
        self.assertLess(handle_pos[1], eye[1])

        # Pickaxe head extends steeply up (+Y) and slightly left (-X) towards screen center
        min_x = positions[:, 0].min()
        max_y = positions[:, 1].max()
        self.assertLess(min_x, handle_pos[0] - 0.15)
        self.assertGreater(max_y, handle_pos[1] + 0.5)

        # Front face is turned inward towards player (yaw ~38 deg), revealing 3D extruded edge
        front_normals = normals[normals[:, 2] > 0.5]
        self.assertGreater(len(front_normals), 0)
        self.assertLess(front_normals[:, 0].mean(), -0.4)
        self.assertGreater(front_normals[:, 2].mean(), 0.6)


class ViewModeTests(unittest.TestCase):
    def test_view_modes_cycle(self):
        mode = VIEW_FIRST
        mode = (mode + 1) % 3
        self.assertEqual(mode, VIEW_THIRD_BACK)
        mode = (mode + 1) % 3
        self.assertEqual(mode, VIEW_THIRD_FRONT)
        mode = (mode + 1) % 3
        self.assertEqual(mode, VIEW_FIRST)

    def test_head_and_hat_pitch_rotation(self):
        from client.player_render import third_person_pose
        from server.player_model import limb_pose

        world = World(seed=1, caves=[])
        player = Player(world)
        player.pitch = 0.5  # Looking up
        pose = third_person_pose(player, 0.0)
        self.assertAlmostEqual(pose["head"][0], 0.5)
        self.assertAlmostEqual(pose["hat"][0], 0.5)

        player.pitch = -0.7  # Looking down
        pose = third_person_pose(player, 0.0)
        self.assertAlmostEqual(pose["head"][0], -0.7)
        self.assertAlmostEqual(pose["hat"][0], -0.7)

    def test_walking_animation_arms_and_legs(self):
        from client.player_render import third_person_pose

        world = World(seed=1, caves=[])
        player = Player(world)
        player.vel = [3.0, 0.0, 0.0]  # Walking

        pose1 = third_person_pose(player, 0.1)
        pose2 = third_person_pose(player, 0.3)

        # Arms and legs swing in opposite directions
        self.assertAlmostEqual(pose1["arm_r"][0], -pose1["arm_l"][0])
        self.assertAlmostEqual(pose1["leg_r"][0], -pose1["leg_l"][0])
        self.assertAlmostEqual(pose1["arm_r"][0], -pose1["leg_r"][0])

        # Swing value changes across time (active animation)
        self.assertNotEqual(pose1["arm_r"][0], pose2["arm_r"][0])

        # When stopped, limbs are at rest (no walking in place)
        player.vel = [0.0, 0.0, 0.0]
        pose_stopped = third_person_pose(player, 0.5)
        self.assertEqual(pose_stopped["arm_r"][0], 0.0)
        self.assertEqual(pose_stopped["arm_l"][0], 0.0)
        self.assertEqual(pose_stopped["leg_r"][0], 0.0)
        self.assertEqual(pose_stopped["leg_l"][0], 0.0)

    def test_nametag_collection(self):
        from unittest.mock import MagicMock
        from client.app import Game
        from client.geom import look_at, perspective

        mock_game = MagicMock()
        mock_game.width = 1280
        mock_game.height = 720
        mock_game.camera = (0.0, 20.0, 0.0)

        remote_player = MagicMock()
        remote_player.username = "Alice"
        remote_player.pos = [0.0, 20.0, -5.0]
        mock_game.other_players = {"uuid-alice": remote_player}

        proj = perspective(75, 1280 / 720, 0.05, 220)
        view = look_at((0, 20, 0), (0, 20, -10))
        vp = proj @ view

        rects, labels = Game._collect_nametags(mock_game, vp, 1280, 720)
        self.assertEqual(len(rects), 1)
        self.assertEqual(len(labels), 1)

        label = labels[0]
        self.assertEqual(label.text, "Alice")
        self.assertEqual(label.anchor_x, "center")
        self.assertEqual(label.anchor_y, "center")
        # Target pos is (0, 22.25, -5). X is center of screen (640)
        self.assertAlmostEqual(label.x, 640.0, delta=1.0)
        # Y is above center of screen (> 360)
        self.assertGreater(label.y, 360.0)

        # Test line of sight dimming when obstructed by real world blocks
        real_world = World(seed=1, caves=[])
        real_world.load_chunk(0, 5, -1)
        mock_game.camera = (0.0, 80.0, 0.0)
        remote_player.pos = [0.0, 80.0, -5.0]
        # View matrix looking from (0, 80, 0) towards (0, 80, -10)
        view_sky = look_at((0, 80, 0), (0, 80, -10))
        vp_sky = proj @ view_sky
        mock_game.world = real_world

        # Unobstructed line of sight in sky
        rects_clear, labels_clear = Game._collect_nametags(mock_game, vp_sky, 1280, 720)
        self.assertEqual(len(labels_clear), 1)
        self.assertEqual(labels_clear[0].color[3], 255)
        self.assertEqual(rects_clear[0][4][3], 0.45)

        # Place a stone block directly in line of sight between camera (0, 80, 0) and (0, 82.25, -5)
        real_world.load_chunk(0, 5, 0)
        for y in (80, 81, 82):
            real_world.set_block(0, y, -2, "stone")
        rects_obstructed, labels_obstructed = Game._collect_nametags(mock_game, vp_sky, 1280, 720)
        self.assertEqual(len(labels_obstructed), 1)
        # Obstructed nametag has dimmed alpha (120 vs 255)
        self.assertEqual(labels_obstructed[0].color[3], 120)
        self.assertEqual(rects_obstructed[0][4][3], 0.2)

    def test_walk_speed_stationary_and_walls(self):
        from client.player_render import walk_speed

        world = World(seed=1, caves=[])
        player = Player(world)

        # Stationary local player
        player.vel = [0.0, 0.0, 0.0]
        self.assertEqual(walk_speed(player), 0.0)

        # Moving local player
        player.vel = [4.3, 0.0, 0.0]
        player.actual_speed = 4.3
        self.assertAlmostEqual(walk_speed(player), 4.3)

        # Local player running into wall (vel > 0 but actual_speed == 0)
        player.vel = [4.3, 0.0, 0.0]
        player.actual_speed = 0.0
        self.assertEqual(walk_speed(player), 0.0)

        # Remote player stationary
        remote = Player(world)
        remote.vel = [0.0, 0.0, 0.0]
        remote._move_speed = 0.0
        self.assertEqual(walk_speed(remote), 0.0)

        # Remote player with zero vel but stale move speed
        remote.vel = [0.0, 0.0, 0.0]
        remote._move_speed = 4.3
        self.assertEqual(walk_speed(remote), 0.0)

        # Remote player moving
        remote.vel = [3.0, 0.0, 0.0]
        remote._move_speed = 3.0
        self.assertAlmostEqual(walk_speed(remote), 3.0)

        # Remote player blocked (zero move_speed despite non-zero vel)
        remote.vel = [3.0, 0.0, 0.0]
        remote._move_speed = 0.0
        self.assertEqual(walk_speed(remote), 0.0)

    def test_on_deactivate_clears_keys(self):
        from unittest.mock import MagicMock
        from client.app import Game
        import pyglet.window.key as key

        game = MagicMock()
        game.keys = {key.W, key.A}
        game.player = Player(World(seed=1, caves=[]))
        game.player.input.forward = True
        game.player.input.left = True
        game.network_client = MagicMock()
        game.mode = "play"
        game.chat = MagicMock()
        game.chat.open = False
        game._apply_input = lambda: Game._apply_input(game)
        game._clear_input = lambda: Game._clear_input(game)

        # Calling on_deactivate should clear keys and player inputs
        Game.on_deactivate(game)
        self.assertEqual(len(game.keys), 0)
        self.assertFalse(game.player.input.forward)
        self.assertFalse(game.player.input.left)

    def test_remote_player_move_packet_and_timeout(self):
        import time
        from unittest.mock import MagicMock
        from client.app import Game

        game = MagicMock()
        world = World(seed=1, caves=[])
        game.world = world
        game.profile = {"uuid": "local-uuid", "skin": "default"}
        game.other_players = {}
        game.ready = False
        game.player = None
        game._net_timer = 0.0

        # Add remote player
        Game._add_remote_player(game, {
            "uuid": "remote-1",
            "pos": [0.0, 10.0, 0.0],
            "yaw": 0.0,
            "pitch": 0.0,
            "username": "Bob",
        })
        remote = game.other_players["remote-1"]
        self.assertEqual(remote._move_speed, 0.0)

        # Move packet with horizontal displacement
        game.network_client = MagicMock()
        game.network_client.poll_events.return_value = [{
            "type": "player_move",
            "uuid": "remote-1",
            "pos": [0.5, 10.0, 0.0],
            "vel": [4.0, 0.0, 0.0],
            "yaw": 0.0,
            "pitch": 0.0,
        }]
        # Set last move time slightly earlier so dt_move > 0
        remote._last_move_time = time.monotonic() - 0.05
        Game._network_tick(game, 0.05)
        self.assertGreater(remote._move_speed, 1.0)

        # Move packet where player has stopped (dist <= 0.02)
        game.network_client.poll_events.return_value = [{
            "type": "player_move",
            "uuid": "remote-1",
            "pos": [0.5, 10.0, 0.0],
            "vel": [0.0, 0.0, 0.0],
            "yaw": 0.0,
            "pitch": 0.0,
        }]
        Game._network_tick(game, 0.05)
        self.assertEqual(remote._move_speed, 0.0)

        # Timeout in tick resets move speed and velocity if packets stop
        remote._move_speed = 3.0
        remote.vel = [3.0, 0.0, 0.0]
        remote._last_move_time = time.monotonic() - 0.2  # 200ms ago (> 0.1s)
        game._network_tick = MagicMock()
        game._play_time = 0.0
        game.mode = "play"
        game.ready = False
        Game.tick(game, 0.016)
        self.assertEqual(remote._move_speed, 0.0)
        self.assertEqual(remote.vel, [0.0, 0.0, 0.0])
