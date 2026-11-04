import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure pyglet doesn't attempt to open a display in headless test environments
import pyglet
pyglet.options["shadow_window"] = False

from server.blocks import BLOCKS, Block
from server.sound import (
    DEFAULT_SOUND_SETS,
    SOUND_SETS,
    get_block_break_sound,
    get_block_breaking_sound,
    get_block_sound_group,
    get_block_walk_sound,
    infer_sound_category,
    load_sound_sets,
)
from client.sound import SoundManager


class TestSoundSystem(unittest.TestCase):
    def test_sound_sets_contain_walk_and_breaking(self):
        for cat in ("grass", "dirt", "wood"):
            self.assertIn(cat, SOUND_SETS)
            self.assertIn("walk", SOUND_SETS[cat])
            self.assertIn("breaking", SOUND_SETS[cat])
            self.assertTrue(len(SOUND_SETS[cat]["walk"]) > 0)
            self.assertTrue(len(SOUND_SETS[cat]["breaking"]) > 0)

    def test_asset_definable_sound_sets(self):
        # Verify that assets/sound_sets/*.json are loaded into SOUND_SETS
        from server.blocks import ASSETS
        sound_sets_dir = ASSETS / "sound_sets"
        self.assertTrue(sound_sets_dir.is_dir())
        self.assertTrue((sound_sets_dir / "grass.json").is_file())
        self.assertTrue((sound_sets_dir / "wood.json").is_file())

        # Test loading custom sound set
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            sets_dir = tmp_path / "sound_sets"
            sets_dir.mkdir()
            custom_file = sets_dir / "custom.json"
            custom_file.write_text(json.dumps({
                "id": "crystal",
                "walk": ["crystal_step_1"],
                "breaking": ["crystal_breaking_hit"],
                "break": ["crystal_shatter"]
            }))
            with patch("server.sound.ASSETS", tmp_path):
                sets = load_sound_sets()
                self.assertIn("crystal", sets)
                self.assertEqual(sets["crystal"]["walk"], ("crystal_step_1",))
                self.assertEqual(sets["crystal"]["breaking"], ("crystal_breaking_hit",))
                self.assertEqual(sets["crystal"]["break"], ("crystal_shatter",))

        # Restore original sound sets
        load_sound_sets()

    def test_sound_category_inference(self):
        self.assertEqual(infer_sound_category(None, "grass"), "grass")
        self.assertEqual(infer_sound_category(None, "tall_grass"), "grass")
        self.assertEqual(infer_sound_category(None, "dirt"), "dirt")
        self.assertEqual(infer_sound_category(None, "sand"), "dirt")
        self.assertEqual(infer_sound_category(None, "planks"), "wood")
        self.assertEqual(infer_sound_category(None, "oak_log"), "wood")

    def test_get_block_sound_group_and_breaking_sound(self):
        # Grass block
        grass_group = get_block_sound_group("grass")
        self.assertIsNotNone(grass_group)
        self.assertIn("grass_walking_1", grass_group["walk"])
        self.assertIn("soil_breaking", grass_group["breaking"])
        self.assertEqual(get_block_breaking_sound("grass"), "soil_breaking")

        # Dirt block
        dirt_group = get_block_sound_group("dirt")
        self.assertIsNotNone(dirt_group)
        self.assertEqual(get_block_breaking_sound("dirt"), "soil_breaking")

        # Planks block
        wood_group = get_block_sound_group("planks")
        self.assertIsNotNone(wood_group)
        self.assertEqual(get_block_breaking_sound("planks"), "wood_breaking")

        # Air returns None
        self.assertIsNone(get_block_sound_group("air"))
        self.assertIsNone(get_block_sound_group(None))
        self.assertIsNone(get_block_breaking_sound("air"))

    def test_block_object_sound_methods(self):
        grass_block = BLOCKS.get("grass")
        self.assertIsNotNone(grass_block)
        walk_sound = grass_block.get_walk_sound()
        self.assertIn(walk_sound, ("grass_walking_1", "grass_walking_2"))
        breaking_sound = grass_block.get_breaking_sound()
        self.assertEqual(breaking_sound, "soil_breaking")
        # Final break sound is None unless explicit break sound is set
        self.assertIsNone(grass_block.get_break_sound())

        planks_block = BLOCKS.get("planks")
        self.assertIsNotNone(planks_block)
        walk_sound = planks_block.get_walk_sound()
        self.assertIn(walk_sound, ("walking_wood_1", "wood_walking_2"))
        breaking_sound = planks_block.get_breaking_sound()
        self.assertEqual(breaking_sound, "wood_breaking")
        self.assertIsNone(planks_block.get_break_sound())

    def test_sound_manager_playback(self):
        sm = SoundManager()
        mock_source = MagicMock()
        sm._sources["soil_breaking"] = mock_source
        sm._sources["grass_walking_1"] = mock_source

        # Test starting breaking sound once with looping
        with patch("pyglet.media.Player") as mock_player_cls:
            mock_player = MagicMock()
            mock_player_cls.return_value = mock_player
            player = sm.start_breaking_sound("grass", loop=True)
            self.assertEqual(player, mock_player)
            self.assertTrue(mock_player.loop)
            mock_player.play.assert_called_once()

            # Second call for the same active sound should NOT restart
            mock_player.playing = True
            player2 = sm.start_breaking_sound("grass", loop=True)
            self.assertEqual(player2, mock_player)
            mock_player.play.assert_called_once()

            # Stopping breaking sound pauses and deletes player midway
            sm.stop_breaking_sound()
            mock_player.pause.assert_called_once()
            mock_player.delete.assert_called_once()
            self.assertIsNone(sm._breaking_player)

        # Test playing walk sound
        mock_walk_player = MagicMock()
        mock_source.play.return_value = mock_walk_player
        walk_player = sm.play_walk_sound("grass")
        self.assertIsNotNone(walk_player)

    def test_breaking_sound_in_survival(self):
        from client.app import Game
        game = Game.__new__(Game)
        game.sound_manager = MagicMock()
        game.world = MagicMock()
        game.world.get_block.return_value = "dirt"
        game.world.get_sub_cell.return_value = None
        game.gamemode = "survival"
        game.mode = "play"
        game._mouse_held_left = True
        game._aim = MagicMock()
        game.target = {"x": 5, "y": 60, "z": 5}
        game.hotbar = []
        game.selected = 0
        game._breaking_block = None
        game._breaking_target_block = None
        game._break_time = 0.0
        game._break_required = 0.0
        game._break_progress = 0.0

        # Start breaking a block with required time > 0
        with patch("client.app.calculate_break_time", return_value=1.0):
            game._start_breaking(game.target)

        # Breaking sound should start playing once with looping
        game.sound_manager.start_breaking_sound.assert_called_once_with(BLOCKS.get("dirt"), loop=True)

        # Advance breaking time halfway (0.5s of 1.0s)
        game._break_time += 0.5
        game._break_progress = game._break_time / game._break_required
        # Sound manager should not have been called again during progress
        self.assertEqual(game.sound_manager.start_breaking_sound.call_count, 1)

        # When block is broken midway or finished, _reset_breaking stops the sound midway
        game._reset_breaking()
        game.sound_manager.stop_breaking_sound.assert_called_once()

    def test_walk_sound_distance_accumulation(self):
        from client.app import Game
        game = Game.__new__(Game)
        game.sound_manager = MagicMock()
        game._walk_sound_distance = 0.0

        mock_player = MagicMock()
        mock_player.pos = [10.0, 65.0, 10.0]
        mock_player.on_ground = True
        mock_player.in_water = MagicMock(return_value=False)
        mock_player.input = MagicMock(sprint=False)
        game.player = mock_player
        game._get_block_under_player = MagicMock(return_value="dirt")

        # Small movement should not trigger walk sound yet (threshold 2.2)
        prev_pos = [10.0, 65.0, 9.5]  # moved 0.5
        game._update_walk_sound(prev_pos)
        game.sound_manager.play_walk_sound.assert_not_called()
        self.assertAlmostEqual(game._walk_sound_distance, 0.5)

        # Move additional 1.8 blocks -> total 2.3 >= 2.2 -> triggers walk sound
        prev_pos = [10.0, 65.0, 10.0]
        game.player.pos = [10.0, 65.0, 11.8]
        game._update_walk_sound(prev_pos)
        game.sound_manager.play_walk_sound.assert_called_once_with("dirt")
        self.assertEqual(game._walk_sound_distance, 0.0)


if __name__ == "__main__":
    unittest.main()
