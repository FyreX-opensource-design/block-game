import tempfile
import unittest

from pyglet.window import key

from client.app import Game
from client.menu import MenuState
from client.settings import load_settings, save_settings


class DummyChat:
    open = False


class DummyPlayer:
    def __init__(self):
        self.yaw = 0.0
        self.pitch = 0.0


class DummyApp:
    def __init__(self, enabled=True):
        self.arrow_key_camera = enabled
        self.player = DummyPlayer()
        self.chat = DummyChat()
        self.keys = set()


class ArrowCameraTest(unittest.TestCase):
    def _apply(self, app, keys, dt=0.5):
        app.keys = set(keys)
        Game._apply_arrow_look(app, dt)
        return app.player.yaw, app.player.pitch

    def test_disabled_does_not_rotate(self):
        app = DummyApp(enabled=False)
        yaw, pitch = self._apply(app, [key.LEFT, key.UP])
        self.assertEqual((yaw, pitch), (0.0, 0.0))

    def test_arrow_keys_rotate_in_expected_directions(self):
        app = DummyApp()
        yaw, _ = self._apply(app, [key.LEFT])
        self.assertGreater(yaw, 0.0)

        app.player.yaw = 0.0
        yaw, _ = self._apply(app, [key.RIGHT])
        self.assertLess(yaw, 0.0)

        app.player.pitch = 0.0
        _, pitch = self._apply(app, [key.UP])
        self.assertGreater(pitch, 0.0)

        app.player.pitch = 0.0
        _, pitch = self._apply(app, [key.DOWN])
        self.assertLess(pitch, 0.0)

    def test_pitch_is_clamped(self):
        app = DummyApp()
        _, pitch = self._apply(app, [key.UP], dt=100.0)
        self.assertLessEqual(pitch, 1.45)
        _, pitch = self._apply(app, [key.DOWN], dt=100.0)
        self.assertGreaterEqual(pitch, -1.45)

    def test_chat_input_is_not_affected(self):
        app = DummyApp()
        app.chat.open = True
        yaw, pitch = self._apply(app, [key.LEFT, key.UP])
        self.assertEqual((yaw, pitch), (0.0, 0.0))

    def test_setting_persists(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = load_settings(folder)
            self.assertFalse(settings["arrow_key_camera"])

            settings["arrow_key_camera"] = True
            save_settings(settings, folder)
            self.assertTrue(load_settings(folder)["arrow_key_camera"])

    def test_alternate_setting_names_are_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = load_settings(folder)
            settings["arrow_keys_look"] = True
            save_settings(settings, folder)

            loaded = load_settings(folder)
            self.assertTrue(loaded["arrow_key_camera"])
            self.assertTrue(loaded["arrow_keys_look"])

    def test_options_menu_has_toggle(self):
        menu = MenuState()
        menu.apply("options")
        frame = menu.layout(800, 600)
        labels = [label.text for label in frame.labels]
        self.assertIn("Arrow Key Camera: OFF", labels)
        self.assertIn("toggle-arrow-camera", [hit.action for hit in frame.hits])

        self.assertEqual(menu.apply("toggle-arrow-camera"), "toggle-arrow-camera")
        self.assertTrue(menu.arrow_key_camera)
        frame = menu.layout(800, 600)
        self.assertIn("Arrow Key Camera: ON", [label.text for label in frame.labels])


if __name__ == "__main__":
    unittest.main()
