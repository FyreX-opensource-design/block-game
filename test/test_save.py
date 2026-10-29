import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from client.menu import MenuState
from server.blocks import AIR, STONE
from server.noise import terrain_height
from server.storage import WorldCatalog, default_saves_dir
from server.world import World


class SaveTests(unittest.TestCase):
    def test_windows_saves_use_local_appdata(self):
        local = Path("/tmp/winhome/AppData/Local")
        with mock.patch("server.storage.sys.platform", "win32"):
            with mock.patch.dict(os.environ, {"LOCALAPPDATA": str(local)}):
                self.assertEqual(default_saves_dir(), local / "block-game" / "worlds")
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("server.storage.Path.home", return_value=Path("/home/ada")):
                    self.assertEqual(
                        default_saves_dir(),
                        Path("/home/ada/AppData/Local/block-game/worlds"),
                    )

    def test_linux_saves_stay_in_xdg_data_home(self):
        with mock.patch("server.storage.sys.platform", "linux"):
            custom = Path("/tmp/xdg-data")
            with mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(custom)}):
                self.assertEqual(default_saves_dir(), custom / "block-game" / "worlds")

    def test_edited_chunks_survive_unload_and_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = WorldCatalog(folder)
            info = catalog.create("Hills", 4)
            store = catalog.store_for(info.id)
            world = World(seed=info.seed, store=store)
            world.generate_chunk(0, 7, 0)
            self.assertEqual(world.get_block(3, 120, 5), AIR)
            self.assertTrue(world.set_block(3, 120, 5, STONE))
            world.evict_chunk((0, 7, 0))
            self.assertTrue(store.path_for(0, 7, 0).is_file())

            restored = World(seed=info.seed, store=store)
            restored.load_chunk(0, 7, 0)
            self.assertEqual(restored.get_block(3, 120, 5), STONE)

    def test_unedited_chunks_are_not_written(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = WorldCatalog(folder)
            info = catalog.create("Plain", 1)
            store = catalog.store_for(info.id)
            world = World(seed=info.seed, store=store)
            world.generate_chunk(0, 0, 0)
            world.evict_chunk((0, 0, 0))
            self.assertFalse(store.path_for(0, 0, 0).exists())
            again = World(seed=info.seed, store=store)
            again.load_chunk(0, 0, 0)
            self.assertNotEqual(again.get_block(1, 0, 1), AIR)

    def test_world_list_create_and_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = WorldCatalog(folder)
            first = catalog.create("Alpha", 3)
            second = catalog.create("Alpha", 9)
            self.assertNotEqual(first.id, second.id)
            second.last_played = first.last_played + 10
            catalog.write(second)
            listed = catalog.list_worlds()
            self.assertEqual([item.id for item in listed], [second.id, first.id])
            catalog.delete(first.id)
            self.assertEqual([item.id for item in catalog.list_worlds()], [second.id])
            self.assertFalse((Path(folder) / first.id).exists())

    def test_player_position_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = WorldCatalog(folder)
            info = catalog.create("Camp", 12)
            info.player = {"x": 4.5, "y": 22.0, "z": -3.5, "yaw": 1.2, "pitch": -0.2, "slot": 3}
            catalog.write(info)
            loaded = catalog.read(info.id)
            self.assertEqual(loaded.player["x"], 4.5)
            self.assertEqual(loaded.player["slot"], 3)
            self.assertEqual(loaded.seed, 12)

    def test_different_seeds_change_the_terrain(self):
        differed = False
        for x in range(30):
            for z in range(30):
                if terrain_height(x, z, 1) != terrain_height(x, z, 2):
                    differed = True
                    break
            if differed:
                break
        self.assertTrue(differed)


class MenuTests(unittest.TestCase):
    def test_title_singleplayer_button_is_clickable(self):
        menu = MenuState()
        frame = menu.layout(1280, 720)
        hit = next(item for item in frame.hits if item.action == "singleplayer")
        self.assertEqual(menu.hit_test(frame, hit.x + 4, hit.y + 4), "singleplayer")
        self.assertIsNone(menu.hit_test(frame, 2, 2))

    def test_world_row_selects_and_double_click_plays(self):
        menu = MenuState()
        menu.screen = "worlds"
        menu.worlds = [_FakeWorld("One", 1), _FakeWorld("Two", 2)]
        menu.selected = 0
        frame = menu.layout(1280, 720)
        row = next(item for item in frame.hits if item.action == "row:1")
        self.assertIsNone(menu.apply(menu.hit_test(frame, row.x + 3, row.y + 3)))
        self.assertEqual(menu.selected, 1)
        self.assertEqual(menu.apply("row:1"), "play")

    def test_delete_is_disabled_without_a_world(self):
        menu = MenuState()
        menu.screen = "worlds"
        frame = menu.layout(1280, 720)
        delete = next(item for item in frame.hits if item.action == "delete")
        self.assertFalse(delete.enabled)
        self.assertIsNone(menu.hit_test(frame, delete.x + 2, delete.y + 2))


class _FakeWorld:
    def __init__(self, name, seed):
        self.name = name
        self.seed = seed
        self.last_played = 0


if __name__ == "__main__":
    unittest.main()
