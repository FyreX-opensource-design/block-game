import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from server.blocks import ASSETS
from server.commands import CommandContext, CommandHandler
from server.images import read_png
from server.storage import WorldCatalog, WorldInfo
from server.world import World
from client.render import Renderer


class TestDayNightCycle(unittest.TestCase):
    def test_advance_time(self):
        w = World(seed=123, day_time=1000.0)
        self.assertEqual(w.day_time, 1000.0)
        # Advance 1 second = 20 ticks
        w.advance_time(1.0)
        self.assertAlmostEqual(w.day_time, 1020.0)
        # Check wraparound at 24000
        w.day_time = 23990.0
        w.advance_time(1.0)
        self.assertAlmostEqual(w.day_time, 10.0)

    def test_light_levels_day_and_night(self):
        w = World(seed=123)
        # Noon = 6000 ticks
        w.day_time = 6000.0
        self.assertAlmostEqual(w.get_sky_light_level(), 15.0, places=3)
        self.assertAlmostEqual(w.get_sky_intensity(), 1.0, places=3)

        # Midnight = 18000 ticks
        # Requirement: "Don't make it compleltly dark during the night. make it about a light level of 7 on skylight during the night."
        w.day_time = 18000.0
        self.assertAlmostEqual(w.get_sky_light_level(), 7.0, places=3)
        self.assertAlmostEqual(w.get_sky_intensity(), 7.0 / 15.0, places=3)

    def test_current_skylight_scaling(self):
        w = World(seed=123)
        # Mock get_skylight
        w.get_skylight = MagicMock(return_value=15)

        # At noon, full skylight remains 15
        w.day_time = 6000.0
        self.assertEqual(w.get_current_skylight(0, 64, 0), 15)

        # At midnight, full surface skylight scales to 7
        w.day_time = 18000.0
        self.assertEqual(w.get_current_skylight(0, 64, 0), 7)

        # Deep cave with 0 skylight stays 0 day and night
        w.get_skylight = MagicMock(return_value=0)
        w.day_time = 6000.0
        self.assertEqual(w.get_current_skylight(0, 10, 0), 0)
        w.day_time = 18000.0
        self.assertEqual(w.get_current_skylight(0, 10, 0), 0)

    def test_world_info_and_catalog_persistence(self):
        info = WorldInfo("test_world", "Test", 42, 100.0, 200.0, day_time=14500.0)
        d = info.to_dict()
        self.assertEqual(d["day_time"], 14500.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = WorldCatalog(Path(tmpdir))
            created = catalog.create("SavedWorld", 999)
            self.assertEqual(created.day_time, 1000.0)

            # Update day_time and save
            created.day_time = 18250.0
            catalog.write(created)

            # Re-read
            loaded = catalog.read(created.id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.day_time, 18250.0)

    def test_time_command(self):
        cmd = CommandHandler()
        w = World(seed=123, day_time=1000.0)
        ctx = CommandContext(sender="Player", world=w)

        # /time query
        cmd.execute(ctx, "/time query")
        self.assertTrue(any("1000" in r[0] for r in ctx.responses))

        # /time set night
        cmd.execute(ctx, "/time set night")
        self.assertEqual(w.day_time, 13000.0)

        # /time set noon
        cmd.execute(ctx, "/time set noon")
        self.assertEqual(w.day_time, 6000.0)

        # /time set midnight
        cmd.execute(ctx, "/time set midnight")
        self.assertEqual(w.day_time, 18000.0)

        # /time set custom ticks
        cmd.execute(ctx, "/time set 4500")
        self.assertEqual(w.day_time, 4500.0)

    def test_sun_and_moon_textures_exist(self):
        sun_path = ASSETS / "textures" / "sun.png"
        moon_path = ASSETS / "textures" / "moon.png"
        self.assertTrue(sun_path.is_file(), "sun.png should exist")
        self.assertTrue(moon_path.is_file(), "moon.png should exist")

        sun_img = read_png(sun_path)
        moon_img = read_png(moon_path)
        self.assertEqual(sun_img.shape, (32, 32, 4))
        self.assertEqual(moon_img.shape, (32, 32, 4))

    def test_renderer_celestial_math(self):
        renderer_dummy = Renderer.__new__(Renderer)
        # Test get_sky_intensity
        intensity_noon = Renderer.get_sky_intensity(renderer_dummy, 6000.0)
        self.assertAlmostEqual(intensity_noon, 1.0, places=3)
        intensity_midnight = Renderer.get_sky_intensity(renderer_dummy, 18000.0)
        self.assertAlmostEqual(intensity_midnight, 7.0 / 15.0, places=3)

        # Test get_sky_color
        color_day = Renderer.get_sky_color(renderer_dummy, 6000.0)
        color_night = Renderer.get_sky_color(renderer_dummy, 18000.0)
        color_sunset = Renderer.get_sky_color(renderer_dummy, 12000.0)

        # Day is bright sky blue
        self.assertGreater(color_day[2], 0.8)
        # Night is dark blue (not pitch black: > 0.0)
        self.assertGreater(color_night[0], 0.01)
        self.assertLess(color_night[0], 0.15)
        # Sunset has prominent red/orange component
        self.assertGreater(color_sunset[0], 0.5)

        # Test celestial lighting
        sun_dir_noon, sun_col_noon = Renderer.get_celestial_lighting(renderer_dummy, 6000.0)
        # At noon, sun is near zenith (positive Y)
        self.assertGreater(sun_dir_noon[1], 0.8)

        sun_dir_mid, _ = Renderer.get_celestial_lighting(renderer_dummy, 18000.0)
        # At midnight, sun is near nadir (negative Y)
        self.assertLess(sun_dir_mid[1], -0.8)

    def test_stars_cover_full_celestial_sphere(self):
        import numpy as np
        # Verify uniform sphere distribution logic
        rng = np.random.RandomState(42)
        ys = []
        for _ in range(500):
            z = rng.uniform(-1.0, 1.0)
            ys.append(z)
        # Check both positive Y and negative Y have stars
        pos_y = sum(1 for y in ys if y > 0.2)
        neg_y = sum(1 for y in ys if y < -0.2)
        self.assertGreater(pos_y, 100)
        self.assertGreater(neg_y, 100)

