import time
import unittest

import numpy as np

from server.biomes import BIOMES, all_biomes
from server.blocks import WORLD_HEIGHT
from server.lighting import WORLD_HEIGHT as LIGHT_HEIGHT
from server.weather import CLOUD_Y, SlaveClock, WeatherField, ptp_offset, rain_at, rainfall_of
from server.network import MultiplayerServer
from server.weather_link import WeatherPort
from server.weather_server import WeatherClient
from server.world import World


class ClockTests(unittest.TestCase):
    def test_ptp_offset_matches_the_four_timestamps(self):
        # Slave is 0.1s ahead of the master, and the path delay is 0.2s.
        offset, delay = ptp_offset(t1=100.0, t2=100.3, t3=100.4, t4=100.5)
        self.assertAlmostEqual(offset, 0.1)
        self.assertAlmostEqual(delay, 0.2)
        clock = SlaveClock()
        clock.observe(100.0, 100.3, 100.4, 100.5)
        self.assertTrue(clock.synced)
        self.assertAlmostEqual(clock.master_now(100.4), 100.3)


class CloudTests(unittest.TestCase):
    def test_world_is_tall_enough_for_the_cloud_deck(self):
        self.assertEqual(WORLD_HEIGHT, 256)
        self.assertEqual(LIGHT_HEIGHT, WORLD_HEIGHT)
        self.assertGreater(CLOUD_Y, 128)
        self.assertLess(CLOUD_Y, WORLD_HEIGHT)

    def test_a_column_in_the_clouds_blocks_rain(self):
        field = WeatherField(seed=3)
        field.wind[:] = (8.0, 0.0)
        field.density[:] = 0.8
        heights = np.full((field.size, field.size), 80.0)
        heights[:, 20] = CLOUD_Y + 4
        rainfall = np.ones_like(heights)
        field.step(0.2, heights, rainfall)
        self.assertTrue(np.all(field.rain[:, 20] == 0.0))
        self.assertTrue(np.all(field.density[:, 20] == 0.0))

    def test_windward_rain_exceeds_the_lee_of_a_ridge(self):
        field = WeatherField(seed=5)
        field.wind[:] = (8.0, 0.0)
        field.density[:] = 0.7
        heights = np.full((field.size, field.size), 70.0)
        for column in range(8, 16):
            heights[:, column] = 70.0 + (column - 8) * 16.0
        heights[:, 16:20] = CLOUD_Y + 8
        for column in range(20, 28):
            heights[:, column] = (CLOUD_Y + 8) - (column - 19) * 16.0
        rainfall = np.ones_like(heights)
        for _ in range(30):
            field.step(0.2, heights, rainfall)
        windward = float(field.rain[:, 8:16].mean())
        leeward = float(field.rain[:, 20:28].mean())
        self.assertGreater(windward, leeward)
        self.assertTrue(np.all(field.rain[:, 16:20] == 0.0))

    def test_a_tall_building_casts_a_rain_shadow(self):
        field = WeatherField(seed=8)
        field.wind[:] = (8.0, 0.0)
        field.density[:] = 0.85
        heights = np.full((field.size, field.size), 70.0)
        heights[20, 20] = 170.0
        rainfall = np.ones_like(heights)
        for _ in range(20):
            field.step(0.2, heights, rainfall)
        self.assertGreater(float(field.rain[20, 19]), float(field.rain[20, 21]))

    def test_desert_rain_is_lighter_than_forest_rain(self):
        heights = np.full((40, 40), 80.0)

        def total(rainfall):
            field = WeatherField(seed=11)
            field.wind[:] = (0.0, 0.0)
            field.density[:] = 0.9
            wet = np.full_like(heights, rainfall)
            amount = 0.0
            for _ in range(8):
                field.step(0.2, heights, wet)
                amount += float(field.rain.sum())
            return amount

        self.assertLess(total(0.12), total(1.0) * 0.5)

    def test_rain_stays_under_the_cloud_that_made_it(self):
        field = WeatherField(seed=13)
        field.wind[:] = (0.0, 0.0)
        field.density[:] = 0.0
        field.density[18:23, 18:23] = 1.2
        heights = np.full((field.size, field.size), 75.0)
        rainfall = np.ones_like(heights)
        field.step(0.05, heights, rainfall)
        self.assertGreater(float(field.rain[20, 20]), 0.2)
        self.assertEqual(float(field.rain[0, 0]), 0.0)
        self.assertEqual(float(field.rain[39, 5]), 0.0)

    def test_flat_country_keeps_a_few_storms_and_mostly_fair_sky(self):
        field = WeatherField(seed=4)
        field.wind[:] = (4.0, 0.0)
        heights = np.full((field.size, field.size), 70.0)
        rainfall = np.ones_like(heights)
        for _ in range(40):
            field.step(0.2, heights, rainfall)
        share = float((field.rain > 0.08).mean())
        self.assertGreater(share, 0.05)
        self.assertLess(share, 0.45)

    def test_dry_biomes_carry_less_rainfall(self):
        catalog = {biome.id: biome for biome in all_biomes(BIOMES)}
        self.assertLess(rainfall_of(catalog["desert"]), 0.3)
        self.assertLess(rainfall_of(catalog["savanna"]), 0.6)
        self.assertGreater(rainfall_of(catalog["ocean"]), 1.0)
        self.assertEqual(rainfall_of(catalog["plains"]), 1.0)

    def test_sliding_the_weather_grid_matches_a_fresh_sample(self):
        from server.biomes import biome_at

        field = WeatherField(seed=496724168)
        field.recenter(-411.0, -289.0)
        field.maps([])
        field.recenter(-411.0 + field.cell, -289.0 + field.cell)
        heights, rain = field.maps([])
        fresh = WeatherField(seed=496724168)
        fresh.origin = field.origin
        fresh_heights, fresh_rain = fresh.maps([])
        self.assertTrue(np.allclose(heights, fresh_heights))
        self.assertTrue(np.allclose(rain, fresh_rain))
        ox, oz = field.origin
        for i in range(0, field.size, 13):
            for j in range(0, field.size, 13):
                x = int(ox + (j + 0.5) * field.cell)
                z = int(oz + (i + 0.5) * field.cell)
                self.assertAlmostEqual(rain[i, j], rainfall_of(biome_at(x, z, field.seed, BIOMES)))


class WeatherLinkTests(unittest.TestCase):
    def test_weather_server_follows_the_world_clock(self):
        world = World(seed=4, caves=[])
        port = WeatherPort(world, "127.0.0.1", 0)
        client = None
        try:
            client = WeatherClient("127.0.0.1", port.port)
            deadline = time.time() + 8.0
            while time.time() < deadline and world.weather is None:
                port.poll(0.05, (0.0, 0.0))
                client.pump()
                time.sleep(0.02)
            self.assertIsNotNone(world.weather)
            self.assertTrue(client.clock.synced)
            self.assertLess(abs(client.clock.offset), 0.25)
            self.assertEqual(int(world.weather["cloud_y"]), CLOUD_Y)
            self.assertGreater(rain_at(world.weather, 0.0, 0.0), -0.01)
        finally:
            if client is not None:
                client.close()
            port.close()

    def test_game_server_accepts_weather_on_its_own_port(self):
        world = World(seed=2, caves=[])
        server = MultiplayerServer(host="127.0.0.1", port=0, world=world)
        server.start()
        client = None
        try:
            client = WeatherClient("127.0.0.1", server.actual_port)
            deadline = time.time() + 8.0
            while time.time() < deadline and world.weather is None:
                client.pump()
                time.sleep(0.02)
            self.assertIsNotNone(world.weather)
            self.assertTrue(client.clock.synced)
            self.assertLess(abs(client.clock.offset), 0.25)
        finally:
            if client is not None:
                client.close()
            server.stop()


class CloudLookTests(unittest.TestCase):
    def test_a_full_deck_still_leaves_clear_sky(self):
        from client.render import cloud_cover

        covers = [cloud_cover(x, z) for x in range(-240, 240, 8) for z in range(-240, 240, 8)]
        cloudy = sum(1 for value in covers if value >= 0.56)
        self.assertGreater(cloudy, len(covers) * 0.15)
        self.assertLess(cloudy, len(covers) * 0.55)

    def test_sun_and_moon_sit_behind_the_clouds(self):
        from client.render import CELESTIAL_DISTANCE, CELESTIAL_FRAG, _celestial_quad

        self.assertGreater(CELESTIAL_DISTANCE, CLOUD_Y)
        self.assertIn("discard", CELESTIAL_FRAG)
        quad = _celestial_quad((10.0, 80.0, -4.0), (0.0, 1.0, 0.0), 14.0)
        center = quad.reshape(-1, 5)[:, :3].mean(axis=0)
        dist = float(np.linalg.norm(center - np.array([10.0, 80.0, -4.0])))
        self.assertGreater(dist, CLOUD_Y)

    def test_rain_falls_only_under_a_cloud_and_stays_cheap(self):
        import time

        from client.render import _RAIN_CAP, _cloud_vertices, _rain_vertices

        class _OpenSky:
            def column_top(self, x, z):
                return 64

        # A dry puff overhead, and a wet one too far away to reach the player.
        gap = {"cloud_y": CLOUD_Y, "_cloud_ready": True, "_puffs": np.array([[0.0, 0.0, 5.0, 0.0], [40.0, 0.0, 6.0, 0.9]], dtype=np.float32)}
        self.assertIsNone(_rain_vertices(_OpenSky(), gap, (0.0, 80.0, 0.0), 1.0))

        wet = {"cloud_y": CLOUD_Y, "_cloud_ready": True, "_puffs": np.array([[4.0, 1.0, 5.0, 0.8]], dtype=np.float32)}
        streaks = _rain_vertices(_OpenSky(), wet, (4.0, 80.0, 1.0), 1.0)
        self.assertIsNotNone(streaks)
        centers = streaks.reshape(-1, 6, 9).mean(axis=1)
        self.assertLessEqual(len(centers), _RAIN_CAP * 2)
        self.assertTrue(np.all(np.hypot(centers[:, 0] - 4.0, centers[:, 2] - 1.0) <= 5.0))
        indoors = _rain_vertices(type("Roof", (), {"column_top": lambda self, x, z: 90})(), wet, (4.0, 80.0, 1.0), 1.0)
        self.assertIsNone(indoors)

        def _mean_height(now):
            falling = _rain_vertices(_OpenSky(), wet, (4.0, 80.0, 1.0), now)
            return float(falling.reshape(-1, 6, 9)[:, :, 1].mean())

        self.assertLess(_mean_height(1.05), _mean_height(1.0))

        size = 40
        from server.weather import pack_grid

        broad = {
            "origin": [-160.0, -160.0],
            "cell": 8,
            "size": size,
            "cloud_y": CLOUD_Y,
            "wind": [4.0, 1.0],
            "density": pack_grid(np.full((size, size), 0.8)),
            "rain": pack_grid(np.full((size, size), 0.7)),
        }
        started = time.perf_counter()
        _cloud_vertices(broad, 0.0, 0.0)
        self.assertLess(time.perf_counter() - started, 0.02)
        started = time.perf_counter()
        _cloud_vertices(broad, 1.0, 0.0)
        self.assertLess(time.perf_counter() - started, 0.001)
        wet_puffs = broad["_puffs"]
        wet_puffs = wet_puffs[wet_puffs[:, 3] >= 0.08]
        self.assertGreater(len(wet_puffs), 0)
        spot = wet_puffs[len(wet_puffs) // 2]
        started = time.perf_counter()
        streaks = None
        for step in range(20):
            streaks = _rain_vertices(_OpenSky(), broad, (float(spot[0]), 80.0, float(spot[1])), step * 0.016)
        self.assertLess((time.perf_counter() - started) / 20.0, 0.002)
        self.assertIsNotNone(streaks)
        centers = streaks.reshape(-1, 6, 9).mean(axis=1)
        self.assertLessEqual(len(centers), _RAIN_CAP * 2)
        dx = centers[:, None, 0] - wet_puffs[None, :, 0]
        dz = centers[:, None, 2] - wet_puffs[None, :, 1]
        inside = np.hypot(dx, dz) <= wet_puffs[None, :, 2] + 0.05
        self.assertTrue(np.all(inside.any(axis=1)))


if __name__ == "__main__":
    unittest.main()
