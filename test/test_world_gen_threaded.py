import time
import unittest

from server.blocks import CHUNK_SIZE, LOAD_RADIUS, UNLOAD_RADIUS
from server.world import World


class TestWorldGenThreaded(unittest.TestCase):
    def test_generate_chunk_remember_flag(self):
        world = World(seed=42)
        chunk = world.generate_chunk(10, 4, 10, remember=False)
        self.assertIsNotNone(chunk)
        self.assertEqual((chunk.cx, chunk.cy, chunk.cz), (10, 4, 10))
        self.assertNotIn((10, 4, 10), world.chunks)

        world.commit_chunk(chunk)
        self.assertIn((10, 4, 10), world.chunks)
        self.assertIn((10, 4, 10), world.dirty)

    def test_missing_chunks_ordering(self):
        world = World(seed=42)
        # Missing chunks around (0, 4, 0)
        missing = world.missing_chunks([(0, 4, 0)], limit=5)
        self.assertEqual(len(missing), 5)
        # Closest horizontal distance should be first (dx=0, dz=0)
        first = missing[0]
        self.assertEqual((first[0], first[2]), (0, 0))

    def test_evict_distant(self):
        world = World(seed=42)
        world.load_chunk(0, 4, 0)
        world.load_chunk(20, 4, 20)
        self.assertIn((0, 4, 0), world.chunks)
        self.assertIn((20, 4, 20), world.chunks)

        # Evict distant from center (0, 4, 0)
        removed = world.evict_distant([(0, 4, 0)])
        self.assertIn((20, 4, 20), removed)
        self.assertNotIn((20, 4, 20), world.chunks)
        self.assertIn((0, 4, 0), world.chunks)

    def test_background_generator_async_completion(self):
        world = World(seed=999)
        try:
            world.start_background_generator(max_workers=2, use_processes=True)
            center = (5, 4, 5)
            requested = world.request_chunks([center], max_in_flight=4)
            self.assertEqual(len(requested), 4)

            start_time = time.monotonic()
            committed = []
            while time.monotonic() - start_time < 5.0 and len(committed) < 4:
                got = world.poll_generated_chunks(max_count=4, centers=[center])
                committed.extend(got)
                if len(committed) < 4:
                    time.sleep(0.02)

            self.assertEqual(len(committed), 4)
            for key in committed:
                self.assertIn(key, world.chunks)
                self.assertIn(key, world.dirty)
        finally:
            world.stop_background_generator()

    def test_background_generator_threadpool_fallback(self):
        world = World(seed=888)
        try:
            world.start_background_generator(max_workers=2, use_processes=False)
            center = (3, 4, 3)
            requested = world.request_chunks([center], max_in_flight=2)
            self.assertEqual(len(requested), 2)

            start_time = time.monotonic()
            committed = []
            while time.monotonic() - start_time < 5.0 and len(committed) < 2:
                got = world.poll_generated_chunks(max_count=2, centers=[center])
                committed.extend(got)
                if len(committed) < 2:
                    time.sleep(0.02)

            self.assertEqual(len(committed), 2)
            for key in committed:
                self.assertIn(key, world.chunks)
        finally:
            world.stop_background_generator()


if __name__ == "__main__":
    unittest.main()

