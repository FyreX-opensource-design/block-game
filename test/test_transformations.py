import json
import tempfile
import unittest
from pathlib import Path

from server.blocks import AIR
from server.transformations import (
    TransformationSystem,
    load_transformations,
    transformation_from_mapping,
)
from server.world import World


def _flat_world():
    world = World(seed=1)
    world.load_chunk(0, 5, 0)
    world.chunks[(0, 5, 0)].data.fill(0)
    return world


def _run(world, seconds, step=0.5):
    changed = []
    for _ in range(int(seconds / step)):
        changed += world.update_transformations(step)
    return changed


class TransformationTests(unittest.TestCase):
    def test_default_assets_load(self):
        rules = {r.id: r for r in load_transformations()}
        self.assertEqual(rules["dirt_to_grass"].to_block, "grass")
        self.assertEqual(rules["mud_to_muddy_grass"].to_block, "muddy_grass")
        for rule in rules.values():
            self.assertGreaterEqual(rule.min_time, 120)
            self.assertLessEqual(rule.max_time, 180)

    def test_dirt_next_to_grass_converts_within_three_minutes(self):
        world = _flat_world()
        world.set_block(5, 80, 5, "grass")
        world.set_block(6, 80, 5, "dirt")
        _run(world, 100)
        self.assertEqual(world.get_block(6, 80, 5), "dirt")
        _run(world, 100)
        self.assertEqual(world.get_block(6, 80, 5), "grass")

    def test_spreads_one_block_at_a_time(self):
        world = _flat_world()
        world.set_block(5, 80, 5, "grass")
        world.set_block(6, 80, 5, "dirt")
        world.set_block(7, 80, 5, "dirt")
        _run(world, 100)
        _run(world, 100)
        self.assertEqual(world.get_block(6, 80, 5), "grass")
        self.assertEqual(world.get_block(7, 80, 5), "dirt")

    def test_covered_dirt_does_not_convert(self):
        world = _flat_world()
        world.set_block(5, 80, 5, "grass")
        world.set_block(6, 80, 5, "dirt")
        world.set_block(6, 81, 5, "stone")
        _run(world, 400)
        self.assertEqual(world.get_block(6, 80, 5), "dirt")

    def test_block_placed_while_waiting_cancels(self):
        world = _flat_world()
        world.set_block(5, 80, 5, "grass")
        world.set_block(6, 80, 5, "dirt")
        _run(world, 60)
        world.set_block(6, 81, 5, "stone")
        _run(world, 300)
        self.assertEqual(world.get_block(6, 80, 5), "dirt")

    def test_no_source_nearby_does_nothing(self):
        world = _flat_world()
        world.set_block(6, 80, 5, "dirt")
        _run(world, 400)
        self.assertEqual(world.get_block(6, 80, 5), "dirt")

    def test_mud_becomes_muddy_grass(self):
        world = _flat_world()
        world.set_block(5, 80, 5, "grass")
        world.set_block(6, 80, 5, "mud")
        _run(world, 200)
        self.assertEqual(world.get_block(6, 80, 5), "muddy_grass")

    def test_custom_rule_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "sand_to_gravel.json").write_text(
                json.dumps(
                    {
                        "id": "sand_to_gravel",
                        "from": "sand",
                        "to": "gravel",
                        "sources": ["stone"],
                        "min_time": 1,
                        "max_time": 2,
                        "require_clear_above": False,
                    }
                )
            )
            rules = load_transformations(tmp)
        world = _flat_world()
        world.transformations = TransformationSystem(world, rules)
        world.set_block(5, 81, 5, "stone")
        world.set_block(6, 80, 5, "stone")  # support so the sand does not fall
        world.set_block(6, 81, 5, "sand")
        world.set_block(6, 82, 5, "stone")
        _run(world, 5)
        self.assertEqual(world.get_block(6, 81, 5), "gravel")

    def test_invalid_rules_rejected(self):
        with self.assertRaises(ValueError):
            transformation_from_mapping({"id": "x", "from": "nope", "to": "grass", "sources": ["grass"]}, "t")
        with self.assertRaises(ValueError):
            transformation_from_mapping({"id": "x", "from": "dirt", "to": "grass", "sources": []}, "t")
        with self.assertRaises(ValueError):
            transformation_from_mapping(
                {"id": "x", "from": "dirt", "to": "grass", "sources": ["grass"], "min_time": 10, "max_time": 5}, "t"
            )


if __name__ == "__main__":
    unittest.main()
