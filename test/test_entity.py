import unittest
from unittest.mock import MagicMock

from server.blocks import STONE, DIRT, GRASS, AIR
from server.entity import ItemEntity, DESPAWN_TIME
from server.items import stack_limit
from server.stacks import empty_slots, make_stack, stack_count, stack_id, add_to_inventory
from server.world import World


class TestItemEntity(unittest.TestCase):
    def test_entity_creation(self):
        entity = ItemEntity("dirt", count=5, pos=[1.0, 20.0, 1.0], vel=[0.5, 2.0, -0.5])
        self.assertEqual(entity.item, "dirt")
        self.assertEqual(entity.count, 5)
        self.assertEqual(entity.pos, [1.0, 20.0, 1.0])
        self.assertEqual(entity.vel, [0.5, 2.0, -0.5])
        self.assertEqual(entity.despawn_timer, DESPAWN_TIME)
        self.assertEqual(entity.pickup_delay, 0.0)
        self.assertFalse(entity.on_ground)

    def test_serialization(self):
        entity = ItemEntity("cobble", count=12, pos=[5.0, 10.0, 15.0], vel=[0.1, 0.2, 0.3], despawn_timer=250.0, pickup_delay=1.5)
        data = entity.to_dict()
        self.assertEqual(data["item"], "cobble")
        self.assertEqual(data["count"], 12)
        self.assertEqual(data["pos"], [5.0, 10.0, 15.0])
        self.assertEqual(data["despawn_timer"], 250.0)
        self.assertEqual(data["pickup_delay"], 1.5)

        restored = ItemEntity.from_dict(data)
        self.assertEqual(restored.id, entity.id)
        self.assertEqual(restored.item, "cobble")
        self.assertEqual(restored.count, 12)
        self.assertEqual(restored.pos, [5.0, 10.0, 15.0])

    def test_gravity_and_ground_collision(self):
        world = MagicMock()
        # Pretend y=10 is solid ground
        world.is_solid.side_effect = lambda x, y, z: y <= 10

        entity = ItemEntity("dirt", count=1, pos=[0.5, 12.0, 0.5], vel=[0.0, 0.0, 0.0])
        # Update 1 second: entity should fall towards ground
        for _ in range(20):
            entity.update(0.05, world)

        # Entity should hit ground and land at y=11.0 (on top of block y=10)
        self.assertTrue(entity.on_ground)
        self.assertAlmostEqual(entity.pos[1], 11.0, delta=0.01)
        self.assertEqual(entity.vel[1], 0.0)

    def test_despawn_timer(self):
        entity = ItemEntity("dirt", count=1, pos=[0.5, 10.0, 0.5], despawn_timer=5.0)
        entity.update(2.0, world=None)
        self.assertAlmostEqual(entity.despawn_timer, 3.0)

        world = World(seed=1)
        world.entities = [entity]
        world.update_entities(4.0)
        # Should now be despawned and removed from world
        self.assertEqual(len(world.entities), 0)

    def test_pickup_delay(self):
        entity = ItemEntity("dirt", count=1, pickup_delay=1.0)
        self.assertEqual(entity.pickup_delay, 1.0)
        entity.update(0.4, world=None)
        self.assertAlmostEqual(entity.pickup_delay, 0.6)
        entity.update(0.8, world=None)
        self.assertEqual(entity.pickup_delay, 0.0)

    def test_merging(self):
        e1 = ItemEntity("dirt", count=10, pos=[1.0, 10.0, 1.0])
        e2 = ItemEntity("dirt", count=5, pos=[1.1, 10.0, 1.1])
        merged = e1.try_merge(e2)
        self.assertTrue(merged)
        self.assertEqual(e1.count, 15)
        self.assertEqual(e2.count, 0)

        # Distinct items do not merge
        e3 = ItemEntity("stone", count=5, pos=[1.1, 10.0, 1.1])
        self.assertFalse(e1.try_merge(e3))

    def test_player_pickup_inventory_space(self):
        hotbar = empty_slots(9)
        storage = empty_slots(27)

        # Space available: picks up
        entity = ItemEntity("dirt", count=3, pos=[0.0, 0.0, 0.0])
        leftover = add_to_inventory([hotbar, storage], entity.item, entity.count)
        self.assertEqual(leftover, 0)
        self.assertEqual(stack_id(hotbar[0]), "dirt")
        self.assertEqual(stack_count(hotbar[0]), 3)

        # When inventory is completely full:
        limit = stack_limit("dirt")
        full_hotbar = [make_stack("stone", 64) for _ in range(9)]
        full_storage = [make_stack("stone", 64) for _ in range(27)]
        entity_dirt = ItemEntity("dirt", count=5)
        leftover = add_to_inventory([full_hotbar, full_storage], entity_dirt.item, entity_dirt.count)
        self.assertEqual(leftover, 5)  # None picked up

    def test_world_spawn_and_remove(self):
        world = World(seed=1)
        entity = world.spawn_item("log", count=2, pos=[10.0, 20.0, 10.0], pickup_delay=0.5)
        self.assertIn(entity, world.entities)
        self.assertEqual(entity.item, "log")
        self.assertEqual(entity.count, 2)

        world.remove_entity(entity.id)
        self.assertNotIn(entity, world.entities)


if __name__ == "__main__":
    unittest.main()

