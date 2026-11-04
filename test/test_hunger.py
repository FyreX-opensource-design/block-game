import pytest
from server.world import World
from server.player import Player
from server.items import ITEMS


def test_initial_hunger_state():
    world = World(seed=1)
    player = Player(world)
    player.survival = True

    assert player.max_hunger == 20.0
    assert player.hunger == 20.0
    assert player.saturation == 5.0
    assert player.exhaustion == 0.0
    assert player.hunger_fraction == 1.0


def test_exhaustion_depletes_saturation_then_hunger():
    world = World(seed=1)
    player = Player(world)
    player.survival = True

    # Saturation starts at 5.0. 4.0 exhaustion drains 1.0 saturation.
    player.add_exhaustion(4.0)
    assert player.saturation == 4.0
    assert player.hunger == 20.0

    # Drain remaining saturation (16.0 exhaustion drains 4.0 saturation)
    player.add_exhaustion(16.0)
    assert player.saturation == 0.0
    assert player.hunger == 20.0

    # Exhaustion now drains hunger
    player.add_exhaustion(8.0)
    assert player.saturation == 0.0
    assert player.hunger == 18.0


def test_sprint_restriction_when_hunger_low():
    world = World(seed=1)
    player = Player(world)
    player.survival = True

    player.hunger = 6.0
    player.input.sprint = True
    player.update(0.05)
    assert not player.input.sprint

    player.hunger = 7.0
    player.input.sprint = True
    player.update(0.05)
    assert player.input.sprint


def test_eat_food_restores_hunger_and_saturation():
    world = World(seed=1)
    player = Player(world)
    player.survival = True

    player.hunger = 10.0
    player.saturation = 0.0

    # Eat an apple (nutrition = 4, saturation = 2.4)
    apple = ITEMS.get("apple")
    assert apple is not None
    assert apple.food == 4.0
    assert getattr(apple, "saturation", 0) > 0

    player.eat(apple.food, apple.saturation)
    assert player.hunger == 14.0
    assert player.saturation == apple.saturation

    # Cannot exceed max_hunger
    player.eat(10.0, 10.0)
    assert player.hunger == 20.0
    assert player.saturation <= player.hunger


def test_natural_regen_when_well_fed():
    world = World(seed=1)
    player = Player(world)
    player.survival = True
    player.health = 10.0
    player.hunger = 20.0
    player.saturation = 5.0

    # 3.0s does not regen yet (interval is 4.0s)
    player.update_health(3.0)
    assert player.health == 10.0

    # Another 1.1s reaches 4.0s, heals 1 HP and consumes 3.0 exhaustion
    player.update_health(1.1)
    assert player.health == 11.0
    assert player.exhaustion > 0.0

    # If hunger < 18, regen does not occur
    player.hunger = 17.0
    player.update_health(5.0)
    assert player.health == 11.0


def test_starvation_damage_when_empty():
    world = World(seed=1)
    player = Player(world)
    player.survival = True
    player.health = 20.0
    player.hunger = 0.0

    # 3.0s does not starve yet
    events = player.update_health(3.0)
    assert player.health == 20.0
    assert len(events) == 0

    # Reaching 4.0s deals starvation damage
    events = player.update_health(1.1)
    assert player.health == 19.0
    assert len(events) == 1
    assert events[0].damage_type.id == "starvation"

    # In standard survival, starves down to 1.0 HP
    player.health = 1.0
    events = player.update_health(5.0)
    assert player.health == 1.0
    assert len(events) == 0


def test_respawn_resets_hunger_and_saturation():
    world = World(seed=1)
    player = Player(world)
    player.survival = True
    player.hunger = 2.0
    player.saturation = 0.0
    player.dead = True

    player.respawn()
    assert not player.dead
    assert player.health == 20.0
    assert player.hunger == 20.0
    assert player.saturation == 5.0


def test_multiplayer_hunger_sync_and_persistence():
    import tempfile
    from unittest.mock import MagicMock
    from server.network import MultiplayerServer, RemoteClient

    with tempfile.TemporaryDirectory() as tmp:
        server = MultiplayerServer(host="127.0.0.1", port=0, saves_dir=tmp, world=World(seed=1))
        mock_client = MagicMock(spec=RemoteClient)
        mock_client.alive = True
        mock_client.uuid = "player-hunger-1"
        mock_client.username = "Tester"
        mock_client.pos = [10.0, 5.0, 10.0]
        mock_client.yaw = 0.0
        mock_client.pitch = 0.0
        mock_client.held_item = ""
        server.clients["player-hunger-1"] = mock_client

        player = Player(server.world)
        player.uuid = "player-hunger-1"
        player.survival = True
        player.hunger = 10.0
        player.saturation = 0.0
        server.world.register_player(player)

        sent_packets = []
        mock_client.send = lambda pkt: sent_packets.append(pkt)

        # Send eat_item packet
        server._handle_packet(mock_client, {"type": "eat_item", "item": "apple"})

        assert player.hunger == 14.0
        assert len(sent_packets) == 1
        health_pkt = sent_packets[0]
        assert health_pkt.get("type") == "health"
        assert health_pkt.get("hunger") == 14.0
        assert health_pkt.get("max_hunger") == 20.0
        assert health_pkt.get("saturation") == 2.4

        # Save player data
        server._save_player_data(mock_client)
        loaded = server._load_player_data("player-hunger-1")
        assert loaded is not None
        assert loaded.get("hunger") == 14.0
        assert loaded.get("saturation") == 2.4

