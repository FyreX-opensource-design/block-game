import argparse
import base64
import json
import math
import os
from pathlib import Path
import random
import re
import select
import socket
import struct
import sys
import threading
import time
import zlib

import numpy as np

from server.blocks import CHUNK_SIZE, LOAD_RADIUS, VERTICAL_RADIUS, SECTION_COUNT
from server.commands import CommandContext, CommandHandler
from server.damage import DEFAULT_MAX_HEALTH
from server.player import Player
from server.profile import SKIN_SIZE
from server.stacks import normalize_stack, stack_count, stack_id
from server.storage import WorldCatalog
from server.world import World

DEFAULT_PORT = 25565
_HEADER = struct.Struct(">I")


def encode_skin(skin_rgba):
    if skin_rgba is None:
        return ""
    raw = np.ascontiguousarray(skin_rgba, dtype=np.uint8).tobytes()
    return base64.b64encode(zlib.compress(raw, level=6)).decode("ascii")


def decode_skin(encoded_str):
    if not encoded_str:
        return None
    try:
        raw = zlib.decompress(base64.b64decode(encoded_str))
        if len(raw) != SKIN_SIZE * SKIN_SIZE * 4:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape(SKIN_SIZE, SKIN_SIZE, 4).copy()
    except Exception:
        return None


def encode_chunk(chunk_data):
    raw = np.ascontiguousarray(chunk_data, dtype=np.uint16).tobytes()
    return base64.b64encode(zlib.compress(raw, level=1)).decode("ascii")


def decode_chunk(encoded_str):
    raw = zlib.decompress(base64.b64decode(encoded_str))
    if len(raw) == CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE:
        return np.frombuffer(raw, dtype=np.uint8).astype(np.uint16).reshape(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE).copy()
    return np.frombuffer(raw, dtype=np.uint16).reshape(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE).copy()


def send_packet(sock, packet):
    payload = json.dumps(packet).encode("utf-8")
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def parse_packets(buffer):
    """Parse length-prefixed JSON packets from bytearray buffer.
    
    Yields parsed dict packets and modifies the buffer in-place.
    """
    packets = []
    while len(buffer) >= 4:
        length = _HEADER.unpack_from(buffer, 0)[0]
        if len(buffer) < 4 + length:
            break
        data = buffer[4 : 4 + length]
        del buffer[: 4 + length]
        try:
            packet = json.loads(data.decode("utf-8"))
            packets.append(packet)
        except Exception:
            continue
    return packets


class RemoteClient:
    def __init__(self, sock, addr):
        self.sock = sock
        self.addr = addr
        self.uuid = ""
        self.username = ""
        self.skin_rgba = None
        self.pos = [0.5, 30.0, 0.5]
        self.vel = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.pitch = 0.0
        self.held_item = ""
        self.sent_chunks = set()
        self.in_buffer = bytearray()
        self.out_queue = []
        self.alive = True

    def send(self, packet):
        if not self.alive:
            return
        payload = json.dumps(packet).encode("utf-8")
        self.out_queue.append(_HEADER.pack(len(payload)) + payload)

    def close(self):
        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass


class MultiplayerServer:
    def __init__(
        self,
        host="0.0.0.0",
        port=DEFAULT_PORT,
        world=None,
        saves_dir=None,
        world_name="multiplayer",
        seed=1337,
        gamemode="survival",
    ):
        self.host = host
        self.requested_port = port
        self.actual_port = port
        self.saves_dir = saves_dir
        self.world_name = world_name
        self.seed = seed
        self.gamemode = str(gamemode or "survival").lower()
        self.world = world
        if self.world is not None:
            self.world.on_block_set = lambda bx, by, bz, b: self.broadcast(
                {"type": "block_change", "x": bx, "y": by, "z": bz, "block": b}
            )
            self.world.on_entity_spawn = self._broadcast_entity_spawn
        self.world_info = None
        self.world_dir = None
        self.player_dir = None
        self.players_data = {}
        self.catalog = WorldCatalog(saves_dir) if saves_dir is not None or world is None else None
        self.command_handler = CommandHandler()
        self.clients = {}  # uuid -> RemoteClient
        self.pending_clients = []  # unauthenticated RemoteClients
        self.running = False
        self.server_sock = None
        self._thread = None
        self._last_save = time.monotonic()
        self._time_sync_timer = 0.0
        self._mob_sync_timer = 0.0
        self.open_furnaces = {}  # uuid -> (x, y, z)
        self._furnace_sync_timer = 0.0

    def _player_file(self, uuid):
        if not self.player_dir:
            return None
        safe_uuid = re.sub(r"[^a-zA-Z0-9_-]", "", str(uuid))
        if not safe_uuid:
            return None
        return self.player_dir / f"{safe_uuid}.json"

    def _load_player_data(self, uuid):
        if uuid in self.players_data:
            return self.players_data[uuid]
        path = self._player_file(uuid)
        if path and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.players_data[uuid] = data
                return data
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        return None

    def _save_player_data(self, client):
        if not client or not client.uuid:
            return
        existing = self.players_data.get(client.uuid) or {}
        player = self.world.players.get(client.uuid) if self.world is not None else None
        spawn = getattr(player, "spawn_point", None) or existing.get("spawn")
        health = getattr(player, "health", None)
        if health is None:
            health = existing.get("health", DEFAULT_MAX_HEALTH)
        max_health = getattr(player, "max_health", None) or existing.get("max_health", DEFAULT_MAX_HEALTH)
        data = {
            "uuid": client.uuid,
            "username": client.username,
            "pos": [float(v) for v in client.pos],
            "yaw": float(client.yaw),
            "pitch": float(client.pitch),
            "held": str(client.held_item),
            "spawn": [float(v) for v in spawn] if spawn else None,
            "health": float(health),
            "max_health": float(max_health),
            "last_played": time.time(),
        }
        self.players_data[client.uuid] = data
        path = self._player_file(client.uuid)
        if path:
            try:
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                os.replace(tmp, path)
            except OSError:
                pass

    def set_player_spawn(self, uuid, spawn):
        """Store a player's personal spawn point, persisting it immediately."""
        uuid = str(uuid or "")
        if not uuid:
            return None
        data = self._load_player_data(uuid) or {"uuid": uuid}
        client = self.clients.get(uuid)
        if client is not None:
            player = self.world.players.get(uuid) if self.world is not None else None
            data.update(
                {
                    "username": client.username,
                    "pos": [float(v) for v in client.pos],
                    "yaw": float(client.yaw),
                    "pitch": float(client.pitch),
                    "held": str(client.held_item),
                    "health": float(getattr(player, "health", data.get("health", DEFAULT_MAX_HEALTH))),
                    "max_health": float(getattr(player, "max_health", data.get("max_health", DEFAULT_MAX_HEALTH))),
                    "last_played": time.time(),
                }
            )
        data["spawn"] = [float(v) for v in spawn] if spawn else None
        self.players_data[uuid] = data
        path = self._player_file(uuid)
        if path:
            try:
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                os.replace(tmp, path)
            except OSError:
                pass
        return data.get("spawn")

    def start(self):
        """Start the server on a background daemon thread."""
        if self.world is None:
            existing = None
            if self.catalog is not None:
                for w in self.catalog.list_worlds():
                    if w.name.lower() == self.world_name.lower():
                        existing = w
                        break
            if existing is not None:
                info = existing
                self.gamemode = getattr(existing, "gamemode", self.gamemode)
            else:
                seed_val = self.seed if self.seed is not None else random.randint(0, 2_147_483_647)
                info = self.catalog.create(self.world_name, seed_val, gamemode=self.gamemode)
            self.seed = info.seed
            self.world = World(
                seed=info.seed,
                store=self.catalog.store_for(info.id),
                day_time=getattr(info, "day_time", 1000.0),
                spawn_point=getattr(info, "spawn", None),
            )
            self.world.on_block_set = lambda bx, by, bz, b: self.broadcast(
                {"type": "block_change", "x": bx, "y": by, "z": bz, "block": b}
            )
            self.world.on_entity_spawn = self._broadcast_entity_spawn
            self.world_info = info
            self.world_dir = self.catalog._folder(info.id)
        elif self.world.store is not None and hasattr(self.world.store, "folder"):
            self.world_dir = Path(self.world.store.folder).parent

        if self.world_dir is not None:
            self.player_dir = Path(self.world_dir) / "players"
            try:
                self.player_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.player_dir = None
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.requested_port))
        self.actual_port = self.server_sock.getsockname()[1]
        self.server_sock.listen(16)
        self.server_sock.setblocking(False)
        self.running = True

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def run(self):
        """Run the server blocking on the current thread."""
        self.start()
        try:
            while self.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.world:
            self.world.flush()
            if self.world_info and self.catalog:
                self.world_info.day_time = float(self.world.day_time)
                self.world_info.spawn = list(self.world.get_spawn_point())
                self.catalog.write(self.world_info)
        for client in list(self.clients.values()):
            self._save_player_data(client)
        for client in list(self.clients.values()) + list(self.pending_clients):
            try:
                send_packet(client.sock, {"type": "disconnect", "reason": "Server closed"})
            except OSError:
                pass
            client.close()
        self.clients.clear()
        self.pending_clients.clear()
        if self.server_sock:
            try:
                self.server_sock.close()
            except OSError:
                pass
            self.server_sock = None
        if self.world:
            self.world.flush()

    def _run_loop(self):
        while self.running:
            self.tick()
            time.sleep(0.02)  # ~50 ticks/sec

    def tick(self):
        if self.server_sock is not None and self.running:
            all_socks = [self.server_sock]
            active_clients = [c for c in self.pending_clients + list(self.clients.values()) if c.alive and getattr(c, "sock", None) is not None]
            client_by_sock = {c.sock: c for c in active_clients}
            all_socks.extend(client_by_sock.keys())

            # Select readable sockets
            try:
                readable, writable, _ = select.select(
                    all_socks,
                    [c.sock for c in active_clients if c.out_queue],
                    [],
                    0.01,
                )
            except (ValueError, OSError):
                readable, writable = [], []

            for s in readable:
                if s is self.server_sock:
                    try:
                        conn, addr = self.server_sock.accept()
                        conn.setblocking(False)
                        client = RemoteClient(conn, addr)
                        self.pending_clients.append(client)
                    except OSError:
                        pass
                else:
                    client = client_by_sock.get(s)
                    if client is None:
                        continue
                    try:
                        data = s.recv(16384)
                        if not data:
                            self._disconnect_client(client)
                        else:
                            client.in_buffer.extend(data)
                            packets = parse_packets(client.in_buffer)
                            for pkt in packets:
                                self._handle_packet(client, pkt)
                    except OSError:
                        self._disconnect_client(client)

            for s in writable:
                client = client_by_sock.get(s)
                if client is None or not client.out_queue:
                    continue
                try:
                    while client.out_queue:
                        data = client.out_queue[0]
                        sent = s.send(data)
                        if sent >= len(data):
                            if client.out_queue:
                                client.out_queue.pop(0)
                        else:
                            if client.out_queue:
                                client.out_queue[0] = data[sent:]
                            break
                except BlockingIOError:
                    pass
                except OSError:
                    self._disconnect_client(client)

        # Sync chunks around all online players
        self._sync_player_chunks()

        # Update world entities, furnaces, fluids, and time
        if self.world:
            # Player physics runs on the clients, so the server never calls
            # Player.update. Run the support scan under each player itself,
            # otherwise generated sand/gravel only falls on the stepping
            # client and the other players never see it.
            self._update_player_falling()
            prev_entities = {e.id: e for e in self.world.entities}
            self.world.advance_time(0.02)
            self._time_sync_timer += 0.02
            if self._time_sync_timer >= 2.0:
                self._time_sync_timer = 0.0
                if self.clients:
                    self.broadcast({"type": "time_update", "day_time": float(self.world.day_time)})
            self.world.update_entities(0.02)
            curr_entities = {e.id: e for e in self.world.entities}
            # New entities are announced by World.on_entity_spawn as they are
            # created. Only removals need diffing here.
            for eid in prev_entities:
                if eid not in curr_entities:
                    self.broadcast({"type": "remove_entity", "id": eid})

            # Item pickups & magnetic pull for connected players (even if tabbed out)
            if getattr(self.world, "entities", None) and self.clients:
                for entity in list(self.world.entities):
                    if (
                        not getattr(entity, "count", 0)
                        or getattr(entity, "pickup_delay", 0) > 0
                        or not hasattr(entity, "item")
                        or getattr(entity, "is_falling_block", False)
                    ):
                        continue
                    ex, ey, ez = entity.pos
                    for client in list(self.clients.values()):
                        if not client.alive or not client.uuid:
                            continue
                        cx, cy, cz = client.pos
                        dx = ex - cx
                        dy = ey - (cy + 0.8)
                        dz = ez - cz
                        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                        if dist < 2.5:
                            pull = (2.5 - dist) * 4.0 * 0.02
                            entity.vel[0] -= dx * pull
                            entity.vel[2] -= dz * pull
                        if dist <= 1.6:
                            item_id = entity.item
                            count = entity.count
                            eid = entity.id
                            self.world.remove_entity(eid)
                            self.broadcast({"type": "remove_entity", "id": eid})
                            client.send({
                                "type": "give_item",
                                "id": eid,
                                "item": item_id,
                                "count": count,
                            })
                            break

            self._mob_sync_timer += 0.02
            if self._mob_sync_timer >= 0.1:
                self._mob_sync_timer = 0.0
                mobs = [e.to_dict() for e in curr_entities.values() if getattr(e, "is_mob", False)]
                if mobs and self.clients:
                    self.broadcast({"type": "mobs_update", "mobs": mobs})

            # Track furnace slot and lit state transitions before update
            pre_slots = {}
            pre_lit = {}
            if getattr(self.world, "furnaces", None):
                for pos, f in self.world.furnaces.items():
                    pre_lit[pos] = f.is_lit
                    pre_slots[pos] = (
                        (stack_id(f.input), stack_count(f.input)),
                        (stack_id(f.fuel), stack_count(f.fuel)),
                        (stack_id(f.output), stack_count(f.output)),
                    )

            self.world.update_furnaces(0.02)

            if getattr(self.world, "furnaces", None):
                for pos, f in self.world.furnaces.items():
                    lit_changed = pos in pre_lit and pre_lit[pos] != f.is_lit
                    cur_slots = (
                        (stack_id(f.input), stack_count(f.input)),
                        (stack_id(f.fuel), stack_count(f.fuel)),
                        (stack_id(f.output), stack_count(f.output)),
                    )
                    slots_changed = pos in pre_slots and pre_slots[pos] != cur_slots
                    if lit_changed:
                        self.broadcast({"type": "furnace_sync", "furnace": f.to_dict()})
                    elif slots_changed:
                        # Full sync to viewers because an item finished smelting or fuel was consumed
                        sync_msg = {"type": "furnace_sync", "furnace": f.to_dict()}
                        for c_uuid, f_pos in list(self.open_furnaces.items()):
                            if f_pos == pos and c_uuid in self.clients:
                                self.clients[c_uuid].send(sync_msg)

            # Periodic progress update (cook_time and burn_time ONLY, does NOT touch item slots)
            self._furnace_sync_timer += 0.02
            if self._furnace_sync_timer >= 0.1:
                self._furnace_sync_timer = 0.0
                if self.open_furnaces and getattr(self.world, "furnaces", None):
                    for pos in set(self.open_furnaces.values()):
                        furnace = self.world.furnaces.get(pos)
                        if furnace is not None:
                            prog_msg = {
                                "type": "furnace_progress",
                                "x": furnace.x,
                                "y": furnace.y,
                                "z": furnace.z,
                                "burn_time": furnace.burn_time,
                                "burn_duration": furnace.burn_duration,
                                "cook_time": furnace.cook_time,
                                "cook_duration": furnace.cook_duration,
                            }
                            for c_uuid, f_pos in list(self.open_furnaces.items()):
                                if f_pos == pos and c_uuid in self.clients:
                                    self.clients[c_uuid].send(prog_msg)
            self.world.update_leaves(0.02)
            self.world.update_transformations(0.02)
            self.world.update_saplings(0.02)
            changed = self.world.update_fluids(0.02)
            if getattr(self.world, "on_block_set", None) is None:
                for x, y, z, block in changed:
                    self.broadcast(
                        {
                            "type": "block_change",
                            "x": x,
                            "y": y,
                            "z": z,
                            "block": block,
                        }
                    )

        # Survival health (the server is authoritative in multiplayer).
        self._update_player_health(0.02)

        # Autosave world and players
        now = time.monotonic()
        if now - self._last_save > 10.0:
            self._last_save = now
            if self.world:
                self.world.flush()
                if self.world_info and self.catalog:
                    self.world_info.day_time = float(self.world.day_time)
                    self.world_info.spawn = list(self.world.get_spawn_point())
                    self.catalog.write(self.world_info)
            for client in list(self.clients.values()):
                self._save_player_data(client)

    def _send_player_health(self, client, player):
        client.send(
            {
                "type": "health",
                "health": float(player.health),
                "max_health": float(player.max_health),
                "hurt_time": float(player.hurt_time),
                "dead": bool(player.dead),
            }
        )

    def _respawn_client_player(self, client, player, cause=""):
        had_spawn = player.get_spawn_point() is not None
        pos = player.respawn()
        client.pos = list(pos)
        client.vel = [0.0, 0.0, 0.0]
        client.send({"type": "teleport", "pos": list(pos), "vel": [0.0, 0.0, 0.0]})
        self._send_player_health(client, player)
        where = "your spawn point" if had_spawn else "the world spawn"
        text = "You died" + (f": {cause}" if cause else "") + f". You respawned at {where}."
        client.send({"type": "chat", "sender": "", "text": text, "color": [255, 100, 100, 255]})
        self._save_player_data(client)

    def _update_player_falling(self):
        """Re-check block support under every connected player.

        Player movement is simulated on the clients, so ``Player.update`` (and
        its falling-block scan) never runs on the server. Generated sand or
        gravel can therefore sit on air until a player walks over it. Do the
        scan server-side so the resulting fall is authoritative and broadcast
        to every client, instead of only happening in the stepping player's
        local world.
        """
        if self.world is None:
            return
        for uuid, client in list(self.clients.items()):
            if not getattr(client, "alive", True) or not client.uuid:
                continue
            player = self.world.players.get(uuid)
            radius = getattr(player, "radius", 0.3)
            pos = getattr(client, "pos", None)
            if not pos:
                continue
            by = math.floor(pos[1] - 0.05)
            min_x = math.floor(pos[0] - radius)
            max_x = math.floor(pos[0] + radius - 1e-6)
            min_z = math.floor(pos[2] - radius)
            max_z = math.floor(pos[2] + radius - 1e-6)
            for x in range(min_x, max_x + 1):
                for z in range(min_z, max_z + 1):
                    self.world.notify_neighbors_falling(x, by, z)

    def _update_player_health(self, dt):
        if self.world is None:
            return
        for uuid, client in list(self.clients.items()):
            if not getattr(client, "alive", True):
                continue
            player = self.world.players.get(uuid)
            if player is None:
                continue
            vel = getattr(client, "vel", None) or [0.0, 0.0, 0.0]
            try:
                grounded = abs(float(vel[1])) < 1e-3
            except (IndexError, TypeError, ValueError):
                grounded = True
            try:
                in_water = player.in_water()
            except Exception:
                in_water = False
            events = player.update_health(dt, grounded=grounded, in_water=in_water)
            if player.dead:
                cause = ""
                for event in events:
                    if event.fatal and event.message:
                        cause = event.message
                self._respawn_client_player(client, player, cause)
            elif events:
                self._send_player_health(client, player)

    def _handle_packet(self, client, packet):
        p_type = packet.get("type")
        if p_type == "login":
            uuid = str(packet.get("uuid", "")).strip()
            if not uuid:
                client.close()
                return
            client.uuid = uuid
            client.username = str(packet.get("username", "Player"))
            client.skin_rgba = decode_skin(packet.get("skin"))

            if client in self.pending_clients:
                self.pending_clients.remove(client)
            self.clients[uuid] = client

            # Register player in world
            player = Player(self.world)
            player.apply_identity(uuid, client.skin_rgba)
            player.gamemode = str(getattr(self, "gamemode", "survival") or "survival").lower()
            self.world.register_player(player)

            # Determine spawn position: personal spawn point, then saved position,
            # then the world spawn point.
            saved = self._load_player_data(uuid)
            personal = None
            if saved:
                raw_spawn = saved.get("spawn")
                if isinstance(raw_spawn, (list, tuple)) and len(raw_spawn) >= 3:
                    try:
                        personal = [float(raw_spawn[0]), float(raw_spawn[1]), float(raw_spawn[2])]
                    except (TypeError, ValueError):
                        personal = None
            if personal is not None:
                player.spawn_point = list(personal)
                client.pos = list(personal)
                client.yaw = float(saved.get("yaw", 0.0))
                client.pitch = float(saved.get("pitch", 0.0))
                client.held_item = str(saved.get("held", ""))
                player.pos = list(client.pos)
                player.yaw = client.yaw
                player.pitch = client.pitch
            elif saved and "pos" in saved:
                client.pos = [float(v) for v in saved["pos"]]
                client.yaw = float(saved.get("yaw", 0.0))
                client.pitch = float(saved.get("pitch", 0.0))
                client.held_item = str(saved.get("held", ""))
                player.pos = list(client.pos)
                player.yaw = client.yaw
                player.pitch = client.pitch
            else:
                sp_x, sp_y, sp_z = self.world.get_spawn_point()
                ground = self.world.column_top(int(math.floor(sp_x)), int(math.floor(sp_z)))
                client.pos = [float(sp_x), float(max(sp_y, ground + 1.0)), float(sp_z)]
                player.pos = list(client.pos)

            # Restore health; a saved fatal value means the player respawns.
            try:
                player.max_health = float(saved.get("max_health", DEFAULT_MAX_HEALTH)) if saved else DEFAULT_MAX_HEALTH
            except (TypeError, ValueError):
                player.max_health = DEFAULT_MAX_HEALTH
            saved_health = saved.get("health") if saved else None
            try:
                saved_health = float(saved_health) if saved_health is not None else None
            except (TypeError, ValueError):
                saved_health = None
            if saved_health is None or saved_health <= 0.0:
                player.health = player.max_health
                player.dead = False
            else:
                player.set_health(saved_health)
            player.reset_damage_state()

            # Build list of already-connected players
            existing_players = []
            for other_uuid, other in self.clients.items():
                if other_uuid != uuid and other.alive:
                    existing_players.append(
                        {
                            "uuid": other.uuid,
                            "username": other.username,
                            "pos": list(other.pos),
                            "yaw": other.yaw,
                            "pitch": other.pitch,
                            "skin": encode_skin(other.skin_rgba),
                            "held": other.held_item,
                        }
                    )

            # Send login_ack
            client.send(
                {
                    "type": "login_ack",
                    "uuid": uuid,
                    "seed": self.world.seed,
                    "gamemode": self.gamemode,
                    "pos": list(client.pos),
                    "yaw": client.yaw,
                    "pitch": client.pitch,
                    "players": existing_players,
                    "entities": [e.to_dict() for e in self.world.entities] if self.world else [],
                    "day_time": self.world.day_time if self.world else 1000.0,
                    "health": float(player.health),
                    "max_health": float(player.max_health),
                    "hurt_time": float(player.hurt_time),
                }
            )

            # Broadcast player_join to all others
            join_msg = {
                "type": "player_join",
                "uuid": uuid,
                "username": client.username,
                "pos": list(client.pos),
                "yaw": client.yaw,
                "pitch": client.pitch,
                "skin": encode_skin(client.skin_rgba),
                "held": client.held_item,
            }
            self.broadcast(join_msg, exclude=uuid)
            self.broadcast(
                {
                    "type": "chat",
                    "sender": "",
                    "text": f"{client.username} joined the game",
                    "color": [255, 255, 85, 255],
                }
            )

        elif p_type == "player_move":
            if not client.uuid:
                return
            client.pos = list(packet.get("pos", client.pos))
            client.vel = list(packet.get("vel", client.vel))
            client.yaw = float(packet.get("yaw", client.yaw))
            client.pitch = float(packet.get("pitch", client.pitch))
            client.held_item = str(packet.get("held", client.held_item))

            # Update world player
            player = self.world.players.get(client.uuid)
            if player is not None:
                player.pos = list(client.pos)
                player.vel = list(client.vel)
                player.yaw = client.yaw
                player.pitch = client.pitch

            # Keep in-memory player record up to date
            previous = self.players_data.get(client.uuid) or {}
            self.players_data[client.uuid] = {
                "uuid": client.uuid,
                "username": client.username,
                "pos": list(client.pos),
                "yaw": client.yaw,
                "pitch": client.pitch,
                "held": client.held_item,
                "spawn": previous.get("spawn"),
                "health": previous.get("health", DEFAULT_MAX_HEALTH),
                "max_health": previous.get("max_health", DEFAULT_MAX_HEALTH),
                "last_played": time.time(),
            }

            # Relay movement to other clients
            move_msg = {
                "type": "player_move",
                "uuid": client.uuid,
                "pos": client.pos,
                "vel": client.vel,
                "yaw": client.yaw,
                "pitch": client.pitch,
                "held": client.held_item,
            }
            self.broadcast(move_msg, exclude=client.uuid)

        elif p_type == "set_block":
            x = int(packet.get("x", 0))
            y = int(packet.get("y", 0))
            z = int(packet.get("z", 0))
            block = str(packet.get("block", ""))
            if block not in ("", "air"):
                from server.blocks import BLOCKS, can_place_block, placement_block

                if block in BLOCKS and BLOCKS[block].properties.get("rotation") == "horizontal" and not BLOCKS[block].is_variant:
                    block = placement_block(block, (0, 1, 0), yaw=getattr(client, "yaw", 0.0))

                if not can_place_block(self.world, x, y, z, block):
                    return
            cx, cy, cz = self.world.chunk_key(x, y, z)
            if (cx, cy, cz) not in self.world.chunks:
                self.world.load_chunk(cx, cy, cz)
            if self.world.set_block(x, y, z, block):
                if getattr(self.world, "on_block_set", None) is None:
                    self.broadcast(
                        {
                            "type": "block_change",
                            "x": x,
                            "y": y,
                            "z": z,
                            "block": block,
                        }
                    )

        elif p_type == "hit_block":
            x = int(packet.get("x", 0))
            y = int(packet.get("y", 0))
            z = int(packet.get("z", 0))
            item = str(packet.get("item", ""))
            if self.world is not None and item:
                triggered, drops, _consume = self.world.hit_block_with_item(x, y, z, item)
                if triggered:
                    for drop_id, count in drops:
                        if count > 0:
                            pos = [x + 0.5, y + 0.8, z + 0.5]
                            self.world.spawn_item(drop_id, count=count, pos=pos)


        elif p_type == "drop_item":
            item = str(packet.get("item", ""))
            count = int(packet.get("count", 1))
            pos = list(packet.get("pos", [0.0, 0.0, 0.0]))
            vel = list(packet.get("vel", [0.0, 0.0, 0.0]))
            delay = float(packet.get("pickup_delay", 0.0))
            if self.world is not None and item:
                self.world.spawn_item(item, count, pos=pos, vel=vel, pickup_delay=delay)

        elif p_type == "pickup_item":
            entity_id = int(packet.get("id", 0))
            if self.world is not None:
                entity = next((e for e in self.world.entities if e.id == entity_id), None)
                if entity is not None:
                    self.world.remove_entity(entity_id)
                    self.broadcast({"type": "remove_entity", "id": entity_id})

        elif p_type == "furnace_open":
            fx = int(packet.get("x", 0))
            fy = int(packet.get("y", 0))
            fz = int(packet.get("z", 0))
            self.open_furnaces[client.uuid] = (fx, fy, fz)
            if self.world is not None:
                cx, cy, cz = self.world.chunk_key(fx, fy, fz)
                if (cx, cy, cz) not in self.world.chunks:
                    self.world.load_chunk(cx, cy, cz)
                furnace = self.world.get_or_create_furnace(fx, fy, fz)
                client.send({"type": "furnace_sync", "furnace": furnace.to_dict()})

        elif p_type == "furnace_close":
            self.open_furnaces.pop(client.uuid, None)

        elif p_type == "furnace_update":
            fx = int(packet.get("x", 0))
            fy = int(packet.get("y", 0))
            fz = int(packet.get("z", 0))
            if self.world is not None:
                from server.stacks import normalize_stack
                cx, cy, cz = self.world.chunk_key(fx, fy, fz)
                if (cx, cy, cz) not in self.world.chunks:
                    self.world.load_chunk(cx, cy, cz)
                furnace = self.world.get_or_create_furnace(fx, fy, fz)

                slots = packet.get("slots")
                if slots is None:
                    slots = {}
                    for k in ("input", "fuel", "output"):
                        if k in packet:
                            slots[k] = packet[k]

                if "input" in slots:
                    furnace.input = normalize_stack(slots["input"])
                if "fuel" in slots:
                    furnace.fuel = normalize_stack(slots["fuel"])
                if "output" in slots:
                    furnace.output = normalize_stack(slots["output"])

                # Broadcast update only to OTHER clients viewing this furnace
                sync_msg = {"type": "furnace_sync", "furnace": furnace.to_dict()}
                for c_uuid, pos in list(self.open_furnaces.items()):
                    if pos == (fx, fy, fz) and c_uuid != client.uuid and c_uuid in self.clients:
                        self.clients[c_uuid].send(sync_msg)

        elif p_type == "attack_entity":
            entity_id = int(packet.get("id", 0))
            if self.world is not None:
                entity = next((e for e in self.world.entities if e.id == entity_id), None)
                if entity is not None and getattr(entity, "is_mob", False) and getattr(entity, "alive", True):
                    p_pos = client.pos
                    dist = math.hypot(entity.pos[0] - p_pos[0], entity.pos[2] - p_pos[2])
                    if dist <= 6.0:
                        damaged = entity.take_damage(2.0, attacker=client, world=self.world)
                        if damaged:
                            self.broadcast({
                                "type": "damage_mob",
                                "id": entity.id,
                                "health": entity.health,
                                "hurt_time": entity.hurt_time,
                                "vel": entity.vel,
                                "pos": entity.pos,
                            })

        elif p_type == "interact_entity":
            entity_id = int(packet.get("id", 0))
            item_id = str(packet.get("item", ""))
            if self.world is not None:
                entity = next((e for e in self.world.entities if e.id == entity_id), None)
                if entity is not None and getattr(entity, "is_mob", False) and getattr(entity, "alive", True):
                    p_pos = client.pos
                    dist = math.hypot(entity.pos[0] - p_pos[0], entity.pos[2] - p_pos[2])
                    if dist <= 6.0:
                        if getattr(entity, "mob_type", "") == "sheep":
                            if item_id in ("shears", ""):
                                drops = entity.shear(world=self.world)
                                if drops:
                                    self.broadcast({"type": "mobs_update", "mobs": [entity.to_dict()]})
                            elif hasattr(entity, "dye") and entity.dye(item_id):
                                self.broadcast({"type": "mobs_update", "mobs": [entity.to_dict()]})

        elif p_type == "chat":
            text = str(packet.get("message", "")).strip()
            if not text:
                return
            if text.startswith("/"):
                player = self.world.players.get(client.uuid)
                ctx = CommandContext(
                    sender=client.username,
                    player=player,
                    world=self.world,
                    server=self,
                )
                self.command_handler.execute(ctx, text)
                for reply_text, color in ctx.responses:
                    client.send(
                        {
                            "type": "chat",
                            "sender": "",
                            "text": reply_text,
                            "color": list(color),
                        }
                    )
            else:
                self.broadcast(
                    {
                        "type": "chat",
                        "sender": client.username,
                        "text": text,
                    }
                )

        elif p_type == "disconnect":
            self._disconnect_client(client)

    def _sync_player_chunks(self):
        """Load and stream chunks around all connected players."""
        centers = []
        for client in self.clients.values():
            if client.alive and client.uuid:
                cx = math.floor(client.pos[0] / CHUNK_SIZE)
                cy = min(SECTION_COUNT - 1, max(0, math.floor(client.pos[1] / CHUNK_SIZE)))
                cz = math.floor(client.pos[2] / CHUNK_SIZE)
                centers.append((cx, cy, cz))

        if not centers:
            return

        # Generate / load nearest missing chunks around players
        for _ in range(4):  # up to 4 chunks per tick
            built, removed = self.world.sync_players(centers)
            if removed:
                for rem_key in removed:
                    for cl in self.clients.values():
                        cl.sent_chunks.discard(rem_key)
            if built is None:
                break

        # Send newly available chunks to players who need them
        for client in self.clients.values():
            if not client.alive or not client.uuid:
                continue
            cx = math.floor(client.pos[0] / CHUNK_SIZE)
            cy = min(SECTION_COUNT - 1, max(0, math.floor(client.pos[1] / CHUNK_SIZE)))
            cz = math.floor(client.pos[2] / CHUNK_SIZE)

            candidates = []
            for dz in range(-LOAD_RADIUS, LOAD_RADIUS + 1):
                for dx in range(-LOAD_RADIUS, LOAD_RADIUS + 1):
                    h_dist = dx * dx + dz * dz
                    for dy in range(VERTICAL_RADIUS, -VERTICAL_RADIUS - 1, -1):
                        ty = cy + dy
                        if ty < 0 or ty >= SECTION_COUNT:
                            continue
                        key = (cx + dx, ty, cz + dz)
                        if key in client.sent_chunks:
                            continue
                        if key in self.world.chunks:
                            candidates.append((h_dist, -ty, key))

            candidates.sort(key=lambda item: (item[0], item[1]))
            for _, _, key in candidates[:12]:
                chunk = self.world.chunks.get(key)
                if chunk is not None:
                    client.send(
                        {
                            "type": "chunk",
                            "cx": key[0],
                            "cy": key[1],
                            "cz": key[2],
                            "data": encode_chunk(chunk.data),
                        }
                    )
                    client.sent_chunks.add(key)

    def _broadcast_entity_spawn(self, entity):
        """Announce a newly spawned entity to every client.

        Called by :class:`~server.world.World` the moment an entity is added.
        Announcing at spawn time (rather than diffing the entity list once per
        tick) matters because entities can be created while handling packets or
        during fluid updates, i.e. outside the narrow window the old diff saw.
        Falling blocks were the visible symptom: the sand fell on the server but
        no client ever received ``spawn_falling_block``.
        """
        if not getattr(self, "clients", None):
            return
        if getattr(entity, "is_falling_block", False):
            self.broadcast(
                {
                    "type": "spawn_falling_block",
                    "id": entity.id,
                    "block": entity.block_id,
                    "pos": entity.pos,
                    "vel": entity.vel,
                }
            )
        elif getattr(entity, "is_mob", False):
            self.broadcast(
                {
                    "type": "spawn_mob",
                    "id": entity.id,
                    "mob_type": entity.mob_type,
                    "pos": entity.pos,
                    "vel": entity.vel,
                    "yaw": entity.yaw,
                    "pitch": entity.pitch,
                    "health": entity.health,
                }
            )
        elif hasattr(entity, "item"):
            self.broadcast(
                {
                    "type": "spawn_item",
                    "id": entity.id,
                    "item": entity.item,
                    "count": entity.count,
                    "pos": entity.pos,
                    "vel": entity.vel,
                    "pickup_delay": entity.pickup_delay,
                }
            )

    def broadcast(self, packet, exclude=None):
        for uuid, client in self.clients.items():
            if uuid != exclude and client.alive:
                client.send(packet)

    def _disconnect_client(self, client):
        self._save_player_data(client)
        client.close()
        self.open_furnaces.pop(client.uuid, None)
        if client in self.pending_clients:
            self.pending_clients.remove(client)
        username = client.username
        if client.uuid and client.uuid in self.clients:
            del self.clients[client.uuid]
            self.world.unregister_player(client.uuid)
            self.broadcast({"type": "player_leave", "uuid": client.uuid})
            if username:
                self.broadcast(
                    {
                        "type": "chat",
                        "sender": "",
                        "text": f"{username} left the game",
                        "color": [255, 255, 85, 255],
                    }
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Block Game Multiplayer Server")
    parser.add_argument("--server", action="store_true", help="Run dedicated server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on (default 25565)")
    parser.add_argument("--world", default="multiplayer", help="World name")
    parser.add_argument("--seed", type=int, default=None, help="World generation seed (random if omitted)")
    parser.add_argument("--gamemode", default="survival", choices=["survival", "creative"], help="Default server game mode (default: survival)")
    parser.add_argument("--saves-dir", default=None, help="Custom world storage directory")
    args = parser.parse_args(argv)

    server = MultiplayerServer(
        host=args.host,
        port=args.port,
        world_name=args.world,
        seed=args.seed,
        gamemode=args.gamemode,
        saves_dir=args.saves_dir,
    )
    server.start()
    print(f"Starting Block Game server on {args.host}:{server.actual_port} (World: {server.world_name}, Seed: {server.seed})...")
    try:
        while server.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
