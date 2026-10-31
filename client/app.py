import argparse
import math
import random
import sys
import tempfile
import threading
import time

import pyglet

pyglet.options["shadow_window"] = False

import moderngl
import numpy as np
from pyglet.gl import Config, glBindVertexArray, glGetError
from pyglet.window import NoSuchConfigException, key, mouse

from client.chat import ChatState
from client.geom import look_at, perspective
from client.inventory import CreativeInventory
from client.menu import LabelSpec, MenuState
from client.network import NetworkClient, decode_chunk, decode_skin
from client.player_render import (
    VIEW_DISTANCE,
    VIEW_FIRST,
    VIEW_THIRD_BACK,
    VIEW_THIRD_FRONT,
    selected_item_id,
)
from client.render import SKY, Renderer, chunk_origin
from client.texture import build_atlas, build_item_atlas
from server.commands import CommandContext, CommandHandler
from server.items import ITEMS
from server.servers import load_servers, parse_address, save_servers
from server.blocks import (
    AIR,
    BLOCKS,
    CHUNK_SIZE,
    HOTBAR_SLOTS,
    LOAD_RADIUS,
    SECTION_COUNT,
    VERTICAL_RADIUS,
    WORLD_HEIGHT,
    placement_block,
)
from server.stacks import (
    CRAFT_SLOTS,
    INVENTORY_SLOTS,
    add_to_inventory,
    empty_slots,
    empty_stack,
    make_stack,
    normalize_slots,
    normalize_stack,
    pack_stack,
    stack_count,
    stack_id,
)
from server.entity import FallingBlockEntity, ItemEntity, entity_from_dict, Mob, Pig, Sheep
from server.loot import calculate_break_time, get_block_drops
from server.mesh import build_chunk_mesh
from server.player import Player, look_direction
from server.profile import load_profile, load_skin_rgba, save_profile
from server.storage import WorldCatalog, display_path
from server.world import World
from client.settings import get_local_ips, load_settings, save_settings
from client.web_ui import WebCompanion


class LabelPool:
    def __init__(self):
        self.labels = []

    def draw(self, specs):
        while len(self.labels) < len(specs):
            self.labels.append(pyglet.text.Label(""))
        for label, spec in zip(self.labels, specs):
            label.text = spec.text
            label.x = spec.x
            label.y = spec.y
            label.font_size = spec.size
            label.anchor_x = spec.anchor_x
            label.anchor_y = spec.anchor_y
            color = spec.color
            if color and any(isinstance(v, float) for v in color):
                color = tuple(int(v * 255 if v <= 1.0 else v) for v in color)
            label.color = color
            label.draw()


class Game(pyglet.window.Window):
    def __init__(
        self,
        smoke=False,
        saves_dir=None,
        profile_dir=None,
        username=None,
        player_uuid=None,
        title=None,
        width=1280,
        height=720,
        window_pos=None,
        connect=None,
    ):
        self._smoke = smoke
        caption = title or ("Block Game" if not username else f"Block Game - {username}")
        display = dict(
            width=width,
            height=height,
            caption=caption,
            resizable=True,
            vsync=not smoke,
            visible=True,
        )
        config = Config(
            major_version=3,
            minor_version=3,
            forward_compatible=True,
            depth_size=24,
            double_buffer=True,
        )
        try:
            super().__init__(config=config, **display)
        except NoSuchConfigException:
            super().__init__(**display)
        self.set_minimum_size(min(width, 480), min(height, 360))
        if window_pos is not None:
            try:
                self.set_location(int(window_pos[0]), int(window_pos[1]))
            except Exception:
                pass

        self.ctx = moderngl.create_context()
        self.ctx.viewport = (0, 0, *self.get_framebuffer_size())
        atlas, size = build_atlas()
        item_atlas, item_size = build_item_atlas()
        self.renderer = Renderer(self.ctx, atlas, size, item_atlas, item_size)
        self.catalog = WorldCatalog(saves_dir)
        self.profile_dir = profile_dir
        self.profile = load_profile(profile_dir)
        self._custom_title = bool(title)
        if player_uuid:
            self.profile["uuid"] = str(player_uuid)
        if username:
            self.profile["username"] = str(username)
        elif "username" not in self.profile:
            self.profile["username"] = "Player"
        self.menu = MenuState()
        self.menu.username = self.profile.get("username", "Player")
        self.lock = threading.RLock()
        self.settings = load_settings(profile_dir)
        self.menu.web_ui_enabled = self.settings.get("web_ui_enabled", False)
        self.menu.web_ui_port = self.settings.get("web_ui_port", 5000)
        self.web_companion = None
        if self.settings.get("web_ui_enabled", False):
            self._start_web_companion()
        self.inventory = CreativeInventory()
        self.hotbar = empty_slots(HOTBAR_SLOTS)
        self.storage = empty_slots(INVENTORY_SLOTS)
        self.craft = empty_slots(CRAFT_SLOTS)
        self.menu.saves_hint = display_path(self.catalog.root)
        self.labels = LabelPool()
        self.status_label = pyglet.text.Label(
            "",
            font_size=22,
            anchor_x="center",
            anchor_y="center",
            color=(244, 240, 230, 255),
        )
        self.readout = pyglet.text.Label(
            "",
            x=16,
            y=self.height - 16,
            width=360,
            multiline=True,
            font_size=15,
            anchor_y="top",
            color=(255, 255, 255, 255),
        )
        self.world = None
        self.player = None
        self.info = None
        self.mode = "title"
        self.keys = set()
        self.selected = 0
        self.ready = False
        self.target = None
        self.eye = (0.0, 20.0, 0.0)
        self.camera = self.eye
        self.forward = look_direction(0.0, 0.0)
        self.view_mode = VIEW_FIRST
        self._play_time = 0.0
        self._mouse = (self.width / 2, self.height / 2)
        self._center = (0, 0, 0)
        self._gen_left = None
        self._mesh_queue = []
        self._generated = 0
        self._meshed = 0
        self._total = (LOAD_RADIUS * 2 + 1) ** 2 * (VERTICAL_RADIUS * 2 + 1)
        self._status = ""
        self._fresh_spawn = False
        self._autosave = 0.0
        self._text_ok = True
        self.network_client = None
        self.other_players = {}
        self._net_timer = 0.0
        self._closing = False
        self._client_picked_entity_ids = set()
        self.chat = ChatState()
        self.command_handler = CommandHandler()
        self.vision_mode = 0
        self._ignore_next_char = None
        self._chat_labels = []
        self._nametag_labels = []
        self.gamemode = "survival"
        self._mouse_held_left = False
        self._breaking_block = None
        self._break_time = 0.0
        self._break_required = 0.0
        self._break_progress = 0.0
        pyglet.clock.schedule_interval(self.tick, 1 / 60)
        if connect:
            self._open_server(connect)

    def tick(self, dt):
        if self.network_client is not None:
            self._network_tick(dt)
        self._play_time += min(dt, 0.05)
        now = time.monotonic()
        for other in self.other_players.values():
            if hasattr(other, "_move_speed"):
                if now - getattr(other, "_last_move_time", 0) > 0.1:
                    other._move_speed = 0.0
                    other.vel = [0.0, 0.0, 0.0]
        if self.mode == "loading":
            if self.network_client is None:
                self._bootstrap_step()
            elif self.world is not None:
                for key in self.world.take_dirty(8):
                    if key in self.world.chunks:
                        self._upload(key)
            return
        if self.mode not in ("play", "inventory") or not self.ready:
            return
        if self.mode == "play":
            self._apply_input()
            self.player.update(min(dt, 0.05))
        if self.network_client is None:
            self._stream()
            self._autosave += dt
            if self._autosave >= 8:
                self._autosave = 0
                self._persist_world()

        if self.world is not None:
            pos = self.player.pos if self.player is not None else None
            for key in self.world.take_dirty(4, player_pos=pos):
                if key in self.world.chunks:
                    self._upload(key)

        if self.world is not None:
            if self.network_client is None:
                self.world.advance_time(min(dt, 0.05))
            self.world.update_entities(min(dt, 0.05))
            if self.network_client is None:
                self.world.update_fluids(min(dt, 0.05))
                self.world.update_leaves(min(dt, 0.05))
                self.world.update_transformations(min(dt, 0.05))
                self.world.update_saplings(min(dt, 0.05))
                self.world.update_furnaces(min(dt, 0.05))
            if self.player is not None and self.mode == "play":
                px = self.player.pos[0]
                py = self.player.pos[1] + 0.8
                pz = self.player.pos[2]
                for entity in list(self.world.entities):
                    if not isinstance(entity, ItemEntity):
                        continue
                    if entity.pickup_delay <= 0 and entity.count > 0:
                        dx = entity.pos[0] - px
                        dy = entity.pos[1] - py
                        dz = entity.pos[2] - pz
                        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                        if dist < 2.5:
                            pull = (2.5 - dist) * 4.0 * min(dt, 0.05)
                            entity.vel[0] -= dx * pull
                            entity.vel[2] -= dz * pull
                        if dist <= 1.6:
                            leftover = add_to_inventory([self.hotbar, self.storage], entity.item, entity.count)
                            picked = entity.count - leftover
                            if picked > 0:
                                entity.count = leftover
                                if self.network_client is not None and entity.count <= 0:
                                    self._client_picked_entity_ids.add(entity.id)
                                    self.network_client.send_pickup_item(entity.id)
                                if entity.count <= 0:
                                    self.world.remove_entity(entity.id)

        if self.gamemode == "survival" and self.mode == "play":
            if self._mouse_held_left:
                self._aim()
                if self.target is None:
                    self._reset_breaking()
                else:
                    target_key = (self.target["x"], self.target["y"], self.target["z"], self.target.get("slab_half"))
                    if self._breaking_block != target_key:
                        self._start_breaking(self.target)
                    else:
                        self._break_time += dt
                        self._break_progress = min(1.0, self._break_time / max(0.001, self._break_required))
                        if self._break_time >= self._break_required:
                            self._break_block()
                            self._reset_breaking()
            else:
                self._reset_breaking()

    def _network_tick(self, dt):
        if self.network_client is None:
            return

        events = self.network_client.poll_events()
        for pkt in events:
            p_type = pkt.get("type")
            if p_type == "disconnect":
                reason = pkt.get("reason", "Server closed connection")
                self._handle_server_disconnect(reason)
                return
            elif p_type == "login_ack":
                self.gamemode = pkt.get("gamemode", "survival")
                self._reset_breaking()
                self.world = World(seed=pkt.get("seed", 0), day_time=pkt.get("day_time", 1000.0))
                self.world.spawner = None
                self.player = Player(self.world)
                skin = load_skin_rgba(self.profile["skin"])
                self.player.apply_identity(self.profile["uuid"], skin)
                self.renderer.set_skin(skin)
                pos = pkt.get("pos", [0.5, 30.0, 0.5])
                self.player.pos = list(pos)
                if "yaw" in pkt:
                    self.player.yaw = float(pkt["yaw"])
                if "pitch" in pkt:
                    self.player.pitch = float(pkt["pitch"])
                self.eye = (pos[0], pos[1] + self.player.eye, pos[2])
                self.camera = self.eye
                self.ready = True
                self.mode = "play"
                self.other_players.clear()
                for p in pkt.get("players", []):
                    self._add_remote_player(p)
                if "entities" in pkt:
                    self.world.entities.clear()
                    for ed in pkt["entities"]:
                        e = entity_from_dict(ed)
                        if getattr(e, "is_mob", False):
                            e.is_remote = True
                            e.target_pos = list(e.pos)
                            e.target_yaw = float(e.yaw)
                            e.target_pitch = float(e.pitch)
                        self.world.entities.append(e)
                if not self._smoke:
                    self.set_exclusive_mouse(True)
            elif p_type == "gamemode":
                self.gamemode = str(pkt.get("gamemode") or "survival").lower()
                self._reset_breaking()
            elif p_type == "player_join":
                self._add_remote_player(pkt)
            elif p_type == "player_move":
                u = pkt.get("uuid")
                other = self.other_players.get(u)
                if other is not None:
                    new_pos = list(pkt.get("pos", other.pos))
                    dx = new_pos[0] - other.pos[0]
                    dz = new_pos[2] - other.pos[2]
                    dist = math.hypot(dx, dz)
                    now = time.monotonic()
                    last_time = getattr(other, "_last_move_time", now)
                    dt_move = max(0.001, now - last_time)
                    other._last_move_time = now
                    other.pos = new_pos
                    other.vel = list(pkt.get("vel", other.vel))
                    other.yaw = float(pkt.get("yaw", other.yaw))
                    other.pitch = float(pkt.get("pitch", other.pitch))
                    other.held_item = str(pkt.get("held", other.held_item))
                    hspeed = math.hypot(other.vel[0], other.vel[2])
                    if dist > 0.02 and hspeed >= 0.1:
                        other._move_speed = dist / dt_move
                    elif dist > 0.02:
                        other._move_speed = dist / dt_move
                    else:
                        other._move_speed = 0.0
            elif p_type == "player_leave":
                self.other_players.pop(pkt.get("uuid"), None)
            elif p_type == "chunk":
                if self.world is not None:
                    cx = pkt.get("cx")
                    cy = pkt.get("cy")
                    cz = pkt.get("cz")
                    chunk_data = decode_chunk(pkt.get("data"))
                    from server.world import Chunk
                    ch = Chunk(cx, cy, cz)
                    ch.data[:, :, :] = chunk_data
                    self.world._remember(ch)
                    self._upload((cx, cy, cz))
                    self.world.dirty.discard((cx, cy, cz))
            elif p_type == "block_change":
                x = pkt.get("x")
                y = pkt.get("y")
                z = pkt.get("z")
                block = pkt.get("block")
                if self.world is not None:
                    self.world.set_block(x, y, z, block)
            elif p_type == "spawn_item":
                if self.world is not None:
                    if not any(e.id == pkt["id"] for e in self.world.entities):
                        self.world.entities.append(ItemEntity.from_dict(pkt))
            elif p_type == "spawn_falling_block":
                if self.world is not None:
                    if not any(e.id == pkt["id"] for e in self.world.entities):
                        self.world.entities.append(FallingBlockEntity.from_dict(pkt))
            elif p_type == "spawn_mob":
                if self.world is not None:
                    if not any(e.id == pkt["id"] for e in self.world.entities):
                        mob = entity_from_dict(pkt)
                        if getattr(mob, "is_mob", False):
                            mob.is_remote = True
                            mob.target_pos = list(mob.pos)
                            mob.target_yaw = float(mob.yaw)
                            mob.target_pitch = float(mob.pitch)
                        self.world.entities.append(mob)
            elif p_type == "mobs_update":
                if self.world is not None:
                    by_id = {e.id: e for e in self.world.entities if getattr(e, "is_mob", False)}
                    for md in pkt.get("mobs", []):
                        mid = md.get("id")
                        mob = by_id.get(mid)
                        if mob is not None:
                            mob.is_remote = True
                            mob.target_pos = list(md.get("pos", mob.pos))
                            mob.target_yaw = float(md.get("yaw", mob.yaw))
                            mob.target_pitch = float(md.get("pitch", mob.pitch))
                            mob.vel = list(md.get("vel", mob.vel))
                            mob.health = float(md.get("health", mob.health))
                            mob.hurt_time = float(md.get("hurt_time", mob.hurt_time))
                            if "sheared" in md:
                                mob.sheared = bool(md["sheared"])
                            if "color" in md:
                                mob.color = str(md["color"])
                        else:
                            mob = entity_from_dict(md)
                            if getattr(mob, "is_mob", False):
                                mob.is_remote = True
                                mob.target_pos = list(mob.pos)
                                mob.target_yaw = float(mob.yaw)
                                mob.target_pitch = float(mob.pitch)
                            self.world.entities.append(mob)
            elif p_type == "damage_mob":
                if self.world is not None:
                    mid = pkt.get("id")
                    mob = next((e for e in self.world.entities if e.id == mid), None)
                    if mob is not None:
                        mob.health = float(pkt.get("health", mob.health))
                        mob.hurt_time = float(pkt.get("hurt_time", 0.4))
                        if mob.health <= 0:
                            mob.alive = False
                        if "vel" in pkt:
                            mob.vel = list(pkt["vel"])
                        if "pos" in pkt:
                            mob.pos = list(pkt["pos"])
                            mob.target_pos = list(pkt["pos"])
            elif p_type == "give_item":
                eid = pkt.get("id")
                if eid is not None and eid in self._client_picked_entity_ids:
                    self._client_picked_entity_ids.discard(eid)
                else:
                    item = pkt.get("item")
                    count = int(pkt.get("count", 1))
                    if item and count > 0:
                        add_to_inventory([self.hotbar, self.storage], item, count)
            elif p_type == "remove_entity":
                eid = pkt.get("id")
                if self.world is not None:
                    self.world.remove_entity(eid)
                self._client_picked_entity_ids.discard(eid)
            elif p_type == "furnace_sync":
                f_data = pkt.get("furnace")
                if f_data and self.world is not None:
                    from server.stacks import normalize_stack
                    fx = int(f_data.get("x", 0))
                    fy = int(f_data.get("y", 0))
                    fz = int(f_data.get("z", 0))
                    furnace = self.world.get_or_create_furnace(fx, fy, fz, block_id=f_data.get("block_id"))
                    furnace.input = normalize_stack(f_data.get("input", furnace.input))
                    furnace.fuel = normalize_stack(f_data.get("fuel", furnace.fuel))
                    furnace.output = normalize_stack(f_data.get("output", furnace.output))
                    furnace.burn_time = float(f_data.get("burn_time", furnace.burn_time))
                    furnace.burn_duration = float(f_data.get("burn_duration", furnace.burn_duration))
                    furnace.cook_time = float(f_data.get("cook_time", furnace.cook_time))
                    furnace.cook_duration = float(f_data.get("cook_duration", furnace.cook_duration))
                    furnace.cook_speed = float(f_data.get("cook_speed", furnace.cook_speed))
                    if "recipe_types" in f_data:
                        furnace.recipe_types = tuple(f_data["recipe_types"])
                    if (
                        getattr(self.inventory, "container_type", None) == "furnace"
                        and self.inventory.furnace is not None
                        and (self.inventory.furnace.x, self.inventory.furnace.y, self.inventory.furnace.z) == (fx, fy, fz)
                    ):
                        self.inventory.furnace = furnace
            elif p_type == "furnace_progress":
                if self.world is not None:
                    fx = int(pkt.get("x", 0))
                    fy = int(pkt.get("y", 0))
                    fz = int(pkt.get("z", 0))
                    furnace = self.world.get_furnace(fx, fy, fz)
                    if furnace is not None:
                        if "burn_time" in pkt:
                            furnace.burn_time = float(pkt["burn_time"])
                        if "burn_duration" in pkt:
                            furnace.burn_duration = float(pkt["burn_duration"])
                        if "cook_time" in pkt:
                            furnace.cook_time = float(pkt["cook_time"])
                        if "cook_duration" in pkt:
                            furnace.cook_duration = float(pkt["cook_duration"])
                        if (
                            getattr(self.inventory, "container_type", None) == "furnace"
                            and self.inventory.furnace is not None
                            and (self.inventory.furnace.x, self.inventory.furnace.y, self.inventory.furnace.z) == (fx, fy, fz)
                        ):
                            self.inventory.furnace = furnace
            elif p_type == "chat":
                self.chat.add_message(
                    pkt.get("text", ""),
                    sender=pkt.get("sender", ""),
                    color=pkt.get("color"),
                )
            elif p_type == "time_update":
                if self.world is not None and "day_time" in pkt:
                    self.world.day_time = float(pkt["day_time"])

        if not self.network_client.connected:
            err = self.network_client.error or "Server closed connection"
            self._handle_server_disconnect(err)
            return

        if self.ready and self.player is not None:
            self._net_timer += dt
            if self._net_timer >= 0.04:
                self._net_timer = 0.0
                held = selected_item_id(self.hotbar, self.selected)
                self.network_client.send_move(
                    self.player.pos,
                    self.player.vel,
                    self.player.yaw,
                    self.player.pitch,
                    held,
                )

    def _handle_server_disconnect(self, message="Connection lost"):
        was_loading = (self.mode == "loading")
        if self.network_client is not None:
            self.network_client.disconnect()
            self.network_client = None
        self.other_players.clear()
        self._drop_view()
        if not self._smoke:
            try:
                self.set_exclusive_mouse(False)
            except Exception:
                pass
        self.keys.clear()
        self.mode = "servers"
        self.menu.screen = "servers"
        self.menu.servers = load_servers()
        self.menu.selected = 0 if self.menu.servers else -1
        self.menu.scroll = 0
        prefix = "Failed to connect" if was_loading else "Disconnected"
        self.menu.message = f"{prefix}: {message}"

    def _add_remote_player(self, data):
        u = data.get("uuid")
        if not u or u == self.profile["uuid"]:
            return
        skin = decode_skin(data.get("skin"))
        remote = Player(self.world)
        remote.apply_identity(u, skin)
        remote.pos = list(data.get("pos", [0.5, 30.0, 0.5]))
        remote.yaw = float(data.get("yaw", 0.0))
        remote.pitch = float(data.get("pitch", 0.0))
        remote.username = str(data.get("username", "Player"))
        remote.held_item = str(data.get("held", ""))
        remote._move_speed = 0.0
        remote._last_move_time = time.monotonic()
        self.other_players[u] = remote
        if skin is not None:
            self.renderer.set_skin(skin, player_id=u)

    def _bootstrap_step(self):
        if self.player is None or self.world is None:
            return
        if self._gen_left is None:
            cx = math.floor(self.player.pos[0] / CHUNK_SIZE)
            cy = min(SECTION_COUNT - 1, max(0, math.floor(self.player.pos[1] / CHUNK_SIZE)))
            cz = math.floor(self.player.pos[2] / CHUNK_SIZE)
            self._center = (cx, cy, cz)
            self._total = self.world.load_span(cx, cy, cz)
            self._gen_left = self._total
            self._generated = 0
            self._meshed = 0
            self._mesh_queue = []
        if self._gen_left:
            built, _removed = self.world.sync(*self._center)
            if built is None:
                self._gen_left = 0
            else:
                self._mesh_queue.append(built)
                self._generated += 1
                self._gen_left -= 1
                self._status = f"Generating terrain  {self._generated}/{self._total}"
            return
        if self._mesh_queue:
            key = self._mesh_queue.pop(0)
            self._upload(key)
            self.world.dirty.discard(key)
            self._meshed += 1
            self._status = f"Building the world  {self._meshed}/{self._total}"
            return
        if self.world.dirty:
            key = self.world.dirty.pop()
            if key in self.world.chunks:
                self._upload(key)
            return
        if self._fresh_spawn:
            self.player.spawn(math.floor(self.player.pos[0]), math.floor(self.player.pos[2]))
        self.ready = True
        self.mode = "play"
        self._persist_world()
        if not self._smoke:
            self.set_exclusive_mouse(True)

    def _stream(self):
        cx = math.floor(self.player.pos[0] / CHUNK_SIZE)
        cy = min(SECTION_COUNT - 1, max(0, math.floor(self.player.pos[1] / CHUNK_SIZE)))
        cz = math.floor(self.player.pos[2] / CHUNK_SIZE)
        built, removed = self.world.sync(cx, cy, cz)
        for key in removed:
            self.renderer.drop(key)
        if built is not None:
            self._upload(built)
            self.world.dirty.discard(built)

    def _upload(self, key):
        mesh = build_chunk_mesh(self.world, key[0], key[1], key[2])
        self.renderer.set_chunk(key, mesh, chunk_origin(key))

    def _upload_block_change(self, x, y, z):
        cx = x // CHUNK_SIZE
        cy = y // CHUNK_SIZE
        cz = z // CHUNK_SIZE
        lx = x % CHUNK_SIZE
        ly = y % CHUNK_SIZE
        lz = z % CHUNK_SIZE
        affected = {(cx, cy, cz)}
        if lx == 0:
            affected.add((cx - 1, cy, cz))
        elif lx == CHUNK_SIZE - 1:
            affected.add((cx + 1, cy, cz))
        if ly == 0 and cy > 0:
            affected.add((cx, cy - 1, cz))
        elif ly == CHUNK_SIZE - 1 and cy < SECTION_COUNT - 1:
            affected.add((cx, cy + 1, cz))
        if lz == 0:
            affected.add((cx, cy, cz - 1))
        elif lz == CHUNK_SIZE - 1:
            affected.add((cx, cy, cz + 1))
        for k in affected:
            if k in self.world.chunks:
                self._upload(k)
                self.world.dirty.discard(k)
        adjacent_dirty = [
            k for k in self.world.dirty
            if abs(k[0] - cx) <= 1 and abs(k[1] - cy) <= 1 and abs(k[2] - cz) <= 1
        ]
        for k in adjacent_dirty:
            if k in self.world.chunks:
                self._upload(k)
                self.world.dirty.discard(k)

    def _apply_input(self):
        inp = self.player.input
        if self.chat.open:
            inp.forward = False
            inp.back = False
            inp.left = False
            inp.right = False
            inp.jump = False
            inp.sprint = False
            inp.sneak = False
            inp.down = False
            return
        inp.forward = key.W in self.keys
        inp.back = key.S in self.keys
        inp.left = key.A in self.keys
        inp.right = key.D in self.keys
        inp.jump = key.SPACE in self.keys
        inp.sprint = key.LSHIFT in self.keys or key.RSHIFT in self.keys
        inp.sneak = (
            key.LSHIFT in self.keys
            or key.RSHIFT in self.keys
            or key.LCTRL in self.keys
            or key.RCTRL in self.keys
            or key.C in self.keys
        )
        inp.down = inp.sneak

    def _clear_input(self):
        self.keys.clear()
        if self.player is not None and self.mode == "play":
            self._apply_input()

    def _aim(self):
        if self.player is None:
            self.target = None
            return
        ex, ey, ez = self.player.pos
        ey += self.player.eye
        self.eye = (ex, ey, ez)
        self.forward = look_direction(self.player.yaw, self.player.pitch)
        self.camera = self._camera_from_view()
        if self.mode == "play":
            self.target = self.world.raycast(*self.eye, *self.forward, 6)
        else:
            self.target = None

    def _camera_from_view(self):
        if self.view_mode == VIEW_FIRST or self.player is None:
            return self.eye
        fx, fy, fz = self.forward
        if self.view_mode == VIEW_THIRD_BACK:
            rx = -fx * VIEW_DISTANCE
            ry = -fy * VIEW_DISTANCE + 0.4
            rz = -fz * VIEW_DISTANCE
        else:
            rx = fx * VIEW_DISTANCE
            ry = fy * VIEW_DISTANCE + 0.4
            rz = fz * VIEW_DISTANCE

        if self.world is None:
            return (self.eye[0] + rx, self.eye[1] + ry, self.eye[2] + rz)

        max_dist = math.sqrt(rx * rx + ry * ry + rz * rz)
        if max_dist < 1e-4:
            return self.eye

        dx = rx / max_dist
        dy = ry / max_dist
        dz = rz / max_dist

        ox, oy, oz = self.eye
        margin = 0.2
        min_allowed = max_dist

        # Sample rays from center and 8 near-frustum offsets
        offsets = (
            (0.0, 0.0, 0.0),
            (-0.1, -0.1, -0.1),
            (-0.1, -0.1, 0.1),
            (-0.1, 0.1, -0.1),
            (-0.1, 0.1, 0.1),
            (0.1, -0.1, -0.1),
            (0.1, -0.1, 0.1),
            (0.1, 0.1, -0.1),
            (0.1, 0.1, 0.1),
        )
        for sx, sy, sz in offsets:
            rox = ox + sx
            roy = oy + sy
            roz = oz + sz
            if self.world.is_solid(math.floor(rox), math.floor(roy), math.floor(roz)):
                continue
            hit = self.world.raycast(rox, roy, roz, dx, dy, dz, max_dist, solid_only=True)
            if hit is not None and isinstance(hit, dict):
                dist_val = float(hit.get("dist", max_dist))
                allowed = max(0.0, dist_val - margin)
                if allowed < min_allowed:
                    min_allowed = allowed

        cam_x = ox + dx * min_allowed
        cam_y = oy + dy * min_allowed
        cam_z = oz + dz * min_allowed

        # Pull forward out of solid blocks if still inside any
        step = 0.05
        while min_allowed > 0.0 and self.world.is_solid(math.floor(cam_x), math.floor(cam_y), math.floor(cam_z)):
            min_allowed = max(0.0, min_allowed - step)
            cam_x = ox + dx * min_allowed
            cam_y = oy + dy * min_allowed
            cam_z = oz + dz * min_allowed

        return (cam_x, cam_y, cam_z)

    def _cycle_view(self):
        self.view_mode = (self.view_mode + 1) % 3

    def _reset_breaking(self):
        self._breaking_block = None
        self._break_time = 0.0
        self._break_required = 0.0
        self._break_progress = 0.0

    def _start_breaking(self, target_or_pos):
        if isinstance(target_or_pos, dict):
            target = target_or_pos
            x, y, z = target["x"], target["y"], target["z"]
            target_half = target.get("slab_half")
        elif len(target_or_pos) == 4:
            x, y, z, target_half = target_or_pos
            target = {"x": x, "y": y, "z": z, "slab_half": target_half}
        else:
            x, y, z = target_or_pos
            target_half = None
            target = {"x": x, "y": y, "z": z}

        target_key = (x, y, z, target_half)
        block_id = self.world.get_block(x, y, z)
        if not block_id or block_id == AIR:
            self._reset_breaking()
            return

        from server.blocks import get_offset_info, get_offset_torch_info, get_slab_info, is_slab

        o_info = get_offset_info(block_id)
        t_info = get_offset_torch_info(block_id)
        s_info = get_slab_info(block_id)
        if o_info is not None and o_info.get("capped"):
            half = target_half or ("top" if target.get("local_y", 0.5) >= 0.5 else "offset")
            sub_id = o_info["top_slab"] if half == "top" else o_info["base_id"]
            block = BLOCKS.get(sub_id) or BLOCKS.get(block_id)
        elif t_info is not None and t_info.get("capped"):
            half = target_half or ("top" if target.get("local_y", 0.5) >= 0.5 else "offset_torch")
            sub_id = t_info["top_slab"] if half == "top" else t_info["base_id"]
            block = BLOCKS.get(sub_id) or BLOCKS.get(block_id)
        elif s_info is not None and s_info.get("double"):
            half = target_half or ("top" if target.get("local_y", 0.5) >= 0.5 else "bottom")
            sub_id = s_info["top"] if half == "top" else s_info["bottom"]
            block = BLOCKS.get(sub_id) or BLOCKS.get(block_id)
        else:
            block = BLOCKS.get(block_id)

        held_id = stack_id(self.hotbar[self.selected]) if self.selected < len(self.hotbar) else ""
        held_item = ITEMS.get(held_id)
        req_time = calculate_break_time(block, held_item)
        if req_time <= 0.0:
            self._break_block()
            self._reset_breaking()
        else:
            self._breaking_block = target_key
            self._break_time = 0.0
            self._break_required = req_time
            self._break_progress = 0.0

    def _break_block(self):
        self._aim()
        hit = self.target
        if hit is None:
            return
        x, y, z = hit["x"], hit["y"], hit["z"]
        block_id = self.world.get_block(x, y, z)
        if not block_id or block_id == AIR:
            return

        from server.blocks import get_offset_info, get_offset_torch_info, get_slab_info, is_slab

        o_info = get_offset_info(block_id)
        t_info = get_offset_torch_info(block_id)
        if (o_info is not None and o_info.get("capped")) or (t_info is not None and t_info.get("capped")):
            cap_info = o_info if (o_info is not None and o_info.get("capped")) else t_info
            target_half = hit.get("slab_half") or ("top" if hit.get("local_y", 0.5) >= 0.5 else "offset")
            base_id = cap_info["base_id"]
            top_slab = cap_info["top_slab"]
            if target_half == "top":
                broken_item = top_slab
                remaining_block = f"{base_id}_offset"
                drop_y = y + 0.75
            else:
                broken_item = base_id
                remaining_block = f"{top_slab}_top"
                drop_y = y + 0.0

            if self.gamemode == "survival":
                held_id = stack_id(self.hotbar[self.selected]) if self.selected < len(self.hotbar) else ""
                held_item = ITEMS.get(held_id)
                drops = get_block_drops(broken_item, held_item)
                for drop_id, count in drops:
                    vx = random.uniform(-0.8, 0.8)
                    vy = random.uniform(2.2, 3.8)
                    vz = random.uniform(-0.8, 0.8)
                    pos = [x + 0.5, drop_y, z + 0.5]
                    self.world.spawn_item(drop_id, count=count, pos=pos, vel=[vx, vy, vz], pickup_delay=0.3)
                    if self.network_client is not None:
                        self.network_client.send_drop_item(drop_id, count, pos, [vx, vy, vz], pickup_delay=0.3)

            if self.network_client is not None:
                self.network_client.send_set_block(x, y, z, remaining_block)
            if self.world.set_block(x, y, z, remaining_block):
                self._upload_block_change(x, y, z)
            return

        s_info = get_slab_info(block_id)
        if s_info is not None and s_info.get("double"):
            target_half = hit.get("slab_half") or ("top" if hit.get("local_y", 0.5) >= 0.5 else "bottom")
            bottom_id = s_info["bottom"]
            top_id = s_info["top"]
            if target_half == "top":
                broken_item = top_id
                remaining_block = bottom_id
                drop_y = y + 0.75
            else:
                broken_item = bottom_id
                remaining_block = f"{top_id}_top"
                drop_y = y + 0.25

            if self.gamemode == "survival":
                held_id = stack_id(self.hotbar[self.selected]) if self.selected < len(self.hotbar) else ""
                held_item = ITEMS.get(held_id)
                drops = get_block_drops(broken_item, held_item)
                for drop_id, count in drops:
                    vx = random.uniform(-0.8, 0.8)
                    vy = random.uniform(2.2, 3.8)
                    vz = random.uniform(-0.8, 0.8)
                    pos = [x + 0.5, drop_y, z + 0.5]
                    self.world.spawn_item(drop_id, count=count, pos=pos, vel=[vx, vy, vz], pickup_delay=0.3)
                    if self.network_client is not None:
                        self.network_client.send_drop_item(drop_id, count, pos, [vx, vy, vz], pickup_delay=0.3)

            if self.network_client is not None:
                self.network_client.send_set_block(x, y, z, remaining_block)
            if self.world.set_block(x, y, z, remaining_block):
                self._upload_block_change(x, y, z)
            return

        next_block = AIR
        if o_info is not None:
            broken_id = o_info["base_id"]
        elif s_info is not None:
            broken_id = s_info["base_id"]
            if s_info.get("waterlogged"):
                from server.fluid import WATER_LEVEL_BLOCKS

                lvl = s_info.get("level", 8)
                if lvl >= 8:
                    next_block = "water"
                elif lvl in WATER_LEVEL_BLOCKS:
                    next_block = WATER_LEVEL_BLOCKS[lvl]
                else:
                    next_block = AIR
        else:
            broken_id = block_id

        if self.gamemode == "survival":
            held_id = stack_id(self.hotbar[self.selected]) if self.selected < len(self.hotbar) else ""
            held_item = ITEMS.get(held_id)
            drops = get_block_drops(broken_id, held_item)
            for drop_id, count in drops:
                vx = random.uniform(-0.8, 0.8)
                vy = random.uniform(2.2, 3.8)
                vz = random.uniform(-0.8, 0.8)
                pos = [x + 0.5, y + 0.3, z + 0.5]
                self.world.spawn_item(drop_id, count=count, pos=pos, vel=[vx, vy, vz], pickup_delay=0.3)
                if self.network_client is not None:
                    self.network_client.send_drop_item(drop_id, count, pos, [vx, vy, vz], pickup_delay=0.3)

        if self.network_client is not None:
            self.network_client.send_set_block(x, y, z, next_block)
        if self.world.set_block(x, y, z, next_block):
            self._upload_block_change(x, y, z)

    def _drop_held_item(self):
        if self.player is None or self.world is None:
            return
        if self.selected >= len(self.hotbar):
            return
        slot = self.hotbar[self.selected]
        item_id = stack_id(slot)
        count = stack_count(slot)
        if not item_id or count <= 0:
            return

        if count > 1:
            self.hotbar[self.selected] = make_stack(item_id, count - 1)
        else:
            self.hotbar[self.selected] = empty_stack()

        px = self.player.pos[0]
        py = self.player.pos[1] + self.player.eye - 0.2
        pz = self.player.pos[2]
        fx, fy, fz = self.forward
        vx = fx * 5.5 + self.player.vel[0] * 0.4
        vy = fy * 5.5 + 1.2
        vz = fz * 5.5 + self.player.vel[2] * 0.4
        pos = [px, py, pz]
        vel = [vx, vy, vz]
        self.world.spawn_item(item_id, count=1, pos=pos, vel=vel, pickup_delay=1.0)
        if self.network_client is not None:
            self.network_client.send_drop_item(item_id, 1, pos, vel, pickup_delay=1.0)

    def _place_block(self):
        self._aim()
        hit = self.target
        if hit is None:
            return

        held = ITEMS.get(stack_id(self.hotbar[self.selected]) if self.selected < len(self.hotbar) else "")
        if held is None or held.right != "place" or not held.block:
            return

        from server.blocks import (
            can_place_block,
            get_capped_offset_block,
            get_capped_offset_torch,
            get_double_slab,
            get_offset_block,
            get_offset_info,
            get_offset_torch,
            get_offset_torch_info,
            get_slab_info,
            has_half_offset_top,
            is_offset_block,
            is_offset_torch,
            is_slab,
            is_torch,
            placement_block,
        )
        from server.fluid import is_water

        held_is_slab = is_slab(held.block)

        # 1. Try combining with targeted block if it's a single slab or an uncapped offset block
        target_pos = (hit["x"], hit["y"], hit["z"])
        target_block = self.world.get_block(*target_pos)
        target_slab_info = get_slab_info(target_block)
        target_offset_info = get_offset_info(target_block)

        if held_is_slab and target_offset_info is not None and not target_offset_info["capped"]:
            tx, ty, tz = target_pos
            above_target = self.world.get_block(tx, ty + 1, tz)
            if hit["ny"] == 1 and not is_offset_block(above_target) and not is_offset_torch(above_target):
                capped_id = get_capped_offset_block(target_block, held.block)
                if capped_id is not None:
                    if self.player is not None:
                        px, py, pz = self.player.pos
                        rad, hgt = self.player.radius, self.player.height
                        if (
                            px + rad > tx
                            and px - rad < tx + 1
                            and py + hgt > ty + 0.5
                            and py < ty + 1.0
                            and pz + rad > tz
                            and pz - rad < tz + 1
                        ):
                            return
                    if self.network_client is not None:
                        self.network_client.send_set_block(tx, ty, tz, capped_id)
                    if self.world.set_block(tx, ty, tz, capped_id):
                        self._upload_block_change(tx, ty, tz)
                        if self.gamemode == "survival":
                            cur_slot = self.hotbar[self.selected]
                            c = stack_count(cur_slot)
                            if c > 1:
                                self.hotbar[self.selected] = make_stack(stack_id(cur_slot), c - 1)
                            else:
                                self.hotbar[self.selected] = empty_stack()
                        return

        if held_is_slab and target_slab_info is not None and not target_slab_info["double"]:
            tx, ty, tz = target_pos
            above_target = self.world.get_block(tx, ty + 1, tz)
            if not is_offset_block(above_target) and not is_offset_torch(above_target):
                can_combine = False
                double_id = None
                if target_slab_info["half"] == "bottom":
                    if hit["ny"] == 1 or (hit["ny"] == 0 and hit.get("local_y", 0.0) >= 0.5):
                        double_id = get_double_slab(target_slab_info["base_id"], held.block)
                        can_combine = True
                elif target_slab_info["half"] == "top":
                    if hit["ny"] == -1 or (hit["ny"] == 0 and hit.get("local_y", 1.0) < 0.5):
                        double_id = get_double_slab(held.block, target_slab_info["base_id"])
                        can_combine = True

                if can_combine and double_id is not None:
                    if self.player is not None and self.player.overlaps_block(tx, ty, tz):
                        return
                    if self.network_client is not None:
                        self.network_client.send_set_block(tx, ty, tz, double_id)
                    if self.world.set_block(tx, ty, tz, double_id):
                        self._upload_block_change(tx, ty, tz)
                        if self.gamemode == "survival":
                            cur_slot = self.hotbar[self.selected]
                            c = stack_count(cur_slot)
                            if c > 1:
                                self.hotbar[self.selected] = make_stack(stack_id(cur_slot), c - 1)
                            else:
                                self.hotbar[self.selected] = empty_stack()
                        return

        # 2. Placing into the adjacent block (or combining with existing slab / uncapped offset block there)
        x = hit["x"] + hit["nx"]
        y = hit["y"] + hit["ny"]
        z = hit["z"] + hit["nz"]
        place_local_y = hit.get("local_y")
        if held_is_slab and hit["ny"] == 0 and hit.get("slab_half") == "offset" and (place_local_y or 0.0) < 0.0:
            y -= 1
            place_local_y = (place_local_y or 0.0) + 1.0

        adj_block = self.world.get_block(x, y, z)
        adj_slab_info = get_slab_info(adj_block)
        adj_offset_info = get_offset_info(adj_block)

        if held_is_slab and adj_offset_info is not None and not adj_offset_info["capped"]:
            above_adj = self.world.get_block(x, y + 1, z)
            if not is_offset_block(above_adj) and not is_offset_torch(above_adj):
                capped_id = get_capped_offset_block(adj_block, held.block)
                if capped_id is not None:
                    if self.player is not None:
                        px, py, pz = self.player.pos
                        rad, hgt = self.player.radius, self.player.height
                        if (
                            px + rad > x
                            and px - rad < x + 1
                            and py + hgt > y + 0.5
                            and py < y + 1.0
                            and pz + rad > z
                            and pz - rad < z + 1
                        ):
                            return
                    if self.network_client is not None:
                        self.network_client.send_set_block(x, y, z, capped_id)
                    if self.world.set_block(x, y, z, capped_id):
                        self._upload_block_change(x, y, z)
                        if self.gamemode == "survival":
                            cur_slot = self.hotbar[self.selected]
                            c = stack_count(cur_slot)
                            if c > 1:
                                self.hotbar[self.selected] = make_stack(stack_id(cur_slot), c - 1)
                            else:
                                self.hotbar[self.selected] = empty_stack()
                        return

        if held_is_slab and adj_slab_info is not None and not adj_slab_info["double"]:
            above_adj = self.world.get_block(x, y + 1, z)
            if not is_offset_block(above_adj) and not is_offset_torch(above_adj):
                double_id = None
                if adj_slab_info["half"] == "bottom":
                    double_id = get_double_slab(adj_slab_info["base_id"], held.block)
                elif adj_slab_info["half"] == "top":
                    double_id = get_double_slab(held.block, adj_slab_info["base_id"])
                if double_id is not None:
                    if self.player is not None and self.player.overlaps_block(x, y, z):
                        return
                    if self.network_client is not None:
                        self.network_client.send_set_block(x, y, z, double_id)
                    if self.world.set_block(x, y, z, double_id):
                        self._upload_block_change(x, y, z)
                        if self.gamemode == "survival":
                            cur_slot = self.hotbar[self.selected]
                            c = stack_count(cur_slot)
                            if c > 1:
                                self.hotbar[self.selected] = make_stack(stack_id(cur_slot), c - 1)
                            else:
                                self.hotbar[self.selected] = empty_stack()
                        return

        player_yaw = self.player.yaw if self.player is not None else None
        block_to_place = placement_block(
            held.block,
            (hit["nx"], hit["ny"], hit["nz"]),
            yaw=player_yaw,
            local_y=place_local_y,
        )

        if is_torch(block_to_place) and hit["ny"] == 1 and has_half_offset_top(target_block):
            offset_torch_id = get_offset_torch(block_to_place)
            if offset_torch_id is not None:
                block_to_place = offset_torch_id

        adj_torch_info = get_offset_torch_info(adj_block)
        if held_is_slab and block_to_place.endswith("_top") and adj_torch_info is not None and not adj_torch_info["capped"]:
            above_adj = self.world.get_block(x, y + 1, z)
            if not is_offset_block(above_adj) and not is_offset_torch(above_adj):
                capped_t = get_capped_offset_torch(adj_block, held.block)
                if capped_t is not None:
                    block_to_place = capped_t
        elif (
            is_offset_torch(block_to_place)
            and adj_slab_info is not None
            and not adj_slab_info["double"]
            and adj_slab_info["half"] == "top"
        ):
            capped_t = get_capped_offset_torch(block_to_place, adj_block)
            if capped_t is not None:
                block_to_place = capped_t

        # If placing a slab beneath an offset block (clicking underside ny == -1), place a bottom slab
        if held_is_slab and hit["ny"] == -1 and is_offset_block(self.world.get_block(x, y + 1, z)):
            s_base = get_slab_info(held.block)
            if s_base is not None:
                block_to_place = s_base["base_id"]

        # If placing a full solid block on top of a bottom slab or an uncapped offset block
        # (or adjacent to an offset block where the space below is half-offset/open), place it offset by half height
        if not held_is_slab:
            offset_candidate = get_offset_block(block_to_place)
            if offset_candidate is not None:
                below_block = self.world.get_block(x, y - 1, z)
                should_offset = False
                if hit["ny"] == 1 and has_half_offset_top(target_block):
                    should_offset = True
                elif has_half_offset_top(below_block):
                    should_offset = True
                elif (
                    hit["ny"] == 0
                    and hit.get("slab_half") == "offset"
                    and (below_block in ("", AIR, None) or not self.world.is_solid(x, y - 1, z))
                ):
                    should_offset = True
                if should_offset:
                    block_to_place = offset_candidate

        # Check waterlogging if placing into water
        if held_is_slab and is_water(adj_block):
            if block_to_place.endswith("_top"):
                base_b = block_to_place[:-4]
                block_to_place = f"{base_b}_top_waterlogged"
            else:
                block_to_place = f"{block_to_place}_waterlogged"

        b_obj = BLOCKS.get(block_to_place)
        is_solid = b_obj is not None and b_obj.solid
        if is_solid:
            if hasattr(self.world, "collides_box") and self.player is not None:
                px, py, pz = self.player.pos
                rad = self.player.radius
                hgt = self.player.height
                p_min_x, p_max_x = px - rad, px + rad
                p_min_y, p_max_y = py, py + hgt
                p_min_z, p_max_z = pz - rad, pz + rad

                slab_info = get_slab_info(block_to_place)
                offset_info = get_offset_info(block_to_place)
                torch_info = get_offset_torch_info(block_to_place)
                if torch_info is not None and torch_info["capped"]:
                    b_boxes = [(float(x), float(y) + 0.5, float(z), float(x + 1), float(y) + 1.0, float(z + 1))]
                elif offset_info is not None:
                    b_boxes = [(float(x), float(y) - 0.5, float(z), float(x + 1), float(y) + 0.5, float(z + 1))]
                elif slab_info is not None and slab_info["half"] == "bottom":
                    b_boxes = [(float(x), float(y), float(z), float(x + 1), float(y) + 0.5, float(z + 1))]
                elif slab_info is not None and slab_info["half"] == "top":
                    b_boxes = [(float(x), float(y) + 0.5, float(z), float(x + 1), float(y) + 1, float(z + 1))]
                else:
                    b_boxes = [(float(x), float(y), float(z), float(x + 1), float(y + 1), float(z + 1))]

                collides = False
                for bx0, by0, bz0, bx1, by1, bz1 in b_boxes:
                    if p_max_x > bx0 and p_min_x < bx1 and p_max_y > by0 and p_min_y < by1 and p_max_z > bz0 and p_min_z < bz1:
                        collides = True
                        break
                if collides:
                    return
            elif self.player.overlaps_block(x, y, z):
                return

        placing_capped_torch = (t_place_info := get_offset_torch_info(block_to_place)) is not None and t_place_info["capped"]
        if not placing_capped_torch and (self.world.is_solid(x, y, z) or is_torch(self.world.get_block(x, y, z))):
            return

        from server.blocks import can_place_block

        if not can_place_block(self.world, x, y, z, block_to_place, hit=hit):
            return
        if self.network_client is not None:
            self.network_client.send_set_block(x, y, z, block_to_place)
        if self.world.set_block(x, y, z, block_to_place):
            self._upload_block_change(x, y, z)
            if self.gamemode == "survival":
                cur_slot = self.hotbar[self.selected]
                c = stack_count(cur_slot)
                if c > 1:
                    self.hotbar[self.selected] = make_stack(stack_id(cur_slot), c - 1)
                else:
                    self.hotbar[self.selected] = empty_stack()

    def _open_world(self, info):
        self._drop_view()
        self.info = info
        self.gamemode = getattr(info, "gamemode", "survival")
        self._reset_breaking()
        self.world = World(
            seed=info.seed,
            store=self.catalog.store_for(info.id),
            day_time=getattr(info, "day_time", 1000.0),
            spawn_point=getattr(info, "spawn", None),
        )
        self.player = Player(self.world)
        skin = load_skin_rgba(self.profile["skin"])
        self.player.apply_identity(self.profile["uuid"], skin)
        self.world.register_player(self.player)
        self.renderer.set_skin(skin)
        self.inventory.clear_held()
        self.inventory.scroll = 0
        self.inventory.hover = None
        self.view_mode = VIEW_FIRST
        self._play_time = 0.0
        saved = info.player or None
        if saved:
            self.player.pos[:] = [saved["x"], saved["y"], saved["z"]]
            self.player.yaw = saved.get("yaw", 0.0)
            self.player.pitch = saved.get("pitch", 0.0)
            self.hotbar = normalize_slots(saved.get("hotbar"), HOTBAR_SLOTS)
            self.storage = normalize_slots(saved.get("inventory"), INVENTORY_SLOTS)
            self.craft = normalize_slots(saved.get("craft"), CRAFT_SLOTS)
            carried = normalize_stack(saved.get("carried"))
            if stack_id(carried):
                self.inventory.held = stack_id(carried)
                self.inventory.held_count = stack_count(carried)
                self.inventory.stash_held(self.hotbar, self.storage, self.craft)
            self.selected = int(saved.get("slot", 0)) % HOTBAR_SLOTS
            self._fresh_spawn = False
        else:
            self.hotbar = empty_slots(HOTBAR_SLOTS)
            self.storage = empty_slots(INVENTORY_SLOTS)
            self.craft = empty_slots(CRAFT_SLOTS)
            self.selected = 0
            sp_x, sp_y, sp_z = self.world.get_spawn_point()
            self.player.spawn(int(math.floor(sp_x)), int(math.floor(sp_z)))
            self.player.pos[:] = [float(sp_x), max(float(self.player.pos[1]), float(sp_y)), float(sp_z)]
            self.player.yaw = 0.6
            self._fresh_spawn = True
        self.ready = False
        self._gen_left = None
        self._mesh_queue = []
        self.target = None
        self._autosave = 0.0
        self.keys.clear()
        self.mode = "loading"
        self._status = "Generating terrain"

    def _persist_world(self):
        if self.world is None or self.info is None or self.player is None:
            return
        self.world.flush()
        if hasattr(self.world, "day_time"):
            self.info.day_time = float(self.world.day_time)
        if hasattr(self.world, "get_spawn_point"):
            self.info.spawn = list(self.world.get_spawn_point())
        self.info.player = {
            "x": float(self.player.pos[0]),
            "y": float(self.player.pos[1]),
            "z": float(self.player.pos[2]),
            "yaw": float(self.player.yaw),
            "pitch": float(self.player.pitch),
            "slot": int(self.selected),
            "hotbar": [pack_stack(slot) for slot in self.hotbar],
            "inventory": [pack_stack(slot) for slot in self.storage],
            "craft": [pack_stack(slot) for slot in self.craft],
            "carried": pack_stack(make_stack(self.inventory.held, self.inventory.held_count))
            if self.inventory.held and not self.inventory.held_infinite
            else "",
        }
        self.info.last_played = time.time()
        self.catalog.write(self.info)

    def _submit_chat(self, line):
        line = line.strip()
        if not line:
            return
        if self.network_client is not None:
            self.network_client.send_chat(line)
        else:
            if line.startswith("/"):
                ctx = CommandContext(
                    sender="Player",
                    player=self.player,
                    world=self.world,
                )
                ctx.app = self
                self.command_handler.execute(ctx, line)
                for reply_text, color in ctx.responses:
                    self.chat.add_message(reply_text, sender="", color=color)
            else:
                self.chat.add_message(line, sender=self.profile.get("username", "Player"))

    def _drop_view(self):
        for key in list(self.renderer.chunks):
            self.renderer.drop(key)
        if self.world is not None and self.player is not None and self.player.uuid:
            self.world.unregister_player(self.player.uuid)
        self.world = None
        self.player = None
        self.ready = False
        self.target = None
        self._gen_left = None
        self._mesh_queue = []
        self.chat.close_chat()
        self.chat.messages.clear()
        self.inventory.clear_held()
        self._mouse_held_left = False
        self._reset_breaking()

    def _return_to_worlds(self):
        if self.network_client is not None:
            self.network_client.disconnect()
            self.network_client = None
            self.other_players.clear()
            self._drop_view()
            self.mode = "servers"
            self.menu.screen = "servers"
            self.menu.servers = load_servers()
            self.menu.selected = 0 if self.menu.servers else -1
            self.menu.scroll = 0
            self.set_exclusive_mouse(False)
            self.keys.clear()
            return
        self._persist_world()
        self._drop_view()
        self.info = None
        self.mode = "worlds"
        self.menu.screen = "worlds"
        self.menu.worlds = self.catalog.list_worlds()
        self.menu.selected = 0 if self.menu.worlds else -1
        self.menu.scroll = 0
        self.menu.ensure_visible()
        self.set_exclusive_mouse(False)
        self.keys.clear()

    def _save_username(self):
        name = self.menu.username.strip() or "Player"
        self.profile["username"] = name
        save_profile(self.profile, getattr(self, "profile_dir", None))
        if not getattr(self, "_custom_title", False):
            self.set_caption(f"Block Game - {name}")

    def _start_web_companion(self):
        if getattr(self, "web_companion", None) is not None:
            self.web_companion.stop()
        port = int(self.settings.get("web_ui_port", 5000))
        self.web_companion = WebCompanion(self, host="0.0.0.0", port=port)
        self.web_companion.start()

    def _stop_web_companion(self):
        if getattr(self, "web_companion", None) is not None:
            self.web_companion.stop()
            self.web_companion = None

    def _save_settings(self):
        save_settings(self.settings, getattr(self, "profile_dir", None))
        if getattr(self, "player", None) is not None:
            self.player.name = name

    def _open_server(self, address):
        self._drop_view()
        self.info = None
        host, port = parse_address(address)
        skin = load_skin_rgba(self.profile["skin"])
        self.network_client = NetworkClient(
            host, port, self.profile["uuid"], username=self.profile.get("username", "Player"), skin_rgba=skin
        )
        if not self.network_client.connect(timeout=4.0):
            err = self.network_client.error or "Connection failed"
            self.network_client = None
            self.mode = "servers"
            self.menu.screen = "servers"
            self.menu.message = f"Failed to connect: {err}"
            return
        self.ready = False
        self.keys.clear()
        self.mode = "loading"
        self._status = f"Connecting to {host}:{port}"

    def _create_world(self):
        name = self.menu.name.strip()
        if not name:
            self.menu.message = "Give the world a name."
            return
        if self.menu.seed_text.strip():
            try:
                seed = int(self.menu.seed_text)
            except ValueError:
                self.menu.message = "Seed should be a whole number."
                return
            if seed < 0 or seed > 2_147_483_647:
                self.menu.message = "Seed should be from 0 to 2147483647."
                return
        else:
            seed = random.SystemRandom().randint(0, 2_147_483_647)
        self._open_world(self.catalog.create(name, seed, gamemode=self.menu.gamemode))

    def _run_action(self, action):
        if action in ("singleplayer", "multiplayer", "options", "quit"):
            self._save_username()
            self.menu.focus = ""
        if action == "singleplayer":
            self.mode = "worlds"
            self.menu.screen = "worlds"
            self.menu.worlds = self.catalog.list_worlds()
            self.menu.selected = 0 if self.menu.worlds else -1
            self.menu.scroll = 0
        elif action == "multiplayer":
            self.mode = "servers"
            self.menu.screen = "servers"
            self.menu.servers = load_servers()
            self.menu.selected = -1
            self.menu._row_click = 0.0
            self.menu.scroll = 0
            self.menu.message = ""
        elif action == "add-server":
            self.mode = "add_server"
            self.menu.screen = "add_server"
            self.menu.server_name = ""
            self.menu.server_address = ""
            self.menu.focus = "server_name"
            self.menu.message = ""
        elif action == "save-server":
            name = self.menu.server_name.strip() or "Block Game Server"
            addr = self.menu.server_address.strip() or "127.0.0.1:25565"
            self.menu.servers.append({"name": name, "address": addr})
            save_servers(self.menu.servers)
            self.mode = "servers"
            self.menu.screen = "servers"
            self.menu.selected = len(self.menu.servers) - 1
            self.menu.message = ""
        elif action == "cancel-server":
            self.mode = "servers"
            self.menu.screen = "servers"
            self.menu.message = ""
        elif action == "delete-server":
            server = self.menu.selected_server()
            if server is not None:
                self.menu.servers.remove(server)
                save_servers(self.menu.servers)
                self.menu.selected = min(self.menu.selected, len(self.menu.servers) - 1)
        elif action == "direct-connect":
            self.mode = "direct_connect"
            self.menu.screen = "direct_connect"
            self.menu.server_address = "127.0.0.1:25565"
            self.menu.focus = "server_address"
            self.menu.message = ""
        elif action == "cancel-direct":
            self.mode = "servers"
            self.menu.screen = "servers"
            self.menu.message = ""
        elif action == "join-direct":
            addr = self.menu.server_address.strip() or "127.0.0.1:25565"
            self._open_server(addr)
        elif action == "join-server":
            server = self.menu.selected_server()
            if server is not None:
                self._open_server(server["address"])
        elif action == "quit":
            self._quit()
        elif action == "options":
            self.mode = "options"
            self.menu.screen = "options"
            self.menu.local_ips = get_local_ips()
            self.menu.focus = ""
        elif action == "toggle-web-ui":
            self.settings["web_ui_enabled"] = self.menu.web_ui_enabled
            self.menu.local_ips = get_local_ips()
            self._save_settings()
            if self.menu.web_ui_enabled:
                self._start_web_companion()
            else:
                self._stop_web_companion()
        elif action == "back-options":
            self.mode = self.menu.screen
            self.settings["web_ui_port"] = self.menu.web_ui_port
            self._save_settings()
            if self.menu.web_ui_enabled:
                self._start_web_companion()
        elif action == "back":
            self.mode = "title"
            self.menu.screen = "title"
            self.menu.username = self.profile.get("username", "Player")
            self.menu.focus = ""
        elif action == "new":
            self.mode = "create"
            self.menu.screen = "create"
            self.menu.name = ""
            self.menu.seed_text = ""
            self.menu.gamemode = "survival"
            self.menu.focus = "name"
            self.menu.message = ""
        elif action == "back-worlds":
            self.mode = "worlds"
            self.menu.screen = "worlds"
        elif action == "play":
            info = self.menu.selected_world()
            if info is not None:
                self._open_world(info)
        elif action == "delete":
            if self.menu.selected_world() is not None:
                self.mode = "confirm"
                self.menu.screen = "confirm"
        elif action == "cancel-delete":
            self.mode = "worlds"
            self.menu.screen = "worlds"
        elif action == "confirm-delete":
            info = self.menu.selected_world()
            if info is not None:
                self.catalog.delete(info.id)
            self.menu.worlds = self.catalog.list_worlds()
            self.menu.selected = 0 if self.menu.worlds else -1
            self.menu.scroll = 0
            self.mode = "worlds"
            self.menu.screen = "worlds"
        elif action == "create":
            self._create_world()
        elif action == "resume":
            self.mode = "play"
            if not self._smoke:
                self.set_exclusive_mouse(True)
        elif action == "save-quit":
            self._return_to_worlds()

    def _menu_frame(self):
        self.menu.screen = self.mode
        frame = self.menu.layout(self.width, self.height)
        hover = self.menu.hit_test(frame, *self._mouse)
        if hover != self.menu.hover:
            self.menu.hover = hover
            frame = self.menu.layout(self.width, self.height)
        return frame

    def _collect_nametags(self, vp, fw, fh):
        if not self.other_players or vp is None:
            return [], []

        rects = []
        labels = []
        cam = np.array(self.camera, dtype=np.float32)

        for other in self.other_players.values():
            name = getattr(other, "username", "")
            if not name:
                continue

            # Tag is placed above head: feet + 2.25 units
            pos_world = np.array([other.pos[0], other.pos[1] + 2.25, other.pos[2], 1.0], dtype=np.float32)
            clip = vp @ pos_world
            w_clip = clip[3]
            if w_clip <= 0.1:
                continue

            ndc_x = clip[0] / w_clip
            ndc_y = clip[1] / w_clip
            ndc_z = clip[2] / w_clip

            if not (-1.2 <= ndc_x <= 1.2 and -1.2 <= ndc_y <= 1.2 and 0.0 <= ndc_z <= 1.0):
                continue

            dist = float(np.linalg.norm(pos_world[:3] - cam))
            if dist > 64.0:
                continue

            has_los = True
            if self.world is not None and hasattr(self.world, "raycast"):
                dx = float(pos_world[0] - cam[0])
                dy = float(pos_world[1] - cam[1])
                dz = float(pos_world[2] - cam[2])
                hit = self.world.raycast(float(cam[0]), float(cam[1]), float(cam[2]), dx, dy, dz, dist, solid_only=True)
                if isinstance(hit, dict) and hit.get("dist", dist) < dist - 0.6:
                    has_los = False

            # Window space coordinates
            sx = (ndc_x * 0.5 + 0.5) * self.width
            sy = (ndc_y * 0.5 + 0.5) * self.height

            box_w = max(50, len(name) * 9 + 16)
            box_h = 22
            box_x = sx - box_w / 2
            box_y = sy - box_h / 2

            bg_alpha = 0.45 if has_los else 0.2
            text_alpha = 255 if has_los else 120

            rects.append((box_x, box_y, box_w, box_h, (0.0, 0.0, 0.0, bg_alpha)))
            labels.append(LabelSpec(name, sx, sy, size=13, anchor_x="center", anchor_y="center", color=(255, 255, 255, text_alpha)))

        return rects, labels

    def on_draw(self):
        fw, fh = self.get_framebuffer_size()
        self.ctx.viewport = (0, 0, max(fw, 1), max(fh, 1))
        day_time = float(getattr(self.world, "day_time", 1000.0)) if self.world is not None else 1000.0
        sky_color = self.renderer.get_sky_color(day_time) if hasattr(self.renderer, "get_sky_color") else SKY
        self.ctx.clear(*sky_color, 1.0, depth=1.0)
        self._aim()
        vp = None
        if self.mode in ("play", "pause", "loading", "inventory", "options") and self.world is not None:
            aspect = max(fw, 1) / max(fh, 1)
            # Near plane stays short for world detail; the FP hand is placed past 0.2.
            projection = perspective(75, aspect, 0.05, 220)
            if self.view_mode == VIEW_THIRD_FRONT:
                view = look_at(self.camera, self.eye)
            else:
                lx, ly, lz = self.forward
                view = look_at(self.camera, (self.camera[0] + lx, self.camera[1] + ly, self.camera[2] + lz))
            vp = projection @ view
            if hasattr(self.renderer, "draw_sky"):
                self.renderer.draw_sky(vp, self.camera, day_time)
            self.renderer.draw_world(vp, self.camera, cutout=True, day_time=day_time)
            if self.mode in ("play", "pause", "inventory", "options") and self.player is not None and self.ready:
                if self.other_players:
                    self.renderer.draw_other_players(
                        vp,
                        self.camera,
                        self.other_players,
                        self._play_time,
                        day_time=day_time,
                    )
                if self.world is not None and getattr(self.world, "entities", None):
                    self.renderer.draw_item_entities(
                        vp,
                        self.camera,
                        self.world.entities,
                        self._play_time,
                        day_time=day_time,
                    )
            if self.mode == "play" and self.target is not None:
                target_coords = (self.target["x"], self.target["y"], self.target["z"])
                target_box = self.target.get("box")
                self.renderer.draw_outline(vp, target_coords, box=target_box)
                target_key = (self.target["x"], self.target["y"], self.target["z"], self.target.get("slab_half"))
                if self.gamemode == "survival" and self._breaking_block == target_key:
                    stage = min(9, max(0, int(self._break_progress * 10)))
                    self.renderer.draw_crack(vp, target_coords, stage, box=target_box)
            if self.mode in ("play", "pause", "inventory", "options") and self.player is not None and self.ready:
                if self.view_mode != VIEW_FIRST:
                    self.renderer.draw_player(
                        vp,
                        self.camera,
                        self.player,
                        self.hotbar,
                        self.selected,
                        self.view_mode,
                        self._play_time,
                        day_time=day_time,
                    )
            self.renderer.draw_translucent(vp, self.camera, day_time=day_time)
            if self.mode in ("play", "pause", "inventory", "options") and self.player is not None and self.ready:
                if self.view_mode == VIEW_FIRST:
                    self.renderer.draw_player(
                        vp,
                        self.camera,
                        self.player,
                        self.hotbar,
                        self.selected,
                        self.view_mode,
                        self._play_time,
                        day_time=day_time,
                    )
        scale = fw / max(self.width, 1)
        if vp is not None and self.other_players and self.mode in ("play", "pause", "inventory", "options"):
            tag_rects, self._nametag_labels = self._collect_nametags(vp, fw, fh)
            if tag_rects:
                self.renderer.draw_rects(
                    fw,
                    fh,
                    [(r[0] * scale, r[1] * scale, r[2] * scale, r[3] * scale, r[4]) for r in tag_rects],
                )
        else:
            self._nametag_labels = []

        if self.mode == "play" and self.ready:
            break_prog = self._break_progress if (self.gamemode == "survival" and self._breaking_block is not None) else 0.0
            self.renderer.draw_hud(fw, fh, self.hotbar, self.selected, crosshair=True, break_progress=break_prog)
            chat_rects, self._chat_labels = self.chat.layout(self.width, self.height, scale)
            if chat_rects:
                self.renderer.draw_rects(
                    fw,
                    fh,
                    [(r[0] * scale, r[1] * scale, r[2] * scale, r[3] * scale, r[4]) for r in chat_rects],
                )
        elif self.mode == "inventory":
            frame = self._inventory_frame()
            self.renderer.draw_rects(
                fw,
                fh,
                [(rect[0] * scale, rect[1] * scale, rect[2] * scale, rect[3] * scale, rect[4]) for rect in frame.rects],
            )
            self.renderer.draw_icons(
                fw,
                fh,
                [
                    (icon[0] * scale, icon[1] * scale, icon[2] * scale, icon[3] * scale, *icon[4:])
                    for icon in frame.icons
                ],
            )
        elif self.mode == "loading":
            self.renderer.draw_panel(fw, fh, 560 * scale, 120 * scale)
        elif self.mode in ("title", "worlds", "servers", "add_server", "direct_connect", "create", "confirm", "pause", "options"):
            frame = self._menu_frame()
            self.renderer.draw_rects(
                fw,
                fh,
                [(rect[0] * scale, rect[1] * scale, rect[2] * scale, rect[3] * scale, rect[4]) for rect in frame.rects],
            )
        self._draw_text(scale)

    def _draw_text(self, scale):
        if not self._text_ok:
            return
        try:
            while glGetError():
                pass
            glBindVertexArray(0)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            if self.mode == "play" and self.ready and self.player is not None:
                px, py, pz = self.player.pos
                biome_str = ""
                if self.world is not None and getattr(self.world, "biomes", None):
                    from server.biomes import biome_at
                    cur_biome = biome_at(int(px), int(pz), getattr(self.world, "seed", 0), self.world.biomes)
                    if cur_biome:
                        biome_title = getattr(cur_biome, "name", None) or cur_biome.id.replace("_", " ").title()
                        biome_str = f"   {biome_title}"
                name = ""
                if self.target is not None:
                    block = BLOCKS.get(self.world.get_block(self.target["x"], self.target["y"], self.target["z"]))
                    name = f"\n{block.name}" if block else ""
                held_block = ITEMS.get(stack_id(self.hotbar[self.selected]))
                held = held_block.name if held_block else "Empty"
                self.labels.draw(self._hotbar_counts(scale))
                if self._nametag_labels:
                    self.labels.draw(self._nametag_labels)
                if self._chat_labels:
                    self.labels.draw(self._chat_labels)
                self.readout.text = f"{px:.1f}   {py:.1f}   {pz:.1f}{biome_str}{name}\nHolding {held}"
                self.readout.x = 16
                self.readout.y = self.height - 16
                self.readout.draw()
                self.readout.x = 16
                self.readout.y = self.height - 16
                self.readout.draw()
            elif self.mode == "loading":
                self.status_label.text = self._status
                self.status_label.x = self.width / 2
                self.status_label.y = self.height / 2
                self.status_label.draw()
            elif self.mode == "inventory":
                self.labels.draw(self._inventory_frame().labels)
            elif self.mode in ("title", "worlds", "servers", "add_server", "direct_connect", "create", "confirm", "pause", "options"):
                self.labels.draw(self._menu_frame().labels)
        except Exception as exc:
            print(f"On-screen text error: {exc}", file=sys.stderr)

    def on_resize(self, width, height):
        super().on_resize(width, height)
        if hasattr(self, "ctx"):
            self.ctx.viewport = (0, 0, *self.get_framebuffer_size())

    def on_key_press(self, symbol, modifiers):
        if self.chat.open:
            if symbol == key.ESCAPE:
                self.chat.close_chat()
                if not self._smoke:
                    self.set_exclusive_mouse(True)
                return
            if symbol in (key.ENTER, key.RETURN):
                line = self.chat.send_current()
                if not self._smoke:
                    self.set_exclusive_mouse(True)
                if line:
                    self._submit_chat(line)
                return
            if symbol == key.BACKSPACE:
                self.chat.backspace()
                return
            if symbol == key.DELETE:
                self.chat.delete()
                return
            if symbol == key.LEFT:
                self.chat.cursor_left()
                return
            if symbol == key.RIGHT:
                self.chat.cursor_right()
                return
            if symbol == key.HOME:
                self.chat.cursor_home()
                return
            if symbol == key.END:
                self.chat.cursor_end()
                return
            if symbol == key.UP:
                self.chat.history_prev()
                return
            if symbol == key.DOWN:
                self.chat.history_next()
                return
            return

        if self.mode == "play":
            if symbol == key.T:
                self.chat.open_chat("")
                self._ignore_next_char = "t"
                self.set_exclusive_mouse(False)
                self.keys.clear()
                if self.player is not None:
                    self.player.input.forward = self.player.input.back = self.player.input.left = self.player.input.right = self.player.input.jump = self.player.input.sprint = False
                return
            if symbol == key.SLASH:
                self.chat.open_chat("/")
                self._ignore_next_char = "/"
                self.set_exclusive_mouse(False)
                self.keys.clear()
                if self.player is not None:
                    self.player.input.forward = self.player.input.back = self.player.input.left = self.player.input.right = self.player.input.jump = self.player.input.sprint = False
                return
            if symbol in (key.V, key.L):
                modes = ["Normal", "UV Vision", "Infrared / Thermal", "Full Spectrum"]
                self.vision_mode = (getattr(self, "vision_mode", 0) + 1) % len(modes)
                if hasattr(self, "renderer") and self.renderer is not None:
                    self.renderer.vision_mode = self.vision_mode
                self.chat.add_message(f"Vision Mode: {modes[self.vision_mode]}", sender="")
                return

        if symbol == key.BACKSPACE:
            if self.mode == "title":
                if self.menu.focus == "username":
                    self.menu.username = self.menu.username[:-1]
                    self._save_username()
                    return
            elif self.mode == "create":
                if self.menu.focus == "name":
                    self.menu.name = self.menu.name[:-1]
                elif self.menu.focus == "seed":
                    self.menu.seed_text = self.menu.seed_text[:-1]
                self.menu.message = ""
            elif self.mode == "options":
                if self.menu.focus == "web_ui_port":
                    port_str = str(self.menu.web_ui_port)
                    self.menu.web_ui_port = int(port_str[:-1]) if len(port_str) > 1 else 0
                    self.settings["web_ui_port"] = self.menu.web_ui_port
                    self._save_settings()
                    return
                return
            elif self.mode in ("add_server", "direct_connect"):
                if self.menu.focus == "server_name":
                    self.menu.server_name = self.menu.server_name[:-1]
                elif self.menu.focus == "server_address":
                    self.menu.server_address = self.menu.server_address[:-1]
                self.menu.message = ""
                return
        if symbol == key.ESCAPE:
            self._on_escape()
            return
        if symbol in (key.ENTER, key.RETURN):
            if self.mode == "title":
                self.menu.focus = ""
                self._save_username()
                return
            elif self.mode == "worlds":
                self._run_action("play")
            elif self.mode == "create":
                self._run_action("create")
            elif self.mode == "pause":
                self._run_action("resume")
            elif self.mode == "options":
                self._run_action("back-options")
            elif self.mode == "servers":
                self._run_action("join-server")
            elif self.mode == "add_server":
                self._run_action("save-server")
            elif self.mode == "direct_connect":
                self._run_action("join-direct")
            return
        if self.mode in ("worlds", "servers") and symbol in (key.UP, key.DOWN):
            self.menu.move_selection(-1 if symbol == key.UP else 1)
            return
        if symbol == key.E and self.mode in ("play", "inventory"):
            self._set_inventory(self.mode != "inventory")
            return
        if symbol == key.F5 and self.mode in ("play", "inventory", "pause"):
            self._cycle_view()
            return
        if symbol == key.Q and self.mode == "play" and not self.chat.open:
            self._drop_held_item()
            return
        number = {
            key._1: 0,
            key._2: 1,
            key._3: 2,
            key._4: 3,
            key._5: 4,
            key._6: 5,
            key._7: 6,
            key._8: 7,
            key._9: 8,
        }
        if symbol in number and self.mode in ("play", "inventory"):
            self.selected = number[symbol]
            return
        if self.mode != "play":
            return
        self.keys.add(symbol)

    def _hotbar_counts(self, scale):
        slot = 50 / scale
        gap = 8 / scale
        total = HOTBAR_SLOTS * slot + (HOTBAR_SLOTS - 1) * gap
        x = (self.width - total) / 2
        y = 28 / scale
        specs = []
        for index, slot_stack in enumerate(self.hotbar):
            count = stack_count(slot_stack)
            if count <= 1:
                continue
            sx = x + index * (slot + gap)
            specs.append(LabelSpec(str(count), sx + slot - 6, y + 4, 12, "right", "bottom"))
        return specs

    def _inventory_frame(self):
        frame = self.inventory.layout(
            self.width,
            self.height,
            self.hotbar,
            self.selected,
            self._mouse,
            self.storage,
            self.craft,
            gamemode=self.gamemode,
            container_type=self.inventory.container_type,
            furnace=self.inventory.furnace,
        )
        hover = self.menu.hit_test(frame, *self._mouse)
        if hover != self.inventory.hover:
            self.inventory.hover = hover
            frame = self.inventory.layout(
                self.width,
                self.height,
                self.hotbar,
                self.selected,
                self._mouse,
                self.storage,
                self.craft,
                gamemode=self.gamemode,
                container_type=self.inventory.container_type,
                furnace=self.inventory.furnace,
            )
        return frame

    def _capture_furnace_slots(self):
        if (
            self.network_client is not None
            and getattr(self.inventory, "container_type", None) == "furnace"
            and self.inventory.furnace is not None
        ):
            f = self.inventory.furnace
            from server.stacks import stack_count, stack_id
            return {
                "input": (stack_id(f.input), stack_count(f.input)),
                "fuel": (stack_id(f.fuel), stack_count(f.fuel)),
                "output": (stack_id(f.output), stack_count(f.output)),
            }
        return None

    def _sync_furnace_changes(self, prev_slots):
        if (
            prev_slots is not None
            and self.network_client is not None
            and getattr(self.inventory, "container_type", None) == "furnace"
            and self.inventory.furnace is not None
        ):
            f = self.inventory.furnace
            from server.stacks import stack_count, stack_id
            changed = {}
            if (stack_id(f.input), stack_count(f.input)) != prev_slots.get("input"):
                changed["input"] = f.input
            if (stack_id(f.fuel), stack_count(f.fuel)) != prev_slots.get("fuel"):
                changed["fuel"] = f.fuel
            if (stack_id(f.output), stack_count(f.output)) != prev_slots.get("output"):
                changed["output"] = f.output
            if changed:
                self.network_client.send_furnace_update_slots(f.x, f.y, f.z, changed)

    def _set_inventory(self, opened, container_type="player", furnace_pos=None, container_block=None):
        if self.chat.open:
            self.chat.close_chat()
        if not opened:
            if (
                self.network_client is not None
                and getattr(self.inventory, "container_type", None) == "furnace"
                and self.inventory.furnace is not None
            ):
                f = self.inventory.furnace
                self.network_client.send_furnace_close(f.x, f.y, f.z)
            player_pos = tuple(self.player.pos) if self.player and getattr(self.player, "pos", None) is not None else None
            self.inventory.dismiss_container(self.hotbar, self.storage, self.world, player_pos)
            self.inventory.dismiss_held(self.hotbar, self.storage, self.craft)
        self.keys.clear()
        if self.player is not None:
            self._apply_input()
        if opened:
            self.inventory.container_type = container_type
            self.inventory.container_block = container_block
            if container_type == "furnace" and furnace_pos is not None and self.world is not None:
                self.inventory.furnace = self.world.get_or_create_furnace(*furnace_pos, block_id=container_block)
                if self.network_client is not None:
                    self.network_client.send_furnace_open(furnace_pos[0], furnace_pos[1], furnace_pos[2])
            else:
                self.inventory.furnace = None
            self.mode = "inventory"
            self.set_exclusive_mouse(False)
            return
        self.mode = "play"
        if not self._smoke:
            self.set_exclusive_mouse(True)

    def _on_right_click(self, modifiers=0):
        if self._handle_mob_interaction():
            return
        self._aim()
        hit = self.target
        if hit is not None:
            is_shift = bool(modifiers & key.MOD_SHIFT) if hasattr(key, "MOD_SHIFT") else False
            if not is_shift and self.world is not None:
                target_block = self.world.get_block(hit["x"], hit["y"], hit["z"])
                target_obj = BLOCKS.get(target_block)
                station = getattr(target_obj, "station", None)
                base = getattr(target_obj, "base_id", target_block) if target_obj else target_block
                if station == "crafting_table" or base == "crafting_table":
                    self._set_inventory(True, container_type="crafting_table", container_block=target_block)
                    return
                if station == "furnace" or base == "furnace":
                    self._set_inventory(True, container_type="furnace", furnace_pos=(hit["x"], hit["y"], hit["z"]), container_block=target_block)
                    return

        if self._handle_bucket_interaction():
            return

        if self._handle_block_hit_interaction(hit):
            return

        self._place_block()

    def _handle_mob_interaction(self):
        if self.player is None or self.world is None:
            return False
        ex, ey, ez = self.player.pos
        ey += self.player.eye
        fx, fy, fz = look_direction(self.player.yaw, self.player.pitch)
        hit_mob = self.world.raycast_entity(ex, ey, ez, fx, fy, fz, max_dist=3.5)
        if hit_mob is None or not getattr(hit_mob, "alive", True):
            return False

        held_slot = self.hotbar[self.selected] if self.selected < len(self.hotbar) else empty_stack()
        held_id = stack_id(held_slot)

        if getattr(hit_mob, "mob_type", "") == "sheep":
            if held_id == "shears":
                if getattr(hit_mob, "sheared", False):
                    return True
                if self.network_client is not None:
                    self.network_client.send_interact_mob(hit_mob.id, held_id)
                else:
                    hit_mob.shear(world=self.world)
                return True
            if hasattr(hit_mob, "dye") and held_id.endswith("_dye"):
                if self.network_client is not None:
                    if hit_mob.dye(held_id):
                        self.network_client.send_interact_mob(hit_mob.id, held_id)
                        if self.gamemode != "creative":
                            cnt = stack_count(held_slot)
                            if cnt <= 1:
                                self.hotbar[self.selected] = empty_stack()
                            else:
                                self.hotbar[self.selected] = make_stack(held_id, cnt - 1)
                    return True
                else:
                    if hit_mob.dye(held_id):
                        if self.gamemode != "creative":
                            cnt = stack_count(held_slot)
                            if cnt <= 1:
                                self.hotbar[self.selected] = empty_stack()
                            else:
                                self.hotbar[self.selected] = make_stack(held_id, cnt - 1)
                    return True
        return False

    def _handle_block_hit_interaction(self, hit):
        if self.player is None or self.world is None or hit is None:
            return False
        if self.selected >= len(self.hotbar):
            return False
        held_slot = self.hotbar[self.selected]
        held_id = stack_id(held_slot)
        if not held_id:
            return False

        hx, hy, hz = hit["x"], hit["y"], hit["z"]
        from server.recipes import find_block_hit_recipe

        target_block = self.world.get_block(hx, hy, hz)
        recipe = find_block_hit_recipe(held_id, target_block)
        if recipe is None:
            return False

        if self.network_client is not None:
            self.network_client.send_hit_block(hx, hy, hz, held_id)
            if recipe.consume_item and self.gamemode != "creative":
                cnt = stack_count(held_slot)
                if cnt <= 1:
                    self.hotbar[self.selected] = empty_stack()
                else:
                    self.hotbar[self.selected] = make_stack(held_id, cnt - 1)
        else:
            triggered, drops, consume = self.world.hit_block_with_item(hx, hy, hz, held_id)
            if triggered:
                for drop_id, count in drops:
                    if count > 0:
                        vx = random.uniform(-0.8, 0.8)
                        vy = random.uniform(1.8, 3.2)
                        vz = random.uniform(-0.8, 0.8)
                        pos = [hx + 0.5 + hit["nx"] * 0.4, hy + 0.5 + hit["ny"] * 0.4, hz + 0.5 + hit["nz"] * 0.4]
                        self.world.spawn_item(drop_id, count=count, pos=pos, vel=[vx, vy, vz], pickup_delay=0.3)

                if consume and self.gamemode != "creative":
                    cnt = stack_count(held_slot)
                    if cnt <= 1:
                        self.hotbar[self.selected] = empty_stack()
                    else:
                        self.hotbar[self.selected] = make_stack(held_id, cnt - 1)
        return True

    def _handle_bucket_interaction(self):
        if self.player is None or self.world is None:
            return False
        if self.selected >= len(self.hotbar):
            return False
        held_slot = self.hotbar[self.selected]
        held_id = stack_id(held_slot)
        if not held_id:
            return False

        from server.fluid import (
            get_bucket_empty,
            get_bucket_fluid,
            get_filled_bucket,
            is_empty_bucket,
            is_filled_bucket,
            is_water,
            is_water_source,
        )

        # 1. Picking up fluid with an empty bucket
        if is_empty_bucket(held_id):
            fluid_hit = self.world.raycast(*self.eye, *self.forward, 6, fluids=True)
            if fluid_hit is not None:
                fx, fy, fz = fluid_hit["x"], fluid_hit["y"], fluid_hit["z"]
                target_block = self.world.get_block(fx, fy, fz)
                from server.blocks import is_waterlogged_slab, get_slab_info
                from server.fluid import get_fluid_type, is_water_source

                if is_waterlogged_slab(target_block) and is_water_source(target_block):
                    s_info = get_slab_info(target_block)
                    unlogged_id = s_info["base_id"] if s_info["half"] == "bottom" else f"{s_info['base_id']}_top"
                    filled_id = get_filled_bucket(held_id, "water")
                    if filled_id:
                        if self.network_client is not None:
                            self.network_client.send_set_block(fx, fy, fz, unlogged_id)
                        self.world.set_block(fx, fy, fz, unlogged_id)
                        self._upload_block_change(fx, fy, fz)

                        if self.gamemode == "creative":
                            self.hotbar[self.selected] = make_stack(filled_id, 1)
                        else:
                            cnt = stack_count(held_slot)
                            if cnt <= 1:
                                self.hotbar[self.selected] = make_stack(filled_id, 1)
                            else:
                                self.hotbar[self.selected] = make_stack(held_id, cnt - 1)
                                leftover = add_to_inventory([self.hotbar, self.storage], filled_id, 1)
                                if leftover > 0:
                                    self.world.spawn_item(filled_id, count=leftover, pos=list(self.player.pos))
                        return True
                elif is_water_source(target_block):
                    filled_id = get_filled_bucket(held_id, get_fluid_type(target_block) or "water")
                    if filled_id:
                        if self.network_client is not None:
                            self.network_client.send_set_block(fx, fy, fz, AIR)
                        self.world.set_block(fx, fy, fz, AIR)
                        self._upload_block_change(fx, fy, fz)

                        if self.gamemode == "creative":
                            self.hotbar[self.selected] = make_stack(filled_id, 1)
                        else:
                            cnt = stack_count(held_slot)
                            if cnt <= 1:
                                self.hotbar[self.selected] = make_stack(filled_id, 1)
                            else:
                                self.hotbar[self.selected] = make_stack(held_id, cnt - 1)
                                leftover = add_to_inventory([self.hotbar, self.storage], filled_id, 1)
                                if leftover > 0:
                                    self.world.spawn_item(filled_id, count=leftover, pos=list(self.player.pos))
                        return True
            return False

        # 2. Placing fluid from a filled bucket
        if is_filled_bucket(held_id):
            fluid_id = get_bucket_fluid(held_id)
            empty_id = get_bucket_empty(held_id)
            if not fluid_id:
                return False

            hit = self.world.raycast(*self.eye, *self.forward, 6, fluids=True)
            if hit is None:
                return False

            hx, hy, hz = hit["x"], hit["y"], hit["z"]
            cur_target = self.world.get_block(hx, hy, hz)
            from server.blocks import is_slab, get_slab_info
            from server.fluid import is_water_source

            s_info = get_slab_info(cur_target)
            if s_info is not None and not s_info["double"] and not is_water_source(cur_target):
                logged_id = (
                    f"{s_info['base_id']}_waterlogged"
                    if s_info["half"] == "bottom"
                    else f"{s_info['base_id']}_top_waterlogged"
                )
                if self.network_client is not None:
                    self.network_client.send_set_block(hx, hy, hz, logged_id)
                self.world.set_block(hx, hy, hz, logged_id)
                self._upload_block_change(hx, hy, hz)

                if self.gamemode != "creative":
                    cnt = stack_count(held_slot)
                    if cnt <= 1:
                        self.hotbar[self.selected] = make_stack(empty_id, 1)
                    else:
                        self.hotbar[self.selected] = make_stack(held_id, cnt - 1)
                        leftover = add_to_inventory([self.hotbar, self.storage], empty_id, 1)
                        if leftover > 0:
                            self.world.spawn_item(empty_id, count=leftover, pos=list(self.player.pos))
                return True
            target_obj = BLOCKS.get(cur_target)
            is_plant_or_breakable = (
                target_obj is not None
                and not is_water(cur_target)
                and (
                    getattr(target_obj, "fluid_breakable", False)
                    or getattr(target_obj, "plant", False)
                    or getattr(target_obj, "requires_soil", False)
                )
            )
            if (is_water(cur_target) and not is_water_source(cur_target)) or is_plant_or_breakable:
                px, py, pz = hx, hy, hz
            else:
                px = hx + hit["nx"]
                py = hy + hit["ny"]
                pz = hz + hit["nz"]

            if py < 0 or py >= WORLD_HEIGHT:
                return False
            if self.world.is_solid(px, py, pz):
                return False

            if self.network_client is not None:
                self.network_client.send_set_block(px, py, pz, fluid_id)
            self.world.set_block(px, py, pz, fluid_id)
            self._upload_block_change(px, py, pz)

            if self.gamemode != "creative":
                cnt = stack_count(held_slot)
                if cnt <= 1:
                    self.hotbar[self.selected] = make_stack(empty_id, 1)
                else:
                    self.hotbar[self.selected] = make_stack(held_id, cnt - 1)
                    leftover = add_to_inventory([self.hotbar, self.storage], empty_id, 1)
                    if leftover > 0:
                        self.world.spawn_item(empty_id, count=leftover, pos=list(self.player.pos))
            return True

        return False

    def _on_escape(self):
        self._mouse_held_left = False
        self._reset_breaking()
        if self.chat.open:
            self.chat.close_chat()
            if not self._smoke:
                self.set_exclusive_mouse(True)
            return
        if self.mode == "inventory":
            self._set_inventory(False)
            return
        if self.mode == "play":
            self.mode = "pause"
            self.menu.screen = "pause"
            self.set_exclusive_mouse(False)
            self.keys.clear()
            if self.player is not None:
                self.player.input.forward = False
                self.player.input.back = False
                self.player.input.left = False
                self.player.input.right = False
                self.player.input.jump = False
                self.player.input.sprint = False
        elif self.mode == "pause":
            self._run_action("resume")
        elif self.mode == "options":
            self._run_action("back-options")
        elif self.mode == "loading":
            self._return_to_worlds()
        elif self.mode == "confirm":
            self._run_action("cancel-delete")
        elif self.mode == "create":
            self._run_action("back-worlds")
        elif self.mode == "worlds":
            self._run_action("back")
        elif self.mode in ("servers", "add_server", "direct_connect"):
            if self.mode in ("add_server", "direct_connect"):
                self.mode = "servers"
                self.menu.screen = "servers"
            else:
                self.mode = "title"
                self.menu.screen = "title"
        elif self.mode == "title":
            self._quit()

    def on_key_release(self, symbol, modifiers):
        self.keys.discard(symbol)

    def on_text(self, text):
        if not text.isprintable():
            return
        if self._ignore_next_char is not None:
            ignore = self._ignore_next_char
            self._ignore_next_char = None
            if text.lower() == ignore.lower():
                return
        if self.chat.open:
            self.chat.add_char(text)
            return
        if self.mode == "title":
            if self.menu.focus == "username" and len(self.menu.username) < 24:
                self.menu.username += text
                self._save_username()
                return
        elif self.mode == "create":
            if self.menu.focus == "name" and len(self.menu.name) < 32:
                self.menu.name += text
                self.menu.message = ""
            elif self.menu.focus == "seed" and text.isdigit() and len(self.menu.seed_text) < 10:
                self.menu.seed_text += text
                self.menu.message = ""
        elif self.mode in ("add_server", "direct_connect"):
            if self.menu.focus == "server_name" and len(self.menu.server_name) < 32:
                self.menu.server_name += text
                self.menu.message = ""
            elif self.menu.focus == "server_address" and len(self.menu.server_address) < 64:
                self.menu.server_address += text
                self.menu.message = ""
        elif self.mode == "options":
            if self.menu.focus == "web_ui_port" and text.isdigit():
                port_str = str(self.menu.web_ui_port) if self.menu.web_ui_port else ""
                if len(port_str) < 5:
                    self.menu.web_ui_port = int(port_str + text)
                    self.settings["web_ui_port"] = self.menu.web_ui_port
                    self._save_settings()

    def on_mouse_motion(self, x, y, dx, dy):
        self._mouse = (x, y)
        if self.mode == "play" and not self.chat.open:
            self._look(dx, dy)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self._mouse = (x, y)
        if self.mode == "play" and not self.chat.open:
            self._look(dx, dy)
        elif self.mode == "inventory":
            if (buttons & mouse.RIGHT) and self.inventory.held and self.inventory.held_count > 0:
                frame = self._inventory_frame()
                action = self.menu.hit_test(frame, x, y)
                if action and action not in self.inventory.drag_slots:
                    if any(
                        action.startswith(prefix)
                        for prefix in ("craft:", "table_craft:", "store:", "hotbar:", "furnace:input", "furnace:fuel")
                    ):
                        prev_slots = self._capture_furnace_slots()
                        self.inventory.drag_slots.add(action)
                        self.inventory.apply(
                            action,
                            "right_drag",
                            self.hotbar,
                            self.selected,
                            self.storage,
                            self.craft,
                            gamemode=self.gamemode,
                            furnace=self.inventory.furnace,
                        )
                        self._sync_furnace_changes(prev_slots)

    def _look(self, dx, dy):
        if self.player is None:
            return
        self.player.yaw -= dx * 0.0022
        self.player.pitch += dy * 0.0022
        self.player.pitch = max(-1.45, min(1.45, self.player.pitch))

    def on_mouse_press(self, x, y, button, modifiers):
        self._mouse = (x, y)
        if self.chat.open:
            return
        if self.mode == "inventory":
            frame = self._inventory_frame()
            button_name = "left" if button == mouse.LEFT else "right"
            action = self.menu.hit_test(frame, x, y)
            shift = bool(modifiers & key.MOD_SHIFT) if hasattr(key, "MOD_SHIFT") else False
            if button == mouse.RIGHT:
                self.inventory.drag_button = "right"
                self.inventory.drag_slots = {action} if action else set()
            prev_slots = self._capture_furnace_slots()
            picked = self.inventory.apply(
                action,
                button_name,
                self.hotbar,
                self.selected,
                self.storage,
                self.craft,
                gamemode=self.gamemode,
                furnace=self.inventory.furnace,
                shift=shift,
            )
            if picked is not None:
                self.selected = picked
            self._sync_furnace_changes(prev_slots)
            return
        if button != mouse.LEFT:
            if self.mode == "play":
                self._on_right_click(modifiers)
            return
        if self.mode == "play":
            # Check if hitting a living mob first
            if self.player is not None and self.world is not None:
                ex, ey, ez = self.player.pos
                ey += self.player.eye
                fx, fy, fz = look_direction(self.player.yaw, self.player.pitch)
                hit_mob = self.world.raycast_entity(ex, ey, ez, fx, fy, fz, max_dist=3.5)
                if hit_mob is not None:
                    if self.network_client is not None:
                        self.network_client.send_attack_mob(hit_mob.id)
                        hit_mob.hurt_time = 0.4
                    else:
                        hit_mob.take_damage(2.0, attacker=self.player, world=self.world)
                    return
            is_shift = bool(modifiers & key.MOD_SHIFT) if hasattr(key, "MOD_SHIFT") else False
            self._aim()
            hit = self.target
            if not is_shift and hit is not None and self._handle_block_hit_interaction(hit):
                return

            if self.gamemode == "creative":
                self._break_block()
            else:
                self._mouse_held_left = True
                self._aim()
                if self.target is not None:
                    self._start_breaking(self.target)
            return
        if self.mode == "loading":
            return
        frame = self._menu_frame()
        self._run_action(self.menu.apply(self.menu.hit_test(frame, x, y)))

    def on_mouse_release(self, x, y, button, modifiers):
        if self.mode == "inventory":
            self.inventory.drag_slots.clear()
            self.inventory.drag_button = None
        if button == mouse.LEFT:
            self._mouse_held_left = False
            self._reset_breaking()

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if scroll_y == 0:
            return
        if self.mode == "play":
            step = -1 if scroll_y > 0 else 1
            self.selected = (self.selected + step) % HOTBAR_SLOTS
        elif self.mode == "inventory":
            self.inventory.scroll_by(-1 if scroll_y > 0 else 1)
        elif self.mode in ("worlds", "servers"):
            self.menu.scroll_by(-1 if scroll_y > 0 else 1)

    def on_deactivate(self):
        self._mouse_held_left = False
        self._reset_breaking()
        self._clear_input()
        if self.network_client is not None:
            return
        if self.mode == "inventory":
            player_pos = tuple(self.player.pos) if self.player and getattr(self.player, "pos", None) is not None else None
            self.inventory.dismiss_container(self.hotbar, self.storage, self.world, player_pos)
            self.inventory.dismiss_held(self.hotbar, self.storage, self.craft)
            self.mode = "pause"
            self.menu.screen = "pause"
            return
        if self.mode == "play":
            self._on_escape()

    def on_close(self):
        if self._closing:
            return
        self._closing = True
        self._stop_web_companion()
        if self.network_client is not None:
            self.network_client.disconnect()
            self.network_client = None
        self._persist_world()
        self._release_text()
        super().on_close()

    def _release_text(self):
        import gc

        self.labels.labels.clear()
        self.status_label = None
        self.readout = None
        gc.collect()
        while glGetError():
            pass

    def _quit(self):
        self.on_close()

    def run_smoke(self):
        from pyglet.gl import glGetError

        self.switch_to()
        self.on_draw()
        fw, fh = self.get_framebuffer_size()
        menu_ok = self._pixel_is_panel(fw, fh)
        self._save_shot(fw, fh, "/tmp/blockgame-menu.png")
        info = self.catalog.create("Smoke Hills", 7)
        self.menu.worlds = self.catalog.list_worlds()
        self.menu.selected = 0
        self.mode = "worlds"
        self.on_draw()
        self._save_shot(fw, fh, "/tmp/blockgame-worlds.png")
        self.mode = "create"
        self.menu.name = "Smoke Hills"
        self.menu.seed_text = "7"
        self.menu.focus = "name"
        self.on_draw()
        self._save_shot(fw, fh, "/tmp/blockgame-create.png")
        self.mode = "confirm"
        self.on_draw()
        self._save_shot(fw, fh, "/tmp/blockgame-confirm.png")
        self.mode = "worlds"
        self._open_world(info)
        while self.mode == "loading":
            self._bootstrap_step()
        self.on_draw()
        terrain_ok = self._pixel_is_terrain(fw, fh)
        self._save_shot(fw, fh, "/tmp/blockgame-smoke.png")
        if self.renderer.chunks:
            self._upload(next(iter(self.renderer.chunks)))
            self.on_draw()
        self.hotbar[0] = make_stack("grass", 1)
        self.selected = 0
        self.view_mode = VIEW_FIRST
        self.on_draw()
        self._save_shot(fw, fh, "/tmp/blockgame-player-fp.png")
        self.view_mode = VIEW_THIRD_BACK
        self.player.yaw = 0.9
        self.player.pitch = -0.25
        self.on_draw()
        self._save_shot(fw, fh, "/tmp/blockgame-player-tp.png")
        self.view_mode = VIEW_FIRST
        self.mode = "inventory"
        self.on_draw()
        self._save_shot(fw, fh, "/tmp/blockgame-inventory.png")
        self.inventory.tab = "natural"
        self.on_draw()
        self._save_shot(fw, fh, "/tmp/blockgame-inventory-natural.png")
        self.inventory.tab = "tools"
        self.on_draw()
        self._save_shot(fw, fh, "/tmp/blockgame-inventory-tools.png")
        self.inventory.tab = "inventory"
        self.craft[0] = make_stack("log", 1)
        self.on_draw()
        self._save_shot(fw, fh, "/tmp/blockgame-inventory-player.png")
        self.mode = "pause"
        self.on_draw()
        self._save_shot(fw, fh, "/tmp/blockgame-pause.png")
        self.flip()
        while glGetError():
            pass
        self.close()
        return 0 if menu_ok and terrain_ok else 1

    def _pixel_is_panel(self, fw, fh):
        raw = self.ctx.screen.read(viewport=(fw // 2, fh // 2, 1, 1), components=3, alignment=1)
        while glGetError():
            pass
        pixel = (raw[0], raw[1], raw[2])
        sky = tuple(int(channel * 255) for channel in SKY)
        if max(pixel) < 20:
            return False
        return any(abs(pixel[i] - sky[i]) > 24 for i in range(3))

    def _pixel_is_terrain(self, fw, fh):
        sky = tuple(int(channel * 255) for channel in SKY)
        raw = self.ctx.screen.read(viewport=(fw // 2, fh // 6, 1, 1), components=3, alignment=1)
        while glGetError():
            pass
        pixel = (raw[0], raw[1], raw[2])
        if max(pixel) < 40:
            return False
        return any(abs(pixel[i] - sky[i]) > 18 for i in range(3))

    def _save_shot(self, fw, fh, path):
        raw = self.ctx.screen.read(components=3, alignment=1)
        while glGetError():
            pass
        image = pyglet.image.ImageData(fw, fh, "RGB", raw)
        image.save(path)


def main(argv=None):
    raw_args = sys.argv[1:] if argv is None else argv
    if "--server" in raw_args:
        from server.network import main as server_main
        server_args = [a for a in raw_args if a != "--server"]
        return server_main(server_args)

    parser = argparse.ArgumentParser(description="Block Game Client")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test headless")
    parser.add_argument("--saves-dir", default=None, help="Custom world storage directory")
    parser.add_argument("--profile-dir", default=None, help="Custom user profile directory")
    parser.add_argument("--username", default=None, help="Player username")
    parser.add_argument("--uuid", default=None, help="Player UUID")
    parser.add_argument("--title", default=None, help="Custom window title")
    parser.add_argument("--connect", default=None, help="Auto-connect to server address (e.g. 127.0.0.1:25565)")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument("--pos", default=None, help="Window position X,Y (e.g. 50,50)")

    parsed = parser.parse_args(raw_args)

    window_pos = None
    if parsed.pos:
        try:
            parts = [int(p.strip()) for p in parsed.pos.split(",")]
            if len(parts) >= 2:
                window_pos = (parts[0], parts[1])
        except ValueError:
            pass

    saves_dir = tempfile.mkdtemp(prefix="blockgame-") if parsed.smoke else parsed.saves_dir
    game = Game(
        smoke=parsed.smoke,
        saves_dir=saves_dir,
        profile_dir=parsed.profile_dir,
        username=parsed.username,
        player_uuid=parsed.uuid,
        title=parsed.title,
        width=parsed.width,
        height=parsed.height,
        window_pos=window_pos,
        connect=parsed.connect,
    )
    if parsed.smoke:
        return game.run_smoke()
    pyglet.app.run()
    return 0
