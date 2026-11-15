"""The world server's side of the weather connection.

The world is the master clock. It answers Precision Time Protocol
exchanges and sends the height of loaded columns so built structures
can push the clouds around. The weather process connects in; this side
never dials out.
"""

import math
import queue
import select
import socket
import threading
import time

from server.blocks import CHUNK_SIZE
from server.weather import CELL, CLOUD_Y, dump_packet, read_packets


class WeatherSession:
    """Protocol state for one connected weather server."""

    def __init__(self, world):
        self.world = world
        self.outgoing = []
        self.sync_left = 0.0
        self.map_left = 0.0
        self.state = None
        self.state_seq = 0

    def welcome(self):
        return {
            "type": "weather_welcome",
            "seed": int(self.world.seed),
            "day_time": float(self.world.day_time),
            "cloud_y": CLOUD_Y,
            "cell": CELL,
            "time": time.time(),
        }

    def send(self, packet):
        self.outgoing.append(packet)

    def take_outgoing(self):
        pending = self.outgoing
        self.outgoing = []
        return pending

    def on_packet(self, packet):
        kind = packet.get("type")
        if kind == "ptp_delay_req":
            self.send(
                {
                    "type": "ptp_delay_resp",
                    "t1": packet.get("t1"),
                    "t2": packet.get("t2"),
                    "t3": packet.get("t3"),
                    "t4": time.time(),
                    "day_time": float(self.world.day_time),
                    "rate": 20.0,
                }
            )
        elif kind == "weather_state":
            self.state = packet
            self.state_seq += 1
            self.world.weather = packet
            self.world.weather_at = time.perf_counter()

    def tick(self, dt, focus):
        self.sync_left -= float(dt)
        self.map_left -= float(dt)
        if self.sync_left <= 0.0:
            self.sync_left = 1.0
            self.send({"type": "ptp_sync", "t1": time.time()})
        if self.map_left <= 0.0:
            self.map_left = 0.5
            self.send(cover_packet(self.world, focus))

    def client_packet(self):
        if not self.state:
            return None
        packet = dict(self.state)
        packet["type"] = "weather"
        packet.pop("_rain", None)
        packet.pop("_density", None)
        return packet


def weather_focus(world):
    points = []
    for player in list(getattr(world, "players", {}).values()):
        if getattr(player, "dead", False):
            continue
        pos = getattr(player, "pos", None)
        if pos is not None and len(pos) >= 3:
            points.append(pos)
    if not points:
        spawn = world.get_spawn_point()
        return float(spawn[0]), float(spawn[2])
    return (
        sum(float(pos[0]) for pos in points) / len(points),
        sum(float(pos[2]) for pos in points) / len(points),
    )


def cover_packet(world, focus, radius=80, cell=CELL):
    """Highest solid block on loaded columns around the players.

    Unloaded ground is left out. The weather server already knows the
    generated terrain from the seed, and uses these tops where a player
    has built or dug.
    """
    fx, fz = float(focus[0]), float(focus[1])
    x0 = int(math.floor((fx - radius) / cell)) * cell
    z0 = int(math.floor((fz - radius) / cell)) * cell
    x1 = int(math.floor((fx + radius) / cell)) * cell
    z1 = int(math.floor((fz + radius) / cell)) * cell
    columns = getattr(world, "columns", {})
    tops = []
    for x in range(x0, x1 + 1, cell):
        for z in range(z0, z1 + 1, cell):
            if (x // CHUNK_SIZE, z // CHUNK_SIZE) not in columns:
                continue
            tops.append([int(x), int(z), int(world.column_top(x, z))])
    return {"type": "cover", "focus": [fx, fz], "tops": tops}


class WeatherPort:
    """Listens for one weather server. Socket IO runs on its own thread."""

    def __init__(self, world, host="127.0.0.1", port=0):
        self.world = world
        self.session = WeatherSession(world)
        self.inbox = queue.Queue()
        self.outbox = queue.Queue()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, int(port)))
        self.server.listen(1)
        self.server.setblocking(False)
        self.host, self.port = self.server.getsockname()
        self.running = True
        self._welcomed = False
        self._chunks = []
        self.thread = threading.Thread(target=self._io, name="weather-link", daemon=True)
        self.thread.start()

    def poll(self, dt, focus=None):
        while True:
            try:
                packet = self.inbox.get_nowait()
            except queue.Empty:
                break
            kind = packet.get("type")
            if kind == "_connected":
                self._welcomed = True
                self.session = WeatherSession(self.world)
                self.outbox.put(self.session.welcome())
                continue
            if kind == "_disconnected":
                self._welcomed = False
                continue
            self.session.on_packet(packet)
        if not self._welcomed:
            return
        if focus is None:
            focus = weather_focus(self.world)
        self.session.tick(dt, focus)
        for packet in self.session.take_outgoing():
            self.outbox.put(packet)

    def close(self):
        self.running = False
        try:
            self.server.close()
        except OSError:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _io(self):
        client = None
        buf = bytearray()
        while self.running:
            self._gather()
            read = [self.server]
            write = []
            if client is not None:
                read.append(client)
                if self._chunks:
                    write.append(client)
            try:
                readable, writable, _ = select.select(read, write, [], 0.05)
            except (OSError, ValueError):
                break
            if self.server in readable and client is None:
                try:
                    conn, _addr = self.server.accept()
                except OSError:
                    conn = None
                if conn is not None:
                    conn.setblocking(False)
                    client = conn
                    buf = bytearray()
                    self.inbox.put({"type": "_connected"})
            if client is not None and client in readable:
                try:
                    data = client.recv(65536)
                except OSError:
                    data = b""
                if not data:
                    try:
                        client.close()
                    except OSError:
                        pass
                    client = None
                    self._chunks.clear()
                    self.inbox.put({"type": "_disconnected"})
                else:
                    buf.extend(data)
                    for packet in read_packets(buf):
                        self.inbox.put(packet)
            if client is not None and client in writable:
                self._flush(client)
        if client is not None:
            try:
                client.close()
            except OSError:
                pass

    def _gather(self):
        while True:
            try:
                packet = self.outbox.get_nowait()
            except queue.Empty:
                break
            self._chunks.append(dump_packet(packet))

    def _flush(self, client):
        while self._chunks:
            data = self._chunks[0]
            try:
                sent = client.send(data)
            except BlockingIOError:
                return
            except OSError:
                return
            if sent < len(data):
                self._chunks[0] = data[sent:]
                return
            self._chunks.pop(0)
