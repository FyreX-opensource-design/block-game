import queue
import select
import socket
import struct
import threading
import time

from server.network import decode_chunk, decode_skin, encode_skin, parse_packets, send_packet

_HEADER = struct.Struct(">I")


class NetworkClient:
    def __init__(self, host, port, uuid, username="Player", skin_rgba=None):
        self.host = host
        self.port = port
        self.uuid = uuid
        self.username = username
        self.skin_rgba = skin_rgba
        self.sock = None
        self.connected = False
        self.running = False
        self._thread = None
        self.in_queue = queue.Queue()
        self.out_queue = []
        self.error = None

    def connect(self, timeout=5.0):
        """Connect to the server and send login packet."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
            self.connected = True
            self.running = True

            # Send login handshake
            login_packet = {
                "type": "login",
                "uuid": self.uuid,
                "username": self.username,
                "skin": encode_skin(self.skin_rgba),
            }
            send_packet(self.sock, login_packet)

            # Start background reader/writer thread
            self._thread = threading.Thread(target=self._io_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as exc:
            self.error = str(exc)
            self.disconnect()
            return False

    def send_move(self, pos, vel, yaw, pitch, held_item=""):
        if not self.connected:
            return
        packet = {
            "type": "player_move",
            "pos": [float(p) for p in pos],
            "vel": [float(v) for v in vel],
            "yaw": float(yaw),
            "pitch": float(pitch),
            "held": str(held_item or ""),
        }
        self._enqueue(packet)

    def send_set_block(self, x, y, z, block_id):
        if not self.connected:
            return
        packet = {
            "type": "set_block",
            "x": int(x),
            "y": int(y),
            "z": int(z),
            "block": str(block_id or ""),
        }
        self._enqueue(packet)

    def send_hit_block(self, x, y, z, item_id):
        if not self.connected:
            return
        packet = {
            "type": "hit_block",
            "x": int(x),
            "y": int(y),
            "z": int(z),
            "item": str(item_id or ""),
        }
        self._enqueue(packet)

    def send_chat(self, message):
        if not self.connected:
            return
        packet = {
            "type": "chat",
            "message": str(message or ""),
        }
        self._enqueue(packet)

    def send_drop_item(self, item, count, pos, vel, pickup_delay=0.0):
        if not self.connected:
            return
        packet = {
            "type": "drop_item",
            "item": str(item),
            "count": int(count),
            "pos": [float(p) for p in pos],
            "vel": [float(v) for v in vel],
            "pickup_delay": float(pickup_delay),
        }
        self._enqueue(packet)

    def send_pickup_item(self, entity_id):
        if not self.connected:
            return
        packet = {
            "type": "pickup_item",
            "id": int(entity_id),
        }
        self._enqueue(packet)

    def send_attack_mob(self, entity_id):
        """Send attack mob packet to the server."""
        if not self.connected:
            return
        packet = {
            "type": "attack_entity",
            "id": int(entity_id),
        }
        self._enqueue(packet)

    def send_furnace_open(self, x, y, z):
        """Notify the server that this player opened a furnace."""
        if not self.connected:
            return
        packet = {
            "type": "furnace_open",
            "x": int(x),
            "y": int(y),
            "z": int(z),
        }
        self._enqueue(packet)

    def send_furnace_close(self, x, y, z):
        """Notify the server that this player closed the furnace."""
        if not self.connected:
            return
        packet = {
            "type": "furnace_close",
            "x": int(x),
            "y": int(y),
            "z": int(z),
        }
        self._enqueue(packet)

    def send_furnace_update_slots(self, x, y, z, slots):
        """Send only the changed slots of a furnace to the server."""
        if not self.connected or not slots:
            return
        from server.stacks import normalize_stack
        clean_slots = {k: normalize_stack(v) for k, v in slots.items() if k in ("input", "fuel", "output")}
        packet = {
            "type": "furnace_update",
            "x": int(x),
            "y": int(y),
            "z": int(z),
            "slots": clean_slots,
        }
        self._enqueue(packet)

    def send_furnace_update(self, x, y, z, input_stack, fuel_stack, output_stack):
        """Send updated slot state of a furnace to the server."""
        slots = {
            "input": input_stack,
            "fuel": fuel_stack,
            "output": output_stack,
        }
        self.send_furnace_update_slots(x, y, z, slots)

    def send(self, packet):
        """Send a raw packet dictionary to the server."""
        if not self.connected:
            return
        self._enqueue(packet)

    def _enqueue(self, packet):
        import json
        payload = json.dumps(packet).encode("utf-8")
        self.out_queue.append(_HEADER.pack(len(payload)) + payload)

    def poll_events(self):
        """Return list of all parsed events received from the server."""
        events = []
        while True:
            try:
                events.append(self.in_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def disconnect(self):
        self.running = False
        self.connected = False
        sock = self.sock
        self.sock = None
        if sock is not None:
            try:
                send_packet(sock, {"type": "disconnect"})
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        if self._thread and self._thread.is_alive() and threading.current_thread() != self._thread:
            self._thread.join(timeout=0.5)

    def _io_loop(self):
        buffer = bytearray()
        while self.running and self.sock:
            try:
                writable = [self.sock] if self.out_queue else []
                readable, write_ready, _ = select.select([self.sock], writable, [self.sock], 0.02)
            except (ValueError, OSError):
                if not self.error:
                    self.error = "Connection lost"
                break

            if self.sock in readable:
                try:
                    data = self.sock.recv(32768)
                    if not data:
                        if not self.error:
                            self.error = "Server closed connection"
                        break
                    buffer.extend(data)
                    for pkt in parse_packets(buffer):
                        self.in_queue.put(pkt)
                except OSError:
                    if not self.error:
                        self.error = "Connection reset by server"
                    break

            if self.sock in write_ready and self.out_queue:
                try:
                    while self.out_queue:
                        chunk = self.out_queue[0]
                        sent = self.sock.send(chunk)
                        if sent >= len(chunk):
                            self.out_queue.pop(0)
                        else:
                            self.out_queue[0] = chunk[sent:]
                            break
                except BlockingIOError:
                    pass
                except OSError:
                    if not self.error:
                        self.error = "Connection lost"
                    break

        if not self.error:
            self.error = "Server disconnected"
        self.connected = False
        self.running = False
        sock = self.sock
        self.sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
