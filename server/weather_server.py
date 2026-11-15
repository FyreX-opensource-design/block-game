"""Weather process.

Connects to a world server, follows its clock with Precision Time
Protocol, and sends back the cloud deck and the rain. Run it on this
machine or another one:

    python -m server.weather_server --world 127.0.0.1:25565

The process is separate from the world so it can sit in its own cgroup.
"""

import argparse
import time

from server.servers import parse_address
from server.weather import SlaveClock, WeatherField, dump_packet, read_packets


class WeatherClient:
    """PTP slave. The world server is the master clock."""

    def __init__(self, host, port, timeout=5.0):
        import socket

        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.settimeout(0.05)
        self.buf = bytearray()
        self.out = []
        self.clock = SlaveClock()
        self.field = None
        self.focus = (0.0, 0.0)
        self.overrides = []
        self.day_time = 1000.0
        self.day_at = None
        self.acc = 0.0
        self._last_master = None
        self.alive = True
        self.out.append({"type": "weather_hello", "protocol": 1})

    def close(self):
        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass

    def pump(self):
        if not self.alive:
            return False
        self._send()
        self._recv()
        if not self.alive or self.field is None or not self.clock.synced:
            return self.alive
        now = self.clock.master_now(time.time())
        if self._last_master is None:
            self._last_master = now
            return True
        dt = max(0.0, min(0.25, now - self._last_master))
        self._last_master = now
        self.acc += dt
        if self.acc < 0.1:
            return True
        step = min(self.acc, 0.25)
        self.acc = 0.0
        self.field.recenter(self.focus[0], self.focus[1])
        day = self.estimated_day(now)
        heights, rainfall = self.field.maps(self.overrides)
        self.field.step(step, heights, rainfall, day_time=day)
        self.out.append(self.field.snapshot(day))
        self._send()
        return True

    def estimated_day(self, master_now):
        if self.day_at is None:
            return self.day_time
        return (self.day_time + (float(master_now) - self.day_at) * 20.0) % 24000.0

    def _on_packet(self, packet):
        kind = packet.get("type")
        if kind == "weather_welcome":
            self.field = WeatherField(int(packet.get("seed", 0)))
            self.day_time = float(packet.get("day_time", 1000.0))
            self.day_at = None
        elif kind == "ptp_sync":
            t2 = time.time()
            t3 = time.time()
            self.out.append(
                {
                    "type": "ptp_delay_req",
                    "t1": packet.get("t1"),
                    "t2": t2,
                    "t3": t3,
                }
            )
        elif kind == "ptp_delay_resp":
            t1, t2, t3, t4 = packet.get("t1"), packet.get("t2"), packet.get("t3"), packet.get("t4")
            if None not in (t1, t2, t3, t4):
                self.clock.observe(t1, t2, t3, t4)
            if "day_time" in packet and packet.get("t4") is not None:
                self.day_time = float(packet["day_time"])
                self.day_at = float(packet["t4"])
        elif kind == "cover":
            focus = packet.get("focus") or [0.0, 0.0]
            try:
                self.focus = (float(focus[0]), float(focus[1]))
            except (TypeError, ValueError, IndexError):
                self.focus = (0.0, 0.0)
            tops = packet.get("tops") or []
            self.overrides = tops if isinstance(tops, list) else []
        elif kind == "disconnect":
            self.alive = False

    def _recv(self):
        import socket

        try:
            data = self.sock.recv(65536)
        except socket.timeout:
            return
        except OSError:
            self.alive = False
            return
        if not data:
            self.alive = False
            return
        self.buf.extend(data)
        for packet in read_packets(self.buf):
            self._on_packet(packet)

    def _send(self):
        while self.out:
            try:
                self.sock.sendall(dump_packet(self.out[0]))
            except OSError:
                self.alive = False
                return
            self.out.pop(0)


def serve(address):
    """Follow a world server until interrupted, reconnecting if it drops."""
    host, port = parse_address(address)
    while True:
        try:
            client = WeatherClient(host, port)
        except OSError:
            time.sleep(0.5)
            continue
        try:
            while client.alive:
                client.pump()
                time.sleep(0.02)
        except KeyboardInterrupt:
            client.close()
            return
        finally:
            client.close()
        time.sleep(0.5)


def spawn_weather_process(address):
    """Start a weather server aimed at a world. The caller owns the process."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return subprocess.Popen(
        [sys.executable, "-m", "server.weather_server", "--world", str(address)],
        cwd=str(root),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Block Game weather server")
    parser.add_argument(
        "--world",
        required=True,
        help="World server host:port. That server is the PTP master clock.",
    )
    args = parser.parse_args(argv)
    host, port = parse_address(args.world)
    print(f"Weather server following {host}:{port} (world server is the master clock).")
    print("This process is separate from the world, so it can run under its own cgroup.")
    try:
        serve(args.world)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    main()
