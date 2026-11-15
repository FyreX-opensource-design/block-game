"""Clouds and the rain that falls out of them.

The deck sits at a fixed height. Wind carries a moisture field across a
grid. The height of each column — terrain or a building, whichever is
taller — turns that wind, lifts moisture where the air is forced uphill,
and dries the lee side. A column that reaches the deck blocks the cloud
and the rain under it.

The world server is the master clock. A weather process elsewhere asks
for the time with the same four timestamps Precision Time Protocol uses,
then steps this field on the master's clock.
"""

import base64
import json
import math
import struct
import zlib

import numpy as np

CLOUD_Y = 200
CELL = 8
GRID = 40

_HEADER = struct.Struct(">I")


def ptp_offset(t1, t2, t3, t4):
    """Slave-clock error and the one-way delay.

    t1 is when the master sent Sync, t2 when the slave received it,
    t3 when the slave sent Delay_Req, and t4 when the master received
    that request. The offset is slave time minus master time.
    """
    offset = ((float(t2) - float(t1)) - (float(t4) - float(t3))) / 2.0
    delay = ((float(t2) - float(t1)) + (float(t4) - float(t3))) / 2.0
    return offset, delay


class SlaveClock:
    """A clock corrected so it reads the master's time."""

    def __init__(self):
        self.offset = 0.0
        self.delay = 0.0
        self.synced = False

    def observe(self, t1, t2, t3, t4):
        self.offset, self.delay = ptp_offset(t1, t2, t3, t4)
        self.synced = True

    def master_now(self, now):
        return float(now) - self.offset


def _storm_pattern(origin, cell, size, seed):
    """Where storm clouds can hold together. Most of the deck is fair sky."""
    from server.noise import fbm_grid

    n = int(size)
    xs = float(origin[0]) + (np.arange(n) + 0.5) * float(cell)
    zs = float(origin[1]) + (np.arange(n) + 0.5) * float(cell)
    xx, zz = np.meshgrid(xs, zs)
    broad = fbm_grid(xx * 0.016, zz * 0.016, 3, int(seed) + 9100)
    detail = fbm_grid(xx * 0.042 + 4.0, zz * 0.042 - 2.0, 2, int(seed) + 9300)
    field = broad * 0.75 + detail * 0.25
    edge = np.clip((field - 0.60) / 0.14, 0.0, 1.0)
    return edge * edge * (3.0 - 2.0 * edge)


def rainfall_of(biome):
    value = getattr(biome, "rainfall", None)
    if value is None:
        return 1.0
    return float(value)


def prevailing_wind(day_time, seed):
    """Steady wind for this moment of the day, in blocks per second."""
    angle = (float(day_time) / 24000.0) * math.tau + (int(seed) % 997) * 0.001
    speed = 7.0
    return (math.cos(angle) * speed, math.sin(angle) * speed)


def dump_packet(packet):
    payload = json.dumps(packet).encode("utf-8")
    return _HEADER.pack(len(payload)) + payload


def read_packets(buffer):
    packets = []
    while len(buffer) >= 4:
        length = _HEADER.unpack_from(buffer, 0)[0]
        if length < 0 or len(buffer) < 4 + length:
            break
        data = buffer[4 : 4 + length]
        del buffer[: 4 + length]
        try:
            packets.append(json.loads(data.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return packets


def pack_grid(values):
    raw = np.clip(np.asarray(values) * 255.0, 0, 255).astype(np.uint8).tobytes()
    return base64.b64encode(zlib.compress(raw, 1)).decode("ascii")


def unpack_grid(text, size):
    raw = zlib.decompress(base64.b64decode(text))
    grid = np.frombuffer(raw, dtype=np.uint8)
    if grid.size != int(size) * int(size):
        raise ValueError("weather grid size mismatch")
    return grid.reshape(int(size), int(size)).copy()


def rain_at(state, x, z):
    """Rain strength at a world column, from 0 to 1."""
    if not state:
        return 0.0
    try:
        cell = float(state["cell"])
        ox, oz = state["origin"]
        size = int(state["size"])
        j = int((float(x) - float(ox)) / cell)
        i = int((float(z) - float(oz)) / cell)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return 0.0
    if cell <= 0.0 or i < 0 or j < 0 or i >= size or j >= size:
        return 0.0
    grid = state.get("_rain")
    if grid is None:
        try:
            grid = unpack_grid(state["rain"], size)
        except (KeyError, ValueError, zlib.error):
            return 0.0
        state["_rain"] = grid
    return float(grid[i, j]) / 255.0


def _shift_upwind(mass, di, dj):
    """Slide each cell's mass one step back against the wind, without wrapping."""
    out = np.zeros_like(mass)
    n = mass.shape[0]
    shift_i = -int(di)
    shift_j = -int(dj)

    def span(shift):
        if shift > 0:
            return slice(0, n - shift), slice(shift, n)
        if shift < 0:
            return slice(-shift, n), slice(0, n + shift)
        return slice(0, n), slice(0, n)

    src_i, dst_i = span(shift_i)
    src_j, dst_j = span(shift_j)
    out[dst_i, dst_j] = mass[src_i, src_j]
    return out


def _sample(field, si, sj):
    n = field.shape[0]
    outside = (si < 0.0) | (sj < 0.0) | (si > n - 1) | (sj > n - 1)
    si = np.clip(si, 0.0, n - 1.001)
    sj = np.clip(sj, 0.0, n - 1.001)
    i0 = np.floor(si).astype(np.int32)
    j0 = np.floor(sj).astype(np.int32)
    i1 = np.minimum(i0 + 1, n - 1)
    j1 = np.minimum(j0 + 1, n - 1)
    ti = si - i0
    tj = sj - j0
    sampled = (
        field[i0, j0] * (1.0 - ti) * (1.0 - tj)
        + field[i0, j1] * (1.0 - ti) * tj
        + field[i1, j0] * ti * (1.0 - tj)
        + field[i1, j1] * ti * tj
    )
    return sampled, outside


class WeatherField:
    """Moisture and rainfall on one square of the world."""

    def __init__(self, seed, size=GRID, cell=CELL):
        self.seed = int(seed)
        self.size = int(size)
        self.cell = int(cell)
        self.density = np.zeros((self.size, self.size), dtype=np.float64)
        self.rain = np.zeros((self.size, self.size), dtype=np.float64)
        self.wind = np.array(prevailing_wind(1000.0, self.seed), dtype=np.float64)
        span = self.size * self.cell
        self.origin = (-span / 2.0, -span / 2.0)
        self._biome_origin = None
        self._biome_rain = np.ones((self.size, self.size), dtype=np.float64)
        self._seed_density()

    def _seed_density(self):
        pattern = _storm_pattern(self.origin, self.cell, self.size, self.seed)
        # Fair sky stays below the rain line. Storm masses start already wet.
        self.density[:] = 0.2 + 0.95 * pattern

    def recenter(self, focus_x, focus_z):
        cx = self.origin[0] + self.size * self.cell * 0.5
        cz = self.origin[1] + self.size * self.cell * 0.5
        dx = int(np.clip(round((float(focus_x) - cx) / self.cell), -self.size, self.size))
        dz = int(np.clip(round((float(focus_z) - cz) / self.cell), -self.size, self.size))
        if dx == 0 and dz == 0:
            return
        self.density = np.roll(self.density, shift=(-dz, -dx), axis=(0, 1))
        self.rain = np.roll(self.rain, shift=(-dz, -dx), axis=(0, 1))
        if dz > 0:
            self.density[-dz:, :] = 0.2
            self.rain[-dz:, :] = 0.0
        elif dz < 0:
            self.density[:-dz, :] = 0.2
            self.rain[:-dz, :] = 0.0
        if dx > 0:
            self.density[:, -dx:] = 0.2
            self.rain[:, -dx:] = 0.0
        elif dx < 0:
            self.density[:, :-dx] = 0.2
            self.rain[:, :-dx] = 0.0
        self.origin = (self.origin[0] + dx * self.cell, self.origin[1] + dz * self.cell)

    def step(self, dt, heights, rainfall, day_time=None):
        """Advance the clouds. Heights and rainfall are arrays aligned to the grid."""
        dt = float(max(0.0, min(0.5, dt)))
        if day_time is not None:
            self.wind = np.array(prevailing_wind(day_time, self.seed), dtype=np.float64)
        heights = np.asarray(heights, dtype=np.float64)
        rainfall = np.clip(np.asarray(rainfall, dtype=np.float64), 0.0, 2.0)
        n = self.size
        cell = float(self.cell)
        wx = float(self.wind[0])
        wz = float(self.wind[1])
        speed = math.hypot(wx, wz) or 1.0
        ux, uz = wx / speed, wz / speed
        px, pz = -uz, ux
        grad_z, grad_x = np.gradient(heights, cell)
        side = grad_x * px + grad_z * pz
        along = grad_x * ux + grad_z * uz
        turn = np.clip(side / 48.0, -1.2, 1.2)
        slow = np.clip((CLOUD_Y - heights) / 36.0, 0.0, 1.0)
        blocked = heights >= (CLOUD_Y - 1.0)
        slow = np.where(blocked, 0.0, slow)
        cosine = np.cos(turn)
        sine = np.sin(turn)
        local_x = (ux * cosine + px * sine) * speed * slow
        local_z = (uz * cosine + pz * sine) * speed * slow
        ii, jj = np.mgrid[0:n, 0:n]
        si = ii - (local_z * dt) / cell
        sj = jj - (local_x * dt) / cell
        advected, outside = _sample(self.density, si.astype(np.float64), sj.astype(np.float64))
        advected = np.where(outside, 0.35 * rainfall, advected)
        lift = np.clip(along / 24.0, -1.0, 1.0)
        windward = np.clip(lift, 0.0, 1.0)
        leeward = np.clip(-lift, 0.0, 1.0)
        # Moisture gathers in storm masses. The rest of the deck dries, so
        # most clouds stay fair and only the grey ones keep raining.
        pattern = _storm_pattern(self.origin, self.cell, n, self.seed)
        density = advected + dt * (0.28 * rainfall * pattern + 0.55 * windward * rainfall)
        density -= dt * 0.70 * leeward * density
        density = np.clip(density, 0.0, 1.5)
        piled = _shift_upwind(
            np.where(blocked, density, 0.0),
            int(np.sign(uz)) if abs(uz) >= 0.2 else 0,
            int(np.sign(ux)) if abs(ux) >= 0.2 else 0,
        )
        density = np.where(blocked, 0.0, density) + piled
        density = np.clip(density, 0.0, 1.5)
        excess = np.maximum(0.0, density - 0.5)
        rain = np.clip(excess * rainfall * (1.0 - 0.8 * leeward), 0.0, 1.0)
        rain = np.where(blocked, 0.0, rain)
        density = np.where(blocked, 0.0, np.clip(density - rain * dt * 0.4, 0.0, 1.5))
        self.density = density
        self.rain = rain
        return self.density, self.rain

    def maps(self, overrides):
        """Terrain height and biome rainfall for the current grid, plus built columns."""
        from server.biomes import BIOMES, biome_at
        from server.noise import terrain_grid

        n = self.size
        cell = float(self.cell)
        ox, oz = self.origin
        xs = ox + (np.arange(n) + 0.5) * cell
        zs = oz + (np.arange(n) + 0.5) * cell
        xx, zz = np.meshgrid(xs, zs)
        heights = np.asarray(terrain_grid(xx, zz, self.seed), dtype=np.float64)
        if self._biome_origin != self.origin:
            rain = np.ones((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    biome = biome_at(int(xx[i, j]), int(zz[i, j]), self.seed, BIOMES)
                    rain[i, j] = rainfall_of(biome)
            self._biome_rain = rain
            self._biome_origin = self.origin
        rainfall = self._biome_rain
        for item in overrides or ():
            try:
                x, z, y = item
                j = int((float(x) - ox) / cell)
                i = int((float(z) - oz) / cell)
            except (TypeError, ValueError):
                continue
            if 0 <= i < n and 0 <= j < n:
                heights[i, j] = float(y)
        return heights, rainfall

    def snapshot(self, day_time=None):
        return {
            "type": "weather_state",
            "origin": [float(self.origin[0]), float(self.origin[1])],
            "cell": self.cell,
            "size": self.size,
            "cloud_y": CLOUD_Y,
            "wind": [float(self.wind[0]), float(self.wind[1])],
            "density": pack_grid(self.density / 1.5),
            "rain": pack_grid(self.rain),
            "day_time": None if day_time is None else float(day_time),
        }
