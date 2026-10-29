import json
import os
import re
import shutil
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from server.blocks import CHUNK_SIZE

CHUNK_MAGIC = b"BG02"
CHUNK_BYTES = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE
_HEADER = struct.Struct("<iiiI")
META_NAME = "meta.json"


def default_saves_dir():
    return _data_home() / "block-game" / "worlds"


def _data_home():
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local)
        return Path.home() / "AppData" / "Local"
    configured = os.environ.get("XDG_DATA_HOME")
    if configured:
        return Path(configured)
    return Path.home() / ".local" / "share"


def display_path(path):
    text = str(path)
    home = str(Path.home())
    if text == home or text.startswith(home + os.sep):
        return "~" + text[len(home) :]
    return text


class WorldInfo:
    def __init__(self, world_id, name, seed, created, last_played, player=None, gamemode="survival", day_time=1000.0):
        self.id = world_id
        self.name = name
        self.seed = int(seed)
        self.created = float(created)
        self.last_played = float(last_played)
        self.player = player
        self.gamemode = str(gamemode or "survival").lower()
        self.day_time = float(day_time)

    def to_dict(self):
        return {
            "name": self.name,
            "seed": self.seed,
            "created": self.created,
            "last_played": self.last_played,
            "player": self.player,
            "gamemode": self.gamemode,
            "day_time": self.day_time,
        }


class ChunkStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def path_for(self, cx, cy, cz):
        return self.folder / f"{int(cx)}_{int(cy)}_{int(cz)}.bin"

    def write_chunk(self, chunk):
        raw = np.ascontiguousarray(chunk.data, dtype=np.uint8).tobytes()
        if len(raw) != CHUNK_BYTES:
            raise ValueError("chunk has the wrong size")
        payload = _HEADER.pack(int(chunk.cx), int(chunk.cy), int(chunk.cz), len(raw)) + zlib.compress(raw, level=1)
        path = self.path_for(chunk.cx, chunk.cy, chunk.cz)
        temporary = path.with_suffix(".bin.tmp")
        temporary.write_bytes(CHUNK_MAGIC + payload)
        os.replace(temporary, path)

    def read_chunk(self, cx, cy, cz):
        path = self.path_for(cx, cy, cz)
        if not path.is_file():
            return None
        blob = path.read_bytes()
        if len(blob) < 4 + _HEADER.size or blob[:4] != CHUNK_MAGIC:
            return None
        saved_x, saved_y, saved_z, size = _HEADER.unpack_from(blob, 4)
        if saved_x != int(cx) or saved_y != int(cy) or saved_z != int(cz) or size != CHUNK_BYTES:
            return None
        try:
            raw = zlib.decompress(blob[4 + _HEADER.size :])
        except zlib.error:
            return None
        if len(raw) != CHUNK_BYTES:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE).copy()


class WorldCatalog:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else default_saves_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def list_worlds(self):
        worlds = []
        for entry in self.root.iterdir():
            if not entry.is_dir() or not (entry / META_NAME).is_file():
                continue
            try:
                worlds.append(self.read(entry.name))
            except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
                continue
        worlds.sort(key=lambda info: info.last_played, reverse=True)
        return worlds

    def read(self, world_id):
        folder = self._folder(world_id)
        data = json.loads((folder / META_NAME).read_text(encoding="utf-8"))
        return WorldInfo(
            world_id,
            str(data["name"]),
            int(data["seed"]),
            float(data.get("created", 0)),
            float(data.get("last_played", 0)),
            data.get("player"),
            gamemode=str(data.get("gamemode", "survival")),
            day_time=float(data.get("day_time", 1000.0)),
        )

    def create(self, name, seed, gamemode="survival"):
        world_id = self._allocate_id(name)
        now = time.time()
        info = WorldInfo(world_id, name.strip(), int(seed), now, now, None, gamemode=gamemode, day_time=1000.0)
        folder = self.root / world_id
        folder.mkdir(parents=True)
        self.write(info)
        return info

    def write(self, info):
        folder = self._folder(info.id)
        folder.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(info.to_dict(), indent=2)
        temporary = folder / "meta.json.tmp"
        temporary.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary, folder / META_NAME)

    def delete(self, world_id):
        folder = self._folder(world_id)
        if folder.is_dir():
            shutil.rmtree(folder)

    def store_for(self, world_id):
        return ChunkStore(self._folder(world_id) / "chunks")

    def _folder(self, world_id):
        if world_id != Path(world_id).name or world_id in ("", ".", ".."):
            raise ValueError("invalid world id")
        folder = (self.root / world_id).resolve()
        if folder.parent != self.root.resolve():
            raise ValueError("invalid world id")
        return folder

    def _allocate_id(self, name):
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "world"
        candidate = base
        number = 2
        while (self.root / candidate).exists():
            candidate = f"{base}-{number}"
            number += 1
        return candidate
