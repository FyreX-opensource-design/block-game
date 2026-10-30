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

from server.blocks import CHUNK_SIZE, block_from_index, block_index

CHUNK_MAGIC = b"BG03"
LEGACY_CHUNK_MAGIC = b"BG02"
CHUNK_BYTES = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE
_HEADER = struct.Struct("<iiiII")
_LEGACY_HEADER = struct.Struct("<iiiI")
META_NAME = "meta.json"
# Blocks introduced after the last BG02 release. BG02 chunks stored global block
# indices from before these existed, so they are skipped when mapping legacy data.
_LEGACY_NEW_BLOCKS = frozenset({"oak_sapling", "pine_sapling"})
# The block registry keeps this index free for compatibility with old saves.
_LEGACY_RESERVED_INDEX = 99


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
    def __init__(self, world_id, name, seed, created, last_played, player=None, gamemode="survival", day_time=1000.0, spawn=None):
        self.id = world_id
        self.name = name
        self.seed = int(seed)
        self.created = float(created)
        self.last_played = float(last_played)
        self.player = player
        self.gamemode = str(gamemode or "survival").lower()
        self.day_time = float(day_time)
        self.spawn = spawn

    def to_dict(self):
        return {
            "name": self.name,
            "seed": self.seed,
            "created": self.created,
            "last_played": self.last_played,
            "player": self.player,
            "gamemode": self.gamemode,
            "day_time": self.day_time,
            "spawn": self.spawn,
        }


def _palette_and_local(data):
    """Return a block-id palette and equivalent indices local to that palette."""
    used = np.unique(data)
    palette = []
    for value in used:
        block = block_from_index(int(value))
        palette.append(block.id if block is not None else "")
    local = np.searchsorted(used, data).astype(np.uint16)
    return palette, local


def _palette_lookup(palette):
    """Map palette positions to indices in the current block registry."""
    lookup = np.zeros(len(palette), dtype=np.uint16)
    for position, block_id in enumerate(palette):
        if not block_id:
            continue
        try:
            lookup[position] = block_index(str(block_id))
        except KeyError:
            lookup[position] = 0
    return lookup


def _legacy_lookup():
    """Map BG02 global indices to the current registry (pre-sapling layout)."""
    from server.blocks import BY_INDEX

    # Rebuild the registry as it looked before saplings were added. The reserved
    # hole is recreated at its absolute index rather than carried over, because
    # removing saplings shifts every later index.
    palette = [""]
    for index, block in enumerate(BY_INDEX):
        if index == 0 or block is None or block.id in _LEGACY_NEW_BLOCKS:
            continue
        if len(palette) == _LEGACY_RESERVED_INDEX:
            palette.append("")
        palette.append(block.id)
    return _palette_lookup(palette)


def _remap(data, lookup):
    result = np.zeros(data.shape, dtype=np.uint16)
    valid = data < len(lookup)
    result[valid] = lookup[data[valid]]
    return result


class ChunkStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def path_for(self, cx, cy, cz):
        return self.folder / f"{int(cx)}_{int(cy)}_{int(cz)}.bin"

    def meta_path(self, name):
        return self.folder.parent / Path(name).name

    def read_meta(self, name):
        path = self.meta_path(name)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    def write_meta(self, name, data):
        path = self.meta_path(name)
        try:
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            return False
        return True

    def write_chunk(self, chunk):
        data = np.ascontiguousarray(chunk.data, dtype=np.uint16)
        if data.size != CHUNK_BYTES:
            raise ValueError("chunk has the wrong size")
        palette, local = _palette_and_local(data)
        raw = local.tobytes()
        palette_bytes = json.dumps(palette, separators=(",", ":")).encode("utf-8")
        payload = (
            _HEADER.pack(int(chunk.cx), int(chunk.cy), int(chunk.cz), len(palette_bytes), len(raw))
            + palette_bytes
            + zlib.compress(raw, level=1)
        )
        path = self.path_for(chunk.cx, chunk.cy, chunk.cz)
        temporary = path.with_suffix(".bin.tmp")
        temporary.write_bytes(CHUNK_MAGIC + payload)
        os.replace(temporary, path)

    def read_chunk(self, cx, cy, cz):
        path = self.path_for(cx, cy, cz)
        if not path.is_file():
            return None
        blob = path.read_bytes()
        if blob[:4] == LEGACY_CHUNK_MAGIC:
            return self._read_legacy(blob, cx, cy, cz)
        if len(blob) < 4 + _HEADER.size or blob[:4] != CHUNK_MAGIC:
            return None
        saved_x, saved_y, saved_z, palette_len, data_len = _HEADER.unpack_from(blob, 4)
        if (saved_x, saved_y, saved_z) != (int(cx), int(cy), int(cz)):
            return None
        offset = 4 + _HEADER.size
        end = offset + palette_len
        if palette_len <= 0 or end > len(blob) or data_len != CHUNK_BYTES * 2:
            return None
        try:
            palette = json.loads(blob[offset:end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(palette, list) or not palette:
            return None
        try:
            raw = zlib.decompress(blob[end:])
        except zlib.error:
            return None
        if len(raw) != data_len:
            return None
        local = np.frombuffer(raw, dtype=np.uint16).reshape(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE)
        return _remap(local, _palette_lookup(palette))

    def _read_legacy(self, blob, cx, cy, cz):
        if len(blob) < 4 + _LEGACY_HEADER.size:
            return None
        saved_x, saved_y, saved_z, size = _LEGACY_HEADER.unpack_from(blob, 4)
        if (saved_x, saved_y, saved_z) != (int(cx), int(cy), int(cz)):
            return None
        if size not in (CHUNK_BYTES, CHUNK_BYTES * 2):
            return None
        try:
            raw = zlib.decompress(blob[4 + _LEGACY_HEADER.size :])
        except zlib.error:
            return None
        if len(raw) != size:
            return None
        if size == CHUNK_BYTES:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
        else:
            data = np.frombuffer(raw, dtype=np.uint16)
        return _remap(data.reshape(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), _legacy_lookup())


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
            spawn=data.get("spawn"),
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
