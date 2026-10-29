import json
import uuid
from pathlib import Path

import numpy as np

from server.blocks import ASSETS
from server.images import read_png
from server.storage import _data_home

SKIN_SIZE = 64
PROFILE_NAME = "profile.json"
DEFAULT_SKIN = "default"


def profile_dir():
    path = _data_home() / "block-game"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return path


def skins_dir():
    path = profile_dir() / "skins"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return path


def profile_path():
    return profile_dir() / PROFILE_NAME


def load_profile(folder=None):
    root = Path(folder) if folder is not None else profile_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    path = root / PROFILE_NAME
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
        player_id = str(data.get("uuid", ""))
        skin = str(data.get("skin", DEFAULT_SKIN)) or DEFAULT_SKIN
        username = str(data.get("username", "Player"))
        if not _valid_uuid(player_id):
            player_id = str(uuid.uuid4())
            _write_profile(path, player_id, skin, username)
        return {"uuid": player_id, "skin": skin, "username": username}
    player_id = str(uuid.uuid4())
    skin = DEFAULT_SKIN
    username = "Player"
    _write_profile(path, player_id, skin, username)
    return {"uuid": player_id, "skin": skin, "username": username}


def save_profile(profile, folder=None):
    root = Path(folder) if folder is not None else profile_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    player_id = str(profile["uuid"])
    skin = str(profile.get("skin", DEFAULT_SKIN)) or DEFAULT_SKIN
    username = str(profile.get("username", "Player"))
    if not _valid_uuid(player_id):
        raise ValueError("profile uuid is invalid")
    _write_profile(root / PROFILE_NAME, player_id, skin, username)


def skin_path(name, folder=None):
    name = str(name or DEFAULT_SKIN)
    if name == DEFAULT_SKIN:
        return ASSETS / "skins" / "default.png"
    root = Path(folder) if folder is not None else skins_dir()
    path = (root / name).resolve()
    if path.suffix.lower() != ".png":
        path = Path(str(path) + ".png")
    return path


def load_skin_rgba(name=DEFAULT_SKIN, folder=None):
    path = skin_path(name, folder)
    if not path.is_file():
        path = skin_path(DEFAULT_SKIN)
    image = read_png(path)
    if image.shape[0] != SKIN_SIZE or image.shape[1] != SKIN_SIZE:
        raise ValueError(f"{path} must be a {SKIN_SIZE}x{SKIN_SIZE} skin")
    return np.ascontiguousarray(image, dtype=np.uint8)


def _write_profile(path, player_id, skin, username="Player"):
    try:
        Path(path).write_text(json.dumps({"uuid": player_id, "skin": skin, "username": username}, indent=2) + "\n")
    except OSError:
        pass


def _valid_uuid(value):
    try:
        uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return False
    return True
