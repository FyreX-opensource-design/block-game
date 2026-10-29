import json
from pathlib import Path

from server.profile import profile_dir

SERVERS_FILE = "servers.json"
DEFAULT_PORT = 25565
DEFAULT_SERVERS = [
    {
        "name": "Local Server",
        "address": "127.0.0.1:25565",
    }
]


def servers_path(folder=None):
    root = Path(folder) if folder is not None else profile_dir()
    return root / SERVERS_FILE


def load_servers(folder=None):
    path = servers_path(folder)
    if not path.is_file():
        servers = [dict(s) for s in DEFAULT_SERVERS]
        save_servers(servers, folder)
        return servers
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            servers = []
            for item in data:
                if isinstance(item, dict) and "name" in item and "address" in item:
                    servers.append({"name": str(item["name"]), "address": str(item["address"])})
            return servers
    except (json.JSONDecodeError, OSError):
        pass
    return [dict(s) for s in DEFAULT_SERVERS]


def save_servers(servers, folder=None):
    path = servers_path(folder)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    clean = []
    for item in servers:
        if isinstance(item, dict) and "name" in item and "address" in item:
            clean.append({"name": str(item["name"]).strip(), "address": str(item["address"]).strip()})
    try:
        path.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def parse_address(address, default_port=DEFAULT_PORT):
    raw = str(address or "").strip()
    if not raw:
        return "127.0.0.1", default_port
    if ":" in raw:
        host, port_str = raw.rsplit(":", 1)
        host = host.strip() or "127.0.0.1"
        try:
            port = int(port_str.strip())
            return host, port
        except ValueError:
            return host, default_port
    return raw, default_port

