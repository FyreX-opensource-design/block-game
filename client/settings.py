import json
import socket
import subprocess
from pathlib import Path
from server.profile import profile_dir

SETTINGS_NAME = "settings.json"

DEFAULT_SETTINGS = {
    "web_ui_enabled": False,
    "web_ui_port": 5000,
}


def get_local_ips():
    """Detect non-loopback IPv4 addresses on local network interfaces."""
    ips = []
    # 1. Routing UDP socket queries (no packets sent)
    for test_target in ("8.8.8.8", "1.1.1.1", "10.255.255.255", "192.168.255.255"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
            s.connect((test_target, 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and ip not in ips and not ip.startswith("127.") and not ip.startswith("169.254."):
                ips.append(ip)
        except Exception:
            pass

    # 2. Hostname resolution
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and ip not in ips and not ip.startswith("127.") and not ip.startswith("169.254."):
                ips.append(ip)
    except Exception:
        pass

    # 3. Linux hostname -I fallback
    try:
        res = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=0.5)
        for part in res.stdout.strip().split():
            if part and part not in ips and not part.startswith("127.") and not part.startswith("169.254.") and ":" not in part:
                ips.append(part)
    except Exception:
        pass

    # 4. Linux ip route get
    try:
        res = subprocess.run(["ip", "route", "get", "1.1.1.1"], capture_output=True, text=True, timeout=0.5)
        words = res.stdout.split()
        if "src" in words:
            src_ip = words[words.index("src") + 1]
            if src_ip not in ips and not src_ip.startswith("127.") and not src_ip.startswith("169.254."):
                ips.append(src_ip)
    except Exception:
        pass

    return ips


def load_settings(folder=None):
    root = Path(folder) if folder is not None else profile_dir()
    path = root / SETTINGS_NAME
    settings = dict(DEFAULT_SETTINGS)
    if path.is_file():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in DEFAULT_SETTINGS:
                        if isinstance(DEFAULT_SETTINGS[k], bool):
                            settings[k] = bool(v)
                        elif isinstance(DEFAULT_SETTINGS[k], int):
                            settings[k] = int(v)
                        else:
                            settings[k] = v
        except Exception:
            pass
    return settings


def save_settings(settings, folder=None):
    root = Path(folder) if folder is not None else profile_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / SETTINGS_NAME
        data = {
            "web_ui_enabled": bool(settings.get("web_ui_enabled", False)),
            "web_ui_port": int(settings.get("web_ui_port", 5000)),
        }
        path.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

