import os
import stat
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "block-game.sh"
WINDOWS = ROOT / "block-game.exe"


def _write(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class LauncherTests(unittest.TestCase):
    def test_windows_launcher_is_a_gui_executable(self):
        data = WINDOWS.read_bytes()
        self.assertEqual(data[:2], b"MZ")
        pe = int.from_bytes(data[0x3C:0x40], "little")
        self.assertEqual(data[pe:pe + 4], b"PE\0\0")
        self.assertEqual(int.from_bytes(data[pe + 4:pe + 6], "little"), 0x8664)
        self.assertEqual(int.from_bytes(data[pe + 24:pe + 26], "little"), 0x20B)
        # IMAGE_SUBSYSTEM_WINDOWS_GUI, so itch does not open a spare console.
        self.assertEqual(int.from_bytes(data[pe + 92:pe + 94], "little"), 2)

    def test_linux_script_is_valid_bash(self):
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def _stage(self, home, py_version, with_virtualenv):
        game = Path(home) / "game"
        bindir = Path(home) / "bin"
        game.mkdir()
        bindir.mkdir()
        (game / "requirements.txt").write_text("numpy>=2.2\n")
        (game / "play.py").write_text("print('game')\n")
        script = (game / "block-game.sh")
        script.write_bytes(LAUNCHER.read_bytes())
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        flags = Path(home) / "flags"
        pip_log = Path(home) / "pip.log"
        ran = Path(home) / "ran.log"
        notes = Path(home) / "notify.log"
        _write(bindir / "python3", """#!/bin/bash
if [[ "$1" == "-c" && "$2" == *version_info* ]]; then
    printf '%s\\n' "$FAKE_PY_VERSION"
    exit 0
fi
if [[ "$1" == "-c" && "$2" == *hashlib* ]]; then
    /usr/bin/python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$3"
    exit 0
fi
if [[ "$1" == "-c" && "$2" == *import\\ numpy* ]]; then
    if [[ -f "$DEPS_FLAG" ]]; then exit 0; else exit 1; fi
fi
if [[ "$1" == "-m" && "$2" == "pip" ]]; then
    printf '%s\\n' "$*" >> "$PIP_LOG"
    touch "$DEPS_FLAG"
    exit 0
fi
if [[ "$1" == "-m" && "$2" == "virtualenv" ]]; then
    exit 1
fi
if [[ "$1" == "-m" && "$2" == "ensurepip" ]]; then
    exit 0
fi
printf '%s\\n' "$*" > "$RAN_LOG"
exit 0
""")
        _write(bindir / "notify-send", """#!/bin/bash
printf '%s\\n' "$*" >> "$NOTIFY_LOG"
""")
        if with_virtualenv:
            _write(bindir / "virtualenv", """#!/bin/bash
dest=""
skip=""
for arg in "$@"; do
    if [[ -n "$skip" ]]; then skip=""; continue; fi
    case "$arg" in
        --python) skip=1; continue;;
        --python=*) continue;;
    esac
    dest="$arg"
done
mkdir -p "$dest/bin"
cp "$STUB_PYTHON" "$dest/bin/python"
chmod +x "$dest/bin/python"
""")
        for tool in ("mkdir", "rm", "cp", "chmod", "tr", "cat", "dirname", "touch"):
            os.symlink(f"/usr/bin/{tool}", bindir / tool)
        env = os.environ.copy()
        env["PATH"] = str(bindir)
        env["HOME"] = home
        env["XDG_DATA_HOME"] = str(Path(home) / ".local" / "share")
        env["FAKE_PY_VERSION"] = py_version
        env["DEPS_FLAG"] = str(flags)
        env["PIP_LOG"] = str(pip_log)
        env["RAN_LOG"] = str(ran)
        env["NOTIFY_LOG"] = str(notes)
        env["STUB_PYTHON"] = str(bindir / "python3")
        env.pop("NOTIFY_SEND", None)
        return game, env, pip_log, ran, notes

    def test_missing_python_and_virtualenv_are_reported(self):
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            game, env, _pip, _ran, notes = self._stage(home, "3.9", with_virtualenv=False)
            result = subprocess.run(["/usr/bin/bash", str(game / "block-game.sh")], env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            text = notes.read_text()
            self.assertIn("Python 3.10", text)
            self.assertIn("python-virtualenv", text)
            self.assertNotIn("notify-send", result.stderr)

    def test_virtualenv_then_desktop_file_then_the_game(self):
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            game, env, pip_log, ran, notes = self._stage(home, "3.12", with_virtualenv=True)
            result = subprocess.run(["/usr/bin/bash", str(game / "block-game.sh")], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(notes.exists())
            desktop = Path(env["XDG_DATA_HOME"]) / "applications" / "block-game.desktop"
            entry = desktop.read_text()
            venv_python = game / "runtime" / ".venv" / "bin" / "python"
            self.assertIn(f'"{venv_python}"', entry)
            self.assertIn(f'"{game / "play.py"}"', entry)
            self.assertIn(f"Path={game}", entry)
            self.assertTrue((game / "runtime" / ".requirements.sha256").is_file())
            self.assertIn("pip", pip_log.read_text())
            self.assertEqual(ran.read_text().strip(), str(game / "play.py"))
            first_pip = pip_log.read_text()
            again = subprocess.run(["/usr/bin/bash", str(game / "block-game.sh")], env=env, text=True, capture_output=True)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(pip_log.read_text(), first_pip)
            self.assertEqual(ran.read_text().strip(), str(game / "play.py"))
