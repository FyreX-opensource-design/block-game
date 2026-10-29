#!/usr/bin/env python3
"""Multiplayer Test Environment Launcher

Sets up a local dedicated server and launches two separate client instances
with unique player UUIDs, usernames, and side-by-side window positioning
for local multiplayer testing.

Usage:
    ./.venv/bin/python run_mp_test.py
    ./.venv/bin/python run_mp_test.py --clean --seed 42
    ./.venv/bin/python run_mp_test.py --no-connect
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

DEFAULT_PORT = 25565
DEFAULT_UUID1 = "11111111-1111-4111-8111-111111111111"
DEFAULT_UUID2 = "22222222-2222-4222-8222-222222222222"
ROOT_DIR = Path(__file__).resolve().parent
ENV_DIR = ROOT_DIR / ".mp_env"


def wait_for_server(host, port, timeout=6.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.1)
    return False


def setup_profile(profile_dir, name, player_uuid, server_addr):
    profile_dir.mkdir(parents=True, exist_ok=True)
    # Write profile.json
    prof = {
        "uuid": player_uuid,
        "username": name,
        "skin": "default",
    }
    (profile_dir / "profile.json").write_text(json.dumps(prof, indent=2) + "\n", encoding="utf-8")

    # Write servers.json
    servers = [
        {"name": "Local Test Server", "address": server_addr},
    ]
    (profile_dir / "servers.json").write_text(json.dumps(servers, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Multiplayer Test Environment Launcher")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port (default: 25565)")
    parser.add_argument("--seed", type=int, default=1337, help="Server world seed (default: 1337)")
    parser.add_argument("--world", default="mp_test", help="World name (default: mp_test)")
    parser.add_argument("--name1", default="Player1", help="Client 1 username (default: Player1)")
    parser.add_argument("--name2", default="Player2", help="Client 2 username (default: Player2)")
    parser.add_argument("--uuid1", default=DEFAULT_UUID1, help="Client 1 UUID")
    parser.add_argument("--uuid2", default=DEFAULT_UUID2, help="Client 2 UUID")
    parser.add_argument("--width", type=int, default=720, help="Client window width (default: 720)")
    parser.add_argument("--height", type=int, default=540, help="Client window height (default: 540)")
    parser.add_argument("--clean", action="store_true", default=True, help="Clean existing test environment world & profiles (default: True)")
    parser.add_argument("--keep", "--no-clean", dest="clean", action="store_false", help="Keep existing test environment world & profiles")
    parser.add_argument("--random-seed", action="store_true", help="Use a random seed instead of default 1337")
    parser.add_argument("--no-connect", action="store_true", help="Launch clients to menu without auto-connecting")
    parser.add_argument("--server-only", action="store_true", help="Launch only the server")
    parser.add_argument("--client-only", action="store_true", help="Launch only the two clients")
    args = parser.parse_args()

    if args.random_seed:
        import random
        args.seed = random.randint(0, 2_147_483_647)

    python_bin = sys.executable

    if args.clean and ENV_DIR.exists():
        print(f"Cleaning test environment directory: {ENV_DIR}")
        shutil.rmtree(ENV_DIR, ignore_errors=True)

    ENV_DIR.mkdir(parents=True, exist_ok=True)
    saves_dir = ENV_DIR / "saves"
    prof1_dir = ENV_DIR / "profiles" / "client1"
    prof2_dir = ENV_DIR / "profiles" / "client2"

    server_addr = f"127.0.0.1:{args.port}"
    setup_profile(prof1_dir, args.name1, args.uuid1, server_addr)
    setup_profile(prof2_dir, args.name2, args.uuid2, server_addr)

    processes = []

    try:
        # 1. Start Server
        if not args.client_only:
            server_cmd = [
                python_bin,
                "-m",
                "server.network",
                "--server",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--world",
                args.world,
                "--saves-dir",
                str(saves_dir),
            ]
            if args.seed is not None:
                server_cmd.extend(["--seed", str(args.seed)])

            print(f"[Server] Starting dedicated server on {server_addr} (World: {args.world}, Seed: {args.seed})...")
            server_proc = subprocess.Popen(server_cmd, cwd=str(ROOT_DIR))
            processes.append(("Server", server_proc))

            print("[Server] Waiting for server to accept connections...")
            if not wait_for_server("127.0.0.1", args.port, timeout=6.0):
                print("[Server] ERROR: Timed out waiting for server to start!", file=sys.stderr)
                return 1
            print("[Server] Online and accepting connections!")

        if args.server_only:
            print("\nServer running in server-only mode. Press Ctrl+C to terminate.")
            while True:
                time.sleep(0.5)

        # 2. Launch Client 1 (Left window: pos 40, 50)
        c1_cmd = [
            python_bin,
            "play.py",
            "--profile-dir",
            str(prof1_dir),
            "--username",
            args.name1,
            "--uuid",
            args.uuid1,
            "--title",
            f"Block Game - {args.name1} (Client 1)",
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--pos",
            "40,50",
        ]
        if not args.no_connect:
            c1_cmd.extend(["--connect", server_addr])

        print(f"\n[Client 1] Launching {args.name1} (UUID: {args.uuid1})...")
        c1_proc = subprocess.Popen(c1_cmd, cwd=str(ROOT_DIR))
        processes.append((f"Client 1 ({args.name1})", c1_proc))

        # Stagger client startup slightly
        time.sleep(0.8)

        # 3. Launch Client 2 (Right window: pos 800, 50)
        c2_pos_x = 40 + args.width + 30
        c2_cmd = [
            python_bin,
            "play.py",
            "--profile-dir",
            str(prof2_dir),
            "--username",
            args.name2,
            "--uuid",
            args.uuid2,
            "--title",
            f"Block Game - {args.name2} (Client 2)",
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--pos",
            f"{c2_pos_x},50",
        ]
        if not args.no_connect:
            c2_cmd.extend(["--connect", server_addr])

        print(f"[Client 2] Launching {args.name2} (UUID: {args.uuid2})...")
        c2_proc = subprocess.Popen(c2_cmd, cwd=str(ROOT_DIR))
        processes.append((f"Client 2 ({args.name2})", c2_proc))

        print("\n" + "=" * 60)
        print(" Multiplayer Test Environment Active")
        print(f"  • Server:   {server_addr}")
        print(f"  • Client 1: {args.name1} (UUID: {args.uuid1}) [Left]")
        print(f"  • Client 2: {args.name2} (UUID: {args.uuid2}) [Right]")
        print("=" * 60)
        print("Press Ctrl+C to close all windows and stop the server.\n")

        # Monitor loop
        while True:
            time.sleep(0.5)
            # If both clients have exited, shut down
            clients_alive = [p for name, p in processes if "Client" in name and p.poll() is None]
            if not clients_alive and not args.server_only:
                print("All clients closed. Shutting down environment...")
                break

    except KeyboardInterrupt:
        print("\nShutdown requested (Ctrl+C)...")

    finally:
        print("Stopping all test environment processes...")
        for name, proc in reversed(processes):
            if proc.poll() is None:
                print(f"  Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("All processes stopped. Test environment clean.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

