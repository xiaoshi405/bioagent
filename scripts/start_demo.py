"""Start the local tool API, agent and web interface; stop only owned subprocesses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_environment import check_environment


def port_is_available(port):
    # Check active listeners first: Windows SO_REUSEADDR alone permits sharing.
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return False
    except OSError:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Match uvicorn's reuse policy so a previous TIME_WAIT is not a false conflict.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def ready(url):
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def stop_owned(children):
    for process, _, _ in reversed(children):
        if process.poll() is None:
            process.terminate()
    for process, _, _ in reversed(children):
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-port", type=int, default=8501)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        report = check_environment("agent")
    except (OSError, ValueError, ImportError):
        print("Invalid local environment; run python scripts/check_environment.py --mode agent.", flush=True)
        return 1
    if report["status"] != "PASS":
        print(json.dumps(report, indent=2), flush=True)
        return 1
    from app.core.config import Config
    cfg = Config()
    ports = [cfg.BIO_API_PORT, cfg.SERVER_PORT, args.web_port]
    if len(set(ports)) != 3 or any(port < 1 or port > 65535 for port in ports):
        parser.error("Choose three distinct ports in the range 1..65535.")
    if cfg.BIO_API_BASE.rstrip("/") != f"http://127.0.0.1:{cfg.BIO_API_PORT}":
        parser.error("For the local demo, set BIO_API_BASE=http://127.0.0.1:<BIO_API_PORT>.")
    occupied = [port for port in ports if not port_is_available(port)]
    if occupied:
        print("Ports already in use: " + ", ".join(map(str, occupied)), flush=True)
        return 1
    if args.check_only:
        print("Local demo configuration and ports checked; no model request was made.", flush=True)
        return 0
    commands = [
        ("tools", [sys.executable, "-m", "uvicorn", "ToolsAgent.ToolsFuns.Biology.main:app",
                   "--host", "127.0.0.1", "--port", str(cfg.BIO_API_PORT)],
         f"http://127.0.0.1:{cfg.BIO_API_PORT}/api/runtime-info"),
        ("agent", [sys.executable, str(ROOT / "scripts/start_agent.py"),
                   "--host", "127.0.0.1", "--port", str(cfg.SERVER_PORT)],
         f"http://127.0.0.1:{cfg.SERVER_PORT}/api/runtime-info"),
        ("web", [sys.executable, "-m", "streamlit", "run", str(ROOT / "web_demo.py"),
                 "--server.address", "127.0.0.1", "--server.port", str(args.web_port),
                 "--server.headless", "true", "--browser.gatherUsageStats", "false"],
         f"http://127.0.0.1:{args.web_port}/_stcore/health"),
    ]
    children = []
    handles = []
    try:
        log_dir = Path(cfg.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        for name, command, health in commands:
            log_path = log_dir / f"demo_{name}.log"
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            children.append((process, name, health))
        deadline = time.monotonic() + 120
        while True:
            for process, name, _ in children:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited; inspect logs/demo_{name}.log")
            if all(ready(health) for _, _, health in children):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Startup timed out; inspect logs/demo_*.log")
            time.sleep(0.3)
        print(f"BioAgent ready: http://127.0.0.1:{args.web_port} (Ctrl+C to stop)", flush=True)
        while True:
            for process, name, _ in children:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited; inspect logs/demo_{name}.log")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping BioAgent services...", flush=True)
        return 0
    except (OSError, RuntimeError) as exc:
        print("Demo startup failed: " + str(exc), flush=True)
        return 1
    finally:
        stop_owned(children)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
