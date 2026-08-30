# -*- coding: utf-8 -*-
"""
watchdog.py - OS-level guardian that keeps autofix.py alive forever.

This runs as a separate process. Every 30 seconds it checks if autofix.py
is running. If not, it starts it. This means even if autofix crashes,
the bot comes back within 30 seconds automatically.

Run this INSTEAD of autofix.py:
    py watchdog.py
"""

import subprocess
import sys
import time
import os
from datetime import datetime
from pathlib import Path

AUTOFIX        = Path(__file__).with_name("autofix.py")
LOG_FILE       = Path(__file__).with_name("watchdog_log.txt")
PID_FILE       = Path(__file__).with_name(".watchdog.pid")
CHECK_INTERVAL = 30  # seconds


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [Watchdog] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def acquire_pid_lock():
    """
    Write our PID to .watchdog.pid. If another watchdog is already running,
    exit immediately to prevent dual bot instances.
    Uses subprocess-based check which works on both Windows and Linux.
    """
    if PID_FILE.exists():
        try:
            existing_pid = int(PID_FILE.read_text().strip())
            # Cross-platform alive check via subprocess
            try:
                import psutil
                if psutil.pid_exists(existing_pid):
                    print(f"[Watchdog] Another watchdog is already running (PID {existing_pid}). Exiting.")
                    sys.exit(0)
            except ImportError:
                # psutil not available — use platform-appropriate check
                import platform
                if platform.system() == "Windows":
                    import subprocess as _sp
                    result = _sp.run(
                        ["tasklist", "/FI", f"PID eq {existing_pid}"],
                        capture_output=True, text=True
                    )
                    if str(existing_pid) in result.stdout:
                        print(f"[Watchdog] Another watchdog already running (PID {existing_pid}). Exiting.")
                        sys.exit(0)
                else:
                    # Unix: /proc filesystem check (no signals, no kill)
                    if Path(f"/proc/{existing_pid}").exists():
                        print(f"[Watchdog] Another watchdog already running (PID {existing_pid}). Exiting.")
                        sys.exit(0)
        except (ValueError, Exception):
            pass  # stale or unreadable PID file — overwrite it
    PID_FILE.write_text(str(os.getpid()))


def release_pid_lock():
    PID_FILE.unlink(missing_ok=True)


def is_autofix_running(proc):
    return proc is not None and proc.poll() is None


def start_autofix():
    log("Starting autofix.py...")
    proc = subprocess.Popen(
        [sys.executable, str(AUTOFIX)],
        cwd=str(AUTOFIX.parent),
    )
    log(f"autofix.py started (PID {proc.pid})")
    return proc


def run():
    acquire_pid_lock()
    try:
        log("=" * 50)
        log("Watchdog started — will keep autofix.py alive forever")
        log(f"PID lock acquired: {PID_FILE}")
        log("=" * 50)

        proc = start_autofix()

        while True:
            time.sleep(CHECK_INTERVAL)
            if not is_autofix_running(proc):
                exit_code = proc.returncode
                proc.wait()  # reap zombie handle
                log(f"autofix.py stopped (exit code {exit_code}). Restarting...")
                proc = start_autofix()
    finally:
        release_pid_lock()


if __name__ == "__main__":
    run()
