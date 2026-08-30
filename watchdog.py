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
    Write our PID to .watchdog.pid. If the file already exists and the
    recorded PID is still alive, exit immediately — another watchdog is
    already running. Prevents two watchdog instances from both launching
    autofix.py (which would double-start bot.py and cause dual-token Discord connections).
    """
    if PID_FILE.exists():
        try:
            existing_pid = int(PID_FILE.read_text().strip())
            # Check if that process is still alive
            import psutil
            if psutil.pid_exists(existing_pid):
                print(f"[Watchdog] Another watchdog is already running (PID {existing_pid}). Exiting.")
                sys.exit(0)
        except (ValueError, ImportError, Exception):
            # psutil not available — fall back to os.kill check
            try:
                existing_pid = int(PID_FILE.read_text().strip())
                os.kill(existing_pid, 0)  # signal 0 = check existence
                print(f"[Watchdog] Another watchdog is already running (PID {existing_pid}). Exiting.")
                sys.exit(0)
            except (ValueError, OSError):
                pass  # PID file stale — safe to overwrite
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
