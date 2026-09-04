# -*- coding: utf-8 -*-
"""
autofix.py - Self-healing AI watchdog for bot.py

Triggers a fix attempt on:
  - Python tracebacks (hard crashes)
  - Known bad behaviour patterns in the output (soft failures)
  - RuntimeErrors, unhandled exceptions logged mid-run
  - Repeated identical errors (loop detection)

For each trigger:
  - Captures the surrounding context (last 60 lines + relevant code)
  - Asks qwen2.5-coder:7b to produce a SEARCH/REPLACE patch
  - Applies the patch and hot-restarts the bot
  - Backs up before patching, rolls back if the fix makes things worse
  - Hot-reloads when bot.py is saved on disk
"""

import collections
import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

BOT_FILE            = Path(__file__).with_name("bot.py")
BACKUP_DIR          = Path(__file__).with_name(".autofix_backups")
LOG_FILE            = Path(__file__).with_name("autofix_log.txt")
RELOAD_FLAG         = Path(__file__).with_name(".reload_now")
ENV_FILE            = Path(__file__).with_name(".env")
OLLAMA_URL          = "http://127.0.0.1:11434/api/chat"
CODER_MODEL         = "qwen2.5-coder:7b"
FALLBACK_MODEL      = "llama3.2:3b"
GROQ_URL            = "https://api.groq.com/openai/v1/chat/completions"
GROQ_CODER_MODEL    = "llama-3.3-70b-versatile"   # best free model for code repair
STABILITY_SECS      = 45
MAX_FIX_ATTEMPTS    = 3
FILE_WATCH_INTERVAL = 2
REPEAT_THRESHOLD    = 3


def _load_groq_key():
    """Read GROQ_API_KEY from environment — works both locally and on hosting servers."""
    import os
    # Always check live env vars first (set by hosting platform)
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    # Fallback: read from .env file for local development
    try:
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GROQ_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Behaviour patterns — each is a (label, regex, severity) tuple.
# severity: "fix" = attempt AI patch, "warn" = log only
# ---------------------------------------------------------------------------
BEHAVIOUR_PATTERNS = [
    # Hard errors
    ("traceback",        r"Traceback \(most recent call last\):",           "fix"),    ("name_error",       r"NameError: name '.*' is not defined",            "fix"),
    ("attribute_error",  r"AttributeError:",                                "fix"),
    ("type_error",       r"TypeError:",                                     "fix"),
    ("key_error",        r"KeyError:",                                      "fix"),
    ("index_error",      r"IndexError:",                                    "fix"),
    ("syntax_error",     r"SyntaxError:",                                   "fix"),
    ("runtime_error",    r"RuntimeError:",                                  "fix"),
    ("value_error",      r"ValueError:",                                    "fix"),
    ("import_error",     r"ImportError:|ModuleNotFoundError:",              "fix"),
    ("os_error",         r"OSError:|FileNotFoundError:|PermissionError:",   "fix"),
    # Soft / behavioural failures
    ("ai_not_ready",     r"local AI is not ready|Check that Ollama",        "fix"),
    ("ai_failed",        r"Local AI request failed|\[AI\] Request failed|\[AI\] make_ai_reply failed", "fix"),
    ("voice_failed",     r"Voice reply generation failed",                  "fix"),
    ("db_locked",        r"database is locked",                             "fix"),
    ("db_error",         r"sqlite3\.",                                      "fix"),
    ("event_loop_err",   r"no running event loop|no current event loop",    "fix"),
    ("discord_error",    r"discord\.errors\.",                              "warn"),
    ("rate_limited",     r"429 Too Many Requests",                          "warn"),
    ("command_notfound", r"CommandNotFound:",                               "warn"),
    ("ollama_down",      r"Connection refused.*11434|urlopen error",        "fix"),
    ("all_ai_down",      r"All AI providers unavailable",                   "warn"),
    ("unhandled_exc",    r"Ignoring exception in",                          "fix"),
    ("memory_fail",      r"Memory summarisation failed",                    "warn"),
    ("vision_fail",      r"Local image analysis failed",                    "warn"),
    ("silent_failure",   r"\[SILENT_FAILURE\]",                             "fix"),
]

# Pre-compile for speed
COMPILED_PATTERNS = [
    (label, re.compile(pattern, re.I), severity)
    for label, pattern, severity in BEHAVIOUR_PATTERNS
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def ollama_alive():
    try:
        with urllib.request.urlopen(
            urllib.request.Request("http://127.0.0.1:11434/api/tags"), timeout=3
        ):
            return True
    except Exception:
        return False


def model_present(name):
    try:
        with urllib.request.urlopen(
            urllib.request.Request("http://127.0.0.1:11434/api/tags"), timeout=5
        ) as r:
            models = [m["name"] for m in json.loads(r.read())["models"]]
        return any(m == name or m.startswith(name.split(":")[0] + ":") for m in models)
    except Exception:
        return False


def best_model():
    """Return the best available coding model — local Ollama first, Groq fallback."""
    if ollama_alive():
        if model_present(CODER_MODEL):
            return ("ollama", CODER_MODEL)
        if model_present(FALLBACK_MODEL):
            log(f"[AutoFix] {CODER_MODEL} not available, using {FALLBACK_MODEL}")
            return ("ollama", FALLBACK_MODEL)
    groq_key = _load_groq_key()
    if groq_key:
        log(f"[AutoFix] Ollama not available, using Groq ({GROQ_CODER_MODEL}) for code repair")
        return ("groq", groq_key)
    return None


def ask_ai(model_info, prompt):
    """Send a prompt to the best available AI and return the response."""
    if model_info is None:
        return ""
    backend, model_or_key = model_info

    if backend == "groq":
        payload = json.dumps({
            "model": GROQ_CODER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.05,
        }).encode()
        req = urllib.request.Request(
            GROQ_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {model_or_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"].strip()

    # Ollama backend
    payload = json.dumps({
        "model": model_or_key,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": 1000, "temperature": 0.05},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def backup_bot():
    BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"bot_{ts}.py"
    shutil.copy2(BOT_FILE, dest)
    for old in sorted(BACKUP_DIR.glob("bot_*.py"))[:-15]:
        old.unlink(missing_ok=True)
    return dest


def restore_backup(path):
    shutil.copy2(path, BOT_FILE)
    log(f"[AutoFix] Rolled back to {path.name}")


def apply_patch(patch_text):
    """Apply <<<SEARCH>>>...<<<REPLACE>>>...<<<END>>> blocks."""
    blocks = re.findall(
        r"<<<SEARCH>>>\n(.*?)\n<<<REPLACE>>>\n(.*?)\n<<<END>>>",
        patch_text, re.S
    )
    if not blocks:
        log("[AutoFix] No valid patch blocks in AI response.")
        return False

    source = BOT_FILE.read_text(encoding="utf-8")
    applied = 0
    for search, replace in blocks:
        s, r = search.strip(), replace.strip()
        if s in source:
            source = source.replace(s, r, 1)
            applied += 1
        else:
            log(f"[AutoFix] Block not found (already fixed?): {s[:80]!r}")

    if applied:
        BOT_FILE.write_text(source, encoding="utf-8")
        log(f"[AutoFix] {applied}/{len(blocks)} patch block(s) applied.")
    return applied > 0


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

def get_code_context(line_num, context=14):
    try:
        lines = BOT_FILE.read_text(encoding="utf-8").splitlines()
        start = max(0, line_num - context - 1)
        end   = min(len(lines), line_num + context)
        return "\n".join(
            f"{'>>>' if i + 1 == line_num else '   '} {i+1:4d}: {lines[i]}"
            for i in range(start, end)
        )
    except Exception:
        return ""


def build_fix_prompt(trigger_label, trigger_lines, recent_output):
    """
    Build a prompt for the coder AI.
    trigger_lines: the log lines that matched the problem pattern
    recent_output: last ~60 lines of bot output for context
    """
    # Extract bot.py line references from the output
    refs = re.findall(r'File "([^"]*bot\.py[^"]*)", line (\d+)', recent_output)
    code_sections = []
    for _, ln in refs:
        ctx = get_code_context(int(ln))
        if ctx:
            code_sections.append(f"Code near line {ln}:\n{ctx}")

    # Also check if trigger lines themselves reference a line number
    for tl in trigger_lines:
        m = re.search(r"line (\d+)", tl)
        if m:
            ctx = get_code_context(int(m.group(1)))
            if ctx and ctx not in "\n".join(code_sections):
                code_sections.append(f"Code near line {m.group(1)}:\n{ctx}")

    code_ctx = "\n\n".join(code_sections) or "(no specific line referenced)"

    problem_block = "\n".join(trigger_lines)
    context_block = recent_output[-3000:]  # last 3000 chars of output

    # Try to read brain's known_issues for extra context
    brain_context = ""
    try:
        brain_file = BOT_FILE.parent / "brain_state.txt"
        if brain_file.exists():
            brain_context = f"\nKNOWN ISSUES FROM BOT BRAIN:\n{brain_file.read_text()}\n"
    except Exception:
        pass

    return f"""You are an expert Python bot debugger. The Discord bot produced this problem:

=== PROBLEM TYPE: {trigger_label.upper()} ===
{problem_block}

=== RECENT BOT OUTPUT (last 60 lines) ===
{context_block}

=== RELEVANT CODE IN bot.py ===
{code_ctx}

Your job: produce a minimal, safe fix for this specific problem.

Rules:
- Only fix the exact issue shown. Do not refactor anything else.
- Preserve all existing logic — only change what is broken.
- Copy the SEARCH block character-for-character from the code above, including all spaces.
- Output ONLY patch blocks in this format, nothing else:

<<<SEARCH>>>
exact original lines from bot.py (copy exactly, preserving indentation)
<<<REPLACE>>>
fixed replacement lines
<<<END>>>

Use multiple blocks if needed.
If you cannot safely determine a fix, output exactly: CANNOT_FIX
"""


# ---------------------------------------------------------------------------
# Behaviour monitor — runs in its own thread, watches output line-by-line
# ---------------------------------------------------------------------------

class BehaviourMonitor:
    def __init__(self):
        self.recent_lines    = collections.deque(maxlen=120)
        self.error_counts    = collections.defaultdict(int)
        self.lock            = threading.Lock()
        self._last_fix_time  = 0
        self._pending_reply  = None   # timestamp when [Reply] was seen
        self._silent_fail_injected = False

    def feed(self, line):
        """Feed a new output line. Returns (label, lines) if a fix should fire."""
        self.recent_lines.append(line.rstrip())

        # Never trigger fixes for known non-code errors
        ignorable = [
            "All AI providers unavailable",
            "RuntimeError: All AI",
            "HTTP Error 403",
            "Connection refused",
            "429 Too Many Requests",
            "[AI][sync]",          # background fact extraction / summarisation — not a reply failure
            "quota hit",           # expected when OpenRouter free tier runs out
            "OpenRouter 429",      # same
            "OpenRouter failed",   # transient network error, not a code bug
            "HTTPError: HTTP Error 429",
            "Clean models quota-exhausted",
        ]
        if any(phrase in line for phrase in ignorable):
            return None

        # Track incoming vs sent replies to detect silent failures
        if "[Reply]" in line and "Incoming" in line:
            self._pending_reply = time.time()
            self._silent_fail_injected = False
        if "[Reply sent]" in line:
            self._pending_reply = None
            self._silent_fail_injected = False

        # If a reply was received but not sent within 90 seconds, inject a
        # synthetic silent failure marker that the pattern detector will catch
        if (self._pending_reply and not self._silent_fail_injected
                and time.time() - self._pending_reply > 90):
            self._silent_fail_injected = True
            synthetic = "[SILENT_FAILURE] Bot received message but sent no reply after 90s"
            print(f"[Monitor] {synthetic}", flush=True)
            self.recent_lines.append(synthetic)
            # Feed the synthetic line back through the detector
            return self._check_patterns(synthetic)

        return self._check_patterns(line)

    def _check_patterns(self, line):
        for label, pattern, severity in COMPILED_PATTERNS:
            if pattern.search(line):
                with self.lock:
                    self.error_counts[label] += 1
                    count = self.error_counts[label]

                if severity == "warn":
                    if count == 1:
                        log(f"[Monitor] Warning '{label}': {line.rstrip()}")
                    return None

                if severity == "fix":
                    trigger_lines = list(self.recent_lines)[-10:]
                    now = time.time()
                    if now - self._last_fix_time < 30:
                        return None
                    hard_errors = {"traceback", "name_error", "syntax_error",
                                   "import_error", "attribute_error", "silent_failure",
                                   "type_error", "value_error",
                                   "key_error", "index_error", "os_error"}
                    if label not in hard_errors and count < REPEAT_THRESHOLD:
                        return None
                    self._last_fix_time = now
                    self.error_counts[label] = 0
                    return (label, trigger_lines)
        return None

    def reset(self):
        with self.lock:
            self.error_counts.clear()
        self._last_fix_time  = 0
        self._pending_reply  = None
        self._silent_fail_injected = False
        self.recent_lines.clear()

    def recent_output(self):
        return "\n".join(self.recent_lines)


# ---------------------------------------------------------------------------
# File watcher
# ---------------------------------------------------------------------------

def file_hash(path):
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def watch_files(proc_ref, stop_event):
    """Kill the bot process when bot.py changes or .reload_now appears."""
    last_hash = file_hash(BOT_FILE)
    while not stop_event.is_set():
        time.sleep(FILE_WATCH_INTERVAL)
        cur_hash = file_hash(BOT_FILE)
        reload   = RELOAD_FLAG.exists()
        if cur_hash != last_hash or reload:
            if reload:
                RELOAD_FLAG.unlink(missing_ok=True)
                log("[AutoFix] Reload flag detected — restarting bot...")
            else:
                log("[AutoFix] bot.py changed on disk — hot-reloading...")
            last_hash = cur_hash
            p = proc_ref[0]
            if p and p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Fix runner
# ---------------------------------------------------------------------------

def attempt_fix(label, trigger_lines, recent_output, fix_attempt):
    log(f"[AutoFix] Problem detected: '{label}' (attempt {fix_attempt}/{MAX_FIX_ATTEMPTS})")

    model_info = best_model()
    if not model_info:
        log("[AutoFix] No AI model available (no Ollama + no Groq key) — skipping fix.")
        return False

    backend, _ = model_info
    log(f"[AutoFix] Using {backend.upper()} for code repair")

    backup = backup_bot()
    log(f"[AutoFix] Backed up to {backup.name}")

    prompt = build_fix_prompt(label, trigger_lines, recent_output)
    try:
        log(f"[AutoFix] Sending problem to AI...")
        response = ask_ai(model_info, prompt)
    except Exception as e:
        log(f"[AutoFix] AI request failed: {e}")
        return False

    if "CANNOT_FIX" in response:
        log("[AutoFix] AI cannot safely fix this.")
        return False

    patched = apply_patch(response)
    if not patched:
        log("[AutoFix] Patch produced no changes.")
    return patched


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():
    fix_attempts = 0
    last_backup  = None
    monitor      = BehaviourMonitor()

    log("=" * 60)
    log("AutoFix watchdog started")
    log(f"Bot     : {BOT_FILE}")
    groq_key = _load_groq_key()
    if ollama_alive():
        log(f"AI model: {CODER_MODEL} (Ollama local)")
    elif groq_key:
        log(f"AI model: {GROQ_CODER_MODEL} (Groq cloud fallback)")
    else:
        log("AI model: NONE — no Ollama and no GROQ_API_KEY. Auto-fix disabled.")
    log(f"Watching: crashes + {len(BEHAVIOUR_PATTERNS)} behaviour patterns")
    log("=" * 60)

    while True:
        log(f"[AutoFix] Launching bot.py...")
        start_time = time.time()
        monitor.reset()

        proc = subprocess.Popen(
            [sys.executable, str(BOT_FILE)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(BOT_FILE.parent),
        )
        proc_ref   = [proc]
        stop_watch = threading.Event()
        watcher    = threading.Thread(
            target=watch_files, args=(proc_ref, stop_watch), daemon=True
        )
        watcher.start()

        needs_fix   = None   # (label, trigger_lines) if a fix should fire mid-run
        kill_reason = None   # why we killed the process (if we did)

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

            result = monitor.feed(line)
            if result and needs_fix is None:
                needs_fix   = result
                kill_reason = result[0]
                log(f"[AutoFix] Behaviour trigger '{kill_reason}' — will fix and restart.")
                # Kill after 3 seconds so full traceback prints first
                import threading as _threading
                def _delayed_kill():
                    time.sleep(3)
                    if proc.poll() is None:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                _threading.Thread(target=_delayed_kill, daemon=True).start()

        proc.wait()
        stop_watch.set()

        uptime    = time.time() - start_time
        exit_code = proc.returncode
        log(f"[AutoFix] Bot exited after {uptime:.1f}s (code {exit_code})")

        if uptime >= STABILITY_SECS:
            fix_attempts = 0
            last_backup  = None

        # --- Decide whether to attempt a fix ---
        if needs_fix and fix_attempts < MAX_FIX_ATTEMPTS:
            label, trigger_lines = needs_fix
            patched = attempt_fix(
                label, trigger_lines,
                monitor.recent_output(),
                fix_attempts + 1
            )
            if patched:
                fix_attempts += 1
                last_backup = list(sorted(BACKUP_DIR.glob("bot_*.py")))[-1] if BACKUP_DIR.exists() else None
                log("[AutoFix] Patch applied — restarting bot with fix...")
                # Write a record into brain_state.txt so the bot can read it on next start
                try:
                    brain_file = BOT_FILE.parent / "brain_state.txt"
                    existing = brain_file.read_text(encoding="utf-8") if brain_file.exists() else ""
                    ts = datetime.now().strftime("%H:%M UTC")
                    brain_file.write_text(
                        existing.rstrip() + f"\nAutofix applied: [{ts}] {label} → PATCHED\n",
                        encoding="utf-8"
                    )
                except Exception:
                    pass
            else:
                fix_attempts += 1
                log(f"[AutoFix] Patch failed — will retry or roll back next cycle (attempt {fix_attempts}/{MAX_FIX_ATTEMPTS}).")
            time.sleep(2)
            continue

        elif needs_fix and fix_attempts >= MAX_FIX_ATTEMPTS:
            log(f"[AutoFix] Max fix attempts ({MAX_FIX_ATTEMPTS}) reached.")
            if last_backup and last_backup.exists():
                log("[AutoFix] Rolling back to last known-good backup...")
                restore_backup(last_backup)
            else:
                log("[AutoFix] No backup available — resetting attempt counter and retrying.")
            # Always reset so the bot can recover instead of staying stuck forever.
            fix_attempts = 0
            last_backup  = None
            time.sleep(3)

        log("[AutoFix] Restarting in 5 seconds...")
        time.sleep(5)


if __name__ == "__main__":
    run()
