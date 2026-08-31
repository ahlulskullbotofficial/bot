# -*- coding: utf-8 -*-
import asyncio
import base64
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import tempfile
import traceback
import urllib.error
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

# Force UTF-8 output so non-ASCII usernames/messages never crash the bot
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load secrets from .env so they are never hardcoded in source
from dotenv import load_dotenv
load_dotenv(Path(__file__).with_name(".env"))

import discord
from discord.ext import commands, tasks
from flag_identify import identify_flag
from islamic_quiz import IslamicQuiz, LEVELS
from PIL import Image, UnidentifiedImageError

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
bot_loop = None
console_started = False
slash_commands_synced = False
autoreact_file = Path(__file__).with_name("autoreact_users.json")
memory_file = Path(__file__).with_name("bot_memory.sqlite3")
quiz_file = Path(__file__).with_name("quiz_questions.json")
BOT_REACTION = "\U0001f480"
OLLAMA_MODEL = "llama3.2:3b"
# Groq — loaded from .env. When set, all AI text replies use Groq instead of
# Ollama so the bot can run 24/7 on a server without a local GPU.
# You can add GROQ_API_KEY_2, GROQ_API_KEY_3 etc. for automatic rotation.
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY_2", "")
GROQ_API_KEY_3 = os.environ.get("GROQ_API_KEY_3", "")
GROQ_MODEL    = "llama-3.3-70b-versatile"
GROQ_URL      = "https://api.groq.com/openai/v1/chat/completions"
# OpenRouter fallback — free alternative if Groq is unavailable
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL   = "google/gemma-4-31b-it:free"
# Prefer Qwen2.5-VL; fall back to moondream only if nothing else is installed.
PREFERRED_VISION_MODELS = (
    "qwen2.5vl:3b",
    "qwen2.5vl:7b",
    "qwen2.5vl",
    "llava:7b",
    "llava",
    "moondream",
)
OLLAMA_VISION_MODEL = PREFERRED_VISION_MODELS[0]
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
WHISPER_MODEL_SIZE = "base"
TTS_VOICE = "en-GB-SoniaNeural"
# ElevenLabs — loaded from .env (ELEVENLABS_API_KEY). Leave blank to use edge-tts.
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
# Voice ID for "Aria" — highly expressive, natural-sounding, great for emotions.
ELEVENLABS_VOICE_ID = "9BWtsMINqrJLrRacOk9x"
ELEVENLABS_MODEL = "eleven_multilingual_v2"

# ElevenLabs monthly quota reset tracking (month number when quota was exhausted)
_elevenlabs_quota_reset_month = None

# ============================================================================
# BOT BRAIN — unified cognitive state shared across all subsystems.
# Every subsystem reads from and writes to brain, so they all stay in sync.
# Think of it as the bot's nervous system: one source of truth.
# ============================================================================
class BotBrain:
    def __init__(self):
        # --- Service health ---
        self.ollama_alive      = True   # set by health check task
        self.groq_alive        = True   # set by _call_ai on error
        self.elevenlabs_alive  = True   # set by TTS on quota error

        # --- Active conversation state per user ---
        # {user_id: {"topic": str, "mood": str, "last_seen": datetime, "pending_voice": bool}}
        self.user_state: dict = {}

        # --- System event log (rolling, last 200 entries) ---
        # Each entry: {"ts": iso, "system": str, "event": str, "user_id": int|None}
        self.event_log: list = []

        # --- Self-improvement log: tracks what fixes were attempted ---
        self.fix_history: list = []

        # --- Pending issues that need resolution ---
        self.known_issues: list = []

        # --- Quiz awareness ---
        self.active_quiz_channels: set = set()

        # --- Vision model cache ---
        self.vision_model: str = ""  # set once Ollama is available

        # --- Groq key rotation ---
        self.groq_keys: list = [k for k in [
            os.environ.get("GROQ_API_KEY", ""),
            os.environ.get("GROQ_API_KEY_2", ""),
            os.environ.get("GROQ_API_KEY_3", ""),
        ] if k]
        self.groq_key_index: int = 0
        self.groq_dead_keys: set = set()
        self.groq_key_dead: bool = False  # True when current key is invalid

        # --- Alert tracking (don't spam the same alert) ---
        self.alerted: set = set()   # set of alert IDs already sent

    def log_event(self, system: str, event: str, user_id=None):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "system": system,
            "event": event,
            "user_id": user_id,
        }
        self.event_log.append(entry)
        if len(self.event_log) > 200:
            self.event_log.pop(0)

    def get_user_state(self, user_id: int) -> dict:
        if user_id not in self.user_state:
            self.user_state[user_id] = {
                "topic": "",
                "mood": "neutral",
                "last_seen": datetime.now(timezone.utc),
                "pending_voice": False,
                "message_count": 0,
            }
        return self.user_state[user_id]

    def update_user_state(self, user_id: int, **kwargs):
        state = self.get_user_state(user_id)
        state.update(kwargs)
        state["last_seen"] = datetime.now(timezone.utc)
        state["message_count"] = state.get("message_count", 0) + 1

    def track_groq_request(self):
        pass  # tracking handled by groq_key_index rotation

    def groq_quota_warning(self) -> bool:
        return False  # quota warnings not applicable with key rotation

    def system_status(self) -> str:
        """Human-readable status string injected into AI context."""
        parts = []
        if not self.ollama_alive:
            parts.append("Ollama offline (using Groq for all AI)")
        if not self.groq_alive:
            parts.append("Groq unavailable (using local Ollama)")
        if not self.elevenlabs_alive:
            parts.append("ElevenLabs quota exhausted (using edge-tts)")
        if self.groq_quota_warning():
            parts.append(f"Groq near daily limit ({self.groq_requests_today} requests today)")
        return "; ".join(parts) if parts else "All systems nominal"

    def add_known_issue(self, issue: str):
        if issue not in self.known_issues:
            self.known_issues.append(issue)
            if len(self.known_issues) > 20:
                self.known_issues.pop(0)

    def resolve_issue(self, issue_fragment: str):
        self.known_issues = [i for i in self.known_issues if issue_fragment not in i]

    def active_groq_key(self) -> str:
        """Return the currently active Groq key, skipping dead ones."""
        if not self.groq_keys:
            return GROQ_API_KEY  # fallback to global
        live = [k for k in self.groq_keys if k not in self.groq_dead_keys]
        if not live:
            return ""  # all keys dead
        # Use round-robin index mod live list
        self.groq_key_index = self.groq_key_index % len(live)
        return live[self.groq_key_index]

    def mark_groq_key_dead(self, key: str):
        """Mark a key as invalid (401/403) and rotate to the next one."""
        self.groq_dead_keys.add(key)
        live = [k for k in self.groq_keys if k not in self.groq_dead_keys]
        if live:
            self.groq_key_index = 0
            print(f"[Brain] Groq key rotated — {len(live)} key(s) remaining")
            self.groq_alive = True
            self.resolve_issue("Groq")
        else:
            print("[Brain] ALL Groq keys are dead — no AI available")
            self.groq_alive = False
            self.add_known_issue("All Groq keys expired — update GROQ_API_KEY in env vars")

    def should_alert(self, alert_id: str) -> bool:
        """Return True if this alert hasn't been sent yet (prevents spam)."""
        if alert_id in self.alerted:
            return False
        self.alerted.add(alert_id)
        return True


# Global brain instance — imported and used by all subsystems
brain = BotBrain()

# Per-user AI request rate limiting: max 1 concurrent AI reply per user.
# Uses asyncio.Lock so it never blocks the event loop thread.
_user_ai_locks: dict = {}
_user_ai_locks_lock = threading.Lock()

# Thread lock for WhisperModel initialisation (prevents race on first voice message).
_transcriber_lock = threading.Lock()
MAX_VISION_BYTES = 8 * 1024 * 1024
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AhlulSkullBot/1.1; +https://discord.com)",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

AI_INSTRUCTIONS = """
You are a welcoming assistant for a Muslim Discord server. Speak in a playful
British roadman style: concise, confident, witty, and a little cheeky. Your
swagger is clearly playful, never cruel. Respect Islamic manners and every
member of the community. Do not insult, harass, use slurs, shame people, make
sectarian claims, or present yourself as a scholar. For religious questions,
answer carefully and say that a qualified local imam or scholar is best for
personal rulings. Use phrases such as insha Allah or alhamdulillah only when
they fit naturally. Never claim to be Muslim or speak on behalf of Islam.
Default to one short sentence (around 160 characters or less). Only give more
detail when the user explicitly asks for it or it is necessary for accuracy.
If the user message includes Visual analysis, that is the only source of truth
for what is in the picture or GIF. Repeat the actual objects, text, and scene
from that analysis. Never invent a different country, flag, animal, person, or
object. A possible flag layout is a hint, not a fact, unless the analysis also
says the image is a flag. If the analysis is missing, failed, or unsure, say
you cannot tell - do not guess.
"""

voice_transcriber = None


def initialise_memory():
    with _db() as database:
        # WAL is already set by _db(), but set it explicitly here too for clarity
        database.executescript("""
            CREATE TABLE IF NOT EXISTS preferences (
                user_id INTEGER PRIMARY KEY,
                ai_enabled INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS member_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL DEFAULT 0,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                importance INTEGER NOT NULL DEFAULT 1,
                times_referenced INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, guild_id, memory_key)
            );
            CREATE INDEX IF NOT EXISTS idx_member_memories_lookup
                ON member_memories(user_id, guild_id, importance DESC, updated_at DESC);
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                summary TEXT NOT NULL,
                turn_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, channel_id)
            );
            CREATE TABLE IF NOT EXISTS scheduled_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                requester_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                run_at TEXT NOT NULL,
                delivered INTEGER NOT NULL DEFAULT 0
            );
        """)
        # Migrate existing databases that don't have the times_referenced column yet
        try:
            database.execute("ALTER TABLE member_memories ADD COLUMN times_referenced INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass  # Column already exists


def _db(readonly=False):
    """Open a SQLite connection with WAL mode and a generous timeout.
    WAL mode allows concurrent reads from multiple threads without locking.
    """
    conn = sqlite3.connect(memory_file, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def load_autoreact_users():
    try:
        saved_data = json.loads(autoreact_file.read_text(encoding="utf-8"))
        return {int(user_id) for user_id in saved_data.get("user_ids", [])}
    except (FileNotFoundError, ValueError, TypeError):
        return set()


def save_autoreact_users():
    autoreact_file.write_text(
        json.dumps({"user_ids": sorted(autoreact_users)}, indent=2),
        encoding="utf-8",
    )


def ai_enabled_for(user_id):
    with _db() as database:
        row = database.execute(
            "SELECT ai_enabled FROM preferences WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row is None or bool(row[0])


def set_ai_enabled(user_id, enabled):
    with _db() as database:
        database.execute(
            "INSERT INTO preferences(user_id, ai_enabled) VALUES(?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET ai_enabled = excluded.ai_enabled",
            (user_id, int(enabled)),
        )


def _score_message_importance(content):
    """
    Quickly score a message worth before storing it (0-3).
    0 = trivial noise, skip entirely.
    1 = low value, store but eligible for early pruning.
    2 = normal conversational turn, keep.
    3 = contains personal info, question, or meaningful content, always keep.
    This runs purely on heuristics so it is instant - no AI call needed.
    """
    text = content.strip()
    if not text:
        return 0
    # Very short filler messages \U00002014 not worth storing
    noise_patterns = (
        r"^(lol|lmao|haha|hahaha|ok|okay|k|yep|yup|nope|nah|sure|cool|nice|wow|omg|"
        r"bruh|bro|fr|ngl|ight|aight|yeah|yh|ye|no|yes|thanks|thx|ty|yw|np|"
        "\U0001F602|\U0001F480|\U0001F525|\U0001F440|\U0001F62D|"
        "\u2764\uFE0F|\U0001F44D|\U0001F44E|\u2705|\U0001F64F|\U0001F923|"
        "\U0001F601|\U0001F602|\U0001F603|\U0001F604|\U0001F605|\U0001F606){1,5}"
        r"[!?.]*$"
    )
    if re.match(noise_patterns, text, re.I):
        return 0
    if len(text) < 6:
        return 0
    # High value signals
    high_value = (
        r"\b(remember|my name|i am|i'm|i live|i work|i like|i love|i hate|i speak|"
        r"i want|i need|my age|years old|i study|i'm from|born in|my religion|"
        r"my job|my hobby|my goal|remind me|don't forget|important|please note)\b"
    )
    if re.search(high_value, text, re.I):
        return 3
    # Questions and longer messages are worth keeping
    if "?" in text or len(text) > 60:
        return 2
    # Short statements \U00002014 keep but low priority
    return 1


def remember(user_id, channel_id, role, content):
    # Score assistant replies as always worth keeping (they're the bot's output).
    # For user messages, score and skip trivial noise entirely.
    if role == "user":
        score = _score_message_importance(content)
        if score == 0:
            return  # Not worth storing at all
    with _db() as database:
        database.execute(
            "INSERT INTO conversations(user_id, channel_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, channel_id, role, content[:4000], datetime.now(timezone.utc).isoformat()),
        )
        # Keep 80 turns in the DB. When we exceed 60, the summariser will
        # compress the oldest 40 turns into a rolling summary and delete them,
        # so the live window never balloons unboundedly.
        database.execute(
            "DELETE FROM conversations WHERE user_id = ? AND channel_id = ? AND id NOT IN "
            "(SELECT id FROM conversations WHERE user_id = ? AND channel_id = ? ORDER BY id DESC LIMIT 80)",
            (user_id, channel_id, user_id, channel_id),
        )


def _get_conversation_count(user_id, channel_id):
    with _db() as database:
        row = database.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_id = ? AND channel_id = ?",
            (user_id, channel_id),
        ).fetchone()
    return row[0] if row else 0


def _load_oldest_turns(user_id, channel_id, limit=40):
    """Return the oldest `limit` turns as a list of (id, role, content) tuples."""
    with _db() as database:
        return database.execute(
            "SELECT id, role, content FROM conversations "
            "WHERE user_id = ? AND channel_id = ? ORDER BY id ASC LIMIT ?",
            (user_id, channel_id, limit),
        ).fetchall()


def _save_summary(user_id, channel_id, new_summary, turn_count):
    now = datetime.now(timezone.utc).isoformat()
    with _db() as database:
        database.execute(
            "INSERT INTO conversation_summaries(user_id, channel_id, summary, turn_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, channel_id) DO UPDATE SET "
            "summary = excluded.summary, turn_count = turn_count + excluded.turn_count, updated_at = excluded.updated_at",
            (user_id, channel_id, new_summary[:2000], turn_count, now),
        )


def _delete_turns_by_ids(ids):
    if not ids:
        return
    with _db() as database:
        database.execute(
            f"DELETE FROM conversations WHERE id IN ({','.join('?' * len(ids))})", ids
        )


def summarize_old_turns(user_id, channel_id):
    """
    When the conversation grows past 60 turns, compress the oldest 40 into a
    rolling plain-English summary using Ollama, then delete those raw turns.
    This means context from weeks ago is never lost - it is just compressed.
    """
    if _get_conversation_count(user_id, channel_id) < 60:
        return
    old_turns = _load_oldest_turns(user_id, channel_id, limit=40)
    if not old_turns:
        return
    # Load any existing summary to merge with
    with _db() as database:
        existing = database.execute(
            "SELECT summary FROM conversation_summaries WHERE user_id = ? AND channel_id = ?",
            (user_id, channel_id),
        ).fetchone()
    existing_summary = existing[0] if existing else ""
    # Format the turns into a readable transcript for the model
    transcript = "\n".join(
        f"{role.upper()}: {content[:300]}" for _, role, content in old_turns
    )
    prompt = (
        "You are a memory assistant for a Discord bot. Compress this chat transcript into a "
        "concise summary that preserves every important fact, correction, opinion, and personal "
        "detail the USER shared. Focus on what the USER said, not the bot.\n"
        "Rules:\n"
        "- If the user corrected a previous statement, keep only the corrected version.\n"
        "- Preserve specific numbers, dates, names, and preferences exactly.\n"
        "- Write in third person: 'The user said...', 'They mentioned...'\n"
        "- Never invent information.\n\n"
    )
    if existing_summary:
        prompt += f"EXISTING SUMMARY (merge new info into this):\n{existing_summary}\n\n"
    prompt += f"NEW TRANSCRIPT TO SUMMARISE:\n{transcript}\n\nWrite an updated summary (max 300 words):"
    try:
        messages = [{"role": "user", "content": prompt}]
        new_summary = _call_ai(messages, max_tokens=400, temperature=0.1)
        _save_summary(user_id, channel_id, new_summary, len(old_turns))
        _delete_turns_by_ids([row[0] for row in old_turns])
    except Exception as error:
        print(f"Memory summarisation failed (will retry next turn): {error}")


def get_conversation_summary(user_id, channel_id):
    with _db() as database:
        row = database.execute(
            "SELECT summary FROM conversation_summaries WHERE user_id = ? AND channel_id = ?",
            (user_id, channel_id),
        ).fetchone()
    return row[0] if row else None


def recent_memory(user_id, channel_id):
    """Return up to 20 recent turns PLUS any compressed summary as a system-style prefix."""
    with _db() as database:
        rows = database.execute(
            "SELECT role, content FROM conversations WHERE user_id = ? AND channel_id = ? "
            "ORDER BY id DESC LIMIT 20", (user_id, channel_id)
        ).fetchall()
    turns = [{"role": role, "content": content} for role, content in reversed(rows)]
    summary = get_conversation_summary(user_id, channel_id)
    if summary:
        # Inject the summary as a system message at the very start of history
        # so the model always has the full picture of the conversation.
        turns = [{"role": "system", "content": f"[Earlier conversation summary]: {summary}"}] + turns
    return turns


def _extract_facts_with_ai(content):
    """
    Use Ollama to intelligently extract personal facts from a message.
    Returns a list of (key, value, importance) tuples, or falls back to
    regex if the AI call fails.
    """
    text = re.sub(r"\s+", " ", content).strip()
    if not text or len(text) < 8:
        return []
    blocked = ("password", "passcode", "api key", "token", "credit card", "bank account", "private key")
    if any(term in text.lower() for term in blocked):
        return []

    prompt = (
        "Extract personal facts about the user from this Discord message. "
        "Return ONLY a JSON array of objects with keys: \"key\", \"value\", \"importance\" (1-5).\n"
        "Importance guide: 5=name/explicit memory request, 4=religion/identity, 3=location/language/age, "
        "2=hobby/preference/opinion, 1=casual mention.\n"
        "Valid keys: name, age, location, language, religion, occupation, hobby, preference, "
        "opinion, relationship, goal, explicit_memory, mood, topic_interest.\n"
        "Rules: only extract clear facts, no guessing, return [] if nothing useful, "
        "keep values under 120 characters, never extract passwords or secrets.\n\n"
        f"Message: {text[:400]}\n\nJSON array only, no explanation:"
    )
    try:
        messages = [{"role": "user", "content": prompt}]
        raw = _call_ai(messages, max_tokens=200, temperature=0.0)
        # Extract the JSON array from the response robustly
        match = re.search(r"\[.*\]", raw, re.S)
        if not match:
            return []
        facts_raw = json.loads(match.group())
        facts = []
        valid_keys = {
            "name", "age", "location", "language", "religion", "occupation",
            "hobby", "preference", "opinion", "relationship", "goal",
            "explicit_memory", "mood", "topic_interest",
        }
        for item in facts_raw:
            key = str(item.get("key", "")).strip().lower()
            value = str(item.get("value", "")).strip()
            importance = int(item.get("importance", 2))
            if key in valid_keys and len(value) >= 2:
                facts.append((key, value[:180], max(1, min(5, importance))))
        return facts
    except Exception:
        # Fallback to regex if the AI call times out or returns garbage
        return _extract_facts_regex(text)


def _extract_facts_regex(text):
    """Fallback regex-based fact extraction."""
    facts = []
    patterns = [
        (r"\b(?:my name is|call me)\s+([\w .'-]{2,40})", "name", 5),
        (r"\b(?:i live in|i am from|i'm from)\s+([\w .,'-]{2,60})", "location", 3),
        (r"\b(?:i am|i'm)\s+(\d{1,3})\s*(?:years old|yo)\b", "age", 3),
        (r"\b(?:i prefer|i like|i love|i enjoy|i dislike|i hate)\s+(.{2,100})", "preference", 2),
        (r"\b(?:i want|i need|i am trying to|i'm trying to|i plan to)\s+(.{2,120})", "goal", 2),
        (r"\bremember(?: that)?\s+(.{2,160})", "explicit_memory", 5),
        (r"\b(?:i work as|i am a|i'm a)\s+([\w ]{2,60})", "occupation", 3),
        (r"\b(?:i speak|i am fluent in)\s+([\w ]{2,40})", "language", 3),
    ]
    for pattern, key, importance in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = match.group(1).strip(" .!?")
            if len(value) >= 2:
                facts.append((key, value[:180], importance))
    return facts


def remember_member_facts(user_id, guild_id, content):
    now = datetime.now(timezone.utc).isoformat()
    facts = _extract_facts_with_ai(content)
    if not facts:
        return
    with _db() as database:
        for key, value, importance in facts:
            # Check if a conflicting value already exists for this key
            existing = database.execute(
                "SELECT memory_value, importance FROM member_memories "
                "WHERE user_id = ? AND guild_id = ? AND memory_key = ?",
                (user_id, guild_id, key)
            ).fetchone()
            if existing:
                old_value, old_importance = existing
                # If the new value contradicts the old one and has equal or higher importance,
                # update it. Log the change so the brain is aware.
                if old_value != value and importance >= old_importance:
                    brain.log_event("memory", f"contradiction_resolved:{key}", user_id=user_id)
            database.execute(
                "INSERT INTO member_memories(user_id, guild_id, memory_key, memory_value, importance, times_referenced, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?, ?) "
                "ON CONFLICT(user_id, guild_id, memory_key) DO UPDATE SET "
                "memory_value = CASE WHEN excluded.importance >= importance THEN excluded.memory_value ELSE memory_value END, "
                "importance = MAX(importance, excluded.importance), "
                "updated_at = excluded.updated_at",
                (user_id, guild_id, key, value, importance, now, now),
            )
        # Auto-prune: delete importance-1 facts older than 7 days that were never referenced.
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        database.execute(
            "DELETE FROM member_memories WHERE user_id = ? AND guild_id = ? "
            "AND importance = 1 AND times_referenced = 0 AND updated_at < ?",
            (user_id, guild_id, cutoff),
        )
        # Hard cap: keep the 40 best facts per user per guild
        database.execute(
            "DELETE FROM member_memories WHERE user_id = ? AND guild_id = ? AND id NOT IN "
            "(SELECT id FROM member_memories WHERE user_id = ? AND guild_id = ? "
            "ORDER BY (importance * 2 + times_referenced) DESC, updated_at DESC LIMIT 40)",
            (user_id, guild_id, user_id, guild_id),
        )


def member_memory_context(user_id, guild_id, display_name):
    now = datetime.now(timezone.utc).isoformat()
    with _db() as database:
        rows = database.execute(
            "SELECT id, memory_key, memory_value FROM member_memories "
            "WHERE user_id = ? AND guild_id = ? "
            "ORDER BY (importance * 2 + times_referenced) DESC, updated_at DESC LIMIT 20",
            (user_id, guild_id),
        ).fetchall()
        if rows:
            # Bump the reference counter on every fact we're about to serve to the AI.
            # Facts that get referenced frequently are preserved longer.
            ids = [row[0] for row in rows]
            database.execute(
                f"UPDATE member_memories SET times_referenced = times_referenced + 1 "
                f"WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
    identity = (
        f"You are talking to {display_name} (Discord user ID {user_id}). "
        f"This person is DIFFERENT from all other members - their facts below apply ONLY to them."
    )
    # Inject user brain state for richer context
    user_state = brain.get_user_state(user_id)
    msg_count = user_state.get("message_count", 0)
    if msg_count > 0:
        identity += f" They have sent {msg_count} messages to the bot."
    if not rows:
        return identity + " No personal facts saved yet for this member."
    # Group facts by key, deduplicate values
    grouped = {}
    for _, key, value in rows:
        grouped.setdefault(key, []).append(value)
    fact_lines = "; ".join(
        f"{key.replace('_', ' ')}: {' / '.join(dict.fromkeys(values))}"
        for key, values in grouped.items()
    )
    return (
        identity
        + "\nKnown facts (weave in naturally when relevant, never list them robotically): "
        + fact_lines
    )


def london_time_now():
    """Return London wall time without relying on the optional tzdata package."""
    now_utc = datetime.now(timezone.utc)

    def last_sunday(year, month):
        first_of_next_month = datetime(
            year + (month == 12), (month % 12) + 1, 1, tzinfo=timezone.utc
        )
        last_day = first_of_next_month - timedelta(days=1)
        return last_day - timedelta(days=(last_day.weekday() + 1) % 7)

    # The UK changes clocks at 01:00 UTC on the final Sundays of March/October.
    bst_start = last_sunday(now_utc.year, 3).replace(hour=1)
    bst_end = last_sunday(now_utc.year, 10).replace(hour=1)
    if bst_start <= now_utc < bst_end:
        return now_utc + timedelta(hours=1), "BST"
    return now_utc, "GMT"


def current_time_context():
    """Provide a fresh London clock for each reply; never rely on model guesswork."""
    now, time_zone = london_time_now()
    return (
        "Current server-local date and time: "
        f"{now.strftime('%A, %d %B %Y, %H:%M')} ({time_zone}, London). "
        "Use this for relative dates such as today, tomorrow, later, and next week."
    )


def remember_image_context(user_id, guild_id, description):
    """Keep only the newest useful visual context, rather than every image."""
    description = re.sub(r"\s+", " ", description).strip()[:600]
    if not description:
        return
    now = datetime.now(timezone.utc).isoformat()
    with _db() as database:
        database.execute(
            "INSERT INTO member_memories(user_id, guild_id, memory_key, memory_value, importance, created_at, updated_at) "
            "VALUES (?, ?, 'recent_image', ?, 2, ?, ?) "
            "ON CONFLICT(user_id, guild_id, memory_key) DO UPDATE SET "
            "memory_value = excluded.memory_value, updated_at = excluded.updated_at",
            (user_id, guild_id, description, now, now),
        )


def forget_user(user_id):
    with _db() as database:
        database.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
        database.execute("DELETE FROM member_memories WHERE user_id = ?", (user_id,))
        database.execute("DELETE FROM conversation_summaries WHERE user_id = ?", (user_id,))
        database.execute("DELETE FROM scheduled_messages WHERE requester_id = ?", (user_id,))


autoreact_users = load_autoreact_users()
initialise_memory()
quiz_game = IslamicQuiz(quiz_file, memory_file)


def send_from_console():
    print("Console ready. Type: channel-name | your message")
    print("You can also use a channel ID. Type /quit to stop the bot.")
    while True:
        try:
            entry = input("send> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if entry.lower() == "/quit":
            asyncio.run_coroutine_threadsafe(bot.close(), bot_loop)
            break
        if "|" not in entry:
            print("Format: channel-name | your message")
            continue
        channel_input, message = (part.strip() for part in entry.split("|", 1))
        channel = bot.get_channel(int(channel_input)) if channel_input.isdigit() else discord.utils.get(bot.get_all_channels(), name=channel_input)
        if channel is None or not hasattr(channel, "send"):
            print(f"Channel not found: {channel_input}")
            continue
        try:
            asyncio.run_coroutine_threadsafe(channel.send(message), bot_loop).result(timeout=10)
            print(f"Sent to #{channel.name}")
        except TimeoutError:
            print("Timed out waiting for message to send.")
        except Exception as error:
            print(f"Could not send the message: {error}")


def _call_ai(messages, max_tokens=160, temperature=0.8):
    """
    Unified AI caller. Priority: Groq → OpenRouter → Ollama.
    Falls back automatically if any provider fails.
    """
    def _try_openai_compatible(url, key, model, label):
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()
            brain.groq_alive = True
            brain.log_event(label, "reply_ok")
            return result

    # 1. Try Groq
    if GROQ_API_KEY:
        try:
            result = _try_openai_compatible(GROQ_URL, GROQ_API_KEY, GROQ_MODEL, "groq")
            brain.groq_key_dead = False
            return result
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")[:200]
            except Exception:
                pass
            print(f"[AI] Groq error {e.code}: {body}")
            brain.groq_alive = False
            if e.code in (401, 403):
                brain.groq_key_dead = True
                brain.add_known_issue(f"GROQ KEY INVALID (HTTP {e.code})")
                print(f"[AI] Groq key invalid — trying OpenRouter fallback")
            else:
                brain.add_known_issue(f"Groq HTTP {e.code}")
        except Exception as e:
            print(f"[AI] Groq failed ({type(e).__name__}: {e})")
            brain.groq_alive = False

    # 2. Try OpenRouter
    if OPENROUTER_API_KEY:
        try:
            result = _try_openai_compatible(OPENROUTER_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL, "openrouter")
            brain.resolve_issue("GROQ KEY")
            return result
        except urllib.error.HTTPError as e:
            print(f"[AI] OpenRouter error {e.code}")
        except Exception as e:
            print(f"[AI] OpenRouter failed ({type(e).__name__}: {e})")

    # 3. Try Ollama
    if not _ollama_ping():
        raise RuntimeError("All AI providers unavailable")
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))["message"]["content"].strip()
    except urllib.error.URLError:
        raise RuntimeError("All AI providers unavailable")


def make_ai_reply(history, user_message, member_context, visual=False, user_id=None):
    # Build a rich system prompt that includes brain awareness
    system_parts = [AI_INSTRUCTIONS, current_time_context(), member_context]

    # Inject system status so the AI knows what's available
    status = brain.system_status()
    if status != "All systems nominal":
        system_parts.append(f"[System status: {status}]")

    # Inject user mood/context from brain if available
    if user_id:
        user_state = brain.get_user_state(user_id)
        mood = user_state.get("mood", "neutral")
        if mood and mood != "neutral":
            system_parts.append(f"[User current mood detected: {mood}. Adjust tone accordingly.]")

    messages = [
        {"role": "system", "content": "\n\n".join(system_parts)}
    ] + history + [{"role": "user", "content": user_message}]
    max_tokens  = 180 if visual else 160
    temperature = 0.2 if visual else 0.8
    try:
        return _call_ai(messages, max_tokens=max_tokens, temperature=temperature)
    except urllib.error.URLError as e:
        print(f"[AI] Connection error: {e.reason}")
        raise
    except TimeoutError:
        print("[AI] Request timed out")
        raise
    except Exception as e:
        print(f"[AI] Unexpected error in make_ai_reply: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


def list_ollama_models(silent=False):
    try:
        request = urllib.request.Request(OLLAMA_TAGS_URL, method="GET")
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return [model.get("name", "") for model in payload.get("models", [])]
    except Exception as error:
        if not silent:
            print(f"Could not list Ollama models: {error}")
        return []


def resolve_vision_model():
    # Silence Ollama warnings when running on a Groq-only server
    silent = bool(GROQ_API_KEY) and not brain.ollama_alive
    installed = list_ollama_models(silent=silent)
    names = set(installed)
    for candidate in PREFERRED_VISION_MODELS:
        if candidate in names:
            return candidate
        tagged = next((name for name in installed if name.startswith(candidate + ":")), "")
        if tagged:
            return tagged
    return PREFERRED_VISION_MODELS[0]


def looks_like_video(data):
    return len(data) >= 12 and data[4:8] == b"ftyp"


def prepare_vision_jpeg(image_bytes):
    """Ollama vision models need a still JPEG; GIFs, stickers, and webp often fail raw."""
    if not image_bytes:
        raise ValueError("empty image")
    if looks_like_video(image_bytes):
        raise ValueError("video file is not a still image")
    try:
        image = Image.open(BytesIO(image_bytes))
    except UnidentifiedImageError as error:
        raise ValueError("unreadable image") from error
    frame_count = getattr(image, "n_frames", 1) or 1
    if frame_count > 1:
        # First GIF frames are often a blank or fade-in; pick a later still.
        frame_index = min(max(int(frame_count * 0.25), 1), frame_count - 1)
        image.seek(frame_index)
    image = image.convert("RGB")
    image.thumbnail((1024, 1024))
    if image.size[0] < 8 or image.size[1] < 8:
        raise ValueError("image too small")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def describe_image(image_bytes, caption):
    """Use a local vision model to make a factual description for the chat."""
    global OLLAMA_VISION_MODEL
    # Use cached model from brain if available, only re-resolve if needed
    if brain.vision_model:
        OLLAMA_VISION_MODEL = brain.vision_model
    else:
        OLLAMA_VISION_MODEL = resolve_vision_model()
        brain.vision_model = OLLAMA_VISION_MODEL
    jpeg_bytes = prepare_vision_jpeg(image_bytes)
    flag_note = identify_flag(jpeg_bytes)
    prompt = (
        "Look at the attached still frame. Describe only what is visibly there: "
        "main subject, setting, colours, any readable text, and whether it is a photo, "
        "screenshot, meme, cartoon, or flag. If it is a GIF still, describe the action "
        "implied by that frame. Do not invent objects, people, animals, brands, or countries. "
        "If you cannot tell, say you are unsure. Keep it under 80 words."
    )
    if flag_note:
        prompt += " Extra colour-layout hint (ignore unless this really is a flag): " + flag_note
    if caption and caption not in {
        "Reply briefly to the message above.",
        "What is in this image? Describe it accurately.",
    }:
        prompt += f" The member's caption is: {caption[:500]}"
    payload = {
        "model": OLLAMA_VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(jpeg_bytes).decode("ascii")],
        }],
        "stream": False,
        "options": {"num_predict": 180, "temperature": 0.1},
    }
    request = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        description = (json.loads(response.read().decode("utf-8")).get("message") or {}).get("content", "").strip()
    if not description:
        return "[Image attached; the local vision model returned an empty description.]"
    if flag_note:
        return description + "\n" + flag_note
    return description


def is_visual_attachment(attachment):
    content_type = (attachment.content_type or "").lower()
    suffix = Path(attachment.filename or "").suffix.lower()
    return content_type.startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}


async def resolve_referenced_message(message):
    """Load the message being replied to; on_message often has it unresolved."""
    if not message.reference or not message.reference.message_id:
        return None
    replied = message.reference.resolved
    if isinstance(replied, discord.Message):
        return replied
    try:
        return await message.channel.fetch_message(message.reference.message_id)
    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
        return None


def visual_embed_url(embed):
    """GIFs and pasted images often arrive as embeds, not file attachments."""
    if embed.image and embed.image.url:
        return embed.image.url
    embed_type = (embed.type or "").lower()
    # Tenor/Giphy gifv embeds are videos; the thumbnail is the still we can analyse.
    if embed.thumbnail and embed.thumbnail.url:
        if embed_type in {"image", "gifv", "rich", "article"} or not embed.image:
            return embed.thumbnail.url
    return None


def collect_visual_sources(message):
    sources = []
    seen = set()

    def add(source_type, source):
        key = (source_type, getattr(source, "url", None) or str(source))
        if key in seen:
            return
        seen.add(key)
        sources.append((source_type, source))

    for attachment in message.attachments:
        if is_visual_attachment(attachment):
            add("attachment", attachment)
    for sticker in getattr(message, "stickers", []) or []:
        add("sticker", sticker)
    for embed in message.embeds:
        url = visual_embed_url(embed)
        if url:
            add("url", url)
    return sources


def download_image_url(url):
    request = urllib.request.Request(url, headers=DOWNLOAD_HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_VISION_BYTES + 1)
    if len(data) > MAX_VISION_BYTES:
        raise ValueError("image too large")
    return data


async def read_visual_bytes(source_type, source):
    if source_type == "attachment":
        if source.size and source.size > MAX_VISION_BYTES:
            raise ValueError("image too large")
        try:
            # Fresh uploads (especially on replies) often have no proxy cache yet.
            return await source.read(use_cached=False)
        except Exception:
            return await source.read(use_cached=True)
    if source_type == "sticker":
        return await source.read()
    return await asyncio.to_thread(download_image_url, source)


async def wait_for_visuals(message):
    """Tenor/Giphy GIFs often land as a URL first; the embed arrives a moment later."""
    if collect_visual_sources(message):
        return message
    content = (message.content or "").lower()
    looks_like_gif = any(
        token in content
        for token in ("tenor.com", "giphy.com", "media.discordapp.net", "cdn.discordapp.com", ".gif", ".webp")
    )
    if not looks_like_gif:
        return message
    await asyncio.sleep(1.5)
    try:
        return await message.channel.fetch_message(message.id)
    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
        return message


async def analyse_visual_attachments(message, caption):
    # Check for visual sources first — if there are none, return early without pinging Ollama
    message = await wait_for_visuals(message)
    sources = collect_visual_sources(message)
    if not sources:
        replied = await resolve_referenced_message(message)
        if replied is not None:
            replied = await wait_for_visuals(replied)
            sources = collect_visual_sources(replied)
    if not sources:
        return ""  # no image — skip everything

    # There IS an image — now check if Ollama can handle it
    if _ollama_was_down or not await asyncio.to_thread(_ollama_ping):
        return "[Image attached; vision analysis is only available when running locally with Ollama.]"
    last_failure = "[Image attached; it could not be analysed locally.]"
    for source_type, source in sources[:4]:
        try:
            image_bytes = await read_visual_bytes(source_type, source)
            if not image_bytes:
                last_failure = "[Image attached; it could not be analysed locally.]"
                continue
            return await asyncio.to_thread(describe_image, image_bytes, caption)
        except ValueError as error:
            if "too large" in str(error):
                return "[Image attached, but it is too large for local analysis.]"
            print(f"Local image analysis skipped a source: {error}")
            last_failure = "[Image attached; it could not be analysed locally.]"
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return "[Image attached; the local vision model is not installed yet.]"
            details = error.read().decode("utf-8", errors="replace")[:500]
            print(f"Local image analysis failed: HTTP {error.code}: {details}")
            last_failure = "[Image attached; it could not be analysed locally.]"
        except Exception as error:
            print(f"Local image analysis failed: {error}")
            last_failure = "[Image attached; it could not be analysed locally.]"
    return last_failure


def is_voice_attachment(attachment):
    voice_flag = getattr(attachment, "is_voice_message", False)
    if callable(voice_flag):
        voice_flag = voice_flag()
    content_type = (attachment.content_type or "").lower()
    suffix = Path(attachment.filename or "").suffix.lower()
    return bool(voice_flag) or content_type.startswith("audio/") or suffix in {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".aac", ".flac"}


def transcribe_voice(audio_bytes, filename):
    """
    Transcribe audio. Uses Groq Whisper API if available (works on any server),
    falls back to local faster-whisper if installed, otherwise raises RuntimeError.
    """
    _MIME_TYPES = {
        ".ogg": "audio/ogg", ".opus": "audio/opus", ".mp3": "audio/mpeg",
        ".wav": "audio/wav", ".m4a": "audio/mp4", ".aac": "audio/aac",
        ".flac": "audio/flac",
    }
    suffix = Path(filename or "voice.ogg").suffix.lower() or ".ogg"
    mime_type = _MIME_TYPES.get(suffix, "audio/ogg")

    if GROQ_API_KEY:
        try:
            boundary = "----FormBoundary7MA4YWxkTrZu0gW"
            # Build multipart directly from bytes — no temp file needed
            body = (
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"file\"; filename=\"audio{suffix}\"\r\n"
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode() + audio_bytes + (
                f"\r\n--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"model\"\r\n\r\n"
                f"whisper-large-v3-turbo\r\n"
                f"--{boundary}--\r\n"
            ).encode()
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                data=body,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8")).get("text", "").strip()
        except Exception as e:
            print(f"[Voice] Groq transcription failed: {e} — trying local fallback")

    # Local faster-whisper fallback
    global voice_transcriber
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError("faster-whisper is not installed") from error
    if voice_transcriber is None:
        with _transcriber_lock:
            if voice_transcriber is None:
                voice_transcriber = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    file_descriptor, audio_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(file_descriptor, "wb") as audio_file:
            audio_file.write(audio_bytes)
        segments, _ = voice_transcriber.transcribe(audio_path, beam_size=5, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        Path(audio_path).unlink(missing_ok=True)


async def answer_voice_message(message):
    voices = [attachment for attachment in message.attachments if is_voice_attachment(attachment)]
    if not voices:
        return False
    attachment = voices[0]
    if (attachment.size or 0) > 25 * 1024 * 1024:
        await message.reply("That voice message is too large for my local transcriber - keep it under 25 MB.", mention_author=False)
        return True
    try:
        audio_bytes = await attachment.read(use_cached=True)
        transcript = await asyncio.to_thread(transcribe_voice, audio_bytes, attachment.filename)
    except RuntimeError:
        await message.reply(
            "I can't transcribe voice messages right now — Groq key missing and faster-whisper not installed.",
            mention_author=False,
        )
        print(f"[Reply sent] Text to {message.author.display_name}")
        return True
    except Exception as error:
        print(f"Voice transcription failed: {error}")
        await message.reply("I couldn't transcribe that voice message. Try again in a sec.", mention_author=False)
        print(f"[Reply sent] Text to {message.author.display_name}")
        return True
    if not transcript:
        await message.reply("I couldn't hear clear speech in that voice message.", mention_author=False)
        return True
    guild_id = message.guild.id if message.guild else 0
    asyncio.get_running_loop().run_in_executor(
        None, remember_member_facts, message.author.id, guild_id, transcript
    )
    history = recent_memory(message.author.id, message.channel.id)
    member_context = member_memory_context(message.author.id, guild_id, message.author.display_name)
    try:
        async with message.channel.typing():
            reply = await asyncio.to_thread(make_ai_reply, history, transcript, member_context, False, message.author.id)
    except Exception as error:
        print(f"[AI] Voice transcription AI failed: {type(error).__name__}: {error}")
        brain.log_event("ai", f"voice_ai_failed_{type(error).__name__}", user_id=message.author.id)
        try:
            await asyncio.sleep(3)
            reply = await asyncio.to_thread(make_ai_reply, history, transcript, member_context, False, message.author.id)
        except Exception:
            reply = "I heard you, gimme a sec — my brain just glitched. Try again?"
    reply = (reply or "I heard you, but I need a second - say that again, yeah?")[:1500]
    remember(message.author.id, message.channel.id, "user", transcript)
    remember(message.author.id, message.channel.id, "assistant", reply)
    # Update brain with voice interaction
    brain.update_user_state(message.author.id, pending_voice=True)
    brain.log_event("voice", "replied", user_id=message.author.id)
    asyncio.get_running_loop().run_in_executor(
        None, summarize_old_turns, message.author.id, message.channel.id
    )
    try:
        await send_voice_reply(message, reply)
    except RuntimeError:
        print("Voice reply skipped: edge-tts is not installed.")
        await message.reply(_strip_action_tags(reply), mention_author=False)
    except Exception as error:
        print(f"Voice reply generation failed: {error}")
        await message.reply(_strip_action_tags(reply), mention_author=False)
    return True


def parse_control(message):
    text = message.lower().strip()
    if re.search(r"\b(stop|don't|do not|dont|mute)\b.*\b(reply|respond|talk)\b", text):
        return "mute", None
    if re.search(r"\b(start|resume|unmute)\b.*\b(reply|respond|talk)\b", text):
        return "unmute", None
    if re.search(r"\b(forget|delete|clear)\b.*\b(memory|conversation|chat)\b", text):
        return "forget", None
    match = re.search(r"(?:send|post|remind)\s+(.+?)\s+(?:in)\s+(\d+)\s*(minute|min|hour|hr|day)s?\b", message, re.I)
    if match:
        unit = match.group(3).lower()
        seconds = int(match.group(2)) * ({"minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400}[unit])
        return "schedule", (match.group(1).strip(), seconds)
    return None, None


def requests_voice_reply(message):
    """Recognise natural requests for an audio response, not a special command."""
    text = message.lower()
    patterns = (
        r"\b(?:send|say|reply|answer|read|give|make)\b.{0,60}\b(?:voice|voice message|voice note|audio)\b",
        r"\b(?:voice|voice message|voice note|audio)\b.{0,40}\b(?:reply|message|version)\b",
        r"\bcan you voice\b",
        # Emotional action requests \U00002014 user wants the bot to perform an emotion out loud
        r"\b(?:can you|please|now|go ahead)?\s*(?:cry|laugh|scream|sing|whisper|sob|giggle|"
        r"chuckle|shout|roar|sigh|growl|whimper|yell|weep)\b",
        r"\b(?:do (?:a |your )?(?:laugh|cry|scream|giggle|sob|sigh|shout|yell|whisper|roar|growl|whimper))\b",
        r"\b(?:laugh|cry|scream|giggle|sob|sigh)\s+(?:for me|out loud|please|now|lol|haha)\b",
    )
    return any(re.search(pattern, text, re.I | re.S) for pattern in patterns)


# Prosody presets per emotion: (rate, pitch, volume)
# These are the only knobs edge-tts exposes without full SSML.
_EMOTION_PROSODY = {
    "laughs":       ("+20%", "+8Hz",  "+0%"),
    "laugh":        ("+20%", "+8Hz",  "+0%"),
    "laughing":     ("+20%", "+8Hz",  "+0%"),
    "chuckles":     ("+15%", "+6Hz",  "+0%"),
    "chuckle":      ("+15%", "+6Hz",  "+0%"),
    "giggles":      ("+18%", "+10Hz", "+0%"),
    "giggle":       ("+18%", "+10Hz", "+0%"),
    "whispers":     ("-25%", "-6Hz",  "-20%"),
    "whisper":      ("-25%", "-6Hz",  "-20%"),
    "whispering":   ("-25%", "-6Hz",  "-20%"),
    "sighs":        ("-15%", "-4Hz",  "-5%"),
    "sigh":         ("-15%", "-4Hz",  "-5%"),
    "sighing":      ("-15%", "-4Hz",  "-5%"),
    "cries":        ("-25%", "-10Hz", "-5%"),
    "cry":          ("-25%", "-10Hz", "-5%"),
    "crying":       ("-25%", "-10Hz", "-5%"),
    "sobs":         ("-25%", "-10Hz", "-5%"),
    "sob":          ("-25%", "-10Hz", "-5%"),
    "sobbing":      ("-25%", "-10Hz", "-5%"),
    "sniffles":     ("-20%", "-8Hz",  "-5%"),
    "sniffle":      ("-20%", "-8Hz",  "-5%"),
    "weeps":        ("-25%", "-10Hz", "-5%"),
    "weep":         ("-25%", "-10Hz", "-5%"),
    "whimpers":     ("-20%", "-8Hz",  "-8%"),
    "whimper":      ("-20%", "-8Hz",  "-8%"),
    "excited":      ("+25%", "+12Hz", "+5%"),
    "excitedly":    ("+25%", "+12Hz", "+5%"),
    "sad":          ("-20%", "-8Hz",  "-5%"),
    "sadly":        ("-20%", "-8Hz",  "-5%"),
    "angry":        ("+15%", "+4Hz",  "+10%"),
    "angrily":      ("+15%", "+4Hz",  "+10%"),
    "sarcastically":("+0%",  "-4Hz",  "+0%"),
    "sarcastic":    ("+0%",  "-4Hz",  "+0%"),
    "shocked":      ("+10%", "+14Hz", "+5%"),
    "nervously":    ("-5%",  "+6Hz",  "-5%"),
    "softly":       ("-15%", "-2Hz",  "-15%"),
    "shouting":     ("+20%", "+6Hz",  "+20%"),
    "shouts":       ("+20%", "+6Hz",  "+20%"),
    "shout":        ("+20%", "+6Hz",  "+20%"),
    "screams":      ("+25%", "+8Hz",  "+25%"),
    "scream":       ("+25%", "+8Hz",  "+25%"),
    "screaming":    ("+25%", "+8Hz",  "+25%"),
    "yells":        ("+20%", "+6Hz",  "+20%"),
    "yell":         ("+20%", "+6Hz",  "+20%"),
    "growls":       ("+10%", "-8Hz",  "+10%"),
    "growl":        ("+10%", "-8Hz",  "+10%"),
    "roars":        ("+25%", "-6Hz",  "+20%"),
    "roar":         ("+25%", "-6Hz",  "+20%"),
    "talks":        ("+5%",  "+0Hz",  "+0%"),
    "says":         ("+5%",  "+0Hz",  "+0%"),
    "smiles":       ("+10%", "+4Hz",  "+0%"),
    "grins":        ("+10%", "+4Hz",  "+0%"),
    "pauses":       ("+0%",  "+0Hz",  "+0%"),
    "thinks":       ("-5%",  "-2Hz",  "-5%"),
}
_DEFAULT_PROSODY = ("+5%", "+0Hz", "+0%")

# What to actually SPEAK when an emotion tag appears in voice mode.
# The bot vocalises the emotion rather than going silent.
_EMOTION_SOUNDS = {
    "laughs":       "haha, haha!",
    "laugh":        "haha!",
    "laughing":     "hahaha!",
    "chuckles":     "heh heh.",
    "chuckle":      "heh.",
    "giggles":      "hehe!",
    "giggle":       "hehe!",
    "cries":        "oh no... huh huh huh...",
    "cry":          "huh huh huh...",
    "crying":       "huh... huh huh...",
    "sobs":         "huh... huh huh...",
    "sob":          "huh huh...",
    "sobbing":      "huh... huh huh huh...",
    "sniffles":     "sniff... sniff.",
    "sniffle":      "sniff.",
    "weeps":        "oh... huh huh...",
    "weep":         "huh huh...",
    "whimpers":     "uhh...",
    "whimper":      "uhh...",
    "sighs":        "huhhh...",
    "sigh":         "huh...",
    "sighing":      "huhhh...",
    "screams":      "AAARGH!",
    "scream":       "AAAH!",
    "screaming":    "AAAARGH!",
    "shouts":       "OI!",
    "shout":        "OI!",
    "shouting":     "OI!",
    "yells":        "HEY!",
    "yell":         "HEY!",
    "growls":       "grrrr.",
    "growl":        "grrrr.",
    "roars":        "RAAARGH!",
    "roar":         "RAAH!",
    "whispers":     "",   # no replacement, just quieter prosody on following text
    "whisper":      "",
    "whispering":   "",
    # Neutral/meta \U00002014 just skip these, don't speak them
    "talks":        "",
    "says":         "",
    "smiles":       "",
    "grinning":     "",
    "grins":        "",
    "pauses":       "",
    "thinks":       "",
    "excited":      "",
    "excitedly":    "",
    "sad":          "",
    "sadly":        "",
    "angry":        "",
    "angrily":      "",
    "sarcastically":"",
    "sarcastic":    "",
    "shocked":      "",
    "nervously":    "",
    "softly":       "",
}


def _parse_emotion_segments(text):
    """Split text into (spoken_text, rate, pitch, volume) segments.

    Emotion tags like *laughs* are replaced with the actual vocalised sound
    at the matching prosody. Neutral/meta tags are dropped silently.
    Following plain text inherits the emotion prosody.
    Parenthetical asides are dropped entirely.
    """
    # Drop parenthetical meta-commentary like (with a cheeky accent)
    text = re.sub(r"\([^)]{1,80}\)", "", text)

    token_re = re.compile(r"[*_]{1,2}([^*_\n]{1,40})[*_]{1,2}")
    segments = []
    current_prosody = _DEFAULT_PROSODY
    last_end = 0

    for m in token_re.finditer(text):
        # Plain text before this tag \U00002014 speak it at current prosody
        before = text[last_end:m.start()].strip()
        if before:
            clean = re.sub(r"[*_]+", "", before).strip()
            clean = re.sub(r" {2,}", " ", clean)
            if clean:
                segments.append((clean, *current_prosody))

        tag_full = m.group(1).strip().lower()
        tag_stem = tag_full.rstrip("s")  # e.g. "laughs" -> "laugh"

        # Look up prosody and sound for this tag
        prosody = _EMOTION_PROSODY.get(tag_full) or _EMOTION_PROSODY.get(tag_stem) or _DEFAULT_PROSODY
        sound = _EMOTION_SOUNDS.get(tag_full)
        if sound is None:
            sound = _EMOTION_SOUNDS.get(tag_stem, "")

        current_prosody = prosody

        # Insert the vocalised sound if there is one
        if sound:
            segments.append((sound, *prosody))

        last_end = m.end()

    # Remaining text after last tag
    tail = text[last_end:].strip()
    if tail:
        clean = re.sub(r"[*_]+", "", tail).strip()
        clean = re.sub(r" {2,}", " ", clean)
        if clean:
            segments.append((clean, *current_prosody))

    # If nothing was extracted, fall back to the raw cleaned text at default prosody
    if not segments:
        fallback = re.sub(r"[*_]{1,2}[^*_\n]{1,40}[*_]{1,2}", "", text).strip()
        fallback = re.sub(r"[*_]+", "", fallback).strip()
        if fallback:
            segments.append((fallback, *_DEFAULT_PROSODY))

    return segments


def _strip_action_tags(text):
    """Remove *action* and _action_ stage directions from text replies.
    Used for the plain-text path so members never see *laughs* printed."""
    text = re.sub(r"[*_]{1,2}[^*_\n]{1,60}[*_]{1,2}", "", text)
    text = re.sub(r"\([^)]{1,80}\)", "", text)
    text = re.sub(r"[*_]+", "", text)
    return re.sub(r" {2,}", " ", text).strip()


def _elevenlabs_tts_sync(text, emotion_hint):
    """
    Call ElevenLabs API synchronously and return raw MP3 bytes.
    Raises ElevenLabsQuotaError if the monthly quota is exhausted.
    """
    _EL_EMOTION_SETTINGS = {
        "laugh":    (0.25, 0.75, 0.90, True),
        "cry":      (0.20, 0.80, 0.85, True),
        "whisper":  (0.60, 0.70, 0.30, False),
        "shout":    (0.20, 0.80, 0.95, True),
        "sad":      (0.30, 0.75, 0.70, True),
        "excited":  (0.20, 0.80, 0.90, True),
        "angry":    (0.20, 0.85, 0.90, True),
        "giggle":   (0.25, 0.75, 0.85, True),
        "neutral":  (0.50, 0.75, 0.45, True),
    }
    stability, similarity, style, speaker_boost = _EL_EMOTION_SETTINGS.get(
        emotion_hint, _EL_EMOTION_SETTINGS["neutral"]
    )
    payload = json.dumps({
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity,
            "style": style,
            "use_speaker_boost": speaker_boost,
        },
    }).encode("utf-8")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        # 429 = quota exhausted or rate limit; 401 = bad key
        if e.code in (429, 401) or "quota" in body.lower() or "exceeded" in body.lower():
            raise _ElevenLabsQuotaError(f"ElevenLabs quota/auth error {e.code}: {body[:200]}") from e
        raise


class _ElevenLabsQuotaError(Exception):
    """Raised when ElevenLabs returns a quota-exceeded or auth error."""


async def send_voice_reply(message, text):
    """Send an emotionally expressive voice reply using ElevenLabs (if key set) or edge-tts."""
    segments = _parse_emotion_segments(text)
    # If parsing produced nothing speakable, fall back to the stripped plain text
    if not segments:
        fallback = _strip_action_tags(text)
        if fallback:
            segments = [(fallback, *_DEFAULT_PROSODY)]
        else:
            raise ValueError("No speakable content in reply")

    # Detect dominant emotion directly from the raw text tags
    _TAG_TO_EL = {
        "laugh": "laugh", "laughs": "laugh", "laughing": "laugh",
        "chuckle": "laugh", "chuckles": "laugh", "giggle": "giggle", "giggles": "giggle",
        "cry": "cry", "cries": "cry", "crying": "cry",
        "sob": "cry", "sobs": "cry", "sobbing": "cry",
        "weep": "cry", "weeps": "cry", "sniffle": "cry", "sniffles": "cry",
        "whimper": "cry", "whimpers": "cry",
        "whisper": "whisper", "whispers": "whisper", "whispering": "whisper",
        "shout": "shout", "shouts": "shout", "shouting": "shout",
        "scream": "shout", "screams": "shout", "screaming": "shout",
        "yell": "shout", "yells": "shout", "roar": "shout", "roars": "shout",
        "angry": "angry", "angrily": "angry", "growl": "angry", "growls": "angry",
        "sad": "sad", "sadly": "sad", "sigh": "sad", "sighs": "sad",
        "excited": "excited", "excitedly": "excited", "shocked": "excited",
    }
    # Scan original text for emotion tags to determine dominant emotion
    dominant_emotion = "neutral"
    tag_re = re.compile(r"[*_]{1,2}([^*_\n]{1,40})[*_]{1,2}")
    for m in tag_re.finditer(text):
        tag = m.group(1).strip().lower()
        if tag in _TAG_TO_EL:
            dominant_emotion = _TAG_TO_EL[tag]
            break

    full_text = " ".join(seg[0] for seg in segments)

    final_path = None
    try:
        fd, final_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)

        if ELEVENLABS_API_KEY:
            try:
                # Check if quota was exhausted — auto-reset at start of new month
                global _elevenlabs_quota_exhausted, _elevenlabs_quota_reset_month
                current_month = datetime.now(timezone.utc).month
                if not brain.elevenlabs_alive and _elevenlabs_quota_reset_month != current_month:
                    brain.elevenlabs_alive = True
                    _elevenlabs_quota_reset_month = current_month
                    brain.resolve_issue("ElevenLabs")
                    print("ElevenLabs quota reset — trying ElevenLabs again (new month).")

                if brain.elevenlabs_alive:
                    mp3_bytes = await asyncio.to_thread(_elevenlabs_tts_sync, full_text, dominant_emotion)
                    Path(final_path).write_bytes(mp3_bytes)
                    brain.log_event("tts", "elevenlabs_ok")
                    print(f"ElevenLabs TTS OK: emotion={dominant_emotion}, chars={len(full_text)}")
                else:
                    print("ElevenLabs quota exhausted — using edge-tts fallback.")
                    await _edge_tts_render(segments, final_path)
            except _ElevenLabsQuotaError as quota_err:
                brain.elevenlabs_alive = False
                _elevenlabs_quota_reset_month = datetime.now(timezone.utc).month
                brain.add_known_issue("ElevenLabs quota exhausted")
                brain.log_event("tts", "elevenlabs_quota_exhausted")
                print(f"ElevenLabs quota exhausted — switching to edge-tts for this month. ({quota_err})")
                await _edge_tts_render(segments, final_path)
            except Exception as el_error:
                print(f"ElevenLabs TTS error, falling back to edge-tts: {el_error}")
                brain.log_event("tts", f"elevenlabs_error_{type(el_error).__name__}")
                await _edge_tts_render(segments, final_path)
        else:
            await _edge_tts_render(segments, final_path)

        await message.reply(
            "\U0001f399",
            file=discord.File(final_path, filename="voice-reply.mp3"),
            mention_author=False,
        )
    finally:
        if final_path:
            try:
                Path(final_path).unlink(missing_ok=True)
            except Exception:
                pass


async def _edge_tts_render(segments, output_path):
    """Render emotion segments with edge-tts and write to output_path."""
    try:
        import edge_tts
    except ImportError as error:
        raise RuntimeError("edge-tts is not installed") from error

    seg_files = []
    try:
        for seg_text, rate, pitch, vol in segments:
            fd, seg_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            seg_files.append(seg_path)
            speech = edge_tts.Communicate(seg_text[:1800], voice=TTS_VOICE,
                                          rate=rate, pitch=pitch, volume=vol)
            await speech.save(seg_path)

        with open(output_path, "wb") as out:
            for seg_path in seg_files:
                with open(seg_path, "rb") as inp:
                    out.write(inp.read())
    finally:
        for p in seg_files:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


def schedule_message(user_id, channel_id, content, seconds):
    due = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    with _db() as database:
        database.execute("INSERT INTO scheduled_messages(requester_id, channel_id, content, run_at) VALUES (?, ?, ?, ?)", (user_id, channel_id, content[:1900], due.isoformat()))
    return due


def parse_delay(delay):
    match = re.fullmatch(r"\s*(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)\s*", delay.lower())
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    multiplier = 60 if unit.startswith("m") else 3600 if unit.startswith(("h", "hr")) else 86400
    return amount * multiplier


async def is_reply_to_bot(message):
    replied = await resolve_referenced_message(message)
    return bool(replied and bot.user and replied.author.id == bot.user.id)


async def answer_with_ai(message):
    # Strip the bot mention before any control parsing so natural-language
    # commands like "@Bot remind me to pray in 30 minutes" work correctly.
    stripped_content = message.content
    if bot.user:
        stripped_content = stripped_content.replace(bot.user.mention, "").replace(f"<@!{bot.user.id}>", "").strip()

    # Log every incoming ping/reply so autofix can detect silent failures
    try:
        print(f"[Reply] Incoming from {message.author.display_name}: {stripped_content[:80]!r}".encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
    except Exception:
        print("[Reply] Incoming message (could not print display name)")

    # Update brain user state
    brain.update_user_state(message.author.id, pending_voice=False)
    brain.log_event("reply", "incoming", user_id=message.author.id)

    action, value = parse_control(stripped_content)
    if action == "mute":
        set_ai_enabled(message.author.id, False)
        await message.reply("Say less - I won't reply to you again unless you ask me to resume.")
        print(f"[Reply sent] Text to {message.author.display_name}")
        return
    if action == "unmute":
        set_ai_enabled(message.author.id, True)
        await message.reply("You're back on the list. What's good?")
        print(f"[Reply sent] Text to {message.author.display_name}")
        return
    if action == "forget":
        forget_user(message.author.id)
        await message.reply("Your saved conversation memory has been cleared.")
        print(f"[Reply sent] Text to {message.author.display_name}")
        return
    if action == "schedule":
        content, seconds = value
        due = schedule_message(message.author.id, message.channel.id, content, seconds)
        await message.reply(f"Calm. I'll send that here at {due.strftime('%H:%M UTC')}.")
        print(f"[Reply sent] Text to {message.author.display_name}")
        return
    if not ai_enabled_for(message.author.id):
        print(f"[Reply sent] Skipped (AI disabled) for {message.author.display_name}")
        return

    # Per-user rate limit — one concurrent AI reply per user maximum.
    # Also enforce a 2-second minimum gap between replies to prevent spam.
    if message.author.id not in _user_ai_locks:
        _user_ai_locks[message.author.id] = asyncio.Lock()
    user_lock = _user_ai_locks[message.author.id]
    if user_lock.locked():
        return  # user already has a reply in flight
    # Sequential spam guard — only throttle if a reply was recently SENT
    user_state = brain.get_user_state(message.author.id)
    last_reply = user_state.get("last_reply_sent")
    if last_reply and isinstance(last_reply, datetime):
        elapsed = (datetime.now(timezone.utc) - last_reply).total_seconds()
        if elapsed < 2.0:
            return  # too fast, drop silently
    async with user_lock:
        try:
            await _answer_with_ai_inner(message, stripped_content)
        except Exception as e:
            print(f"[answer_with_ai] Unhandled exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            try:
                await message.reply("Something went wrong on my end — try again in a sec.", mention_author=False)
            except Exception:
                pass
            print(f"[Reply sent] Fallback to {message.author.display_name}")


def _is_reply_sane(user_message: str, reply: str) -> tuple:
    """
    Fast heuristic sanity check — no AI call needed.
    Returns (is_sane, reason). is_sane=False means the reply is wrong.
    """
    reply_lower = reply.lower()
    msg_lower   = user_message.lower()

    # System/error messages leaking into replies
    bad_phrases = [
        "image analysis needs ollama",
        "vision analysis is only available",
        "local vision model is not installed",
        "bear with me, my brain",
        "oi, the local ai is not ready",
        "local ai is not ready",
        "check that ollama is running",
        "i can't transcribe voice messages right now",
        "something went wrong on my end",
        "could not list ollama",
    ]
    for phrase in bad_phrases:
        if phrase in reply_lower:
            return False, f"system_error_leaked:{phrase[:40]}"

    # Empty reply
    if not reply.strip():
        return False, "empty_reply"

    # Image error reply when no image was in the input
    image_error_phrases = ["i can see you sent an image", "image attached", "vision model"]
    image_input_hints   = ["image", "photo", "picture", "pic", ".gif", ".jpg", ".png", "look at this"]
    reply_has_img_error = any(p in reply_lower for p in image_error_phrases)
    input_has_img_hint  = any(p in msg_lower for p in image_input_hints)
    if reply_has_img_error and not input_has_img_hint:
        return False, "image_error_on_text_message"

    return True, ""


async def _answer_with_ai_inner(message, stripped_content):
    """Inner handler called only when the per-user lock is held."""
    user_message = stripped_content
    user_message = user_message.strip()
    guild_id = message.guild.id if message.guild else 0
    image_description = await analyse_visual_attachments(message, user_message)
    if not user_message:
        user_message = (
            "What is in this image? Describe it accurately."
            if image_description
            else "Reply briefly to the message above."
        )
    if image_description:
        if "not installed yet" in image_description or "only available when running locally" in image_description:
            await message.reply(
                "I can see you sent an image but I can't analyse it right now — image analysis needs Ollama running locally. I can still chat though!",
                mention_author=False,
            )
            print(f"[Reply sent] Text to {message.author.display_name}")
            return
        if not image_description.startswith("["):
            remember_image_context(message.author.id, guild_id, image_description)
        user_message += (
            "\n\nVisual analysis (authoritative; do not contradict or invent extra objects):\n"
            + image_description
        )
    # Run fact extraction with the ORIGINAL user message only — not the augmented version
    # containing the visual analysis blob, which would produce garbage facts.
    asyncio.get_running_loop().run_in_executor(
        None, remember_member_facts, message.author.id, guild_id, stripped_content
    )
    # Old chat turns often contain wrong picture guesses; don't let them override a new image.
    history = [] if image_description else recent_memory(message.author.id, message.channel.id)
    member_context = member_memory_context(message.author.id, guild_id, message.author.display_name)

    # Detect emotional voice requests early so we can prime the AI properly.
    # Maps trigger words to the emotion tag the AI should produce.
    _EMOTION_TRIGGERS = {
        "cry": ("*sobs* *sniffles*", "sad"),
        "crying": ("*sobs* *sniffles*", "sad"),
        "laugh": ("*laughs* hahaha!", "laughs"),
        "laughing": ("*laughs* hahaha!", "laughs"),
        "scream": ("*screams*", "shouting"),
        "screaming": ("*screams*", "shouting"),
        "sing": (None, None),
        "whisper": ("*whispers*", "whispers"),
        "whispering": ("*whispers*", "whispers"),
        "sob": ("*sobs*", "sad"),
        "sobbing": ("*sobs*", "sad"),
        "giggle": ("*giggles*", "giggles"),
        "giggling": ("*giggles*", "giggles"),
        "chuckle": ("*chuckles*", "chuckles"),
        "shout": ("*shouts*", "shouting"),
        "shouting": ("*shouts*", "shouting"),
        "yell": ("*shouts*", "shouting"),
        "sigh": ("*sighs*", "sighs"),
        "sighing": ("*sighs*", "sighs"),
        "roar": ("*roars*", "shouting"),
        "growl": ("*growls*", "angry"),
        "whimper": ("*whimpers*", "sad"),
        "weep": ("*sobs*", "sad"),
    }
    voice_emotion_hint = None
    if requests_voice_reply(user_message):
        msg_lower = user_message.lower()
        for trigger, (hint, _) in _EMOTION_TRIGGERS.items():
            if re.search(rf"\b{trigger}\b", msg_lower) and hint:
                voice_emotion_hint = hint
                break

    # If this is a voice+emotion request, override the AI prompt so it produces
    # the right emotion tag that the TTS prosody engine can act on.
    ai_user_message = user_message
    if voice_emotion_hint:
        ai_user_message = (
            f"{user_message}\n\n"
            f"[Voice instruction: the user wants you to perform this emotion out loud. "
            f"Start your reply with the emotion tag exactly as shown: {voice_emotion_hint} "
            f"Then add a short natural reaction in character. Keep it under 2 sentences.]"
        )

    try:
        async with message.channel.typing():
            reply = await asyncio.to_thread(
                make_ai_reply, history, ai_user_message, member_context, bool(image_description), message.author.id
            )
        # Validate reply before sending — catch silent wrong behaviour
        reply = (reply or "I'm drawing a blank for a sec. Try that again, yeah?")[:1900]
        sane, reason = _is_reply_sane(user_message, reply)
        if not sane:
            print(f"[Validator] Bad reply detected ({reason}) — regenerating...")
            brain.log_event("validator", f"bad_{reason}", user_id=message.author.id)
            brain.add_known_issue(f"Bad reply: {reason}")
            safe_prompt = user_message + "\n\n[Reply naturally. Do not mention images, vision, Ollama, or system errors unless the user actually asked about them.]"
            try:
                async with message.channel.typing():
                    reply = await asyncio.to_thread(
                        make_ai_reply, history, safe_prompt, member_context, False, message.author.id
                    )
                reply = (reply or "I'm drawing a blank for a sec. Try that again, yeah?")[:1900]
            except Exception:
                pass
    except Exception as error:
        print(f"[AI] Request failed: {type(error).__name__}: {error}")
        traceback.print_exc()
        brain.log_event("ai", f"reply_failed_{type(error).__name__}", user_id=message.author.id)
        # Groq key invalid — fall through silently, OpenRouter/Ollama will handle it
        if "GROQ_KEY_INVALID" in str(error):
            print("[AI] Groq key invalid — retrying with fallback provider...")
            # Don't send error message to user — just retry silently
        brain.add_known_issue(f"AI reply failed: {type(error).__name__}")
        if not _ollama_was_down:
            asyncio.get_running_loop().run_in_executor(None, _try_start_ollama)
        try:
            await asyncio.sleep(3)
            async with message.channel.typing():
                reply = await asyncio.to_thread(
                    make_ai_reply, history, ai_user_message, member_context, bool(image_description), message.author.id
                )
        except Exception:
            await message.reply("Bear with me, my brain's loading... try again in a sec.", mention_author=False)
            return
    reply = (reply or "I'm drawing a blank for a sec. Try that again, yeah?")[:1900]
    remember(message.author.id, message.channel.id, "user", user_message)
    remember(message.author.id, message.channel.id, "assistant", reply)
    asyncio.get_running_loop().run_in_executor(
        None, summarize_old_turns, message.author.id, message.channel.id
    )
    if requests_voice_reply(user_message):
        try:
            await send_voice_reply(message, reply)
            print(f"[Reply sent] Voice to {message.author.display_name}")
            return
        except RuntimeError as error:
            print(f"Voice reply runtime error: {error}")
            await message.reply(
                _strip_action_tags(reply) + "\n\n*Voice replies need `py -m pip install edge-tts` once, then a restart.*",
                mention_author=False,
            )
            print(f"[Reply sent] Text to {message.author.display_name}")
            return
        except Exception as error:
            print(f"Voice reply generation failed (full error): {type(error).__name__}: {error}")
            traceback.print_exc()
            await message.reply(_strip_action_tags(reply), mention_author=False)
            print(f"[Reply sent] Text to {message.author.display_name}")
            return
    await message.reply(_strip_action_tags(reply), mention_author=False)
    brain.update_user_state(message.author.id, last_reply_sent=datetime.now(timezone.utc))
    print(f"[Reply sent] Text to {message.author.display_name}")


@tasks.loop(seconds=30)
async def deliver_scheduled_messages():
    now = datetime.now(timezone.utc).isoformat()
    with _db() as database:
        due = database.execute("SELECT id, channel_id, content FROM scheduled_messages WHERE delivered = 0 AND run_at <= ?", (now,)).fetchall()
    for message_id, channel_id, content in due:
        try:
            channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
            await channel.send(content)
            try:
                with _db() as database:
                    database.execute("UPDATE scheduled_messages SET delivered = 1 WHERE id = ?", (message_id,))
            except Exception as db_err:
                print(f"Could not mark scheduled message {message_id} delivered: {db_err}")
        except (discord.Forbidden, discord.NotFound, discord.HTTPException) as error:
            print(f"Could not deliver scheduled message {message_id}: {error}")


_ollama_was_down = False
_ollama_consecutive_failures = 0
_self_heal_log: list = []  # rolling log of self-heal events


def _log_heal(msg):
    """Add a timestamped entry to the self-heal log (kept to last 50 events)."""
    entry = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] {msg}"
    _self_heal_log.append(entry)
    if len(_self_heal_log) > 50:
        _self_heal_log.pop(0)
    print(f"[Self-heal] {msg}")


def _ollama_ping():
    """Return True if Ollama responds within 3 seconds."""
    try:
        req = urllib.request.Request(OLLAMA_TAGS_URL, method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def _ollama_model_loaded(model_name):
    """Return True if the model responds to a tiny test prompt."""
    try:
        payload = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "options": {"num_predict": 1},
        }).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        return True
    except Exception:
        return False


def _try_start_ollama():
    """Attempt to launch Ollama serve as a background process."""
    try:
        import subprocess
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return True
    except Exception as e:
        _log_heal(f"Could not auto-start Ollama: {e}")
        return False


@tasks.loop(seconds=45)
async def ollama_health_check():
    """
    Full self-healing watchdog — runs every 45 seconds.

    What it checks and fixes automatically:
    - Ollama process not reachable  -> tries to restart it
    - Model not responding          -> sends a warm-up request to load it into memory
    - Model missing from Ollama     -> logs a clear message with the pull command
    - Recovered from outage         -> logs recovery time
    """
    global _ollama_was_down, _ollama_consecutive_failures

    alive = await asyncio.to_thread(_ollama_ping)

    if not alive:
        _ollama_consecutive_failures += 1
        if not _ollama_was_down:
            _log_heal("Ollama is unreachable. Attempting auto-restart...")
            _ollama_was_down = True
            brain.ollama_alive = False
            brain.add_known_issue("Ollama unreachable")
            brain.log_event("ollama", "went_down")
        if _ollama_consecutive_failures <= 3:
            started = await asyncio.to_thread(_try_start_ollama)
            if started:
                _log_heal("Ollama restart command sent. Will verify next cycle.")
        return

    # Ollama is alive — check if our model is actually loaded and responsive
    if _ollama_was_down:
        _log_heal(f"Ollama is back online after {_ollama_consecutive_failures} failed check(s).")
        _ollama_was_down = False
        _ollama_consecutive_failures = 0
        brain.ollama_alive = True
        brain.vision_model = ""  # force re-resolve vision model now Ollama is back
        brain.resolve_issue("Ollama")
        brain.log_event("ollama", "recovered")

    # Verify the chat model is installed and warm
    model_ok = await asyncio.to_thread(_ollama_model_loaded, OLLAMA_MODEL)
    if not model_ok:
        _log_heal(
            f"Model '{OLLAMA_MODEL}' is not responding. "
            f"If not installed, run: ollama pull {OLLAMA_MODEL}"
        )


@ollama_health_check.error
async def ollama_health_check_error(error):
    print(f"[Self-heal] Health check task error: {error}")
    traceback.print_exc()


# --- Owner ID — set this to your Discord user ID to receive DM alerts ---
# Get your ID: enable Developer Mode in Discord → right-click your name → Copy ID
BOT_OWNER_ID = int(os.environ.get("BOT_OWNER_ID", "0"))


@tasks.loop(minutes=10)
async def groq_health_check():
    """Every 10 minutes verify Groq is reachable. DM the owner if the key dies."""
    if not GROQ_API_KEY or brain.groq_key_dead:
        return
    try:
        payload = json.dumps({
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            GROQ_URL, data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"},
            method="POST",
        )
        await asyncio.to_thread(lambda: urllib.request.urlopen(req, timeout=10).read())
        if not brain.groq_alive:
            print("[Groq] Back online.")
            brain.groq_alive = True
            brain.groq_key_dead = False
            brain.resolve_issue("Groq")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            brain.groq_alive = False
            brain.groq_key_dead = True
            msg = (
                "**[BOT ALERT] Groq API key is INVALID**\n"
                f"Error: HTTP {e.code}\n\n"
                "**To fix:**\n"
                "1. Go to <https://console.groq.com> -> API Keys -> Create new key\n"
                "2. Go to bot-hosting.net -> your deployment -> Env tab\n"
                "3. Update `GROQ_API_KEY` with the new key\n"
                "4. Restart the deployment\n\n"
                "Bot cannot reply to anyone until this is fixed."
            )
            print(f"[GROQ ALERT] Key invalid (HTTP {e.code})")
            if BOT_OWNER_ID and brain.should_alert("groq_key_dead"):
                try:
                    owner = await bot.fetch_user(BOT_OWNER_ID)
                    await owner.send(msg)
                    print(f"[GROQ ALERT] DM sent to owner {BOT_OWNER_ID}")
                except Exception as dm_err:
                    print(f"[GROQ ALERT] Could not DM owner: {dm_err}")
            # Stop checking until manually restarted — key needs human action
            groq_health_check.stop()
    except Exception:
        pass


@groq_health_check.error
async def groq_health_check_error(error):
    print(f"[Groq health] Task error: {error}")


@bot.event
async def on_ready():
    global bot_loop, console_started, slash_commands_synced, OLLAMA_VISION_MODEL
    bot_loop = asyncio.get_running_loop()
    key_status = f"GROQ key: {'set (' + GROQ_API_KEY[:8] + '...)' if GROQ_API_KEY else 'NOT SET'}"
    print(f"Bot is online as {bot.user} | {key_status}")
    # Resolve vision model in background so on_ready doesn't block
    async def _resolve_vision_async():
        global OLLAMA_VISION_MODEL
        OLLAMA_VISION_MODEL = await asyncio.to_thread(resolve_vision_model)
        brain.vision_model = OLLAMA_VISION_MODEL
        print(f"Using vision model: {OLLAMA_VISION_MODEL}")
    asyncio.ensure_future(_resolve_vision_async())
    if not console_started:
        console_started = True
        threading.Thread(target=send_from_console, daemon=True).start()
    if not deliver_scheduled_messages.is_running():
        deliver_scheduled_messages.start()
    if not ollama_health_check.is_running():
        ollama_health_check.start()
    if not groq_health_check.is_running():
        groq_health_check.start()
    if not slash_commands_synced:
        try:
            # Only sync if command schema has changed — avoids hitting rate limits on every restart
            cmd_hash_file = Path(__file__).with_name(".cmd_hash")
            current_cmds = sorted(c.name for c in bot.tree.get_commands())
            current_hash = hashlib.md5(json.dumps(current_cmds).encode()).hexdigest()
            saved_hash = cmd_hash_file.read_text().strip() if cmd_hash_file.exists() else ""
            if current_hash != saved_hash:
                synced = await bot.tree.sync()
                for guild in bot.guilds:
                    bot.tree.copy_global_to(guild=guild)
                    await bot.tree.sync(guild=guild)
                cmd_hash_file.write_text(current_hash)
                print(f"Registered {len(synced)} Discord slash commands.")
            else:
                print("Slash commands unchanged — skipping sync.")
            slash_commands_synced = True
        except discord.HTTPException as error:
            print(f"Could not register slash commands: {error}")


@bot.hybrid_command(description="Turn on skull reactions for your messages.")
async def autoreacton(ctx):
    autoreact_users.add(ctx.author.id)
    save_autoreact_users()
    await ctx.send(f"Auto-reaction \U0001F480 enabled for {ctx.author.mention}.")


@bot.hybrid_command(description="Turn off skull reactions for your messages.")
async def autoreactoff(ctx):
    autoreact_users.discard(ctx.author.id)
    save_autoreact_users()
    await ctx.send(f"Auto-reaction disabled for {ctx.author.mention}.")


@bot.hybrid_command(description="Allow the AI to reply when you ping or reply to it.")
async def aion(ctx):
    set_ai_enabled(ctx.author.id, True)
    await ctx.send(f"AI replies enabled for {ctx.author.mention}.")


@bot.hybrid_command(description="Stop AI replies to you until you turn them back on.")
async def aioff(ctx):
    """Stop AI replies to the command author until !aion is used."""
    set_ai_enabled(ctx.author.id, False)
    await ctx.send(f"AI replies disabled for {ctx.author.mention}. Use `!aion` to enable them again.")


@bot.hybrid_command(description="Check whether AI replies are enabled for you.")
async def aistatus(ctx):
    state = "enabled" if ai_enabled_for(ctx.author.id) else "disabled"
    await ctx.send(f"AI replies are **{state}** for {ctx.author.mention}.")


@bot.hybrid_command(description="Show the bot's system status and health.")
async def botstatus(ctx):
    """Show brain health: AI services, quota, known issues."""
    lines = [
        f"**System status:** {brain.system_status()}",
        f"**Groq requests today:** {brain.groq_requests_today}",
        f"**Ollama:** {'online' if brain.ollama_alive else 'offline'}",
        f"**Groq:** {'online' if brain.groq_alive else 'offline'}",
        f"**ElevenLabs:** {'online' if brain.elevenlabs_alive else 'quota exhausted'}",
    ]
    if brain.known_issues:
        lines.append(f"**Known issues:** {', '.join(brain.known_issues[-5:])}")
    await ctx.send("\n".join(lines))


@bot.hybrid_command(description="Schedule a message for this channel, for example: 2h then your message.")
async def remind(ctx, delay: str = "", *, message: str = ""):
    """Schedule a channel message for later delivery."""
    seconds = parse_delay(delay)
    if not seconds or not message.strip():
        await ctx.send("Use `!remind 2h your message`, `!remind 30m your message`, or `!remind 1d your message`.")
        return
    if seconds > 31_536_000:
        await ctx.send("Please choose a delay shorter than one year.")
        return
    due = schedule_message(ctx.author.id, ctx.channel.id, message.strip(), seconds)
    await ctx.send(f"Reminder scheduled for **{due.strftime('%Y-%m-%d %H:%M UTC')}**.")


@bot.hybrid_command(description="Show your pending scheduled messages in this channel.")
async def scheduled(ctx):
    """Show pending scheduled messages for this user in this channel."""
    with _db() as database:
        rows = database.execute(
            "SELECT id, content, run_at FROM scheduled_messages "
            "WHERE requester_id = ? AND channel_id = ? AND delivered = 0 ORDER BY run_at LIMIT 10",
            (ctx.author.id, ctx.channel.id),
        ).fetchall()
    if not rows:
        await ctx.send("You have no pending messages in this channel.")
        return
    entries = "\n".join(f"`#{item_id}` - {run_at}: {content[:120]}" for item_id, content, run_at in rows)
    await ctx.send("**Your pending messages**\n" + entries)


@bot.hybrid_command(description="Delete the long-term memory saved about you.")
async def forgetme(ctx):
    forget_user(ctx.author.id)
    await ctx.send(f"Saved memory cleared for {ctx.author.mention}.")


@bot.hybrid_command(description="Show the useful facts the bot has saved about you.")
async def memory(ctx):
    """Show the durable facts the bot has saved for the command author."""
    guild_id = ctx.guild.id if ctx.guild else 0
    with _db() as database:
        rows = database.execute(
            "SELECT memory_key, memory_value FROM member_memories WHERE user_id = ? AND guild_id = ? "
            "ORDER BY importance DESC, updated_at DESC LIMIT 24",
            (ctx.author.id, guild_id),
        ).fetchall()
    if not rows:
        await ctx.send(f"I have no long-term facts saved for {ctx.author.mention} yet.")
        return
    lines = [f"\U00002022 **{key.replace('_', ' ').title()}:** {value}" for key, value in rows]
    await ctx.send("\U0001F9E0 **Your saved memory**\n" + "\n".join(lines))


@bot.hybrid_group(name="quiz", invoke_without_command=True, description="Play or manage the Islamic quiz game.")
async def quiz_command(ctx, level: str = "advanced", rounds: int = 5):
    """Start a solo quiz: !quiz [beginner|intermediate|advanced] [rounds]."""
    await quiz_game.create_solo(ctx, level.lower(), rounds)


@quiz_command.command(name="solo", description="Start a solo quiz with a level and number of rounds.")
async def quiz_solo(ctx, level: str = "advanced", rounds: int = 5):
    """Start a solo quiz with a level and number of rounds."""
    await quiz_game.create_solo(ctx, level.lower(), rounds)


@quiz_command.command(name="multi", description="Create a multiplayer quiz for others to join.")
async def quiz_multi(ctx, level: str = "advanced", rounds: int = 5):
    """Create a multiplayer lobby: !quiz multi [level] [rounds]."""
    await quiz_game.create_multiplayer(ctx, level.lower(), rounds)


@quiz_command.command(name="join", description="Join the multiplayer quiz in this channel.")
async def quiz_join(ctx):
    await quiz_game.join(ctx)


@quiz_command.command(name="start", description="Start the waiting multiplayer quiz as its host.")
async def quiz_start(ctx):
    await quiz_game.start(ctx)


@quiz_command.command(name="stop", description="Stop the active quiz in this channel.")
async def quiz_stop(ctx):
    await quiz_game.stop(ctx)


@quiz_command.command(name="leaderboard", description="Show the quiz points leaderboard.")
async def quiz_leaderboard(ctx):
    scores = quiz_game.leaderboard()
    if not scores:
        await ctx.send("No quiz scores have been recorded yet.")
        return
    lines = [f"**{index}.** <@{user_id}> - {points} points | {wins} wins" for index, (user_id, points, wins) in enumerate(scores, 1)]
    await ctx.send("\U0001f3c6 **Islamic Quiz leaderboard**\n" + "\n".join(lines))


@quiz_command.command(name="help", description="Show the Islamic quiz game guide.")
async def quiz_help(ctx):
    """Display the quiz game guide."""
    total = len(quiz_game.questions)
    levels = quiz_game.level_counts()
    embed = discord.Embed(
        title="\U0001f9e0 Islamic Quiz - Game Guide",
        description=(
            f"**{total} source-referenced questions** are currently available - the bank has reached **1,000 questions**. "
            "Every question has four shuffled answers, a 45-second timer, and the source is shown after the round."
        ),
        color=discord.Color.gold(),
    )
    embed.add_field(
        name="\U0001f3af Solo quiz",
        value=(
            "`!quiz` - 5 advanced questions\n"
            "`!quiz intermediate 10` - choose level and rounds\n"
            "`!answer A` - answer the active question"
        ),
        inline=False,
    )
    embed.add_field(
        name="\U0001f3c6 Multiplayer",
        value=(
            "`!quiz multi advanced 10` - create a lobby\n"
            "`!quiz join` - join before it begins\n"
            "`!quiz start` - host starts the game\n"
            "First correct answer gets 1 point. Only the host can use `!quiz stop`."
        ),
        inline=False,
    )
    embed.add_field(
        name="\U0001f4ca Progress",
        value=(
            "`!quiz leaderboard` - top players\n"
            "`!quiz profile` - your points and wins\n"
            "`!quiz categories` - available subjects\n"
            f"Levels: beginner {levels['beginner']} | intermediate {levels['intermediate']} | advanced {levels['advanced']}"
        ),
        inline=False,
    )
    embed.add_field(
        name="\U0001f4da Fairness and sources",
        value=(
            "Answer letters are shuffled every round. The bot reveals the answer and reference after each question. "
            "Games rotate through available subjects so one large category does not dominate. "
            "Questions that represent a specific school or creed should be explicitly labelled."
        ),
        inline=False,
    )
    embed.set_footer(text="Use !quiz help whenever you need this panel again.")
    await ctx.send(embed=embed)


@quiz_command.command(name="profile", description="Show your or another member's quiz profile.")
async def quiz_profile(ctx, member: discord.Member | None = None):
    """Show quiz points and wins for yourself or a server member."""
    member = member or ctx.author
    points, wins = quiz_game.player_stats(member.id)
    await ctx.send(f"\U0001f4c8 **{member.display_name} quiz profile**\nPoints: **{points}** | Quiz wins: **{wins}**")


@quiz_command.command(name="categories", description="Show question counts by subject.")
async def quiz_categories(ctx):
    """Show question counts by subject."""
    counts = quiz_game.category_counts()
    lines = [f"**{category}:** {count}" for category, count in counts.items()]
    await ctx.send("\U0001f4da **Question categories**\n" + "\n".join(lines))


@quiz_command.command(name="reload", description="Admin: reload the quiz question files.")
@commands.has_guild_permissions(manage_guild=True)
async def quiz_reload(ctx):
    count = quiz_game.reload_questions()
    await ctx.send(f"Reloaded {count} valid quiz questions.")


@bot.hybrid_command(name="answer", description="Answer the current quiz question using A, B, C, or D.")
async def quiz_answer(ctx, choice: str):
    await quiz_game.answer(ctx, choice)


@bot.event
async def on_message(message):
    if message.author.bot:
        if bot.user and message.author.id == bot.user.id:
            try:
                await message.add_reaction(BOT_REACTION)
            except (discord.Forbidden, discord.HTTPException) as error:
                print(f"Could not add the bot's self-reaction: {error}")
        return
    if await answer_voice_message(message):
        if message.author.id in autoreact_users:
            try:
                await message.add_reaction("\U0001F480")
            except (discord.Forbidden, discord.HTTPException) as error:
                print(f"Could not add auto-reaction: {error}")
        await bot.process_commands(message)
        return
    is_pinged = bool(bot.user and bot.user in message.mentions)
    if is_pinged or await is_reply_to_bot(message):
        if message.author.id in autoreact_users and not message.content.strip().lower().startswith("!autoreactoff"):
            try:
                await message.add_reaction("\U0001F480")
            except (discord.Forbidden, discord.HTTPException) as error:
                print(f"Could not add auto-reaction: {error}")
        # Process commands first — if the message contains a !command plus a mention,
        # the command takes priority and the AI reply is skipped.
        if message.content.strip().startswith("!"):
            await bot.process_commands(message)
            return
        await answer_with_ai(message)
        return
    if message.author.id in autoreact_users and not message.content.strip().lower().startswith("!autoreactoff"):
        try:
            await message.add_reaction("\U0001F480")
        except (discord.Forbidden, discord.HTTPException) as error:
            print(f"Could not add auto-reaction: {error}")
    await bot.process_commands(message)


_token = os.environ.get("DISCORD_TOKEN", "")
if not _token:
    raise RuntimeError("DISCORD_TOKEN not set. Add it to your .env file.")
bot.run(_token)
