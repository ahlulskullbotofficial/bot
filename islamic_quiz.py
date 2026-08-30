"""Persistent, source-aware Islamic quiz engine for discord.py."""

from __future__ import annotations

import asyncio
import json
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import discord
from discord.ext import commands


LEVELS = ("beginner", "intermediate", "advanced")


@dataclass(frozen=True)
class Question:
    id: str
    level: str
    category: str
    prompt: str
    choices: tuple[str, str, str, str]
    correct: int
    source: str
    note: str = ""


@dataclass
class Session:
    channel_id: int
    host_id: int
    multiplayer: bool
    level: str
    rounds: int
    questions: list[Question]
    players: set[int] = field(default_factory=set)
    scores: dict[int, int] = field(default_factory=dict)
    current_index: int = 0
    answers: dict[int, int] = field(default_factory=dict)
    current_choices: list[str] = field(default_factory=list)
    current_correct: int = 0
    message_id: int | None = None
    timeout_task: asyncio.Task | None = None


class IslamicQuiz:
    def __init__(self, bank_path: Path, database_path: Path) -> None:
        self.bank_path = bank_path
        self.database_path = database_path
        self.sessions: dict[int, Session] = {}
        self.questions = self._load_questions()
        self._initialise_database()

    def _load_questions(self) -> list[Question]:
        questions: list[Question] = []
        seen_ids: set[str] = set()
        files = [self.bank_path] + sorted(self.bank_path.parent.glob("quiz_questions_batch_*.json"))
        for file in files:
            for item in json.loads(file.read_text(encoding="utf-8"))["questions"]:
                required = {"id", "level", "category", "prompt", "choices", "correct", "source"}
                if not required.issubset(item) or item["id"] in seen_ids:
                    continue
                if item["level"] not in LEVELS or len(item["choices"]) != 4:
                    continue
                if not isinstance(item["correct"], int) or not 0 <= item["correct"] <= 3:
                    continue
                seen_ids.add(item["id"])
                questions.append(Question(
                    id=item["id"], level=item["level"], category=item["category"],
                    prompt=item["prompt"], choices=tuple(item["choices"]),
                    correct=item["correct"], source=item["source"], note=item.get("note", ""),
                ))
        return questions

    def reload_questions(self) -> int:
        self.questions = self._load_questions()
        return len(self.questions)

    def _initialise_database(self) -> None:
        with sqlite3.connect(self.database_path) as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS quiz_scores (
                    user_id INTEGER PRIMARY KEY,
                    points INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )"""
            )

    def _record_points(self, user_id: int, points: int, won: bool = False) -> None:
        with sqlite3.connect(self.database_path) as db:
            db.execute(
                """INSERT INTO quiz_scores(user_id, points, wins, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                     points = points + excluded.points,
                     wins = wins + excluded.wins,
                     updated_at = excluded.updated_at""",
                (user_id, points, int(won), datetime.now(timezone.utc).isoformat()),
            )

    def leaderboard(self, limit: int = 10) -> list[tuple[int, int, int]]:
        with sqlite3.connect(self.database_path) as db:
            return db.execute(
                "SELECT user_id, points, wins FROM quiz_scores "
                "ORDER BY points DESC, wins DESC LIMIT ?", (limit,)
            ).fetchall()

    def player_stats(self, user_id: int) -> tuple[int, int]:
        with sqlite3.connect(self.database_path) as db:
            row = db.execute(
                "SELECT points, wins FROM quiz_scores WHERE user_id = ?", (user_id,)
            ).fetchone()
        return row if row else (0, 0)

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for question in self.questions:
            counts[question.category] = counts.get(question.category, 0) + 1
        return dict(sorted(counts.items()))

    def level_counts(self) -> dict[str, int]:
        return {level: sum(question.level == level for question in self.questions) for level in LEVELS}

    def _choose_questions(self, level: str, rounds: int) -> list[Question]:
        pool = [question for question in self.questions if question.level == level]
        if len(pool) < rounds:
            pool = self.questions
        if not pool:
            return []
        # A large subject should not dominate a short game. Select one random
        # item from each available category before returning to a category.
        by_category: dict[str, list[Question]] = {}
        for question in pool:
            by_category.setdefault(question.category, []).append(question)
        for category_pool in by_category.values():
            random.shuffle(category_pool)
        categories = list(by_category)
        random.shuffle(categories)
        target = min(rounds, len(pool))
        selected: list[Question] = []
        while len(selected) < target:
            added = False
            for category in categories:
                if by_category[category]:
                    selected.append(by_category[category].pop())
                    added = True
                    if len(selected) == target:
                        break
            if not added:
                break
        random.shuffle(selected)
        return selected

    async def create_solo(self, ctx: commands.Context, level: str, rounds: int) -> None:
        await self._create_session(ctx, level, rounds, multiplayer=False)

    async def create_multiplayer(self, ctx: commands.Context, level: str, rounds: int) -> None:
        await self._create_session(ctx, level, rounds, multiplayer=True)

    async def _create_session(self, ctx: commands.Context, level: str, rounds: int, multiplayer: bool) -> None:
        if ctx.channel.id in self.sessions:
            await ctx.send("There is already a quiz running in this channel. Use `!quiz stop` first.")
            return
        if level not in LEVELS:
            await ctx.send("Level must be `beginner`, `intermediate`, or `advanced`.")
            return
        rounds = max(1, min(rounds, 20))
        questions = self._choose_questions(level, rounds)
        if not questions:
            await ctx.send("The quiz bank is empty. Ask an admin to add reviewed questions.")
            return
        session = Session(
            channel_id=ctx.channel.id,
            host_id=ctx.author.id,
            multiplayer=multiplayer,
            level=level,
            rounds=len(questions),
            questions=questions,
            players={ctx.author.id},
            scores={ctx.author.id: 0},
        )
        self.sessions[ctx.channel.id] = session
        if multiplayer:
            await ctx.send(
                f"🏆 **Multiplayer {level.title()} quiz lobby** created by {ctx.author.mention}. "
                "Type `!quiz join`, then the host can use `!quiz start`."
            )
        else:
            await ctx.send(f"🧠 **Solo {level.title()} quiz** — {len(questions)} rounds. Answer with `!answer A`.")
            await self._ask_next(ctx.channel, session)

    async def join(self, ctx: commands.Context) -> None:
        session = self.sessions.get(ctx.channel.id)
        if not session or not session.multiplayer or session.current_index > 0:
            await ctx.send("There is no multiplayer quiz lobby to join here.")
            return
        session.players.add(ctx.author.id)
        session.scores.setdefault(ctx.author.id, 0)
        await ctx.send(f"{ctx.author.mention} joined the quiz lobby.")

    async def start(self, ctx: commands.Context) -> None:
        session = self.sessions.get(ctx.channel.id)
        if not session or not session.multiplayer:
            await ctx.send("There is no multiplayer quiz lobby here.")
            return
        if session.host_id != ctx.author.id:
            await ctx.send("Only the quiz host can start this lobby.")
            return
        if session.current_index > 0:
            await ctx.send("This quiz has already started.")
            return
        await ctx.send(f"Quiz started with **{len(session.players)} player(s)**. First correct answer earns the point.")
        await self._ask_next(ctx.channel, session)

    async def answer(self, ctx: commands.Context, choice: str) -> None:
        session = self.sessions.get(ctx.channel.id)
        if not session or session.current_index == 0:
            await ctx.send("There is no active question here.")
            return
        if ctx.author.id not in session.players:
            await ctx.send("Join the quiz first with `!quiz join`.")
            return
        letter = choice.strip().upper()
        if letter not in "ABCD":
            await ctx.send("Answer with `!answer A`, `!answer B`, `!answer C`, or `!answer D`.")
            return
        if ctx.author.id in session.answers:
            await ctx.send("You already answered this question.")
            return
        selected = "ABCD".index(letter)
        question = session.questions[session.current_index - 1]
        session.answers[ctx.author.id] = selected
        if selected == session.current_correct:
            session.scores[ctx.author.id] += 1
            self._record_points(ctx.author.id, 1)
            await ctx.send(f"✅ {ctx.author.mention} is correct — **+1 point**.")
            await self._reveal_and_continue(ctx.channel, session)
            return
        await ctx.send(f"❌ {ctx.author.mention}, not quite.")
        if not session.multiplayer or session.answers.keys() >= session.players:
            await self._reveal_and_continue(ctx.channel, session)

    async def _ask_next(self, channel: discord.abc.Messageable, session: Session) -> None:
        if session.current_index >= session.rounds:
            await self._finish(channel, session)
            return
        session.answers.clear()
        question = session.questions[session.current_index]
        session.current_index += 1
        session.current_choices = list(question.choices)
        random.shuffle(session.current_choices)
        session.current_correct = session.current_choices.index(question.choices[question.correct])
        options = "\n".join(f"**{letter}.** {choice}" for letter, choice in zip("ABCD", session.current_choices))
        message = await channel.send(
            f"**Question {session.current_index}/{session.rounds} · {question.level.title()} · {question.category}**\n"
            f"{question.prompt}\n\n{options}\n\nReply with `!answer A` within 45 seconds."
        )
        session.message_id = message.id
        session.timeout_task = asyncio.create_task(self._timeout(channel, session, session.current_index))

    async def _timeout(self, channel: discord.abc.Messageable, session: Session, round_number: int) -> None:
        await asyncio.sleep(45)
        if self.sessions.get(session.channel_id) is session and session.current_index == round_number:
            await channel.send("⌛ Time is up.")
            await self._reveal_and_continue(channel, session)

    async def _reveal_and_continue(self, channel: discord.abc.Messageable, session: Session) -> None:
        if session.timeout_task:
            session.timeout_task.cancel()
            session.timeout_task = None
        question = session.questions[session.current_index - 1]
        extra = f"\n*Note: {question.note}*" if question.note else ""
        await channel.send(
            f"**Answer: {chr(65 + session.current_correct)} — {session.current_choices[session.current_correct]}**\n"
            f"Source: {question.source}{extra}"
        )
        await self._ask_next(channel, session)

    async def stop(self, ctx: commands.Context) -> None:
        session = self.sessions.get(ctx.channel.id)
        if not session:
            await ctx.send("There is no quiz running here.")
            return
        if ctx.author.id != session.host_id:
            await ctx.send("Only the quiz host can stop this quiz.")
            return
        if session.timeout_task:
            session.timeout_task.cancel()
        del self.sessions[ctx.channel.id]
        await ctx.send("Quiz stopped.")

    async def _finish(self, channel: discord.abc.Messageable, session: Session) -> None:
        if session.timeout_task:
            session.timeout_task.cancel()
        ranked = sorted(session.scores.items(), key=lambda item: item[1], reverse=True)
        if ranked:
            best = ranked[0][1]
            for user_id, score in ranked:
                if score == best:
                    self._record_points(user_id, 0, won=True)
        results = "\n".join(f"<@{user_id}> — **{score}**" for user_id, score in ranked)
        await channel.send(f"🏁 **Quiz complete**\n{results}")
        self.sessions.pop(session.channel_id, None)
