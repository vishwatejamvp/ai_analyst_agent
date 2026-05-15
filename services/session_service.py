"""Lightweight, in-memory session store for short-form follow-ups.

Two responsibilities live here, in two layers:

1. **Routing memory** (original v1 purpose): when a user says a short
   refinement like ``by region``, ``exclude refunds``, or
   ``vs last year``, the orchestrator treats that as a **patch** on
   the previous question's plan rather than a fresh interpretation.
   We persist only ``last_question`` + ``last_decision`` for that.
   Merge logic itself lives in :mod:`services.session_patch`.

2. **Conversation memory** (Build #5): the multi-turn analytical
   thread itself. We append (question, insight) pairs as ``turns``
   and let :class:`services.session_summary.ConversationSummariser`
   compress old turns into a running ``summary`` once the buffer
   exceeds a configurable threshold. The orchestrator injects
   ``summary + remaining recent turns`` into the next prompt so the
   analyst can say "as you saw earlier, March was an outlier...".

Properties:

* In-process only (no Redis, no DB) — sufficient for v1, swappable
  later.
* TTL-bounded so abandoned sessions don't accumulate.
* Thread-safe under a single coarse lock — fine for single-process
  Uvicorn workers, will need rework if we ever shard the API.

A session id is opaque from the API's perspective; the client supplies
it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from models.enums import QueryRoute
from models.schemas import RoutingDecision
from services.session_summary import Turn

DEFAULT_TTL = timedelta(minutes=30)
MAX_SESSIONS = 5_000


@dataclass
class SessionRecord:
    """Per-session memory used by the follow-up resolver and prompt builder.

    Two routing pointers are tracked deliberately:

    * ``last_decision`` — the most recent routing decision of *any*
      route (analytical, semantic, discovery, out-of-scope). Used
      by ``should_reuse_prior_collection`` and other context-aware
      helpers that care about the immediate prior turn.

    * ``last_analytical_decision`` — the most recent decision that
      was *analytical*. Used by the follow-up resolver so a single
      discovery or semantic turn doesn't blow away the analytical
      context. Real conversations look like
      "trend 2026" → analytical, then "what's hajj umrah?" →
      semantic, then "first half" — the user still means
      "first half of the 2026 trend", not "first half of nothing".

    Tracking both pointers is a small memory cost (two refs per
    session) and the only reliable way to keep multi-turn follow-ups
    working across mixed-intent threads.
    """

    session_id: str
    last_question: str
    last_decision: RoutingDecision
    created_at: datetime
    last_used_at: datetime

    last_analytical_question: str | None = None
    last_analytical_decision: RoutingDecision | None = None

    # ---- Conversation memory (Build #5) -----------------------------
    # ``turns`` is the verbatim buffer; ``summary`` is the compressed
    # tail of older turns the buffer no longer holds. The orchestrator
    # builds the prompt from both. Both are optional so legacy callers
    # that only use the routing-memory APIs are unaffected.
    turns: list[Turn] = field(default_factory=list)
    summary: str | None = None

    def is_expired(self, ttl: timedelta) -> bool:
        return datetime.now(timezone.utc) - self.last_used_at > ttl


class SessionService:
    """Thread-safe in-memory session store."""

    def __init__(self, ttl: timedelta = DEFAULT_TTL, max_sessions: int = MAX_SESSIONS) -> None:
        self.ttl = ttl
        self.max_sessions = max_sessions
        self._records: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str | None) -> SessionRecord | None:
        if not session_id:
            return None
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return None
            if record.is_expired(self.ttl):
                self._records.pop(session_id, None)
                return None
            record.last_used_at = datetime.now(timezone.utc)
            return record

    def put(
        self,
        session_id: str,
        question: str,
        decision: RoutingDecision,
    ) -> None:
        if not session_id:
            return
        now = datetime.now(timezone.utc)
        with self._lock:
            existing = self._records.get(session_id)
            created_at = existing.created_at if existing else now

            # Two-pointer routing memory. ``last_*`` always tracks
            # the immediate prior turn; ``last_analytical_*`` only
            # advances when the new decision is analytical. That way
            # a discovery / semantic / out-of-scope turn doesn't
            # erase the analytical context that follow-ups still
            # depend on.
            is_analytical = decision is not None and (
                decision.route == QueryRoute.ANALYTICAL
            )
            if is_analytical:
                last_analytical_question = question
                last_analytical_decision = decision
            elif existing is not None:
                last_analytical_question = existing.last_analytical_question
                last_analytical_decision = existing.last_analytical_decision
            else:
                last_analytical_question = None
                last_analytical_decision = None

            # Preserve conversation memory across ``put`` calls — only
            # routing memory is being replaced here. Without this, every
            # new question would wipe the turn buffer and the summary,
            # which would defeat the whole point of Build #5.
            self._records[session_id] = SessionRecord(
                session_id=session_id,
                last_question=question,
                last_decision=decision,
                created_at=created_at,
                last_used_at=now,
                last_analytical_question=last_analytical_question,
                last_analytical_decision=last_analytical_decision,
                turns=list(existing.turns) if existing else [],
                summary=existing.summary if existing else None,
            )
            if len(self._records) > self.max_sessions:
                self._evict_oldest()

    def reset(self, session_id: str) -> bool:
        with self._lock:
            return self._records.pop(session_id, None) is not None

    # ------------------------------------------------------------------
    # Conversation memory (Build #5)
    # ------------------------------------------------------------------
    #
    # These three methods are intentionally thin: they own the buffer
    # but NOT the compression policy. The orchestrator (which has the
    # active trace context and the summariser instance) decides when
    # to call :meth:`set_summary` and what to compress into. Keeping
    # policy outside of the store lets us swap summarisers (cheaper
    # model, different prompt) without touching the storage layer.

    def add_turn(
        self,
        session_id: str,
        question: str,
        insight: str,
    ) -> int:
        """Append a turn and return the new buffer size.

        The orchestrator always calls :meth:`put` (routing memory)
        before :meth:`add_turn` (conversation memory) within the same
        request, so the record is guaranteed to exist by the time we
        get here. If it doesn't (e.g. a direct caller skipped ``put``),
        we silently no-op and return 0 — better than fabricating a
        placeholder ``RoutingDecision`` and crashing the next router
        follow-up call.
        """
        if not session_id or not (insight or "").strip():
            return 0
        now = datetime.now(timezone.utc)
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return 0
            record.turns.append(
                Turn(question=question or "", insight=insight, ts=now)
            )
            record.last_used_at = now
            return len(record.turns)

    def get_history(
        self,
        session_id: str | None,
    ) -> tuple[str | None, list[Turn]]:
        """Return ``(summary, turns)`` for prompt construction.

        Returns ``(None, [])`` when the session does not exist or has
        expired. The buffer is shallow-copied so the caller can iterate
        without holding the lock.
        """
        if not session_id:
            return None, []
        with self._lock:
            record = self._records.get(session_id)
            if record is None or record.is_expired(self.ttl):
                if record is not None:
                    self._records.pop(session_id, None)
                return None, []
            record.last_used_at = datetime.now(timezone.utc)
            return record.summary, list(record.turns)

    def set_summary(
        self,
        session_id: str,
        new_summary: str,
        keep_last: int,
    ) -> None:
        """Replace the summary and trim the verbatim buffer.

        Atomic with respect to readers: the buffer never goes through
        an inconsistent state where ``summary`` already drops turns
        that are also still in the buffer with no overlap, or where
        the buffer was already trimmed but the summary was not yet
        stored.
        """
        if not session_id:
            return
        keep_last = max(int(keep_last), 0)
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return
            record.summary = (new_summary or "").strip() or None
            if keep_last == 0:
                record.turns = []
            else:
                record.turns = record.turns[-keep_last:]
            record.last_used_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _evict_oldest(self) -> None:
        # Cheap LRU-ish: drop the entry with the oldest last_used_at.
        if not self._records:
            return
        victim = min(self._records.values(), key=lambda r: r.last_used_at)
        self._records.pop(victim.session_id, None)


session_service = SessionService()
