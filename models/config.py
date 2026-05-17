"""Centralized, environment-driven application configuration.

All settings are loaded from environment variables (or a local `.env` file)
via Pydantic Settings. Import the singleton ``settings`` from this module
anywhere in the codebase rather than reading ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- App ---
    app_env: Literal["local", "staging", "production"] = Field(
        default="local", alias="APP_ENV"
    )
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- MongoDB ---
    mongo_uri: str = Field(default="mongodb://localhost:27017", alias="MONGO_URI")
    mongo_db: str = Field(default="ai_analyst", alias="MONGO_DB")
    # Optional Atlas-style credentials (plain text; RFC 3986 escaping applied in
    # ``effective_mongo_uri``). Use when ``MONGO_URI`` would need manual
    # ``urllib.parse.quote_plus`` on username/password (e.g. ``@`` in password).
    mongo_user: str | None = Field(default=None, alias="MONGO_USER")
    mongo_password: str | None = Field(default=None, alias="MONGO_PASSWORD")
    mongo_srv_host: str | None = Field(default=None, alias="MONGO_SRV_HOST")
    # PEM bundle for Atlas TLS (fixes macOS "unable to get local issuer certificate").
    # If unset, the app uses certifi's bundle when ``certifi`` is installed.
    mongo_tls_ca_file: str | None = Field(default=None, alias="MONGO_TLS_CA_FILE")

    # --- MySQL ---
    mysql_host: str = Field(default="localhost", alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    mysql_user: str = Field(default="root", alias="MYSQL_USER")
    mysql_password: str = Field(default="", alias="MYSQL_PASSWORD")
    mysql_db: str = Field(default="ai_analyst", alias="MYSQL_DB")

    # --- Vector DB ---
    faiss_index_path: str = Field(
        default="./data/faiss_index/index.faiss", alias="FAISS_INDEX_PATH"
    )
    faiss_meta_path: str = Field(
        default="./data/faiss_index/meta.json", alias="FAISS_META_PATH"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")

    # --- Reranker (Build #4) ---
    # When ``reranker_enabled`` is True, ``vector_service.search`` fetches
    # ``reranker_fan_out × top_k`` from FAISS and reranks with a cross-encoder
    # to ``top_k``. Defaults are off so existing behaviour is preserved.
    reranker_enabled: bool = Field(default=False, alias="RERANKER_ENABLED")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    reranker_fan_out: int = Field(
        default=4, alias="RERANKER_FAN_OUT", ge=1, le=20,
    )

    # --- Claude ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    claude_model: str = Field(
        default="claude-3-5-sonnet-20240620", alias="CLAUDE_MODEL"
    )
    claude_max_tokens: int = Field(default=1500, alias="CLAUDE_MAX_TOKENS")
    claude_temperature: float = Field(default=0.2, alias="CLAUDE_TEMPERATURE")

    # --- Chart decision strategy ---
    # ``rule``  → deterministic ``select_chart_type`` based on data shape.
    # ``llm``   → ``services.chart_tool.LLMChartPlanner`` decides via tool use.
    # The LLM path adds one Claude call per analytical answer; the trace
    # context records its tokens + cost so you can A/B in evals.
    chart_decider: Literal["rule", "llm"] = Field(
        default="rule", alias="CHART_DECIDER"
    )

    # --- Insight critic (Build #7) ---
    # When ``critic_enabled`` is True, ``AgentService.generate_insight``
    # runs a verification critic over the draft narrative before
    # shipping it. The critic flags claims that are not supported by
    # STRUCTURED DATA / TRUST PANEL / QUALITY WARNINGS. If
    # ``critic_revise_on_flag`` is True (default), one revise round
    # regenerates the draft with the critic's findings appended; if
    # False the agent ships the draft with a verification banner
    # prepended (annotate-only / shadow mode for measuring
    # hallucination rate without doubling the revision cost). The
    # critic call itself is small (~$0.003); the revise call costs as
    # much as a normal generator call. Off by default so existing
    # token budgets are preserved.
    critic_enabled: bool = Field(default=False, alias="CRITIC_ENABLED")
    critic_revise_on_flag: bool = Field(
        default=True, alias="CRITIC_REVISE_ON_FLAG"
    )

    # --- Distilled intent classifier (Build #8) ---
    # When ``intent_distiller_enabled`` is True,
    # ``services.question_intent.classify`` consults a small TF-IDF +
    # LogisticRegression student model trained from the rule
    # classifier. The student NEVER overrides the rule unless three
    # conditions hold simultaneously:
    #   1. the model artifact exists and loaded successfully
    #   2. the student's confidence >= ``intent_distiller_confidence_threshold``
    #   3. the student disagrees with the rule
    # Otherwise the rule wins. The student is therefore a strict
    # *upgrade* path: it can only fix the rule's misses, never make
    # them worse silently.
    #
    # ``intent_distiller_model_path`` lets eval/runtime point at an
    # alternate artifact (e.g. an experiment) without retraining the
    # default. Empty string means "use the default
    # data/intent_distiller.joblib".
    intent_distiller_enabled: bool = Field(
        default=False, alias="INTENT_DISTILLER_ENABLED"
    )
    intent_distiller_confidence_threshold: float = Field(
        default=0.75,
        alias="INTENT_DISTILLER_CONFIDENCE_THRESHOLD",
        ge=0.0,
        le=1.0,
    )
    intent_distiller_model_path: str = Field(
        default="", alias="INTENT_DISTILLER_MODEL_PATH"
    )

    # --- Semantic OOS Detection (Build #9) ---
    # When ``semantic_oos_enabled`` is True, ``question_intent.classify``
    # uses sentence embeddings to detect out-of-scope questions that don't
    # match static keyword lists. The detector compares questions against
    # in-scope and OOS exemplars using cosine similarity. Exemplars can be
    # updated at runtime without retraining. Off by default so existing
    # behavior is preserved.
    semantic_oos_enabled: bool = Field(
        default=False, alias="SEMANTIC_OOS_ENABLED"
    )
    semantic_oos_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="SEMANTIC_OOS_MODEL",
    )
    semantic_oos_threshold: float = Field(
        default=0.65,
        alias="SEMANTIC_OOS_THRESHOLD",
        ge=0.0,
        le=1.0,
    )

    # --- Prompt Injection Defense (Build #10) ---
    # When ``injection_detection_enabled`` is True, all user questions are
    # checked for prompt injection attacks before processing. Uses multi-strategy
    # detection: semantic similarity to known injection patterns, regex patterns,
    # and structural heuristics. Injection patterns can be updated at runtime
    # without redeployment. Off by default for backward compatibility.
    injection_detection_enabled: bool = Field(
        default=False, alias="INJECTION_DETECTION_ENABLED"
    )
    injection_semantic_threshold: float = Field(
        default=0.70,
        alias="INJECTION_SEMANTIC_THRESHOLD",
        ge=0.0,
        le=1.0,
    )

    # --- MongoDB Index Advisor (Build #10) ---
    # When ``auto_create_indexes`` is True, the index advisor automatically
    # creates indexes for frequently slow queries (>100ms). Indexes are created
    # in background mode to avoid blocking writes. When ``auto_drop_unused_indexes``
    # is True, indexes that haven't been used in 7+ days are automatically dropped.
    # Both off by default for safety.
    auto_create_indexes: bool = Field(
        default=False, alias="AUTO_CREATE_INDEXES"
    )
    auto_drop_unused_indexes: bool = Field(
        default=False, alias="AUTO_DROP_UNUSED_INDEXES"
    )

    # --- Hybrid Session Store (Build #10) ---
    # When ``redis_url`` is set, sessions use multi-tier storage:
    # L1 (in-memory) → L2 (Redis) → L3 (MongoDB). Falls back gracefully
    # if Redis is unavailable. Leave empty to use in-memory only.
    redis_url: str = Field(default="", alias="REDIS_URL")
    session_backend: Literal["memory", "hybrid"] = Field(
        default="memory", alias="SESSION_BACKEND"
    )

    # --- Adaptive Query Rewriter (Build #10) ---
    # When ``query_rewriter_enabled`` is True, user queries are checked for
    # typos and misspellings using fuzzy matching and semantic similarity.
    # The rewriter learns from user corrections and updates vocabulary
    # dynamically from actual data. Off by default.
    query_rewriter_enabled: bool = Field(
        default=False, alias="QUERY_REWRITER_ENABLED"
    )
    query_rewriter_fuzzy_threshold: float = Field(
        default=0.85,
        alias="QUERY_REWRITER_FUZZY_THRESHOLD",
        ge=0.0,
        le=1.0,
    )
    query_rewriter_semantic_threshold: float = Field(
        default=0.75,
        alias="QUERY_REWRITER_SEMANTIC_THRESHOLD",
        ge=0.0,
        le=1.0,
    )

    # --- Adaptive Request Queue (Build #10) ---
    # Async request queue with dynamic prioritization and auto-scaling.
    # Premium users and simple queries get higher priority. Workers scale
    # up/down based on queue depth. Off by default (synchronous mode).
    queue_enabled: bool = Field(default=False, alias="QUEUE_ENABLED")
    queue_initial_workers: int = Field(
        default=2, alias="QUEUE_INITIAL_WORKERS", ge=1, le=50
    )
    queue_max_workers: int = Field(
        default=10, alias="QUEUE_MAX_WORKERS", ge=1, le=100
    )
    queue_max_size: int = Field(
        default=1000, alias="QUEUE_MAX_SIZE", ge=10, le=10000
    )

    # --- Follow-up LLM co-reference resolver (Layer 2) ---
    # Deterministic structural follow-up detection
    # (:func:`services.session_patch.analyze_followup`) handles every
    # refinement shape we ship today and stays sub-millisecond. When
    # this flag is True, ambiguous cases (low-confidence structural
    # verdicts) escalate to an LLM co-reference resolver that takes
    # the prior plan plus the new utterance and returns a patched
    # ``AggregationSpec``. Off by default — the structural patcher
    # is sufficient for the cases we have evals for, and turning
    # this on adds per-turn latency + token cost.
    followup_llm_enabled: bool = Field(
        default=False, alias="FOLLOWUP_LLM_ENABLED"
    )

    # --- Router LLM fallback (Build #6) ---
    # When ``router_llm_fallback_enabled`` is True, ``RoutingService.decide``
    # consults an LLM second-opinion (one Claude tool-use call) ONLY when
    # ``services.routing_service._uncertainty_flags`` reports at least one
    # concern about the rule-based decision. The LLM can patch
    # target / operation / metric / group_by / route, but every patched
    # value is validated against an allowed set (target_candidates and
    # actual columns) — hallucinated values are silently rejected and the
    # rule field stands. Off by default so existing eval baselines and
    # token budgets are preserved.
    router_llm_fallback_enabled: bool = Field(
        default=False, alias="ROUTER_LLM_FALLBACK_ENABLED"
    )

    # --- Session summarisation (Build #5) ---
    # When ``session_summary_enabled`` is True the orchestrator threads
    # a running multi-turn summary into the analyst prompt. Compression
    # fires when the verbatim turn buffer reaches
    # ``session_summary_trigger_at`` turns, after which only the most
    # recent ``session_summary_keep_last`` turns survive verbatim. The
    # rest are folded into ``SessionRecord.summary`` by a single Claude
    # call (see :mod:`services.session_summary`). Off by default so
    # legacy single-turn callers see no behaviour change and pay no
    # extra tokens.
    session_summary_enabled: bool = Field(
        default=False, alias="SESSION_SUMMARY_ENABLED"
    )
    session_summary_trigger_at: int = Field(
        default=6, alias="SESSION_SUMMARY_TRIGGER_AT", ge=2, le=50,
    )
    session_summary_keep_last: int = Field(
        default=2, alias="SESSION_SUMMARY_KEEP_LAST", ge=0, le=20,
    )

    # --- Storage ---
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")
    chart_dir: str = Field(default="./data/charts", alias="CHART_DIR")

    # --- Limits ---
    vector_top_k: int = Field(default=5, alias="VECTOR_TOP_K")
    max_rows_for_llm: int = Field(default=50, alias="MAX_ROWS_FOR_LLM")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def _normalize_optional_mongo_strings(self):
        for name in ("mongo_user", "mongo_password", "mongo_srv_host", "mongo_tls_ca_file"):
            val = getattr(self, name)
            if val is not None and not str(val).strip():
                setattr(self, name, None)
        return self

    # ---------------------------------------------------------------------
    # Derived helpers
    # ---------------------------------------------------------------------
    @property
    def effective_mongo_uri(self) -> str:  # pylint: disable=no-member
        """Connection string for PyMongo.

        If ``MONGO_SRV_HOST`` and ``MONGO_USER`` are set, builds a
        ``mongodb+srv://`` URI with ``quote_plus`` on user and password so
        special characters are valid per RFC 3986. Otherwise returns
        ``mongo_uri`` unchanged (credentials in ``MONGO_URI`` must already be
        escaped).

        ``# pylint: disable=no-member`` suppresses a known Pylint false
        positive: Pylint infers Pydantic field attributes as ``FieldInfo``
        (the return type of ``Field()``) rather than the real annotated
        type (``str | None`` here), so any method call like ``.strip()``
        looks unsupported. Pydantic resolves the annotation at runtime,
        so the calls are correct.
        """
        host = (self.mongo_srv_host or "").strip()
        if not host or self.mongo_user is None:
            return self.mongo_uri
        pwd = self.mongo_password if self.mongo_password is not None else ""
        u = quote_plus(self.mongo_user)
        p = quote_plus(pwd)
        base = (
            host.removeprefix("mongodb+srv://")
            .removeprefix("mongodb://")
            .strip("/")
            .split("/")[0]
            .split("?")[0]
        )
        db = self.mongo_db
        return (
            f"mongodb+srv://{u}:{p}@{base}/{db}"
            f"?retryWrites=true&w=majority"
        )

    @property
    def mysql_url(self) -> str:
        """SQLAlchemy connection URL for MySQL."""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}"
            f"?charset=utf8mb4"
        )

    @field_validator("upload_dir", "chart_dir")
    @classmethod
    def _ensure_dir(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("faiss_index_path", "faiss_meta_path")
    @classmethod
    def _ensure_parent(cls, v: str) -> str:
        Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v


@lru_cache
def get_settings() -> Settings:
    """Return cached ``Settings`` (reads ``.env`` on first call per process).

    After changing environment variables, call ``get_settings.cache_clear()``
    then use ``get_settings()`` again — or restart the process. The module
    attribute ``settings`` delegates here, so it always sees the current cache.
    """
    return Settings()


class _SettingsProxy:
    """Delegate to ``get_settings()`` so ``get_settings.cache_clear()`` works."""

    def __getattr__(self, name: str):
        return getattr(get_settings(), name)

    def __repr__(self) -> str:
        return f"SettingsProxy({get_settings()!r})"


settings: Settings = _SettingsProxy()  # type: ignore  # pyright: ignore[reportAssignmentType]
