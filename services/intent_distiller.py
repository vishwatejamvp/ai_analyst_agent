"""Distilled intent classifier — small student trained from the rule teacher (Build #8).

Why this module exists
----------------------
The rule-based :func:`services.question_intent.classify` is fast,
dependency-free, and right ~85% of the time. Where it fails is the
long tail of paraphrases that don't match its keyword tables: "what's
up with hajj data" doesn't say "what data" or "list datasets" so the
rule path treats it as ANALYTICAL even though the user is exploring.

The standard fix is **distillation**: train a small classifier whose
inputs are the same questions and whose labels are produced by the
teacher (the rule classifier) plus a small set of human-curated
edge cases. The student learns the teacher's *behaviour*, then
generalises beyond it via the embedding space.

Architecture in one paragraph
-----------------------------
The student is a scikit-learn pipeline: a ``TfidfVectorizer`` (1-2
grams, lowercased, no stopword removal — short questions need every
token) feeding a ``LogisticRegression`` with ``class_weight="balanced"``
so the under-represented intents (DISCOVERY, OUT_OF_SCOPE) don't get
swamped by the over-represented ANALYTICAL. The whole pipeline
serialises to a single ``.joblib`` file (~50 KB) and runs inference
in microseconds.

Why TF-IDF + LogReg, not a transformer?
---------------------------------------
For a 4-class single-label task with ~60 training examples:

* TF-IDF + LogReg trains in <200 ms on CPU, no GPU needed.
* Inference is sub-millisecond.
* The artifact is small enough to commit to git (or ship in the
  Docker image).
* Zero new heavy dependencies — sklearn was already installed for
  numpy, joblib for the FAISS index.
* The model is interpretable: ``predict_proba`` returns a 4-vector
  the orchestrator can confidence-gate on.

When you eventually have thousands of labels (collected from
production via :func:`collect_label`) you can swap in
``sentence-transformers + LogReg`` or fine-tune a small encoder. The
*interface* on this class will not change — only ``_load`` and
``predict`` get smarter.

Confidence-gated fallback
-------------------------
The distiller never silently overrides the rule classifier. The
caller (``services.question_intent.classify`` when
``settings.intent_distiller_enabled`` is True) consults the student
and applies a simple policy:

    if student is unavailable      → use the rule
    if student.confidence < threshold → use the rule
    if student.intent == rule.intent → use the rule (agreement)
    else                           → trust the student

That last case — high-confidence disagreement — is the only place a
distilled prediction overrides the deterministic teacher. Everything
else is "the rule was already right; the student confirms".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.question_intent import QuestionIntent
from models.config import settings
from utils.logger import logger

# ---------------------------------------------------------------------------
# Default model location
# ---------------------------------------------------------------------------
# We default the model artifact to ``data/intent_distiller.joblib``. The
# training script writes there; the loader reads from there. Settings
# can override the path for experiments.
DEFAULT_MODEL_PATH = Path("data") / "intent_distiller.joblib"


@dataclass
class DistilledIntent:
    """One distilled prediction with its calibrated confidence.

    ``confidence`` is the probability the LogReg head assigned to the
    chosen class — i.e. ``max(predict_proba(question))``. It is
    NOT a calibrated probability of correctness, but it correlates
    well enough on this task to use as a gating threshold.

    ``probabilities`` is the full 4-vector keyed by intent name so
    the trace / eval can see how confident the model was about
    runners-up — useful when triaging "why did this question flip
    classes between releases?".
    """

    intent: QuestionIntent
    confidence: float
    probabilities: dict[str, float]
    model_version: str = "tfidf-logreg-v1"


class IntentDistiller:
    """TF-IDF + LogReg student model for intent classification.

    Lazy-loads the joblib artifact on first ``predict`` so import-time
    cost is zero when the feature is disabled. If the artifact is
    missing or fails to load, predictions return ``None`` and the
    caller falls back to the rule classifier.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        # We resolve at predict-time (not __init__) so settings overrides
        # take effect for in-process eval flag flips.
        self._explicit_path = Path(model_path) if model_path else None
        self._pipeline = None
        self._classes_: list[str] | None = None
        self._load_attempted = False
        self._load_failed_with: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, question: str) -> DistilledIntent | None:
        """Return a :class:`DistilledIntent` or ``None`` if unavailable.

        ``None`` means "the student couldn't run" (no artifact, load
        error, empty input). Callers MUST treat that as "fall back to
        the rule classifier" — never as "no opinion".
        """
        q = (question or "").strip()
        if not q:
            return None
        pipeline = self._get_pipeline()
        if pipeline is None:
            return None
        try:
            probs = pipeline.predict_proba([q])[0]
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            logger.warning(
                f"Intent distiller predict failed; caller will fall "
                f"back to rule classifier: {type(exc).__name__}: {exc}"
            )
            return None

        classes = self._classes_ or []
        # ``classes_`` from sklearn is in the same index order as
        # ``probs``, so we pair them directly. We deliberately avoid
        # ``np.argmax`` to keep the dependency surface tiny.
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        best_label = str(classes[best_idx])
        try:
            intent = QuestionIntent(best_label.lower())
        except ValueError:
            # Model was trained against a class label we don't know
            # about — treat as unavailable and let rules win.
            logger.warning(
                f"Distiller produced unknown class '{best_label}'; "
                f"falling back to rules."
            )
            return None
        return DistilledIntent(
            intent=intent,
            confidence=float(probs[best_idx]),
            probabilities={
                str(c).lower(): float(p) for c, p in zip(classes, probs)
            },
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if self._load_attempted:
            return None
        self._load_attempted = True
        path = self._resolve_model_path()
        if not path.exists():
            self._load_failed_with = (
                f"model artifact not found at {path} — train one with "
                f"`python -m scripts.train_intent_distiller`"
            )
            logger.info(
                f"Intent distiller artifact missing ({path}); "
                f"distillation disabled until trained."
            )
            return None
        try:
            import joblib  # heavy-ish: load only when actually needed

            artifact = joblib.load(path)
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            self._load_failed_with = f"{type(exc).__name__}: {exc}"
            logger.warning(
                f"Intent distiller load failed at {path}; falling back to "
                f"rules: {self._load_failed_with}"
            )
            return None

        # The artifact is a small dict containing the sklearn pipeline
        # plus its class labels (so we don't have to crack the pipeline
        # internals at predict time).
        self._pipeline = artifact.get("pipeline")
        self._classes_ = list(artifact.get("classes") or [])
        if self._pipeline is None or not self._classes_:
            self._load_failed_with = "artifact missing pipeline/classes"
            logger.warning(
                f"Intent distiller artifact at {path} is malformed; "
                f"falling back to rules."
            )
            self._pipeline = None
            return None
        logger.info(
            f"Intent distiller loaded from {path} "
            f"(classes={self._classes_})"
        )
        return self._pipeline

    def _resolve_model_path(self) -> Path:
        if self._explicit_path is not None:
            return self._explicit_path
        configured = getattr(settings, "intent_distiller_model_path", "") or ""
        if configured:
            return Path(configured)
        return DEFAULT_MODEL_PATH

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """For debugging / health endpoints."""
        return {
            "loaded": self._pipeline is not None,
            "load_attempted": self._load_attempted,
            "model_path": str(self._resolve_model_path()),
            "classes": list(self._classes_ or []),
            "load_error": self._load_failed_with,
        }


# ---------------------------------------------------------------------------
# Label collection (for ongoing distillation)
# ---------------------------------------------------------------------------
# In production, every classification is a free training label — the
# rule classifier acts as the teacher. Calling ``collect_label`` from
# the orchestrator (or a sampling middleware) gives the next training
# run a much larger dataset than the bootstrap file. We append-only
# to JSONL so writes are atomic per line and no DB is needed.
def collect_label(
    question: str,
    intent: QuestionIntent,
    *,
    source: str = "auto",
    out_path: str | Path = "data/intent_labels.jsonl",
) -> None:
    """Append a labeled question to the training dataset.

    Safe to call from hot paths — best-effort, never raises.
    """
    try:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "question": question,
                        "intent": intent.value.upper(),
                        "source": source,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        logger.debug(f"Label collection failed (non-fatal): {exc}")


# Module-level singleton — lazy-loaded on first predict.
intent_distiller = IntentDistiller()
