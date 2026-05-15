"""Train the distilled intent classifier (Build #8).

Reads ``data/intent_labels.jsonl`` (or whatever ``--labels`` points
at), trains a small TF-IDF + LogisticRegression pipeline, and writes
the joblib artifact to ``data/intent_distiller.joblib`` (or
``--out``).

Run from the project root::

    python -m scripts.train_intent_distiller

Common flags::

    --labels    path to labelled JSONL (default data/intent_labels.jsonl)
    --out       output joblib path     (default data/intent_distiller.joblib)
    --test-frac fraction held out for eval (default 0.2)
    --seed      random seed (default 17 — deterministic by default)
    --min-per-class minimum samples per class to keep training (default 3)
    --no-stratify  disable stratified split (use when classes < 2 samples)
    --quiet     suppress per-row diagnostics

The script prints a per-class precision / recall / F1 table and a
confusion matrix so you can see where the student disagrees with the
teacher BEFORE you ship the model.

Why each design choice
----------------------
* **TF-IDF (1-2 grams, lowercase)**: short questions need both
  unigrams ("compare") and bigrams ("vs last") to win against the
  rule's keyword tables. We keep stopwords because tokens like
  "what", "how", "vs", "and" carry intent signal here.
* **LogisticRegression with ``class_weight='balanced'``**: the
  bootstrap dataset is intentionally imbalanced (mostly ANALYTICAL,
  to mirror real traffic). Balancing prevents the model from
  collapsing to "always predict ANALYTICAL".
* **Stratified split**: with ~60 examples and 4 classes we cannot
  afford a random split that leaves a class entirely in the
  training half — stratifying preserves the class ratios in both
  splits.
* **Deterministic seed**: same labels → same model. The artifact
  hash should change ONLY when the underlying labels change.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_labels(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        raise FileNotFoundError(f"labels file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_num} not valid JSON: {exc}"
                ) from exc
            q = (row.get("question") or "").strip()
            intent = (row.get("intent") or "").strip().upper()
            if not q or not intent:
                continue
            rows.append(
                {
                    "question": q,
                    "intent": intent,
                    "source": (row.get("source") or "unknown"),
                }
            )
    return rows


def _dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Last write wins per question.

    Lets you append a corrected label to the JSONL without rewriting
    the file: the latest line for a given question overrides earlier
    ones. ``source`` of the kept row is preserved.
    """
    by_q: dict[str, dict[str, str]] = {}
    for r in rows:
        by_q[r["question"].lower()] = r
    return list(by_q.values())


def _enforce_min_per_class(
    rows: list[dict[str, str]], *, min_per_class: int
) -> list[dict[str, str]]:
    counts = Counter(r["intent"] for r in rows)
    bad = [c for c, n in counts.items() if n < min_per_class]
    if bad:
        print(
            f"WARN  classes with <{min_per_class} samples will be "
            f"dropped from training: {bad}",
            file=sys.stderr,
        )
        rows = [r for r in rows if r["intent"] not in bad]
    return rows


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _make_pipeline() -> Pipeline:
    """Build the TF-IDF → LogReg pipeline.

    Hyperparameters are tuned for tiny (~50-200 row) datasets:

    * ``ngram_range=(1, 2)`` — bigrams catch comparison phrases like
      "vs last" / "compare X" without over-fitting on a small corpus.
    * ``min_df=1`` — every token can matter when you only have ~50
      examples. Re-tune to 2 once you have ~500+.
    * ``sublinear_tf=True`` — log-scaled TF damps the effect of
      common tokens on long questions vs short ones.
    * ``max_iter=2000`` — LogReg on TF-IDF converges fast; the
      generous cap absorbs noisy datasets without breaking under the
      ``ConvergenceWarning`` that an over-tight cap would emit.
    * ``solver='lbfgs'`` — the modern default for multiclass softmax.
      We deliberately moved off ``liblinear`` (which is deprecated for
      multiclass since sklearn 1.7+) so this script doesn't need a
      per-version solver branch.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=17,
                ),
            ),
        ]
    )


def _train(
    rows: list[dict[str, str]],
    *,
    test_frac: float,
    seed: int,
    stratify: bool,
) -> tuple[Pipeline, dict[str, Any]]:
    X = [r["question"] for r in rows]
    y = [r["intent"] for r in rows]

    # Stratification requires at least 2 samples per class. If anyone
    # passes --no-stratify we honour it; otherwise we auto-disable
    # stratification when it would crash.
    if stratify and min(Counter(y).values()) < 2:
        print(
            "WARN  some class has only 1 sample; disabling stratified "
            "split (re-enable once every class has >=2 examples)",
            file=sys.stderr,
        )
        stratify = False

    if test_frac > 0:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=test_frac,
            random_state=seed,
            stratify=y if stratify else None,
        )
    else:
        # No held-out eval — train on everything and report training
        # accuracy. Useful right before shipping the artifact.
        X_tr, X_te, y_tr, y_te = X, X, y, y

    pipe = _make_pipeline()
    pipe.fit(X_tr, y_tr)

    # Evaluate.
    pred = pipe.predict(X_te)
    report = classification_report(
        y_te, pred, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(
        y_te, pred, labels=sorted(set(y))
    ).tolist()

    metrics: dict[str, Any] = {
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "classes": sorted(set(y)),
        "report": report,
        "confusion_matrix": cm,
    }
    return pipe, metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_report(metrics: dict[str, Any], rows: list[dict[str, str]]) -> None:
    print()
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  rows total            : {len(rows)}")
    print(f"  rows train            : {metrics['n_train']}")
    print(f"  rows test             : {metrics['n_test']}")
    print(f"  classes               : {metrics['classes']}")
    print()
    print("  per-class metrics:")
    rep = metrics["report"]
    for cls in metrics["classes"]:
        if cls in rep:
            r = rep[cls]
            print(
                f"    {cls:<14} "
                f"P={r['precision']:.2f}  R={r['recall']:.2f}  "
                f"F1={r['f1-score']:.2f}  "
                f"support={int(r['support'])}"
            )
    macro = rep.get("macro avg", {})
    if macro:
        print(
            f"    {'macro avg':<14} "
            f"P={macro['precision']:.2f}  R={macro['recall']:.2f}  "
            f"F1={macro['f1-score']:.2f}"
        )
    print()
    print("  confusion matrix (rows=true, cols=pred):")
    classes = metrics["classes"]
    print("    " + " ".join(f"{c[:8]:>10}" for c in classes))
    for cls, row in zip(classes, metrics["confusion_matrix"]):
        print(f"    {cls[:8]:>4} " + " ".join(f"{n:>10}" for n in row))
    print()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def _save(pipe: Pipeline, *, out: Path, metrics: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipe,
        "classes": list(pipe.classes_),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "model_version": "tfidf-logreg-v1",
    }
    joblib.dump(artifact, out)
    size_kb = out.stat().st_size / 1024
    print(f"  saved artifact        : {out} ({size_kb:.1f} KB)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the intent distiller (TF-IDF + LogReg). Produces a "
            "joblib artifact consumable by services.intent_distiller."
        )
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/intent_labels.jsonl"),
        help="Input JSONL with {question, intent, source} per line.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/intent_distiller.joblib"),
        help="Output joblib artifact path.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction held out for evaluation (0 = train on all).",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=3,
        help="Drop any class with fewer than this many examples.",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified train/test split.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    rows = _load_labels(args.labels)
    if not args.quiet:
        print(f"loaded {len(rows)} rows from {args.labels}")
    rows = _dedupe(rows)
    if not args.quiet:
        print(f"  after dedupe         : {len(rows)} rows")
    rows = _enforce_min_per_class(rows, min_per_class=args.min_per_class)
    if len(rows) < 8:
        print(
            f"ERROR  only {len(rows)} usable rows — collect more labels "
            f"before training.",
            file=sys.stderr,
        )
        return 2

    pipe, metrics = _train(
        rows,
        test_frac=args.test_frac,
        seed=args.seed,
        stratify=not args.no_stratify,
    )
    _print_report(metrics, rows)
    _save(pipe, out=args.out, metrics=metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
