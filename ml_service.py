"""
ml_service.py — Pennywise ML Service (Production)

Loads a pre-trained pipeline from model.pkl, or separately loads
vectorizer.pkl + model.pkl if the model is not a full pipeline.

Falls back to a lightweight rule-based categoriser if no model files
are present (so the app boots cleanly on first deploy before artefacts
are uploaded).

Expected artefacts (either of these layouts work):
  Layout A — Single pipeline file
      model.pkl  : sklearn Pipeline(TfidfVectorizer, MultinomialNB)
                   exposes .predict() and .predict_proba()

  Layout B — Separate vectoriser + classifier
      vectorizer.pkl : sklearn TfidfVectorizer (fitted)
      model.pkl      : sklearn MultinomialNB (fitted)
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (relative to this file so they work both locally and on Render)
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH      = os.path.join(_BASE, "model.pkl")
_VECTORIZER_PATH = os.path.join(_BASE, "vectorizer.pkl")

# ---------------------------------------------------------------------------
# Fallback rule-based categoriser
# ---------------------------------------------------------------------------
_RULES: list[tuple[list[str], str]] = [
    (["uber", "ola", "taxi", "bus", "metro", "fuel", "petrol", "auto", "rapido", "namma"], "Transport"),
    (["zomato", "swiggy", "restaurant", "cafe", "lunch", "dinner", "food", "pizza", "burger", "blinkit", "chai"], "Food"),
    (["amazon", "flipkart", "shopping", "clothes", "shoes", "mall", "myntra", "ajio", "meesho"], "Shopping"),
    (["netflix", "spotify", "movie", "game", "entertainment", "theatre", "prime", "hotstar", "youtube"], "Entertainment"),
    (["electricity", "water", "rent", "broadband", "wifi", "mobile", "recharge", "gas", "jio", "airtel"], "Utilities"),
    (["doctor", "medicine", "pharmacy", "hospital", "health", "apollo", "medplus"], "Healthcare"),
    (["course", "book", "udemy", "education", "school", "college", "tuition", "coursera", "unacademy"], "Education"),
]
_DEFAULT_CATEGORY = "Miscellaneous"


def _rule_predict(text: str) -> Tuple[str, float]:
    lower = text.lower()
    for keywords, category in _RULES:
        if any(kw in lower for kw in keywords):
            return category, 0.75
    return _DEFAULT_CATEGORY, 0.50


# ---------------------------------------------------------------------------
# Model state  (loaded once, then cached)
# ---------------------------------------------------------------------------
_pipeline    = None   # Layout A: full sklearn Pipeline
_vectorizer  = None   # Layout B: separate TfidfVectorizer
_classifier  = None   # Layout B: separate classifier
_mode        = None   # "pipeline" | "split" | "rules"


def _load_model() -> None:
    global _pipeline, _vectorizer, _classifier, _mode

    if _mode is not None:
        return  # already initialised

    # ── Try Layout A: single pipeline ────────────────────────────────────
    if os.path.exists(_MODEL_PATH):
        try:
            with open(_MODEL_PATH, "rb") as fh:
                obj = pickle.load(fh)

            # Check if it looks like a full pipeline (has predict_proba)
            if hasattr(obj, "predict") and hasattr(obj, "predict_proba"):
                _pipeline = obj
                _mode = "pipeline"
                logger.info("Loaded ML pipeline from %s", _MODEL_PATH)
                return
        except Exception as exc:
            logger.warning("Could not load model.pkl as pipeline: %s", exc)

    # ── Try Layout B: separate vectorizer + classifier ────────────────────
    if os.path.exists(_VECTORIZER_PATH) and os.path.exists(_MODEL_PATH):
        try:
            with open(_VECTORIZER_PATH, "rb") as fh:
                _vectorizer = pickle.load(fh)
            with open(_MODEL_PATH, "rb") as fh:
                _classifier = pickle.load(fh)
            _mode = "split"
            logger.info("Loaded split vectorizer + classifier from disk")
            return
        except Exception as exc:
            logger.warning("Could not load split model artefacts: %s", exc)

    logger.warning(
        "No model artefacts found at %s / %s — using rule-based fallback.",
        _MODEL_PATH, _VECTORIZER_PATH,
    )
    _mode = "rules"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_category(title: str) -> Tuple[str, float]:
    """
    Return (category: str, confidence: float ∈ [0, 1]).

    Always runs ML prediction (or rule fallback) before inserting to DB.
    """
    _load_model()

    if _mode == "pipeline":
        try:
            category: str       = _pipeline.predict([title])[0]
            proba               = _pipeline.predict_proba([title])[0]
            confidence: float   = float(max(proba))
            return category, round(confidence, 4)
        except Exception as exc:
            logger.error("Pipeline prediction failed (%s). Falling back to rules.", exc)

    elif _mode == "split":
        try:
            X                   = _vectorizer.transform([title])
            category            = _classifier.predict(X)[0]
            proba               = _classifier.predict_proba(X)[0]
            confidence          = float(max(proba))
            return category, round(confidence, 4)
        except Exception as exc:
            logger.error("Split model prediction failed (%s). Falling back to rules.", exc)

    return _rule_predict(title)
