"""MESBAHI core — real-time istighfar counter with explainable AI.

Pipeline:
    MediaPipe landmarks  ->  per-finger geometric features
        ->  tiny GradientBoosting model predicts how many fingers are open (0-5)
        ->  SHAP explains *why* the model counted that number — live, per frame
        ->  an istighfar is counted each time a NEW finger opens (counted by hand).

The XAI layer answers: "why 3?" — because the index, middle & ring fingers are
extended (their SHAP contributions dominate), thumb & pinky are folded.
"""

from __future__ import annotations

import numpy as np
import shap
from sklearn.ensemble import GradientBoostingRegressor

# MediaPipe hand landmark indices
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP = 0, 4, 8, 12
FINGERS = [  # (tip, pip, mcp) per finger: thumb->index->middle->ring->pinky
    (THUMB_TIP, 3, 2),
    (INDEX_TIP, 6, 5),
    (MIDDLE_TIP, 10, 9),
    (16, 14, 13),
    (20, 18, 17),
]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def finger_features(lm: np.ndarray) -> np.ndarray:
    """Map 21 landmarks (x,y,z normalized) to 5 scale-invariant 'openness' scores.

    A finger is increasingly 'open' as its tip-to-mcp length exceeds its
    pip-to-mcp length (non-thumb) / thumb separated from the index base.
    """
    pts = lm.astype(float)
    palm = float(np.linalg.norm(pts[0] - pts[9])) + 1e-6  # wrist->middle mcp
    feats = []
    for tip, pip, mcp in FINGERS:
        d_tip = np.linalg.norm(pts[tip] - pts[mcp])
        d_pip = np.linalg.norm(pts[pip] - pts[mcp])
        feats.append(float(np.clip(d_tip / (d_pip + 1e-6) - 1.0, 0.0, 1.5)))
    # give thumb extra context: separation from index base, normalized
    feats[0] = float(np.clip(np.linalg.norm(pts[THUMB_TIP] - pts[5]) / palm, 0.0, 1.5))
    return np.asarray(feats, dtype=np.float64)


def build_count_model():
    """A tiny GB regressor trained on realistic feature combos (0-5 open fingers).

    Training samples are drawn *through the real feature extractor* on synthetic
    hands, so live inference sees the same score distributions the model learned
    on (keeps the project self-contained: no dataset to ship).
    """
    rng = np.random.default_rng(42)
    X, y = [], []
    for mask in range(32):                      # all 0..5 finger combinations
        n_open = bin(mask).count("1")
        for s in range(80):
            feats = finger_features(synthetic_hand(mask, seed=s))
            X.append(feats + rng.normal(0, 0.04, 5))
            y.append(n_open)
    X, y = np.asarray(X), np.asarray(y)
    model = GradientBoostingRegressor(
        n_estimators=120, max_depth=3, learning_rate=0.08, random_state=42
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return model, explainer


def count_fingers(model, feats: np.ndarray) -> int:
    """Round the model's continuous count to the nearest natural number."""
    return int(np.clip(round(float(model.predict(feats.reshape(1, -1))[0])), 0, 5))


def explain_count(explainer, feats: np.ndarray) -> dict:
    """Per-finger SHAP contributions for the current frame (why 'n'?)."""
    sv = explainer.shap_values(feats.reshape(1, -1))[0]
    return {name: float(v) for name, v in zip(FINGER_NAMES, sv)}


def count_with_stable_finger(model, feats: np.ndarray, prev: float) -> tuple[int, float]:
    """Return the finger count with a small hysteresis debounce for smoothness."""
    raw = float(model.predict(feats.reshape(1, -1))[0])
    n = round(raw)
    if abs(raw - n) < 0.35:               # confident, keep
        return max(0, min(5, n)), raw
    return max(0, min(5, int(round(prev)))), prev


# ----------------------------------------------------------------- synthesis
def synthetic_hand(open_mask: int, seed: int = 1) -> np.ndarray:
    """Build a plausible 21-landmark hand with the given fingers open (tests/sim).

    Geometry mirrors reality: mcp base at y ~ 0.5; an open finger's tip extends
    far along the finger axis (tip->mcp >> pip->mcp), a folded tip curls back to
    rest near the palm (tip->mcp ~= pip->mcp).
    """
    rng = np.random.default_rng(seed)
    lm = rng.normal(0.5, 0.04, (21, 3))  # base noise
    xs = [0.62, 0.55, 0.50, 0.45, 0.40]  # thumb..pinky lateral position
    for i, (tip, pip, mcp) in enumerate(FINGERS):
        x = xs[i]
        lm[mcp] = [x, 0.50, 0]
        lm[pip] = [x, 0.58, 0]
        if i == 0:                       # thumb abducts sideways when open
            lm[tip] = [0.78 if open_mask & 1 else 0.63, 0.53, 0]
        else:
            open_ = bool(open_mask >> i & 1)
            lm[tip] = [x, 0.74 if open_ else 0.55, 0]
    lm[0] = [0.50, 0.30, 0]              # wrist below palm centre
    lm[WRIST] = [0.50, 0.30, 0]
    return lm