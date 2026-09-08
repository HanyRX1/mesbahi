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


def finger_features(lm: np.ndarray, aspect: float = 1.0) -> np.ndarray:
    """Map 21 landmarks (x,y,z normalized) to 5 scale-invariant 'openness' scores.

    A finger is increasingly 'open' as its tip-to-mcp length exceeds its
    pip-to-mcp length (non-thumb) / thumb separated from the index base.

    `aspect` = width/height of the camera frame. MediaPipe normalizes x by the
    image width and y by the height independently, so raw coordinates are only
    metric-isotropic once x is corrected by the frame aspect ratio. This keeps
    the features invariant to camera resolution (square 720p == 480p 4:3).
    """
    pts = lm.astype(float).copy()
    pts[:, 0] *= aspect               # make x and y share the same physical scale
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


# ------------------------------------------------------------------- counting
def count_thresholds(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Data-driven per-finger open/close thresholds with a hysteresis band.

    Derived from the exact same synthetic score distribution used to train the
    model, so each finger's thresholds sit between its 'open' and 'closed'
    score clusters (thumb has a higher baseline than the other fingers).
    Returns (open_th, close_th), each shape (5,).
    """
    rng = np.random.default_rng(seed)
    acc_open = np.zeros(5)
    acc_closed = np.zeros(5)
    n_open = np.zeros(5, int)
    n_closed = np.zeros(5, int)
    for mask in range(32):
        open_by_finger = np.array([int(bool(mask >> i & 1)) for i in range(5)])
        for s in range(80):
            feats = finger_features(synthetic_hand(mask, seed=s)) + rng.normal(0, 0.04, 5)
            acc_open += feats * open_by_finger
            acc_closed += feats * (1 - open_by_finger)
            n_open += open_by_finger
            n_closed += (1 - open_by_finger)
    mean_open = acc_open / np.maximum(n_open, 1)
    mean_closed = acc_closed / np.maximum(n_closed, 1)
    span = mean_open - mean_closed
    open_th = mean_closed + 0.55 * span    # finger must clearly rise to open
    close_th = mean_closed + 0.30 * span   # once open, stays open lower — band
    return open_th, close_th


class FingerCounter:
    """Accurate real-time istighfar counter (per-finger state machine).

    Rules that kill the common false-count sources in opencv pipelines:
      * Hysteresis band: a finger opens only past ``open_th`` and only closes
        below ``close_th`` (open_th > close_th), so boundary jitter cannot
        toggle it.
      * Debounce: a change only commits after it persists ``hold_frames``
        consecutive frames, so threshold flicker never becomes a count.
      * One count per opening event: close & re-open counts again — exactly
        'each fresh finger = one istighfar'.

    Smoothing (EMA) additionally absorbs MediaPipe landmark jitter.
    """

    def __init__(self, open_th, close_th, hold_frames: int = 3, alpha: float = 0.6):
        self.open_th = np.asarray(open_th, float)
        self.close_th = np.asarray(close_th, float)
        self.hold = max(1, int(hold_frames))
        self.alpha = alpha
        self.smooth = np.zeros(5, float)
        self.state = np.zeros(5, bool)
        self.streak = np.zeros(5, int)
        self.total = 0

    def reset(self) -> None:
        self.smooth[:] = 0.0
        self.state[:] = False
        self.streak[:] = 0
        self.total = 0

    def update(self, feats) -> int:
        """Feed one openness vector; returns current open-finger count."""
        x = self.smooth = (self.alpha * np.asarray(feats, float)
                           + (1 - self.alpha) * self.smooth)
        rising = (x >= self.open_th) & ~self.state
        falling = (x < self.close_th) & self.state
        self.streak = np.where(rising | falling, self.streak + 1, 0)
        opened = rising & (self.streak >= self.hold)
        closed = falling & (self.streak >= self.hold)
        self.total += int(opened.sum())
        self.state[opened] = True
        self.state[closed] = False
        self.streak[opened | closed] = 0
        return int(self.state.sum())


class FistCounter:
    """Count istighfar per completed fist-grip:  open hand → firm fist = +1.

    Uses the noise-robust per-finger machine internally, then adds a
    gesture-level debounce:

      * opening the hand (any finger confirmed open) *arms* the counter;
      * only a fully closed hand (a real fist), stable for ``fist_hold``
        consecutive frames, commits exactly ONE istighfar;
      * while the fist is held it never re-counts; opening again re-arms it.

    This directly maps the user gesture — *one closed fist = one tasbih* —
    and cannot produce spurious counts from jitter, partial grips or holds.
    """

    def __init__(self, open_th, close_th, hold_frames: int = 3,
                 alpha: float = 0.6, fist_hold: int = 4):
        self.fingers = FingerCounter(open_th, close_th, hold_frames, alpha)
        self.fist_hold = max(1, int(fist_hold))
        self.armed = False
        self.fist_frames = 0
        self.committed = False
        self.total = 0

    def reset(self) -> None:
        self.fingers.reset()
        self.armed = False
        self.fist_frames = 0
        self.committed = False
        self.total = 0

    def update(self, feats) -> int:
        """Feed one openness vector; returns open-finger count (0 = fist)."""
        n = self.fingers.update(feats)
        if n > 0:                       # hand open → arm the counter
            self.armed = True
            self.fist_frames = 0
            self.committed = False
        elif self.armed:                # hand closing …
            self.fist_frames += 1
            if self.fist_frames >= self.fist_hold and not self.committed:
                self.total += 1          # … and now a stable fist = one istighfar
                self.committed = True
        return n


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