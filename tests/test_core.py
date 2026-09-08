import numpy as np
import pytest

import core


@pytest.fixture(scope="module")
def model_and_explainer():
    m, e = core.build_count_model()
    return m, e


@pytest.fixture(scope="module")
def thresholds():
    return core.count_thresholds()


def make_counter(thresholds, **kw):
    open_th, close_th = thresholds
    return core.FistCounter(open_th, close_th, hold_frames=3, alpha=0.6, **kw)


def feats_by_mask(mask):
    return core.finger_features(core.synthetic_hand(mask, seed=3))


def open_all(c):
    for _ in range(5):
        c.update(feats_by_mask(31))


def close_all(c):
    for _ in range(8):
        c.update(feats_by_mask(0))


@pytest.mark.parametrize(
    "mask,expected",
    [(0, 0), (1, 1), (2, 1), (4, 1), (8, 1), (16, 1),
     (3, 2), (7, 3), (15, 4), (31, 5)],
)
def test_feature_extraction_and_model_count(model_and_explainer, mask, expected):
    m, _ = model_and_explainer
    feats = core.finger_features(core.synthetic_hand(mask))
    assert core.count_fingers(m, feats) == expected


def test_all_hand_masks_count(model_and_explainer):
    m, _ = model_and_explainer
    for mask in range(32):
        feats = core.finger_features(core.synthetic_hand(mask))
        expected = bin(mask).count("1")
        assert core.count_fingers(m, feats) == expected, f"mask {mask}"


def test_features_are_scale_invariant():
    a = core.finger_features(core.synthetic_hand(7, seed=3))
    scaled = core.synthetic_hand(7, seed=3).copy()
    scaled[:, :2] *= 2.0
    b = core.finger_features(scaled)
    assert np.allclose(a, b, atol=1e-2)


def test_fold_reduces_feature():
    assert core.finger_features(core.synthetic_hand(0)).sum() < \
           core.finger_features(core.synthetic_hand(31)).sum()


def test_shap_explains_each_finger(model_and_explainer):
    _, e = model_and_explainer
    vals = core.explain_count(e, core.finger_features(core.synthetic_hand(7)))
    assert set(vals) == set(core.FINGER_NAMES)
    # mask 7 = 0b00111 -> thumb, index, middle open; ring & pinky folded
    assert vals["thumb"] > 0 and vals["index"] > 0 and vals["middle"] > 0
    assert vals["ring"] < 0 and vals["pinky"] < 0  # folded fingers oppose


def test_hysteresis_stable(model_and_explainer):
    m, _ = model_and_explainer
    # confident frames match the plain count; count stays bounded in [0,5]
    f0 = core.finger_features(core.synthetic_hand(0))
    n0, _ = core.count_with_stable_finger(m, f0, prev=5)
    assert n0 == 0
    f3 = core.finger_features(core.synthetic_hand(3))
    n3, _ = core.count_with_stable_finger(m, f3, prev=0)
    assert n3 == 2


# -------------------------------------------------------------------- FistCounter
def test_fist_counts_open_then_grip(thresholds):
    c = make_counter(thresholds)
    assert c.update(feats_by_mask(0)) == 0   # starts as fist
    open_all(c)
    assert c.armed
    close_all(c)
    assert c.total == 1                       # one grip = one istighfar


def test_fist_never_recounts_while_held(thresholds):
    c = make_counter(thresholds)
    open_all(c)
    close_all(c)
    assert c.total == 1
    for _ in range(50):                       # hold the fist forever
        c.update(feats_by_mask(0))
        assert c.total == 1


def test_fist_reopen_counts_again(thresholds):
    c = make_counter(thresholds)
    open_all(c); close_all(c)
    open_all(c); close_all(c)
    open_all(c); close_all(c)
    assert c.total == 3                       # repeat cycle -> 3 tasbih


def test_partial_grip_does_not_count(thresholds):
    c = make_counter(thresholds)
    open_all(c)
    # close only to 2 fingers and hold forever: not a fist
    for _ in range(30):
        c.update(feats_by_mask(3))
    assert c.total == 0


def test_fist_ignores_jitter_in_band(thresholds):
    c = make_counter(thresholds)
    open_th, close_th = thresholds
    mid = (open_th + close_th) / 2
    rng = np.random.default_rng(0)
    for _ in range(40):                       # fingers dangling mid-band
        c.update(mid + rng.normal(0, 0.03, 5))
    assert c.total == 0


def test_fist_reset(thresholds):
    c = make_counter(thresholds)
    open_all(c); close_all(c)
    c.reset()
    assert c.total == 0
    open_all(c); close_all(c)
    assert c.total == 1