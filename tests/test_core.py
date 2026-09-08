import numpy as np
import pytest

import core


@pytest.fixture(scope="module")
def model_and_explainer():
    m, e = core.build_count_model()
    return m, e


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