import numpy as np
import pytest

from signalgraph.resolution.features import FEATURE_NAMES
from signalgraph.resolution.scorers import (
    FellegiSunterMatcher,
    LogisticMatcher,
    load_default_model,
)


def _toy(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = np.zeros((n, len(FEATURE_NAMES)))
    x[:, 0] = np.where(y == 1, rng.uniform(0.85, 1, n), rng.uniform(0.4, 0.9, n))
    x[:, 1] = np.where(y == 1, rng.uniform(0.6, 1, n), rng.uniform(0, 0.6, n))
    dm, dmm = FEATURE_NAMES.index("domain_match"), FEATURE_NAMES.index("domain_mismatch")
    x[:, dm] = np.where(y == 1, rng.integers(0, 2, n), 0)  # domain match only for matches
    x[:, dmm] = np.where(y == 0, rng.integers(0, 2, n), 0)  # mismatch only for non-matches
    return x, y


@pytest.mark.parametrize("model_cls", [LogisticMatcher, FellegiSunterMatcher])
def test_models_separate_toy_data(model_cls: type) -> None:
    x, y = _toy()
    model = model_cls().fit(x, y)
    p = model.predict_proba(x)
    assert ((p >= 0.5) == (y == 1)).mean() > 0.9
    assert np.all((p >= 0) & (p <= 1))


def test_logistic_learns_sensible_signs() -> None:
    x, y = _toy()
    m = LogisticMatcher().fit(x, y)
    coef = dict(zip(FEATURE_NAMES, m.coef_, strict=True))
    assert coef["jaro_winkler"] > 0
    assert coef["domain_match"] > 0
    assert coef["domain_mismatch"] < 0


def test_logistic_roundtrip() -> None:
    x, y = _toy()
    m = LogisticMatcher().fit(x, y)
    m2 = LogisticMatcher.from_dict(m.to_dict())
    np.testing.assert_allclose(m.predict_proba(x), m2.predict_proba(x))


def test_from_dict_rejects_feature_drift() -> None:
    data = LogisticMatcher().to_dict()
    data["features"] = ["jaro_winkler"]
    with pytest.raises(ValueError, match="feature list"):
        LogisticMatcher.from_dict(data)


def test_shipped_model_loads_and_is_monotone_in_domain_evidence() -> None:
    m = load_default_model()
    base = np.zeros((1, len(FEATURE_NAMES)))
    base[0, :4] = [0.9, 0.7, 1.0, 1.0]
    match, mismatch = base.copy(), base.copy()
    match[0, FEATURE_NAMES.index("domain_match")] = 1
    mismatch[0, FEATURE_NAMES.index("domain_mismatch")] = 1
    assert m.predict_proba(match)[0] > m.predict_proba(base)[0] > m.predict_proba(mismatch)[0]
