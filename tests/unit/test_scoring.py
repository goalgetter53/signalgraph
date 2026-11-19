from datetime import UTC, datetime, timedelta

import pytest

from signalgraph.domain.models import Account, SignalEvent, SignalType, Source
from signalgraph.scoring.config import IcpConfig, Magnitude, ScoringConfig
from signalgraph.scoring.engine import decay, magnitude_factor, score_account
from tests.conftest import CONFIG

AS_OF = datetime(2026, 11, 19, tzinfo=UTC)
CFG = ScoringConfig.load(CONFIG / "scoring.yaml")
ICP = IcpConfig.load(CONFIG / "icp.yaml")
ACCT = Account(
    account_id="a1",
    name="Northwind Robotics",
    domain="northwindrobotics.com",
    country="US",
    member_mention_ids=["m1"],
)


def _ev(
    sig: SignalType,
    sub: str,
    days_ago: float,
    magnitude: float = 1.0,
    native: str = "x",
    **attrs: object,
) -> SignalEvent:
    return SignalEvent(
        source=Source.GREENHOUSE,
        native_id=native,
        signal_type=sig,
        subtype=sub,
        company_name="Northwind",
        occurred_at=AS_OF - timedelta(days=days_ago),
        title=f"{sub} {native}",
        magnitude=magnitude,
        attributes=dict(attrs),
    )


def test_decay_half_life() -> None:
    assert decay(0, 30) == 1.0
    assert decay(30, 30) == pytest.approx(0.5)
    assert decay(90, 30) == pytest.approx(0.125)
    assert decay(-5, 30) == 1.0  # future-dated signals are not boosted


def test_magnitude_factor() -> None:
    spec = Magnitude(kind="log10", reference=1e6, floor=0.25, cap=3.0)
    assert magnitude_factor(1e6, spec) == pytest.approx(1.0)
    assert magnitude_factor(1e7, spec) == pytest.approx(2.0)
    assert magnitude_factor(1e12, spec) == 3.0
    assert magnitude_factor(5e4, spec) == 0.25
    assert magnitude_factor(0, spec) == 0.25
    assert magnitude_factor(123, Magnitude()) == 1.0


def test_contributions_sum_to_raw_points_and_explain() -> None:
    events = [
        _ev(SignalType.HIRING, "gtm", 0, native="ae"),
        _ev(SignalType.HIRING, "ops", 30, native="office"),
        _ev(SignalType.FUNDING, "form_d", 90, magnitude=1e7, native="d"),
    ]
    bd = score_account(ACCT, events, CFG, ICP, AS_OF)
    assert bd.raw_points == pytest.approx(sum(c.points for c in bd.contributions), abs=1e-3)
    by_title = {c.title: c for c in bd.contributions}
    # hiring gtm: 3 * 2.0 * 1 * 1.0 = 6 ; ops: 3 * 0.3 * 0.5 = 0.45
    assert by_title["gtm ae"].points == pytest.approx(6.0)
    assert by_title["ops office"].points == pytest.approx(0.45)
    # funding: 25 * 1.0 * 2.0 (log10 $10M) * 0.5 (one half-life) = 25
    assert by_title["form_d d"].points == pytest.approx(25.0)
    assert bd.contributions[0].title == "form_d d"  # sorted by points
    assert 0 < bd.score < 100
    assert bd.icp_fit


def test_per_type_cap_and_saturation() -> None:
    events = [_ev(SignalType.HIRING, "gtm", 0, native=str(i)) for i in range(100)]
    bd = score_account(ACCT, events, CFG, ICP, AS_OF)
    assert bd.by_type["hiring"] == CFG.signals["hiring"].max_points
    assert bd.score < 100


def test_max_age_drops_stale_signals() -> None:
    bd = score_account(ACCT, [_ev(SignalType.HIRING, "gtm", 400)], CFG, ICP, AS_OF)
    assert bd.contributions == []
    assert bd.score == 0


def test_amendment_multiplier() -> None:
    new = score_account(ACCT, [_ev(SignalType.FUNDING, "form_d", 0, 1e6)], CFG, ICP, AS_OF)
    amended = score_account(
        ACCT, [_ev(SignalType.FUNDING, "form_d", 0, 1e6, is_amendment=True)], CFG, ICP, AS_OF
    )
    assert amended.raw_points == pytest.approx(new.raw_points * 0.5)


def test_icp_reasons() -> None:
    fund = Account(
        account_id="f",
        name="Ember & Oak Ventures Fund I, L.P.",
        domain=None,
        country="KY",
        member_mention_ids=["x"],
    )
    bd = score_account(fund, [_ev(SignalType.FUNDING, "form_d_fund", 1, 4e7)], CFG, ICP, AS_OF)
    assert not bd.icp_fit
    joined = " | ".join(bd.icp_reasons)
    assert "country KY not in ICP" in joined
    assert "excluded pattern" in joined
    assert "qualifying signals" in joined
    assert bd.score == 0  # fund subtype weight is 0
