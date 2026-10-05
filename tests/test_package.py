"""Package-level smoke test: model resolves and decisions flow.

Uses whatever artifact is available locally (env var, sibling repo, or
the HF cache) so it runs offline too; network is only hit on a true
cache miss.
"""

from __future__ import annotations

import pytest

from jev_vad import TurnDecision, TurnDetector


@pytest.fixture(scope="module")
def detector() -> TurnDetector:
    """One shared detector (model load is slow)."""
    return TurnDetector()


def test_barge_contract(detector: TurnDetector) -> None:
    d = detector.decide_barge_in(
        "espera, que me he equivocado de fecha",
        agent_context="Tu reserva está confirmada, referencia ABX123.",
    )
    assert isinstance(d, TurnDecision)
    assert d.label in {"take_floor", "backchannel", "side_talk"}
    assert 0.0 <= d.confidence <= 1.0
    assert set(d.probabilities) == {"take_floor", "backchannel", "side_talk"}
    assert abs(sum(d.probabilities.values()) - 1.0) < 1e-3
    assert isinstance(d.should_interrupt, bool)


def test_barge_clear_correction_interrupts(detector: TurnDetector) -> None:
    d = detector.decide_barge_in(
        "no, no, al revés, quiero el del jueves",
        agent_context="Tengo vuelos el jueves y el viernes, ¿cuál prefiere?",
    )
    assert d.should_interrupt


def test_barge_filler_does_not_interrupt(detector: TurnDetector) -> None:
    d = detector.decide_barge_in(
        "vale, de acuerdo",
        agent_context="Tu reserva está confirmada, referencia ABX123.",
    )
    assert not d.should_interrupt


def test_pure_filler_is_short_circuited(detector: TurnDetector) -> None:
    d = detector.decide_barge_in(
        "eh",
        agent_context="Tu reserva está confirmada, referencia ABX123.",
    )
    assert not d.should_interrupt


def test_turn_contract(detector: TurnDetector) -> None:
    d = detector.decide_turn("quiero reservar un vuelo a París mañana")
    assert d.label in {"turn_complete", "still_speaking"}
    assert 0.0 <= d.confidence <= 1.0


def test_decision_is_fast_enough_warm(detector: TurnDetector) -> None:
    """Warm decisions must stay well under 1s on laptop CPU."""
    detector.decide_turn("calienta la sesión")
    d = detector.decide_turn("quiero reservar un vuelo a París mañana")
    assert d.latency_ms < 1000
