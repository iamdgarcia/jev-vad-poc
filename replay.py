"""Replay partial transcripts through the TurnDetector PoC (Spanish).

Self-check scenarios (default): scripted timelines with known
expectations for both modes — prints the decision timeline and latency.

--perf: latency distribution across input sizes (no assertions).
"""

from __future__ import annotations

import argparse
import statistics
import time

from jev_vad import TurnDetector

# (agent_context, partial, expected)
BARGE_SCENARIO: list[tuple[str, str, str]] = [
    (
        "Reserva confirmada. Referencia ABX123. Recibirás un correo en breve.",
        "sí",
        "backchannel",
    ),
    (
        "Tengo vuelos el jueves y el viernes por la mañana. ¿Cuál prefiere?",
        "perdón, quería un tren en realidad",
        "take_floor",
    ),
    (
        "Reserva confirmada. Referencia ABX123. Recibirás un correo en breve.",
        "espera, ese es el vuelo equivocado",
        "take_floor",
    ),
    (
        "Reserva confirmada. Referencia ABX123. Recibirás un correo en breve.",
        "perfecto, gracias",
        "backchannel",
    ),
]

# (partial, expected) — thin-signal mode; only unambiguous cases asserted
# (semantic-completeness signal; hard-cut fragments read complete — see README)
TURN_SCENARIO: list[tuple[str, str]] = [
    ("quiero reservar un vuelo a París para mañana", "turn_complete"),
    ("eh... no sé", "still_speaking"),
]


def run_scenarios() -> bool:
    """Replay both scenarios; print timeline; return all-OK flag."""
    detector = TurnDetector()
    ok = True

    print("== barge-in (agente hablando) ==")
    print(f"{'partial':<44} {'label':<13} {'ms':>4}")
    for ctx, partial, want in BARGE_SCENARIO:
        d = detector.decide_barge_in(partial, agent_context=ctx)
        flag = "OK " if d.label == want else "BAD"
        ok &= d.label == want
        print(f"{flag} {partial[:42]:<44} {d.label:<13} {d.latency_ms:>4.0f}")

    print("\n== turn-complete (agente en silencio) ==")
    for partial, want in TURN_SCENARIO:
        d = detector.decide_turn(partial)
        flag = "OK " if d.label == want else "BAD"
        ok &= d.label == want
        print(f"{flag} {partial[:42]:<44} {d.label:<13} {d.latency_ms:>4.0f}")
    return ok


def run_perf() -> None:
    """Latency distribution across input sizes."""
    detector = TurnDetector()
    print(f"{'words':>6} {'p50':>7} {'min':>7} {'p90':>7}")
    for n_words in (4, 8, 16, 32):
        partial = " ".join(f"palabra{i}" for i in range(n_words)) + " paris"
        lats: list[float] = []
        for _ in range(9):
            t0 = time.perf_counter()
            detector.decide_turn(partial)
            lats.append(time.perf_counter() - t0)
        lats.sort()
        print(
            f"{n_words:>6} {statistics.median(lats) * 1000:>6.0f} "
            f"{lats[0] * 1000:>6.0f} {lats[-2] * 1000:>6.0f}"
        )


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf", action="store_true", help="latency mode")
    args = parser.parse_args()
    if args.perf:
        run_perf()
        return
    ok = run_scenarios()
    print(f"\nescenarios {'ALL OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
