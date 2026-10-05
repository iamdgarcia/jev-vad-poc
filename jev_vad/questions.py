"""Typed question definitions for the laya-as-VAD PoC.

Winner of the config sweep (debug_sweep.py over 100 Spanish phrases):

- BARGE-IN: **English instructions + English criteria, Spanish state**
  (dict with `agent_context` + `partial_transcript`). mmBERT's task
  framing is English-native; the same question written in Spanish
  loses ~0.2-0.5 take_floor separation. Threshold the `take_floor`
  probability at 0.30 + a pure-filler guard, not the argmax.
- TURN-COMPLETE: **Spanish wording** (the English mirror flattens to
  ~0.5 everywhere). The signal is semantic, not syntactic — hard-cut
  fragments read complete — so treat as best-effort and pair with
  acoustic end-of-speech pause detection.
"""

from __future__ import annotations

from typing import Any

BARGE_QUESTION: dict[str, Any] = {
    "type": "choice",
    "instructions": (
        "The assistant just said `agent_context`. The user interrupted "
        "with `partial_transcript`. Is the user taking the floor from "
        "the assistant?"
    ),
    "criteria": {
        "take_floor": (
            "a deliberate interjection or correction directed at the "
            "assistant; the agent should stop and listen"
        ),
        "backchannel": (
            "a brief acknowledgment or filler; the user is not trying to stop the agent"
        ),
        "side_talk": ("the user is talking to someone else, not to the assistant"),
    },
}

TURN_QUESTION: dict[str, Any] = {
    "type": "choice",
    "instructions": (
        "¿Es `partial_transcript` un enunciado completo, o una "
        "transcripción parcial todavía en curso?"
    ),
    "criteria": {
        "turn_complete": (
            "El enunciado forma una petición o afirmación completa "
            "y autocontenida; el hablante ha terminado y espera "
            "respuesta"
        ),
        "still_speaking": (
            "El enunciado es un fragmento incompleto; el hablante "
            "está en mitad de una frase y se esperan más palabras"
        ),
    },
}

# Short pure fillers: the model scores them near 50/50 (defensible —
# intonation decides), so the pipeline rule shorts them to backchannel
# before consulting the model. Keep in sync with the sweep.
PURE_FILLERS: frozenset[str] = frozenset(
    {"eh", "mhm", "ajá", "uhm", "mm", "em", "ay", "ehm", "ah", "aja", "ej"}
)


def is_pure_filler(text: str) -> bool:
    """True when the partial is only a bare filler token."""
    return text.lower().strip("¡!¿?.,;: ") in PURE_FILLERS


def barge_questions() -> dict[str, dict[str, Any]]:
    """Question set for the agent-speaking mode."""
    return {"barge": BARGE_QUESTION}


def turn_questions() -> dict[str, dict[str, Any]]:
    """Question set for the agent-silent mode."""
    return {"floor": TURN_QUESTION}


def combined_questions() -> dict[str, dict[str, Any]]:
    """Both questions in one pass (~2x latency of one; only when the
    pipeline needs both answers on the same audio frame)."""
    return {"barge": BARGE_QUESTION, "floor": TURN_QUESTION}
