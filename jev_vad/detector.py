"""Turn detector over partial transcripts using the laya decision model.

One `TurnDetector` per process; model load is cached. Two decision
modes plus a combined one (both questions in a single ONNX pass):

- `decide_barge_in(partial, agent_context)`: agent speaking.
- `decide_turn(partial)`: agent silent.
- `decide(partial, agent_context)`: one pass, both answers.

State configuration is the sweep winner (see README): English-framed
barge question over a Spanish dict state, tuned take_floor threshold,
pure-filler guard.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from laya.onnx_agent import ONNXAgent

from .questions import (
    barge_questions,
    combined_questions,
    is_pure_filler,
    turn_questions,
)

ONNX_FILE = "laya-multilingual-int4-blk32.onnx"
INTRA_OP_THREADS = 4

# Sweep-tuned operating point: interrupt when P(take_floor) >= this;
# pure fillers never interrupt (model scores them ~50/50). 0.30 gives
# the best F1 on the 50-phrase barge set (P=0.68 R=0.85 F1=0.76).
TAKE_FLOOR_THRESHOLD = 0.30

# ponytail: single global detector assumed; no per-call isolation.
_SESSION_CACHE: dict[str, ONNXAgent] = {}


def load_agent(model_dir: Path) -> ONNXAgent:
    """Load the ONNX agent once per process, with tuned ORT threads."""
    cached = _SESSION_CACHE.get(str(model_dir))
    if cached is not None:
        return cached
    import onnxruntime as ort

    agent = ONNXAgent(str(model_dir), onnx_path=str(model_dir / ONNX_FILE))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = INTRA_OP_THREADS
    so.inter_op_num_threads = 1
    agent.session = ort.InferenceSession(
        str(model_dir / ONNX_FILE),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    _SESSION_CACHE[str(model_dir)] = agent
    return agent


class TurnDecision:
    """One detection result, ready for the voice pipeline to consume."""

    __slots__ = (
        "label",
        "confidence",
        "probabilities",
        "latency_ms",
        "should_interrupt",
    )

    def __init__(
        self,
        label: str,
        confidence: float,
        probabilities: dict[str, float],
        latency_ms: float,
        should_interrupt: bool = False,
    ) -> None:
        """Store decision fields."""
        self.label = label
        self.confidence = confidence
        self.probabilities = probabilities
        self.latency_ms = latency_ms
        self.should_interrupt = should_interrupt

    def __repr__(self) -> str:
        """Short human-readable decision."""
        return (
            f"TurnDecision({self.label} {self.confidence:.2f}, "
            f"{self.latency_ms:.0f} ms)"
        )


class TurnDetector:
    """Decides floor state from partial transcripts in one model pass."""

    def __init__(
        self,
        model_dir: Path | None = None,
        max_state_words: int = 40,
        take_floor_threshold: float = TAKE_FLOOR_THRESHOLD,
    ) -> None:
        """Resolve (or download) the model and load it once.

        `model_dir=None` uses `JEV_VAD_MODEL_DIR`, then the sibling
        serving-repo artifact, then the Hugging Face hub cache.
        """
        from ._download import ensure_model_dir

        self._dir = ensure_model_dir(model_dir)
        self._agent = load_agent(self._dir)
        self._max_state_words = max_state_words
        self._threshold = take_floor_threshold
        self._last_usage: dict[str, Any] = {}

    def _tail(self, text: str) -> str:
        """Keep the last `max_state_words` whitespace words."""
        words = text.split()
        return " ".join(words[-self._max_state_words :])

    def _predict(self, state: Any, qs: dict) -> tuple[dict, float]:
        """One timed model pass; returns (answers, latency_ms)."""
        t0 = time.perf_counter()
        out = self._agent.predict(state, qs)
        latency = (time.perf_counter() - t0) * 1000
        self._last_usage = out.get("usage", {})
        return out["answers"], latency

    def _should_interrupt(self, partial: str, barge_answer: dict) -> bool:
        """Pipeline action: tuned threshold + pure-filler guard."""
        if is_pure_filler(partial):
            return False
        return barge_answer["probabilities"].get("take_floor", 0.0) >= self._threshold

    def _barge_decision(
        self, answers: dict, latency: float, partial: str
    ) -> TurnDecision:
        """Package a barge answer with the tuned interrupt action."""
        a = answers["barge"]
        return TurnDecision(
            label=a["choice"],
            confidence=a["answer_confidence"],
            probabilities=dict(a["probabilities"]),
            latency_ms=latency,
            should_interrupt=self._should_interrupt(partial, a),
        )

    def decide(
        self,
        partial: str,
        agent_context: str = "",
        with_addressee: bool = True,
    ) -> TurnDecision:
        """Both questions in one pass; keep the barge answer."""
        state = {
            "agent_context": self._tail(agent_context),
            "partial_transcript": self._tail(partial),
        }
        qs = combined_questions() if with_addressee else barge_questions()
        answers, latency = self._predict(state, qs)
        return self._barge_decision(answers, latency, partial)

    def decide_barge_in(
        self,
        partial: str,
        agent_context: str,
    ) -> TurnDecision:
        """Classify a user interjection while the agent was speaking."""
        state = {
            "agent_context": self._tail(agent_context),
            "partial_transcript": self._tail(partial),
        }
        answers, latency = self._predict(state, barge_questions())
        return self._barge_decision(answers, latency, partial)

    def decide_turn(self, partial: str) -> TurnDecision:
        """Guess whether the partial transcript is a complete turn.

        Semantic-only signal (hard-cut fragments read complete); pair
        with acoustic end-of-speech pauses.
        """
        state = {"partial_transcript": self._tail(partial)}
        answers, latency = self._predict(state, turn_questions())
        a = answers["floor"]
        return TurnDecision(
            label=a["choice"],
            confidence=a["answer_confidence"],
            probabilities=dict(a["probabilities"]),
            latency_ms=latency,
        )

    def usage(self) -> dict[str, Any]:
        """Return token usage of the last decision."""
        return dict(self._last_usage)
