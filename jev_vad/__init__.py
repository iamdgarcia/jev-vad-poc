"""jev_vad: semantic VAD over Spanish partial ASR transcripts.

Uses the Laya System-1 decision model (int4 ONNX, mmBERT multilingual
encoder) as a turn/barge-in detector for voice pipelines. The model
artifact (~206 MB) is downloaded from the Hugging Face hub on first
use and cached locally.

Quick start::

    from jev_vad import TurnDetector

    vad = TurnDetector()            # downloads on first run (~206 MB)
    d = vad.decide_barge_in("espera, que me he equivocado",
                            agent_context="Su reserva está confirmada.")
    if d.should_interrupt:          # tuned threshold + filler guard
        ...                         # stop TTS playback and listen
"""

from .detector import (  # noqa: F401
    TAKE_FLOOR_THRESHOLD,
    TurnDecision,
    TurnDetector,
    load_agent,
)
from .questions import (  # noqa: F401
    barge_questions,
    combined_questions,
    is_pure_filler,
    turn_questions,
)

__version__ = "0.1.0"

__all__ = [
    "TAKE_FLOOR_THRESHOLD",
    "TurnDecision",
    "TurnDetector",
    "load_agent",
    "barge_questions",
    "combined_questions",
    "is_pure_filler",
    "turn_questions",
    "__version__",
]
