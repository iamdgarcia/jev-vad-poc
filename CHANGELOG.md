# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-10-05

### Added

- `jev_vad` package: `TurnDetector` with `decide_barge_in`,
  `decide_turn` and `decide` (both questions in one ONNX pass).
- Automatic model download from Hugging Face
  (`androidli/laya-multilingual-onnx-int4`, ~206 MB) with local
  resolution order: `JEV_VAD_MODEL_DIR` env var → sibling serving-repo
  artifact → HF cache. Corporate SSL interception handled via
  `truststore`.
- Sweep-tuned barge-in operating point: `P(take_floor) ≥ 0.30` plus a
  pure-filler guard, exposed as `TurnDecision.should_interrupt`.
- `debug_sweep.py`: 100-phrase labeled Spanish benchmark ×
  question-wording × state-shape grid, with per-class F1, confusion
  matrices and threshold tuning.
- `replay.py`: self-check scenarios + `--perf` latency mode.
- `tests/`: 6 package-level contract tests.
