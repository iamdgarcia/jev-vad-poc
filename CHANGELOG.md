# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-10-06

### Added

- LaTeX paper draft (`paper/`): "Asking Instead of Listening" — 17 pp,
  33 verified citations, 5-reviewer audit applied. Dual-environment
  latency table (laptop + Cloud Run 8 vCPU, n=100/cell, Wilson CIs).
- Google Cloud benchmark: `cloud_bench.py` + Cloud Build config +
  Cloud Run Job `jev-vad-bench` (8 vCPU); results in
  `benchmark/cloud_bench1.json`.
- Pipeline investigation docs (`docs/`): iabto-ms-bc anatomy +
  latency decomposition + smart-turn vs semantic comparison, with
  smart-turn v3.2 verified locally (146 ms p50, 8.7 MB).
- Frozen artifacts: `benchmark/benchmark.json` (100 phrases, md5
  293f9a66d26af383013e85d0ce076256), `sweep_results.json`,
  `operating_point.json`.

### Fixed

- Sweep scorer label mismatch (mid_turn vs still_speaking) that
  invalidated the turn-half accuracies; full sweep re-run — true
  turn-half accuracy is 0.40-0.54 (near chance).
- Operating-point table regenerated from a single probability trace
  with guard/no-guard rows at every threshold.

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
