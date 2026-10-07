"""Model download: resolve (or fetch) the Laya int4 ONNX artifact.

Resolution order:
1. `JEV_VAD_MODEL_DIR` env var — a directory that already contains the
   artifact (offline deployments; the serving repo's `src/models/laya`).
2. HF hub cache via `snapshot_download` (first use downloads ~206 MB;
   later runs are instant).

On corporate networks with SSL interception (self-signed CA in the
system store but not in certifi), `truststore` injects the OS trust
store so huggingface_hub works without disabling verification.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

HF_REPO = "androidli/laya-multilingual-onnx-int4"
ONNX_NAME = "laya-multilingual-int4-blk32.onnx"
SMALL_PATTERNS = ["rl_agent_config.json", "tokenizer/*", "encoder/*"]


def _ssl_ready() -> None:
    """Let huggingface_hub verify against the OS trust store."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass  # normal networks: certifi already works


def default_model_dir() -> Path:
    """Preconfigured local artifact dir, if one exists."""
    env = os.environ.get("JEV_VAD_MODEL_DIR")
    if env:
        return Path(env)
    sibling = (
        Path(__file__).resolve().parents[2] / "ibgc-openapi-models/src/models/laya"
    )
    if (sibling / ONNX_NAME).exists():
        return sibling
    return sibling  # not present; caller falls back to the hub


def ensure_model_dir(model_dir: Path | None = None) -> Path:
    """Return a local dir containing the ONNX artifact, downloading it
    from the Hugging Face hub on cache miss."""
    if model_dir is not None:
        return Path(model_dir)
    env = os.environ.get("JEV_VAD_MODEL_DIR")
    if env and (Path(env) / ONNX_NAME).exists():
        return Path(env)
    sibling = (
        Path(__file__).resolve().parents[2] / "ibgc-openapi-models/src/models/laya"
    )
    if (sibling / ONNX_NAME).exists():
        return sibling

    _ssl_ready()
    from huggingface_hub import hf_hub_download, snapshot_download

    # Small files through snapshot_download (ONNXAgent needs config +
    # tokenizer in one dir); the big graph + weights via hf_hub_download.
    # Everything is then COPIED (real files, no symlinks) into one clean
    # dir: onnxruntime resolves symlinks before validating external data,
    # so a symlinked .onnx whose .data lives in the same snapshot dir
    # still fails validation ("External data path escapes model
    # directory"). Real files keep the graph and its .data together.
    snapshot_dir = Path(snapshot_download(HF_REPO, allow_patterns=SMALL_PATTERNS))
    clean = snapshot_dir.parent / "jev-vad-bundle"
    clean.mkdir(parents=True, exist_ok=True)
    for src in (
        *[snapshot_dir / p for p in ("rl_agent_config.json",)],
        *sorted(snapshot_dir.glob("tokenizer/*")),
        *sorted(snapshot_dir.glob("encoder/*")),
        Path(hf_hub_download(HF_REPO, ONNX_NAME)),
        Path(hf_hub_download(HF_REPO, f"{ONNX_NAME}.data")),
    ):
        dst = clean / src.name if src.parent == snapshot_dir else clean / src.relative_to(snapshot_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dst)
    return clean
