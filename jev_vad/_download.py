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
    snapshot_dir = Path(snapshot_download(HF_REPO, allow_patterns=SMALL_PATTERNS))
    for name in (ONNX_NAME, f"{ONNX_NAME}.data"):
        hf_hub_download(HF_REPO, name)
    return snapshot_dir
