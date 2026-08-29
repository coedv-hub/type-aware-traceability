"""Sentence-BERT and CodeBERT baselines with persistent embedding caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CacheStats:
    """Cache diagnostics for one artifact collection."""

    hit: bool
    embeddings: int
    path: str
    verification_misses: int = 0


def select_device(requested: str = "auto") -> str:
    """Select CUDA/MPS when available and otherwise fall back to CPU."""
    import torch

    requested = requested.casefold()
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if requested == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is not available.")
        if requested not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingCache:
    """Content-addressed, non-pickle NumPy cache for artifact embeddings."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @staticmethod
    def _fingerprint(
        model_name: str,
        model_revision: str,
        max_length: int,
        artifact_ids: list[str],
        texts: list[str],
    ) -> str:
        digest = hashlib.sha256()
        metadata = {
            "schema": 1,
            "model": model_name,
            "revision": model_revision,
            "max_length": max_length,
            "count": len(artifact_ids),
        }
        digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        for artifact_id, text in zip(artifact_ids, texts, strict=True):
            digest.update(b"\0")
            digest.update(artifact_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(text.encode("utf-8", errors="replace"))
        return digest.hexdigest()[:20]

    def path_for(
        self,
        dataset: str,
        method: str,
        role: str,
        model_name: str,
        model_revision: str,
        max_length: int,
        artifact_ids: list[str],
        texts: list[str],
    ) -> Path:
        fingerprint = self._fingerprint(
            model_name, model_revision, max_length, artifact_ids, texts
        )
        return self.root / dataset / method / f"{role}-{fingerprint}.npz"

    @staticmethod
    def load(path: Path, expected_ids: list[str]) -> np.ndarray | None:
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as cached:
                ids = cached["ids"].astype(str).tolist()
                embeddings = cached["embeddings"].astype(np.float32, copy=False)
        except (OSError, ValueError, KeyError):
            return None
        if ids != expected_ids or embeddings.ndim != 2:
            return None
        if embeddings.shape[0] != len(expected_ids):
            return None
        if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
            return None
        if np.any(np.linalg.norm(embeddings, axis=1) <= 0.0):
            return None
        return embeddings

    @staticmethod
    def save(path: Path, artifact_ids: list[str], embeddings: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            ids=np.asarray(artifact_ids, dtype=np.str_),
            embeddings=np.asarray(embeddings, dtype=np.float32),
        )
        temporary.replace(path)


class NeuralBaseline:
    """Common caching and cosine-scoring behavior for neural encoders."""

    name = "base"

    def __init__(
        self,
        model_name: str,
        model_revision: str,
        cache_dir: str | Path,
        dataset_name: str,
        batch_size: int,
        max_length: int,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.cache = EmbeddingCache(cache_dir)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = select_device(device)
        self.code_embeddings: np.ndarray | None = None
        self.cache_stats: dict[str, CacheStats] = {}

    def _encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] == 0:
            raise RuntimeError(f"Invalid embedding shape: {embeddings.shape}")
        if not np.isfinite(embeddings).all():
            raise RuntimeError("Encoder produced non-finite embeddings.")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise RuntimeError("Encoder produced one or more empty embeddings.")
        return embeddings / norms

    def _cached_encode(
        self, role: str, artifact_ids: list[str], texts: list[str]
    ) -> np.ndarray:
        if len(artifact_ids) != len(texts):
            raise ValueError("Artifact IDs and texts must have equal lengths.")
        if not artifact_ids:
            raise ValueError(f"Cannot encode an empty {role} collection.")
        path = self.cache.path_for(
            self.dataset_name,
            self.name,
            role,
            self.model_name,
            self.model_revision,
            self.max_length,
            artifact_ids,
            texts,
        )
        embeddings = self.cache.load(path, artifact_ids)
        hit = embeddings is not None
        if embeddings is None:
            embeddings = self._normalize(self._encode(texts))
            self.cache.save(path, artifact_ids, embeddings)
            verified = self.cache.load(path, artifact_ids)
            if verified is None:
                raise RuntimeError(f"Embedding cache verification failed: {path}")
            embeddings = verified
        self.cache_stats[role] = CacheStats(
            hit=hit,
            embeddings=len(artifact_ids),
            path=str(path),
            verification_misses=0,
        )
        return embeddings

    def fit(self, code_ids: list[str], code_texts: list[str]) -> "NeuralBaseline":
        self.code_embeddings = self._cached_encode("code", code_ids, code_texts)
        return self

    def score_all(
        self, requirement_ids: list[str], requirement_texts: list[str]
    ) -> np.ndarray:
        if self.code_embeddings is None:
            raise RuntimeError("Call fit() before score_all().")
        requirements = self._cached_encode(
            "requirements", requirement_ids, requirement_texts
        )
        scores = requirements @ self.code_embeddings.T
        return np.clip(scores, -1.0, 1.0)


class SentenceBERTBaseline(NeuralBaseline):
    name = "sentencebert"

    def _encode(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        if not hasattr(self, "_model"):
            self._model = SentenceTransformer(
                self.model_name,
                revision=self.model_revision,
                device=self.device,
                trust_remote_code=False,
            )
            self._model.max_seq_length = self.max_length
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )


class CodeBERTBaseline(NeuralBaseline):
    name = "codebert"

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not hasattr(self, "_model"):
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                trust_remote_code=False,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                trust_remote_code=False,
            ).to(self.device)
            self._model.eval()

        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                encoded = self._tokenizer(
                    texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                hidden = self._model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                batches.append(pooled.cpu().numpy())
        return np.concatenate(batches, axis=0)


def create_neural_baseline(
    method: str,
    config: dict[str, Any],
    dataset_name: str,
    cache_dir: str | Path,
    device: str = "auto",
) -> NeuralBaseline:
    """Create a configured neural baseline without loading model weights."""
    method = method.casefold()
    if method not in {"sentencebert", "codebert"}:
        raise ValueError(f"Unknown neural method: {method}")
    item = config["neural"][method]
    common = {
        "model_name": str(item["model_name"]),
        "model_revision": str(item["revision"]),
        "cache_dir": cache_dir,
        "dataset_name": dataset_name,
        "batch_size": int(item["batch_size"]),
        "max_length": int(item["max_length"]),
        "device": device,
    }
    if method == "sentencebert":
        return SentenceBERTBaseline(**common)
    return CodeBERTBaseline(**common)
