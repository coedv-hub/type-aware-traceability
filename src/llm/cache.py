"""Recoverable content-addressed cache for LLM calls."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .base_client import LLMResponse


@dataclass(frozen=True)
class CacheIdentity:
    model: str
    prompt: str
    dataset: str
    requirement_id: str
    code_file: str
    provider: str = ""

    def digest(self) -> str:
        canonical = json.dumps(
            asdict(self), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def legacy_digest(self) -> str:
        """Return the pre-provider cache digest for backward-compatible reads."""
        legacy = asdict(self)
        legacy.pop("provider")
        canonical = json.dumps(
            legacy, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LLMCache:
    """One JSON file per prompt; writes are atomic and safe to resume."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, identity: CacheIdentity) -> Path:
        digest = identity.digest()
        return self.root / digest[:2] / f"{digest}.json"

    def get(
        self, identity: CacheIdentity, allow_estimated: bool = False
    ) -> LLMResponse | None:
        expected = asdict(identity)
        paths = [self.path_for(identity)]
        if identity.provider:
            digest = identity.legacy_digest()
            paths.append(self.root / digest[:2] / f"{digest}.json")
        for path in paths:
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                stored_identity = payload.get("identity")
                is_current = stored_identity == expected
                is_legacy = (
                    identity.provider
                    and isinstance(stored_identity, dict)
                    and "provider" not in stored_identity
                    and stored_identity
                    == {key: value for key, value in expected.items() if key != "provider"}
                )
                if not (is_current or is_legacy):
                    continue
                response = LLMResponse.from_dict(payload["response"])
                if is_legacy and response.provider != identity.provider:
                    continue
                if response.estimated and not allow_estimated:
                    continue
                return response
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return None

    def put(
        self, identity: CacheIdentity, response: LLMResponse
    ) -> Path:
        path = self.path_for(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "identity": asdict(identity),
            "response": response.to_dict(),
        }
        encoded = json.dumps(payload, indent=2, ensure_ascii=False)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return path
