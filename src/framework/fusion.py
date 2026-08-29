"""Fixed signal-fusion helpers for type-aware framework components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixedMixedSignalFusion:
    """Global fixed fusion interface for Mixed FR/NFR signals."""

    functional_weight: float = 0.5

    def combine(self, functional_score: float, quality_score: float) -> float:
        functional = max(0.0, min(1.0, functional_score))
        quality = max(0.0, min(1.0, quality_score))
        weight = max(0.0, min(1.0, self.functional_weight))
        return weight * functional + (1.0 - weight) * quality


DEFAULT_MIXED_FUSION = FixedMixedSignalFusion()
