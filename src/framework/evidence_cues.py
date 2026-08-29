"""Shared type-specific evidence cues for the IST framework."""

from __future__ import annotations

import re
from collections.abc import Iterable


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?")

EVIDENCE_CUE_ALIASES: dict[str, tuple[str, ...]] = {
    "validation": (
        "validation",
        "validate",
        "validator",
        "check",
        "constraint",
        "precondition",
        "postcondition",
        "sanitization",
        "sanitize",
    ),
    "configuration": (
        "configuration",
        "config",
        "setting",
        "settings",
        "property",
        "properties",
        "option",
        "parameter",
    ),
    "authentication": (
        "authentication",
        "authenticate",
        "auth",
        "login",
        "credential",
        "credentials",
        "token",
        "session",
    ),
    "authorization": (
        "authorization",
        "authorize",
        "permission",
        "role",
        "access",
        "privilege",
        "policy",
    ),
    "exception_handling": (
        "exception",
        "error",
        "failure",
        "fault",
        "try",
        "catch",
        "recover",
        "fallback",
        "rollback",
    ),
    "logging_auditing": (
        "logging",
        "log",
        "audit",
        "auditing",
        "trace",
        "monitor",
        "record",
    ),
    "resource_management": (
        "resource",
        "memory",
        "file",
        "connection",
        "close",
        "release",
        "pool",
        "limit",
        "quota",
    ),
    "concurrency": (
        "concurrency",
        "concurrent",
        "thread",
        "lock",
        "synchronize",
        "synchronized",
        "mutex",
        "atomic",
        "parallel",
    ),
    "reliability": (
        "reliability",
        "reliable",
        "availability",
        "available",
        "retry",
        "timeout",
        "fault",
        "recover",
        "resilient",
    ),
    "performance": (
        "performance",
        "latency",
        "throughput",
        "response",
        "cache",
        "efficient",
        "optimize",
        "scalable",
    ),
}

EVIDENCE_CUE_TERMS: tuple[str, ...] = tuple(
    dict.fromkeys(
        term
        for aliases in EVIDENCE_CUE_ALIASES.values()
        for term in aliases
    )
)


def normalize_evidence_cues(*values: object) -> tuple[str, ...]:
    """Return canonical evidence-cue labels detected in free text or lists."""
    tokens: set[str] = set()
    for value in values:
        tokens.update(_tokens_from_value(value))
    cues = [
        cue
        for cue, aliases in EVIDENCE_CUE_ALIASES.items()
        if tokens & {alias.casefold() for alias in aliases}
    ]
    return tuple(cues)


def expand_evidence_cues(cues: Iterable[str]) -> tuple[str, ...]:
    """Expand canonical cue names into their aliases for retrieval/scoring."""
    expanded: list[str] = []
    for cue in cues:
        normalized = str(cue).strip().casefold()
        expanded.extend(EVIDENCE_CUE_ALIASES.get(normalized, (normalized,)))
    return tuple(dict.fromkeys(item for item in expanded if item))


def evidence_cue_text(cues: Iterable[str]) -> str:
    """Build a compact evidence-cue text representation."""
    return " ".join(expand_evidence_cues(cues))


def _tokens_from_value(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.casefold() for item in _TOKEN.findall(value)}
    if isinstance(value, Iterable):
        tokens: set[str] = set()
        for item in value:
            tokens.update(_tokens_from_value(item))
        return tokens
    return {item.casefold() for item in _TOKEN.findall(str(value))}
