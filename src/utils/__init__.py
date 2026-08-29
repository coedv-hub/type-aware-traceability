"""Shared configuration, text, and file utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


CSV_ENCODINGS = ("utf-8-sig", "gb18030", "utf-8", "latin1")
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "latin1")
VALID_TYPES = ("FR", "NFR", "Mixed")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and attach resolved project paths."""
    root = project_root()
    path = Path(config_path) if config_path else root / "configs" / "datasets.yaml"
    if not path.is_absolute():
        path = (root / path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_project_root"] = root
    config["_config_path"] = path
    data_root = Path(config["project"]["data_root"])
    config["_data_root"] = (
        data_root if data_root.is_absolute() else (root / data_root).resolve()
    )
    return config


def read_csv_robust(path: Path) -> tuple[pd.DataFrame, str]:
    """Read a CSV using the encodings used by the curated datasets."""
    errors: list[str] = []
    for encoding in CSV_ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding, dtype=str).fillna("")
            unnamed = [
                column
                for column in frame.columns
                if str(column).startswith("Unnamed") and frame[column].eq("").all()
            ]
            return frame.drop(columns=unnamed), encoding
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"Could not read {path}: {'; '.join(errors)}")


def read_text_robust(path: Path) -> str:
    """Read a text artifact without silently dropping undecodable bytes."""
    errors: list[str] = []
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"Could not read {path}: {'; '.join(errors)}")


def parse_bool(value: object) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def canonical_requirement_type(value: object) -> str:
    """Canonicalize case only; unknown labels remain visible to validation."""
    normalized = str(value).strip()
    lookup = {label.casefold(): label for label in VALID_TYPES}
    return lookup.get(normalized.casefold(), normalized)


def normalize_artifact_id(value: object) -> str:
    return str(value).strip().replace("\\", "/").lstrip("./")


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    """Tokenize prose and source identifiers for classical IR."""
    expanded = _CAMEL_BOUNDARY.sub(" ", text.replace("_", " "))
    return [token.casefold() for token in _TOKEN.findall(expanded)]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
