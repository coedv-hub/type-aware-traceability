"""Load the five curated IST traceability datasets."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterator

import pandas as pd

from ..utils import (
    canonical_requirement_type,
    load_config,
    normalize_artifact_id,
    parse_bool,
    read_csv_robust,
    read_text_robust,
)


@dataclass(frozen=True)
class Requirement:
    req_id: str
    req_type: str
    raw_type: str
    nfr_type: str
    notes: str
    requirement_file: str
    path: Path
    text: str
    is_no_code: bool


@dataclass(frozen=True)
class CodeArtifact:
    code_id: str
    path: Path
    text: str


@dataclass(frozen=True)
class TraceLink:
    req_id: str
    code_id: str
    raw_code_id: str


@dataclass
class TraceDataset:
    name: str
    scope: str
    root: Path
    requirement_dir: Path
    code_dir: Path
    requirements_frame: pd.DataFrame
    links_frame: pd.DataFrame
    requirements: dict[str, Requirement]
    code_files: dict[str, CodeArtifact]
    trace_links: list[TraceLink]
    requirements_encoding: str
    links_encoding: str

    @property
    def positive_links(self) -> set[tuple[str, str]]:
        return {(link.req_id, link.code_id) for link in self.trace_links}

    @property
    def evaluation_requirements(self) -> dict[str, Requirement]:
        return {
            req_id: requirement
            for req_id, requirement in self.requirements.items()
            if not requirement.is_no_code
        }

    @property
    def ground_truth_by_requirement(self) -> dict[str, set[str]]:
        output: dict[str, set[str]] = {
            req_id: set() for req_id in self.evaluation_requirements
        }
        for req_id, code_id in self.positive_links:
            if req_id in output:
                output[req_id].add(code_id)
        return output

    def iter_candidate_pairs(
        self, evaluation_only: bool = True
    ) -> Iterator[tuple[str, str, int]]:
        """Yield every requirement-code pair with its binary gold label."""
        requirements = (
            self.evaluation_requirements if evaluation_only else self.requirements
        )
        positives = self.positive_links
        for req_id, code_id in product(requirements, self.code_files):
            yield req_id, code_id, int((req_id, code_id) in positives)

class DatasetLoader:
    """Configuration-driven loader for the curated benchmark."""

    def __init__(self, config_path: str | Path | None = None):
        self.config = load_config(config_path)
        self.data_root: Path = self.config["_data_root"]
        self.dataset_configs: dict = self.config["datasets"]
        self.requirements_filename = self.config["files"]["requirements"]
        self.links_filename = self.config["files"]["trace_links"]
        self.code_extensions = {
            extension.casefold() for extension in self.config["code_extensions"]
        }

    @property
    def dataset_names(self) -> list[str]:
        return list(self.dataset_configs)

    def _resolve_subdirectory(self, root: Path, candidates: list[str]) -> Path:
        for candidate in candidates:
            path = root / candidate
            if path.is_dir():
                return path
        raise FileNotFoundError(
            f"None of the configured directories exist under {root}: {candidates}"
        )

    def load(self, dataset_name: str) -> TraceDataset:
        if dataset_name not in self.dataset_configs:
            raise KeyError(
                f"Unknown dataset {dataset_name!r}; choose from {self.dataset_names}"
            )
        item = self.dataset_configs[dataset_name]
        root = self.data_root / item["path"]
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        requirement_dir = self._resolve_subdirectory(
            root, item["requirement_dirs"]
        )
        code_dir = self._resolve_subdirectory(root, item["code_dirs"])
        requirements_frame, requirements_encoding = read_csv_robust(
            root / self.requirements_filename
        )
        links_frame, links_encoding = read_csv_robust(root / self.links_filename)

        requirements = self._load_requirements(root, requirements_frame)
        code_files = self._load_code_files(code_dir)
        trace_links = self._load_links(links_frame, code_files)

        return TraceDataset(
            name=dataset_name,
            scope=item["scope"],
            root=root,
            requirement_dir=requirement_dir,
            code_dir=code_dir,
            requirements_frame=requirements_frame,
            links_frame=links_frame,
            requirements=requirements,
            code_files=code_files,
            trace_links=trace_links,
            requirements_encoding=requirements_encoding,
            links_encoding=links_encoding,
        )

    def _load_requirements(
        self, root: Path, frame: pd.DataFrame
    ) -> dict[str, Requirement]:
        required = {
            "Req_ID",
            "Type",
            "NFR_Type",
            "Notes",
            "Requirement_File",
            "Is_No_Code",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{root.name}: missing requirement columns {sorted(missing)}")

        requirements: dict[str, Requirement] = {}
        for row in frame.itertuples(index=False):
            req_id = str(row.Req_ID).strip()
            relative_file = normalize_artifact_id(row.Requirement_File)
            path = root / relative_file if relative_file else root / "__missing__"
            text = read_text_robust(path).strip() if path.is_file() else ""
            raw_type = str(row.Type).strip()
            requirements[req_id] = Requirement(
                req_id=req_id,
                req_type=canonical_requirement_type(raw_type),
                raw_type=raw_type,
                nfr_type=str(row.NFR_Type).strip(),
                notes=str(row.Notes).strip(),
                requirement_file=relative_file,
                path=path,
                text=text,
                is_no_code=parse_bool(row.Is_No_Code),
            )
        return requirements

    def _load_code_files(self, code_dir: Path) -> dict[str, CodeArtifact]:
        artifacts: dict[str, CodeArtifact] = {}
        for path in sorted(code_dir.rglob("*")):
            if (
                not path.is_file()
                or path.suffix.casefold() not in self.code_extensions
                or any(part.startswith(".") for part in path.relative_to(code_dir).parts)
                or path.name.endswith(("~", ".tmp", ".temp", ".swp"))
            ):
                continue
            code_id = path.relative_to(code_dir).as_posix()
            artifacts[code_id] = CodeArtifact(
                code_id=code_id,
                path=path,
                text=read_text_robust(path),
            )
        return artifacts

    def _resolve_code_id(
        self, raw_code_id: str, code_files: dict[str, CodeArtifact]
    ) -> str:
        normalized = normalize_artifact_id(raw_code_id)
        exact = {code_id.casefold(): code_id for code_id in code_files}
        if normalized.casefold() in exact:
            return exact[normalized.casefold()]

        basename_matches = [
            code_id
            for code_id in code_files
            if Path(code_id).name.casefold() == Path(normalized).name.casefold()
        ]
        return basename_matches[0] if len(basename_matches) == 1 else normalized

    def _load_links(
        self, frame: pd.DataFrame, code_files: dict[str, CodeArtifact]
    ) -> list[TraceLink]:
        required = {"Req_ID", "Code_File"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing trace-link columns: {sorted(missing)}")

        if "Label" in frame.columns:
            positive = frame[
                frame["Label"].astype(str).str.strip().str.casefold().eq("positive")
            ]
        else:
            positive = frame

        links: list[TraceLink] = []
        seen: set[tuple[str, str]] = set()
        for row in positive.itertuples(index=False):
            req_id = str(row.Req_ID).strip()
            raw_code_id = normalize_artifact_id(row.Code_File)
            code_id = self._resolve_code_id(raw_code_id, code_files)
            pair = (req_id, code_id)
            if not req_id or not raw_code_id or pair in seen:
                continue
            seen.add(pair)
            links.append(
                TraceLink(
                    req_id=req_id,
                    code_id=code_id,
                    raw_code_id=raw_code_id,
                )
            )
        return links
