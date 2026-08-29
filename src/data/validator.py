"""Structural validation for curated traceability datasets.

Method-independent structural checks on an already-loaded ``TraceDataset``:
requirement/code file existence, positive-link ID resolution,
``Is_No_Code`` consistency, and FR/NFR/Mixed and ``NFR_Type`` label
well-formedness (including noncanonical source labels that were normalized
in memory by the loader). Used by ``scripts/validate_data.py``. A failure
reported here indicates a data-curation problem in the CSVs/text files
themselves, not a bug in any retriever, baseline, or the Proposed
Framework.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .loader import TraceDataset
from ..utils import VALID_TYPES


@dataclass(frozen=True)
class ValidationIssue:
    level: Literal["error", "warning"]
    check: str
    message: str


@dataclass
class ValidationSummary:
    dataset: str
    requirements: int
    code_files: int
    positive_links: int
    no_code_requirements: int
    candidate_pairs: int
    errors: int
    warnings: int
    issues: list[ValidationIssue]

    @property
    def valid(self) -> bool:
        return self.errors == 0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["valid"] = self.valid
        return payload


def validate_dataset(dataset: TraceDataset) -> ValidationSummary:
    issues: list[ValidationIssue] = []
    frame = dataset.requirements_frame

    duplicated = frame.loc[frame["Req_ID"].duplicated(keep=False), "Req_ID"].tolist()
    if duplicated:
        issues.append(
            ValidationIssue(
                "error",
                "unique_req_id",
                f"Duplicated Req_ID values: {sorted(set(duplicated))}",
            )
        )

    missing_requirement_files = [
        requirement.req_id
        for requirement in dataset.requirements.values()
        if not requirement.path.is_file()
    ]
    if missing_requirement_files:
        issues.append(
            ValidationIssue(
                "error",
                "requirement_file_exists",
                f"Missing requirement files: {missing_requirement_files}",
            )
        )

    unknown_link_requirements = sorted(
        {link.req_id for link in dataset.trace_links} - set(dataset.requirements)
    )
    if unknown_link_requirements:
        issues.append(
            ValidationIssue(
                "error",
                "trace_req_id_exists",
                f"Links reference unknown requirements: {unknown_link_requirements}",
            )
        )

    unresolved_code_files = sorted(
        {
            link.raw_code_id
            for link in dataset.trace_links
            if link.code_id not in dataset.code_files
        }
    )
    if unresolved_code_files:
        issues.append(
            ValidationIssue(
                "error",
                "trace_code_file_exists",
                f"Links reference missing or ambiguous code files: {unresolved_code_files}",
            )
        )

    linked_requirements = {link.req_id for link in dataset.trace_links}
    marked_no_code_with_links = sorted(
        requirement.req_id
        for requirement in dataset.requirements.values()
        if requirement.is_no_code and requirement.req_id in linked_requirements
    )
    marked_code_without_links = sorted(
        requirement.req_id
        for requirement in dataset.requirements.values()
        if not requirement.is_no_code and requirement.req_id not in linked_requirements
    )
    if marked_no_code_with_links:
        issues.append(
            ValidationIssue(
                "error",
                "no_code_consistency",
                f"Is_No_Code=True requirements have positive links: {marked_no_code_with_links}",
            )
        )
    if marked_code_without_links:
        issues.append(
            ValidationIssue(
                "error",
                "no_code_consistency",
                f"Is_No_Code=False requirements have no positive links: {marked_code_without_links}",
            )
        )

    invalid_types = sorted(
        {
            requirement.raw_type
            for requirement in dataset.requirements.values()
            if requirement.raw_type not in VALID_TYPES
        }
    )
    if invalid_types:
        issues.append(
            ValidationIssue(
                "warning",
                "canonical_type_labels",
                "Noncanonical Type values were normalized in memory: "
                f"{invalid_types}. Expected {list(VALID_TYPES)}.",
            )
        )

    missing_nfr_type = sorted(
        requirement.req_id
        for requirement in dataset.requirements.values()
        if requirement.req_type in {"NFR", "Mixed"}
        and requirement.nfr_type.strip().casefold() in {"", "-", "none", "nan"}
    )
    if missing_nfr_type:
        issues.append(
            ValidationIssue(
                "error",
                "nfr_type_present",
                f"NFR/Mixed requirements without NFR_Type: {missing_nfr_type}",
            )
        )

    errors = sum(issue.level == "error" for issue in issues)
    warnings = sum(issue.level == "warning" for issue in issues)
    return ValidationSummary(
        dataset=dataset.name,
        requirements=len(dataset.requirements),
        code_files=len(dataset.code_files),
        positive_links=len(dataset.trace_links),
        no_code_requirements=sum(
            requirement.is_no_code for requirement in dataset.requirements.values()
        ),
        candidate_pairs=len(dataset.evaluation_requirements) * len(dataset.code_files),
        errors=errors,
        warnings=warnings,
        issues=issues,
    )
