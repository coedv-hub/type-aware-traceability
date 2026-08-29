"""Stable, intentionally simple prompts for the Stage 4a baselines.

Builds the prompts for the two Stage 4a LLM baselines: Direct Prompting
(``direct_prompt``) and Generic RAG-LLM (``rag_prompt``, which additionally
surfaces the selected candidate's retrieval rank/score as supporting
metadata). Both request the same minimal JSON schema
(``prediction``/``confidence``/``explanation``) with no multi-stage
reasoning scaffold, so these baselines represent a naive "ask an LLM
directly" comparison point -- distinct from the Proposed Framework's
structured, multi-stage prompting in ``src/framework/``.
"""

from __future__ import annotations


OUTPUT_INSTRUCTIONS = """Return only one JSON object with this schema:
{"prediction": "Yes" or "No", "confidence": number from 0 to 1 or null,
 "explanation": "brief reason"}
Decide whether the candidate code file implements or is needed by the requirement.
Do not use knowledge outside the supplied requirement and code."""


def direct_prompt(requirement: str, code_file: str, code: str) -> str:
    return f"""You are performing software requirement-to-code traceability.
{OUTPUT_INSTRUCTIONS}

Requirement:
<requirement>
{requirement}
</requirement>

Candidate code file: {code_file}
<code>
{code}
</code>
"""


def rag_prompt(
    requirement: str,
    code_file: str,
    code: str,
    retrieval_rank: int,
    retrieval_score: float,
) -> str:
    return f"""You are performing software requirement-to-code traceability.
The candidate code file was selected from the ranked candidate list produced
by the retrieval module based on semantic similarity. Retrieval rank and score
are supporting metadata, not a gold label.
{OUTPUT_INSTRUCTIONS}

Requirement:
<requirement>
{requirement}
</requirement>

Retrieved candidate:
- file: {code_file}
- retrieval rank: {retrieval_rank}
- retrieval score: {retrieval_score:.8f}
<code>
{code}
</code>
"""
