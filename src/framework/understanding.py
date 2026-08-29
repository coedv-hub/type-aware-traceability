"""Type-aware requirement understanding for the IST framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .evidence_cues import normalize_evidence_cues
from .llm_support import FrameworkLLMService


RequirementType = Literal["FR", "NFR", "Mixed"]


@dataclass(frozen=True)
class TypeAwareRequirementProfile:
    """Common fields shared by all requirement profiles."""

    requirement_id: str
    requirement_type: RequirementType
    raw_text: str
    prompt: str = ""
    placeholder: bool = False


@dataclass(frozen=True)
class FRRequirementProfile(TypeAwareRequirementProfile):
    """Functional requirement profile from the IST methodology."""

    intent: str = ""
    action: str = ""
    object: str = ""
    constraint: str = ""
    entities: tuple[str, ...] = ()
    expected_behavior: str = ""


@dataclass(frozen=True)
class NFRRequirementProfile(TypeAwareRequirementProfile):
    """Non-functional requirement profile from the IST methodology."""

    quality_attribute: str = ""
    quality_metric: str = ""
    scope: str = ""
    constraint: str = ""
    context: str = ""
    priority: str = ""
    evaluation_condition: str = ""
    evidence_cues: tuple[str, ...] = ()


@dataclass(frozen=True)
class MixedRequirementProfile(TypeAwareRequirementProfile):
    """Mixed requirement profile with functional and quality subprofiles."""

    functional_profile: FRRequirementProfile = field(
        default_factory=lambda: FRRequirementProfile(
            requirement_id="",
            requirement_type="FR",
            raw_text="",
        )
    )
    quality_profile: NFRRequirementProfile = field(
        default_factory=lambda: NFRRequirementProfile(
            requirement_id="",
            requirement_type="NFR",
            raw_text="",
        )
    )

    @property
    def intent(self) -> str:
        return self.functional_profile.intent

    @property
    def action(self) -> str:
        return self.functional_profile.action

    @property
    def object(self) -> str:
        return self.functional_profile.object

    @property
    def constraint(self) -> str:
        return self.functional_profile.constraint

    @property
    def entities(self) -> tuple[str, ...]:
        return self.functional_profile.entities

    @property
    def expected_behavior(self) -> str:
        return self.functional_profile.expected_behavior

    @property
    def quality_attribute(self) -> str:
        return self.quality_profile.quality_attribute

    @property
    def quality_metric(self) -> str:
        return self.quality_profile.quality_metric

    @property
    def context(self) -> str:
        return self.quality_profile.context

    @property
    def scope(self) -> str:
        return self.quality_profile.scope

    @property
    def priority(self) -> str:
        return self.quality_profile.priority

    @property
    def evaluation_condition(self) -> str:
        return self.quality_profile.evaluation_condition

    @property
    def evidence_cues(self) -> tuple[str, ...]:
        return self.quality_profile.evidence_cues


RequirementProfile = (
    FRRequirementProfile | NFRRequirementProfile | MixedRequirementProfile
)


class TypeAwareUnderstandingPromptBuilder:
    """Build prompts only; this class never sends requests to an LLM."""

    def build_fr_prompt(self, requirement_text: str) -> str:
        """Build the FR understanding prompt."""
        return f"""IST type-aware understanding: Functional Requirement (FR).
Return only valid JSON.
Requirement: {requirement_text}
Schema:
{{
  "intent": "high-level functional goal",
  "action": "main operation or behavior",
  "object": "main domain object or target",
  "constraint": "condition, precondition, or rule",
  "entities": ["important domain entities"],
  "expected_behavior": "observable behavior expected from code"
}}"""

    def build_nfr_prompt(self, requirement_text: str) -> str:
        """Build the NFR understanding prompt."""
        return f"""IST type-aware understanding: Non-Functional Requirement (NFR).
Return only valid JSON.
Requirement: {requirement_text}
Schema:
{{
  "quality_attribute": "security, performance, reliability, usability, maintainability, or other quality attribute",
  "quality_metric": "explicit or implied metric, threshold, policy, or quality criterion",
  "scope": "affected feature, component, user role, resource, or runtime scope",
  "constraint": "quality constraint, policy, limit, or required mechanism",
  "context": "system scope or operational context",
  "priority": "critical, high, medium, low, or unspecified",
  "evaluation_condition": "condition under which the quality requirement should be checked",
  "evidence_cues": ["validation, configuration, authentication, authorization, exception_handling, logging_auditing, resource_management, concurrency, reliability, or performance cues"]
}}"""

    def build_mixed_prompt(self, requirement_text: str) -> str:
        """Build the Mixed requirement understanding prompt."""
        return f"""IST type-aware understanding: Mixed Requirement.
Separate functional and quality aspects. Return only valid JSON.
Requirement: {requirement_text}
Schema:
{{
  "functional_profile": {{
    "intent": "high-level functional goal",
    "action": "main operation or behavior",
    "object": "main domain object or target",
    "constraint": "functional condition, precondition, or rule",
    "entities": ["important domain entities"],
    "expected_behavior": "observable behavior expected from code"
  }},
    "quality_profile": {{
    "quality_attribute": "security, performance, reliability, usability, maintainability, or other quality attribute",
    "quality_metric": "explicit or implied metric, threshold, policy, or quality criterion",
    "scope": "affected feature, component, user role, resource, or runtime scope",
    "constraint": "quality constraint, policy, limit, or required mechanism",
    "context": "system scope or operational context",
    "priority": "critical, high, medium, low, or unspecified",
    "evaluation_condition": "condition under which the quality requirement should be checked",
    "evidence_cues": ["validation, configuration, authentication, authorization, exception_handling, logging_auditing, resource_management, concurrency, reliability, or performance cues"]
  }}
}}"""

    def build_prompt(self, requirement_text: str, requirement_type: str) -> str:
        """Dispatch to the type-specific prompt builder."""
        normalized_type = TypeAwareRequirementUnderstanding.normalize_type(
            requirement_type
        )
        if normalized_type == "FR":
            return self.build_fr_prompt(requirement_text)
        if normalized_type == "NFR":
            return self.build_nfr_prompt(requirement_text)
        return self.build_mixed_prompt(requirement_text)


class TypeAwareRequirementUnderstanding:
    """Create FR, NFR, or Mixed requirement profiles without API calls."""

    def __init__(
        self,
        prompt_builder: TypeAwareUnderstandingPromptBuilder | None = None,
        llm_service: FrameworkLLMService | None = None,
    ):
        self.prompt_builder = prompt_builder or TypeAwareUnderstandingPromptBuilder()
        self.llm_service = llm_service

    def understand(
        self,
        requirement_text: str,
        requirement_type: str,
        requirement_id: str = "",
    ) -> RequirementProfile:
        """Return a type-specific requirement profile and its prompt."""
        normalized_type = self.normalize_type(requirement_type)
        prompt = self.prompt_builder.build_prompt(
            requirement_text, normalized_type
        )
        payload: dict[str, Any] = {}
        if self.llm_service is not None and self.llm_service.live_api:
            result = self.llm_service.complete_json(
                module="understanding",
                prompt=prompt,
                requirement_id=requirement_id,
            )
            payload = result.payload
        text = requirement_text.strip()
        if normalized_type == "FR":
            return FRRequirementProfile(
                requirement_id=requirement_id,
                requirement_type="FR",
                raw_text=requirement_text,
                prompt=prompt,
                intent=str(payload.get("intent") or text),
                action=str(payload.get("action") or ""),
                object=str(payload.get("object") or ""),
                constraint=str(payload.get("constraint") or ""),
                entities=_tuple_of_strings(payload.get("entities")),
                expected_behavior=str(payload.get("expected_behavior") or text),
                placeholder=False,
            )
        if normalized_type == "NFR":
            evidence_cues = normalize_evidence_cues(
                payload.get("evidence_cues"),
                payload.get("quality_attribute"),
                payload.get("quality_metric"),
                payload.get("scope"),
                payload.get("constraint"),
                payload.get("context"),
                payload.get("evaluation_condition"),
                text,
            )
            return NFRRequirementProfile(
                requirement_id=requirement_id,
                requirement_type="NFR",
                raw_text=requirement_text,
                prompt=prompt,
                quality_attribute=str(payload.get("quality_attribute") or ""),
                quality_metric=str(payload.get("quality_metric") or ""),
                scope=str(payload.get("scope") or ""),
                constraint=str(payload.get("constraint") or ""),
                context=str(payload.get("context") or text),
                priority=str(payload.get("priority") or "unspecified"),
                evaluation_condition=str(payload.get("evaluation_condition") or ""),
                evidence_cues=evidence_cues,
                placeholder=False,
            )
        functional_payload = (
            payload.get("functional_profile")
            if isinstance(payload.get("functional_profile"), dict)
            else {}
        )
        quality_payload = (
            payload.get("quality_profile")
            if isinstance(payload.get("quality_profile"), dict)
            else {}
        )
        functional_profile = FRRequirementProfile(
            requirement_id=requirement_id,
            requirement_type="FR",
            raw_text=requirement_text,
            prompt=self.prompt_builder.build_fr_prompt(requirement_text),
            intent=str(functional_payload.get("intent") or text),
            action=str(functional_payload.get("action") or ""),
            object=str(functional_payload.get("object") or ""),
            constraint=str(functional_payload.get("constraint") or ""),
            entities=_tuple_of_strings(functional_payload.get("entities")),
            expected_behavior=str(functional_payload.get("expected_behavior") or text),
            placeholder=False,
        )
        evidence_cues = normalize_evidence_cues(
            quality_payload.get("evidence_cues"),
            quality_payload.get("quality_attribute"),
            quality_payload.get("quality_metric"),
            quality_payload.get("scope"),
            quality_payload.get("constraint"),
            quality_payload.get("context"),
            quality_payload.get("evaluation_condition"),
            text,
        )
        quality_profile = NFRRequirementProfile(
            requirement_id=requirement_id,
            requirement_type="NFR",
            raw_text=requirement_text,
            prompt=self.prompt_builder.build_nfr_prompt(requirement_text),
            quality_attribute=str(quality_payload.get("quality_attribute") or ""),
            quality_metric=str(quality_payload.get("quality_metric") or ""),
            scope=str(quality_payload.get("scope") or ""),
            constraint=str(quality_payload.get("constraint") or ""),
            context=str(quality_payload.get("context") or text),
            priority=str(quality_payload.get("priority") or "unspecified"),
            evaluation_condition=str(quality_payload.get("evaluation_condition") or ""),
            evidence_cues=evidence_cues,
            placeholder=False,
        )
        return MixedRequirementProfile(
            requirement_id=requirement_id,
            requirement_type="Mixed",
            raw_text=requirement_text,
            prompt=prompt,
            functional_profile=functional_profile,
            quality_profile=quality_profile,
            placeholder=False,
        )

    @staticmethod
    def normalize_type(requirement_type: str) -> RequirementType:
        """Normalize a requirement type label."""
        lookup: dict[str, RequirementType] = {
            "fr": "FR",
            "functional": "FR",
            "functional requirement": "FR",
            "nfr": "NFR",
            "non-functional": "NFR",
            "non functional": "NFR",
            "non-functional requirement": "NFR",
            "non functional requirement": "NFR",
            "mixed": "Mixed",
        }
        try:
            return lookup[requirement_type.strip().casefold()]
        except KeyError as exc:
            raise ValueError(
                "requirement_type must be one of: FR, NFR, Mixed"
            ) from exc

    _normalize_type = normalize_type


FRProfile = FRRequirementProfile
NFRProfile = NFRRequirementProfile
MixedProfile = MixedRequirementProfile


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, tuple):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()
