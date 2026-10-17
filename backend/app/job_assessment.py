import hashlib
import re
from typing import Annotated
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from app.evidence_extraction import (
    EvidenceExtractionRequest,
    EvidenceExtractionResponse,
    GeneratedEvidenceResponse,
    RequirementInput,
    SourcePassageInput,
    validate_generated_evidence,
)
from app.semantic_comparison import SemanticComparisonResponse
from app.skill_extraction import SKILL_ALIASES, SkillExtractionResult

SkillName = Annotated[str, StringConstraints(min_length=1, max_length=100)]


class JobRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=500)
    importance: Literal["required", "preferred"] = "required"

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Requirement must not be blank.")
        return normalized

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", normalized):
            raise ValueError("Requirement ID contains unsupported characters.")
        return normalized

    def stable_id(self) -> str:
        if self.id is not None:
            return self.id
        slug = re.sub(r"[^a-z0-9]+", "-", self.text.casefold()).strip("-")[:32]
        digest = hashlib.sha256(
            f"{self.importance}:{self.text.casefold()}".encode("utf-8")
        ).hexdigest()[:10]
        return f"req-{slug or 'requirement'}-{digest}"


class JobDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=5000)
    required_skills: list[SkillName] = Field(default_factory=list, max_length=30)
    preferred_skills: list[SkillName] = Field(default_factory=list, max_length=30)
    requirements: list[JobRequirement] = Field(default_factory=list, max_length=10)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Job title must not be blank.")
        return normalized

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        return value.strip()

    @field_validator("required_skills", "preferred_skills")
    @classmethod
    def normalize_skills(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for skill in values:
            cleaned = " ".join(skill.split())
            if not cleaned:
                raise ValueError("Skills must not be blank.")
            key = cleaned.casefold()
            if key not in seen:
                normalized.append(cleaned)
                seen.add(key)
        return normalized

    @field_validator("requirements")
    @classmethod
    def ensure_stable_unique_requirements(
        cls,
        requirements: list[JobRequirement],
    ) -> list[JobRequirement]:
        normalized_texts = [item.text.casefold() for item in requirements]
        if len(normalized_texts) != len(set(normalized_texts)):
            raise ValueError("Explicit requirements must be unique.")
        identifiers = [item.stable_id() for item in requirements]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Explicit requirement IDs must be unique.")
        return [
            item.model_copy(update={"id": item.stable_id()})
            for item in requirements
        ]


class AssessmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1, max_length=64)
    job: JobDefinition
    skills: SkillExtractionResult
    semantic: SemanticComparisonResponse | None = None
    evidence: EvidenceExtractionResponse | None = None
    source_passages: list[SourcePassageInput] = Field(default_factory=list, max_length=20)
    evidence_error: str | None = Field(default=None, max_length=250)


class RequirementAssessment(BaseModel):
    requirement_id: str
    requirement: str
    status: Literal["supported", "explicitly_contradicted", "unclear"]
    evidence: list[dict[str, str]]
    similarity: float | None


class ComponentScore(BaseModel):
    score: float
    numerator: float
    denominator: float


class CandidateAssessment(BaseModel):
    candidate_id: str
    complete: bool
    incomplete_reason: str | None
    skill_coverage: ComponentScore | None
    semantic_relevance: ComponentScore | None
    evidence_relevance: ComponentScore | None
    overall_score: float | None
    arithmetic: str | None
    requirement_assessments: list[RequirementAssessment]
    matched_skills: list[str]
    rank: int | None = None
    evidence_method: Literal[
        "Model-generated assessment — review required",
        "Local dictionary check — review required",
    ] | None = None


class ScoringError(ValueError):
    pass


_SKILL_ALIASES = {
    name.casefold(): tuple(alias.casefold() for alias in aliases)
    for name, aliases in SKILL_ALIASES
}


def _skill_matches(required: str, extracted: str) -> bool:
    required_key = required.casefold().strip()
    extracted_key = extracted.casefold().strip()
    required_aliases = {required_key, *_SKILL_ALIASES.get(required_key, ())}
    extracted_aliases = {extracted_key, *_SKILL_ALIASES.get(extracted_key, ())}
    if required_aliases.intersection(extracted_aliases):
        return True
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", required_key)
        for alias in extracted_aliases
    )


def _rounded(value: float) -> float:
    return round(value + 1e-10, 2)


def _weighted_score(values: list[tuple[float, float]]) -> ComponentScore:
    numerator = sum(score * weight for score, weight in values)
    denominator = sum(weight for _, weight in values)
    score = 100.0 if denominator == 0 else numerator / denominator
    return ComponentScore(
        score=_rounded(score),
        numerator=_rounded(numerator),
        denominator=_rounded(denominator),
    )


def score_candidate(inputs: AssessmentInput) -> CandidateAssessment:
    job = inputs.job
    positive_skills = {
        skill.canonical_name
        for skill in inputs.skills.mentioned_skills
        if skill.label == "Skill mentioned"
    }
    explicitly_negated = {
        skill.canonical_name for skill in inputs.skills.negated_skills
    }
    skill_items = [
        (
            skill,
            2.0,
            any(_skill_matches(skill, candidate) for candidate in positive_skills),
        )
        for skill in job.required_skills
    ] + [
        (
            skill,
            1.0,
            any(_skill_matches(skill, candidate) for candidate in positive_skills),
        )
        for skill in job.preferred_skills
        if not any(
            skill.casefold() == required.casefold()
            for required in job.required_skills
        )
    ]
    skill_component = _weighted_score(
        [(100.0 if matched else 0.0, weight) for _, weight, matched in skill_items]
    )
    matched_skills = [
        skill
        for skill, _, matched in skill_items
        if matched and not any(_skill_matches(skill, negated) for negated in explicitly_negated)
    ]
    if explicitly_negated:
        skill_component = _weighted_score(
            [
                (
                    0.0
                    if any(_skill_matches(skill, negated) for negated in explicitly_negated)
                    else 100.0 if matched else 0.0,
                    weight,
                )
                for skill, weight, matched in skill_items
            ]
        )

    if (
        not job.required_skills
        and not job.preferred_skills
        and not job.requirements
    ):
        return CandidateAssessment(
            candidate_id=inputs.candidate_id,
            complete=False,
            incomplete_reason="No job criteria were provided.",
            skill_coverage=skill_component,
            semantic_relevance=None,
            evidence_relevance=None,
            overall_score=None,
            arithmetic=None,
            requirement_assessments=[],
            matched_skills=matched_skills,
        )

    if job.requirements and (inputs.evidence is None or inputs.evidence_error):
        return CandidateAssessment(
            candidate_id=inputs.candidate_id,
            complete=False,
            incomplete_reason=inputs.evidence_error or "Evidence analysis is unavailable.",
            skill_coverage=skill_component,
            semantic_relevance=None,
            evidence_relevance=None,
            overall_score=None,
            arithmetic=None,
            requirement_assessments=[],
            matched_skills=matched_skills,
        )

    if job.requirements and inputs.semantic is None:
        return CandidateAssessment(
            candidate_id=inputs.candidate_id,
            complete=False,
            incomplete_reason="Semantic comparison is unavailable.",
            skill_coverage=skill_component,
            semantic_relevance=None,
            evidence_relevance=None,
            overall_score=None,
            arithmetic=None,
            requirement_assessments=[],
            matched_skills=matched_skills,
        )

    if job.requirements:
        if not inputs.source_passages:
            return CandidateAssessment(
                candidate_id=inputs.candidate_id,
                complete=False,
                incomplete_reason="Original source passages are unavailable.",
                skill_coverage=skill_component,
                semantic_relevance=None,
                evidence_relevance=None,
                overall_score=None,
                arithmetic=None,
                requirement_assessments=[],
                matched_skills=matched_skills,
            )
        evidence_request = EvidenceExtractionRequest(
            confirmed_fictional_demo=True,
            requirements=[
                RequirementInput(id=item.id or "", text=item.text)
                for item in job.requirements
            ],
            source_passages=inputs.source_passages,
        )
        validate_generated_evidence(
            GeneratedEvidenceResponse(results=inputs.evidence.results),
            evidence_request,
        )

    evidence_by_id = (
        {
            result.requirement_id: result
            for result in inputs.evidence.results
        }
        if inputs.evidence is not None
        else {}
    )
    semantic_by_text = {
        comparison.requirement: comparison
        for comparison in (inputs.semantic.comparisons if inputs.semantic else [])
    }
    required_texts = {requirement.text for requirement in job.requirements}
    if job.requirements and (
        len(semantic_by_text) != len(job.requirements)
        or set(semantic_by_text) != required_texts
    ):
        return CandidateAssessment(
            candidate_id=inputs.candidate_id,
            complete=False,
            incomplete_reason="Semantic comparison is missing or mismatched.",
            skill_coverage=skill_component,
            semantic_relevance=None,
            evidence_relevance=None,
            overall_score=None,
            arithmetic=None,
            requirement_assessments=[],
            matched_skills=matched_skills,
        )
    requirement_scores: list[tuple[float, float]] = []
    semantic_scores: list[tuple[float, float]] = []
    requirement_assessments: list[RequirementAssessment] = []
    for requirement in job.requirements:
        evidence = evidence_by_id.get(requirement.id or "")
        if evidence is None:
            return CandidateAssessment(
                candidate_id=inputs.candidate_id,
                complete=False,
                incomplete_reason="Evidence is missing a job requirement.",
                skill_coverage=skill_component,
                semantic_relevance=None,
                evidence_relevance=None,
                overall_score=None,
                arithmetic=None,
                requirement_assessments=requirement_assessments,
                matched_skills=matched_skills,
            )
        weight = 2.0 if requirement.importance == "required" else 1.0
        evidence_score = {
            "supported": 100.0,
            "explicitly_contradicted": 0.0,
            "unclear": 50.0,
        }[evidence.status]
        requirement_scores.append((evidence_score, weight))
        semantic_match = semantic_by_text.get(requirement.text)
        similarity = None
        if semantic_match and semantic_match.matches:
            similarity = max(
                0.0,
                min(1.0, max(match.similarity for match in semantic_match.matches)),
            )
        semantic_score = (
            0.0
            if evidence.status == "explicitly_contradicted"
            else (similarity * 100.0 if similarity is not None else 0.0)
        )
        semantic_scores.append((semantic_score, weight))
        requirement_assessments.append(
            RequirementAssessment(
                requirement_id=requirement.id or "",
                requirement=requirement.text,
                status=evidence.status,
                evidence=[
                    {
                        "source_id": quote.source_id,
                        "relationship": quote.relationship.value,
                        "quote": quote.quote,
                    }
                    for quote in evidence.evidence
                ],
                similarity=(
                    _rounded(similarity * 100.0)
                    if similarity is not None
                    else None
                ),
            )
        )

    semantic_component = _weighted_score(semantic_scores)
    evidence_component = _weighted_score(requirement_scores)
    overall = _rounded(
        0.5 * skill_component.score
        + 0.3 * semantic_component.score
        + 0.2 * evidence_component.score
    )
    arithmetic = (
        f"({skill_component.score:.2f} × 0.50) + "
        f"({semantic_component.score:.2f} × 0.30) + "
        f"({evidence_component.score:.2f} × 0.20) = {overall:.2f}"
    )
    return CandidateAssessment(
        candidate_id=inputs.candidate_id,
        complete=True,
        incomplete_reason=None,
        skill_coverage=skill_component,
        semantic_relevance=semantic_component,
        evidence_relevance=evidence_component,
        overall_score=overall,
        arithmetic=arithmetic,
        requirement_assessments=requirement_assessments,
        matched_skills=matched_skills,
        evidence_method=inputs.evidence.label if inputs.evidence else None,
    )


def rank_assessments(
    assessments: list[CandidateAssessment],
) -> list[CandidateAssessment]:
    ordered = sorted(
        enumerate(assessments),
        key=lambda pair: (
            not pair[1].complete,
            -(pair[1].overall_score or 0.0) if pair[1].complete else 0.0,
            pair[0],
        ),
    )
    ranks: list[CandidateAssessment] = []
    current_rank = 0
    previous_score: float | None = None
    for position, (_, assessment) in enumerate(ordered, start=1):
        if not assessment.complete or assessment.overall_score is None:
            ranks.append(assessment.model_copy(update={"rank": None}))
            continue
        if assessment.overall_score != previous_score:
            current_rank = position
            previous_score = assessment.overall_score
        ranks.append(assessment.model_copy(update={"rank": current_rank}))
    return ranks
