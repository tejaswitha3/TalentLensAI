"""Conservative, offline citation checks for simple skill requirements.

This deliberately cannot infer experience from semantic similarity or from a
bare skill mention. Anything outside the small explicit pattern set is unclear.
"""

import re

from app.evidence_extraction import (
    EvidenceExtractionRequest,
    EvidenceExtractionResponse,
    EvidenceQuote,
    EvidenceRelationship,
    GeneratedEvidenceResponse,
    NO_EVIDENCE_EXPLANATION,
    RequirementEvidence,
    validate_generated_evidence,
)
from app.skill_extraction import extract_skills

LOCAL_LABEL = "Local dictionary check — review required"
_SIMPLE_REQUIREMENT = re.compile(
    r"^(?:hands-on\s+)?(?:experience\s+(?:with|in)\s+)?"
    r"(?P<skill>.+?)"
    r"(?:\s+(?:hands-on\s+)?(?:experience|development|proficiency))?$",
    re.IGNORECASE,
)
_ACTION = re.compile(
    r"\b(?:built|developed|implemented|created|deployed|used|using|"
    r"worked with|hands-on|experience with|experience in|"
    r"experience using|project|projects)\b",
    re.IGNORECASE,
)
_UNCERTAIN = re.compile(
    r"\b(?:want to|hope to|plan to|learning|studying|interested in|"
    r"seeking|aspiring|not yet|without|no experience|never used)\b",
    re.IGNORECASE,
)


def _single_skill(text: str) -> str | None:
    match = _SIMPLE_REQUIREMENT.fullmatch(text.strip())
    if match is None:
        return None
    skill_text = match.group("skill")
    found = extract_skills(skill_text).mentioned_skills
    if len(found) != 1:
        return None
    mention = found[0].mentions[0]
    if mention.matched_wording.casefold() != skill_text.casefold():
        return None
    return found[0].canonical_name


def check_local_evidence(request: EvidenceExtractionRequest) -> EvidenceExtractionResponse:
    results: list[RequirementEvidence] = []
    for requirement in request.requirements:
        canonical = _single_skill(requirement.text)
        positive: EvidenceQuote | None = None
        negative: EvidenceQuote | None = None
        context: EvidenceQuote | None = None
        if canonical:
            for source in request.source_passages:
                found = extract_skills(source.text)
                for group in found.mentioned_skills:
                    if group.canonical_name != canonical:
                        continue
                    for mention in group.mentions:
                        if len(mention.source_excerpt) > 800:
                            continue
                        quote = EvidenceQuote(
                            source_id=source.id,
                            relationship=EvidenceRelationship.CONTEXT,
                            quote=mention.source_excerpt,
                        )
                        context = context or quote
                        if _ACTION.search(quote.quote) and not _UNCERTAIN.search(quote.quote):
                            positive = positive or quote.model_copy(
                                update={"relationship": EvidenceRelationship.SUPPORTS}
                            )
                for group in found.negated_skills:
                    if group.canonical_name == canonical:
                        mention = group.mentions[0]
                        if len(mention.source_excerpt) > 800:
                            continue
                        negative = negative or EvidenceQuote(
                            source_id=source.id,
                            relationship=EvidenceRelationship.CONTRADICTS,
                            quote=mention.source_excerpt,
                        )

        if positive and not negative:
            status = "supported"
            evidence = [positive]
            explanation = "A local keyword and action pattern found this exact excerpt: " + positive.quote
        elif negative and not positive:
            status = "explicitly_contradicted"
            evidence = [negative]
            explanation = "A local explicit-negation pattern found this exact excerpt: " + negative.quote
        else:
            status = "unclear"
            evidence = [context] if context and not negative and not positive else []
            explanation = (
                "The local rules found this mention but cannot establish the requirement. Exact source: "
                + context.quote
                if evidence else NO_EVIDENCE_EXPLANATION
            )
        results.append(
            RequirementEvidence(
                requirement_id=requirement.id,
                status=status,
                explanation=explanation[:1000],
                evidence=evidence,
            )
        )
    checked = validate_generated_evidence(
        GeneratedEvidenceResponse(results=results), request
    )
    return checked.model_copy(update={"label": LOCAL_LABEL})
