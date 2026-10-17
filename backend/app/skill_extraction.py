import re
from typing import Literal

from pydantic import BaseModel


class SkillMention(BaseModel):
    matched_wording: str
    source_excerpt: str


class SkillEvidence(BaseModel):
    canonical_name: str
    label: Literal["Skill mentioned", "Negated mention"]
    mentions: list[SkillMention]


class SkillExtractionResult(BaseModel):
    mentioned_skills: list[SkillEvidence]
    negated_skills: list[SkillEvidence]


SKILL_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Python", ("python",)),
    ("Java", ("java",)),
    ("JavaScript", ("javascript", "java script", "js")),
    ("TypeScript", ("typescript", "type script", "ts")),
    ("React", ("react.js", "reactjs", "react")),
    ("Next.js", ("next.js", "nextjs", "next js")),
    ("FastAPI", ("fastapi", "fast api")),
    ("Flask", ("flask",)),
    ("SQL", ("sql",)),
    ("PostgreSQL", ("postgresql", "postgres")),
    ("Git", ("git",)),
    ("Docker", ("docker",)),
    ("pandas", ("pandas",)),
    ("NumPy", ("numpy",)),
    ("scikit-learn", ("scikit-learn", "scikit learn", "sklearn")),
    ("REST APIs", ("restful apis", "restful api", "rest apis", "rest api")),
)

_SKILL_PATTERNS = tuple(
    (
        canonical_name,
        re.compile(
            r"(?<![A-Za-z0-9_])(?:"
            + "|".join(
                re.escape(alias)
                for alias in sorted(aliases, key=len, reverse=True)
            )
            + r")(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    )
    for canonical_name, aliases in SKILL_ALIASES
)

_EXCERPT_BOUNDARY = re.compile(r"[.!?](?=\s)|[\r\n]+")
_SENTENCE_BOUNDARY = re.compile(r"[.!?](?=\s)")
_CLAUSE_BOUNDARY = re.compile(r";|\b(?:but|however|yet)\b", re.IGNORECASE)
_NEGATION_PREFIXES = (
    re.compile(
        r"\bno\s+(?:(?:prior|previous|professional|hands-on)\s+)?"
        r"experience\s+(?:with|in|using)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:(?:prior|previous)\s+)?"
        r"(?:hands-on\s+)?(?:knowledge|proficiency)\s+(?:of|with)\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"\bunfamiliar\s+with\s*$", re.IGNORECASE),
    re.compile(r"\bnot\s+familiar\s+with\s*$", re.IGNORECASE),
    re.compile(r"\bnever\s+(?:used|worked\s+with)\s*$", re.IGNORECASE),
    re.compile(
        r"\b(?:do\s+not|don't|does\s+not|doesn't)\s+"
        r"(?:use|have\s+experience\s+with)\s*$",
        re.IGNORECASE,
    ),
)
_NEGATION_SUFFIXES = (
    re.compile(r"^\s+experience\b", re.IGNORECASE),
    re.compile(
        r"^\s+(?:is\s+)?not\s+(?:a\s+)?"
        r"(?:skill|strength|competency)\b",
        re.IGNORECASE,
    ),
)


def _excerpt_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    for boundary in _EXCERPT_BOUNDARY.finditer(text, 0, start):
        left = boundary.end()

    right = len(text)
    boundary = _EXCERPT_BOUNDARY.search(text, end)
    if boundary is not None:
        right = boundary.end()

    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return left, right


def _negation_prefix_start(
    text: str,
    start: int,
) -> int | None:
    sentence_start = 0
    for boundary in _SENTENCE_BOUNDARY.finditer(text, 0, start):
        sentence_start = boundary.end()

    clause_start = sentence_start
    for boundary in _CLAUSE_BOUNDARY.finditer(text, sentence_start, start):
        clause_start = boundary.end()

    prefix = text[clause_start:start]
    for pattern in _NEGATION_PREFIXES:
        match = pattern.search(prefix)
        if match is not None:
            return clause_start + match.start()
    return None


def _is_negated(text: str, start: int, end: int) -> bool:
    if _negation_prefix_start(text, start) is not None:
        return True

    suffix = text[end : _excerpt_bounds(text, start, end)[1]]
    return any(pattern.match(suffix) for pattern in _NEGATION_SUFFIXES)


def extract_skills(text: str) -> SkillExtractionResult:
    matches: list[tuple[int, int, str, str, str]] = []
    for canonical_name, pattern in _SKILL_PATTERNS:
        for match in pattern.finditer(text):
            excerpt_start, excerpt_end = _excerpt_bounds(
                text,
                match.start(),
                match.end(),
            )
            label = (
                "Negated mention"
                if _is_negated(text, match.start(), match.end())
                else "Skill mentioned"
            )
            matches.append(
                (
                    match.start(),
                    match.end(),
                    canonical_name,
                    match.group(0),
                    label,
                )
            )

    matches.sort(key=lambda match: (match[0], -(match[1] - match[0])))
    non_overlapping_matches: list[tuple[int, int, str, str, str]] = []
    for match in matches:
        if non_overlapping_matches and match[0] < non_overlapping_matches[-1][1]:
            continue
        non_overlapping_matches.append(match)

    non_overlapping_matches.sort(key=lambda match: (match[0], match[1]))
    grouped: dict[str, dict[str, list[SkillMention]]] = {
        "Skill mentioned": {},
        "Negated mention": {},
    }
    for start, end, canonical_name, wording, label in non_overlapping_matches:
        excerpt_start, excerpt_end = _excerpt_bounds(text, start, end)
        if label == "Negated mention":
            negation_start = _negation_prefix_start(text, start)
            if negation_start is not None:
                excerpt_start = min(excerpt_start, negation_start)
        grouped[label].setdefault(canonical_name, []).append(
            SkillMention(
                matched_wording=wording,
                source_excerpt=text[excerpt_start:excerpt_end],
            )
        )

    def evidence_for(label: Literal["Skill mentioned", "Negated mention"]):
        return [
            SkillEvidence(
                canonical_name=canonical_name,
                label=label,
                mentions=mentions,
            )
            for canonical_name, mentions in grouped[label].items()
        ]

    return SkillExtractionResult(
        mentioned_skills=evidence_for("Skill mentioned"),
        negated_skills=evidence_for("Negated mention"),
    )
