import logging
import re
from threading import Lock
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, Field, StringConstraints, field_validator

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_RESUME_TEXT_LENGTH = 100_000
MAX_REQUIREMENT_LENGTH = 500
MAX_REQUIREMENTS = 10
MAX_RESULTS_PER_REQUIREMENT = 3

logger = logging.getLogger(__name__)
_model: Any | None = None
_model_lock = Lock()

RequirementText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_REQUIREMENT_LENGTH),
]


class SemanticComparisonRequest(BaseModel):
    resume_text: str = Field(min_length=1, max_length=MAX_RESUME_TEXT_LENGTH)
    requirements: list[RequirementText] = Field(
        min_length=1,
        max_length=MAX_REQUIREMENTS,
    )

    @field_validator("resume_text")
    @classmethod
    def resume_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Resume text must contain non-whitespace characters.")
        return value

    @field_validator("requirements")
    @classmethod
    def requirements_must_not_be_blank(
        cls,
        value: list[str],
    ) -> list[str]:
        if any(not requirement.strip() for requirement in value):
            raise ValueError("Requirements must not be blank.")
        return value


class PassageMatch(BaseModel):
    excerpt: str
    similarity: float = Field(allow_inf_nan=False)


class RequirementComparison(BaseModel):
    requirement: str
    matches: list[PassageMatch]


class SemanticComparisonResponse(BaseModel):
    comparisons: list[RequirementComparison]


class EmbeddingModelUnavailableError(Exception):
    pass


def get_embedding_model() -> Any:
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            _model = SentenceTransformer(MODEL_NAME, device=device)
        except Exception as error:
            logger.error(
                "Could not load the local sentence embedding model (%s).",
                type(error).__name__,
            )
            raise EmbeddingModelUnavailableError from error

    return _model


def split_resume_passages(text: str, tokenizer: Any, max_tokens: int) -> list[str]:
    if max_tokens < 1:
        raise ValueError("Passage token limit must be positive.")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = [
        (int(start), int(end))
        for start, end in encoded["offset_mapping"]
        if end > start
    ]
    if not offsets:
        raise ValueError("Resume text did not produce any model tokens.")

    sentence_spans = []
    start = 0
    for boundary in re.finditer(r"[.!?]+(?=\s)|\n{2,}", text):
        sentence_spans.append((start, boundary.end()))
        start = boundary.end()
    sentence_spans.append((start, len(text)))

    passages: list[str] = []
    token_cursor = 0
    for sentence_start, sentence_end in sentence_spans:
        while (
            token_cursor < len(offsets)
            and offsets[token_cursor][1] <= sentence_start
        ):
            token_cursor += 1
        sentence_token_start = token_cursor
        while (
            token_cursor < len(offsets)
            and offsets[token_cursor][0] < sentence_end
        ):
            token_cursor += 1
        sentence_offsets = offsets[sentence_token_start:token_cursor]
        for start in range(0, len(sentence_offsets), max_tokens):
            token_end = min(start + max_tokens, len(sentence_offsets))
            excerpt = text[
                sentence_offsets[start][0] : sentence_offsets[token_end - 1][1]
            ]
            if excerpt:
                passages.append(excerpt)
    return passages


def compare_resume_requirements(
    resume_text: str,
    requirements: list[str],
    model: Any,
) -> SemanticComparisonResponse:
    model_max_tokens = int(model.max_seq_length)
    max_passage_tokens = min(model_max_tokens - 2, 220)
    if max_passage_tokens < 1:
        raise ValueError("Embedding model has an invalid maximum sequence length.")

    passages = split_resume_passages(
        resume_text,
        model.tokenizer,
        max_passage_tokens,
    )
    encoded = model.encode(
        [*requirements, *passages],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    requirement_vectors = encoded[: len(requirements)]
    passage_vectors = encoded[len(requirements) :]
    similarities = np.matmul(requirement_vectors, passage_vectors.T)

    comparisons = []
    for requirement, scores in zip(requirements, similarities, strict=True):
        top_indexes = sorted(
            range(len(passages)),
            key=lambda index: (-float(scores[index]), index),
        )[:MAX_RESULTS_PER_REQUIREMENT]
        comparisons.append(
            RequirementComparison(
                requirement=requirement,
                matches=[
                    PassageMatch(
                        excerpt=passages[index],
                        similarity=float(scores[index]),
                    )
                    for index in top_indexes
                ],
            )
        )

    return SemanticComparisonResponse(comparisons=comparisons)
