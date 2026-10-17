import logging
import uuid
from pathlib import PurePath

import pymupdf as fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

from app.skill_extraction import (
    SkillEvidence,
    extract_skills,
)
from app.semantic_comparison import (
    EmbeddingModelUnavailableError,
    SemanticComparisonRequest,
    SemanticComparisonResponse,
    compare_resume_requirements,
    get_embedding_model,
)
from app.evidence_extraction import (
    EvidenceExtractionError,
    EvidenceExtractionRequest,
    EvidenceExtractionResponse,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiServiceUnavailableError,
    GeminiTimeoutError,
    InvalidGeminiResponseError,
    MissingGeminiApiKeyError,
    extract_requirement_evidence,
)
from app.local_evidence import check_local_evidence
from app.job_assessment import (
    AssessmentInput,
    CandidateAssessment,
    JobDefinition,
    rank_assessments,
    score_candidate,
)
from app.postgres_store import (
    DatabaseUnavailableError,
    check_database,
    create_job,
    delete_job,
    list_jobs,
)

app = FastAPI()
MAX_UPLOAD_SIZE = 5 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 64 * 1024


class ResumeExtractionResponse(BaseModel):
    filename: str
    page_count: int
    extracted_text: str
    warnings: list[str]
    mentioned_skills: list[SkillEvidence]
    negated_skills: list[SkillEvidence]


class PersistJobRequest(BaseModel):
    job: JobDefinition
    assessments: list[AssessmentInput] = Field(default_factory=list, max_length=10)


class PersistedJobResponse(BaseModel):
    job_id: str
    job: JobDefinition
    assessments: list[CandidateAssessment]


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/database")
def database_health_check() -> dict[str, str]:
    try:
        check_database()
    except DatabaseUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error
    return {"status": "connected"}


@app.get("/api/jobs")
def get_saved_jobs() -> list[dict]:
    try:
        return list_jobs()
    except DatabaseUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/jobs", response_model=PersistedJobResponse)
def save_job(request: PersistJobRequest) -> dict:
    candidate_ids = [item.candidate_id for item in request.assessments]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise HTTPException(
            status_code=422,
            detail="Candidate identifiers must be unique within a saved job.",
        )
    if any(item.job != request.job for item in request.assessments):
        raise HTTPException(
            status_code=422,
            detail="Every candidate assessment must use the saved job criteria.",
        )
    try:
        assessments = rank_assessments(
            [score_candidate(item) for item in request.assessments]
        )
        return create_job(request.job, assessments)
    except HTTPException:
        raise
    except DatabaseUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (InvalidGeminiResponseError, ValidationError) as error:
        logging.getLogger(__name__).warning(
            "Saved assessment validation failed (%s).",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=422,
            detail="Assessment evidence did not pass source validation.",
        ) from error


@app.delete("/api/jobs/{job_id}")
def remove_saved_job(job_id: str) -> dict[str, str]:
    try:
        normalized_id = str(uuid.UUID(job_id))
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid saved job identifier.") from error
    try:
        if not delete_job(normalized_id):
            raise HTTPException(status_code=404, detail="Saved job was not found.")
    except DatabaseUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "deleted"}


@app.post("/api/jobs/validate", response_model=JobDefinition)
def validate_job(request: JobDefinition) -> JobDefinition:
    return request


@app.post("/api/assessment/score", response_model=CandidateAssessment)
def score_assessment(request: AssessmentInput) -> CandidateAssessment:
    try:
        return score_candidate(request)
    except (InvalidGeminiResponseError, ValidationError) as error:
        logging.getLogger(__name__).warning(
            "Assessment validation failed (%s).",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=422,
            detail="Assessment evidence did not pass source validation.",
        ) from error


class AssessmentRankRequest(BaseModel):
    assessments: list[CandidateAssessment] = Field(max_length=10)


@app.post("/api/assessment/rank", response_model=list[CandidateAssessment])
def rank_assessment_endpoint(
    request: AssessmentRankRequest,
) -> list[CandidateAssessment]:
    return rank_assessments(request.assessments)


@app.post(
    "/api/resumes/extract",
    response_model=ResumeExtractionResponse,
)
async def extract_resume(file: UploadFile = File(...)) -> ResumeExtractionResponse:
    filename = file.filename or ""
    if PurePath(filename).suffix.lower() != ".pdf":
        await file.close()
        raise HTTPException(status_code=400, detail="Only .pdf files can be uploaded.")

    content = bytearray()
    try:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            if len(content) + len(chunk) > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail="The PDF exceeds the 5 MB upload limit.",
                )
            content.extend(chunk)
    finally:
        await file.close()

    if b"%PDF-" not in content[:1024]:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file content is not a valid PDF.",
        )

    document: fitz.Document | None = None
    try:
        document = fitz.open(stream=bytes(content), filetype="pdf")
        if document.needs_pass:
            raise HTTPException(
                status_code=422,
                detail="This PDF is password-protected and cannot be processed.",
            )

        page_texts: list[str] = []
        warnings: list[str] = []
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()
            page_texts.append(page_text)
            if not page_text:
                warnings.append(
                    f"Page {page_number} contains no extractable text."
                )

        extracted_text = "\n\n".join(text for text in page_texts if text)
        if not extracted_text:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No selectable text was found. This may be a scanned PDF; "
                    "OCR is not supported yet."
                ),
            )

        return ResumeExtractionResponse(
            filename=filename,
            page_count=document.page_count,
            extracted_text=extracted_text,
            warnings=warnings,
            **extract_skills(extracted_text).model_dump(),
        )
    except HTTPException:
        raise
    except (fitz.FileDataError, fitz.EmptyFileError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file could not be opened as a valid PDF.",
        ) from error
    finally:
        if document is not None:
            document.close()


@app.post(
    "/api/resumes/compare",
    response_model=SemanticComparisonResponse,
)
def compare_resume(
    request: SemanticComparisonRequest,
) -> SemanticComparisonResponse:
    try:
        model = get_embedding_model()
    except EmbeddingModelUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "The local embedding model is unavailable. On first use, "
                "ensure the model can be downloaded, then try again."
            ),
        ) from error

    try:
        return compare_resume_requirements(
            request.resume_text,
            request.requirements,
            model,
        )
    except Exception as error:
        logging.getLogger(__name__).error(
            "Local semantic comparison failed (%s).",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Local semantic comparison could not be completed.",
        ) from error


@app.post(
    "/api/resumes/evidence/local",
    response_model=EvidenceExtractionResponse,
)
def check_local_evidence_endpoint(
    request: EvidenceExtractionRequest,
) -> EvidenceExtractionResponse:
    return check_local_evidence(request)


@app.post(
    "/api/resumes/evidence",
    response_model=EvidenceExtractionResponse,
)
def extract_requirement_evidence_endpoint(
    request: EvidenceExtractionRequest,
) -> EvidenceExtractionResponse:
    try:
        return extract_requirement_evidence(request)
    except MissingGeminiApiKeyError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is not configured. Add GEMINI_API_KEY to backend/.env "
                "for fictional demo data."
            ),
        ) from error
    except GeminiRateLimitError as error:
        raise HTTPException(
            status_code=429,
            detail="Gemini's free-tier rate limit was reached. Try again later.",
        ) from error
    except GeminiServiceUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini's generation service is temporarily unavailable. "
                "No evidence was assessed; try again later."
            ),
        ) from error
    except GeminiTimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail="The Gemini evidence request timed out. Please try again.",
        ) from error
    except InvalidGeminiResponseError as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini returned evidence that failed validation. "
                "No assessment was returned."
            ),
        ) from error
    except GeminiProviderError as error:
        raise HTTPException(
            status_code=502,
            detail="Gemini could not complete the evidence request.",
        ) from error
    except EvidenceExtractionError as error:
        raise HTTPException(
            status_code=502,
            detail="Evidence extraction could not be completed.",
        ) from error
