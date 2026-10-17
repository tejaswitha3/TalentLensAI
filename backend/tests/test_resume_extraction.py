import io
import json
import re
import unittest
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pymupdf as fitz

from app.main import app
from app.evidence_extraction import (
    EvidenceExtractionResponse,
    EvidenceQuote,
    EvidenceRelationship,
    RequirementEvidence,
)
from app.semantic_comparison import (
    EmbeddingModelUnavailableError,
    MAX_REQUIREMENT_LENGTH,
    MAX_RESUME_TEXT_LENGTH,
    SemanticComparisonRequest,
    compare_resume_requirements,
    split_resume_passages,
)
from app.skill_extraction import extract_skills


def make_pdf(
    text: str | None = None,
    *,
    password: str | None = None,
    empty_first_page: bool = False,
    wrap_text: bool = False,
) -> bytes:
    document = fitz.open()
    if empty_first_page:
        document.new_page()
    page = document.new_page()
    if text:
        if wrap_text:
            page.insert_textbox((40, 50, 555, 300), text, fontsize=12)
        else:
            page.insert_text((72, 72), text)

    options = {}
    if password:
        options = {
            "encryption": fitz.PDF_ENCRYPT_AES_256,
            "owner_pw": "owner-password",
            "user_pw": password,
        }

    pdf_bytes = document.tobytes(**options)
    document.close()
    return pdf_bytes


class ResumeExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def upload(self, filename: str, content: bytes):
        return await self.client.post(
            "/api/resumes/extract",
            files={"file": (filename, io.BytesIO(content), "application/pdf")},
        )

    async def test_extracts_text_from_valid_pdf(self) -> None:
        response = await self.upload(
            "fictional_resume.pdf",
            make_pdf("Jordan Avery | Fictional Software Engineer"),
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["filename"], "fictional_resume.pdf")
        self.assertEqual(result["page_count"], 1)
        self.assertIn("Jordan Avery", result["extracted_text"])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["mentioned_skills"], [])
        self.assertEqual(result["negated_skills"], [])

    async def test_fictional_resume_flows_through_job_evidence_and_scoring(self) -> None:
        fictional_text = (
            "Casey Example\nBuilt HTTP endpoints using FastAPI. "
            "I have no experience with Docker."
        )
        uploaded = await self.upload(
            "casey_example_fictional.pdf",
            make_pdf(fictional_text, wrap_text=True),
        )
        self.assertEqual(uploaded.status_code, 200)
        extracted = uploaded.json()
        source_text = extracted["extracted_text"]

        job_response = await self.client.post(
            "/api/jobs/validate",
            json={
                "title": "Backend Engineer",
                "description": "Build API services.",
                "required_skills": ["FastAPI", "Docker"],
                "preferred_skills": ["Kubernetes"],
                "requirements": [
                    {"text": "REST API development"},
                    {"text": "Docker experience"},
                    {"text": "Kubernetes experience"},
                ],
            },
        )
        self.assertEqual(job_response.status_code, 200)
        job = job_response.json()
        requirements = job["requirements"]

        semantic_result = {
            "comparisons": [
                {
                    "requirement": requirement["text"],
                    "matches": [
                        {
                            "excerpt": "Fictional retrieval-only passage.",
                            "similarity": 0.9,
                        }
                    ],
                }
                for requirement in requirements
            ]
        }
        with patch(
            "app.main.get_embedding_model",
            return_value=MagicMock(),
        ), patch(
            "app.main.compare_resume_requirements",
            return_value=semantic_result,
        ):
            compared = await self.client.post(
                "/api/resumes/compare",
                json={
                    "resume_text": source_text,
                    "requirements": [
                        requirement["text"] for requirement in requirements
                    ],
                },
            )
        self.assertEqual(compared.status_code, 200)

        docker_start = source_text.index("I have no experience")
        docker_quote = source_text[
            docker_start : source_text.index(".", docker_start) + 1
        ]
        rest_quote = "Built HTTP endpoints using FastAPI."
        evidence_result = EvidenceExtractionResponse(
            results=[
                RequirementEvidence(
                    requirement_id=requirements[0]["id"],
                    status="supported",
                    explanation=f"The source says: {rest_quote}",
                    evidence=[
                        EvidenceQuote(
                            source_id="source-1",
                            relationship=EvidenceRelationship.SUPPORTS,
                            quote=rest_quote,
                        )
                    ],
                ),
                RequirementEvidence(
                    requirement_id=requirements[1]["id"],
                    status="explicitly_contradicted",
                    explanation=f"The source says: {docker_quote}",
                    evidence=[
                        EvidenceQuote(
                            source_id="source-1",
                            relationship=EvidenceRelationship.CONTRADICTS,
                            quote=docker_quote,
                        )
                    ],
                ),
                RequirementEvidence(
                    requirement_id=requirements[2]["id"],
                    status="unclear",
                    explanation="The source does not provide Kubernetes experience.",
                    evidence=[],
                ),
            ]
        )
        with patch(
            "app.main.extract_requirement_evidence",
            return_value=evidence_result,
        ) as mocked_provider:
            evidence_response = await self.client.post(
                "/api/resumes/evidence",
                json={
                    "confirmed_fictional_demo": True,
                    "requirements": [
                        {
                            "id": requirement["id"],
                            "text": requirement["text"],
                        }
                        for requirement in requirements
                    ],
                    "source_passages": [
                        {"id": "source-1", "text": source_text}
                    ],
                },
            )
        self.assertEqual(evidence_response.status_code, 200)
        mocked_provider.assert_called_once()

        scored = await self.client.post(
            "/api/assessment/score",
            json={
                "candidate_id": "candidate-01",
                "job": job,
                "skills": {
                    "mentioned_skills": extracted["mentioned_skills"],
                    "negated_skills": extracted["negated_skills"],
                },
                "semantic": compared.json(),
                "evidence": evidence_response.json(),
                "source_passages": [
                    {"id": "source-1", "text": source_text}
                ],
            },
        )
        self.assertEqual(scored.status_code, 200, scored.text)
        assessment = scored.json()
        self.assertEqual(
            [item["status"] for item in assessment["requirement_assessments"]],
            ["supported", "explicitly_contradicted", "unclear"],
        )
        self.assertEqual(
            assessment["requirement_assessments"][0]["evidence"][0]["quote"],
            rest_quote,
        )
        self.assertEqual(
            assessment["requirement_assessments"][1]["evidence"][0]["quote"],
            docker_quote,
        )
        self.assertEqual(
            assessment["requirement_assessments"][2]["evidence"],
            [],
        )

    async def test_rejects_invalid_content_disguised_as_pdf(self) -> None:
        response = await self.upload(
            "fictional_resume.pdf",
            b"not a PDF document",
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("not a valid PDF", response.json()["detail"])

    async def test_rejects_forged_pdf_signature_with_invalid_content(self) -> None:
        response = await self.upload(
            "fictional_resume.pdf",
            b"%PDF-1.7\nThis is not a valid PDF document.",
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("could not be opened", response.json()["detail"])

    async def test_extracts_skills_from_pdf_response(self) -> None:
        response = await self.upload(
            "fictional_skills_resume.pdf",
            make_pdf("Built a fictional API using FastAPI and PostgreSQL."),
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        skills = {
            skill["canonical_name"]: skill
            for skill in result["mentioned_skills"]
        }
        self.assertIn("FastAPI", skills)
        self.assertEqual(skills["FastAPI"]["label"], "Skill mentioned")
        self.assertEqual(
            skills["FastAPI"]["mentions"][0]["source_excerpt"],
            "Built a fictional API using FastAPI and PostgreSQL.",
        )
        self.assertIn("PostgreSQL", skills)
        self.assertEqual(result["negated_skills"], [])

    async def test_detects_negated_skill_when_pdf_wraps_statement(self) -> None:
        response = await self.upload(
            "fictional_wrapped_resume.pdf",
            make_pdf(
                "Fictional candidate Casey Example. Built projects with "
                "Python and NEXT.JS. No experience with Docker.",
                wrap_text=True,
            ),
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("No experience with\nDocker.", result["extracted_text"])
        self.assertEqual(
            [skill["canonical_name"] for skill in result["mentioned_skills"]],
            ["Python", "Next.js"],
        )
        docker = next(
            skill
            for skill in result["negated_skills"]
            if skill["canonical_name"] == "Docker"
        )
        self.assertEqual(docker["label"], "Negated mention")
        self.assertEqual(
            docker["mentions"][0]["source_excerpt"],
            "No experience with\nDocker.",
        )

    async def test_fictional_resume_evidence_uses_exact_pdf_sentences(self) -> None:
        extraction = await self.upload(
            "Casey_Example_Fictional_Resume.pdf",
            make_pdf(
                "Fictional candidate Casey Example. "
                "Built HTTP endpoints using FastAPI. "
                "I have no experience with Docker.",
                wrap_text=True,
            ),
        )
        self.assertEqual(extraction.status_code, 200)
        extracted_text = extraction.json()["extracted_text"]
        self.assertIn("Built HTTP endpoints using", extracted_text)
        self.assertIn("FastAPI.", extracted_text)
        self.assertIn("I have no experience", extracted_text)
        self.assertIn("with Docker.", extracted_text)

        provider_client = MagicMock()
        cited_sources: dict[str, dict[str, str]] = {}

        def generate_content(*, contents: str, **kwargs):
            payload = json.loads(contents)
            passages = payload["source_passages"]
            api_source = next(
                passage
                for passage in passages
                if "Built HTTP endpoints using" in passage["text"]
                and "FastAPI." in passage["text"]
            )
            docker_source = next(
                passage
                for passage in passages
                if "I have no experience" in passage["text"]
                and "with Docker." in passage["text"]
            )
            cited_sources["api"] = api_source
            cited_sources["docker"] = docker_source
            provider_client.models.generate_content.return_value.text = json.dumps(
                {
                    "results": [
                        {
                            "requirement_id": "requirement-1",
                            "status": "supported",
                            "evidence": [
                                {
                                    "source_id": api_source["id"],
                                    "relationship": "supports",
                                }
                            ],
                        },
                        {
                            "requirement_id": "requirement-2",
                            "status": "explicitly_contradicted",
                            "evidence": [
                                {
                                    "source_id": docker_source["id"],
                                    "relationship": "contradicts",
                                }
                            ],
                        },
                        {
                            "requirement_id": "requirement-3",
                            "status": "unclear",
                            "evidence": [],
                        },
                    ]
                }
            )
            return provider_client.models.generate_content.return_value

        provider_client.models.generate_content.side_effect = generate_content
        with (
            patch("app.evidence_extraction.get_gemini_api_key", return_value="mock"),
            patch("google.genai.Client", return_value=provider_client),
        ):
            evidence = await self.client.post(
                "/api/resumes/evidence",
                json={
                    "confirmed_fictional_demo": True,
                    "requirements": [
                        {"id": "requirement-1", "text": "REST API development"},
                        {
                            "id": "requirement-2",
                            "text": "Docker hands-on experience",
                        },
                        {"id": "requirement-3", "text": "Kubernetes experience"},
                    ],
                    "source_passages": [
                        {"id": "source-1", "text": extracted_text}
                    ],
                },
            )

        self.assertEqual(evidence.status_code, 200)
        results = evidence.json()["results"]
        self.assertEqual(
            [result["status"] for result in results],
            ["supported", "explicitly_contradicted", "unclear"],
        )
        self.assertEqual(
            results[0]["evidence"][0]["quote"],
            cited_sources["api"]["text"],
        )
        self.assertEqual(
            results[1]["evidence"][0]["quote"],
            cited_sources["docker"]["text"],
        )
        self.assertEqual(results[2]["evidence"], [])
        self.assertEqual(
            provider_client.models.generate_content.call_count,
            1,
        )

    async def test_rejects_upload_larger_than_five_megabytes(self) -> None:
        response = await self.upload(
            "fictional_resume.pdf",
            b"%PDF-" + b"x" * (5 * 1024 * 1024),
        )

        self.assertEqual(response.status_code, 413)
        self.assertIn("5 MB", response.json()["detail"])

    async def test_explains_that_scanned_pdf_requires_unsupported_ocr(self) -> None:
        response = await self.upload("fictional_scan.pdf", make_pdf())

        self.assertEqual(response.status_code, 422)
        self.assertIn("OCR is not supported yet", response.json()["detail"])

    async def test_rejects_encrypted_pdf(self) -> None:
        response = await self.upload(
            "fictional_protected_resume.pdf",
            make_pdf("Fictional protected resume", password="sample-password"),
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("password-protected", response.json()["detail"])

    async def test_rejects_non_pdf_extension(self) -> None:
        response = await self.upload(
            "fictional_resume.txt",
            b"fictional resume text",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Only .pdf", response.json()["detail"])

    async def test_warns_when_a_page_contains_no_text(self) -> None:
        response = await self.upload(
            "fictional_resume_with_blank_page.pdf",
            make_pdf("Jordan Avery", empty_first_page=True),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["page_count"], 2)
        self.assertEqual(
            response.json()["warnings"],
            ["Page 1 contains no extractable text."],
        )

    async def test_health_endpoint_remains_available(self) -> None:
        response = await self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


class SkillExtractionTests(unittest.TestCase):
    def test_matches_aliases_case_insensitively(self) -> None:
        result = extract_skills(
            "Used NEXT.JS, sklearn, and REST API tooling in a fictional project."
        )

        self.assertEqual(
            [skill.canonical_name for skill in result.mentioned_skills],
            ["Next.js", "scikit-learn", "REST APIs"],
        )
        self.assertEqual(
            [skill.mentions[0].matched_wording for skill in result.mentioned_skills],
            ["NEXT.JS", "sklearn", "REST API"],
        )

    def test_deduplicates_canonical_skill_but_preserves_repeated_mentions(self) -> None:
        result = extract_skills("Python scripting; PYTHON testing.")

        self.assertEqual(len(result.mentioned_skills), 1)
        python = result.mentioned_skills[0]
        self.assertEqual(python.canonical_name, "Python")
        self.assertEqual(
            [mention.matched_wording for mention in python.mentions],
            ["Python", "PYTHON"],
        )

    def test_respects_term_boundaries(self) -> None:
        result = extract_skills("JavaScript and GitHub, but Java and Git are listed.")

        names = [skill.canonical_name for skill in result.mentioned_skills]
        self.assertEqual(names, ["JavaScript", "Java", "Git"])
        self.assertNotIn("GitHub", names)

    def test_returns_exact_source_excerpt(self) -> None:
        text = "Fictional work: built with FastAPI.\nNext line mentions Docker."
        result = extract_skills(text)
        fastapi = result.mentioned_skills[0].mentions[0]

        self.assertEqual(fastapi.source_excerpt, "Fictional work: built with FastAPI.")
        self.assertIn(fastapi.source_excerpt, text)

    def test_separates_clear_negated_skill_mentions(self) -> None:
        result = extract_skills("No experience with Docker. Uses Python.")

        self.assertEqual(
            [skill.canonical_name for skill in result.negated_skills],
            ["Docker"],
        )
        self.assertEqual(
            result.negated_skills[0].label,
            "Negated mention",
        )
        self.assertEqual(
            [skill.canonical_name for skill in result.mentioned_skills],
            ["Python"],
        )

    def test_detects_negation_across_pdf_line_wrap_and_keeps_exact_excerpt(
        self,
    ) -> None:
        text = "Built fictional projects.\nNo experience with\nDocker."

        result = extract_skills(text)

        self.assertEqual(result.mentioned_skills, [])
        self.assertEqual(
            [skill.canonical_name for skill in result.negated_skills],
            ["Docker"],
        )
        self.assertEqual(
            result.negated_skills[0].mentions[0].source_excerpt,
            "No experience with\nDocker.",
        )

    def test_returns_no_matches_for_unrecognized_text(self) -> None:
        result = extract_skills("Fictional candidate enjoys hiking and music.")

        self.assertEqual(result.mentioned_skills, [])
        self.assertEqual(result.negated_skills, [])


class WhitespaceTokenizer:
    def __call__(self, text: str, **_kwargs):
        return {
            "offset_mapping": [
                (match.start(), match.end())
                for match in re.finditer(r"\S+", text)
            ]
        }


class FakeEmbeddingModel:
    max_seq_length = 7
    tokenizer = WhitespaceTokenizer()

    def encode(self, texts: list[str], **_kwargs):
        vectors = []
        for text in texts:
            if text == "REST API development":
                vectors.append([1.0, 0.0])
            elif text == "Built HTTP endpoints using FastAPI.":
                vectors.append([0.8, 0.6])
            elif text == "Fictional gardening and music.":
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.6, 0.8])
        return np.array(vectors)


class SemanticComparisonTests(unittest.TestCase):
    def test_validates_resume_size_and_requirement_count(self) -> None:
        with self.assertRaises(ValueError):
            SemanticComparisonRequest.model_validate(
                {
                    "resume_text": "x" * (MAX_RESUME_TEXT_LENGTH + 1),
                    "requirements": ["Python"],
                }
            )
        with self.assertRaises(ValueError):
            SemanticComparisonRequest.model_validate(
                {
                    "resume_text": "Fictional resume",
                    "requirements": [f"Requirement {index}" for index in range(11)],
                }
            )
        with self.assertRaises(ValueError):
            SemanticComparisonRequest.model_validate(
                {"resume_text": "Fictional resume", "requirements": ["  "]}
            )
        with self.assertRaises(ValueError):
            SemanticComparisonRequest.model_validate(
                {
                    "resume_text": "Fictional resume",
                    "requirements": ["x" * (MAX_REQUIREMENT_LENGTH + 1)],
                }
            )

    def test_splits_passages_without_losing_exact_resume_text(self) -> None:
        text = "Built HTTP endpoints using FastAPI.\nFictional gardening and music."

        passages = split_resume_passages(text, WhitespaceTokenizer(), 5)

        self.assertEqual(
            passages,
            [
                "Built HTTP endpoints using FastAPI.",
                "Fictional gardening and music.",
            ],
        )
        self.assertTrue(all(passage in text for passage in passages))

    def test_splits_long_sentences_into_bounded_exact_passages(self) -> None:
        text = "one two three four five six seven eight nine ten eleven twelve."

        passages = split_resume_passages(text, WhitespaceTokenizer(), 5)

        self.assertEqual(len(passages), 3)
        self.assertEqual(" ".join(passages), text)

    def test_returns_exact_excerpts_and_cosine_similarity_order(self) -> None:
        text = "Built HTTP endpoints using FastAPI. Fictional gardening and music."

        response = compare_resume_requirements(
            text,
            ["REST API development"],
            FakeEmbeddingModel(),
        )

        comparison = response.comparisons[0]
        self.assertEqual(comparison.requirement, "REST API development")
        self.assertEqual(
            [match.excerpt for match in comparison.matches],
            [
                "Built HTTP endpoints using FastAPI.",
                "Fictional gardening and music.",
            ],
        )
        self.assertAlmostEqual(comparison.matches[0].similarity, 0.8)
        self.assertAlmostEqual(comparison.matches[1].similarity, 0.0)

    def test_returns_no_more_than_three_passages_per_requirement(self) -> None:
        text = (
            "Built HTTP endpoints using FastAPI. "
            "Fictional gardening and music. "
            "Additional fictional candidate details. "
            "Another unrelated sentence."
        )

        response = compare_resume_requirements(
            text,
            ["REST API development"],
            FakeEmbeddingModel(),
        )

        self.assertEqual(len(response.comparisons[0].matches), 3)


class SemanticComparisonEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_comparison_endpoint_returns_mocked_local_results(self) -> None:
        with patch("app.main.get_embedding_model", return_value=FakeEmbeddingModel()):
            response = await self.client.post(
                "/api/resumes/compare",
                json={
                    "resume_text": (
                        "Built HTTP endpoints using FastAPI. "
                        "Fictional gardening and music."
                    ),
                    "requirements": ["REST API development"],
                },
            )

        self.assertEqual(response.status_code, 200)
        result = response.json()["comparisons"][0]
        self.assertEqual(result["requirement"], "REST API development")
        self.assertEqual(
            result["matches"][0]["excerpt"],
            "Built HTTP endpoints using FastAPI.",
        )
        self.assertAlmostEqual(result["matches"][0]["similarity"], 0.8)

    async def test_endpoint_rejects_more_than_ten_requirements(self) -> None:
        response = await self.client.post(
            "/api/resumes/compare",
            json={
                "resume_text": "Fictional resume",
                "requirements": [f"Requirement {index}" for index in range(11)],
            },
        )

        self.assertEqual(response.status_code, 422)

    async def test_endpoint_rejects_resume_text_above_limit(self) -> None:
        response = await self.client.post(
            "/api/resumes/compare",
            json={
                "resume_text": "x" * (MAX_RESUME_TEXT_LENGTH + 1),
                "requirements": ["Python"],
            },
        )

        self.assertEqual(response.status_code, 422)

    async def test_endpoint_reports_unavailable_model_clearly(self) -> None:
        with patch(
            "app.main.get_embedding_model",
            side_effect=EmbeddingModelUnavailableError,
        ):
            response = await self.client.post(
                "/api/resumes/compare",
                json={
                    "resume_text": "Fictional resume",
                    "requirements": ["Python"],
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("local embedding model is unavailable", response.json()["detail"])

    async def test_endpoint_reports_processing_failure_without_resume_text(self) -> None:
        class FailingEmbeddingModel(FakeEmbeddingModel):
            def encode(self, _texts: list[str], **_kwargs):
                raise RuntimeError("fictional resume content must not be logged")

        with patch(
            "app.main.get_embedding_model",
            return_value=FailingEmbeddingModel(),
        ):
            response = await self.client.post(
                "/api/resumes/compare",
                json={
                    "resume_text": "Fictional private resume text",
                    "requirements": ["Python"],
                },
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "Local semantic comparison could not be completed.",
        )
        self.assertNotIn("Fictional private resume text", response.text)


if __name__ == "__main__":
    unittest.main()
