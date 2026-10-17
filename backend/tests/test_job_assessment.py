import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.evidence_extraction import (
    EvidenceExtractionResponse,
    InvalidGeminiResponseError,
    EvidenceQuote,
    EvidenceRelationship,
    RequirementEvidence,
    SourcePassageInput,
)
from app.job_assessment import (
    AssessmentInput,
    CandidateAssessment,
    JobDefinition,
    JobRequirement,
    rank_assessments,
    score_candidate,
)
from app.main import app
from app.postgres_store import (
    DatabaseUnavailableError,
    check_database,
    create_job,
    delete_job,
)
from app.semantic_comparison import (
    PassageMatch,
    RequirementComparison,
    SemanticComparisonResponse,
)
from app.skill_extraction import extract_skills


CASEY_TEXT = (
    "Built HTTP endpoints using FastAPI. "
    "I have no experience with Docker."
)


def make_job() -> JobDefinition:
    return JobDefinition(
        title="  Backend Engineer  ",
        description="Build and operate web services.",
        required_skills=["REST API development", "Docker"],
        preferred_skills=["Kubernetes"],
        requirements=[
            {"text": "REST API development"},
            {"text": "Docker hands-on experience"},
            {"text": "Kubernetes experience"},
        ],
    )


def make_assessment(
    *,
    candidate_id: str = "candidate-1",
    text: str = CASEY_TEXT,
    semantic_scores: tuple[float, float, float] = (0.99, 0.99, 0.9),
) -> AssessmentInput:
    job = make_job()
    api, docker, kubernetes = job.requirements
    evidence = EvidenceExtractionResponse(
        results=[
            RequirementEvidence(
                requirement_id=api.id,
                status="supported",
                explanation="The resume says: Built HTTP endpoints using FastAPI.",
                evidence=[
                    EvidenceQuote(
                        source_id="source-1",
                        relationship=EvidenceRelationship.SUPPORTS,
                        quote="Built HTTP endpoints using FastAPI.",
                    )
                ],
            ),
            RequirementEvidence(
                requirement_id=docker.id,
                status="explicitly_contradicted",
                explanation="The resume says: I have no experience with Docker.",
                evidence=[
                    EvidenceQuote(
                        source_id="source-1",
                        relationship=EvidenceRelationship.CONTRADICTS,
                        quote="I have no experience with Docker.",
                    )
                ],
            ),
            RequirementEvidence(
                requirement_id=kubernetes.id,
                status="unclear",
                explanation="No Kubernetes evidence appears in the source.",
                evidence=[],
            ),
        ]
    )
    semantic = SemanticComparisonResponse(
        comparisons=[
            RequirementComparison(
                requirement=requirement.text,
                matches=[
                    PassageMatch(
                        excerpt="Retrieval excerpt is not proof.",
                        similarity=similarity,
                    )
                ],
            )
            for requirement, similarity in zip(
                job.requirements,
                semantic_scores,
                strict=True,
            )
        ]
    )
    return AssessmentInput(
        candidate_id=candidate_id,
        job=job,
        skills=extract_skills(text),
        semantic=semantic,
        evidence=evidence,
        source_passages=[SourcePassageInput(id="source-1", text=text)],
    )


class JobDefinitionTests(unittest.TestCase):
    def test_normalizes_job_inputs_and_generates_stable_requirement_ids(self) -> None:
        first = make_job()
        second = make_job()

        self.assertEqual(first.title, "Backend Engineer")
        self.assertEqual(first.required_skills, ["REST API development", "Docker"])
        self.assertEqual(
            [item.id for item in first.requirements],
            [item.id for item in second.requirements],
        )

    def test_rejects_duplicate_requirements_and_excessive_requirements(self) -> None:
        with self.assertRaises(ValidationError):
            JobDefinition(
                title="Engineer",
                requirements=[{"text": "Python"}, {"text": " python "}],
            )
        with self.assertRaises(ValidationError):
            JobDefinition(
                title="Engineer",
                requirements=[{"text": f"Requirement {index}"} for index in range(11)],
            )


class CandidateScoringTests(unittest.TestCase):
    def test_scores_support_contradiction_and_unverified_separately(self) -> None:
        result = score_candidate(make_assessment())

        self.assertTrue(result.complete)
        self.assertEqual(
            [item.status for item in result.requirement_assessments],
            ["supported", "explicitly_contradicted", "unclear"],
        )
        self.assertEqual(
            [item.evidence[0]["quote"] if item.evidence else None
             for item in result.requirement_assessments],
            [
                "Built HTTP endpoints using FastAPI.",
                "I have no experience with Docker.",
                None,
            ],
        )
        self.assertEqual(result.evidence_relevance.score, 50.0)
        self.assertEqual(result.semantic_relevance.score, 63.0)
        self.assertEqual(result.overall_score, 28.9)
        self.assertIn("× 0.50", result.arithmetic)

    def test_similarity_cannot_override_an_explicit_contradiction(self) -> None:
        result = score_candidate(
            make_assessment(semantic_scores=(0.99, 1.0, 0.9))
        )

        self.assertEqual(
            result.requirement_assessments[1].similarity,
            100.0,
        )
        self.assertEqual(result.semantic_relevance.score, 63.0)
        self.assertEqual(
            result.requirement_assessments[1].status,
            "explicitly_contradicted",
        )

    def test_rejects_quote_that_is_not_an_exact_source_substring(self) -> None:
        invalid = make_assessment()
        evidence = invalid.evidence.model_copy(
            update={
                "results": [
                    invalid.evidence.results[0].model_copy(
                        update={
                            "explanation": (
                                "The resume says: Paraphrased FastAPI work."
                            ),
                            "evidence": [
                                invalid.evidence.results[0].evidence[0].model_copy(
                                    update={"quote": "Paraphrased FastAPI work."}
                                )
                            ]
                        }
                    ),
                    *invalid.evidence.results[1:],
                ]
            }
        )
        with self.assertRaises(InvalidGeminiResponseError):
            score_candidate(invalid.model_copy(update={"evidence": evidence}))

    def test_provider_failure_returns_incomplete_without_overall_score(self) -> None:
        result = score_candidate(
            make_assessment().model_copy(
                update={"evidence": None, "evidence_error": "Provider unavailable."}
            )
        )

        self.assertFalse(result.complete)
        self.assertIsNone(result.overall_score)
        self.assertIsNone(result.arithmetic)

    def test_empty_requirements_do_not_divide_by_zero(self) -> None:
        inputs = make_assessment()
        result = score_candidate(
            inputs.model_copy(
                update={
                    "job": JobDefinition(
                        title="General",
                        required_skills=["FastAPI"],
                        preferred_skills=[],
                        requirements=[],
                    ),
                    "evidence": None,
                    "semantic": None,
                }
            )
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.overall_score, 100.0)

    def test_empty_job_criteria_is_incomplete_not_a_100_score(self) -> None:
        inputs = make_assessment().model_copy(
            update={
                "job": JobDefinition(title="Unspecified role"),
                "evidence": None,
                "semantic": None,
            }
        )
        result = score_candidate(inputs)
        self.assertFalse(result.complete)
        self.assertIsNone(result.overall_score)
        self.assertEqual(result.incomplete_reason, "No job criteria were provided.")

    def test_ranking_is_deterministic_and_keeps_ties_in_input_order(self) -> None:
        first = score_candidate(make_assessment(candidate_id="first"))
        tied = score_candidate(make_assessment(candidate_id="tied"))
        lower = score_candidate(
            make_assessment(
                candidate_id="lower",
                semantic_scores=(0.5, 0.5, 0.5),
            )
        )
        incomplete = score_candidate(
            make_assessment().model_copy(
                update={"evidence": None, "evidence_error": "Unavailable."}
            )
        )

        ranked = rank_assessments([first, tied, lower, incomplete])

        self.assertEqual(
            [item.candidate_id for item in ranked],
            ["first", "tied", "lower", "candidate-1"],
        )
        self.assertEqual([item.rank for item in ranked], [1, 1, 3, None])

    def test_fictional_evaluation_set_has_expected_statuses_and_ranking(self) -> None:
        evaluation_path = (
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "fictional_cases.json"
        )
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        job = JobDefinition.model_validate(evaluation["job"])
        ranked_inputs = []
        for candidate in evaluation["candidates"]:
            statuses = candidate["expected_statuses"]
            requirement_evidence = []
            for requirement, status in zip(
                job.requirements,
                statuses,
                strict=True,
            ):
                evidence_items = []
                if status != "unclear":
                    if candidate["candidate_id"] == "casey-example":
                        quote = (
                            "Built HTTP endpoints using FastAPI."
                            if requirement.text == "REST API development"
                            else "I have no experience with Docker."
                        )
                        relationship = (
                            EvidenceRelationship.SUPPORTS
                            if status == "supported"
                            else EvidenceRelationship.CONTRADICTS
                        )
                    else:
                        quote = (
                            "Built REST API services with FastAPI."
                            if requirement.text == "REST API development"
                            else "Deployed workloads using Docker."
                        )
                        relationship = EvidenceRelationship.SUPPORTS
                    evidence_items.append(
                        EvidenceQuote(
                            source_id="source-1",
                            relationship=relationship,
                            quote=quote,
                        )
                    )
                    explanation = f"Source says: {quote}"
                else:
                    explanation = "No source evidence appears for this requirement."
                requirement_evidence.append(
                    RequirementEvidence(
                        requirement_id=requirement.id,
                        status=status,
                        explanation=explanation,
                        evidence=evidence_items,
                    )
                )
            semantic = SemanticComparisonResponse(
                comparisons=[
                    RequirementComparison(
                        requirement=requirement.text,
                        matches=[
                            PassageMatch(
                                excerpt="Fictional retrieval passage.",
                                similarity=similarity,
                            )
                        ],
                    )
                    for requirement, similarity in zip(
                        job.requirements,
                        candidate["semantic_scores"],
                        strict=True,
                    )
                ]
            )
            ranked_inputs.append(
                score_candidate(
                    AssessmentInput(
                        candidate_id=candidate["candidate_id"],
                        job=job,
                        skills=extract_skills(candidate["source_text"]),
                        semantic=semantic,
                        evidence=EvidenceExtractionResponse(
                            results=requirement_evidence
                        ),
                        source_passages=[
                            SourcePassageInput(
                                id="source-1",
                                text=candidate["source_text"],
                            )
                        ],
                    )
                )
            )

        self.assertEqual(
            [
                [
                    result.status
                    for result in assessment.requirement_assessments
                ]
                for assessment in ranked_inputs
            ],
            [candidate["expected_statuses"] for candidate in evaluation["candidates"]],
        )
        self.assertEqual(
            [
                item.candidate_id
                for item in rank_assessments(ranked_inputs)
            ],
            evaluation["expected_ranking"],
        )
        self.assertEqual(
            [item.overall_score for item in ranked_inputs],
            [48.9, 80.57],
        )


class AssessmentApiTests(unittest.TestCase):
    def test_job_validation_returns_normalized_stable_ids(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/api/jobs/validate",
                json={
                    "title": "  Platform Engineer ",
                    "description": "  Build services. ",
                    "required_skills": [" Python ", "Python"],
                    "requirements": [{"text": " REST API development "}],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["title"], "Platform Engineer")
        self.assertEqual(body["description"], "Build services.")
        self.assertEqual(body["required_skills"], ["Python"])
        self.assertRegex(body["requirements"][0]["id"], r"^req-rest-api-development-")

    def test_score_endpoint_revalidates_exact_source_quotes(self) -> None:
        payload = make_assessment().model_dump(mode="json")
        with TestClient(app) as client:
            response = client.post("/api/assessment/score", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["requirement_assessments"][1]["status"], "explicitly_contradicted")

        payload["evidence"]["results"][0]["evidence"][0]["quote"] = (
            "Paraphrased FastAPI work."
        )
        payload["evidence"]["results"][0]["explanation"] = (
            "The resume says: Paraphrased FastAPI work."
        )
        with TestClient(app) as client:
            rejected = client.post("/api/assessment/score", json=payload)
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(
            rejected.json()["detail"],
            "Assessment evidence did not pass source validation.",
        )

    def test_health_is_independent_of_unavailable_database(self) -> None:
        with patch.dict("os.environ", {"DATABASE_URL": ""}), patch(
            "app.postgres_store.dotenv_values",
            return_value={},
        ):
            with TestClient(app) as client:
                health = client.get("/health")
                database = client.get("/health/database")
                jobs = client.get("/api/jobs")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(database.status_code, 503)
        self.assertEqual(jobs.status_code, 503)

    def test_save_recomputes_scores_before_persisting(self) -> None:
        assessment_input = make_assessment()
        job = assessment_input.job
        computed = score_candidate(assessment_input)
        stored_payload = {
            "job_id": "00000000-0000-4000-8000-000000000001",
            "job": job.model_dump(mode="json"),
            "assessments": [computed.model_dump(mode="json")],
        }
        with patch("app.main.create_job", return_value=stored_payload) as create:
            with TestClient(app) as client:
                response = client.post(
                    "/api/jobs",
                    json={
                        "job": job.model_dump(mode="json"),
                        "assessments": [assessment_input.model_dump(mode="json")],
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["assessments"][0]["overall_score"], 28.9)
        self.assertNotIn(CASEY_TEXT, response.text)
        create.assert_called_once()

    def test_rank_and_delete_routes_are_wired(self) -> None:
        assessment = score_candidate(make_assessment())
        job_id = "00000000-0000-4000-8000-000000000001"
        with patch("app.main.delete_job", return_value=True):
            with TestClient(app) as client:
                ranked = client.post(
                    "/api/assessment/rank",
                    json={"assessments": [assessment.model_dump(mode="json")]},
                )
                deleted = client.delete(f"/api/jobs/{job_id}")

        self.assertEqual(ranked.status_code, 200)
        self.assertEqual(ranked.json()[0]["rank"], 1)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json(), {"status": "deleted"})


class PostgresStoreTests(unittest.TestCase):
    def test_missing_database_url_fails_explicitly(self) -> None:
        with patch.dict("os.environ", {"DATABASE_URL": ""}), patch(
            "app.postgres_store.dotenv_values",
            return_value={},
        ):
            with self.assertRaises(DatabaseUnavailableError):
                check_database()

    def test_create_job_stores_only_validated_assessment_report(self) -> None:
        connection = MagicMock()
        connection.__enter__.return_value = connection
        with patch("app.postgres_store._connect", return_value=connection):
            stored = create_job(make_job(), [score_candidate(make_assessment())])

        self.assertEqual(len(stored["job_id"]), 36)
        self.assertEqual(len(stored["assessments"]), 1)
        self.assertEqual(connection.execute.call_count, 2)
        insert_assessment = connection.execute.call_args_list[1]
        self.assertIn("candidate_assessments", insert_assessment.args[0])
        self.assertNotIn(CASEY_TEXT, str(insert_assessment.args[1]))

    def test_delete_job_uses_cascading_job_record(self) -> None:
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.return_value.rowcount = 1
        with patch("app.postgres_store._connect", return_value=connection):
            self.assertTrue(delete_job("00000000-0000-4000-8000-000000000001"))
        self.assertIn("DELETE FROM jobs", connection.execute.call_args.args[0])
