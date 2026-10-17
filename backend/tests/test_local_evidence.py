import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.evidence_extraction import EvidenceExtractionRequest
from app.local_evidence import check_local_evidence
from app.main import app
from app.job_assessment import AssessmentInput, score_candidate
from app.skill_extraction import extract_skills


class LocalEvidenceTests(unittest.TestCase):
    def test_casey_uses_exact_source_and_never_calls_gemini(self):
        source = (
            "Built HTTP endpoints using FastAPI. "
            "Python, FastAPI, and REST API development in a student project. "
            "I have no experience\nwith Docker."
        )
        payload = {
            "confirmed_fictional_demo": True,
            "requirements": [
                {"id": "api", "text": "REST API development"},
                {"id": "docker", "text": "Docker hands-on experience"},
                {"id": "k8s", "text": "Kubernetes experience"},
            ],
            "source_passages": [{"id": "source-1", "text": source}],
        }
        with patch("app.evidence_extraction._call_gemini") as provider:
            response = TestClient(app).post("/api/resumes/evidence/local", json=payload)
        provider.assert_not_called()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["label"], "Local dictionary check — review required")
        self.assertEqual(
            [item["status"] for item in body["results"]],
            ["supported", "explicitly_contradicted", "unclear"],
        )
        self.assertEqual(body["results"][2]["evidence"], [])
        for result in body["results"]:
            for evidence in result["evidence"]:
                self.assertIn(evidence["quote"], source)
        self.assertIn("\n", body["results"][1]["evidence"][0]["quote"])

    def test_ambiguous_and_compound_requirements_remain_unclear(self):
        request = EvidenceExtractionRequest.model_validate(
            {
                "confirmed_fictional_demo": True,
                "requirements": [
                    {"id": "a", "text": "Python experience"},
                    {"id": "b", "text": "Python and Docker"},
                    {"id": "c", "text": "Five years of Python"},
                ],
                "source_passages": [
                    {"id": "s", "text": "Learning Python. Built a Java project."}
                ],
            }
        )
        results = check_local_evidence(request).results
        self.assertTrue(all(item.status == "unclear" for item in results))
        self.assertEqual(results[0].evidence[0].relationship, "context")

    def test_conflicting_mentions_are_not_declared_supported(self):
        request = EvidenceExtractionRequest.model_validate(
            {
                "confirmed_fictional_demo": True,
                "requirements": [{"id": "a", "text": "Docker experience"}],
                "source_passages": [
                    {"id": "s", "text": "Used Docker. I have no experience with Docker."}
                ],
            }
        )
        result = check_local_evidence(request).results[0]
        self.assertEqual(result.status, "unclear")
        self.assertEqual(result.evidence, [])

    def test_malformed_request_is_rejected(self):
        response = TestClient(app).post(
            "/api/resumes/evidence/local",
            json={
                "confirmed_fictional_demo": True,
                "requirements": [{"id": "one", "text": "Docker"}],
                "source_passages": [],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_score_preserves_local_method_for_saved_reports(self):
        source = "Built a Python project."
        request = EvidenceExtractionRequest.model_validate(
            {
                "confirmed_fictional_demo": True,
                "requirements": [{"id": "req-python", "text": "Python experience"}],
                "source_passages": [{"id": "source-1", "text": source}],
            }
        )
        scored = score_candidate(
            AssessmentInput.model_validate(
                {
                    "candidate_id": "fictional",
                    "job": {
                        "title": "Demo",
                        "requirements": [
                            {"id": "req-python", "text": "Python experience"}
                        ],
                    },
                    "skills": extract_skills(source).model_dump(),
                    "semantic": {
                        "comparisons": [
                            {
                                "requirement": "Python experience",
                                "matches": [
                                    {"excerpt": source, "similarity": 0.5}
                                ],
                            }
                        ]
                    },
                    "evidence": check_local_evidence(request).model_dump(),
                    "source_passages": [item.model_dump() for item in request.source_passages],
                }
            )
        )
        self.assertTrue(scored.complete)
        self.assertEqual(scored.evidence_method, "Local dictionary check — review required")
