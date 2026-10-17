import json
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from app.job_assessment import CandidateAssessment, JobDefinition


class DatabaseUnavailableError(RuntimeError):
    pass


def _connect() -> Any:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        database_url = (dotenv_values(env_path).get("DATABASE_URL") or "").strip()
    if not database_url:
        raise DatabaseUnavailableError("DATABASE_URL is not configured.")
    try:
        import psycopg

        return psycopg.connect(database_url, connect_timeout=5)
    except ImportError as error:
        raise DatabaseUnavailableError(
            "The PostgreSQL driver is not installed."
        ) from error
    except Exception as error:
        raise DatabaseUnavailableError(
            "The configured PostgreSQL database is unavailable."
        ) from error


def check_database() -> None:
    try:
        with _connect() as connection:
            connection.execute("SELECT 1")
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise DatabaseUnavailableError(
            "The configured PostgreSQL database is unavailable."
        ) from error


def migrate_database() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1] / "migrations" / "001_initial.sql"
    )
    migration_sql = migration_path.read_text(encoding="utf-8")
    try:
        with _connect() as connection:
            connection.execute(migration_sql)
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise DatabaseUnavailableError(
            "The PostgreSQL migration could not be applied."
        ) from error


def create_job(
    job: JobDefinition,
    assessments: list[CandidateAssessment],
) -> dict[str, Any]:
    identifier = str(uuid.uuid4())
    try:
        from psycopg.types.json import Jsonb

        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (job_id, title, description, job_data)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    identifier,
                    job.title,
                    job.description,
                    Jsonb(job.model_dump(mode="json")),
                ),
            )
            for assessment in assessments:
                connection.execute(
                    """
                    INSERT INTO candidate_assessments
                        (job_id, candidate_id, assessment_data)
                    VALUES (%s, %s, %s)
                    """,
                    (
                        identifier,
                        assessment.candidate_id,
                        Jsonb(assessment.model_dump(mode="json")),
                    ),
                )
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise DatabaseUnavailableError(
            "The job data could not be saved to PostgreSQL."
        ) from error
    return {
        "job_id": identifier,
        "job": job.model_dump(mode="json"),
        "assessments": [
            assessment.model_dump(mode="json") for assessment in assessments
        ],
    }


def list_jobs() -> list[dict[str, Any]]:
    try:
        with _connect() as connection:
            rows = connection.execute(
                """
                SELECT j.job_id, j.job_data, j.created_at,
                       COALESCE(
                           jsonb_agg(c.assessment_data)
                               FILTER (WHERE c.candidate_id IS NOT NULL),
                           '[]'::jsonb
                       ) AS assessments
                FROM jobs AS j
                LEFT JOIN candidate_assessments AS c USING (job_id)
                GROUP BY j.job_id
                ORDER BY j.created_at DESC
                """
            ).fetchall()
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise DatabaseUnavailableError(
            "Saved jobs could not be loaded from PostgreSQL."
        ) from error
    return [
        {
            "job_id": str(row[0]),
            "job": row[1] if isinstance(row[1], dict) else json.loads(row[1]),
            "created_at": row[2].isoformat(),
            "assessments": (
                row[3]
                if isinstance(row[3], list)
                else json.loads(row[3] or "[]")
            ),
        }
        for row in rows
    ]


def delete_job(job_id: str) -> bool:
    try:
        with _connect() as connection:
            result = connection.execute(
                "DELETE FROM jobs WHERE job_id = %s",
                (job_id,),
            )
            return result.rowcount > 0
    except DatabaseUnavailableError:
        raise
    except Exception as error:
        raise DatabaseUnavailableError(
            "The job could not be deleted from PostgreSQL."
        ) from error
