CREATE TABLE IF NOT EXISTS jobs (
    job_id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    job_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS candidate_assessments (
    job_id UUID NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL,
    assessment_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (job_id, candidate_id)
);

CREATE INDEX IF NOT EXISTS candidate_assessments_job_id_idx
    ON candidate_assessments (job_id);
