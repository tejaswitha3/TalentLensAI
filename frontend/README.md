# TalentLens AI frontend

For full Mac setup, architecture, scoring, privacy, PostgreSQL, Docker, and
evaluation documentation, see the [project README](../README.md).

The browser sends requests to local Next.js route handlers. Those handlers use
`BACKEND_URL` (default `http://127.0.0.1:8000`) to reach the FastAPI backend.

## Run locally

Start the backend from the workspace root in one terminal:

```bash
backend/.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Start the frontend from the workspace root in a second terminal:

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Local evidence check

The website offers one evidence action: **Check evidence locally**. It uses a
small curated skill dictionary and explicit-negation patterns to cite exact
resume excerpts. Unrecognized, compound, or ambiguous requirements remain
unclear. These classifications are not verified proficiency or hiring decisions.

No resume text is sent to Gemini through the website. Run **Compare
requirements** separately for a complete matching measure; the evidence
method appears in the report and CSV export. No Gemini key is required.

## Local semantic comparison

Enter up to 10 job requirements, one per line, then choose **Compare
requirements** for an extracted resume. The backend uses the local
`sentence-transformers/all-MiniLM-L6-v2` model to retrieve up to three resume
passages per requirement. Resume text is limited to 100,000 characters per
comparison, and each requirement is limited to 500 characters.

The first comparison downloads the model files from Hugging Face and needs an
internet connection. The model is licensed under Apache-2.0. Its files are
cached by Hugging Face outside this workspace; after download, inference runs
locally and uses Apple MPS when available, otherwise CPU. Resume text is not
sent to Hugging Face.

Similarity values are semantic similarity only, not proficiency scores or
match probabilities. Retrieved passages—including negated statements—do not
prove that a requirement is satisfied. Review the source excerpts and extracted
resume text.

## Job dashboard, scoring, CSV, and saved jobs

The dashboard accepts a title, description, required and preferred skills, and
up to 10 explicit requirements. Required skills have twice the coverage
weight of preferred skills. After both local semantic retrieval and confirmed
fictional evidence are available, the backend returns the skill coverage,
semantic relevance, and evidence relevance components, their weighted
arithmetic, and the overall matching measure. A failed/missing component makes
the assessment incomplete without an overall score. Similarity is forced to
zero for an explicitly contradicted requirement so retrieval cannot override
the contradiction.

The ranked table offers complete/incomplete filters, score sorting, stable
ties, and side-by-side evidence excerpts. Candidate detail keeps the extracted
text inspectable. CSV contains opaque candidate IDs, scores only when valid,
skill/requirement summaries, and status; it excludes filenames and resume text
and neutralizes spreadsheet formula prefixes.

Uploads and full extracted text remain in browser memory. PostgreSQL is
optional and only stores the job and validated assessment report (including
short cited excerpts), not the original PDF or full extracted text. Start a
PostgreSQL server, set `DATABASE_URL`, and run `python -m app.migrate` from the
backend directory to enable persistence. The saved-jobs panel reports
unavailable database state without blocking extraction or local analysis.
Deleting a saved job deletes its associated candidate assessment records;
**Remove** clears the candidate's PDF and in-memory assessment for the current
browser session.
