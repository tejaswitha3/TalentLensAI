# TalentLens AI

TalentLens AI is a local recruiter-assistance MVP for extracting selectable
text from PDFs, detecting dictionary-based skill mentions, retrieving
semantically similar passages, and producing source-checked fictional-demo
evidence assessments. It reports a transparent **matching measure**, not a
probability of hiring success. It never makes hiring decisions.

## Status and limitations

- PDF upload, selectable-text extraction, skill mentions/negations, local
  semantic retrieval, optional source-validated Gemini evidence, a limited
  offline dictionary evidence check, deterministic scoring,
  ranking, CSV export, and optional PostgreSQL persistence are implemented.
- The website uses the local evidence check only. The older Gemini backend
  endpoint remains in the code for API compatibility, but the website does not
  call it or display an external-provider action.
- **Check evidence locally** makes no Gemini request. It recognizes only simple
  single-skill requirements from the curated dictionary and checks clear action
  or explicit negation patterns in exact source excerpts. Unrecognized,
  compound, ambiguous, or conflicting statements remain unclear. Its label
  distinguishes the local rule check from the optional model assessment; a
  local positive means the text matched a pattern, not verified experience.
- Gemini is asked for JSON without schema-constrained generation. The backend
  validates its output and citations; exact quotes are copied from source
  passages by the backend.
- Sentence-transformer similarity is retrieval information, not proof of
  proficiency. Dictionary mentions are also not proof. Review the extracted
  text and exact source excerpts.
- Missing evidence is **unclear/unverified**, not evidence that a candidate
  lacks a skill. Explicit contradictions remain distinct and cannot be
  overridden by a high semantic similarity.
- The current app has no authentication or multi-tenant boundary and is a
  local-development MVP, not a production hiring system. Use fictional
  candidates only. Do not publish it with real applicant data.
- PostgreSQL is optional. The backend health route works without a database;
  saved-job operations report database unavailability separately. PostgreSQL
  stores job criteria, opaque candidate IDs, scores, statuses, and cited
  excerpts, but not uploaded PDFs or full extracted resume text. Deleting a
  saved job cascades to its assessment rows. Removing a candidate clears its
  PDF and in-memory analysis from the current browser session.
- No pgvector dependency is used: semantic retrieval remains local and has no
  database-backed search path.

## Architecture and request flow

```text
Safari
  -> Next.js UI and same-origin route handlers
      -> FastAPI
          -> PyMuPDF: PDF text extraction
          -> local dictionary: skill mentions and explicit negations
          -> local sentence-transformers model: passage retrieval only
          -> local dictionary evidence check, or Google Gemini Developer API
             (optional, confirmed fictional data)
             (API key remains in the backend)
          -> deterministic Python scorer and ranker
          -> optional PostgreSQL: job and validated assessment metadata
```

The main routes are `POST /api/resumes/extract`,
`POST /api/resumes/compare`, `POST /api/resumes/evidence`,
`POST /api/resumes/evidence/local`,
`POST /api/jobs/validate`, `POST /api/assessment/score`,
`POST /api/assessment/rank`, and the optional saved-job routes
`GET/POST /api/jobs` and `DELETE /api/jobs/{job_id}`. `/health` does not depend
on PostgreSQL; `/health/database` reports the separate database state.

The frontend supports up to 10 PDFs in one selection, per-file upload limits,
extraction progress and errors, job title/description/required and preferred
skills/explicit requirements, a ranked table with sorting and filters,
component visualization, candidate detail and side-by-side evidence, CSV
export, and session removal. The job requirements textarea preserves the
manual one-requirement-per-line workflow. The browser stores the PDFs and
extracted full text in memory only.

Choose **Check evidence locally** for the website's evidence step. It makes no
Gemini call. The older backend API endpoint is retained for compatibility but
is not used by the website.
For a complete matching measure, run **Compare requirements** as well; the
semantic component is independent of either evidence path.

## Matching formula

For a complete assessment, each component is normalized to 0–100:

1. **Skill coverage (50%)**: each required skill has weight 2 and each
   preferred skill has weight 1. A dictionary-recognized positive mention
   contributes 100; absent mentions contribute 0. If a skill is explicitly
   negated, its contribution is 0 even if a positive mention also exists.
   This is mention coverage, not proficiency verification.
2. **Semantic relevance (30%)**: for each explicit requirement, take the best
   returned cosine similarity, clamp it to `[0, 1]`, and multiply by 100.
   An explicit contradiction forces that requirement's semantic contribution
   to 0, regardless of similarity. Missing retrieval matches contribute 0.
   Required explicit requirements have weight 2, preferred requirements
   weight 1.
3. **Evidence relevance (20%)**: supported = 100, explicitly contradicted =
   0, and unclear/unverified = 50. Required explicit requirements have weight
   2, preferred requirements weight 1. Evidence statuses are either model
   assessments or local rule classifications, as labeled in the UI; both need
   human review.

The overall matching measure is:

```text
(skill coverage × 0.50) + (semantic relevance × 0.30)
  + (evidence relevance × 0.20)
```

The API returns each component's weighted numerator and denominator as well
as the arithmetic. A component with no applicable entries is neutral (100),
so jobs with no explicit requirements can still be scored from skill inputs.
A job with no skills and no explicit requirements is marked incomplete rather
than assigned a meaningless score. A missing or failed semantic/evidence
assessment makes the overall assessment incomplete with no overall score.
Ranking is descending, deterministic, preserves input order for exact ties,
uses competition ranks for ties, and does not rank incomplete assessments.
Names, photographs, gender, age, and other unrelated personal characteristics
are not inputs to scoring.

## macOS local setup

Install Python 3.13 and Node.js 22 or later (Next.js 16 requires Node 20.9+).
Homebrew commands, if needed:

```bash
brew install python@3.13 node
```

From the workspace root:

```bash
python3.13 -m venv backend/.venv
backend/.venv/bin/python -m pip install --upgrade pip
backend/.venv/bin/python -m pip install -r backend/requirements.txt
cd frontend
npm ci
cd ..
```

No Gemini key is needed for the website. An existing private `backend/.env`
file can remain in place; the local evidence action does not read or send its
key. Backend-only integrations with the older Gemini endpoint still require a
key and explicit fictional-data confirmation. Keep `.env` private.

Start the backend in one terminal:

```bash
cd /path/to/TalentLensAI
backend/.venv/bin/python -m uvicorn app.main:app \
  --app-dir backend --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```bash
cd /path/to/TalentLensAI/frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The first semantic
comparison downloads `sentence-transformers/all-MiniLM-L6-v2` from
Hugging Face; once downloaded, inference is local (Apple MPS when available,
otherwise CPU). Resume text is not sent to Hugging Face.

To make the local PostgreSQL URL in `backend/.env` available to Uvicorn, start
it with `--env-file backend/.env`. Database URL settings are optional. Without
`DATABASE_URL`, health and local analysis remain available; saved-job actions
show an unavailable state.

## PostgreSQL (optional)

PostgreSQL is not required for app health or local analysis. With a local
PostgreSQL server, set a private `DATABASE_URL` in the environment or
`backend/.env`, then apply the checked-in migration:

```bash
cd /path/to/TalentLensAI
cd backend
.venv/bin/python -m app.migrate
```

The migration creates `jobs` and `candidate_assessments`; deleting a job
cascades to its assessments. The app revalidates exact source citations and
recomputes scores before saving. The database stores no PDF and no full
extracted resume text. Local database availability was not verifiable in the
current environment.

To install a local PostgreSQL 17 server on macOS using Homebrew:

```bash
brew install postgresql@17
brew services start postgresql@17
"$(brew --prefix postgresql@17)/bin/createuser" -P talentlens
"$(brew --prefix postgresql@17)/bin/createdb" -O talentlens talentlens
```

The `createuser` command prompts for a local-only password without echoing it.
Set `DATABASE_URL` in the private backend environment file to the matching
local database URL (URL-encode reserved password characters) before applying
the migration.

## Docker

Docker Compose files are provided for the frontend, backend, and PostgreSQL.
Docker was not available in the current environment, so image builds and
container startup have not been verified.

With Docker Desktop installed, create the backend environment file as above
and replace `POSTGRES_PASSWORD` with a local-only password, then from the
project root:

```bash
docker compose --env-file backend/.env up --build -d
docker compose exec backend python -m app.migrate
```

Open [http://localhost:3000](http://localhost:3000). The backend container's
`/health` check does not query PostgreSQL. To stop containers while keeping
the local database volume:

```bash
docker compose down
```

To remove the database volume and its saved data, use
`docker compose down --volumes` only when you explicitly intend that
destructive removal. No deployment or paid service has been enabled. Before
any deployment, add authentication/authorization, TLS and a reverse proxy,
rate limits, secrets management, data-retention controls, privacy/legal review,
and an appropriately governed model/data-processing arrangement. Do not
deploy this demo with real applicant data.

## Evaluation set and results

`backend/evaluation/fictional_cases.json` contains two small fictional cases
with expected supported, contradicted, and unclear statuses and a fixed
ranking order. The evaluation test exercises the deterministic scoring and
ranking logic using those labels and scores: Jordan Example ranks above Casey
Example (80.57 vs. 48.90 in the checked fixture). This verifies arithmetic and
tie/status handling only; it is not an accuracy, fairness, or hiring-success
claim, and it does not measure provider reliability. See the regression tests
for the actual HTTP/Docker/Kubernetes source examples.

## Tests

Run the full backend suite:

```bash
cd /path/to/TalentLensAI/backend
.venv/bin/python -m unittest discover -s tests
```

Run frontend checks:

```bash
cd /path/to/TalentLensAI/frontend
npm run lint
npm test
npm run build
```

The backend tests cover valid/invalid/oversized/encrypted/text-empty PDFs,
fictional extraction-to-scoring API flow with a mocked provider, exact quote
validation, missing and contradictory evidence, provider failures, stable
ranking, and optional-database behavior. CSV unit tests cover quoting and
formula injection. No live Gemini generation request is part of the test
suite.

## Technology notes for interviews

- **FastAPI / Pydantic**: typed HTTP contracts and server-side validation.
- **PyMuPDF**: extracts selectable text from PDFs; scanned-image OCR is not
  implemented.
- **Sentence Transformers**: local semantic retrieval, used only to surface
  passages for review.
- **Google GenAI SDK**: optional fictional demo assessment with a server-side
  key and explicit confirmation; provider output is not trusted as a quote.
- **PostgreSQL**: optional persistence for job definitions and validated
  assessment reports; no vector extension is needed for this local retrieval
  path.
- **Next.js / React / TypeScript**: responsive recruiter interface and
  same-origin server-side API proxies.
- **Docker Compose**: reproducible local app/database packaging; not verified
  without Docker installed.
