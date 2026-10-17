"use client";

import { useEffect, useRef, useState } from "react";
import { buildCsv } from "../lib/csv-export.mjs";

type ConnectionStatus = "loading" | "connected" | "unavailable";
type FileStatus = "queued" | "processing" | "completed" | "failed";
type SkillMention = {
  matched_wording: string;
  source_excerpt: string;
};
type SkillEvidence = {
  canonical_name: string;
  label: "Skill mentioned" | "Negated mention";
  mentions: SkillMention[];
};
type ExtractionResult = {
  filename: string;
  page_count: number;
  extracted_text: string;
  warnings: string[];
  mentioned_skills: SkillEvidence[];
  negated_skills: SkillEvidence[];
};
type PassageMatch = {
  excerpt: string;
  similarity: number;
};
type RequirementComparison = {
  requirement: string;
  matches: PassageMatch[];
};
type SemanticComparisonResult = {
  comparisons: RequirementComparison[];
};
type EvidenceQuote = {
  source_id: string;
  relationship: "supports" | "contradicts" | "context";
  quote: string;
};
type RequirementEvidence = {
  requirement_id: string;
  status: "supported" | "explicitly_contradicted" | "unclear";
  explanation: string;
  evidence: EvidenceQuote[];
};
type EvidenceAnalysisResult = {
  label:
    | "Model-generated assessment — review required"
    | "Local dictionary check — review required";
  results: RequirementEvidence[];
};
type JobRequirement = {
  id: string;
  text: string;
  importance: "required" | "preferred";
};
type JobDefinition = {
  title: string;
  description: string;
  required_skills: string[];
  preferred_skills: string[];
  requirements: JobRequirement[];
};
type ComponentScore = {
  score: number;
  numerator: number;
  denominator: number;
};
type RequirementAssessment = {
  requirement_id: string;
  requirement: string;
  status: "supported" | "explicitly_contradicted" | "unclear";
  evidence: { source_id: string; relationship: string; quote: string }[];
  similarity: number | null;
};
type CandidateAssessment = {
  candidate_id: string;
  complete: boolean;
  incomplete_reason: string | null;
  skill_coverage: ComponentScore | null;
  semantic_relevance: ComponentScore | null;
  evidence_relevance: ComponentScore | null;
  overall_score: number | null;
  arithmetic: string | null;
  requirement_assessments: RequirementAssessment[];
  matched_skills: string[];
  rank: number | null;
  evidence_method?: EvidenceAnalysisResult["label"] | null;
};
type SavedJob = {
  job_id: string;
  job: JobDefinition;
  created_at: string;
  assessments: CandidateAssessment[];
};
type UploadItem = {
  id: number;
  candidate_key: string;
  file: File;
  status: FileStatus;
  result?: ExtractionResult;
  error?: string;
  comparison?: SemanticComparisonResult;
  comparisonError?: string;
  comparisonStatus?: "processing" | "completed" | "failed";
  evidenceResult?: EvidenceAnalysisResult;
  evidenceError?: string;
  evidenceStatus?: "processing" | "completed" | "failed";
  job?: JobDefinition;
  assessment?: CandidateAssessment;
  assessmentError?: string;
};

const MAX_FILES = 10;
const MAX_REQUIREMENTS = 10;
const MAX_REQUIREMENT_LENGTH = 500;
const MAX_RESUME_TEXT_LENGTH = 100_000;
const MAX_EVIDENCE_SOURCE_LENGTH = 30_000;
const MAX_EVIDENCE_PASSAGE_LENGTH = 2_000;
const MAX_EVIDENCE_PASSAGES = 20;

function isConnectedResponse(value: unknown): value is { status: "connected" } {
  return (
    typeof value === "object" &&
    value !== null &&
    "status" in value &&
    value.status === "connected"
  );
}

function isSkillEvidence(value: unknown): value is SkillEvidence {
  return (
    typeof value === "object" &&
    value !== null &&
    "canonical_name" in value &&
    typeof value.canonical_name === "string" &&
    "label" in value &&
    (value.label === "Skill mentioned" || value.label === "Negated mention") &&
    "mentions" in value &&
    Array.isArray(value.mentions) &&
    value.mentions.every(
      (mention) =>
        typeof mention === "object" &&
        mention !== null &&
        "matched_wording" in mention &&
        typeof mention.matched_wording === "string" &&
        "source_excerpt" in mention &&
        typeof mention.source_excerpt === "string",
    )
  );
}

function isExtractionResult(value: unknown): value is ExtractionResult {
  return (
    typeof value === "object" &&
    value !== null &&
    "filename" in value &&
    typeof value.filename === "string" &&
    "page_count" in value &&
    typeof value.page_count === "number" &&
    "extracted_text" in value &&
    typeof value.extracted_text === "string" &&
    "warnings" in value &&
    Array.isArray(value.warnings) &&
    value.warnings.every((warning) => typeof warning === "string") &&
    "mentioned_skills" in value &&
    Array.isArray(value.mentioned_skills) &&
    value.mentioned_skills.every(isSkillEvidence) &&
    "negated_skills" in value &&
    Array.isArray(value.negated_skills) &&
    value.negated_skills.every(isSkillEvidence)
  );
}

function isSemanticComparisonResult(
  value: unknown,
): value is SemanticComparisonResult {
  return (
    typeof value === "object" &&
    value !== null &&
    "comparisons" in value &&
    Array.isArray(value.comparisons) &&
    value.comparisons.every(
      (comparison: unknown) =>
        typeof comparison === "object" &&
        comparison !== null &&
        "requirement" in comparison &&
        typeof comparison.requirement === "string" &&
        "matches" in comparison &&
        Array.isArray(comparison.matches) &&
        comparison.matches.every(
          (match: unknown) =>
            typeof match === "object" &&
            match !== null &&
            "excerpt" in match &&
            typeof match.excerpt === "string" &&
            "similarity" in match &&
            typeof match.similarity === "number" &&
            Number.isFinite(match.similarity),
        ),
    )
  );
}

function isEvidenceAnalysisResult(
  value: unknown,
): value is EvidenceAnalysisResult {
  return (
    typeof value === "object" &&
    value !== null &&
    "label" in value &&
    (value.label === "Model-generated assessment — review required" ||
      value.label === "Local dictionary check — review required") &&
    "results" in value &&
    Array.isArray(value.results) &&
    value.results.every(
      (result: unknown) =>
        typeof result === "object" &&
        result !== null &&
        "requirement_id" in result &&
        typeof result.requirement_id === "string" &&
        "status" in result &&
        (result.status === "supported" ||
          result.status === "explicitly_contradicted" ||
          result.status === "unclear") &&
        "explanation" in result &&
        typeof result.explanation === "string" &&
        "evidence" in result &&
        Array.isArray(result.evidence) &&
        result.evidence.every(
          (quote: unknown) =>
            typeof quote === "object" &&
            quote !== null &&
            "source_id" in quote &&
            typeof quote.source_id === "string" &&
            "relationship" in quote &&
            (quote.relationship === "supports" ||
              quote.relationship === "contradicts" ||
              quote.relationship === "context") &&
            "quote" in quote &&
            typeof quote.quote === "string",
        ),
    )
  );
}

function isJobDefinition(value: unknown): value is JobDefinition {
  return (
    typeof value === "object" &&
    value !== null &&
    "title" in value &&
    typeof value.title === "string" &&
    "description" in value &&
    typeof value.description === "string" &&
    "required_skills" in value &&
    Array.isArray(value.required_skills) &&
    "preferred_skills" in value &&
    Array.isArray(value.preferred_skills) &&
    "requirements" in value &&
    Array.isArray(value.requirements) &&
    value.requirements.every(
      (requirement: unknown) =>
        typeof requirement === "object" &&
        requirement !== null &&
        "id" in requirement &&
        typeof requirement.id === "string" &&
        "text" in requirement &&
        typeof requirement.text === "string" &&
        "importance" in requirement &&
        (requirement.importance === "required" ||
          requirement.importance === "preferred"),
    )
  );
}

function isCandidateAssessment(value: unknown): value is CandidateAssessment {
  return (
    typeof value === "object" &&
    value !== null &&
    "candidate_id" in value &&
    typeof value.candidate_id === "string" &&
    "complete" in value &&
    typeof value.complete === "boolean" &&
    "overall_score" in value &&
    (typeof value.overall_score === "number" || value.overall_score === null) &&
    "requirement_assessments" in value &&
    Array.isArray(value.requirement_assessments) &&
    "matched_skills" in value &&
    Array.isArray(value.matched_skills)
  );
}

function isSavedJobs(value: unknown): value is SavedJob[] {
  return (
    Array.isArray(value) &&
    value.every(
      (job) =>
        typeof job === "object" &&
        job !== null &&
        "job_id" in job &&
        typeof job.job_id === "string" &&
        "job" in job &&
        isJobDefinition(job.job) &&
        "created_at" in job &&
        typeof job.created_at === "string" &&
        "assessments" in job &&
        Array.isArray(job.assessments) &&
        job.assessments.every(isCandidateAssessment),
    )
  );
}

function splitEvidenceSource(text: string) {
  if (text.length > MAX_EVIDENCE_SOURCE_LENGTH) {
    throw new Error(
      `Resume text exceeds the external demo analysis limit of ${MAX_EVIDENCE_SOURCE_LENGTH.toLocaleString()} characters. No text was sent.`,
    );
  }

  const passages: { id: string; text: string }[] = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + MAX_EVIDENCE_PASSAGE_LENGTH, text.length);
    if (end < text.length) {
      const boundary = Math.max(
        text.lastIndexOf("\n", end),
        text.lastIndexOf(" ", end),
      );
      if (boundary > start) {
        end = boundary;
      }
    }
    const passageText = text.slice(start, end);
    if (passageText.trim()) {
      passages.push({ id: `source-${passages.length + 1}`, text: passageText });
    }
    start = end;
  }

  if (passages.length === 0 || passages.length > MAX_EVIDENCE_PASSAGES) {
    throw new Error("Resume text could not be divided into allowed source passages.");
  }
  return passages;
}

function parseSkillList(value: string) {
  return value
    .split(/[,\r\n]/)
    .map((skill) => skill.trim())
    .filter(Boolean);
}

function candidateIdentifier(item: UploadItem) {
  return `candidate-${item.candidate_key}`;
}

export default function Home() {
  const [status, setStatus] = useState<ConnectionStatus>("loading");
  const [items, setItems] = useState<UploadItem[]>([]);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [jobRequirements, setJobRequirements] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [requiredSkills, setRequiredSkills] = useState("");
  const [preferredSkills, setPreferredSkills] = useState("");
  const [requirementsError, setRequirementsError] = useState<string | null>(
    null,
  );
  const [assessmentFilter, setAssessmentFilter] = useState<
    "all" | "complete" | "incomplete"
  >("all");
  const [sortAscending, setSortAscending] = useState(false);
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<string[]>(
    [],
  );
  const [savedJobs, setSavedJobs] = useState<SavedJob[]>([]);
  const [databaseMessage, setDatabaseMessage] = useState<string | null>(null);
  const [isSavingJob, setIsSavingJob] = useState(false);
  const [rankByCandidateId, setRankByCandidateId] = useState<
    Record<string, number | null>
  >({});
  const [rankingError, setRankingError] = useState<string | null>(null);
  const [isComparing, setIsComparing] = useState(false);
  const [isAnalyzingEvidence, setIsAnalyzingEvidence] = useState(false);
  const [isPreparingEvidence, setIsPreparingEvidence] =
    useState(false);
  const processingRef = useRef(false);
  const comparisonRef = useRef(false);
  const evidenceRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function checkBackend() {
      try {
        const response = await fetch("/api/backend-health", {
          cache: "no-store",
          signal: controller.signal,
        });
        const result: unknown = await response.json();

        setStatus(
          response.ok && isConnectedResponse(result)
            ? "connected"
            : "unavailable",
        );
      } catch {
        if (!controller.signal.aborted) {
          setStatus("unavailable");
        }
      }

      if (controller.signal.aborted) return;
      try {
        const response = await fetch("/api/jobs", {
          cache: "no-store",
          signal: controller.signal,
        });
        const result: unknown = await response.json();
        if (response.ok && isSavedJobs(result)) {
          setSavedJobs(result);
          setDatabaseMessage(null);
        } else if (response.status === 503) {
          setDatabaseMessage(
            "PostgreSQL is not configured or currently unavailable. Saved jobs cannot load; local analysis remains available.",
          );
        } else if (response.status === 404) {
          setDatabaseMessage(
            "The saved-jobs route is unavailable. Restart the TalentLens AI backend and frontend.",
          );
        } else {
          setDatabaseMessage(
            "Saved jobs could not be loaded. Local analysis remains available.",
          );
        }
      } catch {
        if (!controller.signal.aborted) {
          setDatabaseMessage(
            "Saved jobs are unavailable. Local analysis remains available.",
          );
        }
      }
    }

    void checkBackend();
    return () => controller.abort();
  }, []);

  const assessmentSnapshot = JSON.stringify(
    items.flatMap((item) => (item.assessment ? [item.assessment] : [])),
  );

  useEffect(() => {
    let cancelled = false;
    const assessments: unknown = JSON.parse(assessmentSnapshot);
    if (!Array.isArray(assessments) || assessments.length === 0) {
      setRankByCandidateId({});
      setRankingError(null);
      return () => {
        cancelled = true;
      };
    }
    if (!assessments.every(isCandidateAssessment)) {
      setRankingError("Candidate ranking input could not be validated.");
      return () => {
        cancelled = true;
      };
    }

    async function rankCandidates() {
      try {
        const response = await fetch("/api/assessment/rank", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ assessments }),
        });
        const result: unknown = await response.json();
        if (
          !response.ok ||
          !Array.isArray(result) ||
          !result.every(isCandidateAssessment)
        ) {
          throw new Error("Ranking service returned an invalid response.");
        }
        if (!cancelled) {
          setRankByCandidateId(
            Object.fromEntries(
              result.map((assessment) => [
                assessment.candidate_id,
                assessment.rank,
              ]),
            ),
          );
          setRankingError(null);
        }
      } catch {
        if (!cancelled) {
          setRankingError(
            "Backend ranking is unavailable; displayed order uses the same deterministic score order.",
          );
        }
      }
    }

    void rankCandidates();
    return () => {
      cancelled = true;
    };
  }, [assessmentSnapshot]);

  const statusContent = {
    loading: {
      label: "Checking backend connection…",
      dot: "bg-amber-400",
      detail: "Contacting the local API.",
    },
    connected: {
      label: "Backend connected",
      dot: "bg-emerald-500",
      detail: "The TalentLens AI backend is responding.",
    },
    unavailable: {
      label: "Backend unavailable",
      dot: "bg-rose-500",
      detail: "Start or check the backend, then refresh this page.",
    },
  }[status];

  function selectFiles(files: FileList | null) {
    const selected = Array.from(files ?? []);
    setRequirementsError(null);
    setSelectedCandidateIds([]);
    setSelectionError(
      selected.length > MAX_FILES
        ? `Select no more than ${MAX_FILES} PDF files at a time.`
        : null,
    );
    setItems(
      selected.slice(0, MAX_FILES).map((file, id) => ({
        id,
        candidate_key: crypto.randomUUID(),
        file,
        status: "queued",
      })),
    );
  }

  async function extractText(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (
      processingRef.current ||
      comparisonRef.current ||
      evidenceRef.current
    ) {
      return;
    }
    if (selectionError) {
      return;
    }
    if (items.length === 0) {
      setSelectionError("Choose at least one PDF file before extracting text.");
      return;
    }

    processingRef.current = true;
    setIsProcessing(true);
    setItems((current) =>
      current.map((item) => ({
        ...item,
        status: "queued",
        result: undefined,
        error: undefined,
        comparison: undefined,
        comparisonError: undefined,
        comparisonStatus: undefined,
        evidenceResult: undefined,
        evidenceError: undefined,
        evidenceStatus: undefined,
        job: undefined,
        assessment: undefined,
        assessmentError: undefined,
      })),
    );
    try {
      for (const item of items) {
        setItems((current) =>
          current.map((currentItem) =>
            currentItem.id === item.id
              ? {
                  ...currentItem,
                  status: "processing",
                  result: undefined,
                  error: undefined,
                }
              : currentItem,
          ),
        );

        let result: ExtractionResult | undefined;
        let error: string | undefined;
        if (!item.file.name.toLowerCase().endsWith(".pdf")) {
          error = "This file does not have a .pdf extension.";
        } else {
          const formData = new FormData();
          formData.append("file", item.file);
          try {
            const response = await fetch("/api/resumes/extract", {
              method: "POST",
              body: formData,
            });
            const responseBody: unknown = await response.json();

            if (response.ok && isExtractionResult(responseBody)) {
              result = responseBody;
            } else {
              error =
                typeof responseBody === "object" &&
                responseBody !== null &&
                "detail" in responseBody &&
                typeof responseBody.detail === "string"
                  ? responseBody.detail
                  : "The PDF could not be processed.";
            }
          } catch {
            error = "Could not reach the upload service. Please try again.";
          }
        }

        setItems((current) =>
          current.map((currentItem) =>
            currentItem.id === item.id
              ? {
                  ...currentItem,
                  status: error ? "failed" : "completed",
                  result,
                  error,
                }
              : currentItem,
          ),
        );
      }
    } finally {
      processingRef.current = false;
      setIsProcessing(false);
    }
  }

  async function validateJobDefinition(): Promise<JobDefinition> {
    const requirementTexts = jobRequirements
      .split(/\r?\n/)
      .map((requirement) => requirement.trim())
      .filter(Boolean);
    const response = await fetch("/api/jobs/validate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        title: jobTitle,
        description: jobDescription,
        required_skills: parseSkillList(requiredSkills),
        preferred_skills: parseSkillList(preferredSkills),
        requirements: requirementTexts.map((text) => ({ text })),
      }),
    });
    const body: unknown = await response.json();
    if (!response.ok || !isJobDefinition(body)) {
      const detail =
        typeof body === "object" &&
        body !== null &&
        "detail" in body &&
        typeof body.detail === "string"
          ? body.detail
          : "The job inputs could not be validated.";
      throw new Error(
        response.status === 404
          ? "The job-validation route is unavailable. Restart the TalentLens AI backend and frontend."
          : detail,
      );
    }
    return body;
  }

  async function submitAssessment(
    item: UploadItem,
    job: JobDefinition,
    semantic: SemanticComparisonResult | undefined,
    evidence: EvidenceAnalysisResult | undefined,
    evidenceError?: string,
  ) {
    if (!item.result) return;
    try {
      const sourcePassages = splitEvidenceSource(item.result.extracted_text);
      const response = await fetch("/api/assessment/score", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          candidate_id: candidateIdentifier(item),
          job,
          skills: {
            mentioned_skills: item.result.mentioned_skills,
            negated_skills: item.result.negated_skills,
          },
          semantic: semantic ?? null,
          evidence: evidence ?? null,
          source_passages: sourcePassages,
          evidence_error: evidenceError,
        }),
      });
      const body: unknown = await response.json();
      const assessment =
        response.ok && isCandidateAssessment(body) ? body : undefined;
      setItems((current) =>
        current.map((candidate) =>
          candidate.id === item.id
            ? {
                ...candidate,
                assessment,
                assessmentError: assessment
                  ? undefined
                  : "Assessment could not be validated; no score was returned.",
              }
            : candidate,
        ),
      );
    } catch {
      setItems((current) =>
        current.map((candidate) =>
          candidate.id === item.id
            ? {
                ...candidate,
                assessment: undefined,
                assessmentError:
                  "Assessment validation could not be reached; no score was returned.",
              }
            : candidate,
        ),
      );
    }
  }

  async function compareRequirements(itemId: number) {
    if (comparisonRef.current || evidenceRef.current) {
      return;
    }

    const item = items.find((candidate) => candidate.id === itemId);
    if (!item?.result) {
      return;
    }
    let job: JobDefinition;
    try {
      job = await validateJobDefinition();
    } catch (error) {
      setRequirementsError(
        error instanceof Error ? error.message : "Job inputs are invalid.",
      );
      return;
    }
    const requirements = job.requirements.map((requirement) => requirement.text);
    if (requirements.length === 0 || requirements.length > MAX_REQUIREMENTS) {
      setRequirementsError(
        `Enter between 1 and ${MAX_REQUIREMENTS} non-empty requirements, one per line.`,
      );
      return;
    }
    if (
      requirements.some(
        (requirement) => requirement.length > MAX_REQUIREMENT_LENGTH,
      )
    ) {
      setRequirementsError(
        `Each requirement must be no more than ${MAX_REQUIREMENT_LENGTH} characters.`,
      );
      return;
    }
    if (item.result.extracted_text.length > MAX_RESUME_TEXT_LENGTH) {
      setItems((current) =>
        current.map((candidate) =>
          candidate.id === itemId
            ? {
                ...candidate,
                comparisonError:
                  `Resume text exceeds the local comparison limit of ${MAX_RESUME_TEXT_LENGTH.toLocaleString()} characters.`,
                comparisonStatus: "failed",
              }
            : candidate,
        ),
      );
      return;
    }

    comparisonRef.current = true;
    setIsComparing(true);
    setRequirementsError(null);
    setItems((current) =>
      current.map((candidate) =>
        candidate.id === itemId
          ? {
              ...candidate,
              comparison: undefined,
              comparisonError: undefined,
              comparisonStatus: "processing",
            }
          : candidate,
      ),
    );

    try {
      const response = await fetch("/api/resumes/compare", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          resume_text: item.result.extracted_text,
          requirements,
        }),
      });
      const responseBody: unknown = await response.json();
      const comparison =
        response.ok && isSemanticComparisonResult(responseBody)
          ? responseBody
          : undefined;
      const error =
        comparison === undefined
          ? typeof responseBody === "object" &&
            responseBody !== null &&
            "detail" in responseBody &&
            typeof responseBody.detail === "string"
            ? responseBody.detail
            : "Semantic comparison could not be completed."
          : undefined;
      setItems((current) =>
        current.map((candidate) =>
          candidate.id === itemId
            ? {
                ...candidate,
                comparison,
                job,
                comparisonError: error,
                comparisonStatus: error ? "failed" : "completed",
              }
            : candidate,
        ),
      );
      if (!error && item.evidenceResult) {
        await submitAssessment(item, job, comparison, item.evidenceResult);
      } else if (error && item.evidenceResult) {
        await submitAssessment(
          item,
          job,
          undefined,
          item.evidenceResult,
          "Semantic comparison did not complete.",
        );
      }
    } catch {
      setItems((current) =>
        current.map((candidate) =>
          candidate.id === itemId
            ? {
                ...candidate,
                comparisonError:
                  "Could not reach the local comparison service. Please try again.",
                comparisonStatus: "failed",
              }
            : candidate,
        ),
      );
      if (item.evidenceResult) {
        await submitAssessment(
          item,
          job,
          undefined,
          item.evidenceResult,
          "Semantic comparison did not complete.",
        );
      }
    } finally {
      comparisonRef.current = false;
      setIsComparing(false);
    }
  }

  async function analyzeEvidence(
    itemId: number,
    job: JobDefinition,
  ) {
    if (evidenceRef.current || comparisonRef.current) {
      return;
    }
    const item = items.find((candidate) => candidate.id === itemId);
    if (!item?.result) {
      return;
    }

    const requirements = job.requirements;
    let sourcePassages: { id: string; text: string }[];
    try {
      sourcePassages = splitEvidenceSource(item.result.extracted_text);
    } catch (error) {
      setItems((current) =>
        current.map((candidate) =>
          candidate.id === itemId
            ? {
                ...candidate,
                evidenceError:
                  error instanceof Error
                    ? error.message
                    : "Resume text exceeds the demo analysis limit.",
                evidenceStatus: "failed",
              }
            : candidate,
        ),
      );
      return;
    }

    evidenceRef.current = true;
    setIsAnalyzingEvidence(true);
    setItems((current) =>
      current.map((candidate) =>
        candidate.id === itemId
          ? {
              ...candidate,
              evidenceStatus: "processing",
              evidenceError: undefined,
              evidenceResult: undefined,
            }
          : candidate,
      ),
    );

    try {
      const response = await fetch("/api/resumes/evidence/local", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            confirmed_fictional_demo: true,
            requirements: requirements.map(({ id, text }) => ({ id, text })),
            source_passages: sourcePassages,
          }),
      });
      const responseBody: unknown = await response.json();
      const evidenceResult =
        response.ok &&
        isEvidenceAnalysisResult(responseBody) &&
        responseBody.results.length === requirements.length &&
        responseBody.results.every(
          (result, index) => result.requirement_id === requirements[index].id,
        )
          ? responseBody
          : undefined;
      const error =
        evidenceResult === undefined
          ? typeof responseBody === "object" &&
            responseBody !== null &&
            "detail" in responseBody &&
            typeof responseBody.detail === "string"
            ? responseBody.detail
            : "Evidence analysis could not be completed."
          : undefined;
      setItems((current) =>
        current.map((candidate) =>
          candidate.id === itemId
            ? {
                ...candidate,
                evidenceResult,
                evidenceError: error,
                evidenceStatus: error ? "failed" : "completed",
              }
            : candidate,
        ),
      );
      await submitAssessment(
        item,
        job,
        item.comparison,
        evidenceResult,
        error ? "Evidence analysis did not complete." : undefined,
      );
    } catch {
      setItems((current) =>
        current.map((candidate) =>
          candidate.id === itemId
            ? {
                ...candidate,
                evidenceError: "Could not reach the local evidence service.",
                evidenceStatus: "failed",
              }
            : candidate,
        ),
      );
      await submitAssessment(
        item,
        job,
        item.comparison,
        undefined,
        "Evidence analysis did not complete.",
      );
    } finally {
      evidenceRef.current = false;
      setIsAnalyzingEvidence(false);
    }
  }

  async function analyzeLocally(itemId: number) {
    if (evidenceRef.current || comparisonRef.current) return;
    setIsPreparingEvidence(true);
    setRequirementsError(null);
    try {
      const job = await validateJobDefinition();
      if (job.requirements.length === 0) {
        setRequirementsError("Enter at least one explicit requirement for evidence analysis.");
        return;
      }
      await analyzeEvidence(itemId, job);
    } catch (error) {
      setRequirementsError(
        error instanceof Error ? error.message : "Job inputs are invalid.",
      );
    } finally {
      setIsPreparingEvidence(false);
    }
  }

  const processedCount = items.filter(
    (item) => item.status === "completed" || item.status === "failed",
  ).length;

  const statusStyles: Record<FileStatus, string> = {
    queued: "bg-slate-100 text-slate-700",
    processing: "bg-indigo-100 text-indigo-800",
    completed: "bg-emerald-100 text-emerald-800",
    failed: "bg-rose-100 text-rose-800",
  };
  const rankedItems = items
    .filter((item) => item.assessment)
    .sort((first, second) => {
      const firstScore = first.assessment?.overall_score;
      const secondScore = second.assessment?.overall_score;
      if (
        (firstScore === null || firstScore === undefined) &&
        (secondScore === null || secondScore === undefined)
      ) {
        return 0;
      }
      if (firstScore === null || firstScore === undefined) return 1;
      if (secondScore === null || secondScore === undefined) return -1;
      return secondScore - firstScore;
    })
    .map((item) => {
      const score = item.assessment?.overall_score ?? null;
      return {
        ...item,
        rank:
          score === null
            ? null
            : 1 +
              items.filter(
                (candidate) =>
                  candidate.assessment?.overall_score !== null &&
                  candidate.assessment?.overall_score !== undefined &&
                  candidate.assessment.overall_score > score,
              ).length,
      };
    });
  const visibleRankedItems = rankedItems
    .filter((item) =>
      assessmentFilter === "all"
        ? true
        : assessmentFilter === "complete"
          ? item.assessment?.complete
          : !item.assessment?.complete,
    )
    .sort((first, second) =>
      sortAscending
        ? (first.assessment?.overall_score ?? Infinity) -
          (second.assessment?.overall_score ?? Infinity)
        : (second.assessment?.overall_score ?? -Infinity) -
          (first.assessment?.overall_score ?? -Infinity),
    );
  const sideBySideItems = rankedItems.filter((item) =>
    selectedCandidateIds.includes(candidateIdentifier(item)),
  );
  const currentJob = items.find((item) => item.job)?.job;
  const jobInputsDisabled =
    isComparing ||
    isAnalyzingEvidence ||
    isPreparingEvidence;

  function exportCsv() {
    const rows = [
      [
        "candidate_identifier",
        "assessment_status",
        "overall_matching_score",
        "skill_coverage",
        "semantic_relevance",
        "evidence_relevance",
        "evidence_method",
        "matched_skills",
        "supported_requirements",
        "explicitly_contradicted_requirements",
        "unverified_requirements",
      ],
      ...rankedItems.map((item) => {
        const assessment = item.assessment;
        const requirements = assessment?.requirement_assessments ?? [];
        return [
          candidateIdentifier(item),
          assessment?.complete ? "complete" : "incomplete",
          assessment?.complete ? assessment.overall_score : "",
          assessment?.complete ? assessment.skill_coverage?.score : "",
          assessment?.complete ? assessment.semantic_relevance?.score : "",
          assessment?.complete ? assessment.evidence_relevance?.score : "",
          assessment?.evidence_method ?? "",
          assessment?.matched_skills.join("; ") ?? "",
          requirements
            .filter((requirement) => requirement.status === "supported")
            .map((requirement) => requirement.requirement)
            .join("; "),
          requirements
            .filter(
              (requirement) =>
                requirement.status === "explicitly_contradicted",
            )
            .map((requirement) => requirement.requirement)
            .join("; "),
          requirements
            .filter((requirement) => requirement.status === "unclear")
            .map((requirement) => requirement.requirement)
            .join("; "),
        ];
      }),
    ];
    const csv = buildCsv(rows);
    const downloadUrl = URL.createObjectURL(
      new Blob([csv], { type: "text/csv;charset=utf-8" }),
    );
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = "talentlens-assessments.csv";
    link.click();
    URL.revokeObjectURL(downloadUrl);
  }

  function removeCandidate(candidateId: number) {
    const item = items.find((candidate) => candidate.id === candidateId);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    setItems((current) => current.filter((candidate) => candidate.id !== candidateId));
    if (item) {
      const identifier = candidateIdentifier(item);
      setSelectedCandidateIds((current) =>
        current.filter((selectedId) => selectedId !== identifier),
      );
    }
  }

  function invalidateJobAssessments() {
    setSelectedCandidateIds([]);
    setItems((current) =>
      current.map((item) => ({
        ...item,
        comparison: undefined,
        comparisonError: undefined,
        comparisonStatus: undefined,
        evidenceResult: undefined,
        evidenceError: undefined,
        evidenceStatus: undefined,
        job: undefined,
        assessment: undefined,
        assessmentError: undefined,
      })),
    );
  }

  async function saveCurrentJob() {
    setIsSavingJob(true);
    setDatabaseMessage(null);
    try {
      const job = await validateJobDefinition();
      const candidates = items.filter(
        (item) => item.status === "completed" && item.result,
      );
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          job,
          assessments: candidates.map((item) => ({
            candidate_id: candidateIdentifier(item),
            job,
            skills: {
              mentioned_skills: item.result?.mentioned_skills ?? [],
              negated_skills: item.result?.negated_skills ?? [],
            },
            semantic: item.comparison ?? null,
            evidence: item.evidenceResult ?? null,
            source_passages: item.result
              ? splitEvidenceSource(item.result.extracted_text)
              : [],
            evidence_error:
              item.evidenceStatus === "failed"
                ? "Evidence analysis did not complete."
                : undefined,
          })),
        }),
      });
      if (!response.ok) {
        throw new Error(
          response.status === 503
            ? "PostgreSQL is not configured or currently unavailable. The assessment remains only in this browser session."
            : response.status === 404
              ? "The saved-jobs route is unavailable. Restart the TalentLens AI backend and frontend."
              : "The job could not be saved.",
        );
      }
      const reload = await fetch("/api/jobs", { cache: "no-store" });
      const jobs: unknown = await reload.json();
      if (!reload.ok || !isSavedJobs(jobs)) {
        throw new Error(
          reload.status === 503
            ? "The job was saved, but PostgreSQL is unavailable; saved jobs could not be refreshed."
            : "The job was saved, but saved jobs could not be refreshed.",
        );
      }
      setSavedJobs(jobs);
      setDatabaseMessage(null);
    } catch (error) {
      setDatabaseMessage(
        error instanceof Error
          ? error.message
          : "PostgreSQL is unavailable; the assessment remains only in this browser session.",
      );
    } finally {
      setIsSavingJob(false);
    }
  }

  async function deleteSavedJob(jobId: string) {
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        setDatabaseMessage(
          response.status === 503
            ? "PostgreSQL is not configured or currently unavailable. The saved job could not be removed."
            : response.status === 404
              ? "The saved-jobs route or requested job was not found."
              : "The saved job could not be removed.",
        );
        return;
      }
      const savedJob = savedJobs.find((job) => job.job_id === jobId);
      const savedCandidateIds = new Set(
        savedJob?.assessments.map((assessment) => assessment.candidate_id) ?? [],
      );
      if (
        items.some((item) =>
          savedCandidateIds.has(candidateIdentifier(item)),
        )
      ) {
        if (fileInputRef.current) fileInputRef.current.value = "";
        setItems((current) =>
          current.filter(
            (item) => !savedCandidateIds.has(candidateIdentifier(item)),
          ),
        );
        setSelectedCandidateIds([]);
      }
      setSavedJobs((current) => current.filter((job) => job.job_id !== jobId));
      setDatabaseMessage(null);
    } catch {
      setDatabaseMessage("The saved job could not be removed.");
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-10 sm:px-6 sm:py-16">
      <section className="w-full max-w-6xl rounded-3xl border border-slate-200 bg-white p-5 shadow-xl shadow-slate-200/60 sm:p-8 lg:p-12">
        <div className="mb-10 inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-600 text-lg font-bold text-white">
          T
        </div>
        <p className="mb-3 text-sm font-semibold uppercase tracking-[0.18em] text-indigo-600">
          Resume insights, grounded in evidence
        </p>
        <h1 className="text-4xl font-bold tracking-tight text-slate-950 sm:text-5xl">
          TalentLens AI
        </h1>
        <p className="mt-5 max-w-xl text-lg leading-8 text-slate-600">
          AI-powered resume screening and explainable candidate ranking.
        </p>

        <div
          aria-live="polite"
          className="mt-10 rounded-2xl border border-slate-200 bg-slate-50 p-5"
        >
          <div className="flex items-center gap-3">
            <span
              aria-hidden="true"
              className={`h-3 w-3 rounded-full ${statusContent.dot}`}
            />
            <h2 className="font-semibold text-slate-900">
              {statusContent.label}
            </h2>
          </div>
          <p className="mt-2 pl-6 text-sm leading-6 text-slate-600">
            {statusContent.detail}
          </p>
        </div>

        <section className="mt-8 rounded-2xl border border-indigo-100 bg-indigo-50/40 p-5">
          <h2 className="text-lg font-semibold text-slate-900">Job criteria</h2>
          <p className="mt-1 text-sm text-slate-600">
            Enter role context, skill priorities, and up to {MAX_REQUIREMENTS} explicit requirements.
            Required skills count twice as much as preferred skills in skill coverage.
            Empty semantic/evidence components are neutral; a job with no criteria cannot be scored.
          </p>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <label className="block text-sm font-medium text-slate-800">
              Job title
              <input
                className="mt-1 block w-full rounded-lg border border-slate-300 bg-white p-3"
                disabled={jobInputsDisabled}
                maxLength={160}
                onChange={(event) => {
                  setJobTitle(event.target.value);
                  invalidateJobAssessments();
                }}
                placeholder="Backend Engineer"
                value={jobTitle}
              />
            </label>
            <label className="block text-sm font-medium text-slate-800">
              Job description
              <textarea
                className="mt-1 block w-full rounded-lg border border-slate-300 bg-white p-3"
                disabled={jobInputsDisabled}
                maxLength={5000}
                onChange={(event) => {
                  setJobDescription(event.target.value);
                  invalidateJobAssessments();
                }}
                placeholder="Describe the role's responsibilities and context."
                rows={3}
                value={jobDescription}
              />
            </label>
            <label className="block text-sm font-medium text-slate-800">
              Required skills
              <textarea
                className="mt-1 block w-full rounded-lg border border-slate-300 bg-white p-3"
                disabled={jobInputsDisabled}
                onChange={(event) => {
                  setRequiredSkills(event.target.value);
                  invalidateJobAssessments();
                }}
                placeholder={"Python\nREST API development"}
                rows={3}
                value={requiredSkills}
              />
            </label>
            <label className="block text-sm font-medium text-slate-800">
              Preferred skills
              <textarea
                className="mt-1 block w-full rounded-lg border border-slate-300 bg-white p-3"
                disabled={jobInputsDisabled}
                onChange={(event) => {
                  setPreferredSkills(event.target.value);
                  invalidateJobAssessments();
                }}
                placeholder={"Docker\nKubernetes"}
                rows={3}
                value={preferredSkills}
              />
            </label>
            <label
              className="block text-sm font-medium text-slate-800 md:col-span-2"
              htmlFor="job-requirements"
            >
              Explicit requirements
              <textarea
                className="mt-1 block w-full rounded-lg border border-slate-300 bg-white p-3"
                disabled={jobInputsDisabled}
                id="job-requirements"
                maxLength={MAX_REQUIREMENT_LENGTH * MAX_REQUIREMENTS}
                onChange={(event) => {
                  setJobRequirements(event.target.value);
                  setRequirementsError(null);
                  invalidateJobAssessments();
                }}
                placeholder={"REST API development\nDocker hands-on experience\nKubernetes experience"}
                rows={4}
                value={jobRequirements}
              />
              <span className="mt-1 block text-xs text-slate-500">
                One per line; at most {MAX_REQUIREMENTS}, up to {MAX_REQUIREMENT_LENGTH} characters each.
              </span>
            </label>
          </div>
          {requirementsError && (
            <p className="mt-3 text-sm text-rose-700" role="alert">
              {requirementsError}
            </p>
          )}
        </section>

        <form className="mt-8 space-y-4" onSubmit={extractText}>
          <label
            className="block text-sm font-semibold text-slate-900"
            htmlFor="resume-pdf"
          >
            Choose up to 10 PDF resumes
          </label>
          <input
            accept=".pdf,application/pdf"
            className="block w-full rounded-xl border border-slate-300 bg-white p-3 text-sm text-slate-700 file:mr-4 file:rounded-lg file:border-0 file:bg-indigo-50 file:px-4 file:py-2 file:font-semibold file:text-indigo-700 hover:file:bg-indigo-100"
            id="resume-pdf"
            multiple
            ref={fileInputRef}
            disabled={
              isProcessing ||
              isComparing ||
              isAnalyzingEvidence ||
              isPreparingEvidence
            }
            name="files"
            onChange={(event) => {
              selectFiles(event.target.files);
            }}
            type="file"
          />
          <p className="text-xs text-slate-500">
            Up to {MAX_FILES} files, with a 5 MiB per-file limit. Scanned
            documents are not OCR-processed.
          </p>
          {selectionError && (
            <p className="text-sm text-rose-700" role="alert">
              {selectionError}
            </p>
          )}
          <button
            className="rounded-xl bg-indigo-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-400"
            disabled={
              isProcessing ||
              isComparing ||
              isAnalyzingEvidence ||
              isPreparingEvidence ||
              items.length === 0 ||
              Boolean(selectionError)
            }
            type="submit"
          >
            {isProcessing ? "Extracting text…" : "Extract text"}
          </button>
        </form>

        {(databaseMessage || savedJobs.length > 0) && (
          <section className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-4">
            <h2 className="font-semibold text-slate-900">Saved jobs</h2>
            <p className="mt-1 text-xs text-slate-600">
              PostgreSQL stores job criteria and validated assessment reports only; uploaded PDFs and full extracted resume text stay in browser memory.
            </p>
            {databaseMessage && (
              <p className="mt-2 text-sm text-amber-900" role="status">
                {databaseMessage}
              </p>
            )}
            {savedJobs.length > 0 && (
              <ul className="mt-3 divide-y divide-slate-200">
                {savedJobs.map((savedJob) => (
                  <li
                    className="flex flex-wrap items-center justify-between gap-3 py-3"
                    key={savedJob.job_id}
                  >
                    <div>
                      <p className="font-medium text-slate-900">
                        {savedJob.job.title}
                      </p>
                      <p className="text-xs text-slate-600">
                        {savedJob.assessments.length} candidate assessments ·{" "}
                        {new Date(savedJob.created_at).toLocaleString()}
                      </p>
                    </div>
                    <button
                      className="rounded-lg border border-rose-300 bg-white px-3 py-2 text-sm font-semibold text-rose-800"
                      onClick={() => {
                        if (
                          window.confirm(
                            "Delete this job and all of its saved assessment data?",
                          )
                        ) {
                          void deleteSavedJob(savedJob.job_id);
                        }
                      }}
                      type="button"
                    >
                      Delete saved job
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        )}

        <div aria-live="polite" className="mt-6">
          {items.length > 0 && (
            <>
              <p className="mb-3 text-sm font-medium text-slate-700">
                Processed {processedCount} of {items.length} files.
              </p>
              {rankedItems.length > 0 && (
                <section className="mb-6 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h2 className="text-lg font-semibold text-slate-900">
                        Ranked candidate dashboard
                      </h2>
                      <p className="mt-1 text-xs text-slate-600">
                        The matching measure is not a probability of hiring
                        success. Review evidence and make all decisions yourself.
                      </p>
                      {rankingError && (
                        <p className="mt-1 text-xs text-amber-900" role="status">
                          {rankingError}
                        </p>
                      )}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-800"
                        onClick={exportCsv}
                        type="button"
                      >
                        Export CSV
                      </button>
                      <button
                        className="rounded-lg bg-indigo-700 px-3 py-2 text-sm font-semibold text-white disabled:opacity-50"
                        disabled={isSavingJob || !currentJob}
                        onClick={() => void saveCurrentJob()}
                        type="button"
                      >
                        {isSavingJob ? "Saving…" : "Save job and assessments"}
                      </button>
                    </div>
                  </div>
                  <div className="mt-4 flex flex-wrap items-center gap-3">
                    <label className="text-sm text-slate-700">
                      Filter
                      <select
                        className="ml-2 rounded-lg border border-slate-300 bg-white p-2"
                        onChange={(event) =>
                          setAssessmentFilter(
                            event.target.value as "all" | "complete" | "incomplete",
                          )
                        }
                        value={assessmentFilter}
                      >
                        <option value="all">All assessments</option>
                        <option value="complete">Complete</option>
                        <option value="incomplete">Incomplete</option>
                      </select>
                    </label>
                    <button
                      className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
                      onClick={() => setSortAscending((current) => !current)}
                      type="button"
                    >
                      Sort score: {sortAscending ? "low to high" : "high to low"}
                    </button>
                    <p className="text-xs text-slate-600">
                      Select up to two candidates for side-by-side comparison.
                    </p>
                  </div>
                  <div className="mt-4 overflow-x-auto">
                    <table className="w-full min-w-[560px] text-left text-sm">
                      <thead className="border-b border-slate-200 text-xs uppercase text-slate-600">
                        <tr>
                          <th className="p-2">Rank</th>
                          <th className="p-2">Candidate ID</th>
                          <th className="p-2">Status</th>
                          <th className="p-2">Matching score</th>
                          <th className="p-2">Compare</th>
                        </tr>
                      </thead>
                      <tbody>
                        {visibleRankedItems.map((item) => (
                          <tr className="border-b border-slate-200" key={item.id}>
                            <td className="p-2">
                              {rankByCandidateId[candidateIdentifier(item)] ??
                                item.rank ??
                                "—"}
                            </td>
                            <td className="p-2 font-medium">
                              {candidateIdentifier(item)}
                            </td>
                            <td className="p-2">
                              {item.assessment?.complete ? "Complete" : "Incomplete"}
                            </td>
                            <td className="p-2">
                              {item.assessment?.overall_score == null
                                ? "Not scored"
                                : `${item.assessment.overall_score.toFixed(2)} / 100`}
                            </td>
                            <td className="p-2">
                              <input
                                aria-label={`Select ${candidateIdentifier(item)} for comparison`}
                                checked={selectedCandidateIds.includes(candidateIdentifier(item))}
                                disabled={
                                  !selectedCandidateIds.includes(candidateIdentifier(item)) &&
                                  selectedCandidateIds.length >= 2
                                }
                                onChange={(event) =>
                                  setSelectedCandidateIds((current) =>
                                    event.target.checked
                                      ? [...current, candidateIdentifier(item)]
                                      : current.filter(
                                          (identifier) =>
                                            identifier !== candidateIdentifier(item),
                                        ),
                                  )
                                }
                                type="checkbox"
                              />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {sideBySideItems.length === 2 && (
                    <div className="mt-5 grid gap-4 md:grid-cols-2">
                      {sideBySideItems.map((item) => (
                        <article
                          className="rounded-xl border border-slate-200 bg-white p-4"
                          key={item.id}
                        >
                          <h3 className="font-semibold text-slate-900">
                            {candidateIdentifier(item)} ·{" "}
                            {item.assessment?.overall_score?.toFixed(2) ?? "Incomplete"}
                          </h3>
                          <p className="text-xs text-slate-600">
                            {item.assessment?.evidence_method ?? "Evidence method unavailable"}
                          </p>
                          <ul className="mt-3 space-y-3 text-sm">
                            {item.assessment?.requirement_assessments.map((assessment) => (
                              <li
                                className="border-t border-slate-100 pt-2"
                                key={assessment.requirement_id}
                              >
                                <p className="font-medium">
                                  {assessment.requirement} · {assessment.status.replaceAll("_", " ")}
                                </p>
                                {assessment.evidence.map((quote, index) => (
                                  <blockquote
                                    className="mt-1 border-l-2 border-indigo-200 pl-3 text-slate-600"
                                    key={`${quote.source_id}-${index}`}
                                  >
                                    {quote.quote}
                                  </blockquote>
                                ))}
                              </li>
                            ))}
                          </ul>
                        </article>
                      ))}
                    </div>
                  )}
                </section>
              )}
              <ul className="space-y-4">
                {items.map((item) => (
                  <li
                    className="rounded-2xl border border-slate-200 bg-white p-4"
                    key={item.id}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <p className="break-all font-medium text-slate-900">
                        {candidateIdentifier(item)} · {item.file.name}
                      </p>
                      <div className="flex items-center gap-2">
                        <span
                          className={`rounded-full px-3 py-1 text-xs font-semibold capitalize ${statusStyles[item.status]}`}
                        >
                          {item.status}
                        </span>
                        <button
                          className="rounded-lg border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-700"
                          onClick={() => removeCandidate(item.id)}
                          type="button"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                    {item.assessment && (
                      <section className="mt-4 rounded-xl border border-indigo-100 bg-indigo-50/40 p-4">
                        <div className="flex flex-wrap items-baseline justify-between gap-2">
                          <h3 className="font-semibold text-slate-900">
                            Matching assessment
                          </h3>
                          <p className="text-xl font-bold text-indigo-800">
                            {item.assessment.complete
                              ? `${item.assessment.overall_score?.toFixed(2)} / 100`
                              : "Incomplete"}
                          </p>
                        </div>
                        {item.assessment.evidence_method && (
                          <p className="mt-1 text-xs text-indigo-900">
                            Evidence method: {item.assessment.evidence_method}
                          </p>
                        )}
                        {!item.assessment.complete && (
                          <p className="mt-2 text-sm text-amber-900">
                            {item.assessment.incomplete_reason ??
                              "No score is available until all required analyses pass."}
                          </p>
                        )}
                        {item.assessment.complete && (
                          <>
                            <div className="mt-3 grid gap-3 sm:grid-cols-3">
                              {[
                                {
                                  label: "Skill coverage · 50%",
                                  value: item.assessment.skill_coverage,
                                },
                                {
                                  label: "Semantic relevance · 30%",
                                  value: item.assessment.semantic_relevance,
                                },
                                {
                                  label: "Evidence relevance · 20%",
                                  value: item.assessment.evidence_relevance,
                                },
                              ].map(({ label, value }) => {
                                if (!value) return null;
                                return (
                                  <div className="rounded-lg bg-white p-3" key={label}>
                                    <div className="flex justify-between gap-2 text-sm">
                                      <span>{label}</span>
                                      <strong>{value.score.toFixed(2)}</strong>
                                    </div>
                                    <div className="mt-2 h-2 rounded bg-slate-200">
                                      <div
                                        className="h-2 rounded bg-indigo-600"
                                        style={{ width: `${Math.max(0, Math.min(100, value.score))}%` }}
                                      />
                                    </div>
                                    <p className="mt-1 text-xs text-slate-500">
                                      {value.numerator.toFixed(2)} weighted points / {value.denominator.toFixed(2)} weight
                                    </p>
                                  </div>
                                );
                              })}
                            </div>
                            <p className="mt-3 break-words text-sm font-medium text-slate-800">
                              {item.assessment.arithmetic}
                            </p>
                          </>
                        )}
                        {item.assessmentError && (
                          <p className="mt-2 text-sm text-rose-700" role="alert">
                            {item.assessmentError}
                          </p>
                        )}
                        {item.assessment.requirement_assessments.length > 0 && (
                          <ul className="mt-4 space-y-3">
                            {item.assessment.requirement_assessments.map((assessment) => (
                              <li
                                className="rounded-lg border border-slate-200 bg-white p-3"
                                key={assessment.requirement_id}
                              >
                                <p className="font-medium text-slate-900">
                                  {assessment.requirement} · {assessment.status.replaceAll("_", " ")}
                                </p>
                                {assessment.status === "unclear" && assessment.evidence.length === 0 && (
                                  <p className="mt-1 text-sm text-slate-600">
                                    Unverified in the available text; this does not mean the candidate lacks the skill.
                                  </p>
                                )}
                                {assessment.evidence.map((quote, index) => (
                                  <blockquote
                                    className="mt-2 border-l-2 border-indigo-200 pl-3 text-sm text-slate-700"
                                    key={`${quote.source_id}-${index}`}
                                  >
                                    {quote.quote}
                                  </blockquote>
                                ))}
                              </li>
                            ))}
                          </ul>
                        )}
                      </section>
                    )}
                    {item.status === "failed" && item.error && (
                      <p className="mt-3 text-sm text-rose-700" role="alert">
                        {item.error}
                      </p>
                    )}
                    {item.status === "completed" && item.result && (
                      <div className="mt-3 space-y-2">
                        <button
                          className="rounded-lg border border-indigo-200 bg-indigo-50 px-4 py-2 text-sm font-semibold text-indigo-800 transition hover:bg-indigo-100 disabled:cursor-not-allowed disabled:opacity-50"
                          disabled={
                            isComparing ||
                            isAnalyzingEvidence ||
                            isPreparingEvidence ||
                            !jobRequirements.trim()
                          }
                          onClick={() => void compareRequirements(item.id)}
                          type="button"
                        >
                          {item.comparisonStatus === "processing"
                            ? "Comparing locally…"
                            : "Compare requirements"}
                        </button>
                        {item.comparisonStatus === "processing" && (
                          <p className="text-sm text-slate-600" role="status">
                            Comparing with the local model. First use may
                            download model files.
                          </p>
                        )}
                        {item.comparisonError && (
                          <p className="text-sm text-rose-700" role="alert">
                            {item.comparisonError}
                          </p>
                        )}
                      </div>
                    )}
                    {item.status === "completed" && item.result && (
                      <div className="mt-4 space-y-3">
                        <button
                          className="rounded-lg border border-indigo-300 bg-indigo-50 px-4 py-2 text-sm font-semibold text-indigo-950 hover:bg-indigo-100 disabled:opacity-50"
                          disabled={
                            isComparing || isAnalyzingEvidence ||
                            isPreparingEvidence || !jobRequirements.trim()
                          }
                          onClick={() => void analyzeLocally(item.id)}
                          type="button"
                        >
                          {item.evidenceStatus === "processing"
                            ? "Checking evidence…"
                            : "Check evidence locally"}
                        </button>
                        <p className="text-xs text-slate-600">
                          This uses a limited skill dictionary and explicit negation patterns.
                          Unrecognized or ambiguous requirements remain unclear.
                        </p>
                        {isPreparingEvidence && (
                          <p className="text-sm text-slate-600" role="status">
                            Validating job requirements…
                          </p>
                        )}
                        {item.evidenceStatus === "processing" && (
                          <p className="text-sm text-slate-600" role="status">
                            Checking exact skill mentions locally…
                          </p>
                        )}
                        {item.evidenceError && (
                          <p className="text-sm text-rose-700" role="alert">
                            {item.evidenceError}
                          </p>
                        )}
                      </div>
                    )}
                    {item.status === "completed" && item.result && (
                      <details className="mt-3">
                        <summary className="cursor-pointer text-sm font-semibold text-indigo-700">
                          View extracted text · {item.result.page_count}{" "}
                          {item.result.page_count === 1 ? "page" : "pages"}
                        </summary>
                        <div className="mt-4 space-y-4">
                          <p className="text-sm text-slate-700">
                            Filename: {item.result.filename}
                          </p>
                          <section className="space-y-3">
                            <div>
                              <h3 className="font-semibold text-slate-900">
                                Detected skills
                              </h3>
                              <p className="mt-1 text-xs leading-5 text-slate-600">
                                These are text mentions only. An excerpt or
                                keyword does not verify proficiency or applied
                                experience; please review the source. Negation
                                detection recognizes a few clear phrases and
                                can miss nuance or implied meaning.
                              </p>
                            </div>
                            {item.result.mentioned_skills.length === 0 ? (
                              <p className="text-sm text-slate-600">
                                No recognized positive skill mentions were
                                found.
                              </p>
                            ) : (
                              <ul className="space-y-3">
                                {item.result.mentioned_skills.map((skill) => (
                                  <li
                                    className="rounded-xl border border-slate-200 bg-white p-3"
                                    key={skill.canonical_name}
                                  >
                                    <p className="font-medium text-slate-900">
                                      {skill.canonical_name} · {skill.label}
                                    </p>
                                    <ul className="mt-2 space-y-2">
                                      {skill.mentions.map((mention, index) => (
                                        <li
                                          className="text-sm text-slate-700"
                                          key={`${index}-${mention.matched_wording}`}
                                        >
                                          <p>
                                            Matched wording: “
                                            {mention.matched_wording}”
                                          </p>
                                          <blockquote className="mt-1 border-l-2 border-indigo-200 pl-3 text-slate-600">
                                            {mention.source_excerpt}
                                          </blockquote>
                                        </li>
                                      ))}
                                    </ul>
                                  </li>
                                ))}
                              </ul>
                            )}
                            {item.result.negated_skills.length > 0 && (
                              <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
                                <h4 className="font-semibold text-amber-950">
                                  Explicitly negated mentions — review context
                                </h4>
                                <ul className="mt-3 space-y-3">
                                  {item.result.negated_skills.map((skill) => (
                                    <li
                                      className="text-sm text-amber-950"
                                      key={skill.canonical_name}
                                    >
                                      <p>
                                        {skill.canonical_name} · {skill.label}
                                      </p>
                                      <ul className="mt-1 space-y-1">
                                        {skill.mentions.map((mention, index) => (
                                          <li
                                            className="text-amber-900"
                                            key={`${index}-${mention.matched_wording}`}
                                          >
                                            Matched wording: “
                                            {mention.matched_wording}”
                                            <blockquote className="mt-1 border-l-2 border-amber-300 pl-3">
                                              {mention.source_excerpt}
                                            </blockquote>
                                          </li>
                                        ))}
                                      </ul>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </section>
                          {item.result.warnings.length > 0 && (
                            <div className="rounded-xl bg-amber-50 p-4 text-sm text-amber-900">
                              <h3 className="font-semibold">Warnings</h3>
                              <ul className="mt-2 list-inside list-disc space-y-1">
                                {item.result.warnings.map((warning, index) => (
                                  <li key={`${index}-${warning}`}>{warning}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-xl border border-slate-200 bg-slate-50 p-4 font-sans text-sm leading-6 text-slate-800">
                            {item.result.extracted_text}
                          </pre>
                        </div>
                      </details>
                    )}
                    {item.comparison && (
                      <section className="mt-4 space-y-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
                        <div>
                          <h3 className="font-semibold text-slate-900">
                            Relevant resume passages
                          </h3>
                          <p className="mt-1 text-xs leading-5 text-slate-600">
                            Semantic similarity compares text meaning. These
                            values are not proficiency scores or match
                            probabilities, and retrieved passages do not prove
                            a requirement is satisfied. Negated statements can
                            also be similar.
                          </p>
                        </div>
                        {item.comparison.comparisons.map((comparison, index) => (
                          <div
                            className="space-y-2"
                            key={`${index}-${comparison.requirement}`}
                          >
                            <h4 className="font-medium text-slate-900">
                              {comparison.requirement}
                            </h4>
                            {comparison.matches.length === 0 ? (
                              <p className="text-sm text-slate-600">
                                No resume passages were available.
                              </p>
                            ) : (
                              <ul className="space-y-2">
                                {comparison.matches.map((match, matchIndex) => (
                                  <li
                                    className="rounded-lg border border-slate-200 bg-white p-3"
                                    key={`${matchIndex}-${match.excerpt}`}
                                  >
                                    <p className="text-xs font-medium text-indigo-800">
                                      Semantic similarity: {match.similarity}
                                    </p>
                                    <blockquote className="mt-2 whitespace-pre-wrap text-sm text-slate-700">
                                      {match.excerpt}
                                    </blockquote>
                                  </li>
                                ))}
                              </ul>
                            )}
                          </div>
                        ))}
                      </section>
                    )}
                    {item.evidenceResult && (
                      <section className="mt-4 space-y-4 rounded-xl border border-amber-200 bg-amber-50 p-4">
                        <div>
                          <h3 className="font-semibold text-amber-950">
                            {item.evidenceResult.label}
                          </h3>
                          <p className="mt-1 text-xs leading-5 text-amber-900">
                            {item.evidenceResult.label.startsWith("Local")
                              ? "These limited dictionary results are not verified proficiency or decisions. Only simple recognized skill requirements can be assessed; review every excerpt."
                              : "These are model-generated assessments, not verified facts or decisions. Review every status and source excerpt."}
                            {" "}Missing evidence is unverified, not proof that a requirement is absent.
                          </p>
                        </div>
                        {item.evidenceResult.results.map((result) => (
                          <article
                            className="space-y-2 rounded-lg border border-amber-200 bg-white p-3"
                            key={result.requirement_id}
                          >
                            <div className="flex flex-wrap items-center gap-2">
                              <h4 className="font-medium text-slate-900">
                                {jobRequirements
                                  .split(/\r?\n/)
                                  .map((requirement) => requirement.trim())
                                  .filter(Boolean)[
                                  Number(
                                    result.requirement_id.replace(
                                      "requirement-",
                                      "",
                                    ),
                                  ) - 1
                                ] ?? result.requirement_id}
                              </h4>
                              <span className="rounded-full bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-800">
                                {result.status.replace(/_/g, " ")}
                              </span>
                            </div>
                            <p className="text-sm leading-6 text-slate-700">
                              {result.explanation}
                            </p>
                            {result.evidence.length > 0 ? (
                              <ul className="space-y-2">
                                {result.evidence.map((evidence, index) => (
                                  <li
                                    className="rounded-md border-l-2 border-indigo-300 bg-slate-50 p-3 text-sm"
                                    key={`${evidence.source_id}-${index}`}
                                  >
                                    <p className="mb-1 text-xs font-medium text-slate-600">
                                      {evidence.relationship} · source{" "}
                                      {evidence.source_id}
                                    </p>
                                    <blockquote className="whitespace-pre-wrap text-slate-800">
                                      “{evidence.quote}”
                                    </blockquote>
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p className="text-sm text-slate-600">
                                No supporting or contradictory excerpt was
                                identified; treat this requirement as
                                unverified.
                              </p>
                            )}
                          </article>
                        ))}
                      </section>
                    )}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      </section>
    </main>
  );
}
