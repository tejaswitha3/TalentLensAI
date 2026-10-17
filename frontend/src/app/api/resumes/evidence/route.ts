const backendUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";
const requestTimeoutMs = 100_000;
const maxRequestLength = 100_000;

export async function POST(request: Request) {
  if (
    !request.headers
      .get("content-type")
      ?.toLowerCase()
      .includes("application/json")
  ) {
    return Response.json(
      { detail: "A JSON evidence-analysis request is required." },
      { status: 400 },
    );
  }

  const contentLength = Number(request.headers.get("content-length") ?? 0);
  if (contentLength > maxRequestLength) {
    return Response.json(
      { detail: "The evidence-analysis request is too large." },
      { status: 413 },
    );
  }

  let body: string;
  try {
    body = await request.text();
    if (body.length > maxRequestLength) {
      return Response.json(
        { detail: "The evidence-analysis request is too large." },
        { status: 413 },
      );
    }
    JSON.parse(body);
  } catch {
    return Response.json(
      { detail: "The evidence-analysis request could not be read." },
      { status: 400 },
    );
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), requestTimeoutMs);
  try {
    const response = await fetch(new URL("/api/resumes/evidence", backendUrl), {
      method: "POST",
      body,
      headers: { "content-type": "application/json" },
      cache: "no-store",
      signal: controller.signal,
    });
    return new Response(response.body, {
      status: response.status,
      headers: {
        "content-type":
          response.headers.get("content-type") ?? "application/json",
        "cache-control": "no-store",
      },
    });
  } catch (error) {
    console.error(
      "Evidence-analysis request to the backend failed.",
      error instanceof Error ? error.name : "unknown error",
    );
    if (controller.signal.aborted) {
      return Response.json(
        { detail: "Evidence analysis timed out. Please try again." },
        { status: 504 },
      );
    }
    return Response.json(
      { detail: "The backend could not complete evidence analysis." },
      { status: 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}
