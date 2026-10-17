const backendUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";
const requestTimeoutMs = 180_000;

export async function POST(request: Request) {
  if (
    !request.headers
      .get("content-type")
      ?.toLowerCase()
      .includes("application/json")
  ) {
    return Response.json(
      { detail: "A JSON semantic comparison request is required." },
      { status: 400 },
    );
  }

  let body: string;
  try {
    body = await request.text();
    JSON.parse(body);
  } catch {
    console.error("Could not read the semantic comparison request.");
    return Response.json(
      { detail: "The semantic comparison request could not be read." },
      { status: 400 },
    );
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), requestTimeoutMs);

  try {
    const response = await fetch(
      new URL("/api/resumes/compare", backendUrl),
      {
        method: "POST",
        body,
        headers: { "content-type": "application/json" },
        cache: "no-store",
        signal: controller.signal,
      },
    );
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
      "Local semantic comparison request to the backend failed.",
      error instanceof Error ? error.name : "unknown error",
    );
    if (controller.signal.aborted) {
      return Response.json(
        { detail: "Semantic comparison timed out. Please try again." },
        { status: 504 },
      );
    }
    return Response.json(
      { detail: "The backend could not complete semantic comparison." },
      { status: 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}
