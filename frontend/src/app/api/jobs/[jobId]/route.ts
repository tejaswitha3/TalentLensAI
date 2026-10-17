const backendUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function DELETE(
  _request: Request,
  context: { params: Promise<{ jobId: string }> },
) {
  const { jobId } = await context.params;
  if (!/^[0-9a-f-]{36}$/i.test(jobId)) {
    return Response.json(
      { detail: "Invalid saved job identifier." },
      { status: 400 },
    );
  }
  try {
    const response = await fetch(
      new URL(`/api/jobs/${encodeURIComponent(jobId)}`, backendUrl),
      { method: "DELETE", cache: "no-store" },
    );
    return new Response(response.body, {
      status: response.status,
      headers: {
        "content-type":
          response.headers.get("content-type") ?? "application/json",
        "cache-control": "no-store",
      },
    });
  } catch {
    return Response.json(
      { detail: "The backend could not delete the saved job." },
      { status: 502 },
    );
  }
}
