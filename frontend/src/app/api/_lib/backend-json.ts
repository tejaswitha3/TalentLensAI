const backendUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function proxyBackendGet(path: string) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10_000);
  try {
    const response = await fetch(new URL(path, backendUrl), {
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
  } catch {
    return Response.json(
      {
        detail: controller.signal.aborted
          ? "The backend request timed out."
          : "The backend could not complete the request.",
      },
      { status: controller.signal.aborted ? 504 : 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}

export async function proxyBackendJson(
  request: Request,
  path: string,
  maxRequestLength = 120_000,
) {
  if (
    !request.headers
      .get("content-type")
      ?.toLowerCase()
      .includes("application/json")
  ) {
    return Response.json({ detail: "A JSON request is required." }, { status: 400 });
  }

  const contentLength = Number(request.headers.get("content-length") ?? 0);
  if (contentLength > maxRequestLength) {
    return Response.json({ detail: "The request is too large." }, { status: 413 });
  }

  let body: string;
  try {
    body = await request.text();
    if (body.length > maxRequestLength) {
      return Response.json({ detail: "The request is too large." }, { status: 413 });
    }
    JSON.parse(body);
  } catch {
    return Response.json({ detail: "The JSON request could not be read." }, { status: 400 });
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 35_000);
  try {
    const response = await fetch(new URL(path, backendUrl), {
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
  } catch {
    return Response.json(
      {
        detail: controller.signal.aborted
          ? "The backend request timed out."
          : "The backend could not complete the request.",
      },
      { status: controller.signal.aborted ? 504 : 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}
