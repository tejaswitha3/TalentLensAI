const backendUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";
const requestTimeoutMs = 3_000;

export async function GET(request: Request) {
  // Probe only after an incoming request; builds must not depend on the backend being up.
  void request.url;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), requestTimeoutMs);

  try {
    const response = await fetch(new URL("/health", backendUrl), {
      cache: "no-store",
      signal: controller.signal,
    });

    if (response.status !== 200) {
      console.error(
        `Backend health check returned HTTP ${response.status}.`,
      );
      return Response.json({ status: "unavailable" }, { status: 503 });
    }

    const body: unknown = await response.json();
    if (
      typeof body !== "object" ||
      body === null ||
      !("status" in body) ||
      body.status !== "ok"
    ) {
      console.error("Backend health check returned an unexpected response.");
      return Response.json({ status: "unavailable" }, { status: 503 });
    }

    return Response.json({ status: "connected" });
  } catch (error) {
    console.error("Backend health check failed.", error);
    return Response.json({ status: "unavailable" }, { status: 503 });
  } finally {
    clearTimeout(timeout);
  }
}
