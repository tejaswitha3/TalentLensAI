const backendUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";
const requestTimeoutMs = 30_000;

export async function POST(request: Request) {
  const contentType = request.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().startsWith("multipart/form-data;")) {
    return Response.json(
      { detail: "A multipart PDF upload is required." },
      { status: 400 },
    );
  }

  let incomingForm: FormData;
  try {
    incomingForm = await request.formData();
  } catch (error) {
    console.error("Could not read the incoming resume upload.", error);
    return Response.json(
      { detail: "The upload form could not be read." },
      { status: 400 },
    );
  }

  const file = incomingForm.get("file");
  if (!(file instanceof File)) {
    return Response.json(
      { detail: "Select one PDF file to extract." },
      { status: 400 },
    );
  }

  const backendForm = new FormData();
  backendForm.append("file", file, file.name);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), requestTimeoutMs);

  try {
    const response = await fetch(
      new URL("/api/resumes/extract", backendUrl),
      {
        method: "POST",
        body: backendForm,
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
    console.error("Resume extraction request to the backend failed.", error);
    if (controller.signal.aborted) {
      return Response.json(
        { detail: "Resume extraction timed out. Please try again." },
        { status: 504 },
      );
    }
    return Response.json(
      { detail: "The backend could not process the upload. Check its status." },
      { status: 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}
