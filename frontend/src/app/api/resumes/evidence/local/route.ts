import { proxyBackendJson } from "../../../_lib/backend-json";

export async function POST(request: Request) {
  return proxyBackendJson(request, "/api/resumes/evidence/local", 100_000);
}
