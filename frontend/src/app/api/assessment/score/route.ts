import { proxyBackendJson } from "../../_lib/backend-json";

export function POST(request: Request) {
  return proxyBackendJson(request, "/api/assessment/score");
}
