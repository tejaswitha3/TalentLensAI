import {
  proxyBackendGet,
  proxyBackendJson,
} from "../_lib/backend-json";

export function GET() {
  return proxyBackendGet("/api/jobs");
}

export function POST(request: Request) {
  return proxyBackendJson(request, "/api/jobs");
}
