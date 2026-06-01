import type { DisplayTimeframe } from "@/api/client";

export interface ApiErrorShape {
  code: string;
  message: string;
}

/**
 * CP213 — 단일 에러 분류 표준.
 * 모든 fetch 지점이 이 형태로 변환해서 StatusInline 으로 표시한다.
 */
export type ApiErrorKind =
  | "network"
  | "http_4xx"
  | "http_5xx"
  | "shape_mismatch"
  | "empty_payload"
  | "unknown";

export interface ApiError {
  kind: ApiErrorKind;
  status?: number;
  code?: string;
  message: string;
  requestId?: string;
  endpoint: string;
}

function extractRequestId(error: unknown): string | undefined {
  if (
    typeof error === "object" &&
    error !== null &&
    "response" in error &&
    typeof (error as { response?: unknown }).response === "object"
  ) {
    const meta = (
      error as { response?: { data?: { meta?: { request_id?: unknown } } } }
    ).response?.data?.meta;
    const requestId = meta?.request_id;
    if (typeof requestId === "string" && requestId.length > 0) {
      return requestId;
    }
  }
  return undefined;
}

function isNetworkError(error: unknown) {
  if (!(error instanceof Error)) {
    return false;
  }
  return (
    error.message === "Network Error" ||
    error.message.includes("ECONNREFUSED") ||
    error.message.includes("ECONNRESET") ||
    error.message.includes("ETIMEDOUT") ||
    error.message.includes("Failed to fetch") ||
    error.message.includes("Failed to construct 'URL'") ||
    error.message.includes("Invalid URL")
  );
}

/**
 * fetch 실패(또는 throw된 객체)를 표준 ApiError로 분류한다.
 * empty_payload, shape_mismatch 는 호출자가 응답 본문 검사 후 직접 build 한다 (buildEmptyPayloadError 등 helper 사용).
 */
export function classifyApiError(input: unknown, endpoint: string): ApiError {
  const requestId = extractRequestId(input);
  const apiError = extractApiError(input);
  const status = extractHttpStatus(input);

  if (status != null && status >= 500 && status < 600) {
    return {
      kind: "http_5xx",
      status,
      code: apiError?.code,
      message: apiError?.message ?? `백엔드 ${status} 오류`,
      requestId,
      endpoint,
    };
  }
  if (status != null && status >= 400 && status < 500) {
    return {
      kind: "http_4xx",
      status,
      code: apiError?.code,
      message: apiError?.message ?? `요청이 거부됨 (${status})`,
      requestId,
      endpoint,
    };
  }
  if (isNetworkError(input)) {
    return {
      kind: "network",
      message:
        input instanceof Error && input.message
          ? `네트워크 오류: ${input.message}`
          : "백엔드에 연결할 수 없습니다.",
      endpoint,
    };
  }
  if (input instanceof Error) {
    return {
      kind: "unknown",
      message: input.message || "알 수 없는 오류가 발생했습니다.",
      requestId,
      endpoint,
    };
  }
  return {
    kind: "unknown",
    message: "알 수 없는 오류가 발생했습니다.",
    requestId,
    endpoint,
  };
}

export function buildEmptyPayloadError(endpoint: string, reason?: string): ApiError {
  return {
    kind: "empty_payload",
    message: reason ?? "응답은 정상이지만 데이터가 비어 있습니다.",
    endpoint,
  };
}

export function buildShapeMismatchError(endpoint: string, reason: string): ApiError {
  return {
    kind: "shape_mismatch",
    message: reason,
    endpoint,
  };
}

/**
 * 사람이 읽을 한 줄 요약. StatusInline 본문 또는 alert title용.
 */
export function describeApiError(error: ApiError): string {
  const reqTail =
    error.requestId && error.requestId.length > 0
      ? ` · req:${error.requestId.slice(-8)}`
      : "";
  const kindLabel: Record<ApiErrorKind, string> = {
    network: "네트워크 끊김",
    http_4xx: `요청 거부${error.status ? ` (${error.status})` : ""}`,
    http_5xx: `백엔드 오류${error.status ? ` (${error.status})` : ""}`,
    shape_mismatch: "응답 형식 오류",
    empty_payload: "빈 응답",
    unknown: "알 수 없는 오류",
  };
  const code = error.code ? ` [${error.code}]` : "";
  return `${kindLabel[error.kind]}${code}: ${error.message}${reqTail}`;
}

/**
 * 엔드포인트를 표시용으로 축약 (쿼리스트링 제거 + 마지막 path만).
 * 예: "/api/v1/predictions/band/1d/AAPL?from=2020" → "/predictions/band/1d/AAPL"
 */
export function shortenEndpoint(endpoint: string): string {
  const noQuery = endpoint.split("?")[0];
  return noQuery.replace(/^\/api\/v\d+/, "");
}

export function extractApiError(error: unknown): ApiErrorShape | null {
  if (
    typeof error === "object" &&
    error !== null &&
    "response" in error &&
    typeof (error as { response?: unknown }).response === "object"
  ) {
    const payload = (
      error as { response?: { data?: { error?: { code?: string; message?: string } } } }
    ).response?.data?.error;
    if (payload?.code && payload?.message) {
      return { code: payload.code, message: payload.message };
    }
  }
  return null;
}

export function extractHttpStatus(error: unknown) {
  if (
    typeof error === "object" &&
    error !== null &&
    "response" in error &&
    typeof (error as { response?: unknown }).response === "object"
  ) {
    const status = (error as { response?: { status?: unknown } }).response?.status;
    return typeof status === "number" ? status : null;
  }
  return null;
}

export function isResourceNotFound(error: unknown) {
  const apiError = extractApiError(error);
  return apiError?.code === "RESOURCE_NOT_FOUND" || extractHttpStatus(error) === 404;
}

export function isBackendConnectionError(error: unknown) {
  const status = extractHttpStatus(error);
  if (status != null) {
    return status >= 500;
  }
  if (error instanceof Error) {
    return (
      error.message === "Network Error" ||
      error.message.includes("ECONNREFUSED") ||
      error.message.includes("Failed to construct 'URL'") ||
      error.message.includes("Invalid URL")
    );
  }
  return false;
}

export function extractErrorMessage(error: unknown) {
  const apiError = extractApiError(error);
  if (apiError) {
    return apiError.message;
  }
  if (error instanceof Error) {
    if (error.message === "Network Error" || error.message.includes("ECONNREFUSED")) {
      return "백엔드에 연결할 수 없습니다. NEXT_PUBLIC_BACKEND_URL 설정과 백엔드 상태를 확인해주세요.";
    }
    if (
      error.message.includes("Failed to construct 'URL'") ||
      error.message.includes("Invalid URL")
    ) {
      return "백엔드 주소 설정을 확인할 수 없습니다. NEXT_PUBLIC_BACKEND_URL 값을 확인해주세요.";
    }
    return error.message;
  }
  return "데이터를 불러오지 못했습니다.";
}

export function buildPriceMissingMessage(ticker: string, timeframe: DisplayTimeframe) {
  return `${ticker.toUpperCase()} ${timeframe} 가격 데이터가 아직 연결되지 않았습니다. 티커 또는 데이터 소스를 확인해주세요.`;
}
