import type { DisplayTimeframe } from "@/api/client";

export interface ApiErrorShape {
  code: string;
  message: string;
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
