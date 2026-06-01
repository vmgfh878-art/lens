import type { ApiError } from "@/lib/apiErrors";
import { describeApiError, shortenEndpoint } from "@/lib/apiErrors";

export interface StatusInlineProps {
  kind: "loading" | "empty" | "error";
  /** 영역명 (예: "AI 밴드 (1D)"). loading/empty/error 공통. */
  label?: string;
  /** API 엔드포인트 (있으면 축약해서 표시). */
  endpoint?: string;
  /** 사람이 읽을 한 줄. error 의 경우 describeApiError 결과를 기본으로 사용. */
  message?: string;
  /** 백엔드 request_id. 마지막 8자만 표시. */
  requestId?: string;
  /** 회복 액션 힌트 (예: "백엔드 로그 확인"). */
  hint?: string;
  /** 재시도 콜백. 있으면 재시도 버튼 표시. */
  onRetry?: () => void;
  /** 원본 ApiError. 있으면 message/endpoint/requestId 우선 사용. */
  error?: ApiError;
  /** 컴팩트(인라인 한 줄) vs 영역 박스. 기본 컴팩트. */
  variant?: "compact" | "block";
}

/**
 * CP213 — loading / empty / error 단일 표시 컴포넌트.
 * 에러일 때 반드시 4가지 표시: 영역명 + 엔드포인트 축약 + 사람이 읽을 메시지 + request_id 끝 8자.
 */
export default function StatusInline({
  kind,
  label,
  endpoint,
  message,
  requestId,
  hint,
  onRetry,
  error,
  variant = "compact",
}: StatusInlineProps) {
  const finalEndpoint = error?.endpoint ?? endpoint;
  const finalRequestId = error?.requestId ?? requestId;
  const finalMessage =
    message ??
    (error ? describeApiError(error) : kind === "loading" ? "불러오는 중…" : "표시할 데이터가 없습니다.");

  const shortEndpoint = finalEndpoint ? shortenEndpoint(finalEndpoint) : null;
  const reqTail =
    finalRequestId && finalRequestId.length > 0 ? `req:${finalRequestId.slice(-8)}` : null;

  const rootClass = [
    "status-inline",
    `status-inline--${kind}`,
    variant === "block" ? "status-inline--block" : "status-inline--compact",
  ].join(" ");

  return (
    <div className={rootClass} role={kind === "error" ? "alert" : "status"} aria-live={kind === "error" ? "polite" : "off"}>
      <div className="status-inline__main">
        {label ? <strong className="status-inline__label">{label}</strong> : null}
        <span className="status-inline__message">{finalMessage}</span>
      </div>
      {(shortEndpoint || reqTail || hint) ? (
        <div className="status-inline__meta">
          {shortEndpoint ? <code className="status-inline__endpoint">{shortEndpoint}</code> : null}
          {reqTail ? <code className="status-inline__req">{reqTail}</code> : null}
          {hint ? <span className="status-inline__hint">{hint}</span> : null}
        </div>
      ) : null}
      {onRetry ? (
        <button type="button" className="status-inline__retry" onClick={onRetry}>
          재시도
        </button>
      ) : null}
    </div>
  );
}
