import axios from "axios";

// Backend 호출은 same-origin proxy `/__backend/*` 를 통한다 (next.config.mjs rewrites).
// - browser (CSR): `/__backend` (relative) → Vercel 서버가 NEXT_PUBLIC_BACKEND_URL 로 forward
// - server (SSR): NEXT_PUBLIC_BACKEND_URL 직접 (proxy 없음, build env)
// 이렇게 하면 browser 에서 CORS 가 필요 없고, backend URL 변경 시 env 하나만 바꾼다.

const LOCAL_BACKEND_URL = "http://127.0.0.1:8000";
const BROWSER_PROXY_BASE = "/__backend";

type BackendUrlSource = "env" | "local-default" | "missing-env" | "invalid-env";

interface BackendUrlResolution {
  url: string;
  source: BackendUrlSource;
  warning: string | null;
}

function isLocalBrowserHost() {
  if (typeof window === "undefined") {
    return false;
  }
  const host = window.location.hostname;
  return host === "localhost" || host === "127.0.0.1";
}

function resolveBackendUrl(value: string | undefined): BackendUrlResolution {
  const raw = value?.trim().replace(/^["']|["']$/g, "");
  if (!raw) {
    if (isLocalBrowserHost()) {
      return {
        url: LOCAL_BACKEND_URL,
        source: "local-default",
        warning:
          "NEXT_PUBLIC_BACKEND_URL이 비어 있어 로컬 백엔드로 연결합니다. 배포 환경에서는 환경변수를 반드시 설정해야 합니다.",
      };
    }
    return {
      url: "",
      source: "missing-env",
      warning:
        "NEXT_PUBLIC_BACKEND_URL이 설정되지 않았습니다. 배포 환경에서는 고정 백엔드 fallback을 사용하지 않습니다.",
    };
  }

  const withProtocol = /^https?:\/\//i.test(raw) ? raw : `http://${raw}`;
  try {
    const parsed = new URL(withProtocol);
    return {
      url: parsed.toString().replace(/\/$/, ""),
      source: "env",
      warning: null,
    };
  } catch {
    const fallback = isLocalBrowserHost() ? LOCAL_BACKEND_URL : "";
    return {
      url: fallback,
      source: "invalid-env",
      warning: fallback
        ? `NEXT_PUBLIC_BACKEND_URL 값이 올바르지 않아 로컬 백엔드 ${fallback}로 연결합니다.`
        : "NEXT_PUBLIC_BACKEND_URL 값이 올바르지 않습니다. 배포 환경에서는 고정 백엔드 fallback을 사용하지 않습니다.",
    };
  }
}

const backendUrlResolution = resolveBackendUrl(process.env.NEXT_PUBLIC_BACKEND_URL);

function resolveBrowserBaseUrl() {
  // 로컬 브라우저에서는 프록시보다 직접 백엔드가 더 안정적이다.
  if (isLocalBrowserHost()) {
    return backendUrlResolution.url || LOCAL_BACKEND_URL;
  }
  return backendUrlResolution.url ? BROWSER_PROXY_BASE : "";
}

// browser 에서는 로컬이면 직접 백엔드, 배포면 same-origin proxy 사용.
const apiBaseUrl = typeof window === "undefined" ? backendUrlResolution.url : resolveBrowserBaseUrl();

export const api = axios.create({
  baseURL: apiBaseUrl,
});

export function getBackendBaseUrl() {
  return backendUrlResolution.url;
}

export function getBackendConfigWarning() {
  return backendUrlResolution.warning;
}
