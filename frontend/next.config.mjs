// Frontend → Backend 호출은 `/__backend/*` same-origin proxy 를 통한다.
// 이렇게 하면 browser 입장에서 모든 API 호출이 vercel.app same-origin 으로 보여
// CORS 협상이 필요 없다. Backend URL 변경 시 NEXT_PUBLIC_BACKEND_URL 환경변수만 바꾼다.
function normalizeBackendUrl(value) {
  const raw = value?.trim().replace(/^["']|["']$/g, "");
  if (!raw) {
    return "";
  }

  const withProtocol = /^https?:\/\//i.test(raw) ? raw : `http://${raw}`;
  try {
    const parsed = new URL(withProtocol);
    return parsed.toString().replace(/\/$/, "");
  } catch {
    return "";
  }
}

// production 에서 NEXT_PUBLIC_BACKEND_URL 미설정 시 build 단계에서 빠르게 실패시킨다.
// dev 에서는 127.0.0.1:8000 default 로 떨어진다.
const proxyTarget =
  normalizeBackendUrl(process.env.NEXT_PUBLIC_BACKEND_URL) ||
  (process.env.NODE_ENV === "production" ? "" : "http://127.0.0.1:8000");

if (process.env.NODE_ENV === "production" && !proxyTarget) {
  console.warn(
    "[next.config] NEXT_PUBLIC_BACKEND_URL 미설정 — production build 에서 backend proxy 가 비활성됩니다."
  );
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    if (!proxyTarget) {
      return [];
    }

    return [
      {
        source: "/__backend/:path*",
        destination: `${proxyTarget}/:path*`,
      },
    ];
  },
};

export default nextConfig;
