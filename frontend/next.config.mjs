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

// CP240 — HTTP 보안 헤더 5종.
// CSP 정책:
// - script-src 'unsafe-inline' / 'unsafe-eval': lightweight-charts + Next.js
//   hydration + Microsoft Clarity init 호환 (임시). v2 에서 nonce 적용 검토.
// - *.clarity.ms: Microsoft Clarity script + telemetry (외부 도메인,
//   frontend/src/app/ClarityInit.tsx 의 @microsoft/clarity init).
// - connect-src 의 http://127.0.0.1:8000: 로컬 dev 환경에서 frontend(3000)
//   가 backend(8000) 직접 호출 (baseClient.ts 의 localhost 분기 — proxy 안 씀).
// - production 의 same-origin proxy (/__backend/*) 는 'self' 가 처리.
const securityHeaders = [
  {
    key: "Strict-Transport-Security",
    value: "max-age=63072000; includeSubDomains",
  },
  {
    key: "X-Content-Type-Options",
    value: "nosniff",
  },
  {
    key: "X-Frame-Options",
    value: "DENY",
  },
  {
    key: "Referrer-Policy",
    value: "strict-origin-when-cross-origin",
  },
  {
    // 사용 안 하는 브라우저 기능 명시적 deny — Lens 는 차트/분석 SPA 라
    // camera/mic/geo/payment/USB/sensor 0개 사용. browsing-topics 는 Google
    // Topics API opt-out (프라이버시, Clarity 정책과 정합).
    key: "Permissions-Policy",
    value: [
      "camera=()",
      "microphone=()",
      "geolocation=()",
      "payment=()",
      "usb=()",
      "magnetometer=()",
      "accelerometer=()",
      "gyroscope=()",
      "browsing-topics=()",
    ].join(", "),
  },
  {
    key: "Content-Security-Policy",
    value: [
      "default-src 'self'",
      // www.clarity.ms + *.clarity.ms 둘 다 명시 (브라우저별 wildcard
      // 처리 차이 대비, 안전 우선)
      "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://www.clarity.ms https://*.clarity.ms",
      "style-src 'self' 'unsafe-inline'",
      // Clarity 트래킹 픽셀 포함. blob: 제거 (lightweight-charts 가 안 씀,
      // Playwright e2e 에서 회귀 감지)
      "img-src 'self' data: https://*.clarity.ms",
      "font-src 'self' data:",
      // production same-origin proxy(/__backend/*) 는 'self' 가 처리.
      // 직접 backend URL 호출 대비 lens-backend-7stj.onrender.com 명시.
      // 127.0.0.1:8000 은 로컬 dev 의 baseClient.ts localhost 직접 호출.
      "connect-src 'self' https://www.clarity.ms https://*.clarity.ms https://lens-backend-7stj.onrender.com http://127.0.0.1:8000",
      "frame-ancestors 'none'",
      "base-uri 'self'",
      "form-action 'self'",
    ].join("; "),
  },
];

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
  async headers() {
    return [
      {
        source: "/:path*",
        headers: securityHeaders,
      },
    ];
  },
};

export default nextConfig;
