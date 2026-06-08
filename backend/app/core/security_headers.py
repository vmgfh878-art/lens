"""CP240 — HTTP 보안 헤더 미들웨어.

5 헤더 박음 (모든 응답에):
- Strict-Transport-Security: HTTPS 강제 (2년 + subdomain 포함)
- X-Content-Type-Options: MIME sniffing 차단
- X-Frame-Options: clickjacking 차단 (iframe 금지)
- Referrer-Policy: 외부 이동 시 URL 누출 제어
- Content-Security-Policy: XSS / 외부 스크립트 차단

CSP 는 API 응답에도 박지만 API 는 JSON 만 반환이라 직접 효과 작음. 주된
효과는 frontend (Vercel) 의 next.config.mjs headers() 에서 나옴. 본 미들웨어는
defense in depth + API 응답 일관성 차원.

순서: main.py 에서 CORSMiddleware / GZipMiddleware 다음에 add — outermost
가 되어 모든 응답 후처리 단계에 보안 헤더 박음.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# CSP — API 응답용. frontend next.config 와 정합 유지.
# - default-src 'self': 외부 리소스 차단 (baseline)
# - script-src 'self' 'unsafe-inline': lightweight-charts inline script
#   호환 (임시 — v2 에서 nonce 도입해 'unsafe-inline' 제거 검토)
# - style-src 'self' 'unsafe-inline': Next.js styled-jsx 호환
# - img-src 'self' data:: data URL 이미지 허용 (base64 등)
# - connect-src 'self': API 자체. backend 가 외부 호출 안 함 (Lens 의 backend
#   는 Supabase / EODHD 등 server-side 호출이지만 그건 outgoing 이라 CSP
#   영향 0. CSP 의 connect-src 는 browser 가 발신하는 fetch/XHR/WebSocket
#   대상이라 frontend next.config 의 connect-src 가 본질)
# - frame-ancestors 'none': iframe 삽입 차단 (X-Frame-Options 보강)
# - base-uri 'self': base tag 우회 방지
# - form-action 'self': form submit 대상 제한
_API_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

_HEADERS: dict[str, str] = {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": _API_CSP,
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """모든 응답에 보안 헤더 5종 부착.

    setdefault 로 박아서 별도 미들웨어 / 라우터가 더 strict 한 정책 박은
    경우 덮어쓰지 않음. 예: 향후 특정 endpoint 가 Content-Security-Policy
    더 strict 하게 박으면 그게 우선.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        for key, value in _HEADERS.items():
            response.headers.setdefault(key, value)
        return response
