"""CP240 — 보안 헤더 5종 응답 부착 검증.

backend FastAPI 가 모든 응답에 다음 5 헤더 박는지 검증.
SecurityHeadersMiddleware 가 끊기면 즉시 RED.
파일명에 cp prefix 안 박은 이유: 영구 안전망 (CP223 test_characterization_api.py
와 동일 정책 — .gitignore 의 test_cp*.py 우회).
"""

from __future__ import annotations

from tests.conftest import FIXED_HEADERS

EXPECTED_HEADERS = {
    "strict-transport-security",
    "x-content-type-options",
    "x-frame-options",
    "referrer-policy",
    "content-security-policy",
    "permissions-policy",
}


def test_security_headers_attached_on_health_live(client):
    """health endpoint 응답에 5 헤더 모두 박혀있어야."""
    resp = client.get("/api/v1/health/live", headers=FIXED_HEADERS)
    assert resp.status_code == 200
    actual = {k.lower() for k in resp.headers.keys()}
    missing = EXPECTED_HEADERS - actual
    assert not missing, f"missing security headers on /health/live: {missing}"


def test_security_headers_attached_on_data_endpoint(client):
    """일반 데이터 endpoint 도 동일 헤더 박혀있어야."""
    resp = client.get(
        "/api/v1/stocks",
        params={"limit": 5},
        headers=FIXED_HEADERS,
    )
    assert resp.status_code == 200
    actual = {k.lower() for k in resp.headers.keys()}
    missing = EXPECTED_HEADERS - actual
    assert not missing, f"missing security headers on /stocks: {missing}"


def test_strict_transport_security_value(client):
    """HSTS = 2년 + includeSubDomains."""
    resp = client.get("/api/v1/health/live", headers=FIXED_HEADERS)
    hsts = resp.headers.get("strict-transport-security", "")
    assert "max-age=63072000" in hsts
    assert "includeSubDomains" in hsts


def test_x_content_type_options_nosniff(client):
    resp = client.get("/api/v1/health/live", headers=FIXED_HEADERS)
    assert resp.headers.get("x-content-type-options") == "nosniff"


def test_x_frame_options_deny(client):
    resp = client.get("/api/v1/health/live", headers=FIXED_HEADERS)
    assert resp.headers.get("x-frame-options") == "DENY"


def test_referrer_policy(client):
    resp = client.get("/api/v1/health/live", headers=FIXED_HEADERS)
    assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"


def test_csp_required_directives(client):
    """CSP 의 필수 directive: default-src 'self', script-src 'self', frame-ancestors 'none'."""
    resp = client.get("/api/v1/health/live", headers=FIXED_HEADERS)
    csp = resp.headers.get("content-security-policy", "")
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "base-uri 'self'" in csp


def test_permissions_policy_denies_unused_features(client):
    """Permissions-Policy 의 필수 deny: camera / microphone / geolocation /
    payment / browsing-topics (Topics API opt-out)."""
    resp = client.get("/api/v1/health/live", headers=FIXED_HEADERS)
    pp = resp.headers.get("permissions-policy", "")
    assert "camera=()" in pp
    assert "microphone=()" in pp
    assert "geolocation=()" in pp
    assert "payment=()" in pp
    assert "browsing-topics=()" in pp
