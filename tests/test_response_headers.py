"""Response hardening must cover successful, rejected and static responses."""

import pytest


@pytest.mark.parametrize(
    "path,status",
    [
        ("/", 200),
        ("/api/health", 200),
        ("/missing", 404),
        ("/api/missing", 404),
        ("/static/css/app.css", 200),
        ("/dashboard", 302),
    ],
)
def test_security_headers_cover_every_response(client, path, status):
    response = client.get(path)
    assert response.status_code == status
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "same-origin"
    if not path.startswith("/static/"):
        assert response.headers["Cache-Control"] == "no-store"
    policy = response.headers["Content-Security-Policy"]
    for directive in (
        "default-src 'none'",
        "script-src 'none'",
        "style-src 'self'",
        "img-src 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
    ):
        assert directive in policy.split("; ")


def test_security_headers_cover_server_errors(modelless_app):
    @modelless_app.get("/broken")
    def broken():
        raise RuntimeError("deliberate test error")

    modelless_app.config["PROPAGATE_EXCEPTIONS"] = False
    response = modelless_app.test_client().get("/broken")
    assert response.status_code == 500
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
