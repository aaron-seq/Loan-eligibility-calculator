"""Unit Tests for Decorators

Testing cross-cutting concerns:
- Rate limiting
- Authentication
- Correlation IDs
"""

import pytest
import time
from flask import Flask, session

from src.utils.decorators import (
    rate_limit,
    require_authentication,
    require_admin_authentication,
    add_correlation_id,
)


class TestRateLimiting:
    """Test rate limiting decorator"""

    def test_rate_limit_allows_within_limit(self):
        """Test that requests within limit are allowed
        
        Positive Testing: Normal usage.
        """
        app = Flask(__name__)
        app.secret_key = "test-secret"

        @app.route("/test")
        @rate_limit(max_requests=5, window_seconds=60)
        def test_route():
            return {"success": True}, 200

        with app.test_client() as client:
            # Make requests within limit
            for i in range(5):
                response = client.get("/test")
                assert response.status_code == 200, f"Request {i+1} should succeed"

    def test_rate_limit_blocks_over_limit(self):
        """Test that requests over limit are blocked
        
        Negative Testing: Abuse prevention.
        """
        app = Flask(__name__)
        app.secret_key = "test-secret"

        @app.route("/test")
        @rate_limit(max_requests=3, window_seconds=60)
        def test_route():
            return {"success": True}, 200

        with app.test_client() as client:
            # Make requests up to limit
            for i in range(3):
                response = client.get("/test")
                assert response.status_code == 200

            # Next request should be rate limited
            response = client.get("/test")
            assert response.status_code == 429, "Should return 429 Too Many Requests"
            assert b"rate limit" in response.data.lower()


class TestAuthentication:
    """Test authentication decorators"""

    def test_require_authentication_blocks_unauthenticated(self):
        """Test that unauthenticated requests are blocked
        
        Negative Testing: Security enforcement.
        """
        app = Flask(__name__)
        app.secret_key = "test-secret"

        @app.route("/protected")
        @require_authentication
        def protected_route():
            return {"success": True}, 200

        with app.test_client() as client:
            response = client.get("/protected")
            assert response.status_code == 401, "Should return 401 Unauthorized"
            assert b"authentication required" in response.data.lower()

    def test_require_authentication_allows_authenticated(self):
        """Test that authenticated requests are allowed
        
        Positive Testing: Authenticated access.
        """
        app = Flask(__name__)
        app.secret_key = "test-secret"

        @app.route("/protected")
        @require_authentication
        def protected_route():
            return {"success": True}, 200

        with app.test_client() as client:
            # Create session
            with client.session_transaction() as sess:
                sess["user_id"] = "test_user"

            response = client.get("/protected")
            assert response.status_code == 200, "Authenticated user should access"
