"""Decorators for Cross-Cutting Concerns

Decorators handle concerns that apply across multiple routes:
- Authentication
- Rate limiting
- Request tracking
- Caching
"""

import time
import uuid
from functools import wraps
from typing import Callable, Dict
from flask import request, session, g, jsonify

from src.logger import get_logger
from src.utils.response import error_response

logger = get_logger(__name__)

# Simple in-memory rate limiting (use Redis in production)
_rate_limit_cache: Dict[str, list] = {}


def add_correlation_id(f: Callable) -> Callable:
    """Add correlation ID to request for distributed tracing
    
    Correlation IDs allow tracking requests across services and logs.
    This is critical for debugging in production.
    
    Usage:
        @app.route('/endpoint')
        @add_correlation_id
        def endpoint():
            # g.correlation_id is now available
            pass
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if correlation ID exists in headers (from upstream service)
        correlation_id = request.headers.get(
            "X-Correlation-ID", str(uuid.uuid4())
        )

        # Store in Flask's g object for access in logging
        g.correlation_id = correlation_id

        # Log request with correlation ID
        logger.info(
            f"Request received: {request.method} {request.path}",
            extra={"correlation_id": correlation_id},
        )

        # Execute function
        response = f(*args, **kwargs)

        # Add correlation ID to response headers
        if isinstance(response, tuple):
            response_obj, status_code = response[0], response[1]
            if hasattr(response_obj, "headers"):
                response_obj.headers["X-Correlation-ID"] = correlation_id
        elif hasattr(response, "headers"):
            response.headers["X-Correlation-ID"] = correlation_id

        return response

    return decorated_function


def rate_limit(max_requests: int = 10, window_seconds: int = 60) -> Callable:
    """Rate limiting decorator to prevent abuse
    
    Implements sliding window rate limiting.
    In production, use Redis with INCR and EXPIRE.
    
    Args:
        max_requests: Maximum number of requests allowed
        window_seconds: Time window in seconds
        
    Usage:
        @app.route('/api/endpoint')
        @rate_limit(max_requests=100, window_seconds=3600)
        def endpoint():
            pass
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get client identifier (IP address or user ID)
            client_id = (
                session.get("user_id")
                or session.get("admin_id")
                or request.remote_addr
            )

            # Create cache key
            cache_key = f"{f.__name__}:{client_id}"

            # Get current time
            current_time = time.time()

            # Initialize or get request timestamps
            if cache_key not in _rate_limit_cache:
                _rate_limit_cache[cache_key] = []

            # Remove timestamps outside the window
            _rate_limit_cache[cache_key] = [
                timestamp
                for timestamp in _rate_limit_cache[cache_key]
                if current_time - timestamp < window_seconds
            ]

            # Check if limit exceeded
            if len(_rate_limit_cache[cache_key]) >= max_requests:
                logger.warning(
                    f"Rate limit exceeded for {cache_key}",
                    extra={"correlation_id": getattr(g, "correlation_id", None)},
                )
                return error_response(
                    message="Rate limit exceeded. Please try again later.",
                    status_code=429,
                )

            # Add current request timestamp
            _rate_limit_cache[cache_key].append(current_time)

            # Execute function
            return f(*args, **kwargs)

        return decorated_function

    return decorator


def require_authentication(f: Callable) -> Callable:
    """Require user authentication for endpoint
    
    Checks if user is logged in before allowing access.
    
    Usage:
        @app.route('/protected')
        @require_authentication
        def protected_route():
            # session['user_id'] is guaranteed to exist
            pass
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            logger.warning(
                f"Unauthorized access attempt to {request.path}",
                extra={"correlation_id": getattr(g, "correlation_id", None)},
            )
            return error_response(
                message="Authentication required. Please log in.",
                status_code=401,
            )

        return f(*args, **kwargs)

    return decorated_function


def require_admin_authentication(f: Callable) -> Callable:
    """Require admin authentication for endpoint
    
    Checks if admin is logged in before allowing access.
    
    Usage:
        @app.route('/admin/endpoint')
        @require_admin_authentication
        def admin_route():
            # session['admin_id'] is guaranteed to exist
            pass
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "admin_id" not in session or not session.get("is_admin"):
            logger.warning(
                f"Unauthorized admin access attempt to {request.path}",
                extra={"correlation_id": getattr(g, "correlation_id", None)},
            )
            return error_response(
                message="Admin authentication required.",
                status_code=403,
            )

        return f(*args, **kwargs)

    return decorated_function
