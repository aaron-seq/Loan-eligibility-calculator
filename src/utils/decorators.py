"""Decorators for Controllers

Provides reusable decorators for authentication, rate limiting, and validation.
"""

import time
from functools import wraps
from flask import session, jsonify, request
from collections import defaultdict
from typing import Callable

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Rate limiting storage (in-memory, use Redis in production)
rate_limit_storage = defaultdict(list)


def require_login(f: Callable) -> Callable:
    """Decorator to require user authentication
    
    Usage:
        @app.route('/protected')
        @require_login
        def protected_route():
            return jsonify({'user': session['user_id']})
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            logger.warning(f"Unauthorized access attempt to {request.endpoint}")
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Authentication required",
                    }
                ),
                401,
            )
        return f(*args, **kwargs)

    return decorated_function


def require_admin(f: Callable) -> Callable:
    """Decorator to require admin authentication
    
    Usage:
        @app.route('/admin')
        @require_admin
        def admin_route():
            return jsonify({'admin': session['admin_id']})
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "admin_id" not in session or not session.get("is_admin"):
            logger.warning(f"Unauthorized admin access attempt to {request.endpoint}")
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Admin access required",
                    }
                ),
                403,
            )
        return f(*args, **kwargs)

    return decorated_function


def rate_limit(max_requests: int = 60, window_seconds: int = 60) -> Callable:
    """Decorator for rate limiting endpoints
    
    Args:
        max_requests: Maximum requests allowed
        window_seconds: Time window in seconds
        
    Usage:
        @app.route('/api/endpoint')
        @rate_limit(max_requests=10, window_seconds=60)
        def limited_endpoint():
            return jsonify({'data': 'value'})
    
    Note: This is in-memory implementation. Use Redis in production for
    distributed rate limiting across multiple app instances.
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Use IP address as identifier (use user_id in production)
            identifier = request.remote_addr
            key = f"{f.__name__}:{identifier}"

            current_time = time.time()

            # Clean old requests outside window
            rate_limit_storage[key] = [
                req_time
                for req_time in rate_limit_storage[key]
                if current_time - req_time < window_seconds
            ]

            # Check rate limit
            if len(rate_limit_storage[key]) >= max_requests:
                logger.warning(
                    f"Rate limit exceeded for {identifier} on {f.__name__}"
                )
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": "Rate limit exceeded. Please try again later.",
                        }
                    ),
                    429,
                )

            # Add current request
            rate_limit_storage[key].append(current_time)

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def require_json(f: Callable) -> Callable:
    """Decorator to ensure request has JSON content type
    
    Usage:
        @app.route('/api/endpoint', methods=['POST'])
        @require_json
        def json_endpoint():
            data = request.json
            return jsonify(data)
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            logger.warning(
                f"Non-JSON request to {request.endpoint} from {request.remote_addr}"
            )
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Content-Type must be application/json",
                    }
                ),
                400,
            )
        return f(*args, **kwargs)

    return decorated_function


def log_execution_time(f: Callable) -> Callable:
    """Decorator to log endpoint execution time
    
    Usage:
        @app.route('/api/slow-endpoint')
        @log_execution_time
        def slow_endpoint():
            # expensive operation
            return jsonify({'result': 'done'})
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        execution_time = time.time() - start_time

        logger.info(
            f"{f.__name__} executed in {execution_time:.3f}s for {request.remote_addr}"
        )

        return result

    return decorated_function
