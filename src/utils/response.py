"""Response Formatting Utilities

Standardized response formats for consistency across the API.
Follows best practices for REST API responses.
"""

from typing import Any, Dict, Optional, List
from flask import jsonify, make_response, g


def success_response(
    data: Optional[Any] = None,
    message: str = "Success",
    status_code: int = 200,
) -> tuple:
    """Create standardized success response
    
    All successful responses follow the same structure:
    {
        "success": true,
        "message": "...",
        "data": {...},
        "correlation_id": "..."
    }
    
    Args:
        data: Response data (optional)
        message: Success message
        status_code: HTTP status code
        
    Returns:
        Tuple of (response, status_code)
    """
    response_body = {
        "success": True,
        "message": message,
    }

    if data is not None:
        response_body["data"] = data

    # Add correlation ID if available
    if hasattr(g, "correlation_id"):
        response_body["correlation_id"] = g.correlation_id

    response = make_response(jsonify(response_body), status_code)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    return response


def error_response(
    message: str = "An error occurred",
    errors: Optional[List[Dict[str, Any]]] = None,
    status_code: int = 400,
) -> tuple:
    """Create standardized error response
    
    All error responses follow the same structure:
    {
        "success": false,
        "message": "...",
        "errors": [...],  # Optional detailed errors
        "correlation_id": "..."
    }
    
    Args:
        message: Error message
        errors: Detailed error list (optional)
        status_code: HTTP status code
        
    Returns:
        Tuple of (response, status_code)
    """
    response_body = {
        "success": False,
        "message": message,
    }

    if errors is not None:
        response_body["errors"] = errors

    # Add correlation ID if available
    if hasattr(g, "correlation_id"):
        response_body["correlation_id"] = g.correlation_id

    response = make_response(jsonify(response_body), status_code)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    return response
