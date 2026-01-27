"""Admin Controller - Handles administrative HTTP requests

HTTP concerns only. Business logic in AdminService.
"""

from flask import Blueprint, request, jsonify, session, render_template
from pydantic import ValidationError as PydanticValidationError

from src.models.schemas import AdminLoginRequest
from src.services.admin_service import AdminService
from src.exceptions import InvalidCredentialsError
from src.logger import get_logger
from src.utils.decorators import (
    rate_limit,
    add_correlation_id,
    require_admin_authentication,
)
from src.utils.response import success_response, error_response

logger = get_logger(__name__)

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

# Service injection
admin_service: AdminService = None


def init_admin_controller(service: AdminService):
    """Initialize controller with service dependency"""
    global admin_service
    admin_service = service


@admin_bp.route("/login", methods=["POST"])
@add_correlation_id
@rate_limit(max_requests=5, window_seconds=300)  # Stricter limit for admin
def login():
    """Handle admin login request
    
    HTTP concerns:
    - Parse credentials
    - Create admin session
    - Return response
    """
    try:
        # Parse request data
        data = request.get_json() if request.is_json else request.form.to_dict()

        # Validate using Pydantic schema
        admin_request = AdminLoginRequest(**data)

        # Delegate to service
        admin_info = admin_service.authenticate_admin(admin_request)

        # Create admin session
        session["admin_id"] = admin_info["admin_id"]
        session["is_admin"] = True
        session.permanent = True

        return success_response(
            data={"admin_id": admin_info["admin_id"]},
            message="Admin login successful",
        )

    except PydanticValidationError as e:
        logger.warning(f"Admin login validation error: {e}")
        return error_response(
            message="Invalid credentials format",
            status_code=400,
        )

    except InvalidCredentialsError as e:
        return error_response(
            message=str(e),
            status_code=401,
        )

    except Exception as e:
        logger.error(f"Unexpected admin login error: {e}", exc_info=True)
        return error_response(
            message="Admin login failed",
            status_code=500,
        )


@admin_bp.route("/predictions", methods=["GET"])
@add_correlation_id
@require_admin_authentication
def get_predictions():
    """Get all predictions with pagination
    
    HTTP concerns:
    - Parse query parameters
    - Validate pagination params
    - Format response
    """
    try:
        # Parse pagination parameters
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))

        # Validate parameters
        if limit < 1 or limit > 1000:
            return error_response(
                message="Limit must be between 1 and 1000",
                status_code=400,
            )

        if offset < 0:
            return error_response(
                message="Offset cannot be negative",
                status_code=400,
            )

        # Delegate to service
        predictions = admin_service.get_all_predictions(limit, offset)

        return success_response(
            data={
                "predictions": predictions,
                "count": len(predictions),
                "limit": limit,
                "offset": offset,
            },
            message="Predictions retrieved successfully",
        )

    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}", exc_info=True)
        return error_response(
            message="Failed to retrieve predictions",
            status_code=500,
        )


@admin_bp.route("/statistics", methods=["GET"])
@add_correlation_id
@require_admin_authentication
def get_statistics():
    """Get prediction statistics
    
    Returns aggregate statistics for admin dashboard.
    """
    try:
        stats = admin_service.get_prediction_statistics()

        return success_response(
            data=stats,
            message="Statistics retrieved successfully",
        )

    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}", exc_info=True)
        return error_response(
            message="Failed to retrieve statistics",
            status_code=500,
        )


@admin_bp.route("/users", methods=["GET"])
@add_correlation_id
@require_admin_authentication
def get_users():
    """Get user list with pagination
    
    HTTP concerns:
    - Parse query parameters
    - Format response
    """
    try:
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
        active_only = request.args.get("active_only", "true").lower() == "true"

        # Validate parameters
        if limit < 1 or limit > 1000:
            return error_response(
                message="Limit must be between 1 and 1000",
                status_code=400,
            )

        if offset < 0:
            return error_response(
                message="Offset cannot be negative",
                status_code=400,
            )

        # Delegate to service
        users = admin_service.get_user_list(limit, offset, active_only)

        return success_response(
            data={
                "users": users,
                "count": len(users),
                "limit": limit,
                "offset": offset,
            },
            message="Users retrieved successfully",
        )

    except Exception as e:
        logger.error(f"Error retrieving users: {e}", exc_info=True)
        return error_response(
            message="Failed to retrieve users",
            status_code=500,
        )


@admin_bp.route("/logout", methods=["POST", "GET"])
@add_correlation_id
def logout():
    """Handle admin logout
    
    HTTP concern: Clear admin session
    """
    try:
        admin_id = session.get("admin_id")
        session.clear()

        if admin_id:
            logger.info(f"Admin logged out: {admin_id}")

        return success_response(
            message="Admin logged out successfully",
        )

    except Exception as e:
        logger.error(f"Admin logout error: {e}", exc_info=True)
        return error_response(
            message="Logout failed",
            status_code=500,
        )
