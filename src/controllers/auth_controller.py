"""Authentication Controller - Handles user authentication HTTP requests

This controller handles ONLY HTTP-related concerns:
- Request parsing
- Response formatting 
- Session management

All business logic is delegated to UserService.
"""

from flask import Blueprint, request, jsonify, session, redirect, render_template
from pydantic import ValidationError as PydanticValidationError

from src.models.schemas import UserRegistrationRequest, UserLoginRequest
from src.services.user_service import UserService
from src.exceptions import (
    UserAlreadyExistsError,
    InvalidCredentialsError,
    ValidationError,
)
from src.logger import get_logger
from src.utils.decorators import rate_limit, add_correlation_id
from src.utils.response import success_response, error_response

logger = get_logger(__name__)

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# Service injection will happen during app initialization
user_service: UserService = None


def init_auth_controller(service: UserService):
    """Initialize controller with service dependency
    
    Dependency injection allows for testing and loose coupling.
    """
    global user_service
    user_service = service


@auth_bp.route("/register", methods=["POST"])
@add_correlation_id
@rate_limit(max_requests=5, window_seconds=300)  # 5 requests per 5 minutes
def register():
    """Handle user registration request
    
    HTTP concerns only:
    - Parse JSON/form data
    - Validate request schema
    - Return appropriate HTTP status codes
    """
    try:
        # Parse request data
        data = request.get_json() if request.is_json else request.form.to_dict()

        # Validate using Pydantic schema
        registration_request = UserRegistrationRequest(**data)

        # Delegate to service (business logic)
        result = user_service.register_user(registration_request)

        # Format success response
        return success_response(
            data=result,
            message="User registered successfully",
            status_code=201,
        )

    except PydanticValidationError as e:
        # Pydantic validation errors
        logger.warning(f"Registration validation error: {e}")
        return error_response(
            message="Invalid input data",
            errors=e.errors(),
            status_code=400,
        )

    except (UserAlreadyExistsError, ValidationError) as e:
        # Business validation errors
        return error_response(
            message=str(e),
            status_code=409 if isinstance(e, UserAlreadyExistsError) else 400,
        )

    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected registration error: {e}", exc_info=True)
        return error_response(
            message="Registration failed. Please try again.",
            status_code=500,
        )


@auth_bp.route("/login", methods=["POST"])
@add_correlation_id
@rate_limit(max_requests=10, window_seconds=300)  # 10 requests per 5 minutes
def login():
    """Handle user login request
    
    HTTP concerns:
    - Parse credentials
    - Manage session creation
    - Return appropriate status codes
    """
    try:
        # Parse request data
        data = request.get_json() if request.is_json else request.form.to_dict()

        # Validate using Pydantic schema
        login_request = UserLoginRequest(**data)

        # Delegate to service (business logic)
        user_info = user_service.authenticate_user(login_request)

        # Create session (HTTP concern)
        session["user_id"] = user_info["user_id"]
        session["full_name"] = user_info["full_name"]
        session.permanent = True  # Use permanent session

        # Format success response
        return success_response(
            data={"user_id": user_info["user_id"]},
            message="Login successful",
        )

    except PydanticValidationError as e:
        logger.warning(f"Login validation error: {e}")
        return error_response(
            message="Invalid credentials format",
            errors=e.errors(),
            status_code=400,
        )

    except InvalidCredentialsError as e:
        return error_response(
            message=str(e),
            status_code=401,
        )

    except Exception as e:
        logger.error(f"Unexpected login error: {e}", exc_info=True)
        return error_response(
            message="Login failed. Please try again.",
            status_code=500,
        )


@auth_bp.route("/logout", methods=["POST", "GET"])
@add_correlation_id
def logout():
    """Handle user logout request
    
    HTTP concern: Clear session
    """
    try:
        user_id = session.get("user_id")
        session.clear()

        if user_id:
            logger.info(f"User logged out: {user_id}")

        return success_response(
            message="Logged out successfully",
        )

    except Exception as e:
        logger.error(f"Logout error: {e}", exc_info=True)
        return error_response(
            message="Logout failed",
            status_code=500,
        )
