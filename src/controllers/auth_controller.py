"""Authentication Controller - HTTP Layer

Handles authentication HTTP requests and delegates to AuthService.
No business logic - only request handling and response formatting.
"""

from flask import Blueprint, request, session, jsonify
from pydantic import ValidationError as PydanticValidationError

from src.models.schemas import UserRegistrationRequest, UserLoginRequest, AdminLoginRequest
from src.services.auth_service import AuthService
from src.utils.logger import get_logger
from src.utils.exceptions import (
    ValidationError,
    AuthenticationError,
    DuplicateUserError,
)
from src.utils.decorators import rate_limit, require_json

logger = get_logger(__name__)
auth_blueprint = Blueprint("auth", __name__, url_prefix="/api/auth")

# Service will be injected during app initialization
auth_service: AuthService = None


def init_auth_controller(service: AuthService):
    """Initialize controller with service dependency"""
    global auth_service
    auth_service = service


@auth_blueprint.route("/register", methods=["POST"])
@rate_limit(max_requests=5, window_seconds=60)
@require_json
def register():
    """Register new user endpoint
    
    Request Body:
        - user_id: Unique username
        - email: Email address
        - full_name: Full name
        - password: Password
        - confirm_password: Password confirmation
        - mobile_number: Optional mobile number
        
    Returns:
        201: User created successfully
        400: Validation error
        409: User already exists
        500: Server error
    """
    try:
        # Validate request using Pydantic
        data = UserRegistrationRequest(**request.json)

        # Delegate to service
        user = auth_service.register_user(
            user_id=data.user_id,
            email=data.email,
            full_name=data.full_name,
            password=data.password,
            mobile_number=data.mobile_number,
        )

        logger.info(f"User registered successfully: {data.user_id}")

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Registration successful",
                    "data": user,
                }
            ),
            201,
        )

    except PydanticValidationError as e:
        logger.warning(f"Registration validation error: {e}")
        return jsonify({"success": False, "message": "Validation error", "errors": e.errors()}), 400

    except ValidationError as e:
        logger.warning(f"Registration validation error: {e}")
        return jsonify({"success": False, "message": str(e)}), 400

    except DuplicateUserError as e:
        logger.warning(f"Registration duplicate error: {e}")
        return jsonify({"success": False, "message": str(e)}), 409

    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Registration failed"}), 500


@auth_blueprint.route("/login", methods=["POST"])
@rate_limit(max_requests=10, window_seconds=60)
@require_json
def login():
    """User login endpoint
    
    Request Body:
        - user_id: Username
        - password: Password
        
    Returns:
        200: Login successful
        400: Validation error
        401: Invalid credentials
        500: Server error
    """
    try:
        # Validate request
        data = UserLoginRequest(**request.json)

        # Authenticate user
        user = auth_service.authenticate_user(
            user_id=data.user_id,
            password=data.password,
        )

        # Set session
        session["user_id"] = user["user_id"]
        session["full_name"] = user["full_name"]
        session["is_admin"] = False

        logger.info(f"User logged in successfully: {data.user_id}")

        return jsonify(
            {
                "success": True,
                "message": "Login successful",
                "data": {
                    "user_id": user["user_id"],
                    "full_name": user["full_name"],
                },
            }
        ), 200

    except PydanticValidationError as e:
        return jsonify({"success": False, "message": "Validation error", "errors": e.errors()}), 400

    except (ValidationError, AuthenticationError) as e:
        logger.warning(f"Login failed: {e}")
        return jsonify({"success": False, "message": str(e)}), 401

    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Login failed"}), 500


@auth_blueprint.route("/admin/login", methods=["POST"])
@rate_limit(max_requests=5, window_seconds=60)
@require_json
def admin_login():
    """Admin login endpoint
    
    Request Body:
        - admin_id: Admin username
        - password: Password
        
    Returns:
        200: Login successful
        400: Validation error
        401: Invalid credentials
        500: Server error
    """
    try:
        # Validate request
        data = AdminLoginRequest(**request.json)

        # Authenticate admin
        admin = auth_service.authenticate_admin(
            admin_id=data.admin_id,
            password=data.password,
        )

        # Set session
        session["admin_id"] = admin["admin_id"]
        session["is_admin"] = True

        logger.info(f"Admin logged in successfully: {data.admin_id}")

        return jsonify(
            {
                "success": True,
                "message": "Admin login successful",
                "data": {"admin_id": admin["admin_id"]},
            }
        ), 200

    except PydanticValidationError as e:
        return jsonify({"success": False, "message": "Validation error", "errors": e.errors()}), 400

    except (ValidationError, AuthenticationError) as e:
        logger.warning(f"Admin login failed: {e}")
        return jsonify({"success": False, "message": str(e)}), 401

    except Exception as e:
        logger.error(f"Admin login error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Login failed"}), 500


@auth_blueprint.route("/logout", methods=["POST"])
def logout():
    """Logout endpoint (user or admin)
    
    Returns:
        200: Logout successful
    """
    user_id = session.get("user_id")
    admin_id = session.get("admin_id")

    session.clear()

    logger.info(f"Logout successful: user={user_id}, admin={admin_id}")

    return jsonify({"success": True, "message": "Logout successful"}), 200


@auth_blueprint.route("/session", methods=["GET"])
def check_session():
    """Check current session status
    
    Returns:
        200: Session info
    """
    return jsonify(
        {
            "authenticated": "user_id" in session or "admin_id" in session,
            "user_id": session.get("user_id"),
            "admin_id": session.get("admin_id"),
            "is_admin": session.get("is_admin", False),
        }
    ), 200
