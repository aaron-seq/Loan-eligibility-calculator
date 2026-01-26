"""User Service - Handles user authentication and management

This service contains ONLY business logic related to users.
No HTTP concerns, no database details - pure business rules.
"""

import re
from typing import Optional, Dict, Any
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

from src.exceptions import (
    UserAlreadyExistsError,
    UserNotFoundError,
    InvalidCredentialsError,
    ValidationError,
)
from src.logger import get_logger
from src.models.database import User
from src.models.schemas import UserRegistrationRequest, UserLoginRequest

logger = get_logger(__name__)


class UserService:
    """Service for user-related business operations"""

    def __init__(self, db_connection_factory):
        """Initialize with database connection factory
        
        Args:
            db_connection_factory: Context manager for database connections
        """
        self.db_connection_factory = db_connection_factory

    def register_user(self, request: UserRegistrationRequest) -> Dict[str, Any]:
        """Register a new user with validation
        
        Business Rules:
        - Username must be unique
        - Email must be valid and unique
        - Password must meet security requirements
        - Mobile number must be valid format (if provided)
        
        Args:
            request: Validated user registration request
            
        Returns:
            Dict containing user_id and success message
            
        Raises:
            UserAlreadyExistsError: If username or email already exists
            ValidationError: If input validation fails
        """
        logger.info(f"Registering new user: {request.user_id}")

        # Additional business validation
        self._validate_user_registration(request)

        # Hash password using secure algorithm
        password_hash = generate_password_hash(
            request.password, method="pbkdf2:sha256", salt_length=16
        )

        try:
            with self.db_connection_factory() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO users (user_id, email, mobile_number, full_name, password_hash)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        request.user_id,
                        request.email,
                        request.mobile_number or "",
                        request.full_name,
                        password_hash,
                    ),
                )
                conn.commit()

            logger.info(f"User registered successfully: {request.user_id}")
            return {
                "user_id": request.user_id,
                "message": "User registered successfully",
            }

        except sqlite3.IntegrityError as e:
            logger.warning(
                f"Registration failed - duplicate entry: {request.user_id}"
            )
            raise UserAlreadyExistsError(
                "Username or email already exists. Please choose different credentials."
            )
        except Exception as e:
            logger.error(f"Unexpected error during registration: {e}", exc_info=True)
            raise

    def authenticate_user(self, request: UserLoginRequest) -> Dict[str, Any]:
        """Authenticate user credentials
        
        Business Rules:
        - User must exist and be active
        - Password must match stored hash
        - Failed login attempts should be logged for security monitoring
        
        Args:
            request: Validated login request
            
        Returns:
            Dict containing user information for session
            
        Raises:
            InvalidCredentialsError: If credentials are invalid
            UserNotFoundError: If user does not exist
        """
        logger.info(f"Authentication attempt for user: {request.user_id}")

        try:
            with self.db_connection_factory() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT user_id, password_hash, full_name, is_active
                    FROM users
                    WHERE user_id = ?
                """,
                    (request.user_id,),
                )
                user = cursor.fetchone()

            if not user:
                logger.warning(f"Authentication failed - user not found: {request.user_id}")
                raise InvalidCredentialsError("Invalid username or password")

            if not user["is_active"]:
                logger.warning(f"Authentication failed - user inactive: {request.user_id}")
                raise InvalidCredentialsError("Account is inactive. Please contact support.")

            # Verify password using constant-time comparison
            if not check_password_hash(user["password_hash"], request.password):
                logger.warning(f"Authentication failed - invalid password: {request.user_id}")
                raise InvalidCredentialsError("Invalid username or password")

            logger.info(f"User authenticated successfully: {request.user_id}")
            return {
                "user_id": user["user_id"],
                "full_name": user["full_name"],
            }

        except (InvalidCredentialsError, UserNotFoundError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error during authentication: {e}", exc_info=True)
            raise

    def _validate_user_registration(self, request: UserRegistrationRequest):
        """Additional business-level validation
        
        Pydantic validates structure, this validates business rules.
        """
        # Validate email format (additional check beyond Pydantic)
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, request.email):
            raise ValidationError("Invalid email format")

        # Validate mobile number format if provided
        if request.mobile_number:
            # Remove common separators for validation
            clean_mobile = re.sub(r"[\s\-\(\)]", "", request.mobile_number)
            if not (clean_mobile.isdigit() and 10 <= len(clean_mobile) <= 15):
                raise ValidationError(
                    "Invalid mobile number. Must be 10-15 digits."
                )

        # Validate username doesn't contain invalid characters
        username_pattern = r"^[a-zA-Z0-9_-]{3,30}$"
        if not re.match(username_pattern, request.user_id):
            raise ValidationError(
                "Username must be 3-30 characters and contain only letters, numbers, underscores, and hyphens"
            )

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user information by ID
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            User information dict or None if not found
        """
        try:
            with self.db_connection_factory() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT user_id, email, mobile_number, full_name, created_at, is_active
                    FROM users
                    WHERE user_id = ?
                """,
                    (user_id,),
                )
                user = cursor.fetchone()

            if user:
                return dict(user)
            return None

        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {e}", exc_info=True)
            raise

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account
        
        Business Rule: Users are never deleted, only deactivated for audit trail.
        
        Args:
            user_id: User to deactivate
            
        Returns:
            True if successful
            
        Raises:
            UserNotFoundError: If user does not exist
        """
        try:
            with self.db_connection_factory() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE users SET is_active = 0 WHERE user_id = ?", (user_id,)
                )
                
                if cursor.rowcount == 0:
                    raise UserNotFoundError(f"User {user_id} not found")
                    
                conn.commit()

            logger.info(f"User deactivated: {user_id}")
            return True

        except UserNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error deactivating user {user_id}: {e}", exc_info=True)
            raise
