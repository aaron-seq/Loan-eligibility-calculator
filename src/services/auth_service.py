"""Authentication Service - Business Logic Layer

Handles all authentication-related business logic without HTTP dependencies.
Pure business logic enables easy testing and reusability.
"""

import re
from typing import Optional, Dict, Any
from werkzeug.security import generate_password_hash, check_password_hash

from src.utils.logger import get_logger
from src.utils.exceptions import (
    ValidationError,
    AuthenticationError,
    DuplicateUserError,
)
from src.repositories.user_repository import UserRepository
from src.repositories.admin_repository import AdminRepository

logger = get_logger(__name__)


class AuthService:
    """Authentication service handling user and admin authentication logic"""

    # Password requirements
    MIN_PASSWORD_LENGTH = 8
    PASSWORD_PATTERN = re.compile(
        r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
    )
    EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    def __init__(self, user_repo: UserRepository, admin_repo: AdminRepository):
        """Initialize auth service with repositories
        
        Args:
            user_repo: User data repository
            admin_repo: Admin data repository
        """
        self.user_repo = user_repo
        self.admin_repo = admin_repo

    def register_user(
        self,
        user_id: str,
        email: str,
        full_name: str,
        password: str,
        mobile_number: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new user with validation
        
        Args:
            user_id: Unique user identifier
            email: User email address
            full_name: User's full name
            password: Plain text password
            mobile_number: Optional mobile number
            
        Returns:
            Dict with user information
            
        Raises:
            ValidationError: If input validation fails
            DuplicateUserError: If user already exists
        """
        # Validate inputs
        self._validate_user_registration(user_id, email, full_name, password)

        # Check if user already exists
        if self.user_repo.get_by_user_id(user_id):
            logger.warning(f"Registration attempt with existing user_id: {user_id}")
            raise DuplicateUserError(f"User ID '{user_id}' already exists")

        if self.user_repo.get_by_email(email):
            logger.warning(f"Registration attempt with existing email: {email}")
            raise DuplicateUserError(f"Email '{email}' already registered")

        # Hash password securely
        password_hash = generate_password_hash(password, method="pbkdf2:sha256")

        # Create user
        user = self.user_repo.create(
            user_id=user_id,
            email=email,
            full_name=full_name,
            password_hash=password_hash,
            mobile_number=mobile_number,
        )

        logger.info(f"New user registered successfully: {user_id}")

        return {
            "user_id": user.user_id,
            "email": user.email,
            "full_name": user.full_name,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }

    def authenticate_user(self, user_id: str, password: str) -> Dict[str, Any]:
        """Authenticate user credentials
        
        Args:
            user_id: User identifier
            password: Plain text password
            
        Returns:
            Dict with user session information
            
        Raises:
            AuthenticationError: If credentials are invalid
        """
        # Validate inputs
        if not user_id or not password:
            raise ValidationError("User ID and password are required")

        # Fetch user from database
        user = self.user_repo.get_by_user_id(user_id)

        if not user:
            logger.warning(f"Authentication failed: user not found - {user_id}")
            # Return generic error to prevent user enumeration
            raise AuthenticationError("Invalid credentials")

        if not user.is_active:
            logger.warning(f"Authentication failed: inactive user - {user_id}")
            raise AuthenticationError("Account is inactive")

        # Verify password
        if not check_password_hash(user.password_hash, password):
            logger.warning(f"Authentication failed: incorrect password - {user_id}")
            raise AuthenticationError("Invalid credentials")

        logger.info(f"User authenticated successfully: {user_id}")

        return {
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
        }

    def authenticate_admin(self, admin_id: str, password: str) -> Dict[str, Any]:
        """Authenticate admin credentials
        
        Args:
            admin_id: Admin identifier
            password: Plain text password
            
        Returns:
            Dict with admin session information
            
        Raises:
            AuthenticationError: If credentials are invalid
        """
        if not admin_id or not password:
            raise ValidationError("Admin ID and password are required")

        # Fetch admin
        admin = self.admin_repo.get_by_admin_id(admin_id)

        if not admin:
            logger.warning(f"Admin authentication failed: not found - {admin_id}")
            raise AuthenticationError("Invalid admin credentials")

        # Verify password
        if not check_password_hash(admin.password_hash, password):
            logger.warning(f"Admin authentication failed: incorrect password - {admin_id}")
            raise AuthenticationError("Invalid admin credentials")

        logger.info(f"Admin authenticated successfully: {admin_id}")

        return {"admin_id": admin.admin_id}

    def _validate_user_registration(
        self, user_id: str, email: str, full_name: str, password: str
    ) -> None:
        """Validate user registration inputs
        
        Raises:
            ValidationError: If validation fails
        """
        errors = []

        # User ID validation
        if not user_id or len(user_id) < 3:
            errors.append("User ID must be at least 3 characters")
        elif not user_id.isalnum():
            errors.append("User ID must contain only letters and numbers")

        # Email validation
        if not email:
            errors.append("Email is required")
        elif not self.EMAIL_PATTERN.match(email):
            errors.append("Invalid email format")

        # Full name validation
        if not full_name or len(full_name.strip()) < 2:
            errors.append("Full name must be at least 2 characters")

        # Password validation
        if not password:
            errors.append("Password is required")
        elif len(password) < self.MIN_PASSWORD_LENGTH:
            errors.append(
                f"Password must be at least {self.MIN_PASSWORD_LENGTH} characters"
            )
        elif not self.PASSWORD_PATTERN.match(password):
            errors.append(
                "Password must contain uppercase, lowercase, number, and special character"
            )

        if errors:
            raise ValidationError("; ".join(errors))

    def change_password(
        self, user_id: str, old_password: str, new_password: str
    ) -> bool:
        """Change user password
        
        Args:
            user_id: User identifier
            old_password: Current password
            new_password: New password
            
        Returns:
            True if successful
            
        Raises:
            AuthenticationError: If old password is incorrect
            ValidationError: If new password doesn't meet requirements
        """
        # Authenticate with old password first
        self.authenticate_user(user_id, old_password)

        # Validate new password
        if len(new_password) < self.MIN_PASSWORD_LENGTH:
            raise ValidationError(
                f"Password must be at least {self.MIN_PASSWORD_LENGTH} characters"
            )
        if not self.PASSWORD_PATTERN.match(new_password):
            raise ValidationError(
                "Password must contain uppercase, lowercase, number, and special character"
            )

        # Hash and update password
        new_password_hash = generate_password_hash(new_password, method="pbkdf2:sha256")
        self.user_repo.update_password(user_id, new_password_hash)

        logger.info(f"Password changed successfully for user: {user_id}")
        return True
