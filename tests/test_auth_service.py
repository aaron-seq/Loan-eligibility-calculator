"""Tests for Authentication Service

Tests follow the 7-Step Validation Protocol.
"""

import pytest
from werkzeug.security import check_password_hash

from src.utils.exceptions import (
    ValidationError,
    AuthenticationError,
    DuplicateUserError,
)


class TestUserRegistration:
    """Test user registration with positive and negative cases"""

    def test_successful_registration_with_realistic_data(self, auth_service, sample_user_data, user_repository):
        """Test successful user registration with Faker-generated data
        
        Requirement: User should be able to register with valid credentials
        """
        # Before State: No users in database
        initial_count = user_repository.count("users")
        assert initial_count == 0

        # Action: Register user
        result = auth_service.register_user(**sample_user_data)

        # After State: Verify exact delta
        final_count = user_repository.count("users")
        assert final_count == initial_count + 1

        # Verify specific data
        assert result["user_id"] == sample_user_data["user_id"]
        assert result["email"] == sample_user_data["email"]
        assert result["full_name"] == sample_user_data["full_name"]
        assert "created_at" in result

        # Verify password is hashed
        user = user_repository.get_by_user_id(sample_user_data["user_id"])
        assert user is not None
        assert check_password_hash(user.password_hash, sample_user_data["password"])

    def test_registration_fails_with_duplicate_user_id(self, auth_service, sample_user_data):
        """Test registration failure with duplicate user_id
        
        Requirement: System must prevent duplicate user_id registration
        """
        # Register user once
        auth_service.register_user(**sample_user_data)

        # Attempt to register again with same user_id
        with pytest.raises(DuplicateUserError) as exc_info:
            auth_service.register_user(**sample_user_data)

        assert "already exists" in str(exc_info.value).lower()

    def test_registration_fails_with_weak_password(self, auth_service, sample_user_data):
        """Test registration failure with weak password
        
        Requirement: Password must meet security requirements
        """
        sample_user_data["password"] = "weak"  # Too short, no special chars

        with pytest.raises(ValidationError) as exc_info:
            auth_service.register_user(**sample_user_data)

        error_message = str(exc_info.value).lower()
        assert "password" in error_message

    def test_registration_fails_with_invalid_email(self, auth_service, sample_user_data):
        """Test registration failure with invalid email format
        
        Requirement: Email must be in valid format
        """
        sample_user_data["email"] = "not-an-email"

        with pytest.raises(ValidationError) as exc_info:
            auth_service.register_user(**sample_user_data)

        assert "email" in str(exc_info.value).lower()

    def test_registration_fails_with_short_user_id(self, auth_service, sample_user_data):
        """Test registration failure with too short user_id
        
        Requirement: User ID must be at least 3 characters
        """
        sample_user_data["user_id"] = "ab"  # Only 2 characters

        with pytest.raises(ValidationError) as exc_info:
            auth_service.register_user(**sample_user_data)

        assert "user id" in str(exc_info.value).lower()


class TestUserAuthentication:
    """Test user authentication with positive and negative cases"""

    def test_successful_authentication(self, auth_service, sample_user_data):
        """Test successful user authentication
        
        Requirement: User should be able to login with correct credentials
        """
        # Setup: Register user
        auth_service.register_user(**sample_user_data)

        # Action: Authenticate user
        result = auth_service.authenticate_user(
            user_id=sample_user_data["user_id"],
            password=sample_user_data["password"],
        )

        # Verify: Check returned data
        assert result["user_id"] == sample_user_data["user_id"]
        assert result["email"] == sample_user_data["email"]
        assert result["full_name"] == sample_user_data["full_name"]

    def test_authentication_fails_with_wrong_password(self, auth_service, sample_user_data):
        """Test authentication failure with incorrect password
        
        Requirement: System must reject incorrect passwords
        """
        # Setup: Register user
        auth_service.register_user(**sample_user_data)

        # Action & Verify: Attempt authentication with wrong password
        with pytest.raises(AuthenticationError) as exc_info:
            auth_service.authenticate_user(
                user_id=sample_user_data["user_id"],
                password="WrongPassword123!",
            )

        assert "invalid credentials" in str(exc_info.value).lower()

    def test_authentication_fails_with_nonexistent_user(self, auth_service):
        """Test authentication failure with non-existent user
        
        Requirement: System must not reveal whether user exists
        """
        with pytest.raises(AuthenticationError) as exc_info:
            auth_service.authenticate_user(
                user_id="nonexistent_user",
                password="AnyPassword123!",
            )

        # Generic error message to prevent user enumeration
        assert "invalid credentials" in str(exc_info.value).lower()
