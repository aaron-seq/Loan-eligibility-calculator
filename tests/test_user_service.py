"""Unit Tests for UserService

Requirement Extraction:
- User registration must validate email, username, and password
- User authentication must verify credentials securely
- Failed logins must be logged
- Duplicate users must be rejected

Domain Realism: Uses Faker for all test data.
"""

import pytest
from werkzeug.security import check_password_hash

from src.services.user_service import UserService
from src.models.schemas import UserRegistrationRequest, UserLoginRequest
from src.exceptions import (
    UserAlreadyExistsError,
    InvalidCredentialsError,
    ValidationError,
)


class TestUserRegistration:
    """Test user registration business logic"""

    def test_successful_registration_with_realistic_data(
        self, db_connection, realistic_user_data
    ):
        """Test successful user registration with realistic data
        
        7-Step Process:
        1. Requirement: User can register with valid credentials
        2. Domain Realism: Using Faker-generated data
        3. Positive Testing: Happy path
        4. Specificity: Exact assertions on returned data
        5. Automation: pytest
        6. Manual Curation: Reviewed test data ranges
        7. Reproducibility: Isolated database per test
        """
        # Setup: Create service
        user_service = UserService(db_connection)

        # Arrange: Prepare request
        realistic_user_data["confirm_password"] = realistic_user_data["password"]
        request = UserRegistrationRequest(**realistic_user_data)

        # Before State: Verify no users exist
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM users")
            initial_count = cursor.fetchone()["count"]
            assert initial_count == 0, "Database should start empty"

        # Act: Register user
        result = user_service.register_user(request)

        # After State: Verify exactly one user added
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM users")
            final_count = cursor.fetchone()["count"]
            assert final_count == initial_count + 1, "Exactly one user should be added"

            # Verify user details
            cursor.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (realistic_user_data["user_id"],),
            )
            user = cursor.fetchone()

        # Specificity: Exact assertions
        assert user is not None, "User should exist in database"
        assert user["user_id"] == realistic_user_data["user_id"]
        assert user["email"] == realistic_user_data["email"]
        assert user["full_name"] == realistic_user_data["full_name"]
        assert user["is_active"] == 1, "New users should be active"
        assert check_password_hash(
            user["password_hash"], realistic_user_data["password"]
        ), "Password should be hashed correctly"

        # Verify service response
        assert result["user_id"] == realistic_user_data["user_id"]
        assert "message" in result

    def test_duplicate_username_rejected(self, db_connection, realistic_user_data):
        """Test that duplicate usernames are properly rejected
        
        Negative Testing: Testing failure path.
        """
        user_service = UserService(db_connection)

        # Arrange: Register first user
        realistic_user_data["confirm_password"] = realistic_user_data["password"]
        request1 = UserRegistrationRequest(**realistic_user_data)
        user_service.register_user(request1)

        # Act & Assert: Attempt duplicate registration
        request2 = UserRegistrationRequest(**realistic_user_data)
        with pytest.raises(UserAlreadyExistsError) as exc_info:
            user_service.register_user(request2)

        # Specificity: Check exact error message
        assert "already exists" in str(exc_info.value).lower()

        # Before/After: Verify count didn't change
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM users")
            count = cursor.fetchone()["count"]
            assert count == 1, "Only one user should exist"

    def test_invalid_email_rejected(self, db_connection, realistic_user_data):
        """Test that invalid email formats are rejected
        
        Negative Testing: Invalid input validation.
        """
        user_service = UserService(db_connection)

        # Arrange: Invalid email formats
        invalid_emails = [
            "notanemail",
            "@example.com",
            "user@",
            "user @example.com",
            "user..name@example.com",
        ]

        for invalid_email in invalid_emails:
            realistic_user_data["email"] = invalid_email
            realistic_user_data["confirm_password"] = realistic_user_data["password"]
            request = UserRegistrationRequest(**realistic_user_data)

            # Act & Assert
            with pytest.raises(ValidationError) as exc_info:
                user_service.register_user(request)

            assert "email" in str(exc_info.value).lower()

    def test_short_password_rejected(self, db_connection, realistic_user_data):
        """Test that passwords under minimum length are rejected
        
        Requirement: Passwords must meet security requirements.
        """
        user_service = UserService(db_connection)

        # Arrange: Too short password
        realistic_user_data["password"] = "short"
        realistic_user_data["confirm_password"] = "short"

        # Act & Assert: Pydantic should catch this
        with pytest.raises(Exception):  # Pydantic ValidationError
            request = UserRegistrationRequest(**realistic_user_data)


class TestUserAuthentication:
    """Test user authentication business logic"""

    def test_successful_authentication(self, db_connection, realistic_user_data):
        """Test successful user authentication
        
        Positive Testing: Valid credentials.
        """
        user_service = UserService(db_connection)

        # Arrange: Register user first
        realistic_user_data["confirm_password"] = realistic_user_data["password"]
        reg_request = UserRegistrationRequest(**realistic_user_data)
        user_service.register_user(reg_request)

        # Act: Authenticate
        login_request = UserLoginRequest(
            user_id=realistic_user_data["user_id"],
            password=realistic_user_data["password"],
        )
        result = user_service.authenticate_user(login_request)

        # Assert: Specificity
        assert result["user_id"] == realistic_user_data["user_id"]
        assert result["full_name"] == realistic_user_data["full_name"]
        assert "user_id" in result
        assert "full_name" in result

    def test_wrong_password_rejected(self, db_connection, realistic_user_data):
        """Test that wrong password is rejected
        
        Negative Testing: Invalid credentials.
        """
        user_service = UserService(db_connection)

        # Arrange: Register user
        realistic_user_data["confirm_password"] = realistic_user_data["password"]
        reg_request = UserRegistrationRequest(**realistic_user_data)
        user_service.register_user(reg_request)

        # Act & Assert: Wrong password
        login_request = UserLoginRequest(
            user_id=realistic_user_data["user_id"],
            password="wrong_password_12345",
        )

        with pytest.raises(InvalidCredentialsError) as exc_info:
            user_service.authenticate_user(login_request)

        # Specificity
        assert "invalid" in str(exc_info.value).lower()

    def test_nonexistent_user_rejected(self, db_connection):
        """Test that authentication fails for non-existent users
        
        Negative Testing: User doesn't exist.
        """
        user_service = UserService(db_connection)

        # Act & Assert
        login_request = UserLoginRequest(
            user_id="nonexistent_user_abc123",
            password="some_password_456",
        )

        with pytest.raises(InvalidCredentialsError):
            user_service.authenticate_user(login_request)
