"""Pytest Configuration and Shared Fixtures

Provides reusable fixtures for all tests.
"""

import os
import pytest
import tempfile
import sqlite3
from contextlib import contextmanager
from faker import Faker

from src.config import Settings
from src.utils.database import initialize_database

fake = Faker()


@pytest.fixture(scope="session")
def test_settings():
    """Create test configuration
    
    Uses in-memory database for isolation.
    """
    # Create temporary directory for test database
    temp_dir = tempfile.mkdtemp()
    test_db_path = os.path.join(temp_dir, "test_loan.db")

    settings = Settings(
        SECRET_KEY="test-secret-key-do-not-use-in-production",
        DATABASE_PATH=test_db_path,
        MODEL_PATH="models",
        FLASK_ENV="testing",
        DEBUG=False,
    )

    yield settings

    # Cleanup
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    os.rmdir(temp_dir)


@pytest.fixture(scope="function")
def db_connection(test_settings):
    """Provide clean database connection for each test
    
    Before/After State Methodology:
    - Each test gets fresh database
    - State is isolated between tests
    """
    # Initialize fresh database
    initialize_database(test_settings.DATABASE_PATH)

    @contextmanager
    def get_connection():
        conn = sqlite3.connect(test_settings.DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    yield get_connection

    # Cleanup database after test
    if os.path.exists(test_settings.DATABASE_PATH):
        os.remove(test_settings.DATABASE_PATH)


@pytest.fixture
def realistic_user_data():
    """Generate realistic user registration data
    
    Domain Realism: Uses Faker for production-like data.
    No "foo", "bar", or "test" placeholders.
    """
    return {
        "user_id": fake.user_name(),
        "email": fake.email(),
        "full_name": fake.name(),
        "mobile_number": fake.phone_number()[:15],  # Limit to 15 digits
        "password": fake.password(length=12, special_chars=True),
        "confirm_password": None,  # Will be set same as password
    }


@pytest.fixture
def realistic_loan_data():
    """Generate realistic loan application data
    
    Domain Realism: Uses real-world ranges for loan parameters.
    """
    return {
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "gender": fake.random_int(min=0, max=1),
        "married": fake.random_int(min=0, max=1),
        "dependents": fake.random_int(min=0, max=3),
        "education": fake.random_int(min=0, max=1),
        "self_employed": fake.random_int(min=0, max=1),
        "property_area": fake.random_int(min=0, max=2),
        "credit_history": fake.random_int(min=0, max=1),
        "cibil_score": fake.random_int(min=300, max=900),
        "applicant_income": float(fake.random_int(min=20000, max=500000)),
        "coapplicant_income": float(fake.random_int(min=0, max=300000)),
        "loan_amount": float(fake.random_int(min=50000, max=5000000)),
        "loan_amount_term": float(fake.random_int(min=12, max=360)),
    }


@pytest.fixture
def admin_credentials():
    """Provide admin credentials for testing"""
    return {
        "admin_id": "test_admin",
        "password": "test_admin_password_123",
    }
