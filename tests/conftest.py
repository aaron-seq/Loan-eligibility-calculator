"""Pytest Configuration and Fixtures

Defines shared fixtures for testing.
"""

import os
import sys
import pytest
import tempfile
import shutil
from faker import Faker

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".." )))

from src.repositories.user_repository import UserRepository
from src.repositories.admin_repository import AdminRepository
from src.repositories.prediction_repository import PredictionRepository
from src.services.auth_service import AuthService
from src.services.prediction_service import PredictionService
from src.ml.model_loader import ModelLoader
from src.utils.database import DatabaseManager


@pytest.fixture(scope="session")
def faker_instance():
    """Faker instance for generating realistic test data"""
    return Faker()


@pytest.fixture(scope="function")
def test_db_path(tmp_path):
    """Create temporary database for testing"""
    db_path = tmp_path / "test_loan_prediction.db"
    yield str(db_path)
    # Cleanup after test
    if db_path.exists():
        db_path.unlink()


@pytest.fixture(scope="function")
def db_manager(test_db_path):
    """Database manager with test database"""
    manager = DatabaseManager(test_db_path)
    manager.initialize_database()
    yield manager
    manager.close()


@pytest.fixture(scope="function")
def user_repository(test_db_path, db_manager):
    """User repository fixture"""
    return UserRepository(test_db_path)


@pytest.fixture(scope="function")
def admin_repository(test_db_path, db_manager):
    """Admin repository fixture"""
    return AdminRepository(test_db_path)


@pytest.fixture(scope="function")
def prediction_repository(test_db_path, db_manager):
    """Prediction repository fixture"""
    return PredictionRepository(test_db_path)


@pytest.fixture(scope="function")
def auth_service(user_repository, admin_repository):
    """Authentication service fixture"""
    return AuthService(user_repository, admin_repository)


@pytest.fixture(scope="function")
def model_loader_mock():
    """Mock model loader for testing without actual models"""
    class MockModelLoader:
        def is_model_loaded(self):
            return True

        def predict(self, features):
            import numpy as np
            return np.array([1])  # Always approve

        def predict_proba(self, features):
            import numpy as np
            return np.array([[0.2, 0.8]])  # 80% confidence

        def scale_features(self, features):
            return features

    return MockModelLoader()


@pytest.fixture(scope="function")
def prediction_service(prediction_repository, model_loader_mock):
    """Prediction service fixture with mocked model"""
    return PredictionService(prediction_repository, model_loader_mock)


@pytest.fixture(scope="function")
def sample_user_data(faker_instance):
    """Generate realistic user registration data"""
    return {
        "user_id": faker_instance.user_name(),
        "email": faker_instance.email(),
        "full_name": faker_instance.name(),
        "password": "SecurePass123!",
        "mobile_number": faker_instance.phone_number()[:15],
    }


@pytest.fixture(scope="function")
def sample_loan_data(faker_instance):
    """Generate realistic loan application data"""
    return {
        "first_name": faker_instance.first_name(),
        "last_name": faker_instance.last_name(),
        "gender": 1,
        "marital_status": 1,
        "dependents": 2,
        "education": 0,
        "employment_type": 0,
        "property_area": 1,
        "credit_history": 1.0,
        "cibil_score": 750,
        "applicant_income": 50000.0,
        "coapplicant_income": 30000.0,
        "loan_amount": 200000.0,
        "loan_tenure": 360.0,
    }
