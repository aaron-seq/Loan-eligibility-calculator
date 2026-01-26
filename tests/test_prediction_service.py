"""Unit Tests for PredictionService

Requirement Extraction:
- Predictions must validate loan parameters
- Feature engineering must match training process exactly
- Predictions must be saved for audit trail
- High loan-to-income ratios must be logged

Domain Realism: Uses realistic financial data.
"""

import pytest
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.services.prediction_service import PredictionService
from src.models.schemas import LoanPredictionRequest
from src.exceptions import (
    ModelNotAvailableError,
    ValidationError,
    PredictionError,
)


class TestPredictionService:
    """Test prediction business logic"""

    @pytest.fixture
    def mock_model(self, tmp_path):
        """Create mock ML model for testing
        
        Avoids dependency on actual trained models.
        """
        # Create simple mock model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        scaler = StandardScaler()

        # Create dummy training data with 22 features
        X_train = np.random.rand(100, 22)
        y_train = np.random.randint(0, 2, 100)

        # Train mock model
        X_scaled = scaler.fit_transform(X_train)
        model.fit(X_scaled, y_train)

        # Save to temp directory
        model_path = tmp_path / "models"
        model_path.mkdir()

        with open(model_path / "loan_eligibility_model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open(model_path / "feature_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        return str(model_path)

    def test_prediction_with_valid_data(
        self, db_connection, realistic_loan_data, mock_model
    ):
        """Test successful prediction with realistic data
        
        7-Step Process applied:
        1. Requirement: Generate loan prediction
        2. Domain Realism: Realistic loan amounts and incomes
        3. Positive Testing: Valid input
        4. Specificity: Check exact prediction structure
        5. Automation: pytest
        6. Manual Curation: Verified data ranges
        7. Reproducibility: Mock model for consistency
        """
        # Setup
        prediction_service = PredictionService(db_connection, mock_model)
        user_id = "test_user_for_prediction"

        # Arrange
        request = LoanPredictionRequest(**realistic_loan_data)

        # Before State: No predictions
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM loan_predictions")
            initial_count = cursor.fetchone()["count"]

        # Act
        result = prediction_service.predict_loan_eligibility(user_id, request)

        # After State: One prediction added
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM loan_predictions")
            final_count = cursor.fetchone()["count"]

        # Specificity: Exact assertions
        assert final_count == initial_count + 1, "Exactly one prediction should be saved"
        assert result.prediction_result in [
            "APPROVED",
            "REJECTED",
        ], "Prediction must be APPROVED or REJECTED"
        assert 0 <= result.confidence_score <= 100, "Confidence must be 0-100%"
        assert isinstance(result.message, str)
        assert len(result.message) > 0

    def test_negative_income_rejected(self, db_connection, realistic_loan_data, mock_model):
        """Test that negative income is rejected
        
        Negative Testing: Invalid input validation.
        """
        prediction_service = PredictionService(db_connection, mock_model)
        user_id = "test_user"

        # Arrange: Negative income
        realistic_loan_data["applicant_income"] = -50000.0
        request = LoanPredictionRequest(**realistic_loan_data)

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            prediction_service.predict_loan_eligibility(user_id, request)

        assert "positive" in str(exc_info.value).lower()

    def test_zero_loan_amount_rejected(self, db_connection, realistic_loan_data, mock_model):
        """Test that zero loan amount is rejected
        
        Negative Testing: Invalid loan amount.
        """
        prediction_service = PredictionService(db_connection, mock_model)
        user_id = "test_user"

        # Arrange
        realistic_loan_data["loan_amount"] = 0.0
        request = LoanPredictionRequest(**realistic_loan_data)

        # Act & Assert
        with pytest.raises(ValidationError):
            prediction_service.predict_loan_eligibility(user_id, request)

    def test_invalid_cibil_score_rejected(self, db_connection, realistic_loan_data, mock_model):
        """Test that invalid CIBIL score is rejected
        
        Requirement: CIBIL score must be 300-900.
        """
        prediction_service = PredictionService(db_connection, mock_model)
        user_id = "test_user"

        # Test below minimum
        realistic_loan_data["cibil_score"] = 299
        request = LoanPredictionRequest(**realistic_loan_data)

        with pytest.raises(ValidationError) as exc_info:
            prediction_service.predict_loan_eligibility(user_id, request)

        assert "cibil" in str(exc_info.value).lower()

        # Test above maximum
        realistic_loan_data["cibil_score"] = 901
        request = LoanPredictionRequest(**realistic_loan_data)

        with pytest.raises(ValidationError):
            prediction_service.predict_loan_eligibility(user_id, request)

    def test_model_unavailable_error(self, db_connection, realistic_loan_data, tmp_path):
        """Test proper error when ML model is not available
        
        Negative Testing: Service degradation.
        """
        # Create empty model directory
        empty_model_path = str(tmp_path / "empty_models")
        os.makedirs(empty_model_path, exist_ok=True)

        prediction_service = PredictionService(db_connection, empty_model_path)
        user_id = "test_user"

        request = LoanPredictionRequest(**realistic_loan_data)

        # Act & Assert
        with pytest.raises(ModelNotAvailableError) as exc_info:
            prediction_service.predict_loan_eligibility(user_id, request)

        assert "unavailable" in str(exc_info.value).lower()
