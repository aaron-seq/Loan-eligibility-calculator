"""Tests for Prediction Service

Tests follow the 7-Step Validation Protocol with realistic data.
"""

import pytest

from src.utils.exceptions import ValidationError, PredictionError


class TestLoanPrediction:
    """Test loan eligibility prediction service"""

    def test_successful_prediction_with_valid_data(
        self, prediction_service, sample_loan_data, prediction_repository
    ):
        """Test successful prediction with valid loan application
        
        Requirement: System should generate prediction for valid loan data
        """
        user_id = "test_user_001"

        # Before State: No predictions
        initial_count = prediction_repository.count("loan_predictions")

        # Action: Generate prediction
        result = prediction_service.predict_loan_eligibility(
            user_id=user_id, loan_data=sample_loan_data
        )

        # After State: Verify prediction created
        final_count = prediction_repository.count("loan_predictions")
        assert final_count == initial_count + 1

        # Verify specific result structure
        assert "prediction_id" in result
        assert "decision" in result
        assert "confidence" in result
        assert "risk_level" in result
        assert "timestamp" in result

        # Verify decision is one of expected values
        assert result["decision"] in ["APPROVED", "REJECTED"]

        # Verify confidence is percentage between 0 and 100
        assert 0 <= result["confidence"] <= 100

        # Verify risk level is one of expected values
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_prediction_fails_with_negative_income(
        self, prediction_service, sample_loan_data
    ):
        """Test prediction failure with negative income
        
        Requirement: System must reject negative income values
        """
        sample_loan_data["applicant_income"] = -10000.0

        with pytest.raises(ValidationError) as exc_info:
            prediction_service.predict_loan_eligibility(
                user_id="test_user", loan_data=sample_loan_data
            )

        assert "income" in str(exc_info.value).lower()

    def test_prediction_fails_with_excessive_loan_amount(
        self, prediction_service, sample_loan_data
    ):
        """Test prediction failure with excessive loan amount
        
        Requirement: Loan amount must be within acceptable range
        """
        sample_loan_data["loan_amount"] = 100000000.0  # 10 Crores - excessive

        with pytest.raises(ValidationError) as exc_info:
            prediction_service.predict_loan_eligibility(
                user_id="test_user", loan_data=sample_loan_data
            )

        assert "loan amount" in str(exc_info.value).lower()

    def test_prediction_fails_with_invalid_cibil_score(
        self, prediction_service, sample_loan_data
    ):
        """Test prediction failure with invalid CIBIL score
        
        Requirement: CIBIL score must be between 300 and 900
        """
        sample_loan_data["cibil_score"] = 1000  # Above maximum

        with pytest.raises(ValidationError) as exc_info:
            prediction_service.predict_loan_eligibility(
                user_id="test_user", loan_data=sample_loan_data
            )

        assert "cibil" in str(exc_info.value).lower()

    def test_prediction_fails_with_missing_required_fields(
        self, prediction_service
    ):
        """Test prediction failure when required fields are missing
        
        Requirement: All required fields must be provided
        """
        incomplete_data = {
            "first_name": "John",
            # Missing many required fields
        }

        with pytest.raises(ValidationError) as exc_info:
            prediction_service.predict_loan_eligibility(
                user_id="test_user", loan_data=incomplete_data
            )

        assert "required" in str(exc_info.value).lower()

    def test_get_user_prediction_history(
        self, prediction_service, sample_loan_data, prediction_repository
    ):
        """Test retrieving user prediction history
        
        Requirement: Users should be able to view their prediction history
        """
        user_id = "test_user_history"

        # Create multiple predictions
        for _ in range(3):
            prediction_service.predict_loan_eligibility(
                user_id=user_id, loan_data=sample_loan_data
            )

        # Retrieve history
        history = prediction_service.get_user_prediction_history(user_id, limit=10)

        # Verify: Should have exactly 3 predictions
        assert len(history) == 3

        # Verify structure of history items
        for item in history:
            assert "id" in item
            assert "loan_amount" in item
            assert "prediction" in item
            assert "confidence" in item
            assert "created_at" in item
