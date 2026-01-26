"""Prediction Service - Handles ML prediction logic

This service encapsulates all machine learning prediction logic.
Separated from controllers to enable independent testing and reuse.
"""

import os
import pickle
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import sqlite3

from src.exceptions import (
    ModelNotAvailableError,
    PredictionError,
    ValidationError,
)
from src.logger import get_logger
from src.models.schemas import LoanPredictionRequest, PredictionResponse

logger = get_logger(__name__)


class PredictionService:
    """Service for loan eligibility predictions"""

    def __init__(self, db_connection_factory, model_path: str):
        """Initialize prediction service
        
        Args:
            db_connection_factory: Context manager for database connections
            model_path: Path to ML model files directory
        """
        self.db_connection_factory = db_connection_factory
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self._load_models()

    def _load_models(self):
        """Load ML models with comprehensive error handling
        
        Models are loaded once during initialization to avoid repeated I/O.
        If models fail to load, predictions will be unavailable.
        """
        try:
            model_file = os.path.join(self.model_path, "loan_eligibility_model.pkl")
            scaler_file = os.path.join(self.model_path, "feature_scaler.pkl")

            if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                logger.error(
                    f"Model files not found in {self.model_path}. "
                    "Prediction service will be unavailable."
                )
                return

            with open(model_file, "rb") as f:
                self.model = pickle.load(f)

            with open(scaler_file, "rb") as f:
                self.scaler = pickle.load(f)

            logger.info("ML models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load ML models: {e}", exc_info=True)
            self.model = None
            self.scaler = None

    def is_available(self) -> bool:
        """Check if prediction service is available
        
        Returns:
            True if models are loaded and ready
        """
        return self.model is not None and self.scaler is not None

    def predict_loan_eligibility(
        self, user_id: str, request: LoanPredictionRequest
    ) -> PredictionResponse:
        """Generate loan eligibility prediction
        
        Business Rules:
        - Applicant income must be positive
        - Loan amount must be reasonable relative to income
        - Credit history heavily influences decision
        - CIBIL score is a strong predictor
        
        Args:
            user_id: User making the prediction request
            request: Validated loan prediction request
            
        Returns:
            PredictionResponse with decision and confidence
            
        Raises:
            ModelNotAvailableError: If ML models are not loaded
            PredictionError: If prediction fails
            ValidationError: If business rules are violated
        """
        if not self.is_available():
            logger.error("Prediction requested but models not available")
            raise ModelNotAvailableError(
                "Prediction service is currently unavailable. Please try again later."
            )

        logger.info(f"Processing prediction request for user: {user_id}")

        # Validate business rules
        self._validate_loan_request(request)

        try:
            # Feature engineering
            features = self._engineer_features(request)

            # Create feature vector in correct order
            feature_vector = self._create_feature_vector(features)

            # Scale and predict
            scaled_features = self.scaler.transform(feature_vector)
            prediction = self.model.predict(scaled_features)
            prediction_proba = self.model.predict_proba(scaled_features)

            # Format response
            decision = "APPROVED" if prediction[0] == 1 else "REJECTED"
            confidence = float(prediction_proba[0][prediction[0]] * 100)

            result = PredictionResponse(
                prediction_result=decision,
                confidence_score=round(confidence, 2),
                message=f"Loan application {decision.lower()} with {confidence:.1f}% confidence",
            )

            # Save prediction to database for audit trail
            self._save_prediction(user_id, request, result)

            logger.info(
                f"Prediction completed for user {user_id}: {decision} ({confidence:.2f}%)"
            )
            return result

        except (ModelNotAvailableError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Prediction failed for user {user_id}: {e}", exc_info=True)
            raise PredictionError("Failed to generate prediction. Please try again.")

    def _validate_loan_request(self, request: LoanPredictionRequest):
        """Validate business rules for loan request
        
        Pydantic handles type validation, this handles business logic validation.
        """
        # Income validation
        if request.applicant_income <= 0:
            raise ValidationError("Applicant income must be positive")

        if request.coapplicant_income < 0:
            raise ValidationError("Co-applicant income cannot be negative")

        # Loan amount validation
        if request.loan_amount <= 0:
            raise ValidationError("Loan amount must be positive")

        if request.loan_amount_term <= 0:
            raise ValidationError("Loan term must be positive")

        # Business rule: Loan-to-income ratio check
        total_income = request.applicant_income + request.coapplicant_income
        loan_to_income_ratio = request.loan_amount / total_income if total_income > 0 else float('inf')
        
        if loan_to_income_ratio > 10:
            logger.warning(
                f"High loan-to-income ratio detected: {loan_to_income_ratio:.2f}"
            )
            # Note: Not raising error, but logging for monitoring

        # CIBIL score validation
        if not (300 <= request.cibil_score <= 900):
            raise ValidationError("CIBIL score must be between 300 and 900")

        # Credit history must be binary
        if request.credit_history not in [0, 1]:
            raise ValidationError("Credit history must be 0 or 1")

    def _engineer_features(self, request: LoanPredictionRequest) -> Dict[str, float]:
        """Engineer features for ML model
        
        Feature engineering must match exactly what was done during training.
        This is critical for model accuracy.
        """
        total_income = request.applicant_income + request.coapplicant_income

        # Loan-to-income ratio
        loan_to_income_ratio = (
            request.loan_amount / total_income if total_income > 0 else 0
        )

        # EMI calculation
        emi = (
            request.loan_amount / request.loan_amount_term
            if request.loan_amount_term > 0
            else 0
        )

        # Log transformations (add 1 to avoid log(0))
        applicant_income_log = np.log(request.applicant_income + 1)
        coapplicant_income_log = np.log(request.coapplicant_income + 1)
        loan_amount_log = np.log(request.loan_amount + 1)
        total_income_log = np.log(total_income + 1)

        return {
            "ApplicantIncome": request.applicant_income,
            "ApplicantIncome_log": applicant_income_log,
            "CIBIL_Score": request.cibil_score,
            "CoapplicantIncome": request.coapplicant_income,
            "CoapplicantIncome_log": coapplicant_income_log,
            "Credit_History": request.credit_history,
            "Dependents_1": 1 if request.dependents == 1 else 0,
            "Dependents_2": 1 if request.dependents == 2 else 0,
            "Dependents_3+": 1 if request.dependents >= 3 else 0,
            "Education_Not Graduate": 1 if request.education == 1 else 0,
            "EMI_feature": emi,
            "Gender_Male": 1 if request.gender == 1 else 0,
            "LoanAmount": request.loan_amount,
            "LoanAmount_log": loan_amount_log,
            "Loan_Amount_Term": request.loan_amount_term,
            "Loan_to_Income_Ratio": loan_to_income_ratio,
            "Married_Yes": 1 if request.married == 1 else 0,
            "Property_Area_Semiurban": 1 if request.property_area == 1 else 0,
            "Property_Area_Urban": 1 if request.property_area == 2 else 0,
            "Self_Employed_Yes": 1 if request.self_employed == 1 else 0,
            "Total_Income": total_income,
            "Total_Income_log": total_income_log,
        }

    def _create_feature_vector(self, features: Dict[str, float]) -> pd.DataFrame:
        """Create feature vector in exact order expected by model
        
        Feature order MUST match training exactly.
        Pandas get_dummies() creates features in alphabetical order.
        """
        return pd.DataFrame([features])

    def _save_prediction(
        self, user_id: str, request: LoanPredictionRequest, result: PredictionResponse
    ):
        """Save prediction to database for audit trail
        
        Business Rule: All predictions must be logged for:
        - Compliance and audit requirements
        - Model performance monitoring
        - User history tracking
        """
        try:
            with self.db_connection_factory() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO loan_predictions (
                        user_id, first_name, last_name, gender, marital_status,
                        dependents_count, education_level, employment_type, property_location,
                        credit_history, cibil_score, applicant_income, coapplicant_income,
                        loan_amount, loan_tenure, prediction_result, confidence_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        user_id,
                        request.first_name,
                        request.last_name,
                        request.gender,
                        request.married,
                        request.dependents,
                        request.education,
                        request.self_employed,
                        request.property_area,
                        request.credit_history,
                        request.cibil_score,
                        request.applicant_income,
                        request.coapplicant_income,
                        request.loan_amount,
                        request.loan_amount_term,
                        result.prediction_result,
                        result.confidence_score,
                    ),
                )
                conn.commit()

            logger.info(f"Prediction saved for user: {user_id}")

        except Exception as e:
            # Log error but don't fail the prediction
            # Prediction succeeded, saving is secondary
            logger.error(
                f"Failed to save prediction for user {user_id}: {e}", exc_info=True
            )

    def get_user_predictions(self, user_id: str, limit: int = 10) -> list:
        """Retrieve prediction history for a user
        
        Args:
            user_id: User identifier
            limit: Maximum number of predictions to retrieve
            
        Returns:
            List of prediction records
        """
        try:
            with self.db_connection_factory() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM loan_predictions
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (user_id, limit),
                )
                predictions = cursor.fetchall()

            return [dict(row) for row in predictions]

        except Exception as e:
            logger.error(
                f"Error retrieving predictions for user {user_id}: {e}", exc_info=True
            )
            raise
