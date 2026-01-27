"""Prediction Service - ML Business Logic

Handles loan eligibility prediction logic separate from HTTP layer.
Pure business logic with comprehensive validation and error handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.exceptions import ValidationError, PredictionError, ModelNotAvailableError
from src.repositories.prediction_repository import PredictionRepository
from src.ml.model_loader import ModelLoader

logger = get_logger(__name__)


class PredictionService:
    """Service for handling loan eligibility predictions"""

    # Business rules for validation
    MIN_INCOME = 0
    MAX_INCOME = 100000000  # 10 Crores
    MIN_LOAN_AMOUNT = 1000
    MAX_LOAN_AMOUNT = 50000000  # 5 Crores
    MIN_LOAN_TENURE = 12  # months
    MAX_LOAN_TENURE = 480  # 40 years
    MIN_CIBIL_SCORE = 300
    MAX_CIBIL_SCORE = 900

    def __init__(
        self,
        prediction_repo: PredictionRepository,
        model_loader: ModelLoader,
    ):
        """Initialize prediction service
        
        Args:
            prediction_repo: Repository for storing predictions
            model_loader: ML model loader
        """
        self.prediction_repo = prediction_repo
        self.model_loader = model_loader

    def predict_loan_eligibility(
        self, user_id: str, loan_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict loan eligibility for a user
        
        Args:
            user_id: User making the prediction
            loan_data: Loan application data
            
        Returns:
            Prediction result with confidence score
            
        Raises:
            ValidationError: If input validation fails
            ModelNotAvailableError: If ML model is not loaded
            PredictionError: If prediction fails
        """
        logger.info(f"Starting loan prediction for user: {user_id}")

        # Validate inputs
        self._validate_loan_data(loan_data)

        # Check model availability
        if not self.model_loader.is_model_loaded():
            logger.error("ML model not available for prediction")
            raise ModelNotAvailableError("Prediction service temporarily unavailable")

        try:
            # Engineer features
            features = self._engineer_features(loan_data)

            # Create feature dataframe
            feature_df = self._create_feature_dataframe(features)

            # Scale features
            scaled_features = self.model_loader.scale_features(feature_df)

            # Generate prediction
            prediction = self.model_loader.predict(scaled_features)
            probabilities = self.model_loader.predict_proba(scaled_features)

            # Format result
            result = self._format_prediction_result(prediction, probabilities)

            # Save prediction to database
            prediction_record = self._save_prediction(
                user_id=user_id,
                loan_data=loan_data,
                features=features,
                result=result,
            )

            logger.info(
                f"Prediction completed successfully for user: {user_id}, "
                f"Result: {result['decision']}, Confidence: {result['confidence']}"
            )

            return {
                "prediction_id": prediction_record.id,
                "decision": result["decision"],
                "confidence": result["confidence"],
                "risk_level": result["risk_level"],
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Prediction error for user {user_id}: {str(e)}")
            raise PredictionError(f"Failed to generate prediction: {str(e)}")

    def _validate_loan_data(self, data: Dict[str, Any]) -> None:
        """Validate loan application data
        
        Raises:
            ValidationError: If validation fails
        """
        errors = []

        # Required fields
        required_fields = [
            "first_name",
            "last_name",
            "applicant_income",
            "loan_amount",
            "loan_tenure",
        ]

        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"{field} is required")

        if errors:
            raise ValidationError("; ".join(errors))

        # Validate numeric ranges
        applicant_income = float(data.get("applicant_income", 0))
        coapplicant_income = float(data.get("coapplicant_income", 0))
        loan_amount = float(data.get("loan_amount", 0))
        loan_tenure = float(data.get("loan_tenure", 0))
        cibil_score = int(data.get("cibil_score", 300))

        if not (self.MIN_INCOME <= applicant_income <= self.MAX_INCOME):
            errors.append(
                f"Applicant income must be between {self.MIN_INCOME} and {self.MAX_INCOME}"
            )

        if coapplicant_income < self.MIN_INCOME:
            errors.append(f"Co-applicant income cannot be negative")

        if not (self.MIN_LOAN_AMOUNT <= loan_amount <= self.MAX_LOAN_AMOUNT):
            errors.append(
                f"Loan amount must be between {self.MIN_LOAN_AMOUNT} and {self.MAX_LOAN_AMOUNT}"
            )

        if not (self.MIN_LOAN_TENURE <= loan_tenure <= self.MAX_LOAN_TENURE):
            errors.append(
                f"Loan tenure must be between {self.MIN_LOAN_TENURE} and {self.MAX_LOAN_TENURE} months"
            )

        if not (self.MIN_CIBIL_SCORE <= cibil_score <= self.MAX_CIBIL_SCORE):
            errors.append(
                f"CIBIL score must be between {self.MIN_CIBIL_SCORE} and {self.MAX_CIBIL_SCORE}"
            )

        if errors:
            raise ValidationError("; ".join(errors))

    def _engineer_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer features from raw loan data
        
        Args:
            data: Raw loan application data
            
        Returns:
            Engineered features dictionary
        """
        applicant_income = float(data.get("applicant_income", 0))
        coapplicant_income = float(data.get("coapplicant_income", 0))
        loan_amount = float(data.get("loan_amount", 0))
        loan_tenure = float(data.get("loan_tenure", 1))  # Avoid division by zero

        # Calculate derived features
        total_income = applicant_income + coapplicant_income
        loan_to_income_ratio = loan_amount / total_income if total_income > 0 else 0
        emi = loan_amount / loan_tenure if loan_tenure > 0 else 0

        # Log transformations (add 1 to avoid log(0))
        applicant_income_log = np.log(applicant_income + 1)
        coapplicant_income_log = np.log(coapplicant_income + 1)
        loan_amount_log = np.log(loan_amount + 1)
        total_income_log = np.log(total_income + 1)

        return {
            # Original features
            "applicant_income": applicant_income,
            "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount,
            "loan_tenure": loan_tenure,
            "credit_history": float(data.get("credit_history", 0)),
            "cibil_score": int(data.get("cibil_score", 300)),
            # Demographic features
            "gender": int(data.get("gender", 0)),
            "marital_status": int(data.get("marital_status", 0)),
            "dependents": int(data.get("dependents", 0)),
            "education": int(data.get("education", 0)),
            "employment_type": int(data.get("employment_type", 0)),
            "property_area": int(data.get("property_area", 0)),
            # Engineered features
            "total_income": total_income,
            "loan_to_income_ratio": loan_to_income_ratio,
            "emi": emi,
            "applicant_income_log": applicant_income_log,
            "coapplicant_income_log": coapplicant_income_log,
            "loan_amount_log": loan_amount_log,
            "total_income_log": total_income_log,
        }

    def _create_feature_dataframe(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Create feature dataframe matching model training format
        
        Args:
            features: Engineered features
            
        Returns:
            DataFrame with proper feature encoding
        """
        # One-hot encode categorical variables
        dependents = features["dependents"]
        education = features["education"]
        gender = features["gender"]
        marital_status = features["marital_status"]
        employment_type = features["employment_type"]
        property_area = features["property_area"]

        return pd.DataFrame(
            {
                "ApplicantIncome": [features["applicant_income"]],
                "ApplicantIncome_log": [features["applicant_income_log"]],
                "CIBIL_Score": [features["cibil_score"]],
                "CoapplicantIncome": [features["coapplicant_income"]],
                "CoapplicantIncome_log": [features["coapplicant_income_log"]],
                "Credit_History": [features["credit_history"]],
                "Dependents_1": [1 if dependents == 1 else 0],
                "Dependents_2": [1 if dependents == 2 else 0],
                "Dependents_3+": [1 if dependents >= 3 else 0],
                "Education_Not Graduate": [1 if education == 1 else 0],
                "EMI_feature": [features["emi"]],
                "Gender_Male": [1 if gender == 1 else 0],
                "LoanAmount": [features["loan_amount"]],
                "LoanAmount_log": [features["loan_amount_log"]],
                "Loan_Amount_Term": [features["loan_tenure"]],
                "Loan_to_Income_Ratio": [features["loan_to_income_ratio"]],
                "Married_Yes": [1 if marital_status == 1 else 0],
                "Property_Area_Semiurban": [1 if property_area == 1 else 0],
                "Property_Area_Urban": [1 if property_area == 2 else 0],
                "Self_Employed_Yes": [1 if employment_type == 1 else 0],
                "Total_Income": [features["total_income"]],
                "Total_Income_log": [features["total_income_log"]],
            }
        )

    def _format_prediction_result(
        self, prediction: np.ndarray, probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """Format prediction result with confidence and risk level
        
        Args:
            prediction: Model prediction (0 or 1)
            probabilities: Prediction probabilities
            
        Returns:
            Formatted result dictionary
        """
        pred_class = int(prediction[0])
        confidence = float(probabilities[0][pred_class] * 100)
        decision = "APPROVED" if pred_class == 1 else "REJECTED"

        # Determine risk level based on confidence
        if confidence >= 90:
            risk_level = "LOW"
        elif confidence >= 70:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "decision": decision,
            "confidence": round(confidence, 2),
            "risk_level": risk_level,
        }

    def _save_prediction(
        self,
        user_id: str,
        loan_data: Dict[str, Any],
        features: Dict[str, Any],
        result: Dict[str, Any],
    ):
        """Save prediction record to database
        
        Args:
            user_id: User identifier
            loan_data: Original loan data
            features: Engineered features
            result: Prediction result
            
        Returns:
            Created prediction record
        """
        return self.prediction_repo.create(
            user_id=user_id,
            first_name=loan_data.get("first_name", ""),
            last_name=loan_data.get("last_name", ""),
            gender=features["gender"],
            marital_status=features["marital_status"],
            dependents_count=features["dependents"],
            education_level=features["education"],
            employment_type=features["employment_type"],
            property_location=features["property_area"],
            credit_history=features["credit_history"],
            cibil_score=features["cibil_score"],
            applicant_income=features["applicant_income"],
            coapplicant_income=features["coapplicant_income"],
            loan_amount=features["loan_amount"],
            loan_tenure=features["loan_tenure"],
            prediction_result=result["decision"],
            confidence_score=result["confidence"],
        )

    def get_user_prediction_history(
        self, user_id: str, limit: int = 10
    ) -> list[Dict[str, Any]]:
        """Get prediction history for a user
        
        Args:
            user_id: User identifier
            limit: Maximum number of records
            
        Returns:
            List of prediction records
        """
        predictions = self.prediction_repo.get_by_user_id(user_id, limit=limit)

        return [
            {
                "id": pred.id,
                "loan_amount": pred.loan_amount,
                "prediction": pred.prediction_result,
                "confidence": pred.confidence_score,
                "created_at": pred.created_at.isoformat() if pred.created_at else None,
            }
            for pred in predictions
        ]
