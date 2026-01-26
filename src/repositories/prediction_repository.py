"""Prediction Repository - Prediction Data Access

Handles all loan prediction related database operations.
"""

from typing import Optional, List
from datetime import datetime

from src.repositories.base_repository import BaseRepository
from src.models.database import LoanPrediction
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionRepository(BaseRepository):
    """Repository for loan prediction data operations"""

    def create(
        self,
        user_id: str,
        first_name: str,
        last_name: str,
        gender: int,
        marital_status: int,
        dependents_count: int,
        education_level: int,
        employment_type: int,
        property_location: int,
        credit_history: float,
        cibil_score: int,
        applicant_income: float,
        coapplicant_income: float,
        loan_amount: float,
        loan_tenure: float,
        prediction_result: str,
        confidence_score: float,
    ) -> LoanPrediction:
        """Create new prediction record
        
        Returns:
            Created LoanPrediction object
        """
        query = """
            INSERT INTO loan_predictions (
                user_id, first_name, last_name, gender, marital_status,
                dependents_count, education_level, employment_type, property_location,
                credit_history, cibil_score, applicant_income, coapplicant_income,
                loan_amount, loan_tenure, prediction_result, confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            user_id,
            first_name,
            last_name,
            gender,
            marital_status,
            dependents_count,
            education_level,
            employment_type,
            property_location,
            credit_history,
            cibil_score,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_tenure,
            prediction_result,
            confidence_score,
        )

        last_id = self.execute_query(query, params, commit=True)
        logger.info(f"Prediction created with ID: {last_id}")

        return LoanPrediction(
            id=last_id,
            user_id=user_id,
            first_name=first_name,
            last_name=last_name,
            gender=gender,
            marital_status=marital_status,
            dependents_count=dependents_count,
            education_level=education_level,
            employment_type=employment_type,
            property_location=property_location,
            credit_history=credit_history,
            cibil_score=cibil_score,
            applicant_income=applicant_income,
            coapplicant_income=coapplicant_income,
            loan_amount=loan_amount,
            loan_tenure=loan_tenure,
            prediction_result=prediction_result,
            confidence_score=confidence_score,
            created_at=datetime.utcnow(),
        )

    def get_by_id(self, id: int) -> Optional[LoanPrediction]:
        """Get prediction by ID
        
        Args:
            id: Prediction ID
            
        Returns:
            LoanPrediction object or None
        """
        query = "SELECT * FROM loan_predictions WHERE id = ?"
        row = self.execute_query(query, (id,), fetch_one=True)

        return self._row_to_prediction(row) if row else None

    def get_by_user_id(
        self, user_id: str, limit: int = 10, offset: int = 0
    ) -> List[LoanPrediction]:
        """Get predictions by user ID
        
        Args:
            user_id: User identifier
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of LoanPrediction objects
        """
        query = """
            SELECT * FROM loan_predictions 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        """
        rows = self.execute_query(query, (user_id, limit, offset), fetch_all=True)

        return [self._row_to_prediction(row) for row in rows] if rows else []

    def get_all(self, limit: int = 100, offset: int = 0) -> List[LoanPrediction]:
        """Get all predictions with pagination
        
        Args:
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of LoanPrediction objects
        """
        query = """
            SELECT * FROM loan_predictions 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        """
        rows = self.execute_query(query, (limit, offset), fetch_all=True)

        return [self._row_to_prediction(row) for row in rows] if rows else []

    def count_by_user(self, user_id: str) -> int:
        """Count predictions for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of predictions
        """
        return self.count("loan_predictions", "user_id = ?", (user_id,))

    def _row_to_prediction(self, row) -> LoanPrediction:
        """Convert database row to LoanPrediction object
        
        Args:
            row: SQLite Row object
            
        Returns:
            LoanPrediction object
        """
        return LoanPrediction(
            id=row["id"],
            user_id=row["user_id"],
            first_name=row["first_name"],
            last_name=row["last_name"],
            gender=row["gender"],
            marital_status=row["marital_status"],
            dependents_count=row["dependents_count"],
            education_level=row["education_level"],
            employment_type=row["employment_type"],
            property_location=row["property_location"],
            credit_history=row["credit_history"],
            cibil_score=row["cibil_score"],
            applicant_income=row["applicant_income"],
            coapplicant_income=row["coapplicant_income"],
            loan_amount=row["loan_amount"],
            loan_tenure=row["loan_tenure"],
            prediction_result=row["prediction_result"],
            confidence_score=row["confidence_score"],
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else None,
        )
