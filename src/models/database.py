"""
Database Models
Defines the database schema using dataclasses for simplicity
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class User:
    """User model for customer accounts"""

    id: Optional[int] = None
    user_id: str = ""
    email: str = ""
    mobile_number: str = ""
    full_name: str = ""
    password_hash: str = ""
    created_at: Optional[datetime] = None
    is_active: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding sensitive fields"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "email": self.email,
            "mobile_number": self.mobile_number,
            "full_name": self.full_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active,
        }


@dataclass
class Admin:
    """Admin model for administrative accounts"""

    id: Optional[int] = None
    admin_id: str = ""
    password_hash: str = ""
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding sensitive fields"""
        return {
            "id": self.id,
            "admin_id": self.admin_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class LoanPrediction:
    """Loan prediction record model"""

    id: Optional[int] = None
    user_id: str = ""
    first_name: str = ""
    last_name: str = ""
    gender: int = 0
    marital_status: int = 0
    dependents_count: int = 0
    education_level: int = 0
    employment_type: int = 0
    property_location: int = 0
    credit_history: float = 0.0
    cibil_score: int = 300
    applicant_income: float = 0.0
    coapplicant_income: float = 0.0
    loan_amount: float = 0.0
    loan_tenure: float = 0.0
    prediction_result: str = ""
    confidence_score: float = 0.0
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "gender": gender,
            "marital_status": self.marital_status,
            "dependents_count": self.dependents_count,
            "education_level": self.education_level,
            "employment_type": self.employment_type,
            "property_location": self.property_location,
            "credit_history": self.credit_history,
            "cibil_score": self.cibil_score,
            "applicant_income": self.applicant_income,
            "coapplicant_income": self.coapplicant_income,
            "loan_amount": self.loan_amount,
            "loan_tenure": self.loan_tenure,
            "prediction_result": self.prediction_result,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
