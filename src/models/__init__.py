"""Data Models for Loan Prediction System"""

from src.models.database import User, Admin, LoanPrediction
from src.models.schemas import (
    UserRegistrationRequest,
    UserLoginRequest,
    AdminLoginRequest,
    LoanPredictionRequest,
    PredictionResponse,
)

__all__ = [
    "User",
    "Admin",
    "LoanPrediction",
    "UserRegistrationRequest",
    "UserLoginRequest",
    "AdminLoginRequest",
    "LoanPredictionRequest",
    "PredictionResponse",
]
