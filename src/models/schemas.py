"""
Request/Response Schemas with Pydantic Validation
Provides strict input validation and automatic documentation
"""

import re
from typing import Optional
from pydantic import BaseModel, Field, validator, EmailStr

from src.config import get_settings

settings = get_settings()


class UserRegistrationRequest(BaseModel):
    """User registration request schema"""

    user_id: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: EmailStr = Field(..., description="Valid email address")
    mobile_number: Optional[str] = Field(
        None, max_length=15, description="Mobile number (optional)"
    )
    full_name: str = Field(..., min_length=2, max_length=100, description="Full name")
    password: str = Field(..., min_length=8, max_length=128, description="Password")
    confirm_password: str = Field(..., description="Password confirmation")

    @validator("user_id")
    def validate_user_id(cls, v):
        """Validate user ID format - alphanumeric and underscore only"""
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError(
                "User ID can only contain letters, numbers, and underscores"
            )
        return v.lower()

    @validator("mobile_number")
    def validate_mobile_number(cls, v):
        """Validate mobile number format"""
        if v and not re.match(r"^\+?[1-9]\d{1,14}$", v):
            raise ValueError("Invalid mobile number format")
        return v

    @validator("password")
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        return v

    @validator("confirm_password")
    def passwords_match(cls, v, values):
        """Ensure password confirmation matches"""
        if "password" in values and v != values["password"]:
            raise ValueError("Passwords do not match")
        return v

    class Config:
        schema_extra = {
            "example": {
                "user_id": "john_doe",
                "email": "john.doe@example.com",
                "mobile_number": "+919876543210",
                "full_name": "John Doe",
                "password": "SecurePass123",
                "confirm_password": "SecurePass123",
            }
        }


class UserLoginRequest(BaseModel):
    """User login request schema"""

    user_id: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)

    class Config:
        schema_extra = {
            "example": {"user_id": "john_doe", "password": "SecurePass123"}
        }


class AdminLoginRequest(BaseModel):
    """Admin login request schema"""

    admin_id: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)

    class Config:
        schema_extra = {"example": {"admin_id": "admin", "password": "admin123"}}


class LoanPredictionRequest(BaseModel):
    """Loan prediction request schema with business rule validation"""

    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    married: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    dependents: int = Field(..., ge=0, le=5, description="Number of dependents (0-5)")
    education: int = Field(
        ..., ge=0, le=1, description="0=Graduate, 1=Not Graduate"
    )
    self_employed: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    property_area: int = Field(
        ..., ge=0, le=2, description="0=Rural, 1=Semiurban, 2=Urban"
    )
    credit_history: float = Field(
        ..., ge=0.0, le=1.0, description="0=Bad, 1=Good credit history"
    )
    cibil_score: int = Field(
        ...,
        ge=settings.min_cibil_score,
        le=settings.max_cibil_score,
        description=f"CIBIL score ({settings.min_cibil_score}-{settings.max_cibil_score})",
    )
    applicant_income: float = Field(
        ...,
        gt=0,
        le=settings.max_income,
        description="Applicant's monthly income",
    )
    coapplicant_income: float = Field(
        ..., ge=0, le=settings.max_income, description="Co-applicant's monthly income"
    )
    loan_amount: float = Field(
        ...,
        gt=0,
        ge=settings.min_loan_amount,
        le=settings.max_loan_amount,
        description=f"Loan amount ({settings.min_loan_amount}-{settings.max_loan_amount})",
    )
    loan_amount_term: float = Field(
        ..., gt=0, le=480, description="Loan tenure in months (max 40 years)"
    )

    @validator("first_name", "last_name")
    def validate_name(cls, v):
        """Validate name contains only letters and spaces"""
        if not re.match(r"^[a-zA-Z\s]+$", v):
            raise ValueError("Name can only contain letters and spaces")
        return v.strip().title()

    @validator("loan_amount")
    def validate_loan_amount(cls, v, values):
        """Validate loan amount against income (business rule)"""
        if "applicant_income" in values and "coapplicant_income" in values:
            total_income = values["applicant_income"] + values["coapplicant_income"]
            # Loan amount should not exceed 50x total monthly income
            if v > total_income * 50:
                raise ValueError(
                    "Loan amount exceeds 50 times total monthly income"
                )
        return v

    class Config:
        schema_extra = {
            "example": {
                "first_name": "John",
                "last_name": "Doe",
                "gender": 1,
                "married": 1,
                "dependents": 1,
                "education": 0,
                "self_employed": 0,
                "property_area": 2,
                "credit_history": 1.0,
                "cibil_score": 750,
                "applicant_income": 50000.0,
                "coapplicant_income": 30000.0,
                "loan_amount": 2000000.0,
                "loan_amount_term": 360.0,
            }
        }


class PredictionResponse(BaseModel):
    """Loan prediction response schema"""

    success: bool
    prediction: str = Field(..., description="APPROVED or REJECTED")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Confidence percentage")
    message: str
    prediction_id: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "prediction": "APPROVED",
                "confidence": 87.5,
                "message": "Loan approved with 87.5% confidence",
                "prediction_id": 123,
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response schema"""

    status: str
    timestamp: str
    model_available: bool
    database_available: bool
    version: str
