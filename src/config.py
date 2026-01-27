"""
Configuration Management with Environment Variable Validation
Follows 12-Factor App methodology for configuration
"""

import os
import secrets
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings with validation"""

    # Application Configuration
    app_name: str = Field(default="Loan Eligibility Prediction System", env="APP_NAME")
    flask_env: str = Field(default="production", env="FLASK_ENV")
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    port: int = Field(default=5000, env="PORT")
    host: str = Field(default="0.0.0.0", env="HOST")

    # Security Configuration
    # WARNING: NEVER commit real SECRET_KEY to version control
    secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32), env="SECRET_KEY"
    )
    session_cookie_secure: bool = Field(default=True, env="SESSION_COOKIE_SECURE")
    session_cookie_httponly: bool = Field(
        default=True, env="SESSION_COOKIE_HTTPONLY"
    )
    session_cookie_samesite: str = Field(default="Lax", env="SESSION_COOKIE_SAMESITE")
    permanent_session_lifetime: int = Field(
        default=3600, env="PERMANENT_SESSION_LIFETIME"
    )  # 1 hour

    # Database Configuration
    database_path: str = Field(
        default=str(BASE_DIR / "data" / "loan_prediction.db"), env="DATABASE_PATH"
    )
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")

    # Model Configuration
    model_path: str = Field(default=str(BASE_DIR / "models"), env="MODEL_PATH")
    model_name: str = Field(
        default="loan_eligibility_model.pkl", env="MODEL_NAME"
    )
    scaler_name: str = Field(default="feature_scaler.pkl", env="SCALER_NAME")
    feature_columns_name: str = Field(
        default="feature_columns.pkl", env="FEATURE_COLUMNS_NAME"
    )

    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(
        default=str(BASE_DIR / "logs" / "app.log"), env="LOG_FILE"
    )
    log_max_bytes: int = Field(default=10485760, env="LOG_MAX_BYTES")  # 10MB
    log_backup_count: int = Field(default=10, env="LOG_BACKUP_COUNT")

    # CORS Configuration
    cors_origins: list = Field(
        default=["http://localhost:3000", "http://localhost:5000"], env="CORS_ORIGINS"
    )

    # Rate Limiting Configuration
    ratelimit_enabled: bool = Field(default=True, env="RATELIMIT_ENABLED")
    ratelimit_default: str = Field(default="100 per hour", env="RATELIMIT_DEFAULT")
    ratelimit_storage_url: Optional[str] = Field(default=None, env="RATELIMIT_STORAGE_URL")

    # Admin Configuration
    default_admin_username: str = Field(default="admin", env="DEFAULT_ADMIN_USERNAME")
    default_admin_password: str = Field(
        default="ChangeMe123!", env="DEFAULT_ADMIN_PASSWORD"
    )

    # Model Prediction Thresholds
    min_income: float = Field(default=1000.0, env="MIN_INCOME")
    max_income: float = Field(default=100000000.0, env="MAX_INCOME")
    min_loan_amount: float = Field(default=1000.0, env="MIN_LOAN_AMOUNT")
    max_loan_amount: float = Field(default=50000000.0, env="MAX_LOAN_AMOUNT")
    min_cibil_score: int = Field(default=300, env="MIN_CIBIL_SCORE")
    max_cibil_score: int = Field(default=900, env="MAX_CIBIL_SCORE")

    @validator("flask_env")
    def validate_flask_env(cls, v):
        """Validate flask environment setting"""
        allowed = ["development", "testing", "production"]
        if v not in allowed:
            raise ValueError(f"flask_env must be one of {allowed}")
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level setting"""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.flask_env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.flask_env == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.flask_env == "testing" or self.testing

    class Config:
        """Pydantic configuration"""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance"""
    return settings
