"""
Custom Exception Classes
Provides specific exception types for different error scenarios
Follows: NEVER silently fail - always log with full context
"""


class LoanPredictionBaseException(Exception):
    """Base exception for all application-specific exceptions"""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert exception to dictionary for JSON responses"""
        return {"error": self.__class__.__name__, "message": self.message, "details": self.details}


class ValidationError(LoanPredictionBaseException):
    """Raised when input validation fails"""

    pass


class AuthenticationError(LoanPredictionBaseException):
    """Raised when authentication fails"""

    pass


class AuthorizationError(LoanPredictionBaseException):
    """Raised when user lacks required permissions"""

    pass


class DatabaseError(LoanPredictionBaseException):
    """Raised when database operations fail"""

    pass


class ModelError(LoanPredictionBaseException):
    """Raised when ML model operations fail"""

    pass


class ConfigurationError(LoanPredictionBaseException):
    """Raised when configuration is invalid or missing"""

    pass


class PredictionServiceUnavailableError(LoanPredictionBaseException):
    """Raised when prediction service is unavailable"""

    pass


class RateLimitExceededError(LoanPredictionBaseException):
    """Raised when rate limit is exceeded"""

    pass
