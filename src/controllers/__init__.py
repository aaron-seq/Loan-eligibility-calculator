"""Controllers - HTTP Request/Response Handling

Controllers handle HTTP concerns ONLY:
- Request parsing and validation
- Response formatting
- HTTP status codes
- Session management

NO BUSINESS LOGIC in controllers. Delegate to services.
"""

from src.controllers.auth_controller import auth_bp
from src.controllers.prediction_controller import prediction_bp
from src.controllers.admin_controller import admin_bp

__all__ = ["auth_bp", "prediction_bp", "admin_bp"]
