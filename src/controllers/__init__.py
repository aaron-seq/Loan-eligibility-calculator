"""Controllers Layer - HTTP Request Handlers

Controllers handle HTTP requests, validate inputs using schemas,
delegate to services, and format responses. NO business logic here.
"""

from src.controllers.auth_controller import auth_blueprint
from src.controllers.prediction_controller import prediction_blueprint
from src.controllers.admin_controller import admin_blueprint
from src.controllers.health_controller import health_blueprint

__all__ = [
    "auth_blueprint",
    "prediction_blueprint",
    "admin_blueprint",
    "health_blueprint",
]
