"""Data Access Layer - Repository Pattern

Repositories handle all database interactions, abstracting persistence logic
from business layer. Enables easy testing with mocks and database swapping.
"""

from src.repositories.user_repository import UserRepository
from src.repositories.admin_repository import AdminRepository
from src.repositories.prediction_repository import PredictionRepository

__all__ = ["UserRepository", "AdminRepository", "PredictionRepository"]
