"""User Repository - User Data Access

Handles all user-related database operations.
"""

from typing import Optional
from datetime import datetime

from src.repositories.base_repository import BaseRepository
from src.models.database import User
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UserRepository(BaseRepository):
    """Repository for user data operations"""

    def create(
        self,
        user_id: str,
        email: str,
        full_name: str,
        password_hash: str,
        mobile_number: Optional[str] = None,
    ) -> User:
        """Create new user record
        
        Args:
            user_id: Unique user identifier
            email: User email
            full_name: User's full name
            password_hash: Hashed password
            mobile_number: Optional mobile number
            
        Returns:
            Created User object
        """
        query = """
            INSERT INTO users (user_id, email, mobile_number, full_name, password_hash)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (user_id, email, mobile_number, full_name, password_hash)

        last_id = self.execute_query(query, params, commit=True)
        logger.info(f"User created with ID: {last_id}")

        return User(
            id=last_id,
            user_id=user_id,
            email=email,
            mobile_number=mobile_number,
            full_name=full_name,
            password_hash=password_hash,
            created_at=datetime.utcnow(),
            is_active=True,
        )

    def get_by_id(self, id: int) -> Optional[User]:
        """Get user by database ID
        
        Args:
            id: Database record ID
            
        Returns:
            User object or None
        """
        query = "SELECT * FROM users WHERE id = ?"
        row = self.execute_query(query, (id,), fetch_one=True)

        return self._row_to_user(row) if row else None

    def get_by_user_id(self, user_id: str) -> Optional[User]:
        """Get user by user_id
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            User object or None
        """
        query = "SELECT * FROM users WHERE user_id = ?"
        row = self.execute_query(query, (user_id,), fetch_one=True)

        return self._row_to_user(row) if row else None

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email
        
        Args:
            email: User email address
            
        Returns:
            User object or None
        """
        query = "SELECT * FROM users WHERE email = ?"
        row = self.execute_query(query, (email,), fetch_one=True)

        return self._row_to_user(row) if row else None

    def update_password(self, user_id: str, password_hash: str) -> bool:
        """Update user password
        
        Args:
            user_id: User identifier
            password_hash: New password hash
            
        Returns:
            True if successful
        """
        query = "UPDATE users SET password_hash = ? WHERE user_id = ?"
        self.execute_query(query, (password_hash, user_id), commit=True)
        logger.info(f"Password updated for user: {user_id}")
        return True

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user account
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful
        """
        query = "UPDATE users SET is_active = 0 WHERE user_id = ?"
        self.execute_query(query, (user_id,), commit=True)
        logger.info(f"User deactivated: {user_id}")
        return True

    def _row_to_user(self, row) -> User:
        """Convert database row to User object
        
        Args:
            row: SQLite Row object
            
        Returns:
            User object
        """
        return User(
            id=row["id"],
            user_id=row["user_id"],
            email=row["email"],
            mobile_number=row["mobile_number"],
            full_name=row["full_name"],
            password_hash=row["password_hash"],
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else None,
            is_active=bool(row["is_active"]),
        )
