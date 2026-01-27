"""Admin Repository - Admin Data Access

Handles all admin-related database operations.
"""

from typing import Optional
from datetime import datetime

from src.repositories.base_repository import BaseRepository
from src.models.database import Admin
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdminRepository(BaseRepository):
    """Repository for admin data operations"""

    def create(self, admin_id: str, password_hash: str) -> Admin:
        """Create new admin record
        
        Args:
            admin_id: Unique admin identifier
            password_hash: Hashed password
            
        Returns:
            Created Admin object
        """
        query = """
            INSERT INTO admins (admin_id, password_hash)
            VALUES (?, ?)
        """
        params = (admin_id, password_hash)

        last_id = self.execute_query(query, params, commit=True)
        logger.info(f"Admin created with ID: {last_id}")

        return Admin(
            id=last_id,
            admin_id=admin_id,
            password_hash=password_hash,
            created_at=datetime.utcnow(),
        )

    def get_by_id(self, id: int) -> Optional[Admin]:
        """Get admin by database ID
        
        Args:
            id: Database record ID
            
        Returns:
            Admin object or None
        """
        query = "SELECT * FROM admins WHERE id = ?"
        row = self.execute_query(query, (id,), fetch_one=True)

        return self._row_to_admin(row) if row else None

    def get_by_admin_id(self, admin_id: str) -> Optional[Admin]:
        """Get admin by admin_id
        
        Args:
            admin_id: Unique admin identifier
            
        Returns:
            Admin object or None
        """
        query = "SELECT * FROM admins WHERE admin_id = ?"
        row = self.execute_query(query, (admin_id,), fetch_one=True)

        return self._row_to_admin(row) if row else None

    def _row_to_admin(self, row) -> Admin:
        """Convert database row to Admin object
        
        Args:
            row: SQLite Row object
            
        Returns:
            Admin object
        """
        return Admin(
            id=row["id"],
            admin_id=row["admin_id"],
            password_hash=row["password_hash"],
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else None,
        )
