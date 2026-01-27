"""Admin Service - Handles administrative operations

This service manages admin authentication and reporting.
Separated to maintain clear boundaries and security.
"""

from typing import Dict, Any, List, Optional
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

from src.exceptions import (
    InvalidCredentialsError,
    AdminNotFoundError,
)
from src.logger import get_logger
from src.models.schemas import AdminLoginRequest

logger = get_logger(__name__)


class AdminService:
    """Service for administrative operations"""

    def __init__(self, db_connection_factory):
        """Initialize admin service
        
        Args:
            db_connection_factory: Context manager for database connections
        """
        self.db_connection_factory = db_connection_factory

    def authenticate_admin(self, request: AdminLoginRequest) -> Dict[str, Any]:
        """Authenticate admin credentials
        
        Business Rules:
        - Admin accounts have elevated privileges
        - Failed login attempts are logged for security
        - Admin sessions should have shorter timeouts
        
        Args:
            request: Validated admin login request
            
        Returns:
            Dict containing admin information
            
        Raises:
            InvalidCredentialsError: If credentials are invalid
        """
        logger.info(f"Admin authentication attempt: {request.admin_id}")

        try:
            with self.db_connection_factory() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT admin_id, password_hash FROM admins WHERE admin_id = ?",
                    (request.admin_id,),
                )
                admin = cursor.fetchone()

            if not admin:
                logger.warning(f"Admin authentication failed - not found: {request.admin_id}")
                raise InvalidCredentialsError("Invalid admin credentials")

            if not check_password_hash(admin["password_hash"], request.password):
                logger.warning(
                    f"Admin authentication failed - invalid password: {request.admin_id}"
                )
                raise InvalidCredentialsError("Invalid admin credentials")

            logger.info(f"Admin authenticated successfully: {request.admin_id}")
            return {"admin_id": admin["admin_id"]}

        except InvalidCredentialsError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during admin authentication: {e}", exc_info=True)
            raise

    def get_all_predictions(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieve all loan predictions with user details
        
        Business Rule: Admins can view all predictions for monitoring and reporting.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip (for pagination)
            
        Returns:
            List of prediction records with user information
        """
        logger.info(f"Retrieving predictions (limit={limit}, offset={offset})")

        try:
            with self.db_connection_factory() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 
                        lp.*,
                        u.full_name,
                        u.email
                    FROM loan_predictions lp
                    JOIN users u ON lp.user_id = u.user_id
                    ORDER BY lp.created_at DESC
                    LIMIT ? OFFSET ?
                """,
                    (limit, offset),
                )
                predictions = cursor.fetchall()

            return [dict(row) for row in predictions]

        except Exception as e:
            logger.error(f"Error retrieving predictions: {e}", exc_info=True)
            raise

    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics on predictions
        
        Returns statistics useful for business reporting:
        - Total predictions
        - Approval rate
        - Average confidence score
        - Predictions by time period
        
        Returns:
            Dict containing various statistics
        """
        logger.info("Generating prediction statistics")

        try:
            with self.db_connection_factory() as conn:
                cursor = conn.cursor()

                # Total predictions
                cursor.execute("SELECT COUNT(*) as total FROM loan_predictions")
                total = cursor.fetchone()["total"]

                # Approval rate
                cursor.execute(
                    """
                    SELECT 
                        prediction_result,
                        COUNT(*) as count,
                        AVG(confidence_score) as avg_confidence
                    FROM loan_predictions
                    GROUP BY prediction_result
                """
                )
                by_result = cursor.fetchall()

                # Recent predictions (last 7 days)
                cursor.execute(
                    """
                    SELECT COUNT(*) as recent_count
                    FROM loan_predictions
                    WHERE created_at >= datetime('now', '-7 days')
                """
                )
                recent = cursor.fetchone()["recent_count"]

            stats = {
                "total_predictions": total,
                "predictions_by_result": [dict(row) for row in by_result],
                "recent_predictions_7d": recent,
            }

            # Calculate approval rate
            approved = sum(
                row["count"]
                for row in by_result
                if row["prediction_result"] == "APPROVED"
            )
            stats["approval_rate"] = (
                round((approved / total) * 100, 2) if total > 0 else 0
            )

            logger.info("Statistics generated successfully")
            return stats

        except Exception as e:
            logger.error(f"Error generating statistics: {e}", exc_info=True)
            raise

    def get_user_list(
        self, limit: int = 100, offset: int = 0, active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get list of users
        
        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip
            active_only: If True, only return active users
            
        Returns:
            List of user records (excluding sensitive data)
        """
        logger.info(f"Retrieving user list (active_only={active_only})")

        try:
            with self.db_connection_factory() as conn:
                cursor = conn.cursor()

                query = """
                    SELECT 
                        user_id,
                        email,
                        full_name,
                        created_at,
                        is_active
                    FROM users
                """

                if active_only:
                    query += " WHERE is_active = 1"

                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"

                cursor.execute(query, (limit, offset))
                users = cursor.fetchall()

            return [dict(row) for row in users]

        except Exception as e:
            logger.error(f"Error retrieving user list: {e}", exc_info=True)
            raise
