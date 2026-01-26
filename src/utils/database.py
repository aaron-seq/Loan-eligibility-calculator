"""Database Utilities

Provides database connection management and utilities.
"""

import sqlite3
from contextlib import contextmanager
from typing import Generator

from src.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def get_db_connection(database_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections
    
    Ensures proper connection handling:
    - Automatic commit on success
    - Automatic rollback on error
    - Guaranteed connection closure
    
    Args:
        database_path: Path to SQLite database file
        
    Yields:
        Database connection with row_factory set
        
    Usage:
        with get_db_connection('path/to/db.sqlite') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
            # Commits automatically on success
        # Connection closed automatically
    """
    conn = None
    try:
        # Create connection
        conn = sqlite3.connect(database_path)

        # Enable row factory for dict-like access
        conn.row_factory = sqlite3.Row

        # Enable foreign key constraints (disabled by default in SQLite)
        conn.execute("PRAGMA foreign_keys = ON")

        yield conn

    except Exception as e:
        if conn:
            # Rollback on any error
            conn.rollback()
            logger.error(f"Database error, rolling back: {e}", exc_info=True)
        raise

    finally:
        if conn:
            # Close connection
            conn.close()


def initialize_database(database_path: str) -> None:
    """Initialize database with required tables
    
    Creates all necessary tables if they don't exist.
    Should be called during application startup.
    
    Args:
        database_path: Path to SQLite database file
    """
    logger.info(f"Initializing database: {database_path}")

    with get_db_connection(database_path) as conn:
        cursor = conn.cursor()

        # Create users table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                mobile_number TEXT,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """
        )

        # Create admins table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_id TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create predictions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS loan_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                gender INTEGER,
                marital_status INTEGER,
                dependents_count INTEGER,
                education_level INTEGER,
                employment_type INTEGER,
                property_location INTEGER,
                credit_history REAL,
                cibil_score INTEGER,
                applicant_income REAL,
                coapplicant_income REAL,
                loan_amount REAL,
                loan_tenure REAL,
                prediction_result TEXT,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """
        )

        # Create indexes for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON loan_predictions(user_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON loan_predictions(created_at)"
        )

        conn.commit()

    logger.info("Database initialized successfully")
