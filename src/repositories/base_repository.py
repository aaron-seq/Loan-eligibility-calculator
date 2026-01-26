"""Base Repository - Common Database Operations

Provides common CRUD operations and database connection management.
All specific repositories inherit from this base.
"""

import sqlite3
from contextlib import contextmanager
from typing import Optional, List, Any, Tuple
from abc import ABC, abstractmethod

from src.utils.logger import get_logger
from src.utils.exceptions import DatabaseError

logger = get_logger(__name__)


class BaseRepository(ABC):
    """Abstract base repository for database operations"""

    def __init__(self, db_path: str):
        """Initialize repository with database path
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

    @contextmanager
    def get_connection(self):
        """Context manager for database connections
        
        Yields:
            Database connection with row factory
            
        Raises:
            DatabaseError: If connection fails
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise DatabaseError(f"Database connection failed: {str(e)}")
        finally:
            if conn:
                conn.close()

    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
        commit: bool = False,
    ) -> Optional[Any]:
        """Execute database query with proper error handling
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch_one: Return single result
            fetch_all: Return all results
            commit: Commit transaction
            
        Returns:
            Query result or None
            
        Raises:
            DatabaseError: If query execution fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or ())

                if commit:
                    conn.commit()
                    return cursor.lastrowid

                if fetch_one:
                    return cursor.fetchone()

                if fetch_all:
                    return cursor.fetchall()

        except sqlite3.IntegrityError as e:
            logger.error(f"Integrity error in query: {e}")
            raise DatabaseError(f"Database integrity error: {str(e)}")
        except sqlite3.Error as e:
            logger.error(f"Database error in query: {e}")
            raise DatabaseError(f"Database error: {str(e)}")

    def execute_many(
        self, query: str, params_list: List[Tuple], commit: bool = True
    ) -> None:
        """Execute multiple queries in a transaction
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            commit: Commit transaction
            
        Raises:
            DatabaseError: If execution fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                if commit:
                    conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error in executemany: {e}")
            raise DatabaseError(f"Batch operation failed: {str(e)}")

    @abstractmethod
    def create(self, **kwargs) -> Any:
        """Create new record - must be implemented by subclasses"""
        pass

    @abstractmethod
    def get_by_id(self, id: Any) -> Optional[Any]:
        """Get record by ID - must be implemented by subclasses"""
        pass

    def count(self, table: str, where_clause: str = "", params: Tuple = ()) -> int:
        """Count records in table
        
        Args:
            table: Table name
            where_clause: Optional WHERE clause
            params: Query parameters
            
        Returns:
            Number of records
        """
        query = f"SELECT COUNT(*) FROM {table}"
        if where_clause:
            query += f" WHERE {where_clause}"

        result = self.execute_query(query, params, fetch_one=True)
        return result[0] if result else 0
