import os
import sqlite3
import logging
from contextlib import contextmanager
from config import Config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database abstraction supporting SQLite (dev) and PostgreSQL (Railway prod)."""
    
    def __init__(self):
        self.config = Config()
        self._connection_string = self._get_connection_string()
        
    def _get_connection_string(self):
        """Get appropriate connection string based on environment."""
        if self.config.is_postgres:
            return self.config.DATABASE_URL
        else:
            return self.config.DATABASE_PATH
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup."""
        if self.config.is_postgres:
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor
                conn = psycopg2.connect(self.config.DATABASE_URL, cursor_factory=RealDictCursor)
                conn.autocommit = False
                yield conn
            except ImportError:
                raise ImportError("psycopg2 required for PostgreSQL support. Install with: pip install psycopg2-binary")
            finally:
                if 'conn' in locals():
                    conn.close()
        else:
            conn = sqlite3.connect(self._connection_string)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def initialize_schema(self):
        """Initialize database schema for both SQLite and PostgreSQL."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.config.is_postgres:
                # PostgreSQL schema
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(255) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        mobile_number VARCHAR(20),
                        full_name VARCHAR(255) NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS admins (
                        id SERIAL PRIMARY KEY,
                        admin_id VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS loan_predictions (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        first_name VARCHAR(255) NOT NULL,
                        last_name VARCHAR(255) NOT NULL,
                        gender INTEGER,
                        marital_status INTEGER,
                        dependents_count INTEGER,
                        education_level INTEGER,
                        employment_type INTEGER,
                        property_location INTEGER,
                        credit_history DECIMAL,
                        cibil_score INTEGER,
                        applicant_income DECIMAL,
                        coapplicant_income DECIMAL,
                        loan_amount DECIMAL,
                        loan_tenure DECIMAL,
                        prediction_result VARCHAR(50),
                        confidence_score DECIMAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON loan_predictions(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON loan_predictions(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_result ON loan_predictions(prediction_result)')
                
            else:
                # SQLite schema (existing)
                cursor.execute('''
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
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS admins (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        admin_id TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
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
                ''')
            
            conn.commit()
            logger.info(f"Database schema initialized successfully ({'PostgreSQL' if self.config.is_postgres else 'SQLite'})")
    
    def get_analytics_summary(self, date_range=None):
        """Get analytics summary for admin dashboard."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Base query for analytics
            where_clause = ""
            params = []
            
            if date_range:
                if self.config.is_postgres:
                    where_clause = "WHERE created_at >= %s AND created_at <= %s"
                else:
                    where_clause = "WHERE created_at >= ? AND created_at <= ?"
                params = [date_range['start'], date_range['end']]
            
            # Get total predictions and approval rate
            query = f"""
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN prediction_result = 'APPROVED' THEN 1 ELSE 0 END) as approved_count,
                    AVG(confidence_score) as avg_confidence,
                    AVG(applicant_income + coapplicant_income) as avg_total_income,
                    AVG(loan_amount) as avg_loan_amount
                FROM loan_predictions
                {where_clause}
            """
            
            cursor.execute(query, params)
            summary = dict(cursor.fetchone())
            
            # Calculate approval rate
            if summary['total_predictions'] > 0:
                summary['approval_rate'] = (summary['approved_count'] / summary['total_predictions']) * 100
            else:
                summary['approval_rate'] = 0
            
            return summary

# Global database manager instance
db_manager = DatabaseManager()
