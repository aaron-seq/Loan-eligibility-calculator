import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

class SecurityService:
    """Security utilities for password policy and admin bootstrap."""

    @staticmethod
    def validate_password_strength(password: str) -> Optional[str]:
        if len(password) < 12:
            return 'Password must be at least 12 characters long'
        if not re.search(r'[A-Z]', password):
            return 'Password must contain at least one uppercase letter'
        if not re.search(r'[a-z]', password):
            return 'Password must contain at least one lowercase letter'
        if not re.search(r'\d', password):
            return 'Password must contain at least one digit'
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return 'Password must contain at least one special character'
        return None

    @staticmethod
    def ensure_admin_bootstrap(config, db_manager, generate_password_hash):
        """Create admin user securely from environment variables if not exists."""
        admin_id = config.ADMIN_USERNAME
        admin_password = config.ADMIN_PASSWORD
        
        if not admin_id or not admin_password:
            logger.warning('Admin bootstrap skipped: ADMIN_USERNAME/ADMIN_PASSWORD not set')
            return False
        
        pwd_error = SecurityService.validate_password_strength(admin_password)
        if pwd_error:
            raise ValueError(f'ADMIN_PASSWORD does not meet policy: {pwd_error}')
        
        with db_manager.get_connection() as conn:
            cur = conn.cursor()
            # Check existence
            if db_manager.config.is_postgres:
                cur.execute('SELECT COUNT(*) AS cnt FROM admins WHERE admin_id = %s', (admin_id,))
                row = cur.fetchone()
                exists = (row['cnt'] if isinstance(row, dict) and 'cnt' in row else list(row)[0]) > 0
            else:
                cur.execute('SELECT COUNT(*) FROM admins WHERE admin_id = ?', (admin_id,))
                exists = cur.fetchone()[0] > 0
            
            if not exists:
                pwd_hash = generate_password_hash(admin_password)
                if db_manager.config.is_postgres:
                    cur.execute('INSERT INTO admins (admin_id, password_hash) VALUES (%s, %s)', (admin_id, pwd_hash))
                else:
                    cur.execute('INSERT INTO admins (admin_id, password_hash) VALUES (?, ?)', (admin_id, pwd_hash))
                conn.commit()
                logger.info('Admin user bootstrapped securely')
                return True
        
        return False
