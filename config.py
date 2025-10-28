import os
from datetime import timedelta

class Config:
    """Environment-driven configuration for production-ready deployment."""
    
    # Core Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    
    # Database configuration
    DATABASE_URL = os.environ.get('DATABASE_URL')
    DATABASE_PATH = os.environ.get('DATABASE_PATH', 'loan_prediction.db')
    
    # ML Model settings
    MODEL_PATH = 'models'
    REQUIRE_MODELS_IN_PROD = os.environ.get('REQUIRE_MODELS_IN_PROD', 'false').lower() == 'true'
    
    # GraphQL settings
    ENABLE_GRAPHIQL = os.environ.get('ENABLE_GRAPHIQL', 'true' if os.environ.get('FLASK_ENV', 'development') == 'development' else 'false').lower() == 'true'
    
    # Admin security settings
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')
    
    # Session security
    SESSION_COOKIE_SECURE = os.environ.get('FLASK_ENV') == 'production'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)
    
    @classmethod
    def validate_production_config(cls):
        """Validate required settings for production deployment."""
        if cls.FLASK_ENV == 'production':
            missing = []
            if not cls.SECRET_KEY:
                missing.append('SECRET_KEY')
            if not cls.ADMIN_USERNAME:
                missing.append('ADMIN_USERNAME')
            if not cls.ADMIN_PASSWORD:
                missing.append('ADMIN_PASSWORD')
            if cls.REQUIRE_MODELS_IN_PROD and not os.path.exists(cls.MODEL_PATH):
                missing.append(f'MODEL_PATH directory: {cls.MODEL_PATH}')
            
            if missing:
                raise ValueError(f"Missing required production configuration: {', '.join(missing)}")
    
    @property
    def is_postgres(self):
        """Check if using PostgreSQL (Railway) vs SQLite (local dev)."""
        return self.DATABASE_URL and self.DATABASE_URL.startswith('postgres')
    
    @property
    def is_development(self):
        """Check if running in development mode."""
        return self.FLASK_ENV == 'development'
