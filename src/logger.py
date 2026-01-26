"""
Logging Infrastructure with Rotation and Structured Logging
Follows: Comprehensive logging with correlation IDs and proper rotation
"""

import logging
import sys
import uuid
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
from functools import wraps
from flask import request, g
import traceback

from src.config import get_settings

settings = get_settings()


class ContextFilter(logging.Filter):
    """Add correlation ID to log records for request tracing"""

    def filter(self, record):
        # Add correlation ID from Flask g object or generate new one
        record.correlation_id = getattr(g, "correlation_id", "NO_REQUEST")
        record.user_id = getattr(g, "user_id", "ANONYMOUS")
        return True


def setup_logging(app_name: str = "loan_prediction") -> logging.Logger:
    """
    Configure application logging with rotation and multiple handlers
    
    Implements:
    - Rotating file handler to prevent disk overflow
    - Console handler for development
    - Structured format with correlation IDs
    - Different log levels for different environments
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, settings.log_level))

    # Remove existing handlers to avoid duplication
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - [%(correlation_id)s] - [%(user_id)s] - "
        "%(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add context filter
    context_filter = ContextFilter()

    # Rotating File Handler (Production)
    # Prevents disk overflow by rotating log files
    if not settings.is_testing:
        file_handler = RotatingFileHandler(
            settings.log_file,
            maxBytes=settings.log_max_bytes,  # 10MB default
            backupCount=settings.log_backup_count,  # Keep 10 backup files
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        file_handler.addFilter(context_filter)
        logger.addHandler(file_handler)

    # Console Handler (Development)
    if settings.is_development or settings.is_testing:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # Error File Handler (All Environments)
    # Separate file for errors to make troubleshooting easier
    if not settings.is_testing:
        error_file = Path(settings.log_file).parent / "error.log"
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=settings.log_max_bytes,
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        error_handler.addFilter(context_filter)
        logger.addHandler(error_handler)

    logger.info(f"Logging initialized for {app_name} - Level: {settings.log_level}")

    return logger


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with parameters and exceptions
    Useful for debugging and audit trails
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(
                f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
            )
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(
                    f"{func.__name__} raised {e.__class__.__name__}: {str(e)}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                raise

        return wrapper

    return decorator


def add_correlation_id():
    """
    Flask before_request handler to add correlation ID to each request
    Enables tracing requests across logs
    """
    g.correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    g.user_id = getattr(g, "user_id", "ANONYMOUS")


# Global logger instance
app_logger = setup_logging()
