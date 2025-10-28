import logging
from typing import Dict, Any, Optional
from db import db_manager

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Aggregated analytics for administrative dashboard."""

    @staticmethod
    def get_summary(date_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return db_manager.get_analytics_summary(date_range)
