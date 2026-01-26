"""Prediction Controller - Handles loan prediction HTTP requests

HTTP concerns only. Business logic in PredictionService.
"""

from flask import Blueprint, request, jsonify, session
from pydantic import ValidationError as PydanticValidationError

from src.models.schemas import LoanPredictionRequest
from src.services.prediction_service import PredictionService
from src.exceptions import (
    ModelNotAvailableError,
    PredictionError,
    ValidationError,
)
from src.logger import get_logger
from src.utils.decorators import (
    rate_limit,
    add_correlation_id,
    require_authentication,
)
from src.utils.response import success_response, error_response

logger = get_logger(__name__)

prediction_bp = Blueprint("prediction", __name__, url_prefix="/api")

# Service injection
prediction_service: PredictionService = None


def init_prediction_controller(service: PredictionService):
    """Initialize controller with service dependency"""
    global prediction_service
    prediction_service = service


@prediction_bp.route("/predict", methods=["POST"])
@add_correlation_id
@require_authentication
@rate_limit(max_requests=20, window_seconds=3600)  # 20 predictions per hour
def predict():
    """Handle loan prediction request
    
    HTTP concerns:
    - Parse request data
    - Validate schema
    - Format response
    """
    try:
        # Get user from session
        user_id = session.get("user_id")

        # Parse request data
        data = request.get_json() if request.is_json else request.form.to_dict()

        # Convert string numbers to appropriate types
        for field in [
            "gender",
            "married",
            "dependents",
            "education",
            "self_employed",
            "property_area",
            "credit_history",
            "cibil_score",
        ]:
            if field in data and isinstance(data[field], str):
                data[field] = int(data[field])

        for field in [
            "applicant_income",
            "coapplicant_income",
            "loan_amount",
            "loan_amount_term",
        ]:
            if field in data and isinstance(data[field], str):
                data[field] = float(data[field])

        # Validate using Pydantic schema
        prediction_request = LoanPredictionRequest(**data)

        # Delegate to service
        result = prediction_service.predict_loan_eligibility(user_id, prediction_request)

        # Format response
        return success_response(
            data=result.dict(),
            message=result.message,
        )

    except PydanticValidationError as e:
        logger.warning(f"Prediction validation error: {e}")
        return error_response(
            message="Invalid prediction request data",
            errors=e.errors(),
            status_code=400,
        )

    except ValidationError as e:
        return error_response(
            message=str(e),
            status_code=400,
        )

    except ModelNotAvailableError as e:
        return error_response(
            message=str(e),
            status_code=503,
        )

    except PredictionError as e:
        return error_response(
            message=str(e),
            status_code=500,
        )

    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}", exc_info=True)
        return error_response(
            message="Prediction failed. Please try again.",
            status_code=500,
        )


@prediction_bp.route("/predictions/history", methods=["GET"])
@add_correlation_id
@require_authentication
def get_history():
    """Get user's prediction history
    
    HTTP concerns:
    - Parse query parameters
    - Format response
    """
    try:
        user_id = session.get("user_id")
        limit = int(request.args.get("limit", 10))

        # Validate limit
        if limit < 1 or limit > 100:
            return error_response(
                message="Limit must be between 1 and 100",
                status_code=400,
            )

        # Delegate to service
        predictions = prediction_service.get_user_predictions(user_id, limit)

        return success_response(
            data={"predictions": predictions, "count": len(predictions)},
            message="Prediction history retrieved successfully",
        )

    except Exception as e:
        logger.error(f"Error retrieving prediction history: {e}", exc_info=True)
        return error_response(
            message="Failed to retrieve prediction history",
            status_code=500,
        )


@prediction_bp.route("/health", methods=["GET"])
@add_correlation_id
def health_check():
    """Health check endpoint for monitoring
    
    Returns service health status including model availability.
    """
    try:
        from datetime import datetime

        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "model_available": (
                prediction_service.is_available() if prediction_service else False
            ),
        }

        status_code = 200 if health_status["model_available"] else 503

        return jsonify(health_status), status_code

    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
