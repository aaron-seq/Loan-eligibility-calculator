"""Prediction Controller - HTTP Layer

Handles prediction HTTP requests and delegates to PredictionService.
"""

from flask import Blueprint, request, session, jsonify
from pydantic import ValidationError as PydanticValidationError

from src.models.schemas import LoanPredictionRequest
from src.services.prediction_service import PredictionService
from src.utils.logger import get_logger
from src.utils.exceptions import (
    ValidationError,
    PredictionError,
    ModelNotAvailableError,
)
from src.utils.decorators import require_login, rate_limit, require_json

logger = get_logger(__name__)
prediction_blueprint = Blueprint("prediction", __name__, url_prefix="/api/predict")

# Service will be injected during app initialization
prediction_service: PredictionService = None


def init_prediction_controller(service: PredictionService):
    """Initialize controller with service dependency"""
    global prediction_service
    prediction_service = service


@prediction_blueprint.route("/", methods=["POST"])
@require_login
@rate_limit(max_requests=20, window_seconds=60)
@require_json
def predict():
    """Loan eligibility prediction endpoint
    
    Request Body:
        - first_name: Applicant first name
        - last_name: Applicant last name
        - gender: Gender (0=Female, 1=Male)
        - marital_status: Marital status (0=No, 1=Yes)
        - dependents: Number of dependents
        - education: Education (0=Graduate, 1=Not Graduate)
        - employment_type: Employment (0=No, 1=Yes)
        - applicant_income: Applicant income
        - coapplicant_income: Co-applicant income
        - loan_amount: Loan amount requested
        - loan_tenure: Loan tenure in months
        - credit_history: Credit history (0=No, 1=Yes)
        - property_area: Property area (0=Rural, 1=Semiurban, 2=Urban)
        - cibil_score: CIBIL score (300-900)
        
    Returns:
        200: Prediction successful
        400: Validation error
        401: Not authenticated
        503: Model unavailable
        500: Server error
    """
    try:
        user_id = session.get("user_id")

        # Validate request
        data = LoanPredictionRequest(**request.json)

        # Generate prediction
        result = prediction_service.predict_loan_eligibility(
            user_id=user_id,
            loan_data=data.dict(),
        )

        logger.info(f"Prediction successful for user: {user_id}")

        return jsonify(
            {
                "success": True,
                "message": f"Loan {result['decision'].lower()}",
                "data": result,
            }
        ), 200

    except PydanticValidationError as e:
        return jsonify({"success": False, "message": "Validation error", "errors": e.errors()}), 400

    except ValidationError as e:
        logger.warning(f"Prediction validation error: {e}")
        return jsonify({"success": False, "message": str(e)}), 400

    except ModelNotAvailableError as e:
        logger.error(f"Model unavailable: {e}")
        return jsonify({"success": False, "message": str(e)}), 503

    except PredictionError as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Prediction failed"}), 500


@prediction_blueprint.route("/history", methods=["GET"])
@require_login
def get_history():
    """Get user's prediction history
    
    Query Params:
        - limit: Number of records (default: 10)
        - offset: Pagination offset (default: 0)
        
    Returns:
        200: History retrieved
        401: Not authenticated
        500: Server error
    """
    try:
        user_id = session.get("user_id")
        limit = min(int(request.args.get("limit", 10)), 100)

        history = prediction_service.get_user_prediction_history(
            user_id=user_id,
            limit=limit,
        )

        return jsonify(
            {
                "success": True,
                "data": {"predictions": history, "count": len(history)},
            }
        ), 200

    except Exception as e:
        logger.error(f"History retrieval error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Failed to retrieve history"}), 500
