import strawberry
from typing import Optional
from strawberry.types import Info
from services.analytics_service import AnalyticsService
from services.prediction_service import prediction_service

@strawberry.type
class Summary:
    total_predictions: int
    approved_count: int
    approval_rate: float
    avg_confidence: float
    avg_total_income: float
    avg_loan_amount: float

@strawberry.type
class PredictionResult:
    decision: str
    confidence: float

@strawberry.input
class PredictInput:
    first_name: str
    last_name: str
    gender: int
    married: int
    dependents: int
    education: int
    self_employed: int
    property_area: int
    credit_history: float
    cibil_score: int
    applicant_income: float
    coapplicant_income: float
    loan_amount: float
    loan_amount_term: float

@strawberry.type
class Query:
    @strawberry.field
    def analytics_summary(self, info: Info, start: Optional[str] = None, end: Optional[str] = None) -> Summary:
        date_range = None
        if start and end:
            date_range = {'start': start, 'end': end}
        summary = AnalyticsService.get_summary(date_range)
        return Summary(
            total_predictions=int(summary.get('total_predictions', 0) or 0),
            approved_count=int(summary.get('approved_count', 0) or 0),
            approval_rate=float(summary.get('approval_rate', 0) or 0.0),
            avg_confidence=float(summary.get('avg_confidence', 0) or 0.0),
            avg_total_income=float(summary.get('avg_total_income', 0) or 0.0),
            avg_loan_amount=float(summary.get('avg_loan_amount', 0) or 0.0)
        )

@strawberry.type
class Mutation:
    @strawberry.mutation
    def predict_loan(self, info: Info, input: PredictInput) -> PredictionResult:
        data = {
            'first_name': input.first_name,
            'last_name': input.last_name,
            'gender': input.gender,
            'marital_status': input.married,
            'dependents_count': input.dependents,
            'education_level': input.education,
            'employment_type': input.self_employed,
            'property_location': input.property_area,
            'credit_history': input.credit_history,
            'cibil_score': input.cibil_score,
            'applicant_income': input.applicant_income,
            'coapplicant_income': input.coapplicant_income,
            'loan_amount': input.loan_amount,
            'loan_tenure': input.loan_amount_term
        }
        result = prediction_service.predict(data)
        return PredictionResult(decision=result['decision'], confidence=result['confidence'])

schema = strawberry.Schema(query=Query, mutation=Mutation)
