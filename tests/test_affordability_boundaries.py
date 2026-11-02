"""Configuration and numerical boundaries for the published affordability maths."""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext

import pytest

from loanapp.config import Config, ConfigError, load_config
from loanapp.features import monthly_instalment
from loanapp.rules import compute_affordability


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.5", "0", "1.5"])
def test_invalid_affordability_ceiling_is_rejected_at_startup(monkeypatch, value):
    monkeypatch.setenv("MAX_EMI_TO_INCOME", value)
    with pytest.raises(ConfigError, match="MAX_EMI_TO_INCOME"):
        load_config("testing")


@pytest.mark.parametrize("value", ["0.01", "0.5", "1"])
def test_positive_affordability_fraction_up_to_one_is_supported(monkeypatch, value):
    monkeypatch.setenv("MAX_EMI_TO_INCOME", value)
    assert load_config("testing").max_emi_to_income == float(value)


def _present_value_of_unit_payments(annual_rate: float, term_months: int) -> float:
    """Independent reference: discount each payment using high-precision decimals."""
    with localcontext() as context:
        context.prec = 400
        monthly_rate = Decimal(str(annual_rate)) / 12
        discount = Decimal(1) / (1 + monthly_rate)
        discounted_payment = Decimal(1)
        present_value = Decimal(0)
        for _ in range(term_months):
            discounted_payment *= discount
            present_value += discounted_payment
        return float(present_value)


@pytest.mark.parametrize("annual_rate", [0.0, 5e-324, 1e-320, 1e-15, 1e-8, 0.105, 0.999])
@pytest.mark.parametrize("term_months", [6, 240, 480])
def test_payment_and_affordable_principal_match_discounted_cash_flows(
    sample_application, annual_rate, term_months
):
    principal = 2_000_000
    app = sample_application.replace(loan_amount=principal, loan_term_months=term_months)
    config = Config(annual_interest_rate=annual_rate, max_emi_to_income=0.5)
    present_value = _present_value_of_unit_payments(annual_rate, term_months)

    payment = monthly_instalment(principal, annual_rate, term_months)
    affordability = compute_affordability(app, config)

    assert math.isfinite(payment)
    assert payment == pytest.approx(principal / present_value, rel=5e-13)
    assert math.isfinite(affordability.max_affordable_loan)
    affordable_payment = app.total_income * config.max_emi_to_income
    assert affordability.max_affordable_loan == pytest.approx(
        affordable_payment * present_value, rel=5e-13
    )
    assert monthly_instalment(
        affordability.max_affordable_loan, annual_rate, term_months
    ) == pytest.approx(affordable_payment, rel=5e-13)


def test_tiny_interest_rate_is_accepted_by_the_configuration(monkeypatch):
    monkeypatch.setenv("ANNUAL_INTEREST_RATE", "1e-15")
    assert load_config("testing").annual_interest_rate == 1e-15


def test_assessment_api_accepts_a_tiny_reference_rate(app, sample_application):
    app.config["APP_CONFIG"].annual_interest_rate = 1e-15
    response = app.test_client().post("/api/assess", json=sample_application.to_dict())
    assert response.status_code == 200
    result = response.get_json()["data"]
    assert result["affordability"]["monthly_instalment"] == pytest.approx(8333.33)
    json.dumps(result, allow_nan=False)
