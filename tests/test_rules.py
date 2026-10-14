"""Affordability and credit rules.

These are published formulas, so the tests assert the actual arithmetic rather
than just that a rule fired.
"""

from __future__ import annotations

import pytest

from loanapp.config import Config
from loanapp.features import LoanApplication, monthly_instalment
from loanapp.rules import compute_affordability, evaluate_rules, rules_verdict


@pytest.fixture
def config():
    return Config(env="testing", annual_interest_rate=0.105, max_emi_to_income=0.50)


def application(**overrides) -> LoanApplication:
    defaults = {
        "applicant_income": 50_000,
        "coapplicant_income": 0,
        "loan_amount": 1_000_000,
        "loan_term_months": 240,
        "credit_history": 1,
        "dependents": 0,
        "education": "graduate",
        "self_employed": False,
        "property_area": "urban",
        "cibil_score": 760,
    }
    return LoanApplication(**{**defaults, **overrides})


def code(results, name):
    return next(r for r in results if r.code == name)


class TestAffordability:
    def test_instalment_matches_the_formula(self, config):
        app = application()
        result = compute_affordability(app, config)
        assert result.monthly_instalment == pytest.approx(
            monthly_instalment(1_000_000, 0.105, 240)
        )

    def test_emi_to_income_is_a_fraction_of_household_income(self, config):
        app = application(applicant_income=40_000, coapplicant_income=10_000)
        result = compute_affordability(app, config)
        assert result.emi_to_income == pytest.approx(result.monthly_instalment / 50_000)

    def test_max_affordable_loan_inverts_the_emi_formula(self, config):
        """The largest principal whose instalment lands exactly on the ceiling."""
        app = application()
        result = compute_affordability(app, config)
        implied = monthly_instalment(result.max_affordable_loan, 0.105, 240)
        assert implied == pytest.approx(50_000 * 0.50, rel=1e-6)

    def test_total_interest_is_never_negative(self, config):
        result = compute_affordability(application(loan_term_months=6), config)
        assert result.total_interest >= 0

    def test_zero_income_does_not_raise(self, config):
        result = compute_affordability(application(applicant_income=0), config)
        assert result.emi_to_income == 0.0


class TestRules:
    def test_affordable_loan_passes(self, config):
        app = application(applicant_income=200_000, loan_amount=1_000_000)
        results = evaluate_rules(app, compute_affordability(app, config), config)
        assert code(results, "emi_to_income").severity == "pass"

    def test_unaffordable_loan_fails(self, config):
        app = application(applicant_income=20_000, loan_amount=5_000_000)
        results = evaluate_rules(app, compute_affordability(app, config), config)
        assert code(results, "emi_to_income").severity == "fail"

    def test_near_the_ceiling_warns_rather_than_failing(self, config):
        # Target 45% of income: inside the 50% ceiling, past the 40% warn line.
        income = 50_000
        target_emi = income * 0.45
        r = 0.105 / 12
        growth = (1 + r) ** 240
        principal = target_emi * (growth - 1) / (r * growth)
        app = application(applicant_income=income, loan_amount=principal)
        results = evaluate_rules(app, compute_affordability(app, config), config)
        assert code(results, "emi_to_income").severity == "warn"

    def test_zero_income_reports_the_income_rule_not_a_ratio(self, config):
        app = application(applicant_income=0, coapplicant_income=0)
        results = evaluate_rules(app, compute_affordability(app, config), config)
        assert code(results, "income_present").severity == "fail"

    def test_adverse_credit_history_fails(self, config):
        app = application(credit_history=0)
        results = evaluate_rules(app, compute_affordability(app, config), config)
        assert code(results, "credit_history").severity == "fail"

    @pytest.mark.parametrize(
        "score,expected", [(800, "pass"), (700, "warn"), (550, "fail")]
    )
    def test_cibil_bands(self, config, score, expected):
        app = application(cibil_score=score)
        results = evaluate_rules(app, compute_affordability(app, config), config)
        assert code(results, "cibil_score").severity == expected

    def test_omitted_cibil_score_produces_no_rule(self, config):
        app = application(cibil_score=None)
        results = evaluate_rules(app, compute_affordability(app, config), config)
        assert all(r.code != "cibil_score" for r in results)

    def test_term_beyond_the_training_range_warns_about_extrapolation(self, config):
        app = application(loan_term_months=420)
        results = evaluate_rules(app, compute_affordability(app, config), config)
        assert code(results, "loan_term").severity == "warn"

    def test_rule_details_quote_real_numbers(self, config):
        """Guidance that says "too high" without saying how high is not guidance."""
        app = application(applicant_income=20_000, loan_amount=5_000_000)
        results = evaluate_rules(app, compute_affordability(app, config), config)
        detail = code(results, "emi_to_income").detail
        assert "%" in detail and any(ch.isdigit() for ch in detail)


class TestVerdict:
    def test_worst_severity_wins(self, config):
        app = application(applicant_income=20_000, loan_amount=5_000_000, cibil_score=800)
        assert (
            rules_verdict(evaluate_rules(app, compute_affordability(app, config), config))
            == "fail"
        )

    def test_all_clear_passes(self, config):
        app = application(
            applicant_income=300_000, loan_amount=1_000_000, cibil_score=800
        )
        assert (
            rules_verdict(evaluate_rules(app, compute_affordability(app, config), config))
            == "pass"
        )
