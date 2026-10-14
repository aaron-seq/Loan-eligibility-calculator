"""The feature contract — the thing that was silently broken before.

Training and serving used to build feature vectors independently, in different
column orders. These tests exist so that can never happen again without a red
test.
"""

from __future__ import annotations

import math

import pytest

from loanapp.features import (
    FEATURE_COLUMNS,
    LoanApplication,
    build_feature_frame,
    engineer,
    from_training_row,
    monthly_instalment,
)

RATE = 0.105


def test_feature_frame_matches_declared_column_order(sample_application):
    frame = build_feature_frame(sample_application, RATE)
    assert tuple(frame.columns) == FEATURE_COLUMNS


def test_engineer_produces_exactly_the_declared_features(sample_application):
    assert set(engineer(sample_application, RATE)) == set(FEATURE_COLUMNS)


def test_training_and_serving_paths_agree():
    """The regression test for the original bug.

    A row read from the training CSV and the equivalent application submitted
    through the API must produce byte-identical feature vectors. Previously
    these were two separate code paths and they disagreed.
    """
    row = {
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 130,  # thousands, as stored in the CSV
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Dependents": "2",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Urban",
    }
    from_csv = from_training_row(row)
    from_api = LoanApplication(
        applicant_income=5000,
        coapplicant_income=1500,
        loan_amount=130_000,  # same money, stated in currency units
        loan_term_months=360,
        credit_history=1,
        dependents=2,
        education="graduate",
        self_employed=False,
        property_area="urban",
    )
    assert engineer(from_csv, RATE) == engineer(from_api, RATE)


def test_training_row_converts_loan_amount_from_thousands():
    app = from_training_row({"LoanAmount": 146, "ApplicantIncome": 5000})
    assert app.loan_amount == 146_000


def test_missing_credit_history_is_treated_as_adverse():
    # Imputing "clean" for a missing bureau record would flatter applicants the
    # data does not support flattering.
    assert from_training_row({"Credit_History": None}).credit_history == 0


def test_dependents_3_plus_is_parsed():
    assert from_training_row({"Dependents": "3+"}).dependents == 3


@pytest.mark.parametrize("area", ["Urban", "RURAL", "Semiurban"])
def test_property_area_is_normalised(area):
    assert from_training_row({"Property_Area": area}).property_area == area.lower()


def test_unknown_property_area_falls_back_rather_than_raising():
    assert from_training_row({"Property_Area": "Atlantis"}).property_area == "urban"


class TestZeroAndEdgeInputs:
    """Ratios must degrade safely, and in the right direction."""

    def test_zero_income_gives_the_worst_ratios_not_the_best(self):
        """Regression: the guard used to substitute 0.0.

        0.0 is the *best* possible value for both affordability ratios, so a
        zero-income application was being told it could service the loan more
        comfortably than any real applicant. The substitute has to be the worst
        value, not the most convenient one.
        """
        from loanapp.features import RATIO_CEILING

        app = LoanApplication(0, 0, 500_000, 240, 1, 0, "graduate", False, "urban")
        features = engineer(app, RATE)
        assert features["loan_to_income_ratio"] == RATIO_CEILING
        assert features["emi_to_income_ratio"] == RATIO_CEILING
        # No income means no co-applicant share to speak of; 0.0 is correct
        # here because it is not an affordability signal.
        assert features["coapplicant_share"] == 0.0

    def test_zero_income_and_zero_loan_is_not_penalised(self):
        """Nothing requested, nothing to afford — the ceiling would be wrong."""
        app = LoanApplication(0, 0, 0, 240, 1, 0, "graduate", False, "urban")
        features = engineer(app, RATE)
        assert features["loan_to_income_ratio"] == 0.0
        assert features["emi_to_income_ratio"] == 0.0

    def test_ratios_are_capped_for_real_applications_too(self):
        """An absurd request must not produce an unbounded feature value."""
        from loanapp.features import RATIO_CEILING

        app = LoanApplication(1, 0, 10_000_000_000, 240, 1, 0, "graduate", False, "urban")
        features = engineer(app, RATE)
        assert features["loan_to_income_ratio"] == RATIO_CEILING
        assert features["emi_to_income_ratio"] == RATIO_CEILING

    def test_the_cap_does_not_touch_plausible_applications(self):
        """Measured: the largest loan-to-income ratio in train.csv is 82.7."""
        from loanapp.features import RATIO_CEILING

        app = LoanApplication(50_000, 0, 2_000_000, 240, 1, 0, "graduate", False, "urban")
        features = engineer(app, RATE)
        assert features["loan_to_income_ratio"] < RATIO_CEILING
        assert features["emi_to_income_ratio"] < RATIO_CEILING

    def test_zero_term_gives_zero_instalment(self):
        assert monthly_instalment(100_000, RATE, 0) == 0.0

    def test_zero_rate_is_straight_line_not_division_by_zero(self):
        assert monthly_instalment(120_000, 0.0, 12) == pytest.approx(10_000)


class TestInstalmentMaths:
    def test_matches_the_standard_amortisation_formula(self):
        # 1,000,000 over 240 months at 10.5%: verified against the closed form.
        emi = monthly_instalment(1_000_000, 0.105, 240)
        r = 0.105 / 12
        expected = 1_000_000 * r * (1 + r) ** 240 / ((1 + r) ** 240 - 1)
        assert emi == pytest.approx(expected)

    def test_longer_term_lowers_the_instalment(self):
        assert monthly_instalment(1_000_000, RATE, 360) < monthly_instalment(
            1_000_000, RATE, 120
        )

    def test_interest_makes_repayment_exceed_principal(self):
        assert monthly_instalment(1_000_000, RATE, 240) * 240 > 1_000_000


def test_replace_returns_an_independent_copy(sample_application):
    modified = sample_application.replace(loan_amount=1)
    assert modified.loan_amount == 1
    assert sample_application.loan_amount == 2_000_000
    assert modified.applicant_income == sample_application.applicant_income


def test_total_income_log_uses_log1p(sample_application):
    features = engineer(sample_application, RATE)
    assert features["total_income_log"] == pytest.approx(math.log1p(70_000))
