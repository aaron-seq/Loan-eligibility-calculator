"""Assessment, explanation and what-if guidance.

The properties asserted here are the ones a user would notice if they broke:
the estimate moves in the right direction, the explanation is coherent, and the
advice is actually achievable.
"""

from __future__ import annotations

import pytest

from loanapp.explain import suggest_improvements
from loanapp.features import LoanApplication

pytestmark = pytest.mark.requires_model


def app(**overrides) -> LoanApplication:
    defaults = {
        "applicant_income": 50_000,
        "coapplicant_income": 0,
        "loan_amount": 2_000_000,
        "loan_term_months": 240,
        "credit_history": 1,
        "dependents": 0,
        "education": "graduate",
        "self_employed": False,
        "property_area": "urban",
        "cibil_score": 740,
    }
    return LoanApplication(**{**defaults, **overrides})


class TestScoring:
    def test_probability_is_a_probability(self, assessor):
        assert 0.0 <= assessor.score(app()) <= 1.0

    def test_scoring_is_deterministic(self, assessor):
        assert assessor.score(app()) == assessor.score(app())

    def test_clean_credit_history_scores_at_least_as_well(self, assessor):
        assert assessor.score(app(credit_history=1)) >= assessor.score(
            app(credit_history=0)
        )

    def test_higher_income_scores_at_least_as_well(self, assessor):
        assert assessor.score(app(applicant_income=200_000)) >= assessor.score(
            app(applicant_income=20_000)
        )

    @pytest.mark.parametrize("factor", [0.9, 0.7, 0.5, 0.3, 0.1, 0.02])
    def test_borrowing_less_never_scores_worse(self, assessor, factor):
        """The property that makes the guidance feature honest.

        A model can have a fine ROC-AUC and still be non-monotonic here — an
        early random forest scored a 70%-smaller loan below a 50%-smaller one.
        Training screens for this; this test stops a regression reaching users.
        """
        base = app()
        smaller = base.replace(loan_amount=base.loan_amount * factor)
        assert assessor.score(smaller) >= assessor.score(base) - 0.005


class TestBands:
    def test_bands_follow_the_configured_thresholds(self, assessor):
        config = assessor.config
        assert assessor.band_for(config.approve_threshold + 0.01) == "likely"
        assert assessor.band_for(config.borderline_threshold + 0.01) == "borderline"
        assert assessor.band_for(config.borderline_threshold - 0.01) == "unlikely"


class TestAssessment:
    def test_returns_every_section(self, assessor):
        result = assessor.assess(app()).to_dict()
        for key in (
            "probability",
            "band",
            "affordability",
            "rules",
            "factors",
            "model",
            "disclaimer",
            "assessed_at",
        ):
            assert key in result

    def test_reports_the_model_version_it_used(self, assessor):
        assert assessor.assess(app()).model_version == assessor.bundle.version

    def test_disclaimer_is_always_present(self, assessor):
        assert (
            "not a loan offer" in assessor.assess(app()).to_dict()["disclaimer"].lower()
        )

    def test_no_guidance_offered_to_someone_already_likely(self, assessor):
        strong = app(applicant_income=400_000, loan_amount=500_000)
        result = assessor.assess(strong)
        if result.band == "likely":
            assert result.suggestions == []

    def test_compare_scores_every_scenario(self, assessor):
        results = assessor.compare([app(), app(loan_amount=500_000)])
        assert len(results) == 2
        assert results[0]["index"] == 0
        assert all(0 <= r["probability"] <= 1 for r in results)


class TestExplanations:
    def test_factors_are_applicant_visible_inputs(self, assessor):
        """Not engineered features.

        Attributing to engineered features produced self-contradictory output —
        "your loan amount helps you" alongside "your loan-to-income ratio hurts
        you" from the same application.
        """
        from loanapp.features import EXPLAINABLE_INPUTS

        for factor in assessor.assess(app()).factors:
            assert factor.feature in EXPLAINABLE_INPUTS

    def test_direction_matches_the_sign_of_the_impact(self, assessor):
        for factor in assessor.assess(app()).factors:
            assert factor.direction == ("helps" if factor.impact > 0 else "hurts")

    def test_factors_are_ranked_by_absolute_impact(self, assessor):
        impacts = [abs(f.impact) for f in assessor.assess(app()).factors]
        assert impacts == sorted(impacts, reverse=True)

    def test_adverse_credit_history_is_attributed_as_harmful(self, assessor):
        factors = assessor.assess(app(credit_history=0)).factors
        credit = [f for f in factors if f.feature == "credit_history"]
        if credit:  # only surfaces when its impact clears the noise floor
            assert credit[0].direction == "hurts"

    def test_a_large_loan_is_never_reported_as_helping(self, assessor):
        """The regression test for the collinearity artifact.

        With a raw ``log(loan_amount)`` in the feature set, an oversized
        request was attributed as *helping* the applicant.
        """
        oversized = app(applicant_income=120_000, loan_amount=20_000_000)
        for factor in assessor.assess(oversized).factors:
            if factor.feature == "loan_amount":
                assert factor.direction == "hurts"

    def test_every_factor_has_a_readable_explanation(self, assessor):
        for factor in assessor.assess(app()).factors:
            assert len(factor.detail) > 20
            assert factor.label != factor.feature  # a label, not a column name


class TestGuidance:
    def test_weak_application_gets_actionable_suggestions(self, assessor):
        weak = app(applicant_income=25_000, loan_amount=4_000_000, credit_history=0)
        assert assessor.assess(weak).suggestions

    def test_every_suggestion_actually_raises_the_estimate(self, assessor):
        weak = app(applicant_income=25_000, loan_amount=4_000_000, credit_history=0)
        result = assessor.assess(weak)
        for suggestion in result.suggestions:
            assert suggestion.delta > 0
            assert suggestion.projected_probability > result.probability

    def test_suggestions_are_ranked_by_effect(self, assessor):
        weak = app(applicant_income=25_000, loan_amount=4_000_000, credit_history=0)
        deltas = [s.delta for s in assessor.assess(weak).suggestions]
        assert deltas == sorted(deltas, reverse=True)

    def test_only_one_borrow_less_option_is_offered(self, assessor):
        weak = app(applicant_income=25_000, loan_amount=4_000_000)
        borrow = [
            s for s in assessor.assess(weak).suggestions if s.action.startswith("Borrow")
        ]
        assert len(borrow) <= 1

    def test_prefers_the_smallest_sufficient_reduction(self, assessor):
        """Advising a 30% cut when 10% clears the band is worse advice, even
        though its delta is larger."""
        borderline = app(applicant_income=60_000, loan_amount=2_400_000)
        result = assessor.assess(borderline)
        borrow = [s for s in result.suggestions if s.action.startswith("Borrow")]
        if (
            borrow
            and borrow[0].projected_probability >= assessor.config.approve_threshold
        ):
            assert borrow[0].action == "Borrow 10% less"

    def test_no_co_applicant_suggestion_when_one_already_exists(self, assessor):
        weak = app(
            applicant_income=25_000, coapplicant_income=15_000, loan_amount=4_000_000
        )
        actions = [s.action for s in assessor.assess(weak).suggestions]
        assert "Add a co-applicant" not in actions

    def test_no_credit_suggestion_when_history_is_already_clean(self, assessor):
        weak = app(applicant_income=25_000, loan_amount=4_000_000, credit_history=1)
        actions = [s.action for s in assessor.assess(weak).suggestions]
        assert "Clear the adverse credit record" not in actions

    def test_no_term_extension_beyond_the_maximum(self, assessor):
        maxed = app(loan_term_months=360, applicant_income=25_000, loan_amount=4_000_000)
        actions = [s.action for s in assessor.assess(maxed).suggestions]
        assert not any(a.startswith("Extend the term") for a in actions)

    def test_suggestions_are_dropped_when_they_do_not_help(self, assessor):
        """A scorer that ignores its input must yield no advice, rather than
        advice with a zero effect."""
        strong = app()
        suggestions = suggest_improvements(
            app=strong,
            base_probability=0.5,
            score_application=lambda _: 0.5,
            config=assessor.config,
        )
        assert suggestions == []
