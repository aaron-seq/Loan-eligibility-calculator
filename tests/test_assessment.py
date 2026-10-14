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


class TestOutOfRange:
    """The calibrated model cannot return a probability below ~0.20, because
    Platt scaling extrapolates to its asymptote outside the training range.
    Quoting that floor as a real chance would be dishonest."""

    def test_the_floor_exists_and_is_well_above_zero(self, assessor):
        """Documents the behaviour this whole class exists to handle."""
        worst = LoanApplication(
            0, 0, 50_000_000, 480, 0, 10, "not_graduate", True, "rural"
        )
        assert assessor.score(worst) > 0.1  # not ~0, as a naive reading expects

    def test_zero_income_is_flagged_out_of_range(self, assessor):
        result = assessor.assess(app(applicant_income=0, coapplicant_income=0))
        assert result.out_of_range_reason is not None
        assert "income" in result.out_of_range_reason

    def test_absurd_loan_is_flagged_out_of_range(self, assessor):
        result = assessor.assess(app(applicant_income=1, loan_amount=1_000_000_000))
        assert result.out_of_range_reason is not None

    def test_normal_application_is_not_flagged(self, assessor):
        assert assessor.assess(app()).out_of_range_reason is None

    def test_no_guidance_is_offered_when_the_number_is_meaningless(self, assessor):
        """ "Borrow 10% less" is not useful advice to someone with no income."""
        result = assessor.assess(app(applicant_income=0, coapplicant_income=0))
        assert result.suggestions == []

    def test_the_flag_is_serialised_for_api_clients(self, assessor):
        payload = assessor.assess(app(applicant_income=0, coapplicant_income=0)).to_dict()
        assert payload["out_of_range"] is True
        assert payload["out_of_range_reason"]

    def test_rules_still_state_the_real_reason(self, assessor):
        """Suppressing the number is only acceptable because something else
        answers the question."""
        result = assessor.assess(app(applicant_income=0, coapplicant_income=0))
        assert any(
            r.code == "income_present" and r.severity == "fail" for r in result.rules
        )


class TestFeatureRateContract:
    """The interest rate has two jobs, and they must not be confused.

    1. It computes ``emi_to_income_ratio``, which is a **model feature**. That
       must use the rate the model was *trained* with, or the feature means
       something different at serve time than it did at training time.
    2. It drives the affordability figures shown to the applicant. That is a
       business setting and should follow ``ANNUAL_INTEREST_RATE``.

    Before this was separated, changing ``ANNUAL_INTEREST_RATE`` silently
    skewed every prediction.
    """

    def test_changing_the_configured_rate_does_not_move_the_model(self, assessor):
        from dataclasses import replace

        from loanapp.assessment import Assessor

        baseline = assessor.score(app())
        skewed = Assessor(
            assessor.bundle, replace(assessor.config, annual_interest_rate=0.25)
        )
        assert skewed.score(app()) == pytest.approx(baseline)

    def test_features_use_the_rate_recorded_in_the_bundle(self, assessor):
        assert assessor.feature_rate == assessor.bundle.metadata["annual_interest_rate"]

    def test_affordability_still_follows_the_configured_rate(self, assessor):
        """The applicant-facing figures are the thing the setting is *for*."""
        from dataclasses import replace

        from loanapp.assessment import Assessor

        high = Assessor(
            assessor.bundle, replace(assessor.config, annual_interest_rate=0.25)
        )
        low = Assessor(
            assessor.bundle, replace(assessor.config, annual_interest_rate=0.05)
        )
        assert (
            high.assess(app()).affordability.monthly_instalment
            > low.assess(app()).affordability.monthly_instalment
        )

    def test_bundle_without_a_recorded_rate_falls_back_to_config(self, assessor):
        """Bundles predating the metadata must still load and score."""
        from dataclasses import replace

        from loanapp.assessment import Assessor

        legacy = replace(
            assessor.bundle,
            metadata={
                k: v
                for k, v in assessor.bundle.metadata.items()
                if k != "annual_interest_rate"
            },
        )
        assert (
            Assessor(legacy, assessor.config).feature_rate
            == assessor.config.annual_interest_rate
        )
