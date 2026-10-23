"""Assembles the full eligibility assessment.

The assessment deliberately reports three things side by side rather than
collapsing them into one number:

1. **Model estimate** — a calibrated probability learned from 614 historical
   applications, with its accuracy and limits attached.
2. **Rules** — deterministic affordability and credit checks with published
   thresholds (:mod:`loanapp.rules`).
3. **Explanation and guidance** — why the estimate landed where it did and what
   would move it (:mod:`loanapp.explain`).

Keeping them separate is the point. A single "APPROVED / REJECTED" verdict from
a 0.83-AUC model trained on a few hundred rows would imply an authority this
system does not have.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .config import Config
from .explain import Factor, Suggestion, explain, suggest_improvements
from .features import (
    FEATURE_COLUMNS,
    RATIO_CEILING,
    LoanApplication,
    build_feature_frame,
    engineer,
)
from .ml.registry import ModelBundle
from .rules import (
    Affordability,
    RuleResult,
    compute_affordability,
    evaluate_rules,
    rules_verdict,
)

DISCLAIMER = (
    "This is an educational estimate produced by a model trained on a small "
    "public dataset. It is not a loan offer, not a credit decision, and no "
    "lender is bound by it. Lenders assess documents, verified income and a "
    "full bureau report that this tool never sees."
)

#: Applicant-facing labels for supported bands and the unavailable state. Phrased as likelihood
#: rather than as a verdict, because that is what the model actually produces.
BAND_LABELS = {
    "unavailable": "Estimate unavailable",
    "likely": "Likely to be approved",
    "borderline": "Borderline",
    "unlikely": "Unlikely to be approved",
}

OUT_OF_RANGE_NOTE = (
    "This application is outside the range the model was trained on, so no "
    "meaningful probability can be given for it. The checks below are the "
    "real answer."
)


def _extrapolation_reason(app: LoanApplication, features: dict[str, float]) -> str | None:
    """Why this application sits outside the model's usable range, if it does.

    This exists because the calibrated model has a **hard probability floor of
    about 0.20**: Platt scaling is fitted on the training range, and anything
    far outside it extrapolates to the sigmoid's asymptote rather than to zero.
    So the worst conceivable application — no income, a 50,000,000 request,
    adverse credit — was still shown "20%".

    For a tool whose premise is an honest number, quoting a floor that cannot
    be escaped is worse than quoting nothing. When this returns a reason the
    caller suppresses the probability and leads with the deterministic rules,
    which can state the actual problem.
    """
    if app.total_income <= 0:
        return "no household income was declared"
    if features["loan_to_income_ratio"] >= RATIO_CEILING:
        return "the loan is implausibly large relative to income"
    if features["emi_to_income_ratio"] >= RATIO_CEILING:
        return "the monthly instalment is implausibly large relative to income"
    return None


@dataclass
class Assessment:
    """The complete result returned to callers."""

    probability: float
    band: str
    band_label: str
    affordability: Affordability
    rules: list[RuleResult]
    rules_verdict: str
    factors: list[Factor]
    suggestions: list[Suggestion]
    model_version: str
    model_calibrated: bool
    model_metrics: dict[str, Any] = field(default_factory=dict)
    #: Rows the model was fitted on. Distinct from metrics["holdout_rows"],
    #: which is the held-back test split; conflating them understated the
    #: training set by a factor of five on the result page.
    model_training_rows: int = 0
    #: Set when the application sits outside the model's usable range. The
    #: probability is then not meaningful and must not be presented as one.
    out_of_range_reason: str | None = None
    assessed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": None
            if self.out_of_range_reason
            else round(self.probability, 4),
            "band": self.band,
            "band_label": self.band_label,
            "affordability": self.affordability.to_dict(),
            "rules": [
                {
                    "code": r.code,
                    "severity": r.severity,
                    "title": r.title,
                    "detail": r.detail,
                }
                for r in self.rules
            ],
            "rules_verdict": self.rules_verdict,
            "factors": [f.to_dict() for f in self.factors],
            "suggestions": [s.to_dict() for s in self.suggestions],
            "model": {
                "version": self.model_version,
                "calibrated": self.model_calibrated,
                "metrics": self.model_metrics,
                "training_rows": self.model_training_rows,
            },
            "assessed_at": self.assessed_at,
            "out_of_range": self.out_of_range_reason is not None,
            "out_of_range_reason": self.out_of_range_reason,
            "disclaimer": DISCLAIMER,
        }


class Assessor:
    """Scores applications against a loaded model bundle.

    Constructed once per process and shared; it holds no per-request state.
    """

    def __init__(self, bundle: ModelBundle, config: Config):
        self.bundle = bundle
        self.config = config
        #: The rate used to compute ``emi_to_income_ratio`` for the *model*.
        #:
        #: Deliberately not ``config.annual_interest_rate``. That setting is a
        #: business parameter for the affordability figures shown to the
        #: applicant, and it is expected to change. The model feature must be
        #: computed exactly as it was during training, or it means something
        #: different at serve time: setting ANNUAL_INTEREST_RATE=0.25 moved one
        #: application from 0.563 to 0.400 with no other change. The trainer
        #: records the rate in the bundle, so serving reads it from there.
        #: Bundles predating that metadata fall back to config, which is what
        #: they were implicitly trained against.
        self.feature_rate = float(
            bundle.metadata.get("annual_interest_rate", config.annual_interest_rate)
        )

    # -- scoring -----------------------------------------------------------
    def score_features(self, features: dict[str, float]) -> float:
        """P(approve) for a raw feature dict, in the canonical column order."""
        frame = pd.DataFrame([features], columns=list(FEATURE_COLUMNS))
        return float(self.bundle.pipeline.predict_proba(frame)[0][1])

    def score(self, app: LoanApplication) -> float:
        return self.score_many([app])[0]

    def score_many(self, apps: Sequence[LoanApplication]) -> list[float]:
        """P(approve) for several applications in a single model call.

        The calibrated pipeline's cost is almost entirely per call, not per
        row: measured at 12.9 ms for one row and 14.4 ms for sixteen. A full
        assessment needs a dozen scores, so making them one at a time spent
        most of the request in fixed overhead. Row order is preserved.
        """
        if not apps:
            return []
        frame = build_feature_frame(list(apps), self.feature_rate)
        return [float(p) for p in self.bundle.pipeline.predict_proba(frame)[:, 1]]

    def band_for(self, probability: float) -> str:
        if probability >= self.config.approve_threshold:
            return "likely"
        if probability >= self.config.borderline_threshold:
            return "borderline"
        return "unlikely"

    # -- full assessment ---------------------------------------------------
    def assess(self, app: LoanApplication, with_guidance: bool = True) -> Assessment:
        features = engineer(app, self.feature_rate)
        probability = self.score_features(features)

        # Applicant-facing cost figures use the *configured* reference rate on
        # purpose — see feature_rate above for why the two differ.
        affordability = compute_affordability(app, self.config)
        rules = evaluate_rules(app, affordability, self.config)

        out_of_range_reason = _extrapolation_reason(app, features)
        factors = (
            explain(
                app=app,
                baselines=self.bundle.baselines,
                score_applications=self.score_many,
                base_probability=probability,
            )
            if out_of_range_reason is None
            else []
        )

        suggestions: list[Suggestion] = []
        # No guidance for someone already in the "likely" band (it would be
        # filler), nor when the application is out of range (the number it
        # would try to improve is not meaningful).
        if (
            with_guidance
            and probability < self.config.approve_threshold
            and out_of_range_reason is None
        ):
            suggestions = suggest_improvements(
                app=app,
                base_probability=probability,
                score_applications=self.score_many,
                config=self.config,
            )

        band = "unavailable" if out_of_range_reason else self.band_for(probability)
        return Assessment(
            probability=probability,
            band=band,
            band_label=BAND_LABELS[band],
            affordability=affordability,
            rules=rules,
            rules_verdict=rules_verdict(rules),
            factors=factors,
            suggestions=suggestions,
            model_version=self.bundle.version,
            model_calibrated=self.bundle.is_calibrated,
            model_metrics=self.bundle.metrics,
            model_training_rows=int(self.bundle.metadata.get("training_rows", 0)),
            out_of_range_reason=out_of_range_reason,
            assessed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def compare(self, applications: list[LoanApplication]) -> list[dict[str, Any]]:
        """Score several scenarios side by side for what-if comparison."""
        results = []
        probabilities = self.score_many(applications)
        for index, (app, probability) in enumerate(
            zip(applications, probabilities, strict=True)
        ):
            affordability = compute_affordability(app, self.config)
            reason = _extrapolation_reason(app, engineer(app, self.feature_rate))
            band = "unavailable" if reason else self.band_for(probability)
            results.append(
                {
                    "index": index,
                    "application": app.to_dict(),
                    "probability": None if reason else round(probability, 4),
                    "band": band,
                    "band_label": BAND_LABELS[band],
                    "out_of_range": reason is not None,
                    "out_of_range_reason": reason,
                    "monthly_instalment": round(affordability.monthly_instalment, 2),
                    "emi_to_income": round(affordability.emi_to_income, 4),
                }
            )
        return results
