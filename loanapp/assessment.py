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

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .config import Config
from .explain import Factor, Suggestion, explain, suggest_improvements
from .features import FEATURE_COLUMNS, LoanApplication, engineer
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

#: Applicant-facing labels for the three decision bands. Phrased as likelihood
#: rather than as a verdict, because that is what the model actually produces.
BAND_LABELS = {
    "likely": "Likely to be approved",
    "borderline": "Borderline",
    "unlikely": "Unlikely to be approved",
}


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
    assessed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": round(self.probability, 4),
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
            },
            "assessed_at": self.assessed_at,
            "disclaimer": DISCLAIMER,
        }


class Assessor:
    """Scores applications against a loaded model bundle.

    Constructed once per process and shared; it holds no per-request state.
    """

    def __init__(self, bundle: ModelBundle, config: Config):
        self.bundle = bundle
        self.config = config

    # -- scoring -----------------------------------------------------------
    def score_features(self, features: dict[str, float]) -> float:
        """P(approve) for a raw feature dict, in the canonical column order."""
        frame = pd.DataFrame([features], columns=list(FEATURE_COLUMNS))
        return float(self.bundle.pipeline.predict_proba(frame)[0][1])

    def score(self, app: LoanApplication) -> float:
        return self.score_features(engineer(app, self.config.annual_interest_rate))

    def band_for(self, probability: float) -> str:
        if probability >= self.config.approve_threshold:
            return "likely"
        if probability >= self.config.borderline_threshold:
            return "borderline"
        return "unlikely"

    # -- full assessment ---------------------------------------------------
    def assess(self, app: LoanApplication, with_guidance: bool = True) -> Assessment:
        features = engineer(app, self.config.annual_interest_rate)
        probability = self.score_features(features)

        affordability = compute_affordability(app, self.config)
        rules = evaluate_rules(app, affordability, self.config)

        factors = explain(
            app=app,
            baselines=self.bundle.baselines,
            score_application=self.score,
            base_probability=probability,
        )

        suggestions: list[Suggestion] = []
        # Someone already in the "likely" band does not need advice on how to
        # get there; showing it anyway would be filler.
        if with_guidance and probability < self.config.approve_threshold:
            suggestions = suggest_improvements(
                app=app,
                base_probability=probability,
                score_application=self.score,
                config=self.config,
            )

        band = self.band_for(probability)
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
            assessed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def compare(self, applications: list[LoanApplication]) -> list[dict[str, Any]]:
        """Score several scenarios side by side for what-if comparison."""
        results = []
        for index, app in enumerate(applications):
            probability = self.score(app)
            affordability = compute_affordability(app, self.config)
            band = self.band_for(probability)
            results.append(
                {
                    "index": index,
                    "application": app.to_dict(),
                    "probability": round(probability, 4),
                    "band": band,
                    "band_label": BAND_LABELS[band],
                    "monthly_instalment": round(affordability.monthly_instalment, 2),
                    "emi_to_income": round(affordability.emi_to_income, 4),
                }
            )
        return results
