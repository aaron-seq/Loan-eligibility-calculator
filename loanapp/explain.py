"""Explanations and what-if guidance.

Two questions an applicant actually has, answered separately:

* **"Why this result?"** — :func:`explain` attributes the estimate to individual
  inputs.
* **"What would change it?"** — :func:`suggest_improvements` re-scores concrete
  variations of the application and reports which ones move the number.

Attribution method
------------------
**One-input-at-a-time ablation.** Each applicant-visible input is set to its
training-set baseline, the whole application is re-scored, and the change is
attributed to that input.

The unit of attribution is deliberately the *input*, not the engineered
feature. ``loan_amount`` drives three features at once — the log amount, the
loan-to-income ratio and the instalment-to-income ratio. Perturbing those
independently produces counterfactuals that are not real applications, and it
showed up immediately as nonsense in the UI: an early version reported that a
large loan amount *helped* the applicant while loan-to-income *hurt* them, from
the same application. Varying the input keeps every counterfactual coherent.

The method is model-agnostic, deterministic and cheap (nine extra scores). It
**ignores interactions** — two inputs that only matter jointly will each look
small, and the attributions do not sum exactly to the total. That limit is
stated in the model card and on the result page rather than papered over. It is
not SHAP and is not presented as such.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import Config
from .features import EXPLAINABLE_INPUTS, INPUT_LABELS, LoanApplication

#: Attributions smaller than this are noise, not explanation.
MIN_MEANINGFUL_IMPACT = 0.005


@dataclass(frozen=True)
class Factor:
    """One input's contribution to the estimate."""

    feature: str
    label: str
    impact: float  # signed probability delta vs. the baseline application
    direction: str  # "helps" | "hurts"
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "label": self.label,
            "impact": round(self.impact, 4),
            "direction": self.direction,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Suggestion:
    """A concrete change and its measured effect on the estimate."""

    action: str
    detail: str
    projected_probability: float
    delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "detail": self.detail,
            "projected_probability": round(self.projected_probability, 4),
            "delta": round(self.delta, 4),
        }


def explain(
    app: LoanApplication,
    baselines: dict[str, Any],
    score_application: Callable[[LoanApplication], float],
    base_probability: float,
    limit: int = 5,
) -> list[Factor]:
    """Rank the applicant's inputs by how much each moved the estimate.

    ``score_application`` is injected rather than imported so this module stays
    independent of how the model is loaded, and so tests can supply a trivial
    scorer.
    """
    factors: list[Factor] = []

    for field in EXPLAINABLE_INPUTS:
        if field not in baselines:
            continue
        value = getattr(app, field)
        baseline = baselines[field]
        if value == baseline:
            continue

        impact = base_probability - score_application(app.replace(**{field: baseline}))
        if abs(impact) < MIN_MEANINGFUL_IMPACT:
            continue

        factors.append(
            Factor(
                feature=field,
                label=INPUT_LABELS.get(field, field),
                impact=impact,
                direction="helps" if impact > 0 else "hurts",
                detail=_describe(field, value, baseline, impact),
            )
        )

    factors.sort(key=lambda f: abs(f.impact), reverse=True)
    return factors[:limit]


def _describe(field: str, value: Any, baseline: Any, impact: float) -> str:
    """Plain-language sentence for one factor."""
    verb = "raises" if impact > 0 else "lowers"
    pts = abs(impact) * 100

    if field == "credit_history":
        state = (
            "Having no defaults or missed payments"
            if value == 1
            else "Having an adverse or missing credit record"
        )
        return f"{state} {verb} your estimate by about {pts:.0f} percentage points."

    if field in ("applicant_income", "coapplicant_income", "loan_amount"):
        comparison = "higher" if value > baseline else "lower"
        return (
            f"At {value:,.0f} this is {comparison} than a typical application in "
            f"the training data (around {baseline:,.0f}), which {verb} your "
            f"estimate by about {pts:.0f} points."
        )

    if field == "loan_term_months":
        comparison = "longer" if value > baseline else "shorter"
        return (
            f"A {value}-month term is {comparison} than the typical {baseline:.0f} "
            f"months, which {verb} your estimate by about {pts:.0f} points."
        )

    if field == "dependents":
        return (
            f"Having {value} dependents (typical is {baseline:.0f}) {verb} your "
            f"estimate by about {pts:.0f} points."
        )

    label = INPUT_LABELS.get(field, field)
    return (
        f"{label} recorded as “{value}” rather than “{baseline}” {verb} your "
        f"estimate by about {pts:.0f} points."
    )


def suggest_improvements(
    app: LoanApplication,
    base_probability: float,
    score_application: Callable[[LoanApplication], float],
    config: Config,
    limit: int = 4,
) -> list[Suggestion]:
    """Score realistic variations of the application and report what helps.

    Every candidate is something the applicant could actually do — borrow less,
    stretch the term, add a co-applicant, clear an arrear. We never suggest
    changing an attribute they cannot change, and we never suggest
    misreporting one.
    """
    candidates: list[tuple[str, str, LoanApplication]] = []

    # Ascending order matters: the smallest sufficient reduction is selected
    # below, and that selection relies on this ordering.
    for pct in (0.10, 0.20, 0.30):
        reduced = app.loan_amount * (1 - pct)
        candidates.append(
            (
                f"Borrow {pct:.0%} less",
                f"Requesting {reduced:,.0f} instead of {app.loan_amount:,.0f}.",
                app.replace(loan_amount=reduced),
            )
        )

    if app.loan_term_months < 360:
        longer = min(app.loan_term_months + 60, 360)
        candidates.append(
            (
                f"Extend the term to {longer} months",
                (
                    "A longer term lowers the monthly instalment. It also increases "
                    "the total interest you pay."
                ),
                app.replace(loan_term_months=longer),
            )
        )

    if app.coapplicant_income == 0 and app.applicant_income > 0:
        added = app.applicant_income * 0.5
        candidates.append(
            (
                "Add a co-applicant",
                (
                    f"A co-applicant earning {added:,.0f} per month raises household "
                    "income and spreads the obligation."
                ),
                app.replace(coapplicant_income=added),
            )
        )

    if app.credit_history == 0:
        candidates.append(
            (
                "Clear the adverse credit record",
                (
                    "Settling outstanding defaults and rebuilding a clean repayment "
                    "record has the largest single effect in this model. It "
                    "typically takes 6-12 months."
                ),
                app.replace(credit_history=1),
            )
        )

    suggestions: list[Suggestion] = []
    for action, detail, variant in candidates:
        projected = score_application(variant)
        delta = projected - base_probability
        if delta <= MIN_MEANINGFUL_IMPACT:
            continue
        suggestions.append(
            Suggestion(
                action=action,
                detail=detail,
                projected_probability=projected,
                delta=delta,
            )
        )

    # Keep one "borrow less" variant: the *smallest* reduction that clears the
    # approval band, falling back to the most effective one. Advising a 30% cut
    # when 10% is enough is worse advice, even though its delta is larger.
    borrow_variants = [s for s in suggestions if s.action.startswith("Borrow")]
    others = [s for s in suggestions if not s.action.startswith("Borrow")]
    if borrow_variants:
        sufficient = [
            s
            for s in borrow_variants
            if s.projected_probability >= config.approve_threshold
        ]
        others.append(
            sufficient[0] if sufficient else max(borrow_variants, key=lambda s: s.delta)
        )

    others.sort(key=lambda s: s.delta, reverse=True)
    return others[:limit]
