"""Deterministic affordability and credit rules.

This layer exists because a model trained on 614 historical rows should not be
the only thing standing between an applicant and a number they might act on.
Every check here is a published formula with a stated threshold, so an analyst
can reproduce it by hand and an applicant can understand it.

The rules layer is *advisory and separate* from the model. It never silently
overrides the model probability; the combined view is assembled in
:mod:`loanapp.assessment`, which reports both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .config import Config
from .features import LoanApplication, monthly_instalment

Severity = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class RuleResult:
    code: str
    severity: Severity
    title: str
    detail: str


@dataclass(frozen=True)
class Affordability:
    """Cash-flow arithmetic for the requested loan."""

    monthly_instalment: float
    total_income: float
    emi_to_income: float
    total_repayable: float
    total_interest: float
    max_affordable_loan: float
    annual_interest_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "monthly_instalment": round(self.monthly_instalment, 2),
            "total_income": round(self.total_income, 2),
            "emi_to_income": round(self.emi_to_income, 4),
            "total_repayable": round(self.total_repayable, 2),
            "total_interest": round(self.total_interest, 2),
            "max_affordable_loan": round(self.max_affordable_loan, 2),
            "annual_interest_rate": self.annual_interest_rate,
        }


def compute_affordability(app: LoanApplication, config: Config) -> Affordability:
    """Amortisation maths at the configured reference interest rate.

    ``max_affordable_loan`` inverts the EMI formula: the largest principal
    whose instalment stays within ``MAX_EMI_TO_INCOME`` of household income.
    """
    rate = config.annual_interest_rate
    emi = monthly_instalment(app.loan_amount, rate, app.loan_term_months)
    total_income = app.total_income
    emi_to_income = emi / total_income if total_income > 0 else 0.0

    total_repayable = emi * app.loan_term_months
    affordable_emi = total_income * config.max_emi_to_income

    if app.loan_term_months <= 0:
        max_affordable = 0.0
    elif rate <= 0:
        max_affordable = affordable_emi * app.loan_term_months
    else:
        r = rate / 12
        growth = (1 + r) ** app.loan_term_months
        max_affordable = affordable_emi * (growth - 1) / (r * growth)

    return Affordability(
        monthly_instalment=emi,
        total_income=total_income,
        emi_to_income=emi_to_income,
        total_repayable=total_repayable,
        total_interest=max(total_repayable - app.loan_amount, 0.0),
        max_affordable_loan=max(max_affordable, 0.0),
        annual_interest_rate=rate,
    )


def evaluate_rules(
    app: LoanApplication, affordability: Affordability, config: Config
) -> list[RuleResult]:
    """Run every deterministic check and return one result per rule."""
    results: list[RuleResult] = []
    ratio_pct = affordability.emi_to_income * 100
    limit_pct = config.max_emi_to_income * 100

    # --- Affordability -----------------------------------------------------
    if affordability.total_income <= 0:
        results.append(
            RuleResult(
                "income_present",
                "fail",
                "No income recorded",
                "Lenders cannot assess repayment capacity without declared income.",
            )
        )
    elif affordability.emi_to_income > config.max_emi_to_income:
        results.append(
            RuleResult(
                "emi_to_income",
                "fail",
                "Instalment exceeds affordability limit",
                f"The monthly instalment is {ratio_pct:.0f}% of household income, "
                f"above the {limit_pct:.0f}% ceiling most lenders apply. "
                f"At this income the comfortable maximum is about "
                f"{affordability.max_affordable_loan:,.0f}.",
            )
        )
    elif affordability.emi_to_income > config.max_emi_to_income * 0.8:
        results.append(
            RuleResult(
                "emi_to_income",
                "warn",
                "Instalment is close to the affordability limit",
                f"The instalment is {ratio_pct:.0f}% of household income. "
                f"Lenders start applying extra scrutiny above "
                f"{limit_pct * 0.8:.0f}%.",
            )
        )
    else:
        results.append(
            RuleResult(
                "emi_to_income",
                "pass",
                "Instalment is affordable",
                f"The instalment is {ratio_pct:.0f}% of household income, within "
                f"the {limit_pct:.0f}% ceiling most lenders apply.",
            )
        )

    # --- Credit history ----------------------------------------------------
    if app.credit_history == 1:
        results.append(
            RuleResult(
                "credit_history",
                "pass",
                "Credit history meets guidelines",
                "No adverse repayment record was declared. In the training data "
                "this is the single strongest driver of approval.",
            )
        )
    else:
        results.append(
            RuleResult(
                "credit_history",
                "fail",
                "Adverse or missing credit history",
                "An adverse or absent repayment record is the most common reason "
                "applications in this dataset were declined.",
            )
        )

    # --- Credit bureau score ----------------------------------------------
    # The score is NOT a model input (the training data has none). It is used
    # only here, with published thresholds, so its effect is inspectable.
    if app.cibil_score is not None:
        if app.cibil_score >= config.cibil_strong_threshold:
            results.append(
                RuleResult(
                    "cibil_score",
                    "pass",
                    "Strong credit score",
                    f"A score of {app.cibil_score} is in the band most lenders "
                    f"treat as low risk (>= {config.cibil_strong_threshold}).",
                )
            )
        elif app.cibil_score >= config.cibil_caution_threshold:
            results.append(
                RuleResult(
                    "cibil_score",
                    "warn",
                    "Moderate credit score",
                    f"A score of {app.cibil_score} sits between the caution and "
                    f"strong bands ({config.cibil_caution_threshold}-"
                    f"{config.cibil_strong_threshold}). Expect a higher offered "
                    "interest rate.",
                )
            )
        else:
            results.append(
                RuleResult(
                    "cibil_score",
                    "fail",
                    "Credit score below common cut-offs",
                    f"A score of {app.cibil_score} is below the "
                    f"{config.cibil_caution_threshold} cut-off many lenders use "
                    "for unsecured retail lending.",
                )
            )

    # --- Term sanity -------------------------------------------------------
    if app.loan_term_months > 360:
        results.append(
            RuleResult(
                "loan_term",
                "warn",
                "Unusually long term",
                f"A {app.loan_term_months}-month term is longer than the 360-month "
                "maximum in the training data, so the estimate is extrapolating.",
            )
        )

    return results


def rules_verdict(results: list[RuleResult]) -> Severity:
    """Worst severity across all rules."""
    if any(r.severity == "fail" for r in results):
        return "fail"
    if any(r.severity == "warn" for r in results):
        return "warn"
    return "pass"
