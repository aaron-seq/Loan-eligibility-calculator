"""Single source of truth for the feature contract.

Both training (``loanapp.ml.train``) and inference (``loanapp.assessment``)
import ``build_feature_frame`` from this module. That is the whole point of the
file: the previous implementation engineered features twice, in two places,
in two different column orders, and the mismatch silently broke prediction.

Two deliberate omissions from the model feature set
---------------------------------------------------
1. **Gender and marital status are not model inputs.** ECOA / Regulation B
   prohibits using them as a basis for a credit decision. They are not
   collected by the application form either. ``docs/MODEL.md`` records the
   measured accuracy cost of leaving them out.

2. **CIBIL score is not a model input.** The training dataset has no credit
   bureau score. The previous trainer manufactured one with
   ``np.random.randint(300, 850)`` and fed that noise to the model while the
   UI advertised it as a real feature. We collect a real CIBIL score and use
   it in the deterministic rules layer (``loanapp.rules``) where its effect is
   published and auditable, rather than pretending the model learned from it.

Units
-----
The public API speaks in plain currency units: ``applicant_income`` and
``coapplicant_income`` are *monthly* amounts, ``loan_amount`` is the *total*
principal. The training CSV stores ``LoanAmount`` in thousands; the conversion
happens once, here, in :func:`from_training_row`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Mapping

import pandas as pd

#: Column order handed to the estimator. Persisted with every model bundle and
#: verified on load, so a change here can never silently mis-align inference.
#:
#: The absolute loan amount is deliberately absent. A raw ``log(loan_amount)``
#: term correlates 0.67 with log income and carries a "bigger loans went to
#: better applicants" effect that inverts the direction a decisioning tool
#: needs: it made a 3,000,000 request score *better* than a 128,000 one for the
#: same applicant. Loan size now enters only through the affordability ratios,
#: which point the right way. Measured cost: 0.0009 ROC-AUC for the linear
#: model, and it improves the forest.
FEATURE_COLUMNS: tuple[str, ...] = (
    "credit_history",
    "total_income_log",
    "loan_term_months",
    "loan_to_income_ratio",
    "emi_to_income_ratio",
    "coapplicant_share",
    "dependents",
    "is_graduate",
    "is_self_employed",
    "property_area_semiurban",
    "property_area_urban",
)

#: Human-readable labels used in explanations shown to applicants.
FEATURE_LABELS: dict[str, str] = {
    "credit_history": "Clean credit history",
    "total_income_log": "Total household income",
    "loan_term_months": "Repayment term",
    "loan_to_income_ratio": "Loan size relative to income",
    "emi_to_income_ratio": "Monthly instalment relative to income",
    "coapplicant_share": "Share of income from a co-applicant",
    "dependents": "Number of dependents",
    "is_graduate": "Graduate education",
    "is_self_employed": "Self-employed",
    "property_area_semiurban": "Property in a semi-urban area",
    "property_area_urban": "Property in an urban area",
}

#: Applicant-visible inputs, in the order explanations should consider them.
#: Attribution happens at *this* level, not at the engineered-feature level:
#: one input drives several features (``loan_amount`` moves both the
#: loan-to-income and the instalment-to-income ratio), so perturbing features
#: independently produces counterfactuals that are not real applications and
#: reads as self-contradictory to the person looking at it.
EXPLAINABLE_INPUTS: tuple[str, ...] = (
    "credit_history",
    "applicant_income",
    "coapplicant_income",
    "loan_amount",
    "loan_term_months",
    "dependents",
    "education",
    "self_employed",
    "property_area",
)

INPUT_LABELS: dict[str, str] = {
    "credit_history": "Your credit history",
    "applicant_income": "Your income",
    "coapplicant_income": "Co-applicant income",
    "loan_amount": "The amount you want to borrow",
    "loan_term_months": "Your repayment term",
    "dependents": "Number of dependents",
    "education": "Education",
    "self_employed": "Employment type",
    "property_area": "Property area",
}

PROPERTY_AREAS = ("rural", "semiurban", "urban")

#: Loan amounts in ``data/train.csv`` are recorded in thousands.
TRAINING_LOAN_AMOUNT_UNIT = 1_000


@dataclass(frozen=True)
class LoanApplication:
    """A validated loan application in canonical units.

    This is the only shape the scoring stack accepts. HTTP-level parsing and
    range checks live in :mod:`loanapp.schemas`; by the time an application
    reaches here it is already known to be well-formed.
    """

    applicant_income: float  # monthly
    coapplicant_income: float  # monthly
    loan_amount: float  # total principal
    loan_term_months: int
    credit_history: int  # 1 = no record of default, 0 = adverse history
    dependents: int
    education: str  # "graduate" | "not_graduate"
    self_employed: bool
    property_area: str  # one of PROPERTY_AREAS
    cibil_score: int | None = None  # rules layer only, never a model input

    @property
    def total_income(self) -> float:
        return self.applicant_income + self.coapplicant_income

    def replace(self, **changes: Any) -> "LoanApplication":
        """Return a copy with ``changes`` applied.

        Used by the what-if engine, which needs to score many nearby variants
        of the same application.
        """
        return LoanApplication(**{**asdict(self), **changes})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def monthly_instalment(principal: float, annual_rate: float, term_months: int) -> float:
    """Standard amortising EMI.

    ``annual_rate`` is a fraction (0.105 for 10.5%). A zero rate degenerates to
    straight-line repayment rather than dividing by zero.
    """
    if term_months <= 0:
        return 0.0
    if annual_rate <= 0:
        return principal / term_months
    r = annual_rate / 12
    growth = (1 + r) ** term_months
    return principal * r * growth / (growth - 1)


def engineer(app: LoanApplication, annual_rate: float) -> dict[str, float]:
    """Derive the model feature values for one application.

    Kept separate from :func:`build_feature_frame` so the rules and explanation
    layers can read individual engineered values without building a DataFrame.
    """
    total_income = app.total_income
    emi = monthly_instalment(app.loan_amount, annual_rate, app.loan_term_months)

    # Guard every ratio: a zero-income application is valid input to reject,
    # not a reason to raise ZeroDivisionError inside the model.
    loan_to_income = app.loan_amount / total_income if total_income > 0 else 0.0
    emi_to_income = emi / total_income if total_income > 0 else 0.0
    coapplicant_share = (
        app.coapplicant_income / total_income if total_income > 0 else 0.0
    )

    return {
        "credit_history": float(app.credit_history),
        "total_income_log": math.log1p(max(total_income, 0.0)),
        "loan_term_months": float(app.loan_term_months),
        "loan_to_income_ratio": float(loan_to_income),
        "emi_to_income_ratio": float(emi_to_income),
        "coapplicant_share": float(coapplicant_share),
        "dependents": float(app.dependents),
        "is_graduate": 1.0 if app.education == "graduate" else 0.0,
        "is_self_employed": 1.0 if app.self_employed else 0.0,
        "property_area_semiurban": 1.0 if app.property_area == "semiurban" else 0.0,
        "property_area_urban": 1.0 if app.property_area == "urban" else 0.0,
    }


def build_feature_frame(
    applications: LoanApplication | list[LoanApplication], annual_rate: float
) -> pd.DataFrame:
    """Build the model input matrix in the canonical column order."""
    if isinstance(applications, LoanApplication):
        applications = [applications]
    rows = [engineer(a, annual_rate) for a in applications]
    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))


def from_training_row(row: Mapping[str, Any]) -> LoanApplication:
    """Map one row of ``data/train.csv`` onto :class:`LoanApplication`.

    Missing values are left for the pipeline's imputer wherever possible; the
    two fields that cannot be imputed downstream (``Dependents``,
    ``Property_Area``) fall back to the dataset mode, which is documented in
    ``docs/MODEL.md``.
    """
    dependents_raw = str(row.get("Dependents", "0")).strip()
    if dependents_raw in ("", "nan", "None"):
        dependents_raw = "0"
    dependents = 3 if dependents_raw.startswith("3") else int(float(dependents_raw))

    loan_amount = row.get("LoanAmount")
    loan_amount = (
        float(loan_amount) * TRAINING_LOAN_AMOUNT_UNIT
        if loan_amount is not None and not _is_missing(loan_amount)
        else float("nan")
    )

    term = row.get("Loan_Amount_Term")
    term = 360 if term is None or _is_missing(term) else int(float(term))

    credit_history = row.get("Credit_History")
    # A missing bureau record is materially different from a clean one; the
    # dataset's own approval rate for missing Credit_History sits between the
    # two classes, so we treat it as adverse rather than imputing "clean".
    credit_history = (
        0
        if credit_history is None or _is_missing(credit_history)
        else int(float(credit_history))
    )

    area = str(row.get("Property_Area", "urban")).strip().lower()
    if area not in PROPERTY_AREAS:
        area = "urban"

    return LoanApplication(
        applicant_income=float(row.get("ApplicantIncome") or 0.0),
        coapplicant_income=float(row.get("CoapplicantIncome") or 0.0),
        loan_amount=loan_amount,
        loan_term_months=term,
        credit_history=credit_history,
        dependents=dependents,
        education="not_graduate"
        if str(row.get("Education", "")).strip().lower().startswith("not")
        else "graduate",
        self_employed=str(row.get("Self_Employed", "No")).strip().lower() == "yes",
        property_area=area,
    )


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
