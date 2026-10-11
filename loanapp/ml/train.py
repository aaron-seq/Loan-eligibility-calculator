"""Training pipeline with honest evaluation.

Run with::

    python -m loanapp.ml.train

What this does differently from the previous trainer
----------------------------------------------------
* Reuses :mod:`loanapp.features` instead of re-implementing feature
  engineering, so serving cannot drift from training.
* Does not fabricate a CIBIL score. The old trainer injected
  ``np.random.randint(300, 850)`` as a feature; the model could only learn
  noise from it while the product advertised it as real.
* Compares candidate models on repeated stratified CV and reports the spread,
  because 614 rows cannot support a single headline accuracy number.
* Calibrates probabilities and measures whether calibration actually helped
  (Brier score + reliability bins), since the product shows the probability to
  applicants rather than just the argmax label.
* Reports the accuracy cost of excluding gender and marital status, so the
  fair-lending decision is documented with a number rather than asserted.
* Writes ``models/metrics.json`` and ``docs/MODEL.md`` so the claims in the
  documentation are generated from the run, not typed by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import BASE_DIR, load_config
from ..features import (
    FEATURE_COLUMNS,
    LoanApplication,
    build_feature_frame,
    from_training_row,
)
from .registry import ModelBundle, save_bundle

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_dataset(
    csv_path: str, annual_rate: float
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[LoanApplication]]:
    """Return ``(X, y, raw, applications)`` for the training CSV.

    ``raw`` is kept so the fairness audit can group the holdout by attributes
    deliberately excluded from ``X``; ``applications`` is kept so explainer
    baselines can be computed over applicant-visible inputs.
    """
    raw = pd.read_csv(csv_path)
    if "Loan_Status" not in raw.columns:
        raise ValueError(f"{csv_path} has no Loan_Status column")

    raw = raw[raw["Loan_Status"].notna()].reset_index(drop=True)
    applications = [from_training_row(row) for _, row in raw.iterrows()]
    X = build_feature_frame(applications, annual_rate)
    y = (raw["Loan_Status"].astype(str).str.upper().str[0] == "Y").astype(int)
    return X, y, raw, applications


def fingerprint(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Candidate models
# --------------------------------------------------------------------------
def candidate_pipelines() -> dict[str, Pipeline]:
    """Two candidates: an interpretable linear baseline and a forest.

    ``class_weight="balanced"`` matters here — the dataset is 69% approvals,
    and an uncorrected model drifts toward approving nearly everyone, which is
    exactly the failure mode an applicant-facing tool must avoid.
    """
    common = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    return {
        "logistic_regression": Pipeline(
            [
                *common,
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                *common,
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=6,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def cross_validate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Repeated stratified CV. We report the spread, not just the mean."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=RANDOM_STATE)
    auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return {
        "roc_auc_mean": float(auc.mean()),
        "roc_auc_std": float(auc.std()),
        # A 95% interval across CV folds. It is not a true confidence interval
        # for generalisation error, but it does show how unstable 614 rows are.
        "roc_auc_ci95": [
            float(np.percentile(auc, 2.5)),
            float(np.percentile(auc, 97.5)),
        ],
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std()),
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def reliability_table(
    y_true: np.ndarray, probs: np.ndarray, bins: int = 5
) -> list[dict[str, Any]]:
    """Predicted vs. observed approval rate per probability bin.

    This is what makes "72% likely" an honest statement rather than a
    decoration: if the 0.6-0.8 bucket approves 71% of the time, the number
    means what it says.
    """
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in pairwise(edges):
        mask = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if not mask.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0})
            continue
        rows.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(mask.sum()),
                "mean_predicted": round(float(probs[mask].mean()), 4),
                "observed_rate": round(float(y_true[mask].mean()), 4),
            }
        )
    return rows


def evaluate(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probs)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probs)), 4),
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "reliability": reliability_table(y_true, probs),
    }


def fairness_audit(
    raw_holdout: pd.DataFrame, probs: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Approval rates by attributes the model is not allowed to use.

    Excluding an attribute from the inputs does not guarantee parity in the
    outputs — correlated features can reintroduce it. Measuring is the only
    way to know, so we measure and publish.
    """
    preds = (probs >= threshold).astype(int)
    audit: dict[str, Any] = {}
    for column in ("Gender", "Married"):
        if column not in raw_holdout.columns:
            continue
        groups: dict[str, Any] = {}
        for value, idx in raw_holdout.groupby(column).groups.items():
            positions = [raw_holdout.index.get_loc(i) for i in idx]
            if len(positions) < 10:  # too few to say anything meaningful
                continue
            groups[str(value)] = {
                "n": len(positions),
                "approval_rate": round(float(preds[positions].mean()), 4),
            }
        if len(groups) >= 2:
            rates = [g["approval_rate"] for g in groups.values()]
            groups["max_gap"] = round(max(rates) - min(rates), 4)
        audit[column] = groups
    return audit


def ablation_cost(
    X: pd.DataFrame, y: pd.Series, raw: pd.DataFrame, pipeline: Pipeline
) -> dict[str, Any]:
    """Measure what excluding gender and marital status costs in ROC-AUC."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    without = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    augmented = X.copy()
    augmented["gender_male"] = (
        raw.get("Gender", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .eq("male")
        .astype(float)
    )
    augmented["married_yes"] = (
        raw.get("Married", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .eq("yes")
        .astype(float)
    )
    with_protected = cross_val_score(
        pipeline, augmented, y, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    return {
        "roc_auc_without_protected": round(float(without.mean()), 4),
        "roc_auc_with_protected": round(float(with_protected.mean()), 4),
        "cost_of_exclusion": round(float(with_protected.mean() - without.mean()), 4),
    }


def monotonicity_check(pipeline: Pipeline, annual_rate: float) -> dict[str, Any]:
    """Does borrowing less ever score *worse*?

    This exists because it caught a real defect. A random forest fitted to 614
    rows produced a coarse step function in which cutting the requested loan by
    50% raised the estimate to 0.74 but cutting it by 70% dropped it back to
    0.65. The model's ranking metrics were fine; its response surface was not.

    An applicant-facing tool that answers "what would improve my chances?" must
    be monotonic in the obvious direction, or the advice it gives is wrong. We
    sweep the loan amount downward across a grid of representative applicants
    and count how often the probability falls.
    """
    # A reversal only matters if a person would notice it. Half a percentage
    # point is below the resolution of anything we display, so we count strict
    # reversals for the record but judge on material ones.
    material_tolerance = 0.005

    violations = 0
    material_violations = 0
    comparisons = 0
    worst_drop = 0.0

    for income in (25_000, 50_000, 100_000):
        for term in (120, 240, 360):
            for credit in (0, 1):
                base = LoanApplication(
                    applicant_income=income,
                    coapplicant_income=0,
                    loan_amount=income * 60,
                    loan_term_months=term,
                    credit_history=credit,
                    dependents=0,
                    education="graduate",
                    self_employed=False,
                    property_area="urban",
                )
                # Sweep across three orders of magnitude. An earlier, narrower
                # sweep (down to 20% of the request) reported the model as
                # monotonic while it was still inverted far below that, which
                # produced a visibly wrong explanation for high-income
                # applicants. The check is only as good as its range.
                variants = [
                    base.replace(loan_amount=income * 60 * factor)
                    for factor in (
                        1.0,
                        0.9,
                        0.8,
                        0.7,
                        0.6,
                        0.5,
                        0.4,
                        0.3,
                        0.2,
                        0.1,
                        0.05,
                        0.02,
                        0.01,
                    )
                ]
                probs = pipeline.predict_proba(
                    build_feature_frame(variants, annual_rate)
                )[:, 1]
                for previous, current in pairwise(probs):
                    comparisons += 1
                    drop = float(previous - current)
                    if drop > 1e-9:
                        violations += 1
                        worst_drop = max(worst_drop, drop)
                    if drop > material_tolerance:
                        material_violations += 1

    return {
        "comparisons": comparisons,
        "violations": violations,
        "material_violations": material_violations,
        "worst_drop": round(worst_drop, 4),
        "tolerance": material_tolerance,
        "monotonic": material_violations == 0,
    }


def compute_baselines(applications: list) -> dict[str, Any]:
    """Reference value per *input*, used by the explainer.

    Baselines are computed over applicant-visible inputs rather than engineered
    features, because that is the level at which explanations are produced —
    see the module docstring of :mod:`loanapp.explain`. Numeric inputs use the
    median; categorical ones use the mode, so an explanation compares against
    "the other option" rather than an average that no applicant has.
    """
    frame = pd.DataFrame([a.to_dict() for a in applications])
    baselines: dict[str, Any] = {}

    for field in ("applicant_income", "coapplicant_income", "loan_amount"):
        baselines[field] = float(frame[field].median())

    baselines["loan_term_months"] = int(frame["loan_term_months"].median())
    baselines["dependents"] = int(frame["dependents"].median())
    baselines["credit_history"] = int(frame["credit_history"].mode().iloc[0])
    baselines["education"] = str(frame["education"].mode().iloc[0])
    baselines["self_employed"] = bool(frame["self_employed"].mode().iloc[0])
    baselines["property_area"] = str(frame["property_area"].mode().iloc[0])

    return baselines


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def train(
    data_path: str,
    model_dir: str,
    docs_path: str | None,
    annual_rate: float,
    threshold: float,
) -> dict[str, Any]:
    X, y, raw, applications = load_dataset(data_path, annual_rate)
    print(f"Loaded {len(X)} rows, {X.shape[1]} features, approval rate {y.mean():.1%}")

    X_train, X_test, y_train, y_test, _raw_train, raw_test = train_test_split(
        X, y, raw, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 1. Select a model on cross-validated ROC-AUC over the training split
    #    only. The holdout is never used for selection.
    cv_results: dict[str, dict[str, float]] = {}
    for name, pipeline in candidate_pipelines().items():
        cv_results[name] = cross_validate(pipeline, X_train, y_train)
        print(
            f"  {name:22s} CV ROC-AUC {cv_results[name]['roc_auc_mean']:.4f} "
            f"(+/- {cv_results[name]['roc_auc_std']:.4f})"
        )

    # 1b. Screen candidates for a coherent response surface before ranking
    #     them on accuracy. A model that scores a smaller loan *worse* cannot
    #     back the "what would improve my chances?" feature, however good its
    #     AUC is, so it is disqualified rather than merely penalised.
    monotonicity: dict[str, dict[str, Any]] = {}
    for name, pipeline in candidate_pipelines().items():
        pipeline.fit(X_train, y_train)
        monotonicity[name] = monotonicity_check(pipeline, annual_rate)
        print(
            f"  {name:22s} monotonic in loan amount: "
            f"{monotonicity[name]['monotonic']} "
            f"({monotonicity[name]['material_violations']} material / "
            f"{monotonicity[name]['violations']} strict of "
            f"{monotonicity[name]['comparisons']}, worst drop "
            f"{monotonicity[name]['worst_drop']:.3f})"
        )

    eligible = [n for n in cv_results if monotonicity[n]["monotonic"]] or list(cv_results)
    best_name = max(eligible, key=lambda n: cv_results[n]["roc_auc_mean"])

    # Among eligible models, prefer logistic regression unless another is
    # clearly better — its coefficients are directly readable by an analyst or
    # a regulator, which matters more here than a fraction of a point of AUC
    # that sits well inside the fold-to-fold spread.
    if "logistic_regression" in eligible:
        margin = (
            cv_results[best_name]["roc_auc_mean"]
            - cv_results["logistic_regression"]["roc_auc_mean"]
        )
        if margin < cv_results["logistic_regression"]["roc_auc_std"]:
            best_name = "logistic_regression"

    print(f"Selected: {best_name}")

    base_pipeline = candidate_pipelines()[best_name]

    # 2. Uncalibrated baseline on the holdout, so we can prove calibration
    #    was worth applying rather than assuming it.
    base_pipeline.fit(X_train, y_train)
    uncalibrated_probs = base_pipeline.predict_proba(X_test)[:, 1]
    uncalibrated = evaluate(y_test.to_numpy(), uncalibrated_probs, threshold)

    # 3. Calibrate. Sigmoid, not isotonic: ~490 training rows is far too few
    #    for isotonic regression, which would overfit the calibration curve.
    calibrated = CalibratedClassifierCV(
        candidate_pipelines()[best_name], method="sigmoid", cv=5
    )
    calibrated.fit(X_train, y_train)
    calibrated_probs = calibrated.predict_proba(X_test)[:, 1]
    calibrated_metrics = evaluate(y_test.to_numpy(), calibrated_probs, threshold)

    use_calibrated = calibrated_metrics["brier_score"] <= uncalibrated["brier_score"]
    print(
        f"Brier: uncalibrated {uncalibrated['brier_score']:.4f} vs "
        f"calibrated {calibrated_metrics['brier_score']:.4f} -> "
        f"{'calibrated' if use_calibrated else 'uncalibrated'}"
    )

    holdout_probs = calibrated_probs if use_calibrated else uncalibrated_probs

    # 4. Audits.
    fairness = fairness_audit(raw_test.reset_index(drop=True), holdout_probs, threshold)
    ablation = ablation_cost(X, y, raw, candidate_pipelines()[best_name])

    # 5. Refit the chosen estimator on all data for serving. The reported
    #    metrics remain the holdout metrics above — they are not re-measured
    #    on data the final model has now seen.
    if use_calibrated:
        final = CalibratedClassifierCV(
            candidate_pipelines()[best_name], method="sigmoid", cv=5
        )
    else:
        final = candidate_pipelines()[best_name]
    final.fit(X, y)

    # Re-check the artifact that will actually be served. Calibration wraps and
    # averages several fold models, so monotonicity of the base estimator does
    # not automatically carry over.
    served_monotonicity = monotonicity_check(final, annual_rate)
    print(
        f"Served model monotonic in loan amount: {served_monotonicity['monotonic']} "
        f"({served_monotonicity['material_violations']} material violations of "
        f"{served_monotonicity['comparisons']} comparisons)"
    )

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "path": os.path.relpath(data_path, BASE_DIR),
            "sha256_prefix": fingerprint(data_path),
            "rows": len(X),
            "approval_rate": round(float(y.mean()), 4),
            "holdout_rows": len(X_test),
        },
        "selected_model": best_name,
        "calibrated": use_calibrated,
        "cross_validation": cv_results,
        "monotonicity": monotonicity,
        "served_model_monotonicity": served_monotonicity,
        "holdout_uncalibrated": uncalibrated,
        "holdout_calibrated": calibrated_metrics,
        "fairness_audit": fairness,
        "protected_attribute_ablation": ablation,
        "feature_columns": list(FEATURE_COLUMNS),
    }

    bundle = ModelBundle(
        pipeline=final,
        feature_columns=FEATURE_COLUMNS,
        baselines=compute_baselines(applications),
        metrics={
            "roc_auc": (calibrated_metrics if use_calibrated else uncalibrated)[
                "roc_auc"
            ],
            "brier_score": (calibrated_metrics if use_calibrated else uncalibrated)[
                "brier_score"
            ],
            "accuracy": (calibrated_metrics if use_calibrated else uncalibrated)[
                "accuracy"
            ],
            "holdout_rows": len(X_test),
        },
        metadata={
            "version": f"{best_name}-{datetime.now(UTC):%Y%m%d}-{fingerprint(data_path)}",
            "algorithm": best_name,
            "calibrated": use_calibrated,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "annual_interest_rate": annual_rate,
            "training_rows": len(X),
            "monotonic_in_loan_amount": served_monotonicity["monotonic"],
        },
    )
    path = save_bundle(bundle, model_dir)
    print(f"Saved bundle -> {path}")

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved metrics -> {metrics_path}")

    if docs_path:
        write_model_card(metrics, bundle, docs_path)
        print(f"Wrote model card -> {docs_path}")

    return metrics


def _ablation_verdict(ablation: dict[str, Any]) -> str:
    """Phrase the ablation result according to its sign.

    Excluding protected attributes usually costs a little accuracy. On this
    dataset it does not, and the model card should say so rather than assume.
    """
    cost = ablation["cost_of_exclusion"]
    if cost > 0.005:
        return f"excluding them costs **{cost:.4f} ROC-AUC**, a price worth paying."
    if cost < -0.005:
        return (
            f"excluding them **improves** ROC-AUC by {abs(cost):.4f}. They were "
            "carrying no signal this model could not get elsewhere."
        )
    return (
        f"the difference ({cost:+.4f}) is within fold-to-fold noise, so exclusion "
        "is effectively free."
    )


def write_model_card(metrics: dict[str, Any], bundle: ModelBundle, path: str) -> None:
    """Generate ``docs/MODEL.md`` from the run.

    Written by the trainer on purpose: the previous README claimed "~85-90%
    accuracy" for a model that did not exist in the repository. Numbers in the
    documentation should come from a run or not be there.
    """
    m = (
        metrics["holdout_calibrated"]
        if metrics["calibrated"]
        else metrics["holdout_uncalibrated"]
    )
    cv = metrics["cross_validation"][metrics["selected_model"]]
    ds = metrics["dataset"]
    ablation = metrics["protected_attribute_ablation"]

    reliability_rows = "\n".join(
        f"| {r['bin']} | {r['n']} | "
        f"{r.get('mean_predicted', float('nan')):.2f} | "
        f"{r.get('observed_rate', float('nan')):.2f} |"
        if r["n"]
        else f"| {r['bin']} | 0 | – | – |"
        for r in m["reliability"]
    )

    fairness_rows = []
    for attribute, groups in metrics["fairness_audit"].items():
        for group, stats in groups.items():
            if group == "max_gap":
                continue
            fairness_rows.append(
                f"| {attribute} | {group} | {stats['n']} | {stats['approval_rate']:.2f} |"
            )
    fairness_table = "\n".join(fairness_rows) or "| – | – | – | – |"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"""# Model card

*Generated by `python -m loanapp.ml.train` on {
            metrics["generated_at"]
        }. Do not edit by hand.*

## What this model is

A binary classifier estimating whether a loan application **resembling those in
the training dataset** would have been approved. It is a decision-support
estimate, not a lending decision, and no lender is bound by it.

| | |
|---|---|
| Algorithm | `{metrics["selected_model"]}` |
| Probability calibration | {
            "sigmoid (Platt), applied"
            if metrics["calibrated"]
            else "none — calibration did not improve the Brier score"
        } |
| Bundle version | `{bundle.version}` |
| Features | {len(metrics["feature_columns"])} (listed below) |

## Training data and its limits

| | |
|---|---|
| Source | `{ds["path"]}` (sha256 `{ds["sha256_prefix"]}…`) |
| Rows | {ds["rows"]} |
| Holdout rows | {ds["holdout_rows"]} |
| Base approval rate | {ds["approval_rate"]:.1%} |

**{ds["rows"]} rows is a small dataset.** Every number below carries real
sampling error — the cross-validated ROC-AUC ranges from
{cv["roc_auc_ci95"][0]:.3f} to {cv["roc_auc_ci95"][1]:.3f} across folds. Treat
this as a well-engineered demonstration of a decisioning pipeline, not as a
model validated for production credit underwriting.

The data is a public loan-application dataset. It carries whatever historical
bias the original approval decisions contained; a model fit to it reproduces
those patterns, including any that were unfair.

## Performance (holdout, never used for model selection)

| Metric | Value | How to read it |
|---|---|---|
| ROC-AUC | {m["roc_auc"]:.3f} | Ranking quality. 0.5 = coin flip. |
| PR-AUC | {m["pr_auc"]:.3f} | Precision/recall trade-off on the approved class. |
| Brier score | {
            m[
                "brier_score"
            ]:.3f} | Probability accuracy. Lower is better; 0.25 = always guessing 50%. |
| Accuracy @ {m["threshold"]:.2f} | {m["accuracy"]:.3f} | Compare against the {
            ds["approval_rate"]:.1%} always-approve baseline. |
| Precision | {m["precision"]:.3f} | Of applications called approvable, how many were. |
| Recall | {m["recall"]:.3f} | Of approvable applications, how many were found. |

Cross-validated ROC-AUC on the training split: **{cv["roc_auc_mean"]:.3f} ± {
            cv["roc_auc_std"]:.3f}**.

### Confusion matrix at threshold {m["threshold"]:.2f}

| | Predicted decline | Predicted approve |
|---|---|---|
| **Actually declined** | {m["confusion_matrix"]["true_negative"]} | {
            m["confusion_matrix"]["false_positive"]
        } |
| **Actually approved** | {m["confusion_matrix"]["false_negative"]} | {
            m["confusion_matrix"]["true_positive"]
        } |

### Calibration

The application shows applicants a probability, so that probability has to
mean something. Each row compares what the model predicted against what
actually happened in that band.

| Probability bin | n | Mean predicted | Observed approval rate |
|---|---|---|---|
{reliability_rows}

Uncalibrated Brier score: {metrics["holdout_uncalibrated"]["brier_score"]:.4f} ·
Calibrated: {metrics["holdout_calibrated"]["brier_score"]:.4f}.

## Response-surface check

The product answers "what would improve my chances?", so the model has to be
**monotonic in the loan amount**: asking for less must never lower the estimate.
Ranking metrics do not test this. We sweep the requested amount downward across
18 representative applicants and count reversals. A reversal counts as
material when it exceeds {metrics["served_model_monotonicity"]["tolerance"]}
probability — below that it is invisible in anything we display.

| Candidate | Monotonic | Material reversals | Worst drop |
|---|---|---|---|
{
            chr(10).join(
                f"| `{name}` | {'yes' if result['monotonic'] else '**no**'} | "
                f"{result['material_violations']}/{result['comparisons']} | {result['worst_drop']:.3f} |"
                for name, result in metrics["monotonicity"].items()
            )
        }
| **served model** | {
            "yes" if metrics["served_model_monotonicity"]["monotonic"] else "**no**"
        } | {metrics["served_model_monotonicity"]["material_violations"]}/{
            metrics["served_model_monotonicity"]["comparisons"]
        } | {metrics["served_model_monotonicity"]["worst_drop"]:.3f} |

Non-monotonic candidates are disqualified from selection regardless of AUC.
This check was added after a random forest — with a perfectly respectable
0.83 holdout AUC — scored a 70%-smaller loan *worse* than a 50%-smaller one,
which would have produced actively misleading advice.

## Features

The model uses exactly these {len(metrics["feature_columns"])} features, in this order:

{chr(10).join(f"{i + 1}. `{c}`" for i, c in enumerate(metrics["feature_columns"]))}

The order is stored inside the model bundle and re-checked on load. A mismatch
between this list and `loanapp.features.FEATURE_COLUMNS` refuses to serve
rather than predicting on mis-aligned columns.

## What the model deliberately does not use

**Gender and marital status.** ECOA / Regulation B prohibits using them as a
basis for a credit decision. Measured effect of excluding them:
{ablation["roc_auc_with_protected"]:.4f} ROC-AUC with vs.
{ablation["roc_auc_without_protected"]:.4f} without — {_ablation_verdict(ablation)}
The point is that the fair-lending decision is now backed by a number from this
run rather than by assertion.

**CIBIL / credit bureau score.** The training dataset contains no bureau
score. The previous version of this project generated one with
`np.random.randint(300, 850)` and trained on it while the interface presented
it as a real input. The score is now collected and used only in
`loanapp.rules`, where its thresholds are published and auditable.

### Fairness audit

Removing an attribute from the inputs does not guarantee parity in the
outputs — correlated features can reintroduce it. Holdout approval rates by
excluded attribute:

| Attribute | Group | n | Approval rate |
|---|---|---|---|
{fairness_table}

Groups with fewer than 10 holdout rows are omitted as uninformative.

## Explanations

Factor attributions come from a **one-feature-at-a-time ablation**: each
feature is swapped to its training-set baseline and the probability recomputed;
the change is that feature's attribution. It is model-agnostic and exactly
reproducible, and it **ignores interactions between features** — two factors
can each look small while mattering jointly. It is not SHAP and is not
presented as such.

## Intended use and misuse

**Intended:** helping an applicant understand which parts of their application
drive the outcome, and what would plausibly change it; giving engineers a
worked example of a calibrated, explainable decisioning service.

**Not intended:** making or supporting real credit decisions; any use where an
applicant is affected by the output without a human decision-maker; any
jurisdiction or product whose lending patterns differ from the training data.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the loan eligibility model")
    parser.add_argument("--data", default=str(BASE_DIR / "data" / "train.csv"))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--docs", default=str(BASE_DIR / "docs" / "MODEL.md"))
    parser.add_argument("--no-docs", action="store_true")
    args = parser.parse_args()

    config = load_config(os.environ.get("FLASK_ENV", "development"))
    train(
        data_path=args.data,
        model_dir=args.model_dir or config.model_dir,
        docs_path=None if args.no_docs else args.docs,
        annual_rate=config.annual_interest_rate,
        threshold=config.approve_threshold,
    )


if __name__ == "__main__":
    main()
