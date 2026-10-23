"""Training pipeline components.

Full training takes ~30s, so these tests exercise the pieces individually
rather than the whole run. The end-to-end run is exercised by the ``train``
job in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loanapp.features import EXPLAINABLE_INPUTS, FEATURE_COLUMNS
from loanapp.ml.train import (
    _ablation_verdict,
    candidate_pipelines,
    compute_baselines,
    evaluate,
    fairness_audit,
    load_dataset,
    monotonicity_check,
    reliability_table,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "train.csv"
RATE = 0.105


@pytest.fixture(scope="module")
def dataset():
    if not DATA.exists():
        pytest.skip("data/train.csv is not present")
    return load_dataset(str(DATA), RATE)


class TestDataset:
    def test_features_use_the_declared_contract(self, dataset):
        X, _, _, _ = dataset
        assert tuple(X.columns) == FEATURE_COLUMNS

    def test_target_is_binary(self, dataset):
        _, y, _, _ = dataset
        assert set(y.unique()) <= {0, 1}

    def test_rows_align_across_every_returned_object(self, dataset):
        X, y, raw, applications = dataset
        assert len(X) == len(y) == len(raw) == len(applications)

    def test_gender_and_marital_status_are_not_features(self, dataset):
        """The fair-lending guarantee, asserted rather than assumed."""
        X, _, raw, _ = dataset
        assert "Gender" in raw.columns  # present in the source data...
        columns = " ".join(X.columns).lower()
        assert "gender" not in columns  # ...and absent from the model's view
        assert "married" not in columns
        assert "marital" not in columns

    def test_no_fabricated_credit_score_reaches_the_model(self, dataset):
        """The previous trainer injected np.random.randint(300, 850) as a
        feature and the UI advertised it as real."""
        X, _, _, _ = dataset
        assert not any("cibil" in c.lower() for c in X.columns)


class TestBaselines:
    def test_covers_every_explainable_input(self, dataset):
        _, _, _, applications = dataset
        baselines = compute_baselines(applications)
        assert set(baselines) == set(EXPLAINABLE_INPUTS)

    def test_categorical_baselines_are_valid_values(self, dataset):
        _, _, _, applications = dataset
        baselines = compute_baselines(applications)
        assert baselines["education"] in ("graduate", "not_graduate")
        assert baselines["property_area"] in ("rural", "semiurban", "urban")
        assert baselines["credit_history"] in (0, 1)
        assert isinstance(baselines["self_employed"], bool)

    def test_numeric_baselines_are_positive(self, dataset):
        _, _, _, applications = dataset
        baselines = compute_baselines(applications)
        assert baselines["applicant_income"] > 0
        assert baselines["loan_amount"] > 0


class TestMonotonicity:
    def test_the_selected_linear_model_is_monotonic_in_loan_amount(self, dataset):
        """The screen that disqualified the random forest.

        A model can post a fine ROC-AUC and still score a smaller loan worse,
        which would make the guidance feature give actively wrong advice.
        """
        X, y, _, _ = dataset
        pipeline = candidate_pipelines()["logistic_regression"]
        pipeline.fit(X, y)
        assert monotonicity_check(pipeline, RATE)["monotonic"]

    def test_the_check_detects_a_deliberately_inverted_model(self):
        class Inverted:
            """Scores larger loans better — exactly what we screen for."""

            def predict_proba(self, X):
                ratio = X["loan_to_income_ratio"].to_numpy()
                p = np.clip(ratio / (ratio.max() or 1), 0.01, 0.99)
                return np.column_stack([1 - p, p])

        result = monotonicity_check(Inverted(), RATE)
        assert not result["monotonic"]
        assert result["material_violations"] > 0


class TestMetrics:
    def test_evaluate_reports_every_headline_metric(self):
        y = np.array([0, 0, 1, 1, 1, 0, 1, 1])
        probs = np.array([0.1, 0.3, 0.8, 0.9, 0.7, 0.2, 0.6, 0.85])
        result = evaluate(y, probs, 0.5)
        for key in ("roc_auc", "pr_auc", "brier_score", "accuracy", "confusion_matrix"):
            assert key in result

    def test_confusion_matrix_totals_match_the_sample_size(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        probs = np.array([0.2, 0.9, 0.4, 0.7, 0.8, 0.1])
        cm = evaluate(y, probs, 0.5)["confusion_matrix"]
        assert sum(cm.values()) == len(y)

    def test_a_perfect_ranking_scores_auc_one(self):
        y = np.array([0, 0, 1, 1])
        assert evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)["roc_auc"] == 1.0

    def test_reliability_table_bins_the_full_probability_range(self):
        y = np.array([0, 1, 1, 0, 1, 1, 0, 1])
        probs = np.array([0.05, 0.25, 0.45, 0.35, 0.65, 0.75, 0.15, 0.95])
        rows = reliability_table(y, probs, bins=5)
        assert len(rows) == 5
        assert sum(r["n"] for r in rows) == len(y)

    def test_reliability_table_handles_an_empty_bin(self):
        y = np.array([1, 1, 0])
        rows = reliability_table(y, np.array([0.9, 0.95, 0.85]), bins=5)
        assert any(r["n"] == 0 for r in rows)


class TestFairnessAudit:
    def test_reports_approval_rates_per_group(self):
        raw = pd.DataFrame({"Gender": ["Male"] * 20 + ["Female"] * 20})
        probs = np.array([0.9] * 20 + [0.2] * 20)
        audit = fairness_audit(raw, probs, 0.5)
        assert audit["Gender"]["Male"]["approval_rate"] == 1.0
        assert audit["Gender"]["Female"]["approval_rate"] == 0.0
        assert audit["Gender"]["max_gap"] == 1.0

    def test_groups_too_small_to_interpret_are_omitted(self):
        raw = pd.DataFrame({"Gender": ["Male"] * 20 + ["Female"] * 3})
        audit = fairness_audit(raw, np.full(23, 0.9), 0.5)
        assert "Female" not in audit["Gender"]


class TestAblationVerdict:
    def test_phrases_a_real_cost_as_a_cost(self):
        assert "costs" in _ablation_verdict({"cost_of_exclusion": 0.02})

    def test_phrases_an_improvement_as_an_improvement(self):
        assert "improves" in _ablation_verdict({"cost_of_exclusion": -0.02})

    def test_phrases_a_negligible_difference_as_noise(self):
        assert "noise" in _ablation_verdict({"cost_of_exclusion": 0.001})


def test_calibration_selection_only_receives_training_rows(
    dataset, tmp_path, monkeypatch
):
    """Changing the holdout's calibration ranking must not select the model."""
    from sklearn.model_selection import train_test_split

    import loanapp.ml.train as training

    X, y, _, _ = dataset
    expected, holdout = train_test_split(
        X, test_size=0.2, random_state=training.RANDOM_STATE, stratify=y
    )
    seen = []

    def select(pipeline, features, labels):
        seen.append(set(features.index))
        assert set(features.index) == set(expected.index)
        assert set(features.index).isdisjoint(holdout.index)
        assert set(labels.index) == set(features.index)
        # Deliberately select the opposite of this dataset's holdout ranking.
        return {"use_calibrated": False, "method": "training_cross_validation"}

    monkeypatch.setattr(training, "select_calibration", select, raising=False)
    monkeypatch.setattr(
        training,
        "cross_validate",
        lambda *args: {
            "roc_auc_mean": 0.75,
            "roc_auc_std": 0.05,
        },
    )
    monkeypatch.setattr(
        training,
        "candidate_pipelines",
        lambda: {
            "logistic_regression": candidate_pipelines()["logistic_regression"],
        },
    )
    monkeypatch.setattr(
        training,
        "monotonicity_check",
        lambda *args: {
            "monotonic": True,
            "material_violations": 0,
            "violations": 0,
            "comparisons": 216,
            "worst_drop": 0.0,
        },
    )
    monkeypatch.setattr(training, "ablation_cost", lambda *args: {})
    metrics = training.train(str(DATA), str(tmp_path), None, RATE, 0.65)
    assert seen, "calibration was selected from holdout metrics instead of training CV"
    assert metrics["calibrated"] is False


@pytest.mark.parametrize("calibrated_brier,expected", [(0.12, True), (0.22, False)])
def test_calibration_is_selected_by_cross_validated_brier(
    dataset, monkeypatch, calibrated_brier, expected
):
    import loanapp.ml.train as training

    X, y, _, _ = dataset
    calls = []

    def score(estimator, features, labels, *, cv, scoring, n_jobs):
        calls.append((type(estimator).__name__, cv, scoring, list(features.index)))
        brier = 0.18 if len(calls) == 1 else calibrated_brier
        return np.full(5, -brier)

    monkeypatch.setattr(training, "cross_val_score", score)
    selection = training.select_calibration(
        candidate_pipelines()["logistic_regression"], X, y
    )
    assert selection["use_calibrated"] is expected
    assert len(calls) == 2
    assert all(call[2] == "neg_brier_score" for call in calls)
    assert calls[0][1] is calls[1][1]  # identical outer folds for a paired comparison


def test_generated_model_card_is_utf8(tmp_path):
    from loanapp.ml.registry import load_bundle
    from loanapp.ml.train import write_model_card

    root = DATA.parent.parent
    metrics = json.loads((root / "models" / "metrics.json").read_text(encoding="utf-8"))
    target = tmp_path / "MODEL.md"
    write_model_card(metrics, load_bundle(str(root / "models")), str(target))
    text = target.read_text(encoding="utf-8")
    assert "±" in text
    assert "—" in text
