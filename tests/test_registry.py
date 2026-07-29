"""Model bundle loading and the feature-contract guard.

The guard is the direct fix for the defect that made the previous version
unusable: training and serving each built their own feature list, in different
orders, with nothing comparing them.
"""

from __future__ import annotations

import pytest

from loanapp.features import FEATURE_COLUMNS
from loanapp.ml.registry import (
    BUNDLE_FORMAT_VERSION,
    ModelBundle,
    ModelContractError,
    ModelUnavailable,
    load_bundle,
    save_bundle,
)


class DummyPipeline:
    def predict_proba(self, X):
        return [[0.3, 0.7]] * len(X)


def bundle(**overrides) -> ModelBundle:
    defaults = dict(
        pipeline=DummyPipeline(),
        feature_columns=FEATURE_COLUMNS,
        baselines={"loan_amount": 100.0},
        metrics={"roc_auc": 0.8},
        metadata={"version": "test-1"},
    )
    return ModelBundle(**{**defaults, **overrides})


def test_missing_bundle_says_how_to_create_one(tmp_path):
    with pytest.raises(ModelUnavailable, match="loanapp.ml.train"):
        load_bundle(str(tmp_path))


def test_round_trip_preserves_metadata(tmp_path):
    save_bundle(bundle(), str(tmp_path))
    loaded = load_bundle(str(tmp_path))
    assert loaded.version == "test-1"
    assert loaded.metrics["roc_auc"] == 0.8
    assert tuple(loaded.feature_columns) == FEATURE_COLUMNS


def test_reordered_columns_are_rejected(tmp_path):
    """The original bug: same column names, different order.

    scikit-learn would raise deep inside transform, or worse, silently score a
    mis-aligned vector. We refuse at load time with a message that says what to
    do about it.
    """
    save_bundle(bundle(feature_columns=tuple(reversed(FEATURE_COLUMNS))), str(tmp_path))
    with pytest.raises(ModelContractError, match="mismatch"):
        load_bundle(str(tmp_path))


def test_missing_column_is_rejected(tmp_path):
    save_bundle(bundle(feature_columns=FEATURE_COLUMNS[:-1]), str(tmp_path))
    with pytest.raises(ModelContractError):
        load_bundle(str(tmp_path))


def test_extra_column_is_rejected(tmp_path):
    save_bundle(bundle(feature_columns=FEATURE_COLUMNS + ("surprise",)), str(tmp_path))
    with pytest.raises(ModelContractError):
        load_bundle(str(tmp_path))


def test_stale_bundle_format_is_rejected(tmp_path):
    save_bundle(bundle(format_version=BUNDLE_FORMAT_VERSION - 1), str(tmp_path))
    with pytest.raises(ModelContractError, match="format"):
        load_bundle(str(tmp_path))


def test_a_bare_pickled_estimator_is_rejected(tmp_path):
    """The previous version pickled a raw estimator with no metadata at all,
    so there was nothing to validate against."""
    import joblib

    from loanapp.ml.registry import bundle_path

    joblib.dump(DummyPipeline(), bundle_path(str(tmp_path)))
    with pytest.raises(ModelContractError, match="ModelBundle"):
        load_bundle(str(tmp_path))


def test_app_starts_and_stays_useful_without_a_model(modelless_app):
    assert modelless_app.extensions["assessor"] is None
    assert modelless_app.extensions["model_status"] == "unavailable"
    assert modelless_app.test_client().get("/").status_code == 200


def test_app_refuses_to_serve_a_contract_mismatched_model(tmp_path):
    """A stale bundle is a deployment error. Serving mis-aligned predictions is
    exactly the failure this design exists to prevent, so we degrade rather
    than guess."""
    from loanapp import create_app
    from loanapp.config import Config

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    save_bundle(bundle(feature_columns=("only_one",)), str(model_dir))

    app = create_app(
        Config(
            env="testing",
            secret_key="k" * 40,
            database_path=str(tmp_path / "t.db"),
            model_dir=str(model_dir),
            cookie_secure=False,
        )
    )
    assert app.extensions["model_status"] == "contract_mismatch"
    response = app.test_client().post(
        "/api/assess",
        json={
            "applicant_income": 1,
            "loan_amount": 1,
            "loan_term_months": 12,
            "credit_history": 1,
        },
    )
    assert response.status_code == 503


def test_the_shipped_bundle_matches_the_current_contract(trained_model_dir):
    """Guards against committing code that changes FEATURE_COLUMNS without
    retraining."""
    assert tuple(load_bundle(trained_model_dir).feature_columns) == FEATURE_COLUMNS
