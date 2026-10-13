"""Model bundle loading and the feature-contract guard.

The guard is the direct fix for the defect that made the previous version
unusable: training and serving each built their own feature list, in different
orders, with nothing comparing them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import sklearn

from loanapp.features import FEATURE_COLUMNS
from loanapp.ml.registry import (
    BUNDLE_FORMAT_VERSION,
    ModelBundle,
    ModelContractError,
    ModelUnavailable,
    _runtime_versions,
    bundle_path,
    load_bundle,
    save_bundle,
)


class DummyPipeline:
    def predict_proba(self, X):
        return [[0.3, 0.7]] * len(X)


def bundle(**overrides) -> ModelBundle:
    defaults = {
        "pipeline": DummyPipeline(),
        "feature_columns": FEATURE_COLUMNS,
        "baselines": {"loan_amount": 100.0},
        "metrics": {"roc_auc": 0.8},
        "metadata": {"version": "test-1"},
    }
    return ModelBundle(**{**defaults, **overrides})


def test_missing_bundle_says_how_to_create_one(tmp_path):
    with pytest.raises(ModelUnavailable, match=re.escape("loanapp.ml.train")):
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
    save_bundle(bundle(feature_columns=(*FEATURE_COLUMNS, "surprise")), str(tmp_path))
    with pytest.raises(ModelContractError):
        load_bundle(str(tmp_path))


def test_stale_bundle_format_is_rejected(tmp_path):
    save_bundle(bundle(format_version=BUNDLE_FORMAT_VERSION - 1), str(tmp_path))
    with pytest.raises(ModelContractError, match="format"):
        load_bundle(str(tmp_path))


class TestLibraryContract:
    """The second guard: library versions, not just feature columns.

    scikit-learn pickles are not portable across minor versions. A bundle
    trained under 1.5 loads happily into 1.9 and then raises
    ``AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'``
    inside ``predict`` — a 500 on every request with nothing naming the cause.
    This was a real failure, hit while upgrading the dependency set.
    """

    def test_a_minor_version_drift_is_rejected(self, tmp_path):
        stale = bundle()
        stale.runtime_versions = {"scikit_learn": "1.5.0", "numpy": "1.26.4"}
        save_bundle(stale, str(tmp_path))
        with pytest.raises(ModelContractError, match="library contract"):
            load_bundle(str(tmp_path))

    def test_the_message_names_both_versions_and_the_remedy(self, tmp_path):
        stale = bundle()
        stale.runtime_versions = {**_runtime_versions(), "scikit_learn": "1.5.0"}
        save_bundle(stale, str(tmp_path))
        with pytest.raises(ModelContractError) as caught:
            load_bundle(str(tmp_path))
        message = str(caught.value)
        assert "1.5.0" in message
        assert sklearn.__version__ in message
        assert "loanapp.ml.train" in message

    def test_a_patch_version_drift_is_tolerated(self, tmp_path):
        """Refusing across a patch release would turn a security patch into an
        outage. Patch releases do not change pickle layout."""
        major_minor = ".".join(sklearn.__version__.split(".")[:2])
        tolerated = bundle()
        tolerated.runtime_versions = {
            **_runtime_versions(),
            "scikit_learn": f"{major_minor}.99",
        }
        save_bundle(tolerated, str(tmp_path))
        assert load_bundle(str(tmp_path)).version == "test-1"

    def test_versions_are_recorded_without_the_trainer_asking(self):
        """Defaulted at construction, so a new trainer cannot forget."""
        assert bundle().runtime_versions["scikit_learn"] == sklearn.__version__

    def test_a_bundle_predates_version_recording(self, tmp_path):
        """A v3 bundle with no recorded versions is treated as drifted rather
        than assumed compatible."""
        unrecorded = bundle()
        unrecorded.runtime_versions = {}
        save_bundle(unrecorded, str(tmp_path))
        with pytest.raises(ModelContractError, match="unknown"):
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


class TestCorruptBundle:
    """``save_bundle`` is not atomic, so an interrupted build leaves a
    truncated pickle. The loader must degrade, not crash-loop every worker."""

    def _app(self, tmp_path, model_dir):
        from loanapp import create_app
        from loanapp.config import Config

        return create_app(
            Config(
                env="testing",
                secret_key="k" * 40,
                database_path=str(tmp_path / "t.db"),
                model_dir=str(model_dir),
                cookie_secure=False,
            )
        )

    def test_truncated_bundle_degrades_instead_of_raising(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        save_bundle(bundle(), str(model_dir))

        path = bundle_path(str(model_dir))
        intact = Path(path).read_bytes()
        Path(path).write_bytes(intact[: len(intact) // 2])  # interrupted write

        app = self._app(tmp_path, model_dir)
        assert app.extensions["model_status"] == "load_failed"
        assert "retrain" in app.extensions["model_error"]

        # Everything that does not need the model still works...
        client = app.test_client()
        assert client.get("/").status_code == 200
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.get_json()["status"] == "degraded"
        # ...and assessment fails with an actionable 503, not a 500.
        assert (
            client.post(
                "/api/assess",
                json={
                    "applicant_income": 1,
                    "loan_amount": 1,
                    "loan_term_months": 12,
                    "credit_history": 1,
                },
            ).status_code
            == 503
        )

    def test_garbage_bundle_degrades(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        Path(bundle_path(str(model_dir))).write_bytes(b"not a pickle at all")
        assert self._app(tmp_path, model_dir).extensions["model_status"] == "load_failed"
