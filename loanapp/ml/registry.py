"""Model bundle persistence.

A bundle is a single ``.joblib`` file holding the fitted sklearn pipeline plus
the metadata needed to prove it is safe to use: the feature order it was
trained on, the training data fingerprint, the metrics it achieved, and the
baseline feature values the explainer needs.

:func:`load_bundle` refuses a model whose ``feature_columns`` disagree with
``loanapp.features.FEATURE_COLUMNS``. That check is the direct fix for the bug
that made the previous version unusable: training and serving each built their
own column list, in different orders, with nothing comparing them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import joblib

from ..features import FEATURE_COLUMNS

BUNDLE_FILENAME = "loan_model.joblib"
BUNDLE_FORMAT_VERSION = 2


class ModelUnavailable(RuntimeError):
    """No usable model bundle is present."""


class ModelContractError(RuntimeError):
    """A bundle exists but does not match the current feature contract."""


@dataclass
class ModelBundle:
    """A fitted pipeline plus everything needed to use it honestly."""

    pipeline: Any
    feature_columns: tuple[str, ...]
    #: Median (numeric) / mode (categorical) of each *applicant-visible input*
    #: in the training set. The explainer swaps one input to its baseline and
    #: re-scores the whole application to measure that input's influence.
    baselines: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    format_version: int = BUNDLE_FORMAT_VERSION

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unknown"))

    @property
    def is_calibrated(self) -> bool:
        return bool(self.metadata.get("calibrated", False))

    @property
    def trained_at(self) -> str:
        return str(self.metadata.get("trained_at", "unknown"))


def bundle_path(model_dir: str) -> str:
    return os.path.join(model_dir, BUNDLE_FILENAME)


def save_bundle(bundle: ModelBundle, model_dir: str) -> str:
    os.makedirs(model_dir, exist_ok=True)
    bundle.metadata.setdefault(
        "saved_at", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    path = bundle_path(model_dir)
    joblib.dump(bundle, path)
    return path


def load_bundle(model_dir: str) -> ModelBundle:
    """Load and validate the bundle in ``model_dir``.

    Raises :class:`ModelUnavailable` if there is nothing to load and
    :class:`ModelContractError` if what is there cannot be trusted. Callers
    treat the two differently: the first is a "run the trainer" message, the
    second is a deployment error.
    """
    path = bundle_path(model_dir)
    if not os.path.exists(path):
        raise ModelUnavailable(
            f"No model bundle at {path}. Train one with: python -m loanapp.ml.train"
        )

    # joblib.load executes pickled code. Only ever point MODEL_DIR at artifacts
    # you produced yourself or fetched over a channel you trust.
    bundle = joblib.load(path)

    if not isinstance(bundle, ModelBundle):
        raise ModelContractError(
            f"{path} is not a ModelBundle (got {type(bundle).__name__}). It was "
            "probably produced by an older version; retrain with "
            "python -m loanapp.ml.train"
        )
    if bundle.format_version != BUNDLE_FORMAT_VERSION:
        raise ModelContractError(
            f"{path} uses bundle format v{bundle.format_version}, this build "
            f"expects v{BUNDLE_FORMAT_VERSION}. Retrain the model."
        )
    if tuple(bundle.feature_columns) != FEATURE_COLUMNS:
        raise ModelContractError(
            "Model/feature contract mismatch — refusing to serve predictions.\n"
            f"  bundle expects: {list(bundle.feature_columns)}\n"
            f"  code provides:  {list(FEATURE_COLUMNS)}\n"
            "Retrain with: python -m loanapp.ml.train"
        )
    return bundle
