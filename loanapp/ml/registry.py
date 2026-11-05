"""Model bundle persistence.

A bundle is a single ``.joblib`` file holding the fitted sklearn pipeline plus
the metadata needed to prove it is safe to use: the feature order it was
trained on, the library versions it was pickled under, the training data
fingerprint, the metrics it achieved, and the baseline input values the
explainer needs.

:func:`load_bundle` refuses a bundle that fails either of two contracts.

**The feature contract.** ``feature_columns`` must match
``loanapp.features.FEATURE_COLUMNS``. This is the direct fix for the bug that
made the original version unusable: training and serving each built their own
column list, in different orders, with nothing comparing them.

**The library contract.** scikit-learn pickles are explicitly not portable
across versions. Loading a bundle trained under 1.5 into 1.9 succeeds and then
fails at predict time with ``AttributeError: 'SimpleImputer' object has no
attribute '_fill_dtype'`` — a 500 per request with nothing pointing at the
cause. We record the versions at save time and check them at load, so the same
situation degrades to an actionable 503 instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy
import sklearn

from ..features import FEATURE_COLUMNS

BUNDLE_FILENAME = "loan_model.joblib"

#: Bump when the bundle's own shape changes. Separate from the library check
#: below: this catches *our* format drift, that catches the ecosystem's.
BUNDLE_FORMAT_VERSION = 3


class ModelUnavailable(RuntimeError):
    """No usable model bundle is present."""


class ModelContractError(RuntimeError):
    """A bundle exists but cannot be trusted to serve predictions."""


def _runtime_versions() -> dict[str, str]:
    """Library versions that affect whether a pickled pipeline still works.

    Only scikit-learn and numpy: they own the pickled objects. pandas is used
    to build the input frame, not stored inside the bundle, so a pandas upgrade
    does not invalidate an artifact.
    """
    return {"scikit_learn": sklearn.__version__, "numpy": numpy.__version__}


def _minor(version: str) -> str:
    """``"1.9.1"`` -> ``"1.9"``.

    Local compatibility policy permits patch drift; deployments still use
    exact dependency pins. See docs/ARCHITECTURE.md.
    """
    parts = version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else version


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
    #: Library versions present when the pipeline was pickled. Defaulted at
    #: construction so the trainer cannot forget to record them.
    runtime_versions: dict[str, str] = field(default_factory=_runtime_versions)
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
        "saved_at", datetime.now(UTC).isoformat(timespec="seconds")
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

    current = _runtime_versions()
    drifted = {
        library: (bundle.runtime_versions.get(library, "unknown"), running)
        for library, running in current.items()
        if _minor(bundle.runtime_versions.get(library, "")) != _minor(running)
    }
    if drifted:
        detail = "\n".join(
            f"  {library}: bundle pickled under {was}, running {now}"
            for library, (was, now) in sorted(drifted.items())
        )
        raise ModelContractError(
            "Model/library contract mismatch — refusing to serve predictions.\n"
            f"{detail}\n"
            "scikit-learn pickles are not portable across minor versions. Loading "
            "anyway appears to work and then fails inside predict().\n"
            "Retrain with: python -m loanapp.ml.train"
        )

    return bundle
