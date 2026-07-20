"""Gradient-boosted ablation over the private-prototype game rows.

Fits a LightGBM binary classifier on the same training rows as the logistic
Elo-residual model, with ``logit(elo_home_probability)`` prepended to the
feature vector. Elo enters as an ordinary feature, not as an offset: boosting
is free to learn its own calibration of the Elo signal against the tabular
features. LightGBM is imported lazily inside :func:`fit_gbm` so the core
package stays importable without the ``gbm`` extra. Training uses a fixed
seed, deterministic mode, and a pinned thread count so repeated runs
reproduce identical boosters.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, cast

from forecastfm.elo_residual import probability_logit
from forecastfm.nba_prototype_dataset import PrototypeGameRow, feature_names

if TYPE_CHECKING:
    import lightgbm as lgb
    import numpy as np
    import numpy.typing as npt


class NbaGbmError(ValueError):
    """Raised when GBM training data or settings are invalid."""


def _require_unit_interval(name: str, value: float) -> None:
    if not isfinite(value) or not 0.0 < value <= 1.0:
        raise NbaGbmError(f"{name} must be finite and in (0, 1]")


@dataclass(frozen=True, slots=True)
class GbmParams:
    """LightGBM hyperparameters for the prototype ablation."""

    n_estimators: int = 400
    learning_rate: float = 0.03
    max_depth: int = 3
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    seed: int = 20260719

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise NbaGbmError("n_estimators must be at least one")
        if self.max_depth < 1:
            raise NbaGbmError("max_depth must be at least one")
        if self.min_child_samples < 1:
            raise NbaGbmError("min_child_samples must be at least one")
        if self.seed < 0:
            raise NbaGbmError("seed must be non-negative")
        _require_unit_interval("learning_rate", self.learning_rate)
        _require_unit_interval("subsample", self.subsample)
        _require_unit_interval("colsample_bytree", self.colsample_bytree)
        if not isfinite(self.reg_lambda) or self.reg_lambda < 0.0:
            raise NbaGbmError("reg_lambda must be non-negative and finite")


DEFAULT_GBM_PARAMS = GbmParams()


@dataclass(frozen=True, slots=True)
class GbmModel:
    """A fitted LightGBM booster with its training-time feature layout."""

    feature_names: tuple[str, ...]
    booster: lgb.Booster

    def probability(self, row: PrototypeGameRow, *, include_health: bool) -> float:
        """Return the boosted home-win probability for one prototype row."""
        import numpy as np  # noqa: PLC0415 -- lazy import keeps the extra optional

        vector = _feature_vector(row, include_health=include_health)
        if len(vector) != len(self.feature_names):
            raise NbaGbmError("row feature count differs from the fitted model")
        raw = self.booster.predict(  # pyright: ignore[reportUnknownMemberType]
            np.asarray([vector], dtype=np.float64)
        )
        value = float(cast("npt.NDArray[np.float64]", raw)[0])
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise NbaGbmError("booster returned an invalid probability")
        return value


def fit_gbm(
    rows: list[PrototypeGameRow],
    *,
    include_health: bool,
    params: GbmParams,
) -> GbmModel:
    """Fit a LightGBM binary classifier on prototype rows plus Elo log-odds."""
    import lightgbm as lgb  # noqa: PLC0415 -- lazy import keeps the extra optional
    import numpy as np  # noqa: PLC0415 -- lazy import keeps the extra optional

    if not rows:
        raise NbaGbmError("at least one training row is required")
    names = ("elo_logit", *feature_names(include_health=include_health))
    matrix = np.asarray(
        [_feature_vector(row, include_health=include_health) for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray([1 if row.home_won else 0 for row in rows], dtype=np.int32)
    dataset = lgb.Dataset(matrix, label=labels, feature_name=list(names))
    booster = lgb.train(  # pyright: ignore[reportUnknownMemberType]
        {
            "objective": "binary",
            "learning_rate": params.learning_rate,
            "max_depth": params.max_depth,
            "min_child_samples": params.min_child_samples,
            "subsample": params.subsample,
            "colsample_bytree": params.colsample_bytree,
            "lambda_l2": params.reg_lambda,
            "seed": params.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 4,
            "verbosity": -1,
        },
        dataset,
        num_boost_round=params.n_estimators,
    )
    return GbmModel(feature_names=names, booster=booster)


def _feature_vector(row: PrototypeGameRow, *, include_health: bool) -> tuple[float, ...]:
    features = row.features_standard
    if include_health:
        if row.features_health is None:
            raise NbaGbmError("health features are missing for a row in the health variant")
        features = features + row.features_health
    return (probability_logit(row.elo_home_probability), *features)
