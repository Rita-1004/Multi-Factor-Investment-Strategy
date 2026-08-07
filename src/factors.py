"""Factor construction and cross-sectional transformation utilities."""

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.data import validate_required_columns


def winsorize_series(
    values: pd.Series,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.Series:
    """Winsorize a numeric series at the selected quantiles."""

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    valid_values = numeric_values.dropna()

    if valid_values.empty:
        return numeric_values

    lower_bound = valid_values.quantile(lower_quantile)
    upper_bound = valid_values.quantile(upper_quantile)

    return numeric_values.clip(
        lower=lower_bound,
        upper=upper_bound,
    )


def zscore_series(
    values: pd.Series,
) -> pd.Series:
    """Standardize a numeric series to a cross-sectional z-score."""

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    mean_value = numeric_values.mean()
    standard_deviation = numeric_values.std(ddof=0)

    if (
        pd.isna(standard_deviation)
        or standard_deviation == 0
    ):
        result = pd.Series(
            np.nan,
            index=values.index,
            dtype=float,
        )

        result.loc[numeric_values.notna()] = 0.0
        return result

    return (
        numeric_values - mean_value
    ) / standard_deviation


def winsorize_by_period(
    data: pd.DataFrame,
    value_columns: Sequence[str],
    period_column: str = "month",
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    suffix: str = "_winsorized",
) -> pd.DataFrame:
    """Winsorize factor values within each cross-sectional period."""

    validate_required_columns(
        data=data,
        required_columns=[
            period_column,
            *value_columns,
        ],
        dataset_name="factor dataset",
    )

    result = data.copy()

    for column in value_columns:
        output_column = "{}{}".format(
            column,
            suffix,
        )

        result[output_column] = (
            result.groupby(
                period_column,
                sort=False,
            )[column]
            .transform(
                lambda values: winsorize_series(
                    values=values,
                    lower_quantile=lower_quantile,
                    upper_quantile=upper_quantile,
                )
            )
        )

    return result


def zscore_by_period(
    data: pd.DataFrame,
    value_columns: Sequence[str],
    period_column: str = "month",
    suffix: str = "_zscore",
) -> pd.DataFrame:
    """Calculate cross-sectional z-scores within each period."""

    validate_required_columns(
        data=data,
        required_columns=[
            period_column,
            *value_columns,
        ],
        dataset_name="factor dataset",
    )

    result = data.copy()

    for column in value_columns:
        output_column = "{}{}".format(
            column,
            suffix,
        )

        result[output_column] = (
            result.groupby(
                period_column,
                sort=False,
            )[column]
            .transform(zscore_series)
        )

    return result


def construct_oriented_factor_scores(
    data: pd.DataFrame,
    factor_mapping: Mapping[str, str],
    factor_directions: Optional[Mapping[str, float]] = None,
    period_column: str = "month",
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Construct winsorized and standardized factor scores.

    Parameters
    ----------
    data:
        Security-period panel.
    factor_mapping:
        Mapping from raw variable names to output score names.

        Example:

        {
            "book_to_market": "factor_value",
            "momentum_12_1": "factor_momentum",
        }

    factor_directions:
        Direction applied to each raw factor. A direction of 1 means
        that a higher value is preferred. A direction of -1 means that
        a lower value is preferred.
    """

    raw_columns = list(factor_mapping.keys())

    validate_required_columns(
        data=data,
        required_columns=[
            period_column,
            *raw_columns,
        ],
        dataset_name="raw factor dataset",
    )

    if factor_directions is None:
        factor_directions = {
            column: 1.0
            for column in raw_columns
        }

    result = data.copy()

    for raw_column, output_column in factor_mapping.items():
        direction = float(
            factor_directions.get(
                raw_column,
                1.0,
            )
        )

        result[output_column] = (
            result.groupby(
                period_column,
                sort=False,
            )[raw_column]
            .transform(
                lambda values: zscore_series(
                    winsorize_series(
                        values=values,
                        lower_quantile=lower_quantile,
                        upper_quantile=upper_quantile,
                    )
                )
            )
            * direction
        )

    return result


def combine_factor_scores(
    data: pd.DataFrame,
    factor_columns: Sequence[str],
    output_column: str = "multi_factor_score",
    factor_weights: Optional[Mapping[str, float]] = None,
    minimum_available_factors: int = 1,
    period_column: str = "month",
    standardize_composite: bool = True,
) -> pd.DataFrame:
    """
    Combine several oriented factor scores.

    Missing factors are handled by re-scaling the available weights
    for each observation. Observations with too few available factors
    receive a missing composite score.
    """

    validate_required_columns(
        data=data,
        required_columns=[
            period_column,
            *factor_columns,
        ],
        dataset_name="factor-score dataset",
    )

    if minimum_available_factors < 1:
        raise ValueError(
            "minimum_available_factors must be at least one."
        )

    if minimum_available_factors > len(factor_columns):
        raise ValueError(
            "minimum_available_factors cannot exceed the number of factors."
        )

    if factor_weights is None:
        factor_weights = {
            column: 1.0
            for column in factor_columns
        }

    weights = pd.Series(
        {
            column: float(
                factor_weights.get(
                    column,
                    0.0,
                )
            )
            for column in factor_columns
        }
    )

    if (weights < 0).any():
        raise ValueError(
            "Composite-score weights cannot be negative."
        )

    if weights.sum() == 0:
        raise ValueError(
            "At least one composite-score weight must be positive."
        )

    result = data.copy()
    factor_values = result[list(factor_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )

    available_indicator = factor_values.notna().astype(float)

    weighted_numerator = factor_values.fillna(0.0).mul(
        weights,
        axis=1,
    ).sum(axis=1)

    available_weight = available_indicator.mul(
        weights,
        axis=1,
    ).sum(axis=1)

    available_factor_count = factor_values.notna().sum(axis=1)

    composite_score = weighted_numerator.div(
        available_weight.replace(0, np.nan)
    )

    composite_score.loc[
        available_factor_count < minimum_available_factors
    ] = np.nan

    result[output_column] = composite_score
    result[
        "{}_available_factor_count".format(output_column)
    ] = available_factor_count

    if standardize_composite:
        result[output_column] = (
            result.groupby(
                period_column,
                sort=False,
            )[output_column]
            .transform(zscore_series)
        )

    return result


def assign_cross_sectional_quantiles(
    data: pd.DataFrame,
    score_column: str,
    period_column: str = "month",
    number_of_quantiles: int = 5,
    output_column: str = "quantile",
) -> pd.DataFrame:
    """Assign securities to cross-sectional factor quantiles."""

    validate_required_columns(
        data=data,
        required_columns=[
            period_column,
            score_column,
        ],
        dataset_name="factor dataset",
    )

    if number_of_quantiles < 2:
        raise ValueError(
            "number_of_quantiles must be at least two."
        )

    result = data.copy()

    def assign_one_period(values):
        numeric_values = pd.to_numeric(
            values,
            errors="coerce",
        )

        percentile_ranks = numeric_values.rank(
            method="first",
            pct=True,
        )

        quantiles = np.ceil(
            percentile_ranks * number_of_quantiles
        )

        quantiles = quantiles.clip(
            lower=1,
            upper=number_of_quantiles,
        )

        return quantiles.astype("Int64")

    result[output_column] = (
        result.groupby(
            period_column,
            sort=False,
        )[score_column]
        .transform(assign_one_period)
    )

    return result


def calculate_forward_return(
    data: pd.DataFrame,
    entity_column: str = "permno",
    period_column: str = "month",
    return_column: str = "monthly_return",
    output_column: str = "future_return_1m",
) -> pd.DataFrame:
    """Create a one-period-ahead return without using future information."""

    validate_required_columns(
        data=data,
        required_columns=[
            entity_column,
            period_column,
            return_column,
        ],
        dataset_name="monthly return panel",
    )

    result = data.copy()

    result[period_column] = pd.to_datetime(
        result[period_column],
        errors="coerce",
    )

    result = result.sort_values(
        [
            entity_column,
            period_column,
        ]
    )

    result[output_column] = (
        result.groupby(
            entity_column,
            sort=False,
        )[return_column]
        .shift(-1)
    )

    return result