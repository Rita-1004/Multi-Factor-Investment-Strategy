"""Basic unit tests for the multi-factor research utilities."""

import numpy as np
import pandas as pd

from src.data import (
    compound_simple_returns,
    convert_to_month_end,
    validate_unique_keys,
)
from src.evaluation import (
    calculate_performance_statistics,
    holm_adjust_p_values,
)
from src.factors import (
    assign_cross_sectional_quantiles,
    combine_factor_scores,
    zscore_series,
)
from src.ml import build_ridge_model
from src.portfolio import (
    calculate_target_weights,
    calculate_turnover,
)
from src.risk import (
    historical_var_cvar,
    kupiec_unconditional_coverage_test,
)


def test_compound_simple_returns():
    returns = pd.Series(
        [
            0.10,
            -0.10,
        ]
    )

    result = compound_simple_returns(
        returns
    )

    assert np.isclose(
        result,
        -0.01,
    )


def test_convert_to_month_end():
    dates = pd.Series(
        pd.to_datetime(
            [
                "2025-01-03",
                "2025-02-15",
            ]
        )
    )

    result = convert_to_month_end(
        dates
    )

    assert result.iloc[0] == pd.Timestamp(
        "2025-01-31"
    )

    assert result.iloc[1] == pd.Timestamp(
        "2025-02-28"
    )


def test_validate_unique_keys():
    data = pd.DataFrame(
        {
            "permno": [
                1,
                2,
            ],
            "month": pd.to_datetime(
                [
                    "2025-01-31",
                    "2025-01-31",
                ]
            ),
        }
    )

    validate_unique_keys(
        data=data,
        key_columns=[
            "permno",
            "month",
        ],
    )


def test_zscore_series():
    values = pd.Series(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    standardized = zscore_series(
        values
    )

    assert np.isclose(
        standardized.mean(),
        0.0,
    )

    assert np.isclose(
        standardized.std(ddof=0),
        1.0,
    )


def test_combine_factor_scores():
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2025-01-31",
                    "2025-01-31",
                    "2025-01-31",
                    "2025-01-31",
                ]
            ),
            "factor_value": [
                -1.0,
                0.0,
                1.0,
                2.0,
            ],
            "factor_quality": [
                -2.0,
                -1.0,
                1.0,
                2.0,
            ],
        }
    )

    result = combine_factor_scores(
        data=data,
        factor_columns=[
            "factor_value",
            "factor_quality",
        ],
        minimum_available_factors=2,
    )

    assert result[
        "multi_factor_score"
    ].notna().all()

    assert np.isclose(
        result[
            "multi_factor_score"
        ].mean(),
        0.0,
    )


def test_cross_sectional_quantiles():
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(
                ["2025-01-31"] * 10
            ),
            "score": np.arange(
                1,
                11,
            ),
        }
    )

    result = assign_cross_sectional_quantiles(
        data=data,
        score_column="score",
        number_of_quantiles=5,
    )

    assert result["quantile"].min() == 1
    assert result["quantile"].max() == 5


def test_equal_weights_sum_to_one():
    selected_data = pd.DataFrame(
        {
            "permno": [
                1,
                2,
                3,
                4,
            ],
            "score": [
                4.0,
                3.0,
                2.0,
                1.0,
            ],
        }
    )

    result = calculate_target_weights(
        selected_data=selected_data,
        weighting_method="Equal Weight",
        score_column="score",
    )

    assert np.isclose(
        result["target_weight"].sum(),
        1.0,
    )

    assert np.allclose(
        result["target_weight"],
        0.25,
    )


def test_turnover():
    old_weights = pd.Series(
        {
            1: 0.50,
            2: 0.50,
        }
    )

    new_weights = pd.Series(
        {
            1: 0.25,
            2: 0.25,
            3: 0.50,
        }
    )

    turnover = calculate_turnover(
        old_weights=old_weights,
        new_weights=new_weights,
    )

    assert np.isclose(
        turnover,
        0.50,
    )


def test_performance_statistics():
    returns = pd.Series(
        [
            0.01,
            0.02,
            -0.01,
            0.03,
            0.01,
            -0.02,
        ]
    )

    statistics = (
        calculate_performance_statistics(
            returns=returns
        )
    )

    assert (
        statistics[
            "number_of_periods"
        ]
        == 6
    )

    assert statistics[
        "terminal_wealth"
    ] > 1.0

    assert statistics[
        "maximum_drawdown"
    ] <= 0.0


def test_holm_adjustment():
    p_values = pd.Series(
        [
            0.01,
            0.04,
            0.20,
        ]
    )

    adjusted = holm_adjust_p_values(
        p_values
    )

    assert np.all(
        adjusted.dropna()
        >= p_values.loc[
            adjusted.dropna().index
        ]
    )

    assert np.all(
        adjusted.dropna()
        <= 1.0
    )


def test_historical_var_cvar():
    returns = pd.Series(
        [
            -0.10,
            -0.05,
            -0.02,
            0.01,
            0.02,
            0.03,
        ]
    )

    risk_statistics = historical_var_cvar(
        returns=returns,
        confidence_level=0.95,
    )

    assert (
        risk_statistics[
            "historical_var"
        ]
        >= 0
    )

    assert (
        risk_statistics[
            "historical_cvar"
        ]
        >= risk_statistics[
            "historical_var"
        ]
    )


def test_kupiec_test():
    exceptions = pd.Series(
        [
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    result = (
        kupiec_unconditional_coverage_test(
            exceptions=exceptions,
            confidence_level=0.95,
        )
    )

    assert (
        result[
            "observed_exceptions"
        ]
        == 1
    )

    assert (
        0
        <= result[
            "kupiec_p_value"
        ]
        <= 1
    )


def test_ridge_model():
    features = pd.DataFrame(
        {
            "factor_value": [
                -1.0,
                0.0,
                1.0,
                2.0,
            ],
            "factor_quality": [
                0.5,
                1.0,
                1.5,
                2.0,
            ],
        }
    )

    target = pd.Series(
        [
            -0.01,
            0.00,
            0.01,
            0.02,
        ]
    )

    model = build_ridge_model(
        alpha=1.0
    )

    model.fit(
        features,
        target,
    )

    predictions = model.predict(
        features
    )

    assert len(predictions) == 4
    assert np.isfinite(
        predictions
    ).all()