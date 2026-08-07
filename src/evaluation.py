"""Performance evaluation and statistical-testing utilities."""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data import validate_required_columns


NumericOrSeries = Union[float, int, pd.Series]


def calculate_drawdown_series(
    returns: pd.Series,
) -> pd.Series:
    """Calculate the drawdown series from simple periodic returns."""

    numeric_returns = pd.to_numeric(
        returns,
        errors="coerce",
    ).fillna(0.0)

    cumulative_wealth = (
        1.0 + numeric_returns
    ).cumprod()

    running_peak = cumulative_wealth.cummax()

    drawdown = (
        cumulative_wealth / running_peak
    ) - 1.0

    return drawdown


def calculate_performance_statistics(
    returns: pd.Series,
    risk_free_returns: Optional[NumericOrSeries] = None,
    periods_per_year: int = 12,
    confidence_level: float = 0.95,
) -> pd.Series:
    """Calculate return, risk, and tail-risk performance statistics."""

    numeric_returns = pd.to_numeric(
        returns,
        errors="coerce",
    ).dropna()

    number_of_periods = len(
        numeric_returns
    )

    if number_of_periods == 0:
        raise ValueError(
            "At least one valid return observation is required."
        )

    if risk_free_returns is None:
        aligned_risk_free = pd.Series(
            0.0,
            index=numeric_returns.index,
        )

    elif np.isscalar(
        risk_free_returns
    ):
        aligned_risk_free = pd.Series(
            float(risk_free_returns),
            index=numeric_returns.index,
        )

    else:
        aligned_risk_free = pd.to_numeric(
            risk_free_returns,
            errors="coerce",
        ).reindex(
            numeric_returns.index
        ).fillna(0.0)

    excess_returns = (
        numeric_returns
        - aligned_risk_free
    )

    terminal_wealth = float(
        (
            1.0
            + numeric_returns
        ).prod()
    )

    if terminal_wealth > 0:
        annualized_return = (
            terminal_wealth
            ** (
                periods_per_year
                / number_of_periods
            )
            - 1.0
        )
    else:
        annualized_return = np.nan

    annualized_volatility = (
        numeric_returns.std(
            ddof=1
        )
        * np.sqrt(
            periods_per_year
        )
    )

    excess_volatility = (
        excess_returns.std(
            ddof=1
        )
    )

    if (
        pd.notna(
            excess_volatility
        )
        and excess_volatility > 0
    ):
        annualized_sharpe = (
            excess_returns.mean()
            / excess_volatility
            * np.sqrt(
                periods_per_year
            )
        )
    else:
        annualized_sharpe = np.nan

    downside_returns = (
        excess_returns.loc[
            excess_returns < 0
        ]
    )

    if len(
        downside_returns
    ) > 0:
        annualized_downside_deviation = (
            np.sqrt(
                np.mean(
                    np.square(
                        downside_returns
                    )
                )
            )
            * np.sqrt(
                periods_per_year
            )
        )
    else:
        annualized_downside_deviation = np.nan

    if (
        pd.notna(
            annualized_downside_deviation
        )
        and annualized_downside_deviation > 0
    ):
        annualized_sortino = (
            excess_returns.mean()
            * periods_per_year
            / annualized_downside_deviation
        )
    else:
        annualized_sortino = np.nan

    drawdown = (
        calculate_drawdown_series(
            numeric_returns
        )
    )

    maximum_drawdown = float(
        drawdown.min()
    )

    if maximum_drawdown < 0:
        calmar_ratio = (
            annualized_return
            / abs(
                maximum_drawdown
            )
        )
    else:
        calmar_ratio = np.nan

    if not (
        0 < confidence_level < 1
    ):
        raise ValueError(
            "confidence_level must lie between zero and one."
        )

    lower_tail_probability = (
        1.0
        - confidence_level
    )

    return_quantile = (
        numeric_returns.quantile(
            lower_tail_probability
        )
    )

    historical_var = max(
        0.0,
        float(
            -return_quantile
        ),
    )

    tail_returns = (
        numeric_returns.loc[
            numeric_returns
            <= return_quantile
        ]
    )

    if len(
        tail_returns
    ) > 0:
        historical_cvar = max(
            0.0,
            float(
                -tail_returns.mean()
            ),
        )
    else:
        historical_cvar = np.nan

    statistics = pd.Series(
        {
            "number_of_periods": (
                number_of_periods
            ),
            "annualized_return": (
                annualized_return
            ),
            "annualized_volatility": (
                annualized_volatility
            ),
            "annualized_sharpe": (
                annualized_sharpe
            ),
            "annualized_downside_deviation": (
                annualized_downside_deviation
            ),
            "annualized_sortino": (
                annualized_sortino
            ),
            "maximum_drawdown": (
                maximum_drawdown
            ),
            "calmar_ratio": (
                calmar_ratio
            ),
            "historical_var": (
                historical_var
            ),
            "historical_cvar": (
                historical_cvar
            ),
            "positive_period_rate": float(
                (
                    numeric_returns
                    > 0
                ).mean()
            ),
            "worst_period": float(
                numeric_returns.min()
            ),
            "best_period": float(
                numeric_returns.max()
            ),
            "skewness": float(
                numeric_returns.skew()
            ),
            "excess_kurtosis": float(
                numeric_returns.kurt()
            ),
            "terminal_wealth": (
                terminal_wealth
            ),
        }
    )

    return statistics


def evaluate_return_panel(
    data: pd.DataFrame,
    series_column: str = "series",
    return_column: str = "return",
    risk_free_column: Optional[str] = None,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Calculate performance statistics for multiple return series."""

    required_columns = [
        series_column,
        return_column,
    ]

    if risk_free_column is not None:
        required_columns.append(
            risk_free_column
        )

    validate_required_columns(
        data=data,
        required_columns=required_columns,
        dataset_name="return panel",
    )

    records = []

    for (
        series_name,
        series_data,
    ) in data.groupby(
        series_column,
        sort=True,
    ):
        if risk_free_column is None:
            risk_free_returns = None
        else:
            risk_free_returns = (
                series_data[
                    risk_free_column
                ]
            )

        statistics = (
            calculate_performance_statistics(
                returns=series_data[
                    return_column
                ],
                risk_free_returns=(
                    risk_free_returns
                ),
                periods_per_year=(
                    periods_per_year
                ),
            )
        )

        record = {
            "series": series_name,
        }

        record.update(
            statistics.to_dict()
        )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


def newey_west_mean_test(
    values: pd.Series,
    maximum_lags: int = 6,
    periods_per_year: int = 12,
) -> pd.Series:
    """Test whether a time-series mean differs from zero using HAC errors."""

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(
        numeric_values
    ) < 3:
        raise ValueError(
            "At least three valid observations are required for a HAC test."
        )

    dependent_variable = (
        numeric_values.to_numpy(
            dtype=float
        )
    )

    explanatory_variables = np.ones(
        (
            len(
                dependent_variable
            ),
            1,
        )
    )

    model = sm.OLS(
        dependent_variable,
        explanatory_variables,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": (
                maximum_lags
            ),
        },
    )

    mean_value = float(
        model.params[0]
    )

    result = pd.Series(
        {
            "number_of_observations": len(
                numeric_values
            ),
            "mean_value": (
                mean_value
            ),
            "annualized_mean": (
                mean_value
                * periods_per_year
            ),
            "newey_west_t_statistic": float(
                model.tvalues[0]
            ),
            "newey_west_p_value": float(
                model.pvalues[0]
            ),
            "positive_observation_rate": float(
                (
                    numeric_values
                    > 0
                ).mean()
            ),
        }
    )

    return result


def calculate_monthly_information_coefficients(
    data: pd.DataFrame,
    signal_column: str,
    future_return_column: str,
    month_column: str = "month",
    correlation_method: str = "spearman",
    minimum_observations: int = 20,
) -> pd.DataFrame:
    """Calculate cross-sectional monthly information coefficients."""

    if correlation_method not in {
        "pearson",
        "spearman",
    }:
        raise ValueError(
            "correlation_method must be 'pearson' or 'spearman'."
        )

    validate_required_columns(
        data=data,
        required_columns=[
            month_column,
            signal_column,
            future_return_column,
        ],
        dataset_name="factor-testing panel",
    )

    records = []

    for (
        month,
        month_data,
    ) in data.groupby(
        month_column,
        sort=True,
    ):
        valid_sample = (
            month_data[
                [
                    signal_column,
                    future_return_column,
                ]
            ]
            .copy()
        )

        valid_sample = (
            valid_sample.apply(
                pd.to_numeric,
                errors="coerce",
            )
            .dropna()
        )

        if len(
            valid_sample
        ) < minimum_observations:
            continue

        information_coefficient = (
            valid_sample[
                signal_column
            ].corr(
                valid_sample[
                    future_return_column
                ],
                method=(
                    correlation_method
                ),
            )
        )

        records.append(
            {
                "month": pd.Timestamp(
                    month
                ),
                "signal": (
                    signal_column
                ),
                "correlation_method": (
                    correlation_method
                ),
                "number_of_observations": len(
                    valid_sample
                ),
                "information_coefficient": (
                    information_coefficient
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def summarize_information_coefficients(
    information_coefficients: pd.Series,
    maximum_lags: int = 6,
) -> pd.Series:
    """Summarize a monthly information-coefficient time series."""

    numeric_ic = pd.to_numeric(
        information_coefficients,
        errors="coerce",
    ).dropna()

    if len(
        numeric_ic
    ) < 3:
        raise ValueError(
            "At least three valid IC observations are required."
        )

    hac_result = (
        newey_west_mean_test(
            values=numeric_ic,
            maximum_lags=(
                maximum_lags
            ),
            periods_per_year=12,
        )
    )

    ic_standard_deviation = (
        numeric_ic.std(
            ddof=1
        )
    )

    if (
        pd.notna(
            ic_standard_deviation
        )
        and ic_standard_deviation > 0
    ):
        annualized_icir = (
            numeric_ic.mean()
            / ic_standard_deviation
            * np.sqrt(12)
        )
    else:
        annualized_icir = np.nan

    result = pd.Series(
        {
            "number_of_months": len(
                numeric_ic
            ),
            "mean_ic": float(
                numeric_ic.mean()
            ),
            "annualized_icir": (
                annualized_icir
            ),
            "newey_west_t_statistic": (
                hac_result[
                    "newey_west_t_statistic"
                ]
            ),
            "newey_west_p_value": (
                hac_result[
                    "newey_west_p_value"
                ]
            ),
            "positive_month_rate": float(
                (
                    numeric_ic
                    > 0
                ).mean()
            ),
        }
    )

    return result


def run_factor_regression(
    data: pd.DataFrame,
    portfolio_return_column: str,
    factor_columns: Sequence[str],
    risk_free_column: str = "rf",
    maximum_lags: int = 6,
    periods_per_year: int = 12,
) -> pd.Series:
    """Estimate a factor regression using Newey-West standard errors."""

    validate_required_columns(
        data=data,
        required_columns=[
            portfolio_return_column,
            risk_free_column,
            *factor_columns,
        ],
        dataset_name="factor-regression dataset",
    )

    regression_data = (
        data[
            [
                portfolio_return_column,
                risk_free_column,
                *factor_columns,
            ]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna()
    )

    minimum_required_rows = (
        len(
            factor_columns
        )
        + 3
    )

    if len(
        regression_data
    ) < minimum_required_rows:
        raise ValueError(
            "The regression sample does not contain enough observations."
        )

    excess_return = (
        regression_data[
            portfolio_return_column
        ]
        - regression_data[
            risk_free_column
        ]
    )

    explanatory_variables = (
        sm.add_constant(
            regression_data[
                list(
                    factor_columns
                )
            ],
            has_constant="add",
        )
    )

    model = sm.OLS(
        excess_return,
        explanatory_variables,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": (
                maximum_lags
            ),
        },
    )

    result = {
        "number_of_months": len(
            regression_data
        ),
        "monthly_alpha": float(
            model.params[
                "const"
            ]
        ),
        "annualized_alpha": float(
            model.params[
                "const"
            ]
            * periods_per_year
        ),
        "alpha_t_statistic": float(
            model.tvalues[
                "const"
            ]
        ),
        "alpha_p_value": float(
            model.pvalues[
                "const"
            ]
        ),
        "r_squared": float(
            model.rsquared
        ),
        "adjusted_r_squared": float(
            model.rsquared_adj
        ),
    }

    for factor in factor_columns:
        result[
            "{}_beta".format(
                factor
            )
        ] = float(
            model.params[
                factor
            ]
        )

        result[
            "{}_t_statistic".format(
                factor
            )
        ] = float(
            model.tvalues[
                factor
            ]
        )

        result[
            "{}_p_value".format(
                factor
            )
        ] = float(
            model.pvalues[
                factor
            ]
        )

    return pd.Series(
        result
    )


def holm_adjust_p_values(
    p_values: pd.Series,
) -> pd.Series:
    """Apply the Holm step-down correction for multiple testing."""

    numeric_p_values = pd.to_numeric(
        p_values,
        errors="coerce",
    )

    adjusted_p_values = pd.Series(
        np.nan,
        index=numeric_p_values.index,
        dtype=float,
    )

    valid_p_values = (
        numeric_p_values.dropna()
    )

    number_of_tests = len(
        valid_p_values
    )

    if number_of_tests == 0:
        return adjusted_p_values

    ordered_p_values = (
        valid_p_values.sort_values()
    )

    running_maximum = 0.0

    for (
        position,
        (
            original_index,
            p_value,
        ),
    ) in enumerate(
        ordered_p_values.items()
    ):
        multiplier = (
            number_of_tests
            - position
        )

        candidate_value = min(
            1.0,
            float(
                p_value
            )
            * multiplier,
        )

        running_maximum = max(
            running_maximum,
            candidate_value,
        )

        adjusted_p_values.loc[
            original_index
        ] = running_maximum

    return adjusted_p_values