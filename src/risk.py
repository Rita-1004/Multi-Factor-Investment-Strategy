"""Tail-risk, drawdown, stress-testing, and VaR backtesting utilities."""

from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.data import validate_required_columns
from src.evaluation import calculate_drawdown_series


def historical_var_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> pd.Series:
    """Calculate historical VaR and CVaR as positive loss measures."""

    numeric_returns = pd.to_numeric(
        returns,
        errors="coerce",
    ).dropna()

    if numeric_returns.empty:
        raise ValueError(
            "At least one valid return is required."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must lie between zero and one."
        )

    return_threshold = numeric_returns.quantile(
        1.0 - confidence_level
    )

    tail_returns = numeric_returns.loc[
        numeric_returns <= return_threshold
    ]

    value_at_risk = max(
        0.0,
        float(-return_threshold),
    )

    conditional_value_at_risk = max(
        0.0,
        float(-tail_returns.mean()),
    )

    return pd.Series(
        {
            "confidence_level": confidence_level,
            "number_of_observations": len(
                numeric_returns
            ),
            "return_threshold": (
                return_threshold
            ),
            "historical_var": value_at_risk,
            "historical_cvar": (
                conditional_value_at_risk
            ),
            "number_of_tail_observations": len(
                tail_returns
            ),
        }
    )


def maximum_drawdown_episode(
    returns: pd.Series,
) -> pd.Series:
    """Identify the maximum-drawdown episode and recovery date."""

    numeric_returns = pd.to_numeric(
        returns,
        errors="coerce",
    ).dropna()

    if numeric_returns.empty:
        raise ValueError(
            "At least one valid return is required."
        )

    numeric_returns = numeric_returns.sort_index()

    cumulative_wealth = (
        1.0 + numeric_returns
    ).cumprod()

    running_peak = cumulative_wealth.cummax()

    drawdown = (
        cumulative_wealth / running_peak
    ) - 1.0

    trough_date = drawdown.idxmin()
    maximum_drawdown = float(
        drawdown.loc[trough_date]
    )

    peak_date = cumulative_wealth.loc[
        :trough_date
    ].idxmax()

    peak_wealth = float(
        cumulative_wealth.loc[peak_date]
    )

    recovery_sample = cumulative_wealth.loc[
        trough_date:
    ]

    recovered_observations = (
        recovery_sample.loc[
            recovery_sample >= peak_wealth
        ]
    )

    if recovered_observations.empty:
        recovery_date = pd.NaT
        recovery_status = "Not Recovered"
        peak_to_recovery_months = np.nan
    else:
        recovery_date = (
            recovered_observations.index[0]
        )
        recovery_status = "Recovered"
        peak_to_recovery_months = (
            (
                recovery_date.year
                - peak_date.year
            )
            * 12
            + (
                recovery_date.month
                - peak_date.month
            )
        )

    peak_to_trough_months = (
        (
            trough_date.year
            - peak_date.year
        )
        * 12
        + (
            trough_date.month
            - peak_date.month
        )
    )

    return pd.Series(
        {
            "maximum_drawdown": (
                maximum_drawdown
            ),
            "peak_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": recovery_date,
            "recovery_status": recovery_status,
            "peak_to_trough_months": (
                peak_to_trough_months
            ),
            "peak_to_recovery_months": (
                peak_to_recovery_months
            ),
        }
    )


def create_rolling_historical_var_forecasts(
    returns: pd.Series,
    estimation_window: int = 60,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Create one-period-ahead rolling historical VaR forecasts."""

    numeric_returns = pd.to_numeric(
        returns,
        errors="coerce",
    ).dropna().sort_index()

    if len(numeric_returns) <= estimation_window:
        raise ValueError(
            "The return series is too short for the selected estimation window."
        )

    records = []

    for position in range(
        estimation_window,
        len(numeric_returns),
    ):
        estimation_sample = numeric_returns.iloc[
            position - estimation_window:
            position
        ]

        realized_return = float(
            numeric_returns.iloc[position]
        )

        forecast_date = (
            numeric_returns.index[position]
        )

        return_threshold = float(
            estimation_sample.quantile(
                1.0 - confidence_level
            )
        )

        tail_sample = estimation_sample.loc[
            estimation_sample
            <= return_threshold
        ]

        cvar_threshold = float(
            tail_sample.mean()
        )

        exception = int(
            realized_return
            < return_threshold
        )

        records.append(
            {
                "date": forecast_date,
                "realized_return": (
                    realized_return
                ),
                "var_return_threshold": (
                    return_threshold
                ),
                "var_loss_forecast": max(
                    0.0,
                    -return_threshold,
                ),
                "cvar_return_threshold": (
                    cvar_threshold
                ),
                "cvar_loss_forecast": max(
                    0.0,
                    -cvar_threshold,
                ),
                "var_exception": exception,
                "estimation_window": (
                    estimation_window
                ),
                "confidence_level": (
                    confidence_level
                ),
            }
        )

    return pd.DataFrame(records)


def kupiec_unconditional_coverage_test(
    exceptions: pd.Series,
    confidence_level: float = 0.95,
) -> pd.Series:
    """Perform the Kupiec unconditional-coverage likelihood-ratio test."""

    exception_values = pd.to_numeric(
        exceptions,
        errors="coerce",
    ).dropna().astype(int)

    if not exception_values.isin(
        [0, 1]
    ).all():
        raise ValueError(
            "The exception series must contain only zeros and ones."
        )

    number_of_forecasts = len(
        exception_values
    )

    observed_exceptions = int(
        exception_values.sum()
    )

    expected_probability = (
        1.0 - confidence_level
    )

    observed_probability = (
        observed_exceptions
        / number_of_forecasts
    )

    epsilon = 1e-12

    expected_probability_clipped = np.clip(
        expected_probability,
        epsilon,
        1.0 - epsilon,
    )

    observed_probability_clipped = np.clip(
        observed_probability,
        epsilon,
        1.0 - epsilon,
    )

    log_likelihood_null = (
        (
            number_of_forecasts
            - observed_exceptions
        )
        * np.log(
            1.0
            - expected_probability_clipped
        )
        + observed_exceptions
        * np.log(
            expected_probability_clipped
        )
    )

    log_likelihood_alternative = (
        (
            number_of_forecasts
            - observed_exceptions
        )
        * np.log(
            1.0
            - observed_probability_clipped
        )
        + observed_exceptions
        * np.log(
            observed_probability_clipped
        )
    )

    likelihood_ratio = max(
        0.0,
        -2.0
        * (
            log_likelihood_null
            - log_likelihood_alternative
        ),
    )

    p_value = float(
        1.0 - chi2.cdf(
            likelihood_ratio,
            df=1,
        )
    )

    return pd.Series(
        {
            "number_of_forecasts": (
                number_of_forecasts
            ),
            "expected_exceptions": (
                number_of_forecasts
                * expected_probability
            ),
            "observed_exceptions": (
                observed_exceptions
            ),
            "observed_exception_rate": (
                observed_probability
            ),
            "kupiec_lr_statistic": (
                likelihood_ratio
            ),
            "kupiec_p_value": p_value,
            "reject_correct_coverage_at_5_percent": (
                p_value < 0.05
            ),
        }
    )


def christoffersen_independence_test(
    exceptions: pd.Series,
) -> pd.Series:
    """Test whether VaR exceptions are independent over time."""

    exception_values = pd.to_numeric(
        exceptions,
        errors="coerce",
    ).dropna().astype(int)

    if not exception_values.isin(
        [0, 1]
    ).all():
        raise ValueError(
            "The exception series must contain only zeros and ones."
        )

    if len(exception_values) < 2:
        raise ValueError(
            "At least two exception observations are required."
        )

    previous_values = exception_values.iloc[
        :-1
    ].to_numpy()

    current_values = exception_values.iloc[
        1:
    ].to_numpy()

    n00 = int(
        (
            (previous_values == 0)
            & (current_values == 0)
        ).sum()
    )

    n01 = int(
        (
            (previous_values == 0)
            & (current_values == 1)
        ).sum()
    )

    n10 = int(
        (
            (previous_values == 1)
            & (current_values == 0)
        ).sum()
    )

    n11 = int(
        (
            (previous_values == 1)
            & (current_values == 1)
        ).sum()
    )

    epsilon = 1e-12

    total_transitions = (
        n00 + n01 + n10 + n11
    )

    unconditional_probability = (
        (n01 + n11)
        / total_transitions
    )

    probability_after_zero = (
        n01 / (n00 + n01)
        if (n00 + n01) > 0
        else 0.0
    )

    probability_after_one = (
        n11 / (n10 + n11)
        if (n10 + n11) > 0
        else 0.0
    )

    unconditional_probability = np.clip(
        unconditional_probability,
        epsilon,
        1.0 - epsilon,
    )

    probability_after_zero_clipped = np.clip(
        probability_after_zero,
        epsilon,
        1.0 - epsilon,
    )

    probability_after_one_clipped = np.clip(
        probability_after_one,
        epsilon,
        1.0 - epsilon,
    )

    log_likelihood_independent = (
        (n00 + n10)
        * np.log(
            1.0
            - unconditional_probability
        )
        + (n01 + n11)
        * np.log(
            unconditional_probability
        )
    )

    log_likelihood_dependent = (
        n00
        * np.log(
            1.0
            - probability_after_zero_clipped
        )
        + n01
        * np.log(
            probability_after_zero_clipped
        )
        + n10
        * np.log(
            1.0
            - probability_after_one_clipped
        )
        + n11
        * np.log(
            probability_after_one_clipped
        )
    )

    likelihood_ratio = max(
        0.0,
        -2.0
        * (
            log_likelihood_independent
            - log_likelihood_dependent
        ),
    )

    p_value = float(
        1.0 - chi2.cdf(
            likelihood_ratio,
            df=1,
        )
    )

    return pd.Series(
        {
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "exception_probability_after_no_exception": (
                probability_after_zero
            ),
            "exception_probability_after_exception": (
                probability_after_one
            ),
            "independence_lr_statistic": (
                likelihood_ratio
            ),
            "independence_p_value": (
                p_value
            ),
            "reject_exception_independence_at_5_percent": (
                p_value < 0.05
            ),
        }
    )


def christoffersen_conditional_coverage_test(
    exceptions: pd.Series,
    confidence_level: float = 0.95,
) -> pd.Series:
    """Combine Kupiec coverage and Christoffersen independence tests."""

    kupiec_result = (
        kupiec_unconditional_coverage_test(
            exceptions=exceptions,
            confidence_level=confidence_level,
        )
    )

    independence_result = (
        christoffersen_independence_test(
            exceptions=exceptions,
        )
    )

    conditional_coverage_statistic = (
        kupiec_result[
            "kupiec_lr_statistic"
        ]
        + independence_result[
            "independence_lr_statistic"
        ]
    )

    conditional_coverage_p_value = float(
        1.0 - chi2.cdf(
            conditional_coverage_statistic,
            df=2,
        )
    )

    result = independence_result.copy()

    result[
        "conditional_coverage_lr_statistic"
    ] = conditional_coverage_statistic

    result[
        "conditional_coverage_p_value"
    ] = conditional_coverage_p_value

    result[
        "reject_correct_conditional_coverage_at_5_percent"
    ] = (
        conditional_coverage_p_value
        < 0.05
    )

    return result


def calculate_stress_period_statistics(
    data: pd.DataFrame,
    stress_periods: Mapping[str, Tuple[str, str]],
    date_column: str = "month",
    series_column: str = "series",
    return_column: str = "return",
) -> pd.DataFrame:
    """Calculate cumulative returns and drawdowns during stress periods."""

    validate_required_columns(
        data=data,
        required_columns=[
            date_column,
            series_column,
            return_column,
        ],
        dataset_name="stress-testing panel",
    )

    stress_data = data.copy()

    stress_data[date_column] = pd.to_datetime(
        stress_data[date_column],
        errors="coerce",
    )

    records = []

    for period_name, (
        start_date,
        end_date,
    ) in stress_periods.items():
        period_sample = stress_data.loc[
            (
                stress_data[date_column]
                >= pd.Timestamp(start_date)
            )
            & (
                stress_data[date_column]
                <= pd.Timestamp(end_date)
            )
        ].copy()

        for series_name, series_data in period_sample.groupby(
            series_column,
            sort=True,
        ):
            series_data = series_data.sort_values(
                date_column
            )

            returns = pd.to_numeric(
                series_data[return_column],
                errors="coerce",
            ).dropna()

            if returns.empty:
                continue

            cumulative_return = float(
                (1.0 + returns).prod()
                - 1.0
            )

            maximum_drawdown = float(
                calculate_drawdown_series(
                    returns
                ).min()
            )

            records.append(
                {
                    "stress_period": (
                        period_name
                    ),
                    "series": series_name,
                    "start_date": (
                        pd.Timestamp(start_date)
                    ),
                    "end_date": (
                        pd.Timestamp(end_date)
                    ),
                    "number_of_months": len(
                        returns
                    ),
                    "cumulative_return": (
                        cumulative_return
                    ),
                    "maximum_drawdown": (
                        maximum_drawdown
                    ),
                    "worst_month": float(
                        returns.min()
                    ),
                }
            )

    return pd.DataFrame(records)