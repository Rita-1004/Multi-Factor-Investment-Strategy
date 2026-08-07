"""Portfolio construction and monthly backtesting utilities."""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data import validate_required_columns


SUPPORTED_WEIGHTING_METHODS = {
    "Equal Weight",
    "Inverse Volatility",
    "Score Rank Weight",
    "Value Weight",
}


def select_top_fraction(
    period_data: pd.DataFrame,
    score_column: str,
    selection_fraction: float = 0.20,
) -> pd.DataFrame:
    """Select the highest-scoring fraction of the investment universe."""

    if not 0 < selection_fraction <= 1:
        raise ValueError(
            "selection_fraction must be greater than zero and no greater than one."
        )

    valid_data = period_data.loc[
        period_data[score_column].notna()
    ].copy()

    if valid_data.empty:
        return valid_data

    number_to_select = max(
        1,
        int(
            np.ceil(
                len(valid_data) * selection_fraction
            )
        ),
    )

    selected = (
        valid_data.sort_values(
            score_column,
            ascending=False,
        )
        .head(number_to_select)
        .copy()
    )

    return selected


def calculate_target_weights(
    selected_data: pd.DataFrame,
    weighting_method: str,
    score_column: str,
    volatility_column: str = "volatility_12m",
    market_cap_column: str = "month_end_market_cap",
    output_column: str = "target_weight",
) -> pd.DataFrame:
    """Calculate portfolio target weights for one formation period."""

    if weighting_method not in SUPPORTED_WEIGHTING_METHODS:
        raise ValueError(
            "Unsupported weighting method: {}".format(
                weighting_method
            )
        )

    result = selected_data.copy()

    if result.empty:
        result[output_column] = pd.Series(dtype=float)
        return result

    if weighting_method == "Equal Weight":
        raw_weights = pd.Series(
            1.0,
            index=result.index,
        )

    elif weighting_method == "Inverse Volatility":
        validate_required_columns(
            data=result,
            required_columns=[volatility_column],
            dataset_name="selected portfolio",
        )

        volatility = pd.to_numeric(
            result[volatility_column],
            errors="coerce",
        )

        valid_volatility = (
            volatility.notna()
            & np.isfinite(volatility)
            & (volatility > 0)
        )

        result = result.loc[
            valid_volatility
        ].copy()

        volatility = volatility.loc[
            result.index
        ]

        raw_weights = 1.0 / volatility

    elif weighting_method == "Score Rank Weight":
        scores = pd.to_numeric(
            result[score_column],
            errors="coerce",
        )

        raw_weights = scores.rank(
            method="average",
            ascending=True,
        )

    elif weighting_method == "Value Weight":
        validate_required_columns(
            data=result,
            required_columns=[market_cap_column],
            dataset_name="selected portfolio",
        )

        market_cap = pd.to_numeric(
            result[market_cap_column],
            errors="coerce",
        )

        valid_market_cap = (
            market_cap.notna()
            & np.isfinite(market_cap)
            & (market_cap > 0)
        )

        result = result.loc[
            valid_market_cap
        ].copy()

        raw_weights = market_cap.loc[
            result.index
        ]

    raw_weights = pd.to_numeric(
        raw_weights,
        errors="coerce",
    )

    raw_weights = raw_weights.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    valid_weights = raw_weights.notna() & (raw_weights > 0)

    result = result.loc[
        valid_weights
    ].copy()

    raw_weights = raw_weights.loc[
        result.index
    ]

    total_raw_weight = raw_weights.sum()

    if result.empty or total_raw_weight <= 0:
        result[output_column] = pd.Series(
            dtype=float
        )
        return result

    result[output_column] = (
        raw_weights / total_raw_weight
    )

    return result


def calculate_turnover(
    old_weights: pd.Series,
    new_weights: pd.Series,
) -> float:
    """
    Calculate one-way portfolio turnover.

    Turnover equals one-half of the total absolute change in security
    weights. If there is no previously invested portfolio, initial
    turnover is set equal to one.
    """

    if old_weights.empty:
        return 1.0

    all_securities = old_weights.index.union(
        new_weights.index
    )

    aligned_old = old_weights.reindex(
        all_securities,
        fill_value=0.0,
    )

    aligned_new = new_weights.reindex(
        all_securities,
        fill_value=0.0,
    )

    turnover = 0.5 * np.abs(
        aligned_new - aligned_old
    ).sum()

    return float(turnover)


def drift_portfolio_weights(
    target_weights: pd.Series,
    realized_returns: pd.Series,
) -> pd.Series:
    """Update portfolio weights after the realization of asset returns."""

    aligned_returns = realized_returns.reindex(
        target_weights.index
    ).fillna(0.0)

    ending_values = target_weights * (
        1.0 + aligned_returns
    )

    total_ending_value = ending_values.sum()

    if (
        not np.isfinite(total_ending_value)
        or total_ending_value <= 0
    ):
        return target_weights.copy()

    return ending_values / total_ending_value


def run_monthly_backtest(
    data: pd.DataFrame,
    score_column: str,
    weighting_method: str = "Equal Weight",
    selection_fraction: float = 0.20,
    transaction_cost: float = 0.001,
    entity_column: str = "permno",
    formation_month_column: str = "month",
    future_return_column: str = "future_return_1m",
    volatility_column: str = "volatility_12m",
    market_cap_column: str = "month_end_market_cap",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a long-only monthly portfolio backtest.

    Securities are selected using information available in each
    formation month. Their next-month returns are then used to evaluate
    portfolio performance.
    """

    validate_required_columns(
        data=data,
        required_columns=[
            entity_column,
            formation_month_column,
            score_column,
            future_return_column,
        ],
        dataset_name="backtest panel",
    )

    if transaction_cost < 0:
        raise ValueError(
            "transaction_cost cannot be negative."
        )

    backtest_data = data.copy()

    backtest_data[formation_month_column] = pd.to_datetime(
        backtest_data[formation_month_column],
        errors="coerce",
    )

    backtest_data = backtest_data.loc[
        backtest_data[formation_month_column].notna()
    ].copy()

    duplicate_count = int(
        backtest_data.duplicated(
            subset=[
                entity_column,
                formation_month_column,
            ],
            keep=False,
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "The backtest panel contains duplicate security-month rows."
        )

    performance_records = []
    holding_records = []

    previous_ending_weights = pd.Series(
        dtype=float
    )

    formation_months = sorted(
        backtest_data[
            formation_month_column
        ].dropna().unique()
    )

    for formation_month in formation_months:
        period_data = backtest_data.loc[
            backtest_data[formation_month_column]
            == formation_month
        ].copy()

        selected = select_top_fraction(
            period_data=period_data,
            score_column=score_column,
            selection_fraction=selection_fraction,
        )

        weighted_portfolio = calculate_target_weights(
            selected_data=selected,
            weighting_method=weighting_method,
            score_column=score_column,
            volatility_column=volatility_column,
            market_cap_column=market_cap_column,
            output_column="target_weight",
        )

        if weighted_portfolio.empty:
            continue

        weighted_portfolio = weighted_portfolio.set_index(
            entity_column,
            drop=False,
        )

        target_weights = weighted_portfolio[
            "target_weight"
        ].astype(float)

        turnover = calculate_turnover(
            old_weights=previous_ending_weights,
            new_weights=target_weights,
        )

        realized_returns = pd.to_numeric(
            weighted_portfolio[
                future_return_column
            ],
            errors="coerce",
        )

        realized_returns.index = (
            weighted_portfolio.index
        )

        missing_return_count = int(
            realized_returns.isna().sum()
        )

        # Missing realized returns are assigned a zero return instead
        # of being removed and reweighted with future information.
        realized_returns_for_backtest = (
            realized_returns.fillna(0.0)
        )

        gross_return = float(
            (
                target_weights
                * realized_returns_for_backtest
            ).sum()
        )

        transaction_cost_amount = (
            turnover * transaction_cost
        )

        net_return = (
            gross_return
            - transaction_cost_amount
        )

        return_month = (
            pd.Timestamp(formation_month)
            + pd.offsets.MonthEnd(1)
        )

        performance_records.append(
            {
                "formation_month": pd.Timestamp(
                    formation_month
                ),
                "return_month": return_month,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "transaction_cost": (
                    transaction_cost_amount
                ),
                "number_of_holdings": len(
                    weighted_portfolio
                ),
                "missing_return_holdings": (
                    missing_return_count
                ),
                "score_column": score_column,
                "weighting_method": (
                    weighting_method
                ),
                "selection_fraction": (
                    selection_fraction
                ),
            }
        )

        for security, row in weighted_portfolio.iterrows():
            holding_records.append(
                {
                    "formation_month": pd.Timestamp(
                        formation_month
                    ),
                    "return_month": return_month,
                    entity_column: security,
                    "score": row[score_column],
                    "target_weight": row[
                        "target_weight"
                    ],
                    "realized_return": row[
                        future_return_column
                    ],
                    "score_column": score_column,
                    "weighting_method": (
                        weighting_method
                    ),
                    "selection_fraction": (
                        selection_fraction
                    ),
                }
            )

        previous_ending_weights = (
            drift_portfolio_weights(
                target_weights=target_weights,
                realized_returns=(
                    realized_returns_for_backtest
                ),
            )
        )

    performance_data = pd.DataFrame(
        performance_records
    )

    holdings_data = pd.DataFrame(
        holding_records
    )

    return performance_data, holdings_data


def validate_portfolio_weights(
    holdings_data: pd.DataFrame,
    period_column: str = "formation_month",
    weight_column: str = "target_weight",
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Validate that target weights sum to one in every period."""

    validate_required_columns(
        data=holdings_data,
        required_columns=[
            period_column,
            weight_column,
        ],
        dataset_name="portfolio holdings",
    )

    weight_summary = (
        holdings_data.groupby(
            period_column,
            as_index=False,
        )[weight_column]
        .sum()
        .rename(
            columns={
                weight_column: "weight_sum",
            }
        )
    )

    weight_summary["absolute_error"] = np.abs(
        weight_summary["weight_sum"] - 1.0
    )

    maximum_error = weight_summary[
        "absolute_error"
    ].max()

    if maximum_error > tolerance:
        raise ValueError(
            "Portfolio weights do not sum to one within the permitted tolerance."
        )

    return weight_summary