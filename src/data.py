"""Data-loading and validation utilities."""

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def load_table(
    file_path: PathLike,
    date_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Load a CSV or Parquet dataset.

    Parameters
    ----------
    file_path:
        Location of the input file.
    date_columns:
        Columns that should be converted to pandas datetime values.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            "The requested data file does not exist: {}".format(file_path)
        )

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        data = pd.read_csv(file_path, low_memory=False)
    elif suffix in {".parquet", ".pq"}:
        data = pd.read_parquet(file_path)
    else:
        raise ValueError(
            "Unsupported file format: {}. Use CSV or Parquet.".format(suffix)
        )

    if date_columns is not None:
        for column in date_columns:
            if column not in data.columns:
                raise KeyError(
                    "Date column '{}' is not present in the dataset.".format(
                        column
                    )
                )

            data[column] = pd.to_datetime(
                data[column],
                errors="coerce",
            )

    return data


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str = "dataset",
) -> None:
    """Verify that all required columns are available."""

    required_columns = list(required_columns)

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            "{} is missing the following columns: {}".format(
                dataset_name,
                ", ".join(missing_columns),
            )
        )


def validate_unique_keys(
    data: pd.DataFrame,
    key_columns: Sequence[str],
    dataset_name: str = "dataset",
) -> None:
    """Verify that a set of columns uniquely identifies every row."""

    validate_required_columns(
        data=data,
        required_columns=key_columns,
        dataset_name=dataset_name,
    )

    duplicate_count = int(
        data.duplicated(
            subset=list(key_columns),
            keep=False,
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "{} contains {:,} rows with duplicate keys: {}".format(
                dataset_name,
                duplicate_count,
                ", ".join(key_columns),
            )
        )


def validate_no_missing_values(
    data: pd.DataFrame,
    columns: Sequence[str],
    dataset_name: str = "dataset",
) -> None:
    """Verify that selected columns do not contain missing values."""

    validate_required_columns(
        data=data,
        required_columns=columns,
        dataset_name=dataset_name,
    )

    missing_counts = data[list(columns)].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        missing_description = ", ".join(
            "{}={:,}".format(column, int(count))
            for column, count in missing_counts.items()
        )

        raise ValueError(
            "{} contains missing values: {}".format(
                dataset_name,
                missing_description,
            )
        )


def validate_date_range(
    data: pd.DataFrame,
    date_column: str,
    minimum_date: Optional[str] = None,
    maximum_date: Optional[str] = None,
) -> None:
    """Verify that a dataset falls inside the required date range."""

    validate_required_columns(
        data=data,
        required_columns=[date_column],
    )

    dates = pd.to_datetime(
        data[date_column],
        errors="coerce",
    )

    if dates.isna().any():
        raise ValueError(
            "The date column '{}' contains invalid or missing dates.".format(
                date_column
            )
        )

    if minimum_date is not None:
        minimum_date = pd.Timestamp(minimum_date)

        if dates.min() < minimum_date:
            raise ValueError(
                "The dataset starts before the permitted minimum date."
            )

    if maximum_date is not None:
        maximum_date = pd.Timestamp(maximum_date)

        if dates.max() > maximum_date:
            raise ValueError(
                "The dataset ends after the permitted maximum date."
            )


def convert_to_month_end(
    values: pd.Series,
) -> pd.Series:
    """Convert dates to their corresponding calendar month-end dates."""

    dates = pd.to_datetime(
        values,
        errors="coerce",
    )

    return dates + pd.offsets.MonthEnd(0)


def compound_simple_returns(
    returns: pd.Series,
    minimum_observations: int = 1,
) -> float:
    """
    Compound simple returns over one period.

    The compounded return is calculated as:

        product(1 + return) - 1
    """

    valid_returns = pd.to_numeric(
        returns,
        errors="coerce",
    ).dropna()

    if len(valid_returns) < minimum_observations:
        return np.nan

    return float(
        np.prod(1.0 + valid_returns.to_numpy()) - 1.0
    )


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Sequence[str],
    how: str = "left",
    validate: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge two datasets while checking for unexpected row multiplication.
    """

    left_row_count = len(left)

    merged = left.merge(
        right,
        on=list(on),
        how=how,
        validate=validate,
    )

    if how == "left" and len(merged) < left_row_count:
        raise ValueError(
            "The left merge unexpectedly removed observations."
        )

    return merged


def summarize_panel(
    data: pd.DataFrame,
    entity_column: str,
    date_column: str,
) -> pd.Series:
    """Return a compact summary of a panel dataset."""

    validate_required_columns(
        data=data,
        required_columns=[
            entity_column,
            date_column,
        ],
    )

    dates = pd.to_datetime(
        data[date_column],
        errors="coerce",
    )

    summary = pd.Series(
        {
            "number_of_rows": len(data),
            "number_of_entities": data[entity_column].nunique(),
            "number_of_periods": dates.nunique(),
            "start_date": dates.min(),
            "end_date": dates.max(),
            "duplicate_entity_date_rows": int(
                data.duplicated(
                    subset=[
                        entity_column,
                        date_column,
                    ],
                    keep=False,
                ).sum()
            ),
        }
    )

    return summary