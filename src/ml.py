"""Walk-forward machine-learning utilities for return prediction."""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import validate_required_columns


def build_ridge_model(
    alpha: float = 1.0,
) -> Pipeline:
    """Create a median-imputed and standardized Ridge model."""

    if alpha < 0:
        raise ValueError(
            "Ridge alpha cannot be negative."
        )

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=alpha
                ),
            ),
        ]
    )

    return model


def build_xgboost_model(
    parameters: Optional[Dict] = None,
) -> Pipeline:
    """Create a median-imputed XGBoost regression model."""

    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise ImportError(
            "XGBoost is not installed in the active Python environment."
        ) from error

    default_parameters = {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    if parameters is not None:
        default_parameters.update(
            parameters
        )

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "xgboost",
                XGBRegressor(
                    **default_parameters
                ),
            ),
        ]
    )

    return model


def create_annual_walk_forward_plan(
    data: pd.DataFrame,
    month_column: str,
    target_column: str,
    prediction_start_year: int = 2015,
    prediction_end_year: int = 2025,
) -> pd.DataFrame:
    """
    Create an annual expanding-window walk-forward plan.

    For prediction year Y:

    - Prediction formation months run from December Y-1 to November Y.
    - The corresponding realized returns run from January to December Y.
    - Training formation months end in November Y-1.
    """

    validate_required_columns(
        data=data,
        required_columns=[
            month_column,
            target_column,
        ],
        dataset_name="machine-learning panel",
    )

    working_data = data.copy()

    working_data[month_column] = pd.to_datetime(
        working_data[month_column],
        errors="coerce",
    )

    records = []

    for prediction_year in range(
        prediction_start_year,
        prediction_end_year + 1,
    ):
        training_end_month = pd.Timestamp(
            year=prediction_year - 1,
            month=11,
            day=30,
        )

        prediction_start_month = pd.Timestamp(
            year=prediction_year - 1,
            month=12,
            day=31,
        )

        prediction_end_month = pd.Timestamp(
            year=prediction_year,
            month=11,
            day=30,
        )

        training_sample = working_data.loc[
            (
                working_data[month_column]
                <= training_end_month
            )
            & working_data[
                target_column
            ].notna()
        ]

        prediction_sample = working_data.loc[
            (
                working_data[month_column]
                >= prediction_start_month
            )
            & (
                working_data[month_column]
                <= prediction_end_month
            )
        ]

        if training_sample.empty:
            training_start_month = pd.NaT
        else:
            training_start_month = (
                training_sample[
                    month_column
                ].min()
            )

        records.append(
            {
                "prediction_year": (
                    prediction_year
                ),
                "training_start_month": (
                    training_start_month
                ),
                "training_end_month": (
                    training_end_month
                ),
                "number_of_training_months": (
                    training_sample[
                        month_column
                    ].nunique()
                ),
                "number_of_training_rows": len(
                    training_sample
                ),
                "prediction_start_month": (
                    prediction_start_month
                ),
                "prediction_end_month": (
                    prediction_end_month
                ),
                "number_of_prediction_months": (
                    prediction_sample[
                        month_column
                    ].nunique()
                ),
                "number_of_prediction_rows": len(
                    prediction_sample
                ),
            }
        )

    return pd.DataFrame(records)


def run_annual_walk_forward_predictions(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    month_column: str = "month",
    identifier_columns: Sequence[str] = (
        "permno",
        "ticker",
    ),
    prediction_start_year: int = 2015,
    prediction_end_year: int = 2025,
    ridge_alpha: float = 1.0,
    xgboost_parameters: Optional[Dict] = None,
    include_xgboost: bool = True,
    minimum_training_rows: int = 1000,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Generate genuine out-of-sample annual walk-forward predictions.

    Returns
    -------
    prediction_data:
        Security-month Ridge and XGBoost predictions.

    ridge_coefficient_data:
        Standardized Ridge coefficients for each prediction year.

    xgboost_importance_data:
        XGBoost feature importances for each prediction year.
    """

    required_columns = [
        month_column,
        target_column,
        *feature_columns,
    ]

    available_identifier_columns = [
        column
        for column in identifier_columns
        if column in data.columns
    ]

    validate_required_columns(
        data=data,
        required_columns=required_columns,
        dataset_name="machine-learning panel",
    )

    working_data = data.copy()

    working_data[month_column] = pd.to_datetime(
        working_data[month_column],
        errors="coerce",
    )

    working_data = working_data.loc[
        working_data[month_column].notna()
    ].copy()

    prediction_records = []
    ridge_coefficient_records = []
    xgboost_importance_records = []

    for prediction_year in range(
        prediction_start_year,
        prediction_end_year + 1,
    ):
        training_end_month = pd.Timestamp(
            year=prediction_year - 1,
            month=11,
            day=30,
        )

        prediction_start_month = pd.Timestamp(
            year=prediction_year - 1,
            month=12,
            day=31,
        )

        prediction_end_month = pd.Timestamp(
            year=prediction_year,
            month=11,
            day=30,
        )

        training_sample = working_data.loc[
            (
                working_data[month_column]
                <= training_end_month
            )
            & working_data[
                target_column
            ].notna()
        ].copy()

        prediction_sample = working_data.loc[
            (
                working_data[month_column]
                >= prediction_start_month
            )
            & (
                working_data[month_column]
                <= prediction_end_month
            )
        ].copy()

        if len(training_sample) < minimum_training_rows:
            raise ValueError(
                "Prediction year {} has only {:,} training rows.".format(
                    prediction_year,
                    len(training_sample),
                )
            )

        if prediction_sample.empty:
            continue

        training_features = training_sample[
            list(feature_columns)
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        training_target = pd.to_numeric(
            training_sample[target_column],
            errors="coerce",
        )

        prediction_features = prediction_sample[
            list(feature_columns)
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        ridge_model = build_ridge_model(
            alpha=ridge_alpha
        )

        ridge_model.fit(
            training_features,
            training_target,
        )

        ridge_predictions = ridge_model.predict(
            prediction_features
        )

        ridge_coefficients = (
            ridge_model.named_steps[
                "ridge"
            ].coef_
        )

        for feature, coefficient in zip(
            feature_columns,
            ridge_coefficients,
        ):
            ridge_coefficient_records.append(
                {
                    "prediction_year": (
                        prediction_year
                    ),
                    "feature": feature,
                    "ridge_coefficient": float(
                        coefficient
                    ),
                }
            )

        if include_xgboost:
            xgboost_model = build_xgboost_model(
                parameters=(
                    xgboost_parameters
                )
            )

            xgboost_model.fit(
                training_features,
                training_target,
            )

            xgboost_predictions = (
                xgboost_model.predict(
                    prediction_features
                )
            )

            xgboost_importances = (
                xgboost_model.named_steps[
                    "xgboost"
                ].feature_importances_
            )

            for feature, importance in zip(
                feature_columns,
                xgboost_importances,
            ):
                xgboost_importance_records.append(
                    {
                        "prediction_year": (
                            prediction_year
                        ),
                        "feature": feature,
                        "xgboost_importance": float(
                            importance
                        ),
                    }
                )
        else:
            xgboost_predictions = np.full(
                len(prediction_sample),
                np.nan,
            )

        result_columns = [
            month_column,
            target_column,
            *available_identifier_columns,
        ]

        annual_predictions = (
            prediction_sample[
                result_columns
            ].copy()
        )

        annual_predictions[
            "prediction_year"
        ] = prediction_year

        annual_predictions[
            "ridge_prediction"
        ] = ridge_predictions

        annual_predictions[
            "xgboost_prediction"
        ] = xgboost_predictions

        prediction_records.append(
            annual_predictions
        )

    if prediction_records:
        prediction_data = pd.concat(
            prediction_records,
            ignore_index=True,
        )
    else:
        prediction_data = pd.DataFrame()

    ridge_coefficient_data = pd.DataFrame(
        ridge_coefficient_records
    )

    xgboost_importance_data = pd.DataFrame(
        xgboost_importance_records
    )

    return (
        prediction_data,
        ridge_coefficient_data,
        xgboost_importance_data,
    )


def summarize_model_features(
    ridge_coefficient_data: pd.DataFrame,
    xgboost_importance_data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize feature coefficients and importance across years."""

    validate_required_columns(
        data=ridge_coefficient_data,
        required_columns=[
            "feature",
            "prediction_year",
            "ridge_coefficient",
        ],
        dataset_name="Ridge coefficient data",
    )

    ridge_summary = (
        ridge_coefficient_data.groupby(
            "feature"
        )["ridge_coefficient"]
        .agg(
            ridge_average_coefficient="mean",
            ridge_coefficient_standard_deviation="std",
            ridge_positive_year_rate=lambda values: (
                values > 0
            ).mean(),
        )
        .reset_index()
    )

    if xgboost_importance_data.empty:
        return ridge_summary

    validate_required_columns(
        data=xgboost_importance_data,
        required_columns=[
            "feature",
            "prediction_year",
            "xgboost_importance",
        ],
        dataset_name="XGBoost importance data",
    )

    xgboost_summary = (
        xgboost_importance_data.groupby(
            "feature"
        )["xgboost_importance"]
        .agg(
            xgboost_average_importance="mean",
            xgboost_importance_standard_deviation="std",
        )
        .reset_index()
    )

    return ridge_summary.merge(
        xgboost_summary,
        on="feature",
        how="outer",
        validate="one_to_one",
    )