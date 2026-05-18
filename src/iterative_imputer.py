from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np


def impute_missing_values(
    df,
    impute_cols=["staffing_level_fte", "channel_mix_index", "backlog_days"],
    predictor_cols=[
        "recovered_complaints",
        "media_mentions",
        "staffing_level_fte",
        "channel_mix_index",
        "backlog_days",
    ],
    seed=42,
    max_iter=50,
):

    # Store rows that were originally missing
    missing_mask = df[impute_cols].isna()

    # Copy data for imputation
    temp_df = df[predictor_cols].copy()

    # Create imputer
    imputer = IterativeImputer(random_state=seed, max_iter=max_iter)

    # Impute
    imputed_array = imputer.fit_transform(temp_df)

    imputed_df = pd.DataFrame(imputed_array, columns=predictor_cols, index=df.index)

    # Replace only imputed columns
    for col in impute_cols:
        df[col] = imputed_df[col]

    # -----------------------------------
    # Show previously missing rows
    # -----------------------------------
    for col in impute_cols:

        missing_rows = missing_mask[col]

        if missing_rows.sum() > 0:

            print(f"\nImputed values for: {col}")

            print(df.loc[missing_rows, ["date", "row_id", col]])

    return df, imputer
