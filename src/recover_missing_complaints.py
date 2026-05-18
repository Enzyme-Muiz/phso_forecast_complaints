import pandas as pd


def recover_missing_complaints(df):

    complaints = df["complaints"].copy()

    for i in range(3, len(df) - 3):

        # Skip if complaints already exists
        if pd.notna(complaints.iloc[i]):
            continue

        # Need centered mean available
        mean_val = df.loc[i, "centered_7d_mean"]

        if pd.isna(mean_val):
            continue

        # Get 7-day window
        window = complaints.iloc[i - 3 : i + 4]

        # Recover only if exactly one missing value
        if window.isna().sum() == 1:

            known_sum = window.sum(skipna=True)

            missing_value = (mean_val * 7) - known_sum

            complaints.iloc[i] = missing_value

    df["recovered_complaints"] = complaints

    return df
