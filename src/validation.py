import pandas as pd
from datetime import datetime


###check if a date is missing
###check if weekend is 0 or 1
###check if holiday is 0 or 1
###channel_mix_index not less than 0 and not more than 100
### media mention cannot be less than 0
### row_id int missing


def validate_data(df):
    errors = {}

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # -----------------------------
    # Missing date
    # -----------------------------
    errors["missing_date"] = df[df["date"].isna()]

    # -----------------------------
    # Date not in chronological order
    # -----------------------------
    errors["date_not_in_chronology"] = df[df["date"].diff().dt.days < 0]

    # -----------------------------
    # row_id missing or not integer
    # -----------------------------
    errors["invalid_row_id"] = df[
        df["row_id"].isna()
        | (~df["row_id"].apply(lambda x: pd.isna(x) or float(x).is_integer()))
    ]

    # -----------------------------
    # row_id not in chronological order
    # -----------------------------
    errors["row_id_not_in_order"] = df[df["row_id"].diff() < 0]

    # -----------------------------
    # is_weekend must be 0 or 1
    # -----------------------------
    errors["invalid_is_weekend"] = df[~df["is_weekend"].isin([0, 1])]

    # -----------------------------
    # bank_holiday_flag must be 0 or 1
    # -----------------------------
    errors["invalid_bank_holiday_flag"] = df[~df["bank_holiday_flag"].isin([0, 1])]

    # -----------------------------
    # channel_mix_index between 0 and 100
    # NaN is allowed
    # -----------------------------
    errors["invalid_channel_mix_index"] = df[
        (~df["channel_mix_index"].between(0, 100)) & (df["channel_mix_index"].notna())
    ]

    # -----------------------------
    # media_mentions cannot be negative
    # -----------------------------
    errors["invalid_media_mentions"] = df[
        (df["media_mentions"] < 0) | (df["media_mentions"].isna())
    ]

    return errors
