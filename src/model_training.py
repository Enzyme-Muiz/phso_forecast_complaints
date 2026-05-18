import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import holidays
import optuna
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.config_loader import load_config

config = load_config(".config/analytics.toml")
SEED = config["SEED"]
imputer_trial_count = config["imputer_trial_count"]
model_names = config["model_names"]
forecast_horizon = config["forecast_horizon"]

np.random.seed(SEED)


# -----------------------------
# Metrics
# -----------------------------
def mase(y_true, y_pred, y_train, seasonality=7):
    naive_error = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    model_error = np.mean(np.abs(y_true - y_pred))
    return model_error / naive_error


def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100


# -----------------------------
# Date features + UK holidays
# -----------------------------
def add_date_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    years = range(df["date"].dt.year.min(), df["date"].dt.year.max() + 1)
    uk_holidays = holidays.UK(years=years)

    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    df["bank_holiday_flag"] = (
        df["date"].dt.normalize().isin(pd.to_datetime(list(uk_holidays.keys())))
    ).astype(int)

    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear

    return df


# -----------------------------
# Lag features
# -----------------------------
def create_lag_features(df, target_col, lags=[1, 7, 14, 28]):
    df = df.copy()

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    df[f"{target_col}_rolling_7"] = df[target_col].shift(1).rolling(7).mean()

    return df


# -----------------------------
# Model factory
# -----------------------------
def get_model(trial, model_name):
    if model_name == "ridge":
        alpha = trial.suggest_float("alpha", 0.01, 100, log=True)
        return Ridge(alpha=alpha)

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=SEED,
            n_jobs=-1,
        )

    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            max_iter=trial.suggest_int("max_iter", 100, 500),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 10, 50),
            random_state=SEED,
        )


# -----------------------------
# Bayesian tuning with Optuna
# -----------------------------
def tune_model(X, y, model_name, seed=SEED, n_trials=imputer_trial_count):
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        model = get_model(trial, model_name)
        scores = []

        for train_idx, valid_idx in tscv.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_valid)

            score = mase(y_valid, preds, y_train.values)
            scores.append(score)

        return np.mean(scores)

    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    best_model = get_model(optuna.trial.FixedTrial(study.best_params), model_name)

    best_model.fit(X, y)

    return best_model, study.best_value, study.best_params


# -----------------------------
# Forecast one column recursively
# -----------------------------
def forecast_column(
    df, target_col, models=model_names, forecast_horizon=forecast_horizon
):
    df = add_date_features(df)

    df_model = create_lag_features(df, target_col)
    df_model = df_model.dropna()

    feature_cols = [
        "is_weekend",
        "bank_holiday_flag",
        "dayofweek",
        "month",
        "dayofyear",
        f"{target_col}_lag_1",
        f"{target_col}_lag_7",
        f"{target_col}_lag_14",
        f"{target_col}_lag_28",
        f"{target_col}_rolling_7",
    ]

    X = df_model[feature_cols]
    y = df_model[target_col]

    best = None

    for model_name in models:
        model, score, params = tune_model(
            X, y, seed=SEED, n_trials=imputer_trial_count, model_name=model_name
        )

        if best is None or score < best["score"]:
            best = {
                "model_name": model_name,
                "model": model,
                "score": score,
                "params": params,
            }

    history = df[["date", target_col]].copy()

    future_dates = pd.date_range(
        start=df["date"].max() + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq="D",
    )

    forecasts = []

    for future_date in future_dates:
        temp = pd.DataFrame({"date": [future_date]})

        temp[target_col] = np.nan

        combined = pd.concat([history, temp], ignore_index=True)

        combined = add_date_features(combined)
        combined = create_lag_features(combined, target_col)

        X_future = combined.iloc[[-1]][feature_cols]

        pred = best["model"].predict(X_future)[0]

        forecasts.append(pred)

        history = pd.concat(
            [history, pd.DataFrame({"date": [future_date], target_col: [pred]})],
            ignore_index=True,
        )

    forecast_df = pd.DataFrame({"date": future_dates, target_col: forecasts})

    return forecast_df, best


# -----------------------------
# Full multi-model forecasting pipeline
# -----------------------------


def forecast_recovered_complaints_90_days(
    df,
    horizon=forecast_horizon,
    exog_cols=[
        "staffing_level_fte",
        "channel_mix_index",
        "backlog_days",
        "media_mentions",
    ],
    target_col="recovered_complaints",
    known_exog_future=[
        "is_weekend",
        "bank_holiday_flag",
    ],
    known_derived_future=["dayofweek", "month", "dayofyear"],
    seed=SEED,
):

    np.random.seed(seed)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # -----------------------------------
    # Create future dataframe
    # -----------------------------------
    future_df = pd.DataFrame(
        {
            "date": pd.date_range(
                start=df["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D"
            )
        }
    )

    future_df = add_date_features(future_df)

    model_info = {}

    # -----------------------------------
    # Forecast exogenous variables
    # -----------------------------------
    for col in exog_cols:

        print(f"\nForecasting exogenous column: {col}")

        col_forecast, best_model = forecast_column(
            df[["date", col]].dropna(), target_col=col, forecast_horizon=horizon
        )

        future_df[col] = col_forecast[col].values

        model_info[col] = best_model

    # -----------------------------------
    # Prepare target dataframe
    # -----------------------------------
    required_cols = ["date", target_col] + known_exog_future + exog_cols

    df_target = df[required_cols].copy()

    # -----------------------------------
    # Append future exogenous values
    # -----------------------------------
    full_df = pd.concat(
        [df_target, future_df[["date"] + known_exog_future + exog_cols]],
        ignore_index=True,
    )

    # -----------------------------------
    # Create lag features
    # -----------------------------------
    full_df = add_date_features(full_df)
    full_df = create_lag_features(full_df, target_col)

    lag_features = [
        f"{target_col}_lag_1",
        f"{target_col}_lag_7",
        f"{target_col}_lag_14",
        f"{target_col}_lag_28",
        f"{target_col}_rolling_7",
    ]

    feature_cols = known_exog_future + exog_cols + known_derived_future + lag_features

    # -----------------------------------
    # Training set
    # -----------------------------------
    train_df = full_df[full_df[target_col].notna()].dropna()
    train_df.to_csv("data/outputs/train_df.csv", index=False)
    X = train_df[feature_cols]
    y = train_df[target_col]

    # -----------------------------------
    # Multi-model selection
    # -----------------------------------
    candidate_models = ["ridge", "random_forest", "hist_gradient_boosting"]

    best = None

    for model_name in candidate_models:

        print(f"\nTuning model: {model_name}")

        model, score, params = tune_model(X, y, model_name)

        print(f"MAPE: {score:.4f}")

        if best is None or score < best["score"]:

            best = {
                "model_name": model_name,
                "model": model,
                "score": score,
                "params": params,
            }

    # -----------------------------------
    # Recursive forecasting
    # -----------------------------------
    forecasts = []

    for i in range(horizon):

        row_idx = len(df) + i

        X_future = full_df.loc[[row_idx], feature_cols]

        pred = best["model"].predict(X_future)[0]

        full_df.loc[row_idx, target_col] = pred

        forecasts.append(pred)

        # Recreate lag features
        full_df = create_lag_features(full_df, target_col)

    # -----------------------------------
    # Final forecast dataframe
    # -----------------------------------
    future_df[f"forecast_{target_col}"] = forecasts

    model_info[target_col] = best

    # -----------------------------------
    # Evaluation metrics
    # -----------------------------------
    train_preds = best["model"].predict(X)

    final_mape = mape(y, train_preds)

    final_mase = mase(y_true=y.values, y_pred=train_preds, y_train=y.values)

    print("\n==========================")
    print("FINAL MODEL")
    print("==========================")
    print("Best model:", best["model_name"])
    print("MAPE:", round(final_mape, 4))
    print("MASE:", round(final_mase, 4))
    print("Best params:", best["params"])

    return future_df, model_info, best["model"]
