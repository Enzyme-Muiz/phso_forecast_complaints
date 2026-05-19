##### This is a template for making src available in any folder.
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.logging_config import setup_logger
from src.utils.config_loader import load_config
from src.validation import validate_data
from src.recover_missing_complaints import recover_missing_complaints
from src.iterative_imputer import impute_missing_values
from src.visualizations import plot_time_series, plot_forecast
from src.model_training import forecast_recovered_complaints_90_days

###### Set up logger
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__, base_filename="pipeline")

logger.info("Started run")


# logging = setup_logging()


### Load config
logger.info("Loading config")
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.config_loader import load_config

config = load_config(".config/analytics.toml")
print(config)
SEED = config["SEED"]
forecast_horizon = config["forecast_horizon"]
mlflowlog = config["mlflowlog"]

logger.info(f"Config loaded successfully as {config}")


### load libaries
logger.info("Loading libraries")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Enable IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import pandas as pd
import holidays
import optuna
import joblib

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

logger.info("Libraries loaded successfully")


####mlflow setup
if mlflowlog:
    import os
    import mlflow

    mlflow_tracking_uri = f"file:///{os.getcwd()}/{config['mlflow_tracking_uri']}"
    print(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(config["mlflow_experiment_name"])
    mlflow.sklearn.autolog()
    logger.info("MLflow setup completed successfully")

## loading data
logger.info("Loading data")
df1 = pd.read_excel(
    "data/inputs/Principle_Data_Scientist_Tech_Assessment.xlsx",
    sheet_name="daily records",
)
logger.info("Data loaded successfully")


# Validate the loaded data
logger.info("Validating data")
validation_errors = validate_data(df1)
for error_type, df in validation_errors.items():
    if not df.empty:
        logger.warning(
            f"Found {len(df)} rows with {error_type.replace('_', ' ').title()}"
        )
    else:
        logger.info(f"No issues found for {error_type.replace('_', ' ').title()}")


## missing data points
logger.info("Checking for missing data points")
# Count NaN values in each column
nan_counts = df1.isna().sum()

print(nan_counts)
logger.info(
    f"Missing data points checked successfully and the counts are printed above as {nan_counts}"
)


## Imputation
logger.info("Starting imputation of missing data points")
df1 = recover_missing_complaints(df1)

recovered_aspect = df1[df1["complaints"].isna()][
    ["date", "complaints", "recovered_complaints", "centered_7d_mean"]
]
logger.info(f"Recovered complaints for missing data points:\n{recovered_aspect}")

df1_imputed, imputer = impute_missing_values(
    df1, seed=SEED, max_iter=config["max_iter_iterative_imputer"]
)


# Save imputer model
joblib.dump(imputer, "model/iterative_imputer.pkl")

print("Model saved successfully.")

logger.info("Imputation completed successfully")


### visualization
logger.info("Starting visualization of time series data")
plot_time_series(df1_imputed, date_col="date", columns=config["columns_for_viz"])
logger.info("Visualization completed successfully")


## forecasting
logger.info("Starting model training and forecasting")
forecast_90d, model_info, best_model = forecast_recovered_complaints_90_days(
    df1_imputed, horizon=forecast_horizon
)


###visualize forecast and save the plot and the forecast data
logger.info("Visualizing forecast vs historical data")
result = plot_forecast(
    historical_df=df1_imputed,
    forecast_df=forecast_90d,
    historical_col="recovered_complaints",
    forecast_col=f"forecast_recovered_complaints",
    title=f"{forecast_horizon}-Day Forecast of Recovered Complaints",
)
result.savefig("results/forecast_vs_historical.png")
logger.info("Forecast visualization completed successfully")


# Save best model
joblib.dump(best_model, "model/best_model.pkl")


logger.info("Best model saved successfully")
logger.info("Save the forecasted data")
forecast_90d.to_csv("results/forecast_90d.csv", index=False)
logger.info("Forecasted data saved successfully")

# uv  add ipykernel
# uv run -m ipykernel install --user --name=.venv --display-name "Python (forecast_env)"
## add mlflow
