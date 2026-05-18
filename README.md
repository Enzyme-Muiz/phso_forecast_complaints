## HOW TO RUN THE CODE
1. git clone https://github.com/Enzyme-Muiz/phso_forecast_complaints.git
2. pip install uv
3. uv run main.py

## WHAT DOES THE main.py DO?





## HOW TO RUN THE dev.ipynb NOTEBOOK





## EXPLANATION OF THE main.py


````markdown
## Forecasting Approach

The forecasting pipeline was designed to predict future `recovered_complaints` over the configured forecast horizon. The approach follows a structured end-to-end workflow covering data validation, missing value treatment, feature preparation, model training, forecasting, visualization, and model persistence.

### 1. Data Loading and Configuration

The pipeline begins by loading configuration values from `.config/analytics.toml`, including the random seed, forecast horizon, and imputation settings. The input data is then read from the Excel file using the `daily records` sheet.

Using a configuration file allows the pipeline to be more reusable and easier to adjust without hard-coding parameters directly into the script.

### 2. Data Validation

Before modelling, the dataset is passed through a validation step. This checks for issues such as:

- missing dates
- invalid weekend flags
- invalid bank holiday flags
- invalid `channel_mix_index` values
- negative `media_mentions`
- missing or incorrectly ordered `row_id`
- dates not in chronological order

Any validation issues are logged so that data quality problems can be reviewed before trusting the forecast output.

### 3. Missing Complaint Recovery

The `recovered_complaints` column is created using the relationship between `complaints` and `centered_7d_mean`.

Where `complaints` is missing but `centered_7d_mean` is available, the missing complaint value is recovered from the 7-day centred average where possible. This helps preserve the target signal rather than simply dropping missing complaint rows.

### 4. Imputation of Exogenous Variables

After recovering complaints, missing values in key explanatory variables are imputed using `IterativeImputer`.

The imputed columns are:

- `staffing_level_fte`
- `channel_mix_index`
- `backlog_days`

The predictors used for imputation include:

- `recovered_complaints`
- `media_mentions`
- `staffing_level_fte`
- `channel_mix_index`
- `backlog_days`

A fixed seed is used to make the imputation reproducible. The fitted imputer is saved to disk as:

```text
model/iterative_imputer.pkl
````

This allows the same imputation logic to be reused later.

### 5. Feature Preparation

The forecasting model uses both time-based and business-related features. These include known calendar variables such as:

* weekend flag
* UK bank holiday flag
* day of week
* month
* day of year

It also uses exogenous business drivers such as:

* staffing level
* channel mix index
* backlog days
* media mentions

These variables help the model explain changes in complaint volume beyond simple time trend and seasonality.

### 6. Multi-Model Forecasting

The forecasting step uses a multi-model approach. Multiple candidate models are trained and compared, including models such as:

* Ridge regression
* Random Forest
* HistGradientBoosting

Each model is evaluated using time-series-aware validation, rather than random splitting. This is important because time series data must preserve chronological order during training and testing.

The best-performing model is selected based on forecast error, and it is then used to generate the final forecast for `recovered_complaints`.

### 7. Forecasting Exogenous Variables

Because the final complaints forecast requires future values of exogenous variables, the pipeline first estimates future values for the exogenous drivers. These forecasted drivers are then passed into the final target forecasting model.

Known future features, such as weekend and bank holiday flags, are generated directly from future dates rather than forecasted.

### 8. Evaluation Metrics

The model is evaluated using forecasting-specific error metrics:

* **MAPE**: Mean Absolute Percentage Error
* **MASE**: Mean Absolute Scaled Error

MAPE provides an interpretable percentage error, while MASE compares the model against a naive seasonal forecast. MASE is especially useful for time series because it indicates whether the model improves on a simple baseline.

### 9. Forecast Visualization

The final 90-day forecast is plotted against the historical `recovered_complaints` values. This allows visual inspection of the forecast trend and whether the predicted values are consistent with recent historical patterns.

The plot is saved as:

```text
results/forecast_vs_historical.png
```

### 10. Model Persistence

The best forecasting model is saved to disk as:

```text
model/best_model.pkl
```

Saving the model ensures that the trained model can be reused for future predictions without needing to retrain the full pipeline.

## Summary

Overall, the forecasting solution follows a reproducible machine learning workflow:

```text
Load data
→ Validate data
→ Recover missing complaints
→ Impute missing exogenous variables
→ Engineer calendar and lag-based features
→ Train and compare multiple forecasting models
→ Select best model
→ Generate forecast
→ Visualize forecast
→ Save model artifacts
```

This approach combines data quality checks, statistically informed missing value treatment, exogenous feature modelling, time-series validation, and reproducible model saving to produce a robust forecast of future complaint volumes.

```
```


