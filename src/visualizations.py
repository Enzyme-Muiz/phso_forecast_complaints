import matplotlib.pyplot as plt


def plot_time_series(df, date_col, columns, figsize=(12, 6)):
    """
    Plot multiple time series on a single y-axis.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe

    date_col : str
        Name of date column

    columns : list
        List of columns to plot

    figsize : tuple
        Figure size
    """

    plt.figure(figsize=figsize)

    # Plot each column
    for col in columns:
        plt.plot(df[date_col], df[col], label=col)

    # Labels and title
    plt.xlabel(date_col)
    plt.ylabel("Value")
    plt.title("Time Series Plot")

    # Legend
    plt.legend()

    # Rotate dates for readability
    plt.xticks(rotation=45)

    # Tight layout
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_forecast(
    historical_df,
    forecast_df,
    date_col="date",
    historical_col="recovered_complaints",
    forecast_col="forecast_recovered_complaints",
    figsize=(12, 6),
    title="Forecast vs Historical",
):
    """
    Plot historical and forecasted time series.

    Parameters
    ----------
    historical_df : pd.DataFrame
        Historical dataframe

    forecast_df : pd.DataFrame
        Forecast dataframe

    date_col : str
        Date column name

    historical_col : str
        Historical target column

    forecast_col : str
        Forecast target column

    figsize : tuple
        Figure size

    title : str
        Plot title
    """

    plt.figure(figsize=figsize)

    # Historical data
    plt.plot(
        historical_df[date_col],
        historical_df[historical_col],
        label=f"Historical {historical_col}",
    )

    # Forecast data
    plt.plot(
        forecast_df[date_col],
        forecast_df[forecast_col],
        label=f"Forecast {historical_col}",
    )

    # Labels
    plt.xlabel("Date")
    plt.ylabel(historical_col)

    # Title
    plt.title(title)

    # Legend
    plt.legend()

    # Rotate dates
    plt.xticks(rotation=45)

    # Layout
    plt.tight_layout()

    # Show plot
    plt.show()
