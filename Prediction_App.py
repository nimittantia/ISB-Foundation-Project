import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dateutil.relativedelta import relativedelta

# Page config
st.set_page_config(page_title="Oil Price Forecasting", layout="wide")

# Function to load short-term data and model
@st.cache_data
def load_short_term_model():
    try:
        with open("shortterm_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        import streamlit as st
        st.error(f"Error loading short-term model: {e}")
        return None

@st.cache_resource
def load_long_term_model():
    try:
        with open("norway_oil_forecast_montecarlo.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        import streamlit as st
        st.error(f"Error loading pickle file: {e}")
        return None

# Function to load long-term data and model
@st.cache_data
def load_long_term_data():
    df = pd.read_csv("Long_Term_Data.csv")
    df = df.rename(columns={
        'Year': 'Period',
        'Yearly Average Value': 'Price',
        'Production_TBPD': 'Production'
    })
    df['Period'] = pd.to_datetime(df['Period'])  # Automatically handles various date formats
    return df

@st.cache_resource
def load_long_term_model():
    with open("norway_oil_forecast_montecarlo.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Sidebar for selecting between short-term and long-term predictions
option = st.sidebar.radio("Select Prediction Type", ("Short-Term Prediction", "Long-Term Prediction"))

# Short-term Prediction (SARIMA model)
if option == "Short-Term Prediction":
    st.title("📈 Short-Term Oil Price Forecasting (SARIMA)")
    st.write("This app uses a trained SARIMA model to forecast short-term oil prices based on historical data.")

    # Load short-term model and data
    model = load_short_term_model()
    data = load_short_term_data()

    # Show training data
    st.subheader("Training Data")
    st.write(data.head())

    # Forecast horizon input
    st.subheader("Forecasting Parameters")
    forecast_horizon = st.slider("Select forecast horizon (months):", 1, 12, 6)

    # Forecast button
    if st.button("Forecast"):
        try:
            # Find the last date in the training data
            last_date = data.index[-1]
            
            # Desired forecast start
            forecast_start = pd.to_datetime("2025-01-01")

            # Compute number of steps to get from last date to desired start + horizon
            months_to_skip = (forecast_start.year - last_date.year) * 12 + (forecast_start.month - last_date.month)
            total_steps = months_to_skip + forecast_horizon

            # Forecast using SARIMA model
            forecast_result = model.get_forecast(steps=total_steps)
            forecast_values = forecast_result.predicted_mean

            # Slice only the future window we want
            forecast = forecast_values[-forecast_horizon:]

            # Format forecast
            forecast_df = pd.DataFrame({
                "Period": forecast.index.strftime("%Y-%m-%d"),
                "Predicted_mean": forecast.values
            })

            # Display forecast
            st.subheader(f"Forecasted Values from {forecast_start.strftime('%Y-%m-%d')}")
            st.write(forecast_df)

        except Exception as e:
            st.error(f"Error occurred: {e}")

# Long-term Prediction (Linear Regression + Monte Carlo Simulation)
elif option == "Long-Term Prediction":
    st.title("📈 Long-Term Oil Production Forecasting")
    st.write("This app uses a trained Linear Regression model and Monte Carlo simulations to forecast long-term oil production.")

    # Load long-term model and data
    df = load_long_term_data()
    model = load_long_term_model()

    # Show raw data
    with st.expander("See Raw Data"):
        st.dataframe(df)

    # Forecast parameters
    st.sidebar.header("🔧 Forecast Settings")
    forecast_years = st.sidebar.slider("Number of Years to Forecast:", min_value=1, max_value=5, value=5, step=1)  # Max set to 5 years
    n_simulations = st.sidebar.slider("Monte Carlo Simulations:", min_value=100, max_value=2000, value=1000, step=100)

    # Prepare data for forecasting
    df_model = df[['Period', 'Price', 'Production']].dropna()
    X = df_model[['Price']]
    y = df_model['Production']
    last_date = df_model['Period'].max()
    future_price_mean = X.mean().values[0]
    future_price_std = X.std().values[0]

    # Monte Carlo Simulation
    simulated_forecasts = np.zeros((forecast_years, n_simulations))
    for i in range(n_simulations):
        future_prices = np.random.normal(loc=future_price_mean, scale=future_price_std, size=forecast_years)
        future_preds = model.predict(future_prices.reshape(-1, 1))
        simulated_forecasts[:, i] = future_preds

    forecast_mean = simulated_forecasts.mean(axis=1)
    forecast_lower = np.percentile(simulated_forecasts, 5, axis=1)
    forecast_upper = np.percentile(simulated_forecasts, 95, axis=1)
    future_dates = [last_date + pd.DateOffset(years=i) for i in range(1, forecast_years + 1)]

    # Display forecast results
    st.subheader(f"📅 Forecasted Production (Next {forecast_years} Years)")

    # Ensure 'Year' is an integer and no comma formatting
    forecast_df = pd.DataFrame({
        "Year": [int(d.year) for d in future_dates],  # Force the year to be an integer
        "Mean Production": forecast_mean.round(2),
        "Lower 5%": forecast_lower.round(2),
        "Upper 95%": forecast_upper.round(2)
    })

    # Display dataframe
    st.dataframe(forecast_df.style.format({'Year': '{:d}', 'Mean Production': '{:.2f}', 'Lower 5%': '{:.2f}', 'Upper 95%': '{:.2f}'}))

    # Plot Results
    st.subheader("📊 Forecast Plot")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=df_model['Period'], y=df_model['Production'], label='Historical Production', ax=ax, color='blue')
    sns.lineplot(x=future_dates, y=forecast_mean, label='Forecast Mean', ax=ax, color='orange')
    ax.fill_between(future_dates, forecast_lower, forecast_upper, alpha=0.3, label='90% Confidence Interval', color='orange')
    ax.set_title("Monte Carlo Forecast: Norway Oil Production (Long-Term)", fontsize=14)
    ax.set_ylabel("Production (TBPD)")
    ax.set_xlabel("Year")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
