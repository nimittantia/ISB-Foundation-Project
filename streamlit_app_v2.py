# newapptest.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# Load the trained ARIMA/SARIMAX model
@st.cache_resource
def load_trained_model():
    model = joblib.load('arima_model.pkl')
    return model

# Streamlit App
st.set_page_config(page_title="Oil Production Forecast", layout="centered")

st.title("🛢️ Oil Production Forecasting App")

try:
    data = pd.read_csv('economic_data.csv')

    # Parse Date column
    data['Date'] = pd.to_datetime(data['Date'])
    last_date = data['Date'].max()

    values = data['OILPRODUS'].values.reshape(-1, 1)
    historical_dates = data['Date']

    model = load_trained_model()

    # User Inputs: Short-Term and Long-Term forecasts together
    st.subheader("🔮 Forecast Future Oil Production")
    col1, col2 = st.columns(2)

    with col1:
        short_term_steps = st.number_input(
            "Short-Term Forecast (months)",
            min_value=1,
            max_value=60,
            value=12,
            step=1
        )
    with col2:
        long_term_steps = st.number_input(
            "Long-Term Forecast (months)",
            min_value=1,
            max_value=120,
            value=24,
            step=1
        )

    if st.button("🚀 Forecast"):
        # Short-Term Forecast
        short_term_forecast = model.forecast(steps=short_term_steps)
        short_term_forecast = np.array(short_term_forecast)

        short_term_dates = pd.date_range(start=pd.to_datetime('2025-02-01'), periods=short_term_steps, freq='MS')
        short_term_dates = short_term_dates.strftime('%Y-%m-%d')

        short_forecast_df = pd.DataFrame({
            "Date": short_term_dates,
            "Forecasted_Oil_Production": np.round(short_term_forecast, 2)
        })

        # Long-Term Forecast
        long_term_forecast = model.forecast(steps=long_term_steps)
        long_term_forecast = np.array(long_term_forecast)

        long_term_dates = pd.date_range(start=pd.to_datetime('2025-02-01'), periods=long_term_steps, freq='MS')
        long_term_dates = long_term_dates.strftime('%Y-%m-%d')

        long_forecast_df = pd.DataFrame({
            "Date": long_term_dates,
            "Forecasted_Oil_Production": np.round(long_term_forecast, 2)
        })

        # 📈 Short-Term Forecast Graph
        st.subheader("📈 Short-Term Forecast")
        mask_2000 = historical_dates >= pd.to_datetime('2000-01-01')
        filtered_dates = historical_dates[mask_2000]
        filtered_values = values.flatten()[mask_2000]

        fig_short = go.Figure()
        fig_short.add_trace(go.Scatter(x=filtered_dates, y=filtered_values, mode='lines', name='Actual Production (Historical)', line=dict(color='blue')))
        fig_short.add_trace(go.Scatter(x=short_term_dates, y=short_term_forecast, mode='lines', name='Short-Term Forecast', line=dict(color='red', dash='dash')))

        fig_short.update_layout(
            xaxis_title="Date",
            yaxis_title="Oil Production",
            legend_title="Legend",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_short, use_container_width=True)

        # 📋 Short-Term Forecast Table
        st.subheader("📋 Short-Term Forecasted Values")
        st.dataframe(short_forecast_df)

        # ⬇️ Short-Term Download Button
        csv_short = short_forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Short-Term Forecasted Data as CSV",
            data=csv_short,
            file_name='short_term_forecast.csv',
            mime='text/csv'
        )

        st.divider()

        # 📈 Long-Term Forecast Graph
        st.subheader("📈 Long-Term Forecast")

        fig_long = go.Figure()
        fig_long.add_trace(go.Scatter(x=filtered_dates, y=filtered_values, mode='lines', name='Actual Production (Historical)', line=dict(color='blue')))
        fig_long.add_trace(go.Scatter(x=long_term_dates, y=long_term_forecast, mode='lines', name='Long-Term Forecast', line=dict(color='green', dash='dash')))

        fig_long.update_layout(
            xaxis_title="Date",
            yaxis_title="Oil Production",
            legend_title="Legend",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_long, use_container_width=True)

        # 📋 Long-Term Forecast Table
        st.subheader("📋 Long-Term Forecasted Values")
        st.dataframe(long_forecast_df)

        # ⬇️ Long-Term Download Button
        csv_long = long_forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Long-Term Forecasted Data as CSV",
            data=csv_long,
            file_name='long_term_forecast.csv',
            mime='text/csv'
        )

except FileNotFoundError:
    st.error("❗ The file 'economic_data.csv' was not found. Please ensure it is placed in the same folder as app.py.")
except KeyError:
    st.error("❗ Column 'Date' or 'OILPRODUS' not found in economic_data.csv. Please check your file.")
