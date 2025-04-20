import streamlit as st
import pandas as pd
import requests
import json
import time
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import SimpleExpSmoothing
from datetime import datetime

# --- Streamlit Setup ---
st.set_page_config(layout="wide")
st.title("Oil Market Dashboard: Prices & Production Forecasting")

# --- Section 1: EIA Oil Price Data ---
st.header("📈 Oil Spot Prices (EIA) V3")

api_key = st.text_input("Enter your EIA API Key", type="password")
start_date = st.date_input("Start Date", pd.to_datetime("2000-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))
selected_products = st.multiselect("Select Product(s)", ["EPCBRENT", "EPCWTI"], default=["EPCBRENT", "EPCWTI"])

def fetch_eia_data(api_key, start_date, end_date, products):
    all_records = []
    offset = 0
    while True:
        params = {
            "frequency": "daily",
            "data": ["value"],
            "facets": {"product": products},
            "start": str(start_date),
            "end": str(end_date),
            "sort": [{"column": "period", "direction": "asc"}],
            "length": 5000,
            "offset": offset
        }
        headers = {
            "Accept": "application/json",
            "X-Params": json.dumps(params)
        }
        response = requests.get("https://api.eia.gov/v2/petroleum/pri/spt/data/", headers=headers, params={"api_key": api_key})
        if response.status_code != 200:
            st.error(f"Failed to fetch EIA data: {response.status_code}")
            return None
        records = response.json().get("response", {}).get("data", [])
        if not records:
            break
        all_records.extend(records)
        offset += 5000
        time.sleep(1)
    return pd.DataFrame(all_records)

if st.button("Load Oil Price Data") and api_key:
    df_prices = fetch_eia_data(api_key, start_date, end_date, selected_products)
    if df_prices is not None and not df_prices.empty:
        df_prices['period'] = pd.to_datetime(df_prices['period'])
        df_prices['value'] = pd.to_numeric(df_prices['value'], errors='coerce')
        st.line_chart(df_prices.pivot(index='period', columns='product', values='value'))
        st.dataframe(df_prices)
    else:
        st.warning("No data found or error in fetch.")

# --- Section 2: Norwegian Production Data ---
st.header("🏭 Oil & Gas Production (Norwegian Petroleum)")

@st.cache_data
def load_production_data():
    url = "https://www.econdb.com/api/series/JODI_OIL.LD6EDLD6FCLD781.M.NO/?format=csv"
    # Assuming 'YOUR_API_KEY' is your actual API key from econdb.com
    # Replace 'YOUR_API_KEY' with your actual API Key.
    # Refer to econdb.com documentation for how to pass API Key, it might be in Headers or URL parameters.
    headers = {"Authorization": "Bearer 9b26b5d5fc3c3cd3bd0aca58d14a871215020ff3"}  # Example for Bearer token
    # Or as a URL parameter:
    # url = f"{url}&api_key=YOUR_API_KEY" 
    
    try:
        response = requests.get(url, headers=headers) # If API key is in headers
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        df = pd.read_csv(BytesIO(response.content), parse_dates=["Date"])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of error

    df = df.rename(columns={"Value": "Production (000 bbl/day)"})
    df = df[df["Production (000 bbl/day)"] > 0]
    return df

df_prod = load_production_data()
st.line_chart(df_prod.set_index("date"))
st.write("df_prod columns:", df_prod.columns.tolist())
st.write(df_prod.head())

# --- Section 3: Forecasting ---
st.header("🔮 Forecasting Future Oil Production")

periods = st.slider("Months to Forecast", 6, 60, 24)

# Apply Simple Exponential Smoothing
if st.button("Generate Forecast"):
    model = SimpleExpSmoothing(df_prod["Production (000 bbl/day)"]).fit()
    forecast_index = pd.date_range(start=df_prod["Date"].max(), periods=periods+1, freq="MS")[1:]
    forecast = model.forecast(periods)
    
    forecast_df = pd.DataFrame({"Date": forecast_index, "Forecast": forecast})
    df_plot = df_prod.copy()
    df_plot["Type"] = "Historical"
    forecast_df["Type"] = "Forecast"
    forecast_df["Production (000 bbl/day)"] = forecast_df["Forecast"]

    combined = pd.concat([df_plot[["Date", "Production (000 bbl/day)", "Type"]], forecast_df[["Date", "Production (000 bbl/day)", "Type"]]])

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=combined, x="Date", y="Production (000 bbl/day)", hue="Type", ax=ax)
    st.pyplot(fig)
    st.dataframe(forecast_df)
