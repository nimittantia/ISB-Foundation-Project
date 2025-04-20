import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# --- Load Data ---
@st.cache_data
def load_data():
    df_longterm = pd.read_csv("norway_oil_forecast_montecarlo.csv", parse_dates=["date"])
    df_shortterm = pd.read_csv("df_short_term_forecasted.csv", parse_dates=["date"])
    return df_longterm, df_shortterm

@st.cache_data
def load_models():
    with open("norway_oil_forecast_montecarlo.pkl", "rb") as f:
        model_longterm = pickle.load(f)
    with open("shortterm_model.pkl", "rb") as f:
        model_shortterm = pickle.load(f)
    return model_longterm, model_shortterm

# --- Main App ---
st.set_page_config(page_title="Norway Oil Production Forecast", layout="wide")
st.title("📈 Norway Oil Production Forecasting Dashboard")

df_longterm, df_shortterm = load_data()
model_longterm, model_shortterm = load_models()

# --- Sidebar Filters ---
st.sidebar.header("Filters")
forecast_type = st.sidebar.radio("Select Forecast Type", ["Short Term", "Long Term"])

# --- Visualizations ---
if forecast_type == "Short Term":
    st.subheader("Short Term Forecast (Using Prophet Model)")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_shortterm, x="date", y="production", label="Forecasted", ax=ax)
    ax.set_title("Short Term Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Oil Production")
    st.pyplot(fig)

    st.dataframe(df_shortterm.tail(10), use_container_width=True)

else:
    st.subheader("Long Term Forecast (Using Monte Carlo Simulation)")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_longterm, x="date", y="P10", label="P10", ax=ax)
    sns.lineplot(data=df_longterm, x="date", y="P50", label="P50", ax=ax)
    sns.lineplot(data=df_longterm, x="date", y="P90", label="P90", ax=ax)
    ax.set_title("Long Term Forecast (P10, P50, P90)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Oil Production")
    st.pyplot(fig)

    st.dataframe(df_longterm.tail(10), use_container_width=True)

# --- Model Info ---
st.sidebar.markdown("### Model Details")
st.sidebar.markdown("""
- **Short Term Model**: Trained Prophet model.
- **Long Term Model**: Monte Carlo Simulation using historical decline curves and uncertainties.
""")
