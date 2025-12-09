import streamlit as st
import pandas as pd
import io
import numpy as np
from datetime import datetime

# Prophet import fallback logic
try:
    from prophet import Prophet
except:
    from fbprophet import Prophet

# Time series models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import plotly.express as px

# ----------------------------------------------------------
# STREAMLIT APP
# ----------------------------------------------------------
st.title("📈 Multi-Year Resource Forecasting Tool with Algorithm Choice")

st.write("""
Upload past resource data and generate monthly ML forecasts with:
- Multiple service lines  
- Growth %  
- Manual adjustments  
- Annual Attrition % → monthly replacement positions  
- Algorithm choice: Prophet, ARIMA, SARIMA, Exponential Smoothing  
- Downloadable forecast with cumulative annual replacement  
""")

# ----------------------------------------------------------
# Sample File Download
# ----------------------------------------------------------
st.subheader("📄 Download Sample Excel Template")
sample_df = pd.DataFrame({
    "Month": pd.date_range("2023-01-01", periods=6, freq="MS"),
    "ServiceLine": ["DataScience", "DataScience", "Cloud", "Cloud", "AI", "AI"],
    "Resources": [12, 15, 20, 18, 10, 13]
})

buffer_sample = io.BytesIO()
with pd.ExcelWriter(buffer_sample, engine="xlsxwriter") as writer:
    sample_df.to_excel(writer, index=False, sheet_name="Sample")

st.download_button(
    label="📥 Download Sample Excel",
    data=buffer_sample,
    file_name="sample_resource_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.divider()

# ----------------------------------------------------------
# File Upload
# ----------------------------------------------------------
st.header("Step 1: Upload Past Resource Data")
uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    # Validate required columns
    required_cols = {"Month", "ServiceLine", "Resources"}
    if not required_cols.issubset(df.columns):
        st.error(f"❌ File must contain columns: {required_cols}")
        st.stop()

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    if df["Month"].isna().any():
        st.error("❌ Invalid Month format detected.")
        st.stop()

    # ----------------------------------------------------------
    # Select service lines
    # ----------------------------------------------------------
    st.header("Step 2: Select Service Lines to Forecast")
    service_lines = sorted(df["ServiceLine"].unique().tolist())
    selected_services = st.multiselect(
        "Choose Service Lines",
        service_lines,
        default=service_lines
    )

    if not selected_services:
        st.warning("Please select at least one service line.")
        st.stop()

    # ----------------------------------------------------------
    # Forecast Settings
    # ----------------------------------------------------------
    st.header("Step 3: Forecast Settings")
    forecast_years = st.slider("Years to Forecast", 1, 5, 2)
    periods = forecast_years * 12

    st.subheader("ML Algorithm Selection")
    algorithm = st.selectbox(
        "Choose Forecasting Algorithm",
        ["Prophet", "ARIMA", "Exponential Smoothing"]
    )

    st.subheader("Growth / Adjustment / Attrition Settings")
    growth_inputs = {}
    adjust_inputs = {}
    attrition_inputs = {}

    for sl in selected_services:
        st.markdown(f"### {sl}")
        cols = st.columns(3)

        with cols[0]:
            growth_inputs[sl] = st.number_input(
                f"% Growth for {sl}",
                value=0.0,
                step=0.5
            )

        with cols[1]:
            adjust_inputs[sl] = st.number_input(
                f"Manual Adjustment for {sl}",
                value=0.0,
                step=1.0
            )

        with cols[2]:
            attrition_inputs[sl] = st.number_input(
                f"Annual Attrition % for {sl}",
                value=0.0,
                step=0.5
            )

    # ----------------------------------------------------------
    # Run Forecast
    # ----------------------------------------------------------
    if st.button("Run Forecast"):
        st.header("📊 Forecast Results")
        all_forecasts = []

        for sl in selected_services:
            st.subheader(f"### Service Line: **{sl}**")
            df_sl = df[df["ServiceLine"] == sl].copy().sort_values("Month")
            df_sl.set_index("Month", inplace=True)

            # Base forecast array
            if algorithm == "Prophet":
                prophet_df = df_sl.reset_index().rename(columns={"Month": "ds", "Resources": "y"})
                model = Prophet()
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=periods, freq="MS")
                forecast = model.predict(future)
                forecast["yhat_final"] = forecast["yhat"]

            elif algorithm == "ARIMA":
                model = ARIMA(df_sl["Resources"], order=(1,1,1))
                model_fit = model.fit()
                pred = model_fit.forecast(steps=periods)
                future_dates = pd.date_range(df_sl.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
                forecast = pd.DataFrame({"ds": future_dates, "yhat_final": pred.values})

            elif algorithm == "Exponential Smoothing":
                model = ExponentialSmoothing(df_sl["Resources"], trend="add", seasonal=None)
                model_fit = model.fit()
                pred = model_fit.forecast(steps=periods)
                future_dates = pd.date_range(df_sl.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
                forecast = pd.DataFrame({"ds": future_dates, "yhat_final": pred.values})

            else:
                st.error("Algorithm not implemented.")
                st.stop()

            # Apply Growth %
            forecast["yhat_final"] *= (1 + growth_inputs[sl] / 100)

            # Apply Manual Adjustment
            forecast["yhat_final"] += adjust_inputs[sl]

            # Annual Attrition → Monthly Replacement
            annual_attr = attrition_inputs[sl] / 100
            monthly_attr = 1 - (1 - annual_attr) ** (1/12)
            forecast["Replacement"] = forecast["yhat_final"] * monthly_attr
            forecast["Final_Resource_Need"] = forecast["yhat_final"] + forecast["Replacement"]

            # Cumulative Replacement per Year
            forecast["Year"] = forecast["ds"].dt.year
            forecast["Cumulative_Replacement"] = forecast.groupby("Year")["Replacement"].cumsum()

            forecast["ServiceLine"] = sl
            forecast.rename(columns={"ds": "Month", "yhat_final": "Forecast"}, inplace=True)

            all_forecasts.append(forecast)

            # Plots
            fig1 = px.line(forecast, x="Month", y="Final_Resource_Need", title=f"{sl} Forecast")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.line(forecast, x="Month", y="Cumulative_Replacement", title=f"{sl} Cumulative Replacement")
            st.plotly_chart(fig2, use_container_width=True)

        # Combine all service lines
        final_df = pd.concat(all_forecasts, ignore_index=True)
        st.subheader("Download Forecast Output")
        st.dataframe(final_df, use_container_width=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            final_df.to_excel(writer, index=False, sheet_name="Forecast")

        st.download_button(
            label="Download Forecast Excel",
            data=buffer,
            file_name="resource_forecast_ml_choice.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
