import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Detail Type Anomaly Detection", layout="wide")

st.title("📊 Detail Type Seasonality & Anomaly Detection")

# -----------------------------
# SAFE CHI-SQUARE FUNCTION
# -----------------------------
def safe_chi_square(a, b, c, d):
    table = np.array([[a, b], [c, d]])

    # Skip invalid tables (zero row/column)
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return None

    try:
        chi2, p, _, expected = chi2_contingency(table, correction=True)
        return p
    except ValueError:
        return None


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df.columns = df.columns.str.lower().str.strip()

    if 'detail type' not in df.columns or 'date of occurrence' not in df.columns:
        st.error("CSV must contain 'Detail Type' and 'Date of Occurrence'")
        st.stop()

    # -----------------------------
    # PREP DATA
    # -----------------------------
    df['date of occurrence'] = pd.to_datetime(df['date of occurrence'])
    df['year'] = df['date of occurrence'].dt.year
    df['month'] = df['date of occurrence'].dt.month

    pivot = df.groupby(['year','month','detail type']).size().reset_index(name='count')

    # -----------------------------
    # SIDEBAR CONTROLS
    # -----------------------------
    st.sidebar.header("Controls")
    selected_year = st.sidebar.selectbox("Year", sorted(df['year'].unique()))
    selected_month = st.sidebar.selectbox("Month", range(1,13))

    current = pivot[(pivot['year']==selected_year) & (pivot['month']==selected_month)]
    prev_year = pivot[(pivot['year']==selected_year-1) & (pivot['month']==selected_month)]

    df['year_month'] = df['date of occurrence'].dt.to_period('M')
    selected_period = pd.Period(f"{selected_year}-{selected_month:02d}")

    # -----------------------------
    # 3-MONTH WINDOW
    # -----------------------------
    prev_3 = pivot[pivot.apply(
        lambda row: pd.Period(f"{row['year']}-{row['month']:02d}") in 
        pd.period_range(selected_period-3, selected_period-1, freq='M'),
        axis=1
    )]

    results = []
    detail_types = df['detail type'].unique()

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    for dt in detail_types:

        a = current[current['detail type']==dt]['count'].sum()
        total_current = current['count'].sum()

        b = prev_year[prev_year['detail type']==dt]['count'].sum()
        total_prev = prev_year['count'].sum()

        if total_current == 0 or total_prev == 0:
            continue

        # Minimum threshold (reduces noise)
        if a < 5 and b < 5:
            continue

        c = total_current - a
        d = total_prev - b

        current_rate = a / total_current
        prev_rate = b / total_prev

        # -----------------------------
        # YOY CHI-SQUARE
        # -----------------------------
        p_yoy = safe_chi_square(a, b, c, d)

        if p_yoy is None:
            continue

        direction_yoy = "Increase" if current_rate > prev_rate else "Decrease"
        result_yoy = f"Significant {direction_yoy}" if p_yoy < 0.05 else "Not Significant"

        # -----------------------------
        # 3-MONTH COMPARISON
        # -----------------------------
        prev3_dt = prev_3[prev_3['detail type']==dt]['count'].sum()
        prev3_total = prev_3['count'].sum()

        if prev3_total == 0:
            continue

        prev3_rate = prev3_dt / prev3_total

        p_3mo = safe_chi_square(a, prev3_dt, c, prev3_total - prev3_dt)

        if p_3mo is None:
            continue

        direction_3mo = "Increase" if current_rate > prev3_rate else "Decrease"
        result_3mo = f"Significant {direction_3mo}" if p_3mo < 0.05 else "Not Significant"

        # -----------------------------
        # STORE RESULTS
        # -----------------------------
        results.append({
            "Detail Type": dt,
            "Current Count": a,
            "Prev Year Count": b,
            "Prev 3mo Count": prev3_dt,
            "Current Rate": current_rate,
            "Prev Year Rate": prev_rate,
            "Prev 3mo Rate": prev3_rate,
            "Effect YoY": current_rate - prev_rate,
            "Effect 3mo": current_rate - prev3_rate,
            "YoY Result": result_yoy,
            "3mo Result": result_3mo
        })

    results_df = pd.DataFrame(results)

    # -----------------------------
    # SAFETY CHECK
    # -----------------------------
    if results_df.empty:
        st.warning("No valid statistical comparisons for this selection.")
        st.stop()

    # -----------------------------
    # TOP 5 ANOMALIES (COUNTS ONLY)
    # -----------------------------
    st.subheader("🏆 Top 5 Anomalies")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔺 Largest Increases")
        top_inc = results_df.sort_values("Effect YoY", ascending=False).head(5)
        st.dataframe(top_inc[["Detail Type", "Current Count"]])

    with col2:
        st.markdown("### 🔻 Largest Decreases")
        top_dec = results_df.sort_values("Effect YoY", ascending=True).head(5)
        st.dataframe(top_dec[["Detail Type", "Current Count"]])

    # -----------------------------
    # ALERTS
    # -----------------------------
    st.subheader("🚨 Significant Alerts")

    alerts = results_df[
        (results_df["YoY Result"].str.contains("Significant")) &
        (
            (abs(results_df["Effect YoY"]) > 0.02) |
            (abs(results_df["Effect 3mo"]) > 0.02)
        )
    ]

    st.dataframe(alerts.sort_values("Effect YoY", ascending=False))

    # -----------------------------
    # RISK DASHBOARD
    # -----------------------------
    st.subheader("🎯 Risk Dashboard")

    def color_effect(val):
        if val > 0.02:
            return "background-color: #ff9999"
        elif val > 0:
            return "background-color: #ffe6e6"
        elif val < -0.02:
            return "background-color: #99ff99"
        elif val < 0:
            return "background-color: #e6ffe6"
        return ""

    styled_df = results_df.style.applymap(
        color_effect,
        subset=["Effect YoY", "Effect 3mo"]
    )

    st.dataframe(styled_df)

    # -----------------------------
    # TREND CHART
    # -----------------------------
    st.subheader("📈 Trend Viewer")

    selected_dt = st.selectbox("Select Detail Type", detail_types)

    trend = pivot[pivot['detail type']==selected_dt].copy()
    trend['date'] = pd.to_datetime(trend[['year','month']].assign(day=1))

    st.line_chart(trend.set_index('date')['count'])
