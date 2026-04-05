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

    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return None

    try:
        chi2, p, _, _ = chi2_contingency(table, correction=True)
        return p
    except:
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
    # SIDEBAR
    # -----------------------------
    st.sidebar.header("Controls")
    selected_year = st.sidebar.selectbox("Year", sorted(df['year'].unique()))
    selected_month = st.sidebar.selectbox("Month", range(1,13))

    current = pivot[(pivot['year']==selected_year) & (pivot['month']==selected_month)]
    prev_year = pivot[(pivot['year']==selected_year-1) & (pivot['month']==selected_month)]

    # Last month
    prev_month_period = pd.Period(f"{selected_year}-{selected_month:02d}") - 1
    prev_month = pivot[
        (pivot['year']==prev_month_period.year) &
        (pivot['month']==prev_month_period.month)
    ]

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

        last_month_count = prev_month[prev_month['detail type']==dt]['count'].sum()

        if total_current == 0 or total_prev == 0:
            continue

        if a < 5 and b < 5:
            continue

        c = total_current - a
        d = total_prev - b

        current_rate = a / total_current
        prev_rate = b / total_prev

        # YOY
        p_yoy = safe_chi_square(a, b, c, d)
        if p_yoy is None:
            continue

        direction_yoy = "Increase" if current_rate > prev_rate else "Decrease"
        result_yoy = f"Significant {direction_yoy}" if p_yoy < 0.05 else "Not Significant"

        # 3-MONTH
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

        # Differences
        diff_last_month = a - last_month_count
        diff_yoy = a - b

        results.append({
            "Detail Type": dt,
            "Current Count": a,
            "Diff vs Last Month": diff_last_month,
            "Diff vs Last Year": diff_yoy,
            "Effect YoY": current_rate - prev_rate,
            "Effect 3mo": current_rate - prev3_rate,
            "YoY Result": result_yoy,
            "3mo Result": result_3mo
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        st.warning("No valid statistical comparisons for this selection.")
        st.stop()

    # -----------------------------
    # TOP 5 ANOMALIES
    # -----------------------------
    st.subheader("🏆 Top 5 Anomalies")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔺 Largest Increases")
        top_inc = results_df.sort_values("Effect YoY", ascending=False).head(5)
        st.dataframe(top_inc[[
            "Detail Type",
            "Current Count",
            "Diff vs Last Month",
            "Diff vs Last Year"
        ]])

    with col2:
        st.markdown("### 🔻 Largest Decreases")
        top_dec = results_df.sort_values("Effect YoY", ascending=True).head(5)
        st.dataframe(top_dec[[
            "Detail Type",
            "Current Count",
            "Diff vs Last Month",
            "Diff vs Last Year"
        ]])

    # -----------------------------
    # ALERTS WITH SEARCH
    # -----------------------------
    st.subheader("🚨 Significant Alerts")

    search_term = st.text_input("🔍 Search Detail Type")

    alerts = results_df[
        (results_df["YoY Result"].str.contains("Significant")) &
        (
            (abs(results_df["Effect YoY"]) > 0.02) |
            (abs(results_df["Effect 3mo"]) > 0.02)
        )
    ]

    if search_term:
        alerts = alerts[
            alerts["Detail Type"].str.contains(search_term, case=False, na=False)
        ]

    st.dataframe(alerts.sort_values("Effect YoY", ascending=False))

    # -----------------------------
    # RISK DASHBOARD (SIGNALS ONLY)
    # -----------------------------
    st.subheader("🎯 Risk Dashboard")

    def color_flag(val):
        if val > 0.05:
            return "🔴 High Increase"
        elif val > 0.02:
            return "🟠 Mild Increase"
        elif val < -0.05:
            return "🟢 High Decrease"
        elif val < -0.02:
            return "🟡 Mild Decrease"
        return "⚪ Neutral"

    results_df["YoY Signal"] = results_df["Effect YoY"].apply(color_flag)
    results_df["3mo Signal"] = results_df["Effect 3mo"].apply(color_flag)

    signal_df = results_df[[
        "Detail Type",
        "YoY Signal",
        "3mo Signal"
    ]]

    st.dataframe(signal_df, use_container_width=True)

    # -----------------------------
    # TREND CHART
    # -----------------------------
    st.subheader("📈 Trend Viewer")

    selected_dt = st.selectbox("Select Detail Type", detail_types)

    trend = pivot[pivot['detail type']==selected_dt].copy()
    trend['date'] = pd.to_datetime(trend[['year','month']].assign(day=1))

    st.line_chart(trend.set_index('date')['count'])
