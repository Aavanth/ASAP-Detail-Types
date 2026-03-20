import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Detail Type Anomaly Detection", layout="wide")

st.title("📊 Detail Type Seasonality & Anomaly Detection")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df.columns = df.columns.str.lower().str.strip()

    if 'detail type' not in df.columns or 'date of occurrence' not in df.columns:
        st.error("CSV must contain 'Detail Type' and 'Date of Occurrence'")
    else:
        df['date of occurrence'] = pd.to_datetime(df['date of occurrence'])
        df['year'] = df['date of occurrence'].dt.year
        df['month'] = df['date of occurrence'].dt.month

        pivot = df.groupby(['year','month','detail type']).size().reset_index(name='count')

        # Sidebar controls
        st.sidebar.header("Controls")
        selected_year = st.sidebar.selectbox("Year", sorted(df['year'].unique()))
        selected_month = st.sidebar.selectbox("Month", range(1,13))

        current = pivot[(pivot['year']==selected_year) & (pivot['month']==selected_month)]
        prev_year = pivot[(pivot['year']==selected_year-1) & (pivot['month']==selected_month)]

        df['year_month'] = df['date of occurrence'].dt.to_period('M')
        selected_period = pd.Period(f"{selected_year}-{selected_month:02d}")

        prev_6 = pivot[pivot.apply(
            lambda row: pd.Period(f"{row['year']}-{row['month']:02d}") in 
            pd.period_range(selected_period-6, selected_period-1, freq='M'),
            axis=1
        )]

        results = []
        detail_types = df['detail type'].unique()

        for dt in detail_types:
            a = current[current['detail type']==dt]['count'].sum()
            total_current = current['count'].sum()

            b = prev_year[prev_year['detail type']==dt]['count'].sum()
            total_prev = prev_year['count'].sum()

            if total_current == 0 or total_prev == 0:
                continue

            c = total_current - a
            d = total_prev - b

            current_rate = a / total_current
            prev_rate = b / total_prev

            table_yoy = [[a, b], [c, d]]
            _, p_yoy, _, _ = chi2_contingency(table_yoy)

            # Direction + result
            direction_yoy = "Increase" if current_rate > prev_rate else "Decrease"
            result_yoy = f"Significant {direction_yoy}" if p_yoy < 0.05 else "Not Significant"

            # ---- 6 MONTH ----
            prev6_dt = prev_6[prev_6['detail type']==dt]['count'].sum()
            prev6_total = prev_6['count'].sum()

            if prev6_total == 0:
                continue

            prev6_rate = prev6_dt / prev6_total

            table_6mo = [[a, prev6_dt],
                         [c, prev6_total - prev6_dt]]

            _, p_6mo, _, _ = chi2_contingency(table_6mo)

            direction_6mo = "Increase" if current_rate > prev6_rate else "Decrease"
            result_6mo = f"Significant {direction_6mo}" if p_6mo < 0.05 else "Not Significant"

            results.append({
                "Detail Type": dt,
                "Current Count": a,
                "Prev Year Count": b,
                "Prev 6mo Count": prev6_dt,
                "Current Rate": current_rate,
                "Prev Year Rate": prev_rate,
                "Prev 6mo Rate": prev6_rate,
                "Effect YoY": current_rate - prev_rate,
                "Effect 6mo": current_rate - prev6_rate,
                "YoY Result": result_yoy,
                "6mo Result": result_6mo
            })

        results_df = pd.DataFrame(results)

        # -------------------------
        # 🏆 TOP 5 ANOMALIES
        # -------------------------
        st.subheader("🏆 Top 5 Anomalies")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔺 Largest Increases")
            st.dataframe(results_df.sort_values("Effect YoY", ascending=False).head(5))

        with col2:
            st.markdown("### 🔻 Largest Decreases")
            st.dataframe(results_df.sort_values("Effect YoY", ascending=True).head(5))

        # -------------------------
        # 🚨 ALERTS
        # -------------------------
        st.subheader("🚨 Significant Alerts")

        alerts = results_df[
            (results_df["YoY Result"].str.contains("Significant")) &
            (abs(results_df["Effect YoY"]) > 0.02)
        ]

        st.dataframe(alerts.sort_values("Effect YoY", ascending=False))

        # -------------------------
        # 🎯 RISK DASHBOARD
        # -------------------------
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
            subset=["Effect YoY", "Effect 6mo"]
        )

        st.dataframe(styled_df)

        # -------------------------
        # 📈 TREND CHART
        # -------------------------
        st.subheader("📈 Trend Viewer")

        selected_dt = st.selectbox("Select Detail Type", detail_types)

        trend = pivot[pivot['detail type']==selected_dt]
        trend['date'] = pd.to_datetime(trend[['year','month']].assign(day=1))

        st.line_chart(trend.set_index('date')['count'])
