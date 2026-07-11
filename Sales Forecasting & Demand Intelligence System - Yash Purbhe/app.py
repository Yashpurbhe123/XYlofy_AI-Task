# ============================================================
# app.py — Sales Forecasting & Demand Intelligence Dashboard
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Sales Forecasting & Demand Intelligence Hub", layout="wide", page_icon="🔮")

# Inject Custom CSS for dark-slate theme and premium widgets
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global overrides */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Sidebar text/inputs */
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom cards */
    .custom-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .custom-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Header text */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Radio buttons / selectors */
    div[data-testid="stWidgetLabel"] p {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #f8fafc !important;
    }
    
    /* Custom expanders */
    .st-emotion-cache-1h9z78s {
        background-color: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
    }
    
    /* Make tables fit theme */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        border: 1px solid #1e293b;
        background-color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# Custom Metric Card generator
def create_html_card(title, value, icon, gradient):
    return f"""
    <div class="custom-card" style="background: {gradient}; color: white; border: none; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.85;">{title}</div>
            <div style="font-size: 1.6rem; opacity: 0.95;">{icon}</div>
        </div>
        <div style="font-size: 1.8rem; font-weight: 700; margin-top: 12px; letter-spacing: -0.02em; line-height: 1.1;">{value}</div>
    </div>
    """

# Plotly styling helper
def apply_plotly_theme(fig, title_text, xaxis_title="", yaxis_title=""):
    fig.update_layout(
        title={
            'text': title_text,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'family': 'Outfit, sans-serif', 'color': '#f8fafc'}
        },
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(family="Outfit, sans-serif", size=11, color="#94a3b8"),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(color="#cbd5e1")),
            gridcolor="#1e293b",
            linecolor="#334155",
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(color="#cbd5e1")),
            gridcolor="#1e293b",
            linecolor="#334155",
            tickfont=dict(color="#94a3b8")
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#f8fafc")
        ),
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig

# Load and process data
US_STATE_TO_ABBR = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC'
}

@st.cache_data
def load_data():
    file_path = "Train.csv" if os.path.exists("Train.csv") else "train.csv"
    df = pd.read_csv(file_path, encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["MonthName"] = df["Order Date"].dt.strftime('%B')
    df["Week"] = df["Order Date"].dt.isocalendar().week
    df["StateCode"] = df["State"].map(US_STATE_TO_ABBR)
    return df

df = load_data()

# Robust resampling helpers
def monthly_series(sub_df):
    if len(sub_df) == 0:
        return pd.Series(dtype=float)
    s = sub_df.groupby("Order Date")["Sales"].sum().reset_index()
    s = s.set_index("Order Date")
    return s["Sales"].resample("MS").sum().fillna(0)

def weekly_series(sub_df):
    if len(sub_df) == 0:
        return pd.Series(dtype=float)
    s = sub_df.groupby("Order Date")["Sales"].sum().reset_index()
    s = s.set_index("Order Date")
    return s["Sales"].resample("W").sum().fillna(0)

# Caching prediction models
@st.cache_data
def get_forecast(dim, selected_val, model_type, horizon):
    local_df = load_data()
    segment_df = local_df[local_df[dim] == selected_val]
    series = monthly_series(segment_df)
    
    if len(series) < 12:
        return None, None
        
    if "SARIMA" in model_type:
        try:
            if len(series) >= 24:
                model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            else:
                model = SARIMAX(series, order=(1,1,1), seasonal_order=(0,0,0,0),
                                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            fc = model.get_forecast(steps=horizon)
            pred = fc.predicted_mean
            ci = fc.conf_int()
        except Exception as e:
            mean_val = series.mean()
            pred = pd.Series([mean_val] * horizon)
            ci = pd.DataFrame({
                'lower Sales': [mean_val * 0.8] * horizon,
                'upper Sales': [mean_val * 1.2] * horizon
            })
    else: # Holt-Winters
        try:
            if len(series) >= 24:
                model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12).fit()
            else:
                model = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
            pred = model.forecast(horizon)
        except Exception as e:
            mean_val = series.mean()
            pred = pd.Series([mean_val] * horizon)
            
        std_val = series.std() if len(series) > 1 else series.mean() * 0.1
        ci = pd.DataFrame({
            'lower Sales': (pred - 1.96 * std_val).clip(lower=0),
            'upper Sales': pred + 1.96 * std_val
        })
        
    idx = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    pred.index = idx
    ci.index = idx
    return pred, ci

@st.cache_data
def get_holdout_evaluation(dim, selected_val, model_type, steps):
    local_df = load_data()
    segment_df = local_df[local_df[dim] == selected_val]
    series = monthly_series(segment_df)
    
    if len(series) <= steps + 6:
        return None, None, None
    train = series[:-steps]
    test = series[-steps:]
    
    try:
        if "SARIMA" in model_type:
            if len(train) >= 24:
                model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            else:
                model = SARIMAX(train, order=(1,1,1), seasonal_order=(0,0,0,0),
                                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            pred = model.get_forecast(steps=steps).predicted_mean
        else: # Holt-Winters
            if len(train) >= 24:
                model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
            else:
                model = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
            pred = model.forecast(steps)
            
        mae = mean_absolute_error(test, pred)
        rmse = np.sqrt(mean_squared_error(test, pred))
        non_zero_test = test.copy()
        non_zero_test[non_zero_test == 0] = 1e-5
        mape = np.mean(np.abs((test - pred) / non_zero_test)) * 100
        return mae, rmse, mape
    except Exception as e:
        return None, None, None

# Header Banner
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    padding: 24px;
    border-radius: 16px;
    border: 1px solid rgba(99, 102, 241, 0.15);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    margin-bottom: 25px;
    text-align: center;
">
    <h1 style="margin: 0; color: #f8fafc; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;">
        🔮 DEMAND INTELLIGENCE & FORECAST HUB
    </h1>
    <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 1.05rem; font-weight: 400;">
        Interactive Sales Analytics, Predictive Forecasting Engines, and Advanced Product Segmentation
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("""
<div style="text-align: center; padding-top: 10px; padding-bottom: 10px;">
    <span style="font-size: 2.8rem;">📊</span>
    <h3 style="margin: 5px 0 0 0; color: #f8fafc; font-weight: 700; font-size: 1.3rem;">NAVIGATION PANEL</h3>
</div>
""", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"])

st.sidebar.markdown("<hr style='border-color: #1e293b;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: #f8fafc; font-weight: 600; font-size: 1.1rem; margin-bottom: 10px;'>⚙️ GLOBAL FILTERS</h3>", unsafe_allow_html=True)

# Sidebar global filters
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
selected_years = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))

segments = st.sidebar.multiselect("Segments", sorted(df["Segment"].unique()), default=list(df["Segment"].unique()))
regions = st.sidebar.multiselect("Regions", sorted(df["Region"].unique()), default=list(df["Region"].unique()))
categories = st.sidebar.multiselect("Categories", sorted(df["Category"].unique()), default=list(df["Category"].unique()))

# Apply filters
filtered_df = df[
    (df["Year"] >= selected_years[0]) & (df["Year"] <= selected_years[1]) &
    (df["Segment"].isin(segments)) &
    (df["Region"].isin(regions)) &
    (df["Category"].isin(categories))
]

# Check if data is empty after filters
if len(filtered_df) == 0:
    st.warning("⚠️ No data available with the current combination of filters. Please expand your selection in the sidebar.")
else:
    # ==========================================
    # PAGE 1: SALES OVERVIEW
    # ==========================================
    if page == "Sales Overview":
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(create_html_card("Total Sales", f"${filtered_df['Sales'].sum():,.0f}", "💰", "linear-gradient(135deg, #4f46e5, #3b82f6)"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_html_card("Total Orders", f"{filtered_df['Order ID'].nunique():,}", "📦", "linear-gradient(135deg, #0d9488, #0f766e)"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_html_card("Avg Order Value", f"${filtered_df['Sales'].mean():,.2f}", "📈", "linear-gradient(135deg, #7c3aed, #db2777)"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_html_card("Active Customers", f"{filtered_df['Customer ID'].nunique():,}", "👥", "linear-gradient(135deg, #ea580c, #c2410c)"), unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)
        
        # 2x2 Layout for visuals
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        # 1. Sales Trend
        with row1_col1:
            st.markdown("<h3 style='font-size: 1.25rem; margin-bottom: 10px;'>📈 Monthly Sales Trend</h3>", unsafe_allow_html=True)
            monthly = monthly_series(filtered_df).reset_index()
            monthly.columns = ["Date", "Sales"]
            fig_trend = px.area(monthly, x="Date", y="Sales", color_discrete_sequence=["#6366f1"])
            fig_trend.update_traces(line=dict(width=3, shape="spline"), fillcolor="rgba(99, 102, 241, 0.15)", hovertemplate="Date: %{x|%b %Y}<br>Sales: $%{y:,.2f}")
            apply_plotly_theme(fig_trend, "", "", "Sales ($)")
            st.plotly_chart(fig_trend, use_container_width=True)
            
        # 2. US Map
        with row1_col2:
            st.markdown("<h3 style='font-size: 1.25rem; margin-bottom: 10px;'>🗺️ State-wise Sales Demand</h3>", unsafe_allow_html=True)
            state_sales = filtered_df.groupby(["State", "StateCode"])["Sales"].sum().reset_index()
            fig_map = px.choropleth(
                state_sales,
                locations="StateCode",
                locationmode="USA-states",
                color="Sales",
                color_continuous_scale="viridis",
                scope="usa",
                hover_name="State"
            )
            fig_map.update_layout(
                geo=dict(
                    bgcolor="rgba(0,0,0,0)",
                    lakecolor="#0f172a",
                    landcolor="#1e293b",
                    subunitcolor="#475569",
                    showlakes=True,
                    showland=True
                ),
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_colorbar=dict(title=dict(text="Sales ($)", font=dict(color="#cbd5e1")), tickprefix="$")
            )
            st.plotly_chart(fig_map, use_container_width=True)

        # 3. Category Breakdown (Sunburst)
        with row2_col1:
            st.markdown("<h3 style='font-size: 1.25rem; margin-bottom: 10px;'>🌳 Category Hierarchy Distribution</h3>", unsafe_allow_html=True)
            cat_subcat = filtered_df.groupby(["Category", "Sub-Category"])["Sales"].sum().reset_index()
            fig_sunburst = px.sunburst(
                cat_subcat,
                path=["Category", "Sub-Category"],
                values="Sales",
                color="Sales",
                color_continuous_scale="RdPu"
            )
            fig_sunburst.update_layout(
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_colorbar=dict(title="Sales ($)", tickprefix="$")
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)
            
        # 4. Top Products
        with row2_col2:
            st.markdown("<h3 style='font-size: 1.25rem; margin-bottom: 10px;'>🏆 Top 10 Best Sellers</h3>", unsafe_allow_html=True)
            top_prod = filtered_df.groupby("Product Name")["Sales"].sum().reset_index().sort_values("Sales", ascending=False).head(10)
            fig_prod = px.bar(
                top_prod,
                x="Sales",
                y="Product Name",
                orientation="h",
                color="Sales",
                color_continuous_scale="purples"
            )
            fig_prod.update_layout(yaxis=dict(autorange="reversed"))
            apply_plotly_theme(fig_prod, "", "Sales ($)", "")
            # Truncate long product names on y axis for neat look
            fig_prod.update_yaxes(tickmode="array", tickvals=top_prod["Product Name"], ticktext=[p[:28]+"..." if len(p) > 30 else p for p in top_prod["Product Name"]])
            st.plotly_chart(fig_prod, use_container_width=True)

    # ==========================================
    # PAGE 2: FORECAST EXPLORER
    # ==========================================
    elif page == "Forecast Explorer":
        st.markdown("<h2 style='font-size: 1.6rem; margin-bottom: 15px;'>🔮 Advanced Demand Forecasting</h2>", unsafe_allow_html=True)
        
        # Local Segment selector controls
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dim = st.selectbox("Forecast Dimension", ["Category", "Region", "Segment"])
        with c2:
            selected_val = st.selectbox(f"Select {dim} Value", sorted(df[dim].unique()))
        with c3:
            model_type = st.selectbox("Forecasting Algorithm", ["SARIMA (Seasonal ARIMA)", "Holt-Winters (Exponential Smoothing)"])
        with c4:
            horizon = st.slider("Forecast Horizon (Months)", 1, 12, 6)
            
        # Fetch series for the selected segment
        segment_df = df[df[dim] == selected_val]
        series = monthly_series(segment_df)
        
        if len(series) < 12:
            st.warning("⚠️ The selected segment does not have enough transaction history (minimum 12 months required) to run predictive models.")
        else:
            # Model Selection execution
            with st.spinner("Training predictive models and generating forecasts..."):
                pred, ci = get_forecast(dim, selected_val, model_type, horizon)
                mae, rmse, mape = get_holdout_evaluation(dim, selected_val, model_type, steps=min(3, horizon))
            
            # Growth calculation
            last_actual_avg = series[-3:].mean()
            fc_avg = pred.mean()
            growth_pct = ((fc_avg - last_actual_avg) / last_actual_avg) * 100
            
            # Sub-metrics Row
            met1, met2, met3, met4 = st.columns(4)
            with met1:
                st.markdown(create_html_card("Forecast MAPE", f"{mape:.1f}%" if mape is not None else "N/A", "🎯", "linear-gradient(135deg, #1e293b, #334155)"), unsafe_allow_html=True)
            with met2:
                st.markdown(create_html_card("Model MAE", f"${mae:,.0f}" if mae is not None else "N/A", "🔀", "linear-gradient(135deg, #1e293b, #334155)"), unsafe_allow_html=True)
            with met3:
                growth_color = "linear-gradient(135deg, #064e3b, #065f46)" if growth_pct >= 0 else "linear-gradient(135deg, #7f1d1d, #991b1b)"
                st.markdown(create_html_card("Projected Growth %", f"{growth_pct:+.1f}%", "📊", growth_color), unsafe_allow_html=True)
            with met4:
                st.markdown(create_html_card("Forecast Horizon", f"{horizon} Months", "📅", "linear-gradient(135deg, #1e293b, #334155)"), unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Main Forecast Plot
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=series.index, y=series.values, name="Historical Actuals", line=dict(color="#6366f1", width=3)))
            fig_fc.add_trace(go.Scatter(x=pred.index, y=pred.values, name="Forecast Projections", line=dict(color="#f59e0b", width=3, dash="dash")))
            
            # Confidence interval shading
            fig_fc.add_trace(go.Scatter(
                x=list(ci.index) + list(ci.index[::-1]),
                y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
                fill="toself",
                fillcolor="rgba(245, 158, 11, 0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="95% Confidence Band"
            ))
            
            apply_plotly_theme(fig_fc, f"{horizon}-Month Sales Projection — {selected_val}", "", "Sales ($)")
            st.plotly_chart(fig_fc, use_container_width=True)
            
            # Forecast Table and summary details
            det1, det2 = st.columns([1, 1])
            with det1:
                st.markdown("<h3 style='font-size: 1.25rem;'>📋 Demand Projection Details</h3>", unsafe_allow_html=True)
                fc_df = pd.DataFrame({
                    "Forecasted Sales": pred.round(2),
                    "Lower Limit (95% CI)": ci.iloc[:, 0].round(2),
                    "Upper Limit (95% CI)": ci.iloc[:, 1].round(2)
                })
                fc_df.index.name = "Target Month"
                st.dataframe(fc_df, use_container_width=True)
                
                # Download Button
                csv = fc_df.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Export Forecast Data as CSV",
                    data=csv,
                    file_name=f"{selected_val}_sales_forecast.csv",
                    mime="text/csv"
                )
                
            with det2:
                st.markdown("<h3 style='font-size: 1.25rem;'>💡 Model Performance Insights</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background-color: rgba(30, 41, 59, 0.4); border: 1px solid #1e293b; padding: 20px; border-radius: 12px;">
                    <p style="margin-top: 0;"><b>Algorithm Selected:</b> {model_type}</p>
                    <p><b>Backtest MAPE:</b> {f'{mape:.2f}%' if mape is not None else 'Insufficient history for holdout validation.'}</p>
                    <p><b>Growth Analysis:</b> The forecast average over the next {horizon} months is projected to be <b>{growth_pct:+.1f}%</b> compared to the average of the last 3 historical months.</p>
                    <p style="margin-bottom: 0; font-size: 0.85rem; color: #64748b;">
                        <i>*MAPE (Mean Absolute Percentage Error) represents model error. A value under 10% indicates high forecasting accuracy, while 10%-20% indicates good reliability.</i>
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # ==========================================
    # PAGE 3: ANOMALY REPORT
    # ==========================================
    elif page == "Anomaly Report":
        st.markdown("<h2 style='font-size: 1.6rem; margin-bottom: 15px;'>🚨 Sales Anomaly Detection & Intelligence</h2>", unsafe_allow_html=True)
        
        # Controls card
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            contamination_rate = st.slider("Anomaly Sensitivity (Contamination Rate)", 0.01, 0.15, 0.05, step=0.01)
        with col_c2:
            agg_level = st.selectbox("Data Timeframe Aggregation", ["Weekly", "Monthly"])
            
        # Resample data based on aggregation level
        if agg_level == "Weekly":
            ts_df = weekly_series(filtered_df).to_frame("Sales")
            date_freq = "Week"
        else:
            ts_df = monthly_series(filtered_df).to_frame("Sales")
            date_freq = "Month"
            
        if len(ts_df) < 10:
            st.warning("⚠️ Insufficient data points in the filtered dataset to run anomaly detection. Please expand your global filters.")
        else:
            # Fit Isolation Forest
            iso = IsolationForest(contamination=contamination_rate, random_state=42)
            ts_df["Anomaly"] = iso.fit_predict(ts_df[["Sales"]])
            ts_df["Anomaly"] = ts_df["Anomaly"].map({1: "Normal", -1: "Anomaly"})
            ts_df["Score"] = iso.decision_function(ts_df[["Sales"]])
            
            anomalies = ts_df[ts_df["Anomaly"] == "Anomaly"]
            
            # Anomaly Metrics
            total_anom = len(anomalies)
            avg_normal = ts_df[ts_df["Anomaly"] == "Normal"]["Sales"].mean()
            avg_anom = anomalies["Sales"].mean()
            mag_diff = ((avg_anom - avg_normal) / avg_normal) * 100
            
            met1, met2, met3 = st.columns(3)
            with met1:
                st.markdown(create_html_card("Anomalous Periods", f"{total_anom} Detected", "🚨", "linear-gradient(135deg, #1e293b, #334155)"), unsafe_allow_html=True)
            with met2:
                st.markdown(create_html_card("Baseline Average Sales", f"${avg_normal:,.2f}", "📈", "linear-gradient(135deg, #1e293b, #334155)"), unsafe_allow_html=True)
            with met3:
                mag_gradient = "linear-gradient(135deg, #7f1d1d, #991b1b)" if mag_diff > 0 else "linear-gradient(135deg, #0f766e, #115e59)"
                st.markdown(create_html_card("Anomaly Impact", f"{mag_diff:+.1f}% vs Baseline", "⚡", mag_gradient), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Anomaly Chart
            fig_anom = go.Figure()
            fig_anom.add_trace(go.Scatter(x=ts_df.index, y=ts_df["Sales"], mode="lines", name="Historical Sales", line=dict(color="#6366f1", width=2)))
            fig_anom.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies["Sales"],
                mode="markers",
                name="Anomalous Peaks/Troughs",
                marker=dict(color="#ef4444", size=10, symbol="circle", line=dict(color="#ffffff", width=1))
            ))
            
            apply_plotly_theme(fig_anom, f"{agg_level} Sales Anomaly Markers", "", "Sales ($)")
            st.plotly_chart(fig_anom, use_container_width=True)
            
            # Root Cause Analysis Section
            st.markdown("<hr style='border-color: #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-size: 1.4rem;'>🕵️ Automated Root Cause Analysis (RCA)</h3>", unsafe_allow_html=True)
            
            if total_anom == 0:
                st.info("No anomalies detected. Try increasing the sensitivity (Contamination Rate) slider.")
            else:
                st.markdown("We have analyzed the transaction details for the detected anomaly periods to identify the specific categories, products, and geographies driving the unexpected volatility.")
                
                # Order anomalies by extremity (most negative decision scores)
                sorted_anom = anomalies.sort_values(by="Score")
                
                # Display top 3 anomalies
                for idx_date, row in sorted_anom.head(3).iterrows():
                    # Set date bounds
                    if agg_level == "Weekly":
                        start_date = idx_date
                        end_date = idx_date + pd.Timedelta(days=6)
                        date_str = f"Week of {idx_date.strftime('%d %b %Y')}"
                    else:
                        start_date = idx_date
                        end_date = idx_date + pd.DateOffset(months=1) - pd.Timedelta(days=1)
                        date_str = f"Month of {idx_date.strftime('%B %Y')}"
                        
                    # Filter raw dataset for the anomaly window
                    raw_anom_df = filtered_df[(filtered_df["Order Date"] >= start_date) & (filtered_df["Order Date"] <= end_date)]
                    
                    if len(raw_anom_df) > 0:
                        # Top Category
                        cat_cont = raw_anom_df.groupby("Category")["Sales"].sum().reset_index()
                        top_cat = cat_cont.sort_values("Sales", ascending=False).iloc[0]["Category"]
                        top_cat_sales = cat_cont.sort_values("Sales", ascending=False).iloc[0]["Sales"]
                        top_cat_pct = (top_cat_sales / raw_anom_df["Sales"].sum()) * 100
                        
                        # Top State
                        state_cont = raw_anom_df.groupby("State")["Sales"].sum().reset_index()
                        top_state = state_cont.sort_values("Sales", ascending=False).iloc[0]["State"]
                        
                        # Top Sub-Category
                        subcat_cont = raw_anom_df.groupby("Sub-Category")["Sales"].sum().reset_index()
                        top_sub = subcat_cont.sort_values("Sales", ascending=False).iloc[0]["Sub-Category"]
                        
                        # Largest Order
                        largest_order = raw_anom_df.sort_values("Sales", ascending=False).iloc[0]
                        large_val = largest_order["Sales"]
                        large_prod = largest_order["Product Name"]
                        large_cust = largest_order["Customer Name"]
                        
                        # Generate narrative
                        with st.expander(f"🔴 Anomaly Root Cause Analysis — {date_str}", expanded=True):
                            col_rca1, col_rca2 = st.columns([2, 1])
                            with col_rca1:
                                st.markdown(f"""
                                <p style="font-size: 1.05rem; line-height: 1.6;">
                                    During the {agg_level.lower()} starting <b>{idx_date.strftime('%d %b %Y')}</b>, sales reached <b>${row['Sales']:,.2f}</b>, deviating significantly from the normal baseline of <b>${avg_normal:,.2f}</b>.
                                </p>
                                <p style="font-size: 1.05rem; line-height: 1.6;">
                                    This deviation was primarily driven by the <b>{top_cat}</b> category (specifically <b>{top_sub}</b>), which generated <b>${top_cat_sales:,.2f}</b>, accounting for <b>{top_cat_pct:.1f}%</b> of total period sales. Geographically, <b>{top_state}</b> was the top-performing territory.
                                </p>
                                <p style="font-size: 1.05rem; line-height: 1.6; margin-bottom: 0;">
                                    <b>Key Transaction Alert:</b> A single large transaction of <b>${large_val:,.2f}</b> for <i>"{large_prod[:60]}..."</i> by customer <b>{large_cust}</b> was recorded during this timeframe.
                                </p>
                                """, unsafe_allow_html=True)
                            with col_rca2:
                                # Mini bullet summary cards
                                st.markdown(f"""
                                <div style="background-color: rgba(239, 68, 68, 0.08); border-left: 4px solid #ef4444; padding: 12px; border-radius: 4px; margin-bottom: 8px;">
                                    <small style="color: #ef4444; font-weight: 600; text-transform: uppercase;">Primary Driver</small>
                                    <div style="font-weight: 700; color: #f8fafc;">{top_sub}</div>
                                </div>
                                <div style="background-color: rgba(99, 102, 241, 0.08); border-left: 4px solid #6366f1; padding: 12px; border-radius: 4px; margin-bottom: 8px;">
                                    <small style="color: #6366f1; font-weight: 600; text-transform: uppercase;">Top Region</small>
                                    <div style="font-weight: 700; color: #f8fafc;">{top_state}</div>
                                </div>
                                <div style="background-color: rgba(245, 158, 11, 0.08); border-left: 4px solid #f59e0b; padding: 12px; border-radius: 4px;">
                                    <small style="color: #f59e0b; font-weight: 600; text-transform: uppercase;">Peak Deal</small>
                                    <div style="font-weight: 700; color: #f8fafc;">${large_val:,.2f}</div>
                                </div>
                                """, unsafe_allow_html=True)

    # ==========================================
    # PAGE 4: PRODUCT DEMAND SEGMENTS
    # ==========================================
    elif page == "Product Demand Segments":
        st.markdown("<h2 style='font-size: 1.6rem; margin-bottom: 15px;'>🧩 Product Demand Segments & Inventory Strategy</h2>", unsafe_allow_html=True)
        st.markdown("This section groups sub-categories based on their demand profiles: total revenue volume, average order values, and standard deviation (volatility). Adjust clusters to profile product types and view tailored inventory recommendations.")
        
        k = st.slider("Number of Demand Clusters (K-Means)", 2, 6, 4)
        
        # Calculate features by Sub-Category
        sub_cat = filtered_df.groupby("Sub-Category")["Sales"].agg(["sum", "mean", "std", "count"]).reset_index()
        sub_cat.columns = ["Sub-Category", "TotalSales", "AvgOrderValue", "Volatility", "OrderFrequency"]
        sub_cat["Volatility"] = sub_cat["Volatility"].fillna(0)
        
        if len(sub_cat) < k:
            st.warning("⚠️ Insufficient number of sub-categories to create the requested number of clusters. Reduce your cluster size or broaden the filters.")
        else:
            # Scale and cluster
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(sub_cat[["TotalSales", "AvgOrderValue", "Volatility", "OrderFrequency"]])
            
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            sub_cat["Cluster"] = km.fit_predict(X_scaled).astype(str)
            
            # Reduce dimensional scale to 2D using PCA
            coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
            sub_cat["PCA1"] = coords[:, 0]
            sub_cat["PCA2"] = coords[:, 1]
            
            # Profile mapping
            cluster_stats = sub_cat.groupby("Cluster")[["TotalSales", "AvgOrderValue", "Volatility", "OrderFrequency"]].mean()
            sorted_clusters = cluster_stats.sort_values(by="TotalSales", ascending=False).index.tolist()
            
            profiles = {}
            for i, c_id in enumerate(sorted_clusters):
                if i == 0:
                    profiles[c_id] = {
                        "name": "🏆 Champions (High Volume & Value)",
                        "desc": "These sub-categories represent your core revenue drivers, with high sales and high order frequency.",
                        "inv_strat": "<b>Safety Stock Priority:</b> Maintain high service levels (98%+). Use continuous automated replenishment. Keep 2-3 weeks of safety stock.",
                        "mkt_strat": "Maintain customer relationship campaigns, cross-sell related products, and offer bundle pricing.",
                        "color": "#ef4444"
                    }
                elif i == len(sorted_clusters) - 1:
                    profiles[c_id] = {
                        "name": "🐢 Slow Movers (Long Tail)",
                        "desc": "These sub-categories have low total revenue, low purchase frequency, and low average transaction values.",
                        "inv_strat": "<b>Catalog Rationalization:</b> Keep minimum shelf inventory. Consider make-to-order, direct vendor shipment, or rationalizing out of the catalog.",
                        "mkt_strat": "Perform clearance sales or bundle with high-performing items to free up capital.",
                        "color": "#10b981"
                    }
                else:
                    c_stats = cluster_stats.loc[c_id]
                    if c_stats["Volatility"] > cluster_stats["Volatility"].median():
                        profiles[c_id] = {
                            "name": "🌪️ Volatile Hits (High Seasonal Demand)",
                            "desc": "These sub-categories exhibit erratic demand, characterized by high volatility and seasonal peaks.",
                            "inv_strat": "<b>Dynamic Buffers:</b> Utilize dynamic safety stocks that scale up ahead of known promotional cycles or holidays, then scale down.",
                            "mkt_strat": "Launch focused campaigns during historically strong months to leverage high purchase willingness.",
                            "color": "#f59e0b"
                        }
                    else:
                        profiles[c_id] = {
                            "name": "💎 High-Value Niche (Premium Items)",
                            "desc": "These sub-categories feature high average order values but relatively low transaction frequency.",
                            "inv_strat": "<b>Just-in-Time:</b> Keep minimal stock on hand. Focus on securing vendor agreements for quick shipping when orders occur.",
                            "mkt_strat": "Target corporate buyers, key accounts, and run targeted high-tier ads.",
                            "color": "#8b5cf6"
                        }
            
            sub_cat["Cluster_Profile"] = sub_cat["Cluster"].map(lambda x: profiles[x]["name"])
            
            # Draw PCA Scatter
            fig_cluster = px.scatter(
                sub_cat,
                x="PCA1",
                y="PCA2",
                color="Cluster_Profile",
                size="TotalSales",
                hover_name="Sub-Category",
                size_max=35,
                color_discrete_map={profiles[cid]["name"]: profiles[cid]["color"] for cid in profiles}
            )
            apply_plotly_theme(fig_cluster, "PCA Space Projection of Product Segments")
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            st.markdown("<hr style='border-color: #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-size: 1.4rem;'>💡 Actionable Segment Profiles</h3>", unsafe_allow_html=True)
            
            # Print strategic recommendation profiles in columns
            cols = st.columns(k)
            for j, c_id in enumerate(sorted_clusters):
                prof = profiles[c_id]
                # Filter sub-categories in this cluster
                members = sub_cat[sub_cat["Cluster"] == c_id]["Sub-Category"].tolist()
                members_str = ", ".join(members)
                
                with cols[j]:
                    st.markdown(f"""
                    <div style="
                        background-color: rgba(30, 41, 59, 0.4);
                        border-top: 5px solid {prof['color']};
                        border-left: 1px solid #1e293b;
                        border-right: 1px solid #1e293b;
                        border-bottom: 1px solid #1e293b;
                        padding: 16px;
                        border-radius: 8px;
                        height: 100%;
                    ">
                        <h4 style="margin: 0; color: #f8fafc; font-size: 1.1rem;">{prof['name']}</h4>
                        <p style="font-size: 0.85rem; color: #94a3b8; margin: 8px 0;">{prof['desc']}</p>
                        <hr style="border-color: #1e293b; margin: 8px 0;">
                        <p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 8px;"><b>Inventory Strategy:</b><br>{prof['inv_strat']}</p>
                        <p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 8px;"><b>Marketing Strategy:</b><br>{prof['mkt_strat']}</p>
                        <hr style="border-color: #1e293b; margin: 8px 0;">
                        <p style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0;"><b>Products:</b><br><span style="color: {prof['color']};">{members_str}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-size: 1.25rem;'>📋 Detailed Segment Metrics</h3>", unsafe_allow_html=True)
            
            # Format and display raw clustering metrics
            display_sub_cat = sub_cat[["Sub-Category", "Cluster_Profile", "TotalSales", "AvgOrderValue", "Volatility", "OrderFrequency"]].copy()
            display_sub_cat.columns = ["Sub-Category", "Demand Segment", "Total Sales ($)", "Avg Order Value ($)", "Volatility (StdDev)", "Order Count"]
            st.dataframe(display_sub_cat.sort_values(by="Demand Segment"), use_container_width=True)