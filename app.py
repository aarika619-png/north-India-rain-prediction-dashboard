import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
from datetime import datetime

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM AESTHETICS
# ==============================================================================
st.set_page_config(
    page_title="Rain Prediction & Meteorological Dashboard — North India",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, presentation-ready aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        color: #94a3b8;
        font-size: 1.05rem;
    }
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #0284c7;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    .insight-box {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 0.9rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.92rem;
        color: #065f46;
    }
    .insight-box-blue {
        background: #f0f9ff;
        border-left: 4px solid #0284c7;
        padding: 0.9rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.92rem;
        color: #0369a1;
    }
    .alert-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA & MODEL LOADING
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "north_india_rainfall_cleaned.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        alt_path = os.path.join(BASE_DIR, "data", "datasets", "north_india_rainfall_cleaned.parquet")
        df = pd.read_parquet(alt_path)
    else:
        df = pd.read_parquet(DATA_PATH)
    df['date_of_record'] = pd.to_datetime(df['date_of_record'])
    return df

@st.cache_resource
def load_models():
    rf_pipe_path = os.path.join(MODELS_DIR, "rain_classifier_rf_pipeline.joblib")
    reg_pipe_path = os.path.join(MODELS_DIR, "rain_regressor_xgb_pipeline.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    
    classifier = joblib.load(rf_pipe_path) if os.path.exists(rf_pipe_path) else None
    regressor = joblib.load(reg_pipe_path) if os.path.exists(reg_pipe_path) else None
    
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
    return classifier, regressor, metadata

df_master = load_data()
clf_pipeline, reg_pipeline, model_metadata = load_models()

# ==============================================================================
# 3. SIDEBAR FILTERS
# ==============================================================================
st.sidebar.image("https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=400&q=80", use_container_width=True)
st.sidebar.title("🎛️ Dashboard Filters")
st.sidebar.markdown("<small style='color:#64748b;'>Filter regional weather records</small>", unsafe_allow_html=True)

# State filter
all_states = sorted(df_master['state_name'].unique().tolist())
selected_states = st.sidebar.multiselect(
    "Select State(s):",
    options=all_states,
    default=all_states,
    help="Filter observations by North Indian states and Union Territories"
)

# District filter dynamically dependent on selected states
state_filtered_df = df_master[df_master['state_name'].isin(selected_states)] if selected_states else df_master
available_districts = sorted(state_filtered_df['district'].unique().tolist())
selected_districts = st.sidebar.multiselect(
    "Select District(s):",
    options=available_districts,
    default=[],
    help="Optional: Leave blank to include all districts in selected state(s)"
)

# Year Filter
years_list = sorted(df_master['year'].unique().tolist())
year_range = st.sidebar.slider(
    "Time Horizon (Year Range):",
    min_value=int(min(years_list)),
    max_value=int(max(years_list)),
    value=(int(min(years_list)), int(max(years_list))),
    step=1
)

# Season Filter
all_seasons = ['Winter', 'Summer', 'Monsoon', 'Post-monsoon']
selected_seasons = st.sidebar.multiselect(
    "Select Season(s):",
    options=all_seasons,
    default=all_seasons
)

# Month Filter
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
selected_months = st.sidebar.multiselect(
    "Select Month(s):",
    options=month_names,
    default=[]
)

# Apply filters
filtered_df = df_master.copy()

if selected_states:
    filtered_df = filtered_df[filtered_df['state_name'].isin(selected_states)]
if selected_districts:
    filtered_df = filtered_df[filtered_df['district'].isin(selected_districts)]
filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
if selected_seasons:
    filtered_df = filtered_df[filtered_df['season'].isin(selected_seasons)]
if selected_months:
    filtered_df = filtered_df[filtered_df['month'].isin(selected_months)]

# Non-null rain slice for rainfall calculations
rain_data = filtered_df[filtered_df['is_rainfall_recorded']].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Records Selected:** `{len(filtered_df):,}` / `{len(df_master):,}`")
st.sidebar.markdown(f"**Observatories:** `{filtered_df['station_name'].nunique()}` stations")
st.sidebar.markdown(f"**Districts Active:** `{filtered_df['district'].nunique()}` districts")

# Reset button
if st.sidebar.button("🔄 Reset All Filters"):
    st.rerun()

# ==============================================================================
# 4. MAIN HEADER & PRESENTATION BANNER
# ==============================================================================
st.markdown("""
<div class="main-header">
    <h1>🌧️ Dashboard for Rain Prediction in North India</h1>
    <p>Comprehensive Meteorological Analytics, Historical Pattern Exploration & Real-Time Machine Learning Forecasting Suite</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 5. KPI CARDS ROW
# ==============================================================================
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_rain_vol = rain_data['rainfall_mm'].sum()
avg_rain = rain_data['rainfall_mm'].mean() if len(rain_data) > 0 else 0.0
total_obs = len(filtered_df)
max_rain_record = rain_data.loc[rain_data['rainfall_mm'].idxmax()] if len(rain_data) > 0 else None
rainy_days_pct = (rain_data['rain_today'].mean() * 100) if len(rain_data) > 0 else 0.0

with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Rainfall Recorded</div>
        <div class="kpi-value">{total_rain_vol/1000:,.1f} <span style="font-size:1.1rem; color:#64748b;">m</span></div>
        <div class="kpi-subtext">Sum across active stations</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Average Daily Rainfall</div>
        <div class="kpi-value">{avg_rain:.2f} <span style="font-size:1.1rem; color:#64748b;">mm</span></div>
        <div class="kpi-subtext">Mean per station-day</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Observations</div>
        <div class="kpi-value">{total_obs:,}</div>
        <div class="kpi-subtext">{rain_data.shape[0]:,} with rain gauge records</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    max_val_str = f"{max_rain_record['rainfall_mm']:.1f} mm" if max_rain_record is not None else "N/A"
    max_station_str = f"{max_rain_record['station_name']} ({max_rain_record['state']})" if max_rain_record is not None else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Highest Single-Day Rain</div>
        <div class="kpi-value" style="color:#e11d48;">{max_val_str}</div>
        <div class="kpi-subtext" title="{max_station_str}">{max_station_str[:22]}...</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Precipitation Frequency</div>
        <div class="kpi-value">{rainy_days_pct:.1f}%</div>
        <div class="kpi-subtext">Days with rain ≥ 0.1 mm</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# 6. DASHBOARD NAVIGATION TABS
# ==============================================================================
tab_overview, tab_seasons, tab_districts, tab_drivers, tab_predict, tab_model_perf = st.tabs([
    "📈 Overview & Trends",
    "📅 Monthly & Seasonal",
    "🗺️ Regional & Extremes",
    "🌡️ Meteorological Drivers",
    "🔮 Interactive Rain Predictor",
    "⚙️ Model Performance"
])

# ==============================================================================
# TAB 1: OVERVIEW & TRENDS
# ==============================================================================
with tab_overview:
    st.subheader("1. Rainfall Trends Over Time & State Comparison")
    
    col_t1, col_t2 = st.columns([1.2, 1])
    
    with col_t1:
        # Time Series: Annual / Monthly mean rainfall
        ts_data = rain_data.groupby(['year', 'month_num', 'month']).agg(
            mean_rain=('rainfall_mm', 'mean'),
            rainy_days=('rain_today', lambda x: (x == 1).mean() * 100)
        ).reset_index().sort_values(by=['year', 'month_num'])
        ts_data['date_label'] = ts_data['month'].str[:3] + " " + ts_data['year'].astype(str)
        
        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_ts.add_trace(
            go.Bar(
                x=ts_data['date_label'], y=ts_data['mean_rain'],
                name="Mean Daily Rainfall (mm)", marker_color='#0284c7', opacity=0.75
            ),
            secondary_y=False
        )
        fig_ts.add_trace(
            go.Scatter(
                x=ts_data['date_label'], y=ts_data['rainy_days'],
                name="% Rainy Days", mode='lines+markers', line=dict(color='#ea580c', width=2.5),
                marker=dict(size=5)
            ),
            secondary_y=True
        )
        
        fig_ts.update_layout(
            title="<b>Rainfall Evolution & Frequency Over Time</b>",
            xaxis=dict(title="Timeline", tickangle=-45, showgrid=False),
            yaxis=dict(title="Mean Daily Rainfall (mm)", showgrid=True, gridcolor="#f1f5f9"),
            yaxis2=dict(title="% Rainy Days (Rain ≥ 0.1mm)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            height=420,
            margin=dict(l=40, r=40, t=60, b=50)
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box-blue">
            💡 <b>Insight on Temporal Trends:</b> North Indian rainfall demonstrates persistent multi-year cyclicity with severe monsoon peaks in July–August. The peak storm surge was recorded in <b>July 2023</b>, when the interaction of an active monsoon trough with an intense Western Disturbance delivered record inundation across Chandigarh, Punjab, and Himachal Pradesh.
        </div>
        """, unsafe_allow_html=True)
        
    with col_t2:
        # State Comparison
        state_comp = rain_data.groupby('state_name').agg(
            mean_rain=('rainfall_mm', 'mean'),
            rainy_pct=('rain_today', lambda x: x.mean() * 100),
            max_rain=('rainfall_mm', 'max')
        ).reset_index().sort_values(by='mean_rain', ascending=True)
        
        fig_state = go.Figure()
        fig_state.add_trace(go.Bar(
            y=state_comp['state_name'], x=state_comp['mean_rain'],
            orientation='h',
            marker=dict(
                color=state_comp['mean_rain'],
                colorscale='Teal',
                showscale=True,
                colorbar=dict(title="mm/day", len=0.7)
            ),
            text=[f"{v:.2f} mm" for v in state_comp['mean_rain']],
            textposition='outside',
            name="Mean Rain (mm)"
        ))
        
        fig_state.update_layout(
            title="<b>Rainfall Intensity by North Indian State</b>",
            xaxis=dict(title="Mean Daily Rainfall (mm)", showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title=""),
            height=420,
            margin=dict(l=40, r=40, t=60, b=50)
        )
        st.plotly_chart(fig_state, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            🏔️ <b>Insight on Spatial Differences:</b> <b>Delhi</b> and <b>Himachal Pradesh</b> lead average rainfall depth (~4.3 mm/day), while <b>Himachal Pradesh</b> is the clear frequency leader with rain on <b>54.8% of all days</b>. Western arid states like <b>Rajasthan</b> record less than 2.4 mm/day with lower rain frequency (28.2%).
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: MONTHLY & SEASONAL PATTERNS
# ==============================================================================
with tab_seasons:
    st.subheader("2. Seasonal Progression & Monthly Harmonics")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        # Monthly Progression
        m_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
        month_agg = rain_data.groupby('month').agg(
            mean_rain=('rainfall_mm', 'mean'),
            rainy_pct=('rain_today', lambda x: x.mean() * 100),
            max_rain=('rainfall_mm', 'max')
        ).reindex(m_order).reset_index()
        
        fig_mon = px.bar(
            month_agg, x='month', y='mean_rain',
            color='rainy_pct',
            color_continuous_scale='Blues',
            labels={'mean_rain': 'Mean Daily Rain (mm)', 'rainy_pct': '% Rainy Days', 'month': 'Month'},
            title="<b>Monthly Rainfall Depth & Frequency Progression</b>",
            text=[f"{v:.1f}" for v in month_agg['mean_rain']]
        )
        fig_mon.update_traces(textposition='outside')
        fig_mon.update_layout(height=400, xaxis=dict(tickangle=-30), coloraxis_colorbar=dict(title="% Rainy"))
        st.plotly_chart(fig_mon, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box-blue">
            📅 <b>Monthly Progression Takeaway:</b> <b>July</b> represents the core wet month (mean 10.1 mm/day, 78.6% rainy days), followed by August (8.5 mm/day). Precipitation drops sharply in autumn to hit the annual minimum in <b>November (0.54 mm/day, 11% rainy days)</b>.
        </div>
        """, unsafe_allow_html=True)
        
    with col_s2:
        # Seasonal Share Pie Chart
        season_agg = rain_data.groupby('season').agg(
            total_rain=('rainfall_mm', 'sum'),
            mean_rain=('rainfall_mm', 'mean'),
            obs_count=('rainfall_mm', 'count')
        ).reset_index()
        
        fig_season = px.pie(
            season_agg, names='season', values='total_rain',
            color='season',
            color_discrete_map={
                'Monsoon': '#0284c7',
                'Summer': '#f59e0b',
                'Winter': '#64748b',
                'Post-monsoon': '#10b981'
            },
            hole=0.45,
            title="<b>Total Regional Rainfall Volume Share by Season</b>"
        )
        fig_season.update_traces(textinfo='percent+label', pull=[0.05, 0, 0, 0])
        fig_season.update_layout(height=400)
        st.plotly_chart(fig_season, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            ☀️ <b>Seasonal Distribution Takeaway:</b> Over <b>72% of annual precipitation</b> in North India is concentrated during the 4-month South-West Monsoon. Secondary winter-spring rains driven by Western Disturbances deliver essential moisture for agricultural rabi crops across Punjab, Haryana, and UP.
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 3: REGIONAL & EXTREMES
# ==============================================================================
with tab_districts:
    st.subheader("3. District-Level Extremes & IMD Severity Categories")
    
    col_d1, col_d2 = st.columns([1.1, 1])
    
    with col_d1:
        # District Rankings (Top 10 Wettest & Top 10 Driest)
        dist_agg = rain_data.groupby(['district', 'state']).agg(
            mean_rain=('rainfall_mm', 'mean'),
            obs_count=('rainfall_mm', 'count')
        ).reset_index()
        dist_agg = dist_agg[dist_agg['obs_count'] >= 100]
        dist_agg['label'] = dist_agg['district'] + " (" + dist_agg['state'] + ")"
        
        top10_w = dist_agg.sort_values(by='mean_rain', ascending=False).head(10)
        
        fig_dist = px.bar(
            top10_w, x='mean_rain', y='label', orientation='h',
            color='mean_rain', color_continuous_scale='Mint',
            title="<b>Top 10 Wettest Districts in North India (mm/day)</b>",
            labels={'mean_rain': 'Mean Rain (mm)', 'label': 'District'}
        )
        fig_dist.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col_d2:
        # IMD Classification Breakdown
        imd_counts = rain_data['rainfall_category_today'].value_counts().reset_index()
        imd_counts.columns = ['Category', 'Days']
        
        cat_order = [
            'No Rain', 'Very Light Rain (<2.5mm)', 'Light Rain (2.5-15.5mm)',
            'Moderate Rain (15.6-64.4mm)', 'Heavy Rain (64.5-115.5mm)',
            'Very Heavy Rain (115.6-204.4mm)', 'Extremely Heavy Rain (>204.4mm)'
        ]
        imd_counts['Order'] = imd_counts['Category'].map({c: i for i, c in enumerate(cat_order)})
        imd_counts = imd_counts.sort_values(by='Order')
        
        fig_imd = px.bar(
            imd_counts, x='Category', y='Days',
            color='Category',
            color_discrete_sequence=['#94a3b8', '#93c5fd', '#3b82f6', '#0284c7', '#f59e0b', '#ea580c', '#b91c1c'],
            title="<b>IMD Rainfall Severity Classification</b>",
            log_y=True
        )
        fig_imd.update_layout(height=400, xaxis=dict(tickangle=-35), showlegend=False)
        st.plotly_chart(fig_imd, use_container_width=True)
        
    st.subheader("🚨 Top Documented Extreme Weather Events Preserved")
    extreme_events = rain_data[rain_data['rainfall_mm'] >= 200.0].sort_values(by='rainfall_mm', ascending=False)[[
        'date_of_record', 'station_name', 'state_name', 'district', 'rainfall_mm', 'avg_temp_c', 'temp_range_c', 'air_pressure_hpa'
    ]].head(8)
    
    st.dataframe(
        extreme_events.rename(columns={
            'date_of_record': 'Date', 'station_name': 'Observatory', 'state_name': 'State',
            'district': 'District', 'rainfall_mm': 'Rainfall (mm)', 'avg_temp_c': 'Avg Temp (°C)',
            'temp_range_c': 'DTR (°C)', 'air_pressure_hpa': 'Pressure (hPa)'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("""
    <div class="insight-box">
        🌊 <b>Extreme Weather Takeaway:</b> All 19 historical severe rainfall events exceeding 200 mm/day (led by the <b>Mukteshwar Kumaon 460.5 mm cloudburst</b> and the <b>July 2023 Chandigarh 302 mm flood</b>) were preserved intact without truncation. These rare high-impact events coincide with collapsed Diurnal Temperature Ranges (DTR < 5°C) and steep barometric drops.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4: METEOROLOGICAL DRIVERS & CORRELATIONS
# ==============================================================================
with tab_drivers:
    st.subheader("4. Meteorological Correlation Matrix & Atmospheric Signatures")
    
    col_corr1, col_corr2 = st.columns([1.2, 1])
    
    with col_corr1:
        corr_cols = [
            'rainfall_mm', 'rain_today', 'temp_range_c', 'avg_temp_c', 
            'min_temp_c', 'max_temp_c', 'air_pressure_hpa', 'pressure_diff_1d_hpa', 
            'pressure_anomaly_hpa', 'wind_speed_kmh', 'rainfall_lag1_mm', 'rainfall_roll7_sum_mm'
        ]
        corr_df = rain_data[corr_cols].corr()
        
        fig_heat = px.imshow(
            corr_df,
            labels=dict(x="Weather Variable", y="Weather Variable", color="Pearson r"),
            x=corr_cols, y=corr_cols,
            color_continuous_scale='RdBu_r', zmin=-0.5, zmax=0.5,
            title="<b>Correlation Heatmap: Meteorological Drivers vs Rainfall</b>",
            text_auto='.2f'
        )
        fig_heat.update_layout(height=480, margin=dict(l=40, r=40, t=60, b=40))
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with col_corr2:
        # Boxplot: DTR signature on Rain vs Dry Days
        rain_data['rain_status'] = np.where(rain_data['rain_today'] == 1, 'Rainy Day (≥0.1mm)', 'Dry Day (0mm)')
        
        fig_dtr = px.box(
            rain_data, x='rain_status', y='temp_range_c',
            color='rain_status',
            color_discrete_map={'Dry Day (0mm)': '#64748b', 'Rainy Day (≥0.1mm)': '#0284c7'},
            title="<b>The Diurnal Temperature Range (DTR) Signature</b>",
            labels={'temp_range_c': 'Diurnal Temp Range (°C = Max - Min)', 'rain_status': ''}
        )
        fig_dtr.update_layout(height=480, showlegend=False)
        st.plotly_chart(fig_dtr, use_container_width=True)
        
    st.markdown("""
    <div class="insight-box-blue">
        🌡️ <b>Physical Mechanism:</b> <b>Diurnal Temperature Range ($DTR$) is the single strongest thermodynamic indicator</b> (Pearson $r = -0.47$). On dry days, intense daytime sun and night radiative cooling create a wide swing (median 13.5°C). When clouds and moisture accumulate, day heating is blocked and night cooling is trapped, causing DTR to collapse to <b>7.1°C</b>.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 5: INTERACTIVE RAIN PREDICTION
# ==============================================================================
with tab_predict:
    st.subheader("5. Real-Time Next-Day Rain Prediction Engine")
    st.markdown("Use the trained machine learning pipeline to forecast **Rain Occurrence Probability (%)** and **Quantitative Precipitation Depth (mm)** for any North Indian station.")
    
    # Preset scenarios for user convenience
    preset = st.selectbox(
        "⚡ Choose a Real-World Scenario or Customize Below:",
        options=[
            "Custom User Inputs",
            "Monsoon Trough Depression in New Delhi (Active Cloud Cover)",
            "Scorching Pre-Monsoon Heatwave in Bikaner, Rajasthan (Dry/Clear)",
            "Winter Disturbance Front in Srinagar, Kashmir (Cold & Low Pressure)",
            "Heavy Orographic Influx in Kangra / Dharamsala (High Altitude Uplift)"
        ]
    )
    
    # Preset logic
    if preset == "Monsoon Trough Depression in New Delhi (Active Cloud Cover)":
        p_state = "DL"
        p_sub = "Indo-Gangetic Plains"
        p_season = "Monsoon"
        p_month = 7
        p_day = 200
        p_avg_t = 28.0
        p_min_t = 25.5
        p_max_t = 30.5
        p_press = 996.0
        p_press_diff = -2.5
        p_press_anom = -12.0
        p_wind = 14.0
        p_rain_today = 1
        p_lag1 = 24.0
        p_roll3 = 45.0
        p_roll7 = 80.0
        p_elev = 214
        p_lat = 28.6
        p_lon = 77.2
    elif preset == "Scorching Pre-Monsoon Heatwave in Bikaner, Rajasthan (Dry/Clear)":
        p_state = "RJ"
        p_sub = "Northwestern Arid/Semi-Arid"
        p_season = "Summer"
        p_month = 5
        p_day = 140
        p_avg_t = 41.5
        p_min_t = 29.0
        p_max_t = 47.0
        p_press = 1004.0
        p_press_diff = 0.5
        p_press_anom = 1.2
        p_wind = 18.0
        p_rain_today = 0
        p_lag1 = 0.0
        p_roll3 = 0.0
        p_roll7 = 0.0
        p_elev = 224
        p_lat = 28.0
        p_lon = 73.3
    elif preset == "Winter Disturbance Front in Srinagar, Kashmir (Cold & Low Pressure)":
        p_state = "JK"
        p_sub = "Western Himalayas (Montane)"
        p_season = "Winter"
        p_month = 1
        p_day = 25
        p_avg_t = 2.0
        p_min_t = -3.5
        p_max_t = 6.5
        p_press = 1014.0
        p_press_diff = -3.2
        p_press_anom = -6.0
        p_wind = 8.0
        p_rain_today = 1
        p_lag1 = 8.5
        p_roll3 = 18.0
        p_roll7 = 25.0
        p_elev = 1587
        p_lat = 34.1
        p_lon = 74.8
    elif preset == "Heavy Orographic Influx in Kangra / Dharamsala (High Altitude Uplift)":
        p_state = "HP"
        p_sub = "Western Himalayas (Montane)"
        p_season = "Monsoon"
        p_month = 8
        p_day = 225
        p_avg_t = 22.0
        p_min_t = 19.5
        p_max_t = 24.5
        p_press = 955.0
        p_press_diff = -1.8
        p_press_anom = -8.0
        p_wind = 11.0
        p_rain_today = 1
        p_lag1 = 35.0
        p_roll3 = 90.0
        p_roll7 = 160.0
        p_elev = 1457
        p_lat = 32.2
        p_lon = 76.3
    else:
        # Default custom
        p_state = "UP"
        p_sub = "Indo-Gangetic Plains"
        p_season = "Monsoon"
        p_month = 7
        p_day = 195
        p_avg_t = 29.5
        p_min_t = 26.0
        p_max_t = 33.0
        p_press = 1000.0
        p_press_diff = -1.0
        p_press_anom = -5.0
        p_wind = 9.0
        p_rain_today = 0
        p_lag1 = 0.0
        p_roll3 = 5.0
        p_roll7 = 12.0
        p_elev = 168
        p_lat = 27.2
        p_lon = 78.0

    st.markdown("---")
    c_in1, c_in2, c_in3 = st.columns(3)
    
    with c_in1:
        st.markdown("**1. Thermodynamic & Temperature**")
        in_avg_t = st.slider("Mean Daily Temperature (°C)", -10.0, 48.0, float(p_avg_t), 0.5)
        in_min_t = st.slider("Minimum Night Temperature (°C)", -18.0, 36.0, float(p_min_t), 0.5)
        in_max_t = st.slider("Maximum Day Temperature (°C)", -5.0, 52.0, float(max(in_min_t, p_max_t)), 0.5)
        in_dtr = round(in_max_t - in_min_t, 1)
        st.caption(f"Calculated Diurnal Range (DTR): **{in_dtr:.1f}°C**")
        in_t_anom = st.slider("Temperature Anomaly (°C from normal)", -8.0, 8.0, 0.0, 0.5)
        
    with c_in2:
        st.markdown("**2. Atmospheric Dynamics & Wind**")
        in_press = st.slider("Surface Air Pressure (hPa)", 880.0, 1035.0, float(p_press), 1.0)
        in_p_diff = st.slider("24-Hour Pressure Change (ΔP hPa)", -8.0, 8.0, float(p_press_diff), 0.2)
        in_p_anom = st.slider("Pressure Anomaly from Baseline (hPa)", -25.0, 15.0, float(p_press_anom), 0.5)
        in_wind = st.slider("Wind Speed (km/h)", 0.0, 65.0, float(p_wind), 1.0)
        
    with c_in3:
        st.markdown("**3. Precipitation History & Location**")
        in_rain_today = st.radio("Did it rain today at this station?", [0, 1], index=int(p_rain_today), format_func=lambda x: "Yes (Rainy)" if x==1 else "No (Dry)")
        in_lag1 = st.number_input("Yesterday's Rainfall (mm)", min_value=0.0, max_value=450.0, value=float(p_lag1), step=1.0)
        in_roll7 = st.number_input("7-Day Cumulative Rain (mm)", min_value=0.0, max_value=800.0, value=float(p_roll7), step=5.0)
        in_roll3 = round(in_roll7 * 0.45, 1)
        in_state = st.selectbox("State / Territory", options=['JK', 'HP', 'PB', 'HR', 'CH', 'DL', 'UP', 'RJ'], index=['JK', 'HP', 'PB', 'HR', 'CH', 'DL', 'UP', 'RJ'].index(p_state))
        in_season = st.selectbox("Current Season", options=['Winter', 'Summer', 'Monsoon', 'Post-monsoon'], index=['Winter', 'Summer', 'Monsoon', 'Post-monsoon'].index(p_season))
        in_sub = st.selectbox("Geographic Sub-Region", options=['Western Himalayas (Montane)', 'Indo-Gangetic Plains', 'Northwestern Arid/Semi-Arid'], index=['Western Himalayas (Montane)', 'Indo-Gangetic Plains', 'Northwestern Arid/Semi-Arid'].index(p_sub))

    # Form feature dictionary
    sin_doy = round(np.sin(2 * np.pi * p_day / 365.25), 4)
    cos_doy = round(np.cos(2 * np.pi * p_day / 365.25), 4)
    sin_mon = round(np.sin(2 * np.pi * p_month / 12.0), 4)
    cos_mon = round(np.cos(2 * np.pi * p_month / 12.0), 4)
    
    input_row = pd.DataFrame([{
        'avg_temp_c': in_avg_t,
        'min_temp_c': in_min_t,
        'max_temp_c': in_max_t,
        'temp_range_c': in_dtr,
        'temp_anomaly_c': in_t_anom,
        'air_pressure_hpa': in_press,
        'pressure_diff_1d_hpa': in_p_diff,
        'pressure_anomaly_hpa': in_p_anom,
        'wind_speed_kmh': in_wind,
        'rainfall_lag1_mm': in_lag1,
        'rainfall_lag2_mm': round(in_lag1 * 0.7, 1),
        'rainfall_roll3_sum_mm': in_roll3,
        'rainfall_roll7_sum_mm': in_roll7,
        'rain_today': float(in_rain_today),
        'imd_rainy_day_today': 1.0 if in_lag1 >= 2.5 else 0.0,
        'sin_day_of_year': sin_doy,
        'cos_day_of_year': cos_doy,
        'sin_month': sin_mon,
        'cos_month': cos_mon,
        'elevation_m': int(p_elev),
        'latitude': float(p_lat),
        'longitude': float(p_lon),
        'season': in_season,
        'state': in_state,
        'sub_region': in_sub
    }])
    
    # Run Prediction
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Compute Next-Day Rain Prediction", type="primary", use_container_width=True):
        if clf_pipeline is not None and reg_pipeline is not None:
            prob_rain = clf_pipeline.predict_proba(input_row)[0, 1]
            pred_rain_binary = clf_pipeline.predict(input_row)[0]
            pred_depth = max(0.0, reg_pipeline.predict(input_row)[0])
            
            # Display Results in Gauge / Metric Cards
            out_col1, out_col2, out_col3 = st.columns(3)
            
            with out_col1:
                # Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_rain * 100,
                    number={'suffix': "%", 'font': {'size': 36, 'color': '#0f172a'}},
                    title={'text': "<b>Rain Probability (Tomorrow)</b>", 'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#0284c7"},
                        'steps': [
                            {'range': [0, 35], 'color': "#f1f5f9"},
                            {'range': [35, 65], 'color': "#fed7aa"},
                            {'range': [65, 100], 'color': "#bae6fd"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50.0
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with out_col2:
                # Severity and quantitative depth
                st.markdown("""
                <div class="kpi-card" style="height: 260px; display: flex; flex-direction: column; justify-content: center; text-align: center;">
                    <div class="kpi-title">Expected Rainfall Depth</div>
                    <div class="kpi-value" style="font-size: 2.5rem; color:#0284c7;">
                        {:.1f} <span style="font-size:1.4rem;">mm</span>
                    </div>
                    <div class="kpi-subtext" style="font-size:0.95rem; margin-top:0.8rem;">
                        Status: <b>{}</b>
                    </div>
                </div>
                """.format(
                    pred_depth,
                    "Rain Expected Tomorrow 🌧️" if pred_rain_binary == 1 else "Dry Conditions Likely ☀️"
                ), unsafe_allow_html=True)
                
            with out_col3:
                # IMD Alert Tier
                if pred_depth == 0.0 or prob_rain < 0.35:
                    alert_tier = "GREEN (No Warning / Dry)"
                    badge_color = "#dcfce7"
                    text_color = "#166534"
                    imd_desc = "Standard dry weather or trace precipitation. No disruptions expected."
                elif pred_depth < 15.6:
                    alert_tier = "YELLOW (Watch / Light-Moderate)"
                    badge_color = "#fef9c3"
                    text_color = "#854d0e"
                    imd_desc = "Light to moderate scattered rain. Normal civic operations."
                elif pred_depth < 64.5:
                    alert_tier = "ORANGE (Alert / Moderate-Heavy)"
                    badge_color = "#ffedd5"
                    text_color = "#9a3412"
                    imd_desc = "Sustained rainfall. Risk of localized road waterlogging and drainage overflows."
                else:
                    alert_tier = "RED (Warning / Heavy-Severe)"
                    badge_color = "#fee2e2"
                    text_color = "#991b1b"
                    imd_desc = "Severe rainfall downpour. Flash flood warning, transport and power disruption risk."
                    
                st.markdown(f"""
                <div class="kpi-card" style="height: 260px; display: flex; flex-direction: column; justify-content: center;">
                    <div class="kpi-title">IMD Alert Tier Recommendation</div>
                    <div style="margin: 0.6rem 0;">
                        <span class="alert-badge" style="background:{badge_color}; color:{text_color}; font-size:1rem;">
                            {alert_tier}
                        </span>
                    </div>
                    <div style="font-size:0.85rem; color:#64748b; margin-top:0.5rem;">
                        {imd_desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Prediction pipeline models are loading. Please check models/ directory.")

# ==============================================================================
# TAB 6: MODEL PERFORMANCE METRICS
# ==============================================================================
with tab_model_perf:
    st.subheader("6. Predictive Model Performance & Validation Audit")
    
    col_mp1, col_mp2 = st.columns([1.1, 1])
    
    with col_mp1:
        st.markdown("#### Champion Model: Random Forest Classifier")
        st.markdown("""
        Evaluated strictly on **33,329 out-of-time test observations from 2024–2025** across North India.
        """)
        
        m_table = pd.DataFrame([
            {"Model Architecture": "Logistic Regression (L2 Baseline)", "Accuracy": "84.47%", "Precision": "74.20%", "Recall": "74.16%", "F1-Score": "0.7418", "ROC-AUC": "0.8754", "Brier Score": "0.1224"},
            {"Model Architecture": "Random Forest (Champion)", "Accuracy": "84.33%", "Precision": "72.60%", "Recall": "76.94%", "F1-Score": "0.7471", "ROC-AUC": "0.8865", "Brier Score": "0.1179"},
            {"Model Architecture": "XGBoost Classifier", "Accuracy": "83.78%", "Precision": "71.11%", "Recall": "77.59%", "F1-Score": "0.7421", "ROC-AUC": "0.8848", "Brier Score": "0.1210"}
        ])
        st.table(m_table)
        
        st.markdown("""
        <div class="insight-box">
            🎯 <b>Evaluation Verdict:</b> <b>Random Forest Classifier</b> achieved the top ROC-AUC (<b>0.8865</b>) and the lowest Brier calibration error (<b>0.1179</b>), catching <b>76.94% of all upcoming rain events</b> ahead of time across unseen 2024–2025 test dates.
        </div>
        """, unsafe_allow_html=True)
        
    with col_mp2:
        st.markdown("#### Confusion Matrix (Test Set: 33,329 Station-Days)")
        cm_data = pd.DataFrame(
            [[20393, 2911], [2312, 7713]],
            columns=['Predicted Dry (0)', 'Predicted Rain (1)'],
            index=['Actual Dry (0)', 'Actual Rain (1)']
        )
        fig_cm = px.imshow(
            cm_data, text_auto=True, color_continuous_scale='Blues',
            title="<b>Random Forest Confusion Matrix (2024-25 Test Data)</b>"
        )
        fig_cm.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### Top Predictive Feature Weights (Gini Importance)")
    
    if model_metadata and 'top_predictive_features' in model_metadata:
        feat_df = pd.DataFrame(list(model_metadata['top_predictive_features'].items()), columns=['Feature', 'Importance'])
        feat_df['Feature Clean'] = feat_df['Feature'].str.replace('_c', '').str.replace('_mm', '').str.replace('_hpa', '').str.replace('_', ' ').str.title()
        feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(12)
        
        fig_imp = px.bar(
            feat_df, x='Importance', y='Feature Clean', orientation='h',
            color='Importance', color_continuous_scale='Blues',
            title="<b>Top 12 Predictive Drivers for Rain Forecasts</b>",
            text=[f"{v*100:.1f}%" for v in feat_df['Importance']]
        )
        fig_imp.update_layout(height=380, yaxis=dict(title=""), xaxis=dict(title="Relative Importance Weight"))
        st.plotly_chart(fig_imp, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">
    Dashboard for Rain Prediction in North India | Built with Streamlit, Plotly & Scikit-Learn | Data source: Verified 10-Year Cleaned Meteorological Observations (2015–2025)
</div>
""", unsafe_allow_html=True)
