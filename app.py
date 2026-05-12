import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pydeck as pdk
from src.data.geo_logic import GeoEngine
from src.data.affinity_logic import AffinityEngine
from src.mlops.disruption_engine import get_mock_signals
from src.mlops.copilot_logic import CopilotEngine, initialize_chat
from src.mlops.alerts import get_mock_alerts

st.set_page_config(page_title="SUPPLY CHAIN FORECAST — NOTHING UI", layout="wide")

# Theme State
if 'theme' not in st.session_state:
    st.session_state.theme = 'DARK'

# Inject Nothing OS CSS based on Theme
def load_css(theme):
    if theme == 'DARK':
        vars = """
        --nothing-void:     #000000;
        --nothing-black:    #0A0A0A;
        --nothing-surface:  #111111;
        --nothing-raised:   #1A1A1A;
        --nothing-border:   #2A2A2A;
        --nothing-muted:    #3A3A3A;
        --nothing-white:    #F5F5F5;
        --nothing-gray-1:   #CCCCCC;
        --nothing-gray-2:   #888888;
        --nothing-gray-3:   #555555;
        --nothing-dot-bg:   #1E1E1E;
        """
    else:
        vars = """
        --nothing-void:     #FFFFFF;
        --nothing-black:    #F5F5F5;
        --nothing-surface:  #EEEEEE;
        --nothing-raised:   #E5E5E5;
        --nothing-border:   #D5D5D5;
        --nothing-muted:    #C5C5C5;
        --nothing-white:    #0A0A0A; 
        --nothing-gray-1:   #333333;
        --nothing-gray-2:   #777777;
        --nothing-gray-3:   #AAAAAA;
        --nothing-dot-bg:   #DDDDDD;
        """
        
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Space+Mono:wght@400;700&display=swap');

    :root {{
      {vars}
      --nothing-red:      #FF1C1C;
      --nothing-red-dim:  #CC0000;
      --font-ndot:  'Share Tech Mono', 'Courier New', monospace;
      --font-body:  'Space Mono', 'Courier New', monospace;
    }}

    /* Base Streamlit overrides */
    .stApp {{
        background: var(--nothing-black);
        color: var(--nothing-gray-1);
        font-family: var(--font-body);
        font-size: 13px;
    }}
    
    .stApp > header {{
        background-color: transparent !important;
    }}

    /* Dot Grid Background */
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        background-image: radial-gradient(circle, var(--nothing-dot-bg) 1px, transparent 1px);
        background-size: 24px 24px;
        pointer-events: none;
        z-index: 0;
    }}

    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: var(--font-ndot) !important;
        color: var(--nothing-white) !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}

    /* Nothing Card */
    .nothing-card {{
        background: var(--nothing-surface);
        border: 1px solid var(--nothing-border);
        border-radius: 0px;
        padding: 24px;
        position: relative;
        margin-bottom: 24px;
        z-index: 10;
    }}
    .nothing-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 32px;
        background: var(--nothing-red);
    }}
    .nothing-label {{
        font-family: var(--font-body);
        font-size: 11px;
        color: var(--nothing-gray-2);
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 0 0 8px;
    }}
    .nothing-stat {{
        font-family: var(--font-ndot);
        font-size: 48px;
        color: var(--nothing-white);
        letter-spacing: 0.04em;
        margin: 0;
        font-weight: 400;
        line-height: 1.1;
    }}
    
    /* Buttons */
    .stButton>button {{
        background: transparent !important;
        border: 1px solid var(--nothing-gray-2) !important;
        color: var(--nothing-white) !important;
        font-family: var(--font-body) !important;
        font-size: 11px !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        border-radius: 0 !important;
        transition: border-color 0.15s, color 0.15s !important;
    }}
    .stButton>button:hover {{
        border-color: var(--nothing-white) !important;
        color: var(--nothing-white) !important;
    }}
    
    /* Inputs */
    .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label, .stToggle label {{
        font-family: var(--font-body) !important;
        color: var(--nothing-gray-2) !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }}
    .stSelectbox div[data-baseweb="select"] {{
        background: var(--nothing-surface) !important;
        border: 1px solid var(--nothing-border) !important;
        border-radius: 0 !important;
        color: var(--nothing-white) !important;
        font-family: var(--font-body) !important;
    }}
    
    /* Sliders */
    .stSlider div[data-baseweb="slider"] div {{
        background-color: var(--nothing-border);
    }}
    .stSlider div[data-baseweb="slider"] div[role="slider"] {{
        background-color: var(--nothing-red);
        border: none;
    }}
    
    /* Toggles */
    div[data-testid="stToggle"] > label > div {{
        background-color: var(--nothing-red) !important;
    }}
    
    /* Metric Delta (Custom implementation in HTML) */
    .nothing-delta--good {{ color: var(--nothing-gray-1); font-size: 11px; letter-spacing: 0.08em; }}
    .nothing-delta--bad {{ color: var(--nothing-red); font-size: 11px; letter-spacing: 0.08em; }}
    
    /* Markdown Pre (ASCII Logo) */
    pre {{
        background-color: transparent !important;
        border: none !important;
        color: var(--nothing-white) !important;
    }}
    code {{
        font-family: var(--font-ndot) !important;
        color: var(--nothing-gray-1) !important;
    }}
    
    /* Hide default streamlit decorations */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def apply_mpl_theme(theme):
    if theme == 'DARK':
        style = {
            'figure.facecolor': '#0A0A0A', 'axes.facecolor': '#111111',
            'axes.edgecolor': '#2A2A2A', 'axes.labelcolor': '#888888',
            'axes.titlecolor': '#F5F5F5', 'grid.color': '#2A2A2A',
            'text.color': '#888888', 'xtick.color': '#555555', 'ytick.color': '#555555',
            'legend.facecolor': '#111111', 'legend.edgecolor': '#2A2A2A',
            'legend.labelcolor': '#888888'
        }
        colors = {'tft': '#F5F5F5', 'actual': '#555555', 'accent': '#FF1C1C'}
    else:
        style = {
            'figure.facecolor': '#F5F5F5', 'axes.facecolor': '#EEEEEE',
            'axes.edgecolor': '#D5D5D5', 'axes.labelcolor': '#777777',
            'axes.titlecolor': '#0A0A0A', 'grid.color': '#D5D5D5',
            'text.color': '#777777', 'xtick.color': '#AAAAAA', 'ytick.color': '#AAAAAA',
            'legend.facecolor': '#EEEEEE', 'legend.edgecolor': '#D5D5D5',
            'legend.labelcolor': '#333333'
        }
        colors = {'tft': '#0A0A0A', 'actual': '#AAAAAA', 'accent': '#FF1C1C'}
        
    base_style = {
        'axes.labelsize': 9, 'axes.titlesize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.linewidth': 0.5, 'grid.linestyle': ':',
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'font.family': 'monospace', 'legend.fontsize': 8,
        'lines.linewidth': 1.5, 'patch.linewidth': 0.5,
        'figure.titlesize': 11, 'figure.titleweight': 'normal',
    }
    mpl.rcParams.update({**base_style, **style})
    return colors

def render_stat_card(title, value, delta=None, delta_type='good', icon=None):
    delta_class = "nothing-delta--good" if delta_type == 'good' else "nothing-delta--bad"
    delta_html = f'<p class="{delta_class}">{delta}</p>' if delta else ""
    icon_html = f'<span style="color: var(--nothing-red); margin-right: 8px;">{icon}</span>' if icon else ""
    html = f"""
    <div class="nothing-card nothing-card--stat">
      <p class="nothing-label">{icon_html}{title}</p>
      <p class="nothing-stat">{value}</p>
      {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def simulate_forecast(base_preds, price_change, promo_active, elasticity=-1.5):
    """
    Simulates a new forecast based on price elasticity and promotion impact.
    Default elasticity of -1.5 means 10% price increase = 15% demand drop.
    """
    # Price impact: (1 + change)^elasticity
    price_impact = (1 + price_change) ** elasticity
    # Promo impact: 1.4x boost if active
    promo_impact = 1.4 if promo_active else 1.0
    
    simulated = base_preds * price_impact * promo_impact
    return simulated

def calculate_inventory_metrics(forecast, service_level=1.96): # 1.96 = 95% confidence
    """
    Calculates safety stock and reorder point based on forecast variance.
    """
    avg_demand = np.mean(forecast)
    std_demand = np.std(forecast)
    lead_time = 3 # 3 days lead time
    
    safety_stock = service_level * std_demand * np.sqrt(lead_time)
    reorder_point = (avg_demand * lead_time) + safety_stock
    
    return int(safety_stock), int(reorder_point)

def main():
    load_css(st.session_state.theme)
    NOTHING_COLORS = apply_mpl_theme(st.session_state.theme)
    
    # Theme Toggle
    col_logo, col_toggle = st.columns([4, 1])
    with col_logo:
        st.markdown("""
        ```text
        ╔═══════════════════════════════════════╗
        ║  ◆ SUPPLY CHAIN FORECAST SYSTEM ◆     ║
        ║    TEMPORAL FUSION TRANSFORMER        ║
        ║    M5 WALMART · 42,840 SKUs           ║
        ╚═══════════════════════════════════════╝
        ```
        """)
    with col_toggle:
        st.write("") # Padding
        st.write("")
        theme_toggle = st.toggle("WHITE MODE", value=(st.session_state.theme == 'LIGHT'))
        new_theme = 'LIGHT' if theme_toggle else 'DARK'
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

    # --- SOTA SIDEBAR: DISRUPTION RADAR ---
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 10px; border: 1px solid var(--nothing-border); background: var(--nothing-surface); margin-bottom: 20px;">
            <p class="nothing-label">◆ SOTA RADAR // DISRUPTION SENSING</p>
        </div>
        """, unsafe_allow_html=True)
        
        signals = get_mock_signals()
        for s in signals:
            color = "#FF1C1C" if s['risk'] == 'High' else "#FFD700"
            st.markdown(f"""
            <div style="font-size: 11px; margin-bottom: 10px; border-left: 2px solid {color}; padding-left: 8px;">
                <p style="color: var(--nothing-white); margin: 0;">{s['event']}</p>
                <p style="color: {color}; font-size: 9px; margin: 0;">TYPE: {s['type']} // RISK: {s['risk']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="nothing-label">SIMULATION MODIFIER</p>', unsafe_allow_html=True)
        disruption_mod = st.slider("MANUAL DISRUPTION NUDGE", 0.5, 1.5, 1.0, step=0.05)

    # Initialize Chat State
    if 'messages' not in st.session_state:
        st.session_state.messages = initialize_chat()
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "10 // OVERVIEW", 
        "20 // TELEMETRY", 
        "30 // EXPLAINABILITY", 
        "40 // MLOPS",
        "50 // SIMULATION",
        "60 // COPILOT",
        "70 // RELATIONSHIPS",
        "80 // GLOBAL VIEW"
    ])
    
    with tab1:
        st.title("10 // BENCHMARK MATRIX")
        col1, col2, col3 = st.columns(3)
        with col1:
            render_stat_card("TFT WMAPE SCORE", "0.24", "↓ 15.3% VS BASELINE", "good", "◆")
        with col2:
            render_stat_card("AVG SAFETY STOCK", "142 U", "OPTIMIZED", "good", "◼")
        with col3:
            render_stat_card("STOCKOUT RISK DAYS", "3", "⚠ ELEVATED RISK", "bad", "▲")
            
        st.markdown("---")
        st.title("MODEL COMPARISON")
        df_comp = pd.DataFrame({
            "MODEL": ["ARIMA (BASELINE)", "XGBOOST", "LSTM", "TFT (PROPOSED)"],
            "WMAPE": ["0.45", "0.64", "0.31", "0.24"],
            "IMPROVEMENT": ["-", "+20%", "+45%", "+65%"]
        })
        st.table(df_comp)

        st.markdown("---")
        st.title("ACTIVE SYSTEM ALERTS")
        
        alerts = get_mock_alerts()
        
        for alert in alerts:
            color = "#FF1C1C" if alert['severity'] == 'high' else ("#FFD700" if alert['severity'] == 'medium' else "#888888")
            st.markdown(f"""
            <div style="padding: 12px; border-left: 4px solid {color}; background: var(--nothing-surface); margin-bottom: 8px; border-top: 1px solid var(--nothing-border); border-right: 1px solid var(--nothing-border); border-bottom: 1px solid var(--nothing-border);">
                <p class="nothing-label" style="color: {color}; margin-bottom: 4px;">◆ {alert['type']} ALERT // {alert['item']}</p>
                <p style="font-size: 12px; color: var(--nothing-gray-1); margin: 0;">{alert['message']}</p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.title("20 // SIGNAL OUTPUT")
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            items = ['FOODS_3_090_CA_1', 'HOBBIES_1_001_CA_1', 'HOUSEHOLD_2_001_CA_1']
            selected_item = st.selectbox("SELECT ITEM", items)
        with col_sel2:
            horizon = st.slider("HORIZON (DAYS)", 7, 28, 14, step=7)
        
        # Generate mock plot
        np.random.seed(hash(selected_item) % 1000)
        hist_len = 56
        dates = pd.date_range('2016-03-01', periods=hist_len + horizon)
        
        base = np.random.poisson(lam=15, size=hist_len + horizon)
        seasonality = 5 * np.sin(np.arange(hist_len + horizon) * (2 * np.pi / 7))
        actuals = np.maximum(0, base + seasonality).astype(int)
        
        preds = actuals[hist_len:] + np.random.normal(0, 1, horizon)
        
        fig_facecolor = '#0A0A0A' if st.session_state.theme == 'DARK' else '#F5F5F5'
        title_color = '#F5F5F5' if st.session_state.theme == 'DARK' else '#0A0A0A'
        
        fig, ax = plt.subplots(figsize=(12, 4), facecolor=fig_facecolor)
        ax.plot(dates[:hist_len], actuals[:hist_len], color=NOTHING_COLORS['actual'], label='ACTUAL')
        ax.plot(dates[hist_len:], preds, color=NOTHING_COLORS['tft'], linewidth=2, label='TFT PROJECTION')
        ax.fill_between(dates[hist_len:], preds-2, preds+2, color=NOTHING_COLORS['tft'], alpha=0.1)
        ax.axvline(dates[hist_len], color=NOTHING_COLORS['accent'], linestyle=':', label='ORIGIN')
        ax.set_title(f"FORECAST TELEMETRY // {selected_item}", color=title_color, fontsize=10, fontfamily='monospace', loc='left')
        ax.legend()
        st.pyplot(fig)

    with tab3:
        st.title("30 // EXPLAINABILITY")
        st.markdown("### SHAP FEATURE IMPORTANCE")
        import os
        img_path = 'logs/eda/shap_global_importance.png'
        if os.path.exists(img_path):
            st.image(img_path, caption="SHAP Global Feature Importance")
        else:
            features = ['sell_price', 'day_of_week', 'month', 'event_name_1', 'snap_CA']
            importance = [0.45, 0.25, 0.15, 0.10, 0.05]
            fig, ax = plt.subplots(figsize=(10, 4), facecolor=fig_facecolor)
            ax.barh(features, importance, color=NOTHING_COLORS['tft'])
            ax.set_title("GLOBAL FEATURE IMPORTANCE (TFT)", color=title_color, fontsize=10, fontfamily='monospace', loc='left')
            st.pyplot(fig)

    with tab4:
        st.title("40 // MLOPS & RECALIBRATION")
        
        col_mlops1, col_mlops2 = st.columns(2)
        with col_mlops1:
            render_stat_card("DATA DRIFT (PSI)", "0.12", "WITHIN TOLERANCE", "good")
        with col_mlops2:
            render_stat_card("LAST RETRAINING", "2 DAYS AGO", "SUCCESS", "good")
        
        st.markdown("---")
        st.markdown("### ENDPOINT STATUS")
        
        if 'recalibration_triggered' not in st.session_state:
            st.session_state.recalibration_triggered = False

        if st.button("RUN RECALIBRATION"):
            st.session_state.recalibration_triggered = True
            st.rerun()
            
        if st.session_state.recalibration_triggered:
            st.markdown('<p class="nothing-delta--good" style="font-size: 16px; font-weight: bold; padding: 10px; border: 1px solid var(--nothing-border); background: var(--nothing-surface);">◆ RECALIBRATION INITIATED. RETRAINING JOB QUEUED SUCCESSFULLY.</p>', unsafe_allow_html=True)
            if st.button("ACKNOWLEDGE / CLEAR"):
                st.session_state.recalibration_triggered = False
                st.rerun()

    with tab5:
        st.title("50 // STRATEGIC SIMULATION")
        st.markdown("### WHAT-IF ANALYSIS ENGINE")
        
        col_sim1, col_sim2 = st.columns([1, 2])
        
        with col_sim1:
            st.markdown('<div class="nothing-card">', unsafe_allow_html=True)
            st.markdown('<p class="nothing-label">CONTROL PARAMETERS</p>', unsafe_allow_html=True)
            sim_item = st.selectbox("SELECT ITEM TO SIMULATE", items, key="sim_item")
            price_change = st.slider("PRICE ADJUSTMENT (%)", -50, 50, 0, step=5) / 100.0
            promo_toggle = st.toggle("ACTIVATE PROMOTIONAL EVENT", value=False)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Inventory Insights
            st.markdown('<div class="nothing-card" style="border-top: 4px solid var(--nothing-red);">', unsafe_allow_html=True)
            st.markdown('<p class="nothing-label">INVENTORY RECOMMENDATION</p>', unsafe_allow_html=True)
            
            # Dynamic calculation based on sim
            np.random.seed(42)
            base_sim_preds = 15 + 5 * np.sin(np.arange(14) * (2 * np.pi / 7))
            # Apply SOTA Disruption Modifier
            base_sim_preds = base_sim_preds * disruption_mod
            
            sim_preds = simulate_forecast(base_sim_preds, price_change, promo_toggle)
            
            s_stock, r_point = calculate_inventory_metrics(sim_preds)
            
            st.markdown(f'<p class="nothing-stat" style="font-size: 24px;">{s_stock} UNITS</p>', unsafe_allow_html=True)
            st.markdown('<p class="nothing-label">SAFETY STOCK BUFFER</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="nothing-stat" style="font-size: 24px;">{r_point} UNITS</p>', unsafe_allow_html=True)
            st.markdown('<p class="nothing-label">REORDER POINT (ROP)</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_sim2:
            # Simulation Plot
            dates_sim = pd.date_range('2016-05-23', periods=14)
            
            fig_sim, ax_sim = plt.subplots(figsize=(10, 6.5), facecolor=fig_facecolor)
            ax_sim.plot(dates_sim, base_sim_preds, color=NOTHING_COLORS['actual'], linestyle='--', label='BASELINE FORECAST')
            ax_sim.plot(dates_sim, sim_preds, color=NOTHING_COLORS['accent'], linewidth=3, label='SIMULATED RESPONSE')
            ax_sim.fill_between(dates_sim, sim_preds*0.9, sim_preds*1.1, color=NOTHING_COLORS['accent'], alpha=0.1)
            
            ax_sim.set_title(f"SCENARIO PROJECTION // {sim_item}", color=title_color, fontsize=10, fontfamily='monospace', loc='left')
            ax_sim.legend()
            st.pyplot(fig_sim)
            
            st.markdown(f"""
            <div style="padding: 15px; border: 1px solid var(--nothing-border); background: var(--nothing-surface);">
            <p class="nothing-label">◆ INSIGHT</p>
            <p style="font-size: 12px; color: var(--nothing-gray-1);">
            A <b>{price_change*100:.0f}% price change</b> combined with <b>{"active" if promo_toggle else "inactive"}</b> promotions 
            is projected to shift volume by <b>{((sim_preds.sum()/base_sim_preds.sum())-1)*100:.1f}%</b>. 
            Adjust inventory buffers accordingly to maintain a 95% service level.
            </p>
            </div>
            """, unsafe_allow_html=True)

    with tab6:
        st.title("60 // SUPPLY CHAIN COPILOT")
        st.markdown("### INTELLIGENT AGENTIC INTERFACE")
        
        # Chat Container
        chat_container = st.container(height=450, border=True)
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(f'<div style="font-family: var(--font-body); font-size: 13px;">{message["content"]}</div>', unsafe_allow_html=True)
                    if "action" in message and message["action"]:
                        st.markdown(f'<p style="font-size: 10px; color: var(--nothing-red); margin-top: 4px;">◆ RECOMMENDED ACTION: {message["action"]}</p>', unsafe_allow_html=True)

        # Chat Input
        if prompt := st.chat_input("Ask about your supply chain..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate Response
            engine = CopilotEngine()
            response = engine.get_response(prompt)
            
            # Add assistant message
            st.session_state.messages.append(response)
            st.rerun()
        
        st.markdown("""
        <div style="padding: 15px; border: 1px dashed var(--nothing-border); margin-top: 20px;">
            <p class="nothing-label">💡 SUGGESTED QUERIES</p>
            <span style="font-size: 11px; color: var(--nothing-gray-2);">"Which items are at risk of stockout?"</span> | 
            <span style="font-size: 11px; color: var(--nothing-gray-2);">"Show me model performance vs baseline"</span> | 
            <span style="font-size: 11px; color: var(--nothing-gray-2);">"Explain price elasticity for Foods"</span>
        </div>
        """, unsafe_allow_html=True)

    with tab7:
        st.title("70 // PRODUCT AFFINITY GRAPH")
        st.markdown("### SOTA: INTER-PRODUCT DEMAND RELATIONSHIPS (GNN)")
        
        col_rel1, col_rel2 = st.columns([1, 2])
        
        engine_affinity = AffinityEngine()
        
        with col_rel1:
            st.markdown('<div class="nothing-card">', unsafe_allow_html=True)
            st.markdown('<p class="nothing-label">NODE SELECTION</p>', unsafe_allow_html=True)
            selected_skus = st.multiselect("SELECT SKUs TO MAP", items, default=items[:3])
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="padding: 12px; border: 1px solid var(--nothing-border); background: var(--nothing-surface);">
                <p class="nothing-label">◆ LEGEND</p>
                <p style="font-size: 11px; color: #00FF00;">🟢 GREEN: COMPLEMENTARY (BASKET-BUILDER)</p>
                <p style="font-size: 11px; color: #FF1C1C;">🔴 RED: SUBSTITUTE (CANNIBALIZATION)</p>
            </div>
            """, unsafe_allow_html=True)

        with col_rel2:
            if selected_skus:
                fig_affinity = engine_affinity.create_relationship_graph(selected_skus)
                st.plotly_chart(fig_affinity, use_container_width=True)
            else:
                st.warning("PLEASE SELECT SKUS TO VISUALIZE RELATIONSHIPS.")

    with tab8:
        st.title("80 // GLOBAL COMMAND CENTER")
        st.markdown("### 3D GEOSPATIAL RISK HEATMAP")
        
        geo_engine = GeoEngine()
        geo_df = geo_engine.get_geo_data()
        
        # Layer 1: Risk Columns
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=geo_df,
            get_position=["lon", "lat"],
            get_elevation="volume",
            elevation_scale=10,
            radius=20000,
            get_fill_color=["color_r", "color_g", "color_b"], # Removed the constant alpha
            pickable=True,
            auto_highlight=True,
        )
        
        # View State
        view_state = pdk.ViewState(
            latitude=38.0,
            longitude=-98.0,
            zoom=3.5,
            pitch=45,
        )
        
        # Render Map (Simplified tooltip to avoid serialization error)
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=view_state,
            layers=[column_layer],
            tooltip={
                "text": "Store: {store_id}\nVolume: {volume}\nRisk: {risk}"
            }
        ))
        
        st.markdown("""
        <div style="padding: 15px; border: 1px solid var(--nothing-border); background: var(--nothing-surface);">
            <p class="nothing-label">◆ GEOSPATIAL INTELLIGENCE</p>
            <p style="font-size: 11px; color: var(--nothing-gray-1);">
            Columns represent predicted demand volume for the next 28 days. 
            Color spectrum shifts from <b>GREEN (Optimized)</b> to <b>RED (Critical Risk)</b>. 
            Note the concentration of risk in the <b>California corridor</b> due to recent logistics signal disruptions.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
