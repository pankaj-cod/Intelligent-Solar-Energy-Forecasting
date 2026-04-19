import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

from analysis import summarize_forecast, analyze_risk
from pipeline import run_ai_optimization

# Load environment variables from .env

groq_key = os.getenv("GROQ_API_KEY")
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Energy Forecasting · AI Grid Optimization",
    page_icon="☀️",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* dark background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
    color: #e2e8f0;
}

/* ---------- hero banner (Phase 1 – gold) ---------- */
.hero {
    background: linear-gradient(135deg, #f7971e, #ffd200, #f7971e);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(247,151,30,0.35);
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    color: #1a1a2e;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1rem;
    color: #2d2d2d;
    margin-top: 0.5rem;
    margin-bottom: 0;
}

/* ---------- hero banner (Phase 2 – cyan/teal) ---------- */
.hero-p2 {
    background: linear-gradient(135deg, #0ea5e9, #06b6d4, #0ea5e9);
    border-radius: 20px;
    padding: 2rem 2rem 1.6rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(14,165,233,0.35);
}
.hero-p2 h1 {
    font-size: 2rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-p2 p {
    font-size: 0.9rem;
    color: #1e293b;
    margin-top: 0.4rem;
    margin-bottom: 0;
}

/* ---------- phase badge ---------- */
.phase-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 50px;
    margin-bottom: 0.5rem;
}
.phase-badge.p1 {
    background: rgba(26,26,46,0.2);
    color: #1a1a2e;
}
.phase-badge.p2 {
    background: rgba(15,23,42,0.2);
    color: #0f172a;
}

/* ---------- section cards ---------- */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(12px);
}
.card-title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #ffd200;
    margin-bottom: 1rem;
}

/* Phase 2 card variant – cyan accent */
.card-p2 {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(14,165,233,0.2);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(12px);
}
.card-title-p2 {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 1rem;
}

/* ---------- metric chips ---------- */
.chip-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.chip {
    background: rgba(247,151,30,0.15);
    border: 1px solid rgba(247,151,30,0.3);
    border-radius: 50px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    color: #ffd200;
    font-weight: 600;
}

/* ---------- predict button (gold) ---------- */
div[data-testid="stButton"] > button {
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
}

/* ---------- result banner ---------- */
.result-box {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(16,185,129,0.15));
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    animation: fadeSlideUp 0.5s ease forwards;
    margin-top: 1.2rem;
}
.result-value {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #22c55e, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.result-label {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    letter-spacing: 0.5px;
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---------- sliders & inputs ---------- */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #f7971e, #ffd200) !important;
}
label[data-testid="stWidgetLabel"] p {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ---------- selectbox ---------- */
div[data-testid="stSelectbox"] > div {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    background: rgba(255,255,255,0.06) !important;
}

/* hide default streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* ---------- tabs styling ---------- */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 10px 10px 0 0 !important;
}

/* ---------- Phase 2 metric grid ---------- */
.grid-section {
    margin-top: 1rem;
}
.metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}
.metric-box {
    background: rgba(14,165,233,0.08);
    border: 1px solid rgba(14,165,233,0.2);
    border-radius: 14px;
    padding: 1.3rem 1rem;
    text-align: center;
    backdrop-filter: blur(12px);
    animation: fadeSlideUp 0.5s ease forwards;
}
.metric-box-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #38bdf8;
    line-height: 1.2;
}
.metric-box-label {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}
.var-low    { color: #22c55e !important; }
.var-medium { color: #f59e0b !important; }
.var-high   { color: #ef4444 !important; }

/* ---------- risk banner ---------- */
.risk-banner {
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin-top: 1rem;
    animation: fadeSlideUp 0.5s ease forwards;
}
.risk-banner.risk-stable {
    background: rgba(34,197,94,0.10);
    border: 1px solid rgba(34,197,94,0.35);
}
.risk-banner.risk-low {
    background: rgba(245,158,11,0.10);
    border: 1px solid rgba(245,158,11,0.35);
}
.risk-banner.risk-high {
    background: rgba(239,68,68,0.10);
    border: 1px solid rgba(239,68,68,0.35);
}
.risk-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.risk-title.risk-stable { color: #22c55e; }
.risk-title.risk-low    { color: #f59e0b; }
.risk-title.risk-high   { color: #ef4444; }
.risk-details {
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* ---------- RAG guidelines ---------- */
.guideline-item {
    background: rgba(14,165,233,0.05);
    border-left: 3px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.7rem;
    animation: fadeSlideUp 0.5s ease forwards;
}
.guideline-text {
    font-size: 0.84rem;
    color: #cbd5e1;
    line-height: 1.55;
}
.guideline-score {
    display: inline-block;
    background: rgba(14,165,233,0.15);
    border: 1px solid rgba(14,165,233,0.3);
    border-radius: 50px;
    padding: 0.15rem 0.6rem;
    font-size: 0.68rem;
    color: #38bdf8;
    font-weight: 700;
    margin-top: 0.45rem;
}

/* ---------- LLM recommendation ---------- */
.rec-card {
    background: rgba(14,165,233,0.04);
    border: 1px solid rgba(14,165,233,0.15);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.2rem;
    backdrop-filter: blur(12px);
    animation: fadeSlideUp 0.5s ease forwards;
}
.rec-section {
    margin-bottom: 1.1rem;
}
.rec-section:last-child {
    margin-bottom: 0;
}
.rec-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.35rem;
}
.rec-text {
    font-size: 0.84rem;
    color: #cbd5e1;
    line-height: 1.55;
}

/* ---------- pipeline steps ---------- */
.pipeline-steps {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 1rem 0 1.5rem;
}
.step-chip {
    background: rgba(14,165,233,0.08);
    border: 1px solid rgba(14,165,233,0.2);
    border-radius: 50px;
    padding: 0.35rem 0.9rem;
    font-size: 0.72rem;
    color: #38bdf8;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}
.step-chip .step-num {
    background: rgba(14,165,233,0.3);
    border-radius: 50%;
    width: 18px;
    height: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.6rem;
    font-weight: 800;
}
.step-arrow {
    color: rgba(14,165,233,0.3);
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── Groq API key ──────────────────────────────────────────────────────────────
groq_key = os.getenv("GROQ_API_KEY")

# ── Plant selection (shared) ──────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">⚡ Plant Selection</div>', unsafe_allow_html=True)
plant_choice = st.selectbox("Select Plant", ["Plant 1", "Plant 2"], label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

# Load correct model
if plant_choice == "Plant 1":
    model = joblib.load("model_plant1.pkl")
else:
    model = joblib.load("model_plant2.pkl")

# ══════════════════════════════════════════════════════════════════════════════
#  TABBED LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["☀️  Milestone 1 — Forecast", "🔋 Milestone 2 — AI Grid Optimization"])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PHASE 1: SOLAR FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="hero">
        <span class="phase-badge p1">Milestone 1</span>
        <h1>☀️ Solar Energy Forecasting</h1>
        <p>ML-powered AC power prediction for solar plants</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input columns ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card"><div class="card-title">🕐 Time Features</div>', unsafe_allow_html=True)
        hour  = st.slider("Hour of Day",   0, 23,  12)
        day   = st.slider("Day of Month",  1, 31,  15)
        month = st.slider("Month",         1, 12,   6)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">🌡️ Weather Conditions</div>', unsafe_allow_html=True)
        irradiation  = st.number_input("Irradiation (W/m²)",        value=0.5,  format="%.3f")
        ambient_temp = st.number_input("Ambient Temperature (°C)",  value=25.0, format="%.1f")
        module_temp  = st.number_input("Module Temperature (°C)",   value=30.0, format="%.1f")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Historical AC power ────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">📊 Historical AC Power (kW)</div>', unsafe_allow_html=True)
    hcol1, hcol2, hcol3 = st.columns(3)
    with hcol1:
        prev_1  = st.number_input("1 Hour Ago",   value=100.0, format="%.1f")
    with hcol2:
        prev_2  = st.number_input("2 Hours Ago",  value=100.0, format="%.1f")
    with hcol3:
        prev_24 = st.number_input("24 Hours Ago", value=100.0, format="%.1f")
    st.markdown("</div>", unsafe_allow_html=True)

    roll_3 = (prev_1 + prev_2 + prev_24) / 3

    # ── Input summary chips ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="chip-row">
        <span class="chip">🕐 {hour:02d}:00</span>
        <span class="chip">📅 {day:02d}/{month:02d}</span>
        <span class="chip">☀️ Irr: {irradiation:.3f}</span>
        <span class="chip">🌡️ Amb: {ambient_temp:.1f}°C</span>
        <span class="chip">🔆 Mod: {module_temp:.1f}°C</span>
        <span class="chip">📈 Roll Avg: {roll_3:.1f} kW</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Predict ────────────────────────────────────────────────────────────────
    if st.button("⚡ Predict AC Power"):
        input_dict = {
            'AMBIENT_TEMPERATURE': ambient_temp,
            'MODULE_TEMPERATURE':  module_temp,
            'IRRADIATION':         irradiation,
            'hour':                hour,
            'day':                 day,
            'month':               month,
            'ac_power_prev_1':     prev_1,
            'ac_power_prev_24':    prev_24,
            'ac_power_prev_2':     prev_2,
            'ac_power_roll_3':     roll_3,
        }

        input_data = pd.DataFrame([input_dict])[model.feature_names_in_]
        prediction = model.predict(input_data)[0]

        # Persist for Phase 2 pipeline
        st.session_state['prediction']    = prediction
        st.session_state['last_input']    = input_dict
        st.session_state['last_plant']    = plant_choice
        st.session_state['grid_summary']  = None

    # ── Show persisted prediction ──────────────────────────────────────────────
    if 'prediction' in st.session_state:
        pred  = st.session_state['prediction']
        plant = st.session_state['last_plant']

        st.markdown(f"""
        <div class="result-box">
            <div class="result-value">{pred:,.2f} <span style="font-size:1.2rem;-webkit-text-fill-color:#10b981;">kW</span></div>
            <div class="result-label">Predicted AC Power Output · {plant}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PHASE 2: AI GRID OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="hero-p2">
        <span class="phase-badge p2">Milestone 2</span>
        <h1>🔋 AI Grid Optimization</h1>
        <p>RAG-powered analysis · LLM strategy recommendations · Risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline visualization ─────────────────────────────────────────────────
    st.markdown("""
    <div class="pipeline-steps">
        <div class="step-chip"><span class="step-num">1</span> Forecast Summary</div>
        <span class="step-arrow">→</span>
        <div class="step-chip"><span class="step-num">2</span> Risk Analysis</div>
        <span class="step-arrow">→</span>
        <div class="step-chip"><span class="step-num">3</span> RAG Retrieval</div>
        <span class="step-arrow">→</span>
        <div class="step-chip"><span class="step-num">4</span> LLM Strategy</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Check if Phase 1 prediction exists ─────────────────────────────────────
    if 'prediction' not in st.session_state:
        st.markdown("""
        <div class="risk-banner risk-low">
            <div class="risk-title risk-low">☀️ Run Phase 1 First</div>
            <div class="risk-details">Switch to the Phase 1 tab and click "⚡ Predict AC Power" before running the optimization pipeline.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        pred  = st.session_state['prediction']
        plant = st.session_state['last_plant']

        # Show current prediction context
        st.markdown(f"""
        <div class="card-p2">
            <div class="card-title-p2">📍 Current Prediction Context</div>
            <div style="display:flex; gap:1.5rem; flex-wrap:wrap;">
                <div><span style="color:#94a3b8; font-size:0.78rem;">PLANT</span><br>
                     <span style="color:#e2e8f0; font-weight:700;">{plant}</span></div>
                <div><span style="color:#94a3b8; font-size:0.78rem;">PREDICTED OUTPUT</span><br>
                     <span style="color:#38bdf8; font-weight:700;">{pred:,.2f} kW</span></div>
                <div><span style="color:#94a3b8; font-size:0.78rem;">STATUS</span><br>
                     <span style="color:#22c55e; font-weight:700;">✓ Ready</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Run pipeline ───────────────────────────────────────────────────────
        if st.button("🚀 Run AI Optimization Pipeline"):
            base = st.session_state['last_input'].copy()
            hourly_preds = []
            for h in range(24):
                base['hour'] = h
                df = pd.DataFrame([base])[model.feature_names_in_]
                hourly_preds.append(model.predict(df)[0])

            with st.spinner("Running AI optimization pipeline…"):
                try:
                    result = run_ai_optimization(hourly_preds, api_key=groq_key or None)
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    result = None

            if result:
                st.session_state['grid_summary']      = result['summary']
                st.session_state['risk_analysis']      = result['risk']
                st.session_state['rag_guidelines']     = result['guidelines']
                st.session_state['llm_recommendation'] = result['recommendation']

        # ══════════════════════════════════════════════════════════════════════
        #  SECTION 1 — Forecast Summary
        # ══════════════════════════════════════════════════════════════════════
        if st.session_state.get('grid_summary'):
            s   = st.session_state['grid_summary']
            var = s['variability']

            st.markdown("---")
            st.subheader("📊 24-Hour Forecast Summary")

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-box-value">{s['average_generation']:,.2f} <span style="font-size:0.9rem;">kW</span></div>
                    <div class="metric-box-label">Avg Generation</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-value">{s['max_generation']:,.2f} <span style="font-size:0.9rem;">kW</span></div>
                    <div class="metric-box-label">Peak Generation</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-value">{s['min_generation']:,.2f} <span style="font-size:0.9rem;">kW</span></div>
                    <div class="metric-box-label">Min Generation</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-value var-{var}">{var.upper()}</div>
                    <div class="metric-box-label">Variability</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View raw summary data"):
                st.json(s)

        # ══════════════════════════════════════════════════════════════════════
        #  SECTION 2 — Risk Analysis
        # ══════════════════════════════════════════════════════════════════════
        if st.session_state.get('risk_analysis'):
            r = st.session_state['risk_analysis']

            st.markdown("---")
            st.subheader("⚠️ Risk Analysis")

            if 'High' in r['risk_level']:
                rcls = 'risk-high'
                icon = '🔴'
            elif 'Low' in r['risk_level']:
                rcls = 'risk-low'
                icon = '🟡'
            else:
                rcls = 'risk-stable'
                icon = '🟢'

            st.markdown(f"""
            <div class="risk-banner {rcls}">
                <div class="risk-title {rcls}">{icon} {r['risk_level']}</div>
                <div class="risk-details">{r['details']}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View raw risk data"):
                st.json(r)

        # ══════════════════════════════════════════════════════════════════════
        #  SECTION 3 — AI Recommendations
        # ══════════════════════════════════════════════════════════════════════
        if st.session_state.get('rag_guidelines') or st.session_state.get('llm_recommendation'):
            st.markdown("---")
            st.subheader("🤖 AI-Powered Recommendations")

            # Retrieved guidelines (RAG)
            if st.session_state.get('rag_guidelines'):
                guides = st.session_state['rag_guidelines']
                st.markdown("""
                <div class="card-p2">
                    <div class="card-title-p2">📋 RAG-Retrieved Grid Actions</div>
                """, unsafe_allow_html=True)

                for g in guides:
                    pct = int(g['score'] * 100)
                    st.markdown(f"""
                    <div class="guideline-item">
                        <div class="guideline-text">{g['guideline']}</div>
                        <span class="guideline-score">Relevance: {pct}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            # LLM strategy recommendation
            if st.session_state.get('llm_recommendation'):
                rec = st.session_state['llm_recommendation']
                st.markdown(f"""
                <div class="rec-card">
                    <div class="card-title-p2">💡 LLM Strategy Recommendation</div>
                    <div class="rec-section">
                        <div class="rec-label">Risk Interpretation</div>
                        <div class="rec-text">{rec['risk_interpretation']}</div>
                    </div>
                    <div class="rec-section">
                        <div class="rec-label">Strategy</div>
                        <div class="rec-text">{rec['strategy']}</div>
                    </div>
                    <div class="rec-section">
                        <div class="rec-label">Actions</div>
                        <div class="rec-text">{rec['actions']}</div>
                    </div>
                    <div class="rec-section">
                        <div class="rec-label">Justification</div>
                        <div class="rec-text">{rec['justification']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("View raw recommendation JSON"):
                    st.json(rec)