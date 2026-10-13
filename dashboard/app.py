import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API = "http://localhost:8000"

st.set_page_config(
    page_title="RetainIQ — Telecom Customer Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

with st.sidebar:
    st.markdown("### 📡 RetainIQ")
    st.caption("Telecom Customer Intelligence")
    st.markdown("---")
    theme_choice = st.radio(
        "Appearance", ["🌿 Emerald", "🌙 Midnight"],
        index=0 if st.session_state.theme == "light" else 1,
        horizontal=True,
    )
    st.session_state.theme = "light" if "Emerald" in theme_choice else "dark"
    st.markdown("---")
    st.markdown("**Model Status**")
    try:
        active = requests.get(f"{API}/datasets/active", timeout=15).json()
        if "version" in active:
            st.success(f"Active: {active['algorithm'].upper()}")
            st.caption(f"v{active['version'][-8:]}")
            st.metric("ROC-AUC", f"{active.get('roc_auc', 0):.3f}")
            st.metric("F1 Score", f"{active.get('f1_score', 0):.3f}")
        else:
            st.warning("No active model")
    except Exception:
        st.error("API offline")
    st.markdown("---")
    st.caption("Built by Shafeer Said Alavi")

dark = st.session_state.theme == "dark"

if dark:
    bg = "#0a1418"
    card_bg = "#122028"
    text = "#e8dfd0"
    subtext = "#8a9aa0"
    border = "#1d2f38"
    accent = "#2a9d8f"
    accent_dark = "#1a6b62"
    gold = "#d4a373"
    arabic_letters_color = "%23d4a373"   # copper
    arabic_letters_opacity = "0.07"
else:
    bg = "#f4f1ea"
    card_bg = "#ffffff"
    text = "#1a2b30"
    subtext = "#6b7a80"
    border = "#e2ded4"
    accent = "#0d5c4d"
    accent_dark = "#083d33"
    gold = "#b8865a"
    arabic_letters_color = "%230d5c4d"   # deep emerald
    arabic_letters_opacity = "0.055"

# ---------- Scattered Arabic letters background ----------
# SVG with Arabic letters at random positions and rotations
arabic_bg_svg = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='800' viewBox='0 0 800 800'>"
    f"<g font-family='serif' fill='none' stroke='{arabic_letters_color}' stroke-width='1.2' "
    f"opacity='{arabic_letters_opacity}' font-size='90'>"
    "<text x='80' y='120' transform='rotate(-15 80 120)'>هـ</text>"
    "<text x='300' y='80' transform='rotate(20 300 80)'>و</text>"
    "<text x='560' y='150' transform='rotate(-8 560 150)'>ن</text>"
    "<text x='120' y='320' transform='rotate(30 120 320)'>م</text>"
    "<text x='400' y='280' transform='rotate(-25 400 280)'>ق</text>"
    "<text x='660' y='350' transform='rotate(12 660 350)'>ع</text>"
    "<text x='220' y='520' transform='rotate(-18 220 520)'>ر</text>"
    "<text x='500' y='490' transform='rotate(25 500 490)'>ب</text>"
    "<text x='80' y='700' transform='rotate(15 80 700)'>س</text>"
    "<text x='380' y='680' transform='rotate(-22 380 680)'>ط</text>"
    "<text x='650' y='650' transform='rotate(8 650 650)'>ك</text>"
    "<text x='180' y='200' transform='rotate(-40 180 200)' font-size='55'>ل</text>"
    "<text x='580' y='240' transform='rotate(35 580 240)' font-size='55'>ي</text>"
    "<text x='340' y='420' transform='rotate(-12 340 420)' font-size='55'>ح</text>"
    "<text x='720' y='520' transform='rotate(28 720 520)' font-size='55'>ذ</text>"
    "<text x='260' y='640' transform='rotate(-5 260 640)' font-size='55'>ش</text>"
    "</g>"
    "</svg>"
)

st.markdown(f"""
<style>
    @keyframes heroShimmer {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes floatSoft {{
        0%   {{ transform: translateY(0px); }}
        50%  {{ transform: translateY(-5px); }}
        100% {{ transform: translateY(0px); }}
    }}
    @keyframes fadePulse {{
        0%   {{ opacity: 0.85; }}
        50%  {{ opacity: 1; }}
        100% {{ opacity: 0.85; }}
    }}
    @keyframes ringRotate {{
        0%   {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}

    .stApp {{
        background-color: {bg};
        background-image:
            linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0) 100%),
            url("{arabic_bg_svg}");
        background-attachment: fixed, fixed;
        background-size: auto, 900px 900px;
        background-repeat: repeat, repeat;
        color: {text};
    }}
    h1, h2, h3, h4, h5, h6, p, span, div, label {{ color: {text}; }}
    section[data-testid="stSidebar"] {{ background: {card_bg}; }}
    section[data-testid="stSidebar"] * {{ color: {text} !important; }}

    /* Hero */
    .hero {{
        position: relative;
        background:
            linear-gradient(120deg,
                {accent_dark} 0%,
                {accent} 25%,
                {gold} 50%,
                {accent} 75%,
                {accent_dark} 100%);
        background-size: 300% 300%;
        animation: heroShimmer 16s ease-in-out infinite;
        color: #ffffff;
        padding: 2.6rem 2.6rem 2.2rem 2.6rem;
        border-radius: 24px;
        box-shadow: 0 20px 50px rgba(8,61,51,0.34),
                    inset 0 1px 0 rgba(255,255,255,0.20);
        margin-bottom: 1.5rem;
        overflow: hidden;
    }}
    .hero::after {{
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 260px; height: 260px;
        border: 2px dashed rgba(255,255,255,0.15);
        border-radius: 50%;
        animation: ringRotate 40s linear infinite;
        pointer-events: none;
    }}
    .hero h1 {{
        color: #ffffff;
        margin: 0;
        font-size: 2.15rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 0.9rem;
        position: relative; z-index: 2;
    }}
    .hero p {{
        color: #f7efe0;
        margin: 0.7rem 0 0 0;
        font-size: 1.05rem;
        max-width: 780px;
        position: relative; z-index: 2;
    }}
    .hero .flag-img {{
        border-radius: 6px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.45);
        background: #fff;
        flex-shrink: 0;
    }}
    .badge {{
        display: inline-block;
        background: rgba(255,255,255,0.16);
        color: #fff5df;
        padding: 0.35rem 0.95rem;
        border-radius: 999px;
        font-size: 0.78rem;
        letter-spacing: 0.3px;
        margin-right: 0.5rem;
        margin-top: 1.1rem;
        border: 1px solid rgba(212,163,115,0.55);
        backdrop-filter: blur(4px);
        position: relative; z-index: 2;
    }}

    /* KPI strip below hero */
    .kpi-strip {{
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1.4rem;
        flex-wrap: wrap;
    }}
    .kpi {{
        flex: 1;
        min-width: 160px;
        background: {card_bg};
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border-left: 4px solid {gold};
        box-shadow: 0 4px 18px rgba(13,92,77,0.10);
        animation: floatSoft 6s ease-in-out infinite;
    }}
    .kpi h3 {{ margin: 0; color: {accent}; font-size: 1.5rem; font-weight: 700; }}
    .kpi p  {{ margin: 0.2rem 0 0 0; color: {subtext}; font-size: 0.82rem; letter-spacing: 0.4px; text-transform: uppercase; }}

    .stat-card {{
        background: {card_bg};
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        border-left: 4px solid {gold};
        box-shadow: 0 4px 22px rgba(13,92,77,0.10);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .stat-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(13,92,77,0.20);
    }}
    .stat-card h3 {{ margin: 0; color: {accent}; font-size: 1.55rem; font-weight: 700; }}
    .stat-card p  {{ margin: 0.25rem 0 0 0; color: {subtext}; font-size: 0.85rem; }}

    .answer-card {{
        background: {card_bg};
        padding: 1.7rem 1.9rem;
        border-radius: 18px;
        border-left: 5px solid {gold};
        box-shadow: 0 8px 32px rgba(13,92,77,0.12);
        color: {text};
        line-height: 1.75;
        font-size: 1.02rem;
    }}
    .answer-card h3 {{ color: {accent}; margin-top: 0; letter-spacing: 0.3px; }}

    .factor-row {{
        background: {card_bg};
        padding: 0.75rem 1.05rem;
        border-radius: 12px;
        margin-bottom: 0.45rem;
        border-left: 3px solid {accent};
        font-size: 0.92rem;
        color: {text};
    }}

    .section-title {{
        font-size: 1.15rem;
        font-weight: 600;
        color: {accent};
        margin: 0.6rem 0 0.9rem 0;
        letter-spacing: 0.3px;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid {gold};
        display: inline-block;
    }}

    .stButton>button {{
        background: linear-gradient(135deg, {accent} 0%, {accent_dark} 100%) !important;
        color: #fff5e3 !important;
        border-radius: 12px !important;
        border: 1px solid {gold} !important;
        padding: 0.6rem 1.6rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, {gold} 0%, {accent} 100%) !important;
        transform: translateY(-2px);
    }}
    .stTabs [data-baseweb="tab"] {{ color: {text}; font-weight: 500; font-size: 1rem; }}
    .stTabs [aria-selected="true"] {{ color: {accent} !important; }}
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown("""
<div class="hero">
    <h1>
        <img class="flag-img" src="https://flagcdn.com/w80/sa.png" width="56" height="38" alt="Saudi Arabia" />
        RetainIQ — Telecom Customer Intelligence
    </h1>
    <p>Churn prediction with SHAP explainability and bilingual retention offers (English / العربية). Built for telecom operators in Saudi Arabia and the Gulf.</p>
    <span class="badge">XGBoost + LightGBM</span>
    <span class="badge">SHAP Explainability</span>
    <span class="badge">Groq LLM Advisor</span>
    <span class="badge">Multi-format Upload</span>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📊  Overview", "🎯  Predict", "👥  Batch Score", "📤  Upload & Version"])

# ============ TAB 1: Overview ============
with tabs[0]:
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    try:
        cnt = requests.get(f"{API}/customers/count", timeout=15).json()
        summ = requests.get(f"{API}/analytics/summary", timeout=15).json()
        active = requests.get(f"{API}/datasets/active", timeout=15).json()

        col1.markdown(f"<div class='kpi'><h3>{cnt.get('count', 0):,}</h3><p>Customers</p></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='kpi'><h3>{cnt.get('churn_rate', 0):.1%}</h3><p>Churn Rate</p></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='kpi'><h3>{active.get('roc_auc', 0):.3f}</h3><p>Model ROC-AUC</p></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='kpi'><h3>{summ.get('total_predictions', 0)}</h3><p>Predictions</p></div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load overview: {e}")

    st.markdown("")
    st.markdown('<div class="section-title">Churn by Segment</div>', unsafe_allow_html=True)
    try:
        dist = requests.get(f"{API}/analytics/risk-distribution", timeout=15).json()
        c1, c2 = st.columns(2)

        with c1:
            if "contract_type" in dist:
                df = pd.DataFrame(list(dist["contract_type"].items()), columns=["Contract", "Churn Rate"])
                fig = px.bar(df, x="Contract", y="Churn Rate", color="Churn Rate",
                             color_continuous_scale=["#0d5c4d", "#b8865a", "#a83a3a"],
                             title="Churn Rate by Contract Type")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font_color=text, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "internet_service" in dist:
                df = pd.DataFrame(list(dist["internet_service"].items()), columns=["Internet", "Churn Rate"])
                fig = px.bar(df, x="Internet", y="Churn Rate", color="Churn Rate",
                             color_continuous_scale=["#0d5c4d", "#b8865a", "#a83a3a"],
                             title="Churn Rate by Internet Service")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font_color=text, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Charts unavailable: {e}")

# ============ TAB 2: Predict ============
with tabs[1]:
    st.markdown('<div class="section-title">Score a Single Customer</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            customer_id = st.text_input("Customer ID", "C100042")
            tenure = st.number_input("Tenure (months)", 0, 120, 5)
            monthly = st.number_input("Monthly charges (SAR)", 0.0, 1000.0, 320.0)
            total = st.number_input("Total charges (SAR)", 0.0, 100000.0, 1600.0)
        with c2:
            contract = st.selectbox("Contract", ["month-to-month", "one_year", "two_year"])
            internet = st.selectbox("Internet", ["fiber", "dsl", "none"])
            payment = st.selectbox("Payment", ["credit_card", "bank_transfer", "mada", "cash"])
            support = st.number_input("Support calls", 0, 20, 4)
        with c3:
            days_login = st.number_input("Days since last login", 0, 365, 70)
            avg_gb = st.number_input("Avg monthly GB", 0.0, 500.0, 180.0)
            late = st.number_input("Late payments", 0, 20, 2)
            online_sec = st.checkbox("Online security", False)
            tech_sup = st.checkbox("Tech support", False)

        submitted = st.form_submit_button("🚀 Predict Churn & Generate Offer")

    if submitted:
        payload = {
            "customer_id": customer_id,
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "contract_type": contract,
            "internet_service": internet,
            "num_support_calls": support,
            "payment_method": payment,
            "has_online_security": online_sec,
            "has_tech_support": tech_sup,
        }
        with st.spinner("Running model + SHAP + LLM…"):
            r = requests.post(f"{API}/predictions/predict", json=payload, timeout=60)
        if r.ok:
            d = r.json()
            prob = d["churn_probability"]
            tier = d["risk_tier"]
            color = {"high": "#a83a3a", "medium": "#b8865a", "low": "#0d5c4d"}[tier]

            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(
                    f"<div class='stat-card' style='border-left-color:{color};'>"
                    f"<h3 style='color:{color};'>{prob:.1%}</h3>"
                    f"<p>Churn probability · <b>{tier.upper()}</b> risk</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown("**Top Risk Factors (SHAP)**")
                for f in d["top_factors"]:
                    arrow = "↑" if f["shap_value"] > 0 else "↓"
                    st.markdown(
                        f"<div class='factor-row'>{arrow} <b>{f['feature']}</b> — {f['direction']}</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("")
            st.markdown('<div class="section-title">Recommended Retention Offer</div>', unsafe_allow_html=True)
            st.markdown(f"<div class='answer-card'><h3>English</h3>{d['retention_offer']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-card' style='margin-top:1rem;direction:rtl;text-align:right;'><h3>العربية</h3>{d['retention_offer_ar']}</div>", unsafe_allow_html=True)
            st.caption(f"Model version: {d['model_version']}")
        else:
            st.error(f"Error: {r.status_code} — {r.text}")

# ============ TAB 3: Batch Score ============
with tabs[2]:
    st.markdown('<div class="section-title">Batch Scoring</div>', unsafe_allow_html=True)
    n = st.slider("Sample size", 5, 50, 10)
    if st.button("Run Batch Scoring"):
        with st.spinner(f"Scoring {n} customers… (takes ~{n} seconds)"):
            r = requests.post(f"{API}/customers/batch-score", params={"sample_size": n}, timeout=300)
        if r.ok:
            results = r.json()
            rows = []
            for res in results:
                if "churn_probability" in res:
                    rows.append({
                        "Customer": res["customer_id"],
                        "Churn %": round(res["churn_probability"] * 100, 1),
                        "Risk": res["risk_tier"].upper(),
                    })
            if rows:
                df = pd.DataFrame(rows).sort_values("Churn %", ascending=False)
                st.dataframe(df, use_container_width=True, height=400)
                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download CSV", csv, "retainiq_scores.csv", "text/csv")
        else:
            st.error(r.text)

# ============ TAB 4: Upload & Version ============
with tabs[3]:
    st.markdown('<div class="section-title">Upload a Dataset</div>', unsafe_allow_html=True)
    st.caption("Accepted formats: CSV, Excel (.xlsx, .xls), JSON, Parquet. Column names are auto-mapped.")

    with st.expander("📋 Show accepted column aliases"):
        try:
            schema = requests.get(f"{API}/datasets/schema", timeout=15).json()
            st.json(schema.get("accepted_aliases", {}))
        except Exception:
            st.warning("Could not load schema.")

    uploaded = st.file_uploader("Upload file", type=["csv", "xlsx", "xls", "json", "parquet"])
    auto = st.checkbox("Auto-retrain after upload", value=True)

    if uploaded and st.button("⬆️ Upload & Process"):
        with st.spinner("Ingesting, mapping columns, and (re)training…"):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            r = requests.post(
                f"{API}/datasets/upload",
                files=files,
                params={"auto_retrain": auto},
                timeout=600,
            )
        if r.ok:
            d = r.json()
            if d.get("status") == "duplicate":
                st.warning(f"Duplicate dataset — hash {d['data_hash']}")
            else:
                st.success(f"Uploaded: {d['rows']:,} rows · churn {d['churn_rate']:.1%}")
                if d.get("column_mapping"):
                    st.markdown("**Auto-mapped columns**")
                    st.json(d["column_mapping"])
                if "retrained" in d:
                    rt = d["retrained"]
                    st.markdown("**New model trained**")
                    m = rt["metrics"]
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='kpi'><h3>{m['roc_auc']:.3f}</h3><p>ROC-AUC</p></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='kpi'><h3>{m['f1']:.3f}</h3><p>F1</p></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='kpi'><h3>{rt['algorithm'].upper()}</h3><p>Algorithm</p></div>", unsafe_allow_html=True)
                if "retrain_error" in d:
                    st.error(f"Retrain failed: {d['retrain_error']}")
        else:
            st.error(r.text)

    st.markdown("")
    st.markdown('<div class="section-title">Version History</div>', unsafe_allow_html=True)
    if st.button("🔄 Refresh Versions"):
        r = requests.get(f"{API}/datasets/versions", timeout=15)
        if r.ok:
            v = r.json()
            st.markdown("**Data versions**")
            if v["data_versions"]:
                st.dataframe(pd.DataFrame(v["data_versions"]), use_container_width=True)
            else:
                st.info("No data versions yet.")
            st.markdown("**Model versions**")
            if v["model_versions"]:
                st.dataframe(pd.DataFrame(v["model_versions"]), use_container_width=True)
            else:
                st.info("No model versions yet.")