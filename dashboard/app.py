import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
        "Appearance", ["☀️ Warm", "🌙 Dark"],
        index=0 if st.session_state.theme == "light" else 1,
        horizontal=True,
    )
    st.session_state.theme = "light" if "Warm" in theme_choice else "dark"
    st.markdown("---")
    st.markdown("**Model Status**")
    try:
        active = requests.get(f"{API}/datasets/active", timeout=3).json()
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
    bg = "#14110c"
    card_bg = "#1c1812"
    text = "#ece4d3"
    subtext = "#a89a7c"
    border = "#2e2718"
    accent = "#0a7d47"
    gold = "#c9a44c"
    pattern_opacity = "0.06"
    pattern_stroke = "%23c9a44c"
else:
    bg = "#faf6ee"
    card_bg = "#fffdf8"
    text = "#2b2416"
    subtext = "#7a6a4f"
    border = "#e8dfc9"
    accent = "#0a5c36"
    gold = "#b8860b"
    pattern_opacity = "0.06"
    pattern_stroke = "%230a5c36"

st.markdown(f"""
<style>
    @keyframes heroShimmer {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes floatUp {{
        0%   {{ transform: translateY(0px); opacity: 0.9; }}
        50%  {{ transform: translateY(-6px); opacity: 1; }}
        100% {{ transform: translateY(0px); opacity: 0.9; }}
    }}
    @keyframes glowPulse {{
        0%   {{ box-shadow: 0 0 0 0 rgba(184,134,11,0.4); }}
        70%  {{ box-shadow: 0 0 0 12px rgba(184,134,11,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(184,134,11,0); }}
    }}
    .stApp {{
        background-color: {bg};
        background-image:
            url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'><g fill='none' stroke='{pattern_stroke}' stroke-width='1' opacity='{pattern_opacity}'><path d='M60 10 L75 35 L105 35 L85 55 L95 85 L60 70 L25 85 L35 55 L15 35 L45 35 Z'/><circle cx='60' cy='60' r='48'/><circle cx='60' cy='60' r='30'/></g></svg>");
        background-attachment: fixed;
        color: {text};
    }}
    h1, h2, h3, h4, h5, h6, p, span, div, label {{ color: {text}; }}
    section[data-testid="stSidebar"] {{ background: {card_bg}; }}
    section[data-testid="stSidebar"] * {{ color: {text} !important; }}

    .hero {{
        position: relative;
        background: linear-gradient(120deg, {accent} 0%, #064a2a 30%, {gold} 50%, #064a2a 70%, {accent} 100%);
        background-size: 300% 300%;
        animation: heroShimmer 14s ease-in-out infinite;
        color: #ffffff; padding: 2.2rem 2.4rem; border-radius: 20px;
        box-shadow: 0 14px 40px rgba(10,92,54,0.28);
        margin-bottom: 1.4rem; overflow: hidden;
    }}
    .hero::after {{
        content: ""; position: absolute; right: -50px; top: -50px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(255,255,255,0.15), transparent 70%);
        pointer-events: none;
        animation: floatUp 6s ease-in-out infinite;
    }}
    .hero h1 {{ color: #ffffff; margin: 0; font-size: 2rem; display: flex; align-items: center; gap: 0.9rem; }}
    .hero p  {{ color: #f7f1e3; margin: 0.55rem 0 0 0; }}
    .badge {{
        display: inline-block; background: rgba(255,255,255,0.16); color: #fff8e7;
        padding: 0.32rem 0.9rem; border-radius: 999px; font-size: 0.78rem;
        margin-right: 0.5rem; margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.28);
    }}

    .stat-card {{
        background: {card_bg}; padding: 1.15rem 1.35rem; border-radius: 14px;
        border-left: 4px solid {gold};
        box-shadow: 0 3px 18px rgba(184,134,11,0.10);
        transition: transform 0.25s ease;
    }}
    .stat-card:hover {{ transform: translateY(-3px); }}
    .stat-card h3 {{ margin: 0; color: {accent}; font-size: 1.55rem; font-weight: 700; }}
    .stat-card p  {{ margin: 0.2rem 0 0 0; color: {subtext}; font-size: 0.85rem; }}

    .answer-card {{
        background: {card_bg}; padding: 1.6rem 1.8rem; border-radius: 16px;
        border-left: 5px solid {gold};
        box-shadow: 0 6px 28px rgba(184,134,11,0.12);
        color: {text}; line-height: 1.7;
    }}
    .answer-card h3 {{ color: {accent}; margin-top: 0; }}

    .factor-row {{
        background: {card_bg}; padding: 0.7rem 1rem; border-radius: 10px;
        margin-bottom: 0.4rem; border-left: 3px solid {gold};
        font-size: 0.92rem;
    }}

    .stButton>button {{
        background: linear-gradient(135deg, {accent} 0%, #064a2a 100%) !important;
        color: #ffffff !important; border-radius: 12px !important;
        border: 1px solid {gold} !important;
        padding: 0.55rem 1.55rem !important; font-weight: 600 !important;
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, {gold} 0%, {accent} 100%) !important;
    }}
    .stTabs [data-baseweb="tab"] {{ color: {text}; font-weight: 500; }}
    .stTabs [aria-selected="true"] {{ color: {accent} !important; }}
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown("""
<div class="hero">
    <h1>📡 RetainIQ — Telecom Customer Intelligence</h1>
    <p>Churn prediction with SHAP explainability and bilingual retention offers (English / العربية).</p>
    <span class="badge">XGBoost + LightGBM</span>
    <span class="badge">SHAP Explainability</span>
    <span class="badge">Groq LLM Advisor</span>
    <span class="badge">Multi-format Upload</span>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📊  Overview", "🎯  Predict", "👥  Batch Score", "📤  Upload & Version"])

# ============ TAB 1: Overview ============
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    try:
        cnt = requests.get(f"{API}/customers/count", timeout=5).json()
        summ = requests.get(f"{API}/analytics/summary", timeout=5).json()
        active = requests.get(f"{API}/datasets/active", timeout=5).json()

        col1.markdown(f"<div class='stat-card'><h3>{cnt.get('count', 0):,}</h3><p>Customers</p></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='stat-card'><h3>{cnt.get('churn_rate', 0):.1%}</h3><p>Churn rate</p></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='stat-card'><h3>{active.get('roc_auc', 0):.3f}</h3><p>Model ROC-AUC</p></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='stat-card'><h3>{summ.get('total_predictions', 0)}</h3><p>Predictions made</p></div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load overview: {e}")

    st.markdown("### Churn by Segment")
    try:
        dist = requests.get(f"{API}/analytics/risk-distribution", timeout=5).json()
        c1, c2 = st.columns(2)

        with c1:
            if "contract_type" in dist:
                df = pd.DataFrame(list(dist["contract_type"].items()), columns=["Contract", "Churn Rate"])
                fig = px.bar(df, x="Contract", y="Churn Rate", color="Churn Rate",
                             color_continuous_scale=["#0a5c36", "#b8860b", "#a83232"],
                             title="Churn Rate by Contract Type")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font_color=text, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "internet_service" in dist:
                df = pd.DataFrame(list(dist["internet_service"].items()), columns=["Internet", "Churn Rate"])
                fig = px.bar(df, x="Internet", y="Churn Rate", color="Churn Rate",
                             color_continuous_scale=["#0a5c36", "#b8860b", "#a83232"],
                             title="Churn Rate by Internet Service")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font_color=text, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Charts unavailable: {e}")

# ============ TAB 2: Predict ============
with tabs[1]:
    st.markdown("### Score a single customer")

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

        submitted = st.form_submit_button("🚀 Predict churn & generate offer")

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
            color = {"high": "#a83232", "medium": "#b8860b", "low": "#0a5c36"}[tier]

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
                st.markdown("**Top risk factors (SHAP)**")
                for f in d["top_factors"]:
                    arrow = "↑" if f["shap_value"] > 0 else "↓"
                    st.markdown(
                        f"<div class='factor-row'>{arrow} <b>{f['feature']}</b> — {f['direction']}</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("### 📩 Recommended Retention Offer")
            st.markdown(f"<div class='answer-card'><h3>English</h3>{d['retention_offer']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-card' style='margin-top:1rem;direction:rtl;text-align:right;'><h3>العربية</h3>{d['retention_offer_ar']}</div>", unsafe_allow_html=True)
            st.caption(f"Model version: {d['model_version']}")
        else:
            st.error(f"Error: {r.status_code} — {r.text}")

# ============ TAB 3: Batch Score ============
with tabs[2]:
    st.markdown("### Score a sample of customers")
    n = st.slider("Sample size", 5, 50, 10)
    if st.button("Run batch scoring"):
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
    st.markdown("### Upload a dataset")
    st.caption("Accepted formats: CSV, Excel (.xlsx, .xls), JSON, Parquet. Column names are auto-mapped.")

    with st.expander("📋 Show accepted column aliases"):
        try:
            schema = requests.get(f"{API}/datasets/schema", timeout=5).json()
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
                    c1.markdown(f"<div class='stat-card'><h3>{m['roc_auc']:.3f}</h3><p>ROC-AUC</p></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='stat-card'><h3>{m['f1']:.3f}</h3><p>F1</p></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='stat-card'><h3>{rt['algorithm'].upper()}</h3><p>Algorithm</p></div>", unsafe_allow_html=True)
                if "retrain_error" in d:
                    st.error(f"Retrain failed: {d['retrain_error']}")
        else:
            st.error(r.text)

    st.markdown("---")
    st.markdown("### Version history")
    if st.button("🔄 Refresh versions"):
        r = requests.get(f"{API}/datasets/versions", timeout=10)
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