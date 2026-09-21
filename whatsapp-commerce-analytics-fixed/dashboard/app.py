"""
WhatsApp Commerce Analytics — Streamlit dashboard.

Run locally:
    streamlit run app.py
"""
import os
from pathlib import Path

import pandas as pd
import psycopg2
import streamlit as st

st.set_page_config(
    page_title="WhatsApp Commerce Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSET_DIR = Path(__file__).parent / "assets"

DEFAULT_DSN = (
    "dbname=whatsapp_commerce "
    "user=postgres "
    "password=postgres "
    "host=localhost"
)

DSN = os.environ.get("DB_DSN")

if not DSN:
    try:
        DSN = st.secrets["DB_DSN"]
    except (FileNotFoundError, KeyError):
        DSN = DEFAULT_DSN


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root { --ink:#e9eef5; --muted:#8d9aaa; --panel:#141b24; --panel-2:#1a2430; --line:#273342; --teal:#36d6b0; --amber:#ffbe69; }
    .stApp { background: #0c1117; color: var(--ink); font-family: 'Manrope', sans-serif; }
    [data-testid="stHeader"] { background: rgba(12,17,23,.92); }
    [data-testid="stSidebar"] { background:#101720; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding: 1.35rem 1rem; }
    [data-testid="stSidebar"] .stRadio > label { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; font-weight:800; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap:.35rem; }
    [data-testid="stSidebar"] [role="radio"] { padding:.65rem .75rem; border-radius:10px; color:#b9c5d2; transition:all .18s ease; }
    [data-testid="stSidebar"] [role="radio"]:hover { background:#1a2633; color:white; }
    [data-testid="stSidebar"] [role="radio"] [data-testid="stMarkdownContainer"] p { font-size:.86rem; }
    [data-testid="stSidebar"] [role="radio"]:has(input:checked) { background:rgba(54,214,176,.12); color:var(--teal); }
    .brand { display:flex; align-items:center; gap:.7rem; margin-bottom:2.3rem; }
    .brand-mark { width:35px; height:35px; display:grid; place-items:center; border-radius:10px; background:var(--teal); color:#071711; font-size:1.25rem; font-weight:800; box-shadow:0 0 24px rgba(54,214,176,.2); }
    .brand-name { color:#f4f7fb; font-weight:800; font-size:.95rem; letter-spacing:-.02em; line-height:1.1; }
    .brand-sub { color:var(--muted); font-size:.62rem; letter-spacing:.13em; text-transform:uppercase; margin-top:.2rem; }
    .sidebar-foot { color:#5e6c7a; font-size:.68rem; line-height:1.5; border-top:1px solid var(--line); padding-top:1rem; margin-top:2.5rem; }
    .eyebrow { color:var(--teal); font-family:'DM Mono', monospace; font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.55rem; }
    h1 { font-size:2rem !important; letter-spacing:-.055em; margin:0 !important; color:#f5f8fb !important; }
    h2, h3 { color:#f5f8fb !important; letter-spacing:-.035em; }
    .page-subtitle { color:var(--muted); font-size:.88rem; max-width:780px; margin-top:.5rem; line-height:1.6; }
    .topline { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; margin-bottom:1.5rem; }
    .live-pill { white-space:nowrap; padding:.38rem .65rem; border:1px solid rgba(54,214,176,.28); background:rgba(54,214,176,.08); border-radius:999px; color:var(--teal); font-family:'DM Mono',monospace; font-size:.68rem; }
    .live-dot { display:inline-block; width:6px; height:6px; background:var(--teal); border-radius:50%; margin:0 .35rem .05rem 0; box-shadow:0 0 0 4px rgba(54,214,176,.12); }
    .metric-card { background:linear-gradient(145deg,#17222d,#121922); border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.15rem; min-height:112px; box-shadow:0 10px 30px rgba(0,0,0,.12); }
    .metric-label { color:var(--muted); font-size:.68rem; text-transform:uppercase; letter-spacing:.12em; font-weight:800; }
    .metric-value { color:#f5f8fb; font-size:1.75rem; font-weight:800; letter-spacing:-.06em; margin:.45rem 0 .15rem; }
    .metric-delta { color:var(--teal); font-family:'DM Mono',monospace; font-size:.7rem; }
    .section-head { display:flex; align-items:baseline; gap:.65rem; margin:1.9rem 0 .7rem; }
    .section-head h3 { margin:0; font-size:1rem; }
    .section-head span { color:#667586; font-family:'DM Mono',monospace; font-size:.67rem; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:1rem 1.15rem .75rem; }
    .insight { border-left:3px solid var(--amber); background:rgba(255,190,105,.07); padding:.8rem 1rem; border-radius:0 10px 10px 0; color:#c7d0db; font-size:.82rem; line-height:1.55; }
    .insight strong { color:var(--amber); }
    [data-testid="stMetric"] { background:transparent; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    .stCaption, [data-testid="stCaptionContainer"] { color:var(--muted) !important; }
    .stImage img { border-radius:12px; border:1px solid var(--line); }
    [data-testid="stAlert"] { background:#17222d; border:1px solid var(--line); color:#c7d0db; }
    @media (max-width: 760px) { h1 { font-size:1.55rem !important; } .topline { flex-direction:column; } .live-pill { align-self:flex-start; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def query(sql: str) -> pd.DataFrame:
    """Run a cached analytics query against the configured Postgres database."""
    conn = psycopg2.connect(DSN)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


query = st.cache_data(ttl=600)(query)


def metric_card(label: str, value: str, delta: str, accent: str = "teal") -> None:
    color = "var(--amber)" if accent == "amber" else "var(--teal)"
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-delta" style="color:{color}">{delta}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, detail: str) -> None:
    st.markdown(f'<div class="section-head"><h3>{title}</h3><span>{detail}</span></div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-name">Commerce Intel</div><div class="brand-sub">WhatsApp operations</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">Workspace</div>', unsafe_allow_html=True)
    page = st.radio(
        "View",
        ["Merchant Overview", "Order Fulfillment", "Cohort Retention", "Agent Performance"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="sidebar-foot"><b style="color:#aeb9c6">DATA WINDOW</b><br>6 months · 300 merchants<br><br><b style="color:#aeb9c6">SOURCE</b><br>Synthetic commerce operations dataset<br><br><span style="color:#36d6b0">●</span> PostgreSQL connection ready</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="topline"><div><div class="eyebrow">Operations intelligence / {page.lower()}</div><h1>{page}</h1><div class="page-subtitle">A focused view of the signals that move merchant growth, order reliability, and automated support quality.</div></div><div class="live-pill"><span class="live-dot"></span>DATA REFRESHED · 10 MIN</div></div>',
    unsafe_allow_html=True,
)

if page == "Merchant Overview":
    df = query("""
        SELECT merchant_id, signup_date, first_conversation_at,
               first_order_at, tenth_order_at, lifetime_orders
        FROM merchant_funnel
    """)
    total = len(df)
    engaged = df.first_conversation_at.notna().sum()
    activated = df.first_order_at.notna().sum()
    power_users = df.tenth_order_at.notna().sum()
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Signed up", f"{total:,}", "TOTAL MERCHANTS")
    with c2: metric_card("Engaged", f"{engaged:,}", f"{engaged/total:.1%} started chat")
    with c3: metric_card("Activated", f"{activated:,}", f"{activated/total:.1%} placed an order")
    with c4: metric_card("Power users", f"{power_users:,}", f"{power_users/total:.1%} reached 10 orders", "amber")

    section("Activation funnel", "SIGNUP → REPEAT USAGE")
    funnel_df = pd.DataFrame({"stage": ["Signed up", "Started a conversation", "Placed 1st order", "Reached 10th order"], "merchants": [total, engaged, activated, power_users]}).set_index("stage")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.bar_chart(funnel_df, color="#36d6b0", height=280)
    st.markdown('</div>', unsafe_allow_html=True)

    section("Merchant depth", "LIFETIME ORDERS DISTRIBUTION")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.bar_chart(df.lifetime_orders.value_counts().sort_index(), color="#ffbe69", height=250)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight"><strong>Signal to watch:</strong> the largest drop-off happens after the first order. Retention work should focus on turning first-time buyers into repeat merchants.</div>', unsafe_allow_html=True)

elif page == "Order Fulfillment":
    df = query("SELECT * FROM fulfillment_summary")
    avg_fulfillment = df.fulfillment_rate_pct.mean()
    slowest = df.loc[df.median_fulfillment_hours.idxmax()]
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Avg. fulfillment", f"{avg_fulfillment:.1f}%", "ACROSS CATEGORIES")
    with c2: metric_card("Slowest category", str(slowest["category"]).replace("_", " ").title(), f"{slowest['median_fulfillment_hours']:.1f}h median", "amber")
    with c3: metric_card("Categories tracked", f"{len(df)}", "MERCHANT SEGMENTS")
    section("Fulfillment health", "CATEGORY BENCHMARK")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        section("Fulfillment rate", "% OF ORDERS COMPLETED")
        st.bar_chart(df.set_index("category")["fulfillment_rate_pct"], color="#36d6b0", height=300)
    with c2:
        section("Time to fulfill", "MEDIAN HOURS")
        st.bar_chart(df.set_index("category")["median_fulfillment_hours"], color="#ffbe69", height=300)
    worst = df.sort_values("fulfillment_rate_pct").iloc[0]
    st.markdown(f'<div class="insight"><strong>Priority:</strong> <b>{worst["category"]}</b> has the lowest fulfillment rate at {worst["fulfillment_rate_pct"]}%. Review its delivery process before optimizing already-healthy categories.</div>', unsafe_allow_html=True)

elif page == "Cohort Retention":
    df = query("SELECT * FROM merchant_cohort_retention")
    pivot = df.pivot(index="cohort_week", columns="week_offset", values="retention_pct")
    section("Weekly retention matrix", "SIGNUP COHORT × WEEKS SINCE SIGNUP")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.dataframe(pivot.style.background_gradient(cmap="GnBu", axis=None).format("{:.0f}%"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight"><strong>How to read it:</strong> scan across a row to see how quickly a signup cohort decays. Compare the first four weeks across cohorts to spot changes in onboarding quality.</div>', unsafe_allow_html=True)

else:
    df = query("SELECT * FROM agent_containment")
    avg_containment = df.containment_rate_pct.mean()
    avg_accuracy = df.accuracy_pct.mean()
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Avg. containment", f"{avg_containment:.1f}%", "RESOLVED WITHOUT ESCALATION")
    with c2: metric_card("Avg. accuracy", f"{avg_accuracy:.1f}%", "PREDICTION QUALITY")
    with c3: metric_card("Recommended cutoff", "0.50", "CONFIDENCE → HUMAN ESCALATION", "amber")
    section("Intent performance", "CONTAINMENT + ACCURACY")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        section("Containment rate", "RESOLVED AUTOMATICALLY")
        st.bar_chart(df.set_index("predicted_intent")["containment_rate_pct"], color="#36d6b0", height=320)
    with c2:
        section("Accuracy", "CORRECT INTENT PREDICTIONS")
        st.bar_chart(df.set_index("predicted_intent")["accuracy_pct"], color="#ffbe69", height=320)
    section("Confidence threshold trade-off", "ESCALATION POLICY")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.image(str(ASSET_DIR / "threshold_tradeoff.png"), use_container_width=True)
    with c2:
        st.markdown('<div class="insight"><strong>Recommended policy:</strong><br><br>Below a <b>0.50 confidence threshold</b>, auto-handled error rate climbs above 5%. Escalating low-confidence conversations keeps automated resolutions reliable while still handling roughly 92% of volume automatically.</div>', unsafe_allow_html=True)
        st.image(str(ASSET_DIR / "f1_by_intent.png"), use_container_width=True)
    section("Intent quality detail", "F1 SCORE BY INTENT")
    st.image(str(ASSET_DIR / "confidence_distribution.png"), use_container_width=True)
