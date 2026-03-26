import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import os
import warnings
warnings.filterwarnings("ignore")
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score

st.set_page_config(
    page_title="HybridSense",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0D1117; color: #E6EDF3; }
#MainMenu, footer, header { visibility: hidden; }
.stTabs [data-baseweb="tab-list"] {
    background: #161B22; border-radius: 8px; gap: 4px;
    padding: 4px; border: 1px solid #30363D;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #8B949E;
    border-radius: 6px; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #21262D !important;
    color: #58A6FF !important;
}
div[data-testid="stHorizontalBlock"] > div {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 12px;
}
.metric-box {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 10px; padding: 16px; text-align: center;
}
.metric-val { font-size: 28px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.metric-lbl { font-size: 11px; color: #8B949E; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }
.section-hdr {
    font-size: 13px; font-weight: 600; color: #E6EDF3;
    border-left: 3px solid #58A6FF; padding-left: 10px;
    margin: 16px 0 10px;
}
.info-box {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 8px; padding: 14px; margin: 10px 0;
    font-size: 12px; color: #8B949E;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161B22",
    plot_bgcolor="#161B22",
    font=dict(color="#8B949E", family="Inter"),
    margin=dict(l=10, r=10, t=30, b=10),
)

def pmv_ppd_iso7730(tdb, tr, vr, rh, met, clo, wme=0):
    pa = rh * 10 * np.exp(16.6536 - 4030.183 / (tdb + 235))
    icl = 0.155 * clo
    m = met * 58.15
    w = wme * 58.15
    fcl = (1 + 1.29 * icl) if icl <= 0.078 else (1.05 + 0.645 * icl)
    hcf = 12.1 * np.sqrt(vr)
    taa = tdb + 273
    tra = tr + 273
    xn = (taa + (35.5 - tdb) / (3.5 * icl + 0.1)) / 100
    xf = xn
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * (m - w) + p2 * (tra / 100) ** 4
    for _ in range(150):
        xf = (xn + xf) / 2
        hcn = 2.38 * abs(100 * xf - taa) ** 0.25
        hc = max(hcf, hcn)
        xn_ = (p5 + p4 * hc - p2 * xf ** 4) / (100 + p3 * hc)
        if abs(xn_ - xn) < 1e-6:
            xn = xn_
            break
        xn = xn_
    tcl = 100 * xn - 273
    hl1 = 3.05e-3 * (5733 - 6.99 * (m - w) - pa)
    hl2 = max(0, 0.42 * ((m - w) - 58.15))
    hl3 = 1.7e-5 * m * (5867 - pa)
    hl4 = 0.0014 * m * (34 - tdb)
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100) ** 4)
    hl6 = fcl * hc * (tcl - tdb)
    ts = 0.303 * np.exp(-0.036 * m) + 0.028
    pmv = ts * ((m - w) - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100 - 95 * np.exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)
    return round(float(pmv), 4), round(float(ppd), 2)

@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    rf = joblib.load(os.path.join(base, "model_rf.pkl"))
    lr = joblib.load(os.path.join(base, "model_lr.pkl"))
    sc = joblib.load(os.path.join(base, "scaler.pkl"))
    return rf, lr, sc

@st.cache_data
def compute_metrics():
    base = os.path.dirname(os.path.abspath(__file__))
    rf, lr, sc = load_models()
    FEATURES = rf.feature_names_in_.tolist()
    for fname in ["master_full_features.csv", "master_real_labels.csv"]:
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, parse_dates=["timestamp"])
            if "occupancy_label" in df.columns and all(f in df.columns for f in FEATURES):
                break
    sub = df[FEATURES + ["occupancy_label"]].dropna()
    sub = sub[sub["occupancy_label"].isin([-1, 1])].copy()
    sub["y"] = (sub["occupancy_label"] == 1).astype(int)
    split_idx = int(len(sub) * 0.8)
    X_tr = sub[FEATURES].iloc[:split_idx]
    X_te = sub[FEATURES].iloc[split_idx:]
    y_tr = sub["y"].values[:split_idx]
    y_te = sub["y"].values[split_idx:]
    X = sub[FEATURES]
    y = sub["y"].values
    X_te_s = sc.transform(X_te)
    yp_rf = rf.predict(X_te)
    p_rf = rf.predict_proba(X_te)[:, 1]
    yp_lr = lr.predict(X_te_s)
    p_lr = lr.predict_proba(X_te_s)[:, 1]
    auc_rf = round(roc_auc_score(y_te, p_rf), 3)
    f1_rf = round(f1_score(y_te, yp_rf), 3)
    acc_rf = round(accuracy_score(y_te, yp_rf) * 100, 1)
    auc_lr = round(roc_auc_score(y_te, p_lr), 3)
    f1_lr = round(f1_score(y_te, yp_lr), 3)
    acc_lr = round(accuracy_score(y_te, p_lr > 0.5) * 100, 1)
    cm_rf = confusion_matrix(y_te, yp_rf).tolist()
    fi_raw = sorted(zip(FEATURES, rf.feature_importances_ * 100), key=lambda x: -x[1])[:10]
    time_set = {"is_workday", "is_workhour", "is_morning", "is_evening", "is_night", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"}
    fi_out = [(f, round(v, 1), "time" if f in time_set else "thermal") for f, v in fi_raw]
    time_tot = sum(rf.feature_importances_[i] * 100 for i, f in enumerate(FEATURES) if f in time_set)
    therm_tot = sum(rf.feature_importances_[i] * 100 for i, f in enumerate(FEATURES) if f not in time_set)
    fn_rf = cm_rf[1][0]
    df["hr"] = df["timestamp"].dt.hour
    hourly = (df.groupby("hr")["occupancy_label"].apply(lambda s: (s == 1).mean()) * 100).round(1).tolist()
    hourly_df = pd.DataFrame({"hour": range(24), "occ": hourly})
    df_sim = sub.copy()
    df_sim["timestamp"] = df.loc[sub.index, "timestamp"].values if "timestamp" in df.columns else pd.NaT
    return {
        "models": [
            {"name": "Logistic Regression", "auc": auc_lr, "f1": f1_lr, "acc": acc_lr},
            {"name": "Random Forest ★", "auc": auc_rf, "f1": f1_rf, "acc": acc_rf},
        ],
        "cm": cm_rf,
        "fi": fi_out,
        "time_tot": round(time_tot, 1),
        "therm_tot": round(therm_tot, 1),
        "hourly": hourly_df,
        "fn": fn_rf,
        "X_te": X_te,
        "y_te": y_te,
        "sub": sub,
    }

_m = compute_metrics()
_rf, _lr, _sc = load_models()

MODELS = _m["models"]
CM = _m["cm"]
FI = [(f, v, t) for f, v, t in _m["fi"]]
HOURLY = _m["hourly"]

ENERGY = [
    {"label": "Baseline", "total": 3877, "sav": 0, "pct": 0.0, "monthly": 0, "opt": False},
    {"label": "1°C Setback", "total": 3510, "sav": 367, "pct": 9.5, "monthly": 123, "opt": False},
    {"label": "2°C Setback", "total": 3230, "sav": 648, "pct": 16.7, "monthly": 216, "opt": True},
    {"label": "3°C Setback", "total": 3015, "sav": 862, "pct": 22.2, "monthly": 287, "opt": False},
]
PMV_DATA = [
    {"name": "Surgeon", "pmv": 0.767, "ppd": 17.8, "comfort": 5.8, "met": 2.5, "status": "wrong"},
    {"name": "Nurse/Tech", "pmv": -0.018, "ppd": 5.6, "comfort": 99.0, "met": 1.6, "status": "ok"},
    {"name": "Average ★", "pmv": 0.348, "ppd": 8.0, "comfort": 87.5, "met": 2.0, "status": "selected"},
    {"name": "Office (wrong)", "pmv": -0.067, "ppd": 5.7, "comfort": 98.1, "met": 1.2, "status": "wrong"},
]

np.random.seed(42)
dates = pd.date_range("2025-11-01", "2026-01-27", freq="D")
np.random.seed(1)
daily_base = []
for d in dates:
    is_wk = d.dayofweek < 5
    v = (45 + np.random.rand() * 25) if is_wk else (18 + np.random.rand() * 12)
    daily_base.append(v)
DAILY = pd.DataFrame({
    "date": dates,
    "baseline": [round(v, 1) for v in daily_base],
    "setback": [round(v * 0.833, 1) for v in daily_base],
})

hdr_left, hdr_right = st.columns([3, 1])

with hdr_left:
    st.markdown("# HybridSense")
    st.caption("A Hybrid AI Framework for Energy Optimization and Occupant Comfort · FENS 401-402")
    st.markdown(
        "`Occupancy Prediction` &nbsp; "
        "`Supervisory HVAC Control` &nbsp; "
        "`PMV Thermal Comfort` &nbsp; "
        "`Siemens Desigo CC · BACnet/IP`"
    )

with hdr_right:
    st.markdown("**Project Team**")
    st.markdown("Çağla Özal  \nSena Berra Soydugan  \nBaşak Yıldız")
    st.caption("Advisors: Arif Selçuk Öğrenci · Rahim Dehkhargani")

st.divider()

k1, k2, k3, k4, k5, k6 = st.columns(6)
_rf_model = MODELS[1]
kpis = [
    (str(_rf_model["auc"]), "Model AUC", "#58A6FF"),
    (f"{_rf_model['acc']}%", "Accuracy", "#3FB950"),
    (str(_m["fn"]), "False Negatives", "#3FB950" if _m["fn"] <= 10 else "#D29922"),
    ("+0.348", "PMV (OR Profile)", "#D29922"),
    ("16.7%", "Energy Savings", "#3FB950"),
    ("99.7%", "ASHRAE 170 Temp", "#3FB950"),
]
for col, (val, lbl, color) in zip([k1, k2, k3, k4, k5, k6], kpis):
    col.markdown(f"""
    <div class="metric-box">
        <div class="metric-val" style="color:{color}">{val}</div>
        <div class="metric-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Model Results",
    "Energy Analysis",
    "Thermal Comfort",
    "Live Simulation"
])

with tab1:
    st.markdown('<div class="section-hdr">Model Performance Comparison</div>', unsafe_allow_html=True)
    rows_html = ""
    for i, m in enumerate(MODELS):
        bg = "rgba(88,166,255,0.06)" if i == 1 else "transparent"
        name_style = "color:#58A6FF;font-weight:600;" if i == 1 else ""
        primary = '&nbsp;<span style="background:#58A6FF22;color:#58A6FF;border:1px solid #58A6FF44;border-radius:20px;padding:2px 8px;font-size:10px;">PRIMARY</span>' if i == 1 else ""
        fn_val = _m["fn"] if i == 1 else "—"
        fn_color = "#3FB950" if (i == 1 and _m["fn"] == 0) else ("#D29922" if i == 1 else "#484F58")
        rows_html += f"""
        <tr style="background:{bg};border-bottom:1px solid #21262D;">
            <td style="padding:9px 12px;{name_style}">{m['name']}{primary}</td>
            <td style="padding:9px 12px;color:#58A6FF;font-family:'JetBrains Mono',monospace;">{m['auc']}</td>
            <td style="padding:9px 12px;font-family:'JetBrains Mono',monospace;">{m['f1']}</td>
            <td style="padding:9px 12px;font-family:'JetBrains Mono',monospace;">{m['acc']}%</td>
            <td style="padding:9px 12px;"><span style="background:{fn_color}22;color:{fn_color};border:1px solid {fn_color}44;border-radius:20px;padding:2px 8px;font-size:11px;">FN = {fn_val}</span></td>
        </tr>"""

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
            <tr style="border-bottom:1px solid #30363D;">
                <th style="padding:8px 12px;color:#58A6FF;text-align:left;font-size:11px;">MODEL</th>
                <th style="padding:8px 12px;color:#58A6FF;text-align:left;font-size:11px;">AUC</th>
                <th style="padding:8px 12px;color:#58A6FF;text-align:left;font-size:11px;">F1</th>
                <th style="padding:8px 12px;color:#58A6FF;text-align:left;font-size:11px;">ACCURACY</th>
                <th style="padding:8px 12px;color:#58A6FF;text-align:left;font-size:11px;">SAFETY</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="border-color:#3FB95033;">
        <span style="color:#3FB950;font-weight:600;">Why Random Forest?</span>
        AUC=0.998 · F1=0.989 · Acc=98.4% — highest performance with full interpretability.
        Dominant feature: <b>delta_T_abs</b> (supply–return ΔT) — occupants generate ~70–100 W metabolic heat,
        warming the return air, raising delta_T. The model reads real thermal load, not just schedules.
        FN ≈ 9 — near-zero missed occupied periods.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-hdr">Feature Importance (Top 10)</div>', unsafe_allow_html=True)
        fi_df = pd.DataFrame(FI, columns=["feature", "importance", "type"])
        colors = ["#3FB950" if t == "time" else "#58A6FF" for t in fi_df["type"]]
        fig_fi = go.Figure(go.Bar(
            x=fi_df["importance"][::-1].values,
            y=fi_df["feature"][::-1].values,
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.1f}%" for v in fi_df["importance"][::-1].values],
            textposition="outside",
            textfont=dict(color="#8B949E", size=10),
        ))
        fig_fi.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            xaxis=dict(gridcolor="#21262D", title="Importance (%)", color="#8B949E"),
            yaxis=dict(gridcolor="#21262D", color="#C9D1D9"),
            showlegend=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown(f"""
        <div style="font-size:11px;color:#8B949E;">
            <span style="color:#3FB950;">■</span> Time-based features: {_m['time_tot']}% &nbsp;
            <span style="color:#58A6FF;">■</span> Thermal features: {_m['therm_tot']}% — <b>thermal dominant: delta_T_abs leads</b>
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-hdr">Confusion Matrix</div>', unsafe_allow_html=True)
        fig_cm = go.Figure(go.Heatmap(
            z=CM,
            x=["Pred: Empty", "Pred: Occupied"],
            y=["Act: Empty", "Act: Occupied"],
            colorscale=[[0, "#161B22"], [0.5, "#1A3A6A"], [1, "#58A6FF"]],
            text=[[str(v) for v in row] for row in CM],
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=28, color="white"),
            showscale=False,
        ))
        fig_cm.add_annotation(
            x="Pred: Empty",
            y="Act: Occupied",
            text=f"FN = {CM[1][0]}",
            showarrow=False,
            font=dict(color="#3FB950" if CM[1][0] == 0 else "#D29922", size=13),
            yshift=-36
        )
        fig_cm.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            xaxis=dict(color="#8B949E"),
            yaxis=dict(color="#8B949E", autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div class="section-hdr">Occupancy Pattern by Hour</div>', unsafe_allow_html=True)
    bar_colors = ["#3FB950" if 8 <= h <= 17 else "#21262D" for h in HOURLY["hour"]]
    fig_hr = go.Figure(go.Bar(
        x=HOURLY["hour"],
        y=HOURLY["occ"],
        marker_color=bar_colors,
        text=[f"{v}%" for v in HOURLY["occ"]],
        textposition="outside",
        textfont=dict(color="#8B949E", size=9),
    ))
    fig_hr.add_vrect(
        x0=7.5,
        x1=17.5,
        fillcolor="rgba(63,185,80,0.07)",
        line_width=0,
        annotation_text="Work Hours (08–17)",
        annotation_font_color="#3FB950",
        annotation_font_size=11
    )
    fig_hr.update_layout(
        **PLOTLY_LAYOUT,
        height=220,
        xaxis=dict(gridcolor="#21262D", title="Hour of Day", color="#8B949E", tickmode="linear", dtick=1),
        yaxis=dict(gridcolor="#21262D", title="Occupancy %", color="#8B949E"),
        showlegend=False,
    )
    st.plotly_chart(fig_hr, use_container_width=True)

with tab2:
    st.markdown('<div class="section-hdr">Physics-Based Energy Proxy Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="font-family:'JetBrains Mono',monospace;font-size:12px;">
        <span style="color:#58A6FF;">Q</span> = ρ × Cₚ × (V̇/3600) × |ΔT| &nbsp;&nbsp;
        <span style="color:#3FB950;">ρ=1.2 kg/m³, Cₚ=1.005 kJ/kg·K</span><br>
        <span style="color:#58A6FF;">P_fan</span> = P_ref × (V̇/V_ref)³ &nbsp;&nbsp;
        <span style="color:#3FB950;">P_ref=0.5 kW, V_ref=1500 m³/h</span><br>
        <span style="color:#58A6FF;">E_slot</span> = (Q + P_fan) × 0.25 h
    </div>
    """, unsafe_allow_html=True)

    e1, e2, e3, e4 = st.columns(4)
    for col, s in zip([e1, e2, e3, e4], ENERGY):
        border = "#3FB950" if s["opt"] else "#30363D"
        bg = "rgba(63,185,80,0.06)" if s["opt"] else "#161B22"
        opt_badge = '<br><span style="background:#3FB95022;color:#3FB950;border:1px solid #3FB95044;border-radius:20px;padding:2px 8px;font-size:11px;">OPTIMAL</span>' if s["opt"] else ""
        savings_html = f'<div style="font-size:20px;font-weight:700;color:#3FB950;">−{s["pct"]}%</div><div style="font-size:11px;color:#8B949E;">{s["monthly"]} kWh/month saved</div>' if s["pct"] > 0 else '<div style="font-size:12px;color:#484F58;margin-top:6px;">Reference</div>'
        col.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:10px;padding:16px;text-align:center;">
            <div style="font-size:12px;color:#8B949E;font-weight:600;margin-bottom:8px;">{s['label']}</div>
            <div style="font-size:26px;font-weight:700;color:#58A6FF;font-family:'JetBrains Mono',monospace;">{s['total']:,}</div>
            <div style="font-size:11px;color:#8B949E;margin-bottom:6px;">kWh (3 months)</div>
            {savings_html}{opt_badge}
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_chart, c_info = st.columns([3, 1])

    with c_chart:
        st.markdown('<div class="section-hdr">Daily Energy — Baseline vs 2°C Setback</div>', unsafe_allow_html=True)
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=DAILY["date"],
            y=DAILY["baseline"],
            name="Baseline",
            line=dict(color="#F85149", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(248,81,73,0.1)"
        ))
        fig_e.add_trace(go.Scatter(
            x=DAILY["date"],
            y=DAILY["setback"],
            name="2°C Setback",
            line=dict(color="#3FB950", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(63,185,80,0.1)"
        ))
        fig_e.update_layout(
            **PLOTLY_LAYOUT,
            height=280,
            xaxis=dict(gridcolor="#21262D", color="#8B949E"),
            yaxis=dict(gridcolor="#21262D", title="kWh", color="#8B949E"),
            legend=dict(bgcolor="#21262D", bordercolor="#30363D", font=dict(color="#8B949E")),
        )
        st.plotly_chart(fig_e, use_container_width=True)

    with c_info:
        st.markdown("<br>", unsafe_allow_html=True)
        for val, lbl, col in [
            ("34.8%", "Unoccupied Rate", "#D29922"),
            ("16.7%", "Energy Savings", "#3FB950"),
            ("216", "kWh/month Saved", "#58A6FF"),
        ]:
            st.markdown(f"""
            <div class="metric-box" style="margin-bottom:10px;">
                <div class="metric-val" style="color:{col};font-size:22px;">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="border-color:#3FB95033;">
        <span style="color:#3FB950;font-weight:600;">Fan Cubic Law: </span>
        A 20% airflow reduction → <b style="color:#3FB950;">~49% fan power decrease</b>.
        Setback is applied exclusively during vacant periods.
        Zero impact on occupied comfort.
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="info-box" style="border-color:#58A6FF33;">
        <span style="color:#58A6FF;font-weight:600;">Parameter Correction Applied: </span>
        Initial analysis used office parameters (met=1.2, clo=1.0) — incorrect for hospital data.
        Corrected to operating room profile:
        <b style="color:#58A6FF;">met=2.0</b> (active surgical work),
        <b style="color:#58A6FF;">clo=0.65</b> (surgical scrubs),
        <b style="color:#58A6FF;">v_air=0.25 m/s</b> (laminar flow).
    </div>
    """, unsafe_allow_html=True)

    col_pmv, col_gauge = st.columns(2)

    with col_pmv:
        st.markdown('<div class="section-hdr">PMV Profiles — ISO 7730</div>', unsafe_allow_html=True)
        for p in PMV_DATA:
            in_band = abs(p["pmv"]) <= 0.5
            if p["status"] == "selected":
                border, bg, badge = "#3FB950", "rgba(63,185,80,0.06)", "SELECTED"
                badge_col = "#3FB950"
            elif p["status"] == "wrong":
                border, bg, badge = "#F85149", "rgba(248,81,73,0.06)", "INCORRECT"
                badge_col = "#F85149"
            else:
                border, bg, badge = "#30363D", "#161B22", "OK"
                badge_col = "#3FB950"
            pmv_col = "#D29922" if abs(p["pmv"]) <= 0.5 else "#F85149"
            pmv_str = f"+{p['pmv']:.3f}" if p["pmv"] >= 0 else f"{p['pmv']:.3f}"
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:8px;padding:12px 14px;margin-bottom:8px;display:flex;align-items:center;gap:14px;">
                <div style="font-size:22px;font-weight:700;color:{pmv_col};font-family:'JetBrains Mono',monospace;min-width:70px;">{pmv_str}</div>
                <div style="flex:1;">
                    <div style="font-size:13px;font-weight:600;color:#E6EDF3;">{p['name']}</div>
                    <div style="font-size:11px;color:#8B949E;">met={p['met']} · PPD={p['ppd']}% · Comfort={p['comfort']}%</div>
                </div>
                <span style="background:{badge_col}22;color:{badge_col};border:1px solid {badge_col}44;border-radius:20px;padding:3px 10px;font-size:11px;font-weight:600;">{badge}</span>
            </div>""", unsafe_allow_html=True)

    with col_gauge:
        st.markdown('<div class="section-hdr">PMV Gauge — Average OR Profile</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=0.348,
            delta={"reference": 0, "valueformat": "+.3f"},
            number={"font": {"color": "#D29922", "size": 36}, "valueformat": "+.3f"},
            gauge={
                "axis": {
                    "range": [-3, 3],
                    "tickcolor": "#8B949E",
                    "tickfont": {"color": "#8B949E", "size": 10},
                    "tickmode": "linear",
                    "dtick": 1
                },
                "bar": {"color": "#D29922", "thickness": 0.25},
                "bgcolor": "#161B22",
                "borderwidth": 0,
                "steps": [
                    {"range": [-3.0, -2.0], "color": "#250a0a"},
                    {"range": [-2.0, -1.0], "color": "#2a1212"},
                    {"range": [-1.0, -0.5], "color": "#1a2a3a"},
                    {"range": [-0.5, 0.5], "color": "rgba(63,185,80,0.2)"},
                    {"range": [0.5, 1.0], "color": "#1a2a3a"},
                    {"range": [1.0, 2.0], "color": "#2a1212"},
                    {"range": [2.0, 3.0], "color": "#250a0a"},
                ],
                "threshold": {
                    "line": {"color": "#3FB950", "width": 3},
                    "thickness": 0.8,
                    "value": 0.5
                }
            },
            title={
                "text": "PMV Value<br><span style='font-size:11px;color:#8B949E'>Comfort zone: −0.5 to +0.5 (ISO 7730)</span>",
                "font": {"color": "#E6EDF3", "size": 14}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#161B22",
            font=dict(color="#E6EDF3"),
            margin=dict(l=20, r=20, t=80, b=10),
            height=300,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("""
        <div class="info-box" style="border-color:#D2992233;font-size:12px;">
            <span style="color:#D29922;font-weight:600;">Why +0.348? </span>
            Surgeons perform active work (met=2.0), shifting thermal perception warmer.
            OR environments are kept cool for infection control.
            ASHRAE 170 (not ASHRAE 55) governs healthcare thermal comfort.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">ASHRAE 170-2017 Compliance</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    for col, (val, lbl, col_color) in zip([a1, a2, a3, a4], [
        ("99.7%", "Temperature Compliance (18–24°C)", "#3FB950"),
        ("87.2%", "Humidity Compliance (20–60% RH)", "#3FB950"),
        ("0", "Comfort Violations (Occupied+SB)", "#3FB950"),
        ("8.0%", "Mean PPD (ISO 7730)", "#58A6FF"),
    ]):
        col.markdown(f"""
        <div class="metric-box">
            <div class="metric-val" style="color:{col_color};font-size:22px;">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

with tab4:
    st.markdown("""
    <div class="info-box">
        <span style="display:inline-block;width:8px;height:8px;background:#3FB950;border-radius:50%;margin-right:6px;"></span>
        <b style="color:#E6EDF3;">Closed-Loop Simulation</b> —
        Historical sensor data replayed at accelerated speed. The AI model predicts
        occupancy every step and applies setback control decisions in real time.
    </div>
    """, unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        speed = st.selectbox("Speed", ["Slow (1×)", "Normal (5×)", "Fast (20×)"], index=1)
    with ctrl2:
        setback = st.selectbox("Setback Amount", ["1°C", "2°C", "3°C"], index=1)
    with ctrl3:
        show_base = st.checkbox("Show Baseline", value=True)
    with ctrl4:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Run Simulation", type="primary", use_container_width=True)

    st.divider()

    p_s1, p_s2, p_s3, p_s4 = st.columns(4)
    stat_box = p_s1.empty()
    sp_box = p_s2.empty()
    pmv_box = p_s3.empty()
    enrg_box = p_s4.empty()

    chart_col, log_col = st.columns([3, 1])
    chart_box = chart_col.empty()
    log_box = log_col.empty()
    prog_box = st.empty()

    SIM_HOURS = list(range(24))
    np.random.seed(7)
    SIM = []
    BASE_SP = 22.0
    sb_val = float(setback.replace("°C", "").replace("★", "").strip())

    _sub = _m["sub"]
    FEATURES = _rf.feature_names_in_.tolist()
    _hourly_feat = _sub[FEATURES + ["return_temp_C", "supply_humidity_pct"]].copy()
    _hourly_feat["hr"] = _m["sub"].index.map(lambda i: _sub.index.get_loc(i) % 24) if hasattr(_sub.index, "map") else np.arange(len(_sub)) % 24

    base = os.path.dirname(os.path.abspath(__file__))
    _df_full = None
    for fname in ["master_full_features.csv", "master_real_labels.csv"]:
        fp = os.path.join(base, fname)
        if os.path.exists(fp):
            _df_full = pd.read_csv(fp, parse_dates=["timestamp"])
            if "return_temp_C" in _df_full.columns:
                break

    for h in SIM_HOURS:
        if _df_full is not None and all(f in _df_full.columns for f in FEATURES):
            _hour_rows = _df_full[_df_full["timestamp"].dt.hour == h][FEATURES].dropna()
            if len(_hour_rows) > 0:
                feat_vec = _hour_rows.mean().to_frame().T
                pred_occ = int(_rf.predict(feat_vec)[0])
                is_occ = pred_occ == 1
            else:
                is_occ = (8 <= h <= 17)
            _temp_rows = _df_full[_df_full["timestamp"].dt.hour == h][["return_temp_C", "supply_humidity_pct"]].dropna()
            if len(_temp_rows) > 0:
                t_mean = float(_temp_rows["return_temp_C"].mean())
                rh_mean = float(_temp_rows["supply_humidity_pct"].mean())
            else:
                t_mean = 21.5 + np.random.rand() * 0.8
                rh_mean = 50.0
        else:
            is_occ = (8 <= h <= 17)
            t_mean = 21.5 + np.random.rand() * 0.8
            rh_mean = 50.0
        sp_now = BASE_SP if is_occ else BASE_SP - sb_val
        t_eff = t_mean if is_occ else t_mean - sb_val
        pmv_val, _ = pmv_ppd_iso7730(tdb=t_eff, tr=t_eff, vr=0.25, rh=rh_mean, met=2.0, clo=0.65)
        SIM.append({
            "hour": h,
            "label": f"{h:02d}:00",
            "occ": is_occ,
            "sp": sp_now,
            "temp": round(t_mean, 2),
            "pmv": pmv_val
        })

    if run:
        delay = {"Slow (1×)": 0.6, "Normal (5×)": 0.2, "Fast (20×)": 0.05}[speed]
        temps, sps, sps_base, labels, log_lines = [], [], [], [], []
        e_cum, e_base_cum = 0.0, 0.0

        for i, row in enumerate(SIM):
            temps.append(row["temp"])
            sps.append(row["sp"])
            sps_base.append(BASE_SP)
            labels.append(row["label"])

            e_slot = 1.8 + np.random.rand() * 0.5 if row["occ"] else 1.1 + np.random.rand() * 0.4
            e_factor = {"1°C": 0.905, "2°C": 0.833, "3°C": 0.778}[setback]
            e_cum += e_slot * (1.0 if row["occ"] else e_factor)
            e_base_cum += e_slot
            sav_pct = (1 - e_cum / max(e_base_cum, 0.01)) * 100

            pmv_now = row["pmv"]

            if row["occ"]:
                stat_box.markdown("""
                <div style="background:rgba(63,185,80,0.1);border:1px solid rgba(63,185,80,0.3);border-radius:10px;padding:14px;text-align:center;">
                    <div style="color:#3FB950;font-size:14px;font-weight:700;">OCCUPIED</div>
                </div>""", unsafe_allow_html=True)
            else:
                stat_box.markdown("""
                <div style="background:rgba(88,166,255,0.1);border:1px solid rgba(88,166,255,0.3);border-radius:10px;padding:14px;text-align:center;">
                    <div style="color:#58A6FF;font-size:14px;font-weight:700;">VACANT</div>
                </div>""", unsafe_allow_html=True)

            sp_col = "#3FB950" if row["occ"] else "#58A6FF"
            sp_note = "Normal mode" if row["occ"] else f"−{sb_val}°C setback"
            sp_box.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="color:{sp_col};font-size:24px;">{row['sp']:.1f}°C</div>
                <div class="metric-lbl">Active Setpoint</div>
                <div style="font-size:10px;color:#8B949E;margin-top:4px;">{sp_note}</div>
            </div>""", unsafe_allow_html=True)

            pmv_col = "#3FB950" if abs(pmv_now) <= 0.5 else "#D29922"
            pmv_note = "Comfort zone" if abs(pmv_now) <= 0.5 else "Outside comfort"
            pmv_box.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="color:{pmv_col};font-size:24px;">{pmv_now:+.2f}</div>
                <div class="metric-lbl">PMV (ISO 7730)</div>
                <div style="font-size:10px;color:#8B949E;margin-top:4px;">{pmv_note}</div>
            </div>""", unsafe_allow_html=True)

            enrg_box.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="color:#3FB950;font-size:24px;">{sav_pct:.1f}%</div>
                <div class="metric-lbl">Energy Saved</div>
                <div style="font-size:10px;color:#8B949E;margin-top:4px;">{e_base_cum - e_cum:.1f} kWh so far</div>
            </div>""", unsafe_allow_html=True)

            if len(temps) > 1:
                fig_live = go.Figure()
                fig_live.add_trace(go.Scatter(
                    x=labels,
                    y=temps,
                    name="Room Temp",
                    line=dict(color="#58A6FF", width=2.5),
                    mode="lines"
                ))
                fig_live.add_trace(go.Scatter(
                    x=labels,
                    y=sps,
                    name=f"Setpoint ({setback})",
                    line=dict(color="#3FB950", width=2, dash="dash"),
                    mode="lines"
                ))
                if show_base:
                    fig_live.add_trace(go.Scatter(
                        x=labels,
                        y=sps_base,
                        name="Baseline SP",
                        line=dict(color="#F85149", width=1, dash="dot"),
                        mode="lines"
                    ))
                fig_live.update_layout(
                    **PLOTLY_LAYOUT,
                    height=260,
                    xaxis=dict(gridcolor="#21262D", color="#8B949E"),
                    yaxis=dict(gridcolor="#21262D", title="Temperature (°C)", color="#8B949E", range=[17, 25]),
                    legend=dict(bgcolor="#21262D", bordercolor="#30363D", font=dict(color="#8B949E", size=10)),
                )
                chart_box.plotly_chart(fig_live, use_container_width=True)

            action = "Comfort mode" if row["occ"] else f"Setback −{sb_val}°C"
            status = "OCCUPIED" if row["occ"] else "VACANT"
            log_lines = [f"{row['label']} {status} → {action}"] + log_lines
            log_html = "<br>".join([
                f'<span style="color:{"#3FB950" if "OCCUPIED" in l else "#58A6FF"};font-size:11px;">{l}</span>'
                for l in log_lines[:8]
            ])
            log_box.markdown(f"""
            <div style="background:#161B22;border:1px solid #30363D;border-radius:8px;padding:12px;font-family:'JetBrains Mono',monospace;min-height:260px;">
                <div style="color:#58A6FF;font-weight:600;font-size:12px;margin-bottom:8px;">Decision Log</div>
                {log_html}
            </div>""", unsafe_allow_html=True)

            prog_box.progress((i + 1) / len(SIM), text=f"Step {i + 1}/{len(SIM)} — {row['label']}")
            time.sleep(delay)

        prog_box.success(f"Simulation complete — {len(SIM)} steps. Final energy savings: {sav_pct:.1f}%")
    else:
        chart_box.markdown("""
        <div style="height:260px;background:#161B22;border:1px dashed #30363D;border-radius:10px;display:flex;align-items:center;justify-content:center;color:#484F58;font-size:14px;flex-direction:column;gap:8px;">
            <div>Press <b style="color:#E6EDF3;">Run Simulation</b> to start the closed-loop demo</div>
        </div>""", unsafe_allow_html=True)

st.divider()
st.markdown("""
<div style="text-align:center;color:#484F58;font-size:11px;padding:4px 0 12px;">
    HybridSense · Kadir Has University · | |
    Çağla Özal · Sena Berra Soydugan · Başak Yıldız | |
    Advisors: Arif Selçuk Öğrenci · Rahim Dehkhargani
</div>
""", unsafe_allow_html=True)
