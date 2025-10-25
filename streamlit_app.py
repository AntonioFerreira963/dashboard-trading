# streamlit_app.py — V8.0.2 (Style Excel, multi-timeframes RSI/MACD, correctifs)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timezone

st.set_page_config(page_title="Robot Trading — V8", layout="wide", page_icon="📊")

# ================= THEME =================
st.markdown("""
<style>
:root{
  --bg:#0b0f1a; --panel:#111a2e; --grid:#243557; --txt:#f1f5f9; --mut:#94a3b8;
  --buy:#10c77b; --sell:#ff4d4f; --wait:#a0aec0; --head:#1a2746; --kpi:#0f1629;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--txt)!important;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1428,#0b0f1a);}
.block-container{padding-top:1rem;padding-bottom:2rem;}
h1,h2,h3{color:#e6edf3}
.card{background:var(--panel);border:1px solid var(--grid);border-radius:14px;overflow:hidden}
.card-header{background:var(--head);padding:10px 14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.tag{padding:.15rem .55rem;border-radius:999px;font-weight:700;border:1px solid rgba(255,255,255,.15)}
.tag-buy{color:var(--buy);background:rgba(16,199,123,.14)}
.tag-sell{color:var(--sell);background:rgba(255,77,79,.14)}
.tag-wait{color:var(--wait);background:rgba(160,174,192,.14)}
.grid{display:grid;grid-template-columns:1.05fr .9fr 1.2fr;gap:12px;padding:14px}
.box{background:var(--kpi);border:1px solid var(--grid);border-radius:12px;padding:10px}
.small{color:var(--mut);font-size:.88rem}
.tbl td,.tbl th{border:1px solid var(--grid);padding:6px 10px}
.tbl{border-collapse:collapse;width:100%;font-size:.92rem}
.th-center th{text-align:center}
.center{text-align:center}
.kpi-val{font-weight:800;font-size:1.3rem}
.sep{height:10px}
</style>
""", unsafe_allow_html=True)

st.title("📊 Robot Trading — V8 (Achat / Vente / Attente par timeframe)")

# ================= SIDEBAR =================
with st.sidebar:
    st.subheader("⚙️ Paramètres")
    default_watchlist = ["GC=F","NVDA","BABA","QQQ","IREN","BYND"]
    selection = st.multiselect("Watchlist (Yahoo)", sorted(set(default_watchlist)),
                               default=default_watchlist)
    new_tkr = st.text_input("➕ Ajouter un ticker (Yahoo)")
    if new_tkr:
        selection = list(dict.fromkeys(selection + [new_tkr.strip().upper()]))

    spark_days = st.slider("Historique sparkline (jours)", 5, 90, 30)
    st.caption("Conseil : 30 jours = bonne lisibilité.")

# ================= HELPERS =================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, f=12, s=26) -> pd.Series:
    ema_f = series.ewm(span=f, adjust=False).mean()
    ema_s = series.ewm(span=s, adjust=False).mean()
    return ema_f - ema_s

def pick_period_interval(tf: str):
    # Yahoo n'a pas toujours 240m → fallback sur 60m pour 4H
    if tf == "1m":   return ("1d", "1m")
    if tf == "15m":  return ("7d", "15m")
    if tf == "1H":   return ("30d", "60m")
    if tf == "4H":   return ("60d", "60m")   # compatible
    if tf == "1D":   return ("1y", "1d")
    raise ValueError("tf inconnu")

@st.cache_data(ttl=180, show_spinner=False)
def load_tf(ticker: str, tf: str) -> pd.DataFrame:
    period, interval = pick_period_interval(tf)
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, prepost=True, actions=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.title)
        if "Close" not in df.columns: return pd.DataFrame()
        out = df[["Open","High","Low","Close","Volume"]].copy()
        out["RSI"] = rsi(out["Close"]).round(1)
        out["MACD"] = macd(out["Close"]).round(6)
        return out.dropna()
    except Exception:
        return pd.DataFrame()

def classify_signal(df: pd.DataFrame) -> str:
    """Renvoie 'ACHAT' / 'VENTE' / 'ATTENTE' selon RSI & MACD (croisement)."""
    if df.empty or len(df) < 3: return "ATTENTE"
    rsi_now = float(df["RSI"].iloc[-1])
    macd_now, macd_prev = float(df["MACD"].iloc[-1]), float(df["MACD"].iloc[-2])
    cross_up = (macd_prev < 0 and macd_now > 0)
    cross_down = (macd_prev > 0 and macd_now < 0)

    if rsi_now < 30 or cross_up:
        return "ACHAT"
    if rsi_now > 70 or cross_down:
        return "VENTE"
    return "ATTENTE"

def trend_label(pct: float) -> str:
    if pct > 0.2:  return "Haussière"
    if pct < -0.2: return "Baissière"
    return "Neutre"

def tag_class(label: str) -> str:
    return {"ACHAT":"tag-buy","VENTE":"tag-sell","ATTENTE":"tag-wait"}.get(label,"tag-wait")

@st.cache_data(ttl=180, show_spinner=False)
def load_spark(ticker: str, days: int) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    if days <= 7:   df = tk.history(period="7d", interval="30m", prepost=True)
    elif days <= 30:df = tk.history(period="30d", interval="1d", prepost=True)
    else:           df = tk.history(period="90d", interval="1d", prepost=True)
    if df is None or df.empty: return pd.DataFrame()
    return df.rename(columns=str.title)[["Close"]].dropna().reset_index()

def expert_opinion(signals: dict) -> str:
    # Pondération forte 4H/1D
    score = 0
    weight = {"1m":1, "15m":1, "1H":2, "4H":3, "1D":4}
    for tf, sig in signals.items():
        if sig == "ACHAT":  score += weight.get(tf,1)
        elif sig == "VENTE":score -= weight.get(tf,1)
    if score >= 3:  return "BUY"
    if score <= -3: return "SELL"
    return "NEUTRE"

def forecast_text(signals: dict) -> str:
    major = [signals.get("4H","ATTENTE"), signals.get("1D","ATTENTE")]
    if major.count("ACHAT") >= 1 and "VENTE" not in major: return "Haussière"
    if major.count("VENTE") >= 1 and "ACHAT" not in major: return "Baissière"
    return "Neutre"

# ================= RENDU PAR ACTIF =================
def render_asset(ticker: str, spark_days: int):
    # Snapshot 1D
    day_df = load_tf(ticker, "1D")
    if day_df.empty or len(day_df) < 2:
        st.warning(f"{ticker} : données indisponibles")
        return
    price = float(day_df["Close"].iloc[-1])
    prev = float(day_df["Close"].iloc[-2])
    pct = round(((price - prev) / prev * 100), 2) if prev else 0.0

    # Signals par timeframe
    tfs = ["1m","15m","1H","4H","1D"]
    sigs = {}
    for tf in tfs:
        df_tf = load_tf(ticker, tf)
        sigs[tf] = classify_signal(df_tf)

    # Bannières
    tendance = trend_label(pct)
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    # ---- mapping robuste (corrige le bug KeyError NEUTRE)
    expert_raw = expert_opinion(sigs)          # "BUY" / "SELL" / "NEUTRE"
    mapped = {"BUY": "ACHAT", "SELL": "VENTE", "NEUTRE": "ATTENTE"}.get(expert_raw, "ATTENTE")
    main_tag = {"ACHAT": "Achat", "VENTE": "Vente", "ATTENTE": "Attente"}[mapped]

    # HEADER
    st.markdown(
        f"""
        <div class="card">
          <div class="card-header">
            <div style="font-size:1.05rem;font-weight:800">{ticker}</div>
            <span class="tag">{tendance}</span>
            <div class="small">{now}</div>
            <span class="tag {tag_class(sigs['15m'])}">15m : {sigs['15m']}</span>
            <span class="tag {tag_class(sigs['1D'])}">1D : {sigs['1D']}</span>
            <span class="tag {tag_class(mapped)}">Avis global : {main_tag}</span>
          </div>
          <div class="grid">
        """,
        unsafe_allow_html=True
    )

    # COL 1 — Prix & sparkline
    col1, col2, col3 = st.columns([1.05, .9, 1.2])
    with col1:
        st.markdown('<div class="box">', unsafe_allow_html=True)
        st.markdown(f"<div class='small'>Cours actuel</div><div class='kpi-val'>{price:.2f}</div>", unsafe_allow_html=True)
        colA, colB = st.columns(2)
        with colA:
            st.markdown("<div class='small'>Variation (1D)</div>", unsafe_allow_html=True)
            col_color = "var(--buy)" if pct>=0 else "var(--sell)"
            st.markdown(f"<div style='color:{col_color};font-weight:700'>{pct:.2f}%</div>", unsafe_allow_html=True)
        with colB:
            st.markdown("<div class='small'>Tendance</div>", unsafe_allow_html=True)
            st.markdown(f"<div>{tendance}</div>", unsafe_allow_html=True)

        spark = load_spark(ticker, spark_days)
        if not spark.empty:
            ch = alt.Chart(spark).mark_line().encode(
                x=alt.X('Date:T', axis=alt.Axis(labels=False, ticks=False, title=None)),
                y=alt.Y('Close:Q', axis=alt.Axis(labels=False, ticks=False, title=None))
            ).properties(height=90)
            st.altair_chart(ch, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # COL 2 — Tableau multi-timeframes
    with col2:
        st.markdown('<div class="box">', unsafe_allow_html=True)
        st.markdown("<div class='small' style='margin-bottom:6px'>ACHAT / VENTE / ATTENTE</div>", unsafe_allow_html=True)
        tdata = pd.DataFrame({
            "Timeframe": ["1m","15m","1H","4H","1D"],
            "Signal": [sigs["1m"],sigs["15m"],sigs["1H"],sigs["4H"],sigs["1D"]],
        })
        def color_map(v):
            if v=="ACHAT": return f"<span style='color:var(--buy);font-weight:700'>{v}</span>"
            if v=="VENTE": return f"<span style='color:var(--sell);font-weight:700'>{v}</span>"
            return f"<span style='color:var(--wait);font-weight:700'>{v}</span>"
        tdata["Signal"] = tdata["Signal"].map(color_map)
        html = (
            "<table class='tbl th-center'>"
            "<tr><th>TF</th><th>Signal</th></tr>"
            + "".join([f"<tr class='center'><td>{row.Timeframe}</td><td>{row.Signal}</td></tr>"
                       for row in tdata.itertuples()])
            + "</table>"
        )
        st.markdown(html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # COL 3 — Prévision & Avis experts
    with col3:
        st.markdown('<div class="box">', unsafe_allow_html=True)
        prev_txt = forecast_text(sigs)
        avis = expert_opinion(sigs)  # BUY/SELL/NEUTRE
        colC1, colC2 = st.columns(2)
        with colC1:
            st.markdown("<div class='small'>Prévision (4H & 1D)</div>", unsafe_allow_html=True)
            color = "var(--buy)" if prev_txt=="Haussière" else "var(--sell)" if prev_txt=="Baissière" else "var(--wait)"
            st.markdown(f"<div style='font-weight:800;color:{color}'>{prev_txt}</div>", unsafe_allow_html=True)
        with colC2:
            st.markdown("<div class='small'>Avis experts</div>", unsafe_allow_html=True)
            avis_disp = {"BUY":"BUY","SELL":"SELL","NEUTRE":"NEUTRE"}[avis]
            color = "var(--buy)" if avis_disp=="BUY" else "var(--sell)" if avis_disp=="SELL" else "var(--wait)"
            st.markdown(f"<div style='font-weight:800;color:{color}'>{avis_disp}</div>", unsafe_allow_html=True)

        st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
        st.markdown("<div class='small'>Logique</div>", unsafe_allow_html=True)
        st.markdown(
            "- Signals calculés par **RSI(14)** + **MACD(12-26)**  \n"
            "- Croisement MACD et zones RSI (<30 Achat / >70 Vente)  \n"
            "- **Avis experts** = pondération (1D & 4H > 1H > 15m > 1m)",
            unsafe_allow_html=False
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# ================= PAGE =================
tickers = selection or default_watchlist
for t in tickers:
    render_asset(t, spark_days)
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

st.caption("⚠️ Outil éducatif. Les signaux sont indicatifs (RSI/MACD).")
