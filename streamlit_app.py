# streamlit_app.py — V7 (Style TradingTop, couleurs auto, TP1/TP2/SL serré)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# =========================
#   CONFIG & THEME
# =========================
st.set_page_config(page_title="Robot Trading — V7", layout="wide", page_icon="🤖")

# --- CSS minimal pour un style 'TradingTop' propre
st.markdown("""
<style>
:root{
  --bg:#0b0f1a; --panel:#121829; --muted:#94a3b8;
  --buy:#12d27c; --sell:#ff4d4f; --plus:#8b5cf6; --minus:#eab308; --wait:#64748b;
}
html,body,[class^="st"],[data-testid="stAppViewContainer"]{background:var(--bg)!important;}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#0d1220,#0b0f1a);}
.block-container{padding-top:1.2rem;padding-bottom:2rem;}
h1,h2,h3,h4{color:#e2e8f0;}
.small{font-size:0.86rem;color:var(--muted)}
.card{border-radius:16px;padding:16px;background:var(--panel);border:1px solid rgba(255,255,255,.06)}
.badge{display:inline-block;padding:.25rem .6rem;border-radius:999px;font-weight:700}
.badge-buy{background:rgba(18,210,124,.15);color:var(--buy);border:1px solid rgba(18,210,124,.35)}
.badge-sell{background:rgba(255,77,79,.15);color:var(--sell);border:1px solid rgba(255,77,79,.35)}
.badge-plus{background:rgba(139,92,246,.15);color:var(--plus);border:1px solid rgba(139,92,246,.35)}
.badge-minus{background:rgba(234,179,8,.15);color:var(--minus);border:1px solid rgba(234,179,8,.35)}
.badge-wait{background:rgba(100,116,139,.15);color:var(--wait);border:1px solid rgba(100,116,139,.35)}
.kpi{background:var(--panel);border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:10px 12px;text-align:center}
.kpi .v{font-size:1.35rem;font-weight:800;color:#e5e7eb}
.kpi .l{font-size:.8rem;color:var(--muted)}
table{color:#e2e8f0}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Robot Trading — RSI / MACD / EMA (V7)")

with st.expander("ℹ️ Règles des signaux (rapide)", expanded=False):
    st.markdown("""
**ACHAT** : RSI < 30 **ou** MACD croise haussier (négatif → positif).  
**SHORT** : RSI > 70 **ou** MACD croise baissier (positif → négatif).  
**CASSURE+** : Prix > EMA20 > EMA50 **ET** passage au-dessus de l’EMA20 aujourd’hui.  
**CASSURE-** : Prix < EMA20 < EMA50 **ET** passage en-dessous de l’EMA20 aujourd’hui.  
**TP1 / TP2 / SL** : basés sur l’ATR(14) (TP1 ≈ 0.75×ATR, TP2 ≈ 1.5×ATR, SL serré ≈ 0.8×ATR).
""")

# =========================
#   SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Paramètres")
    default_watchlist = ["GC=F","NVDA","BABA","QQQ","IREN","BYND"]
    selection = st.multiselect(
        "Watchlist (Yahoo)", 
        options=sorted(set(default_watchlist)),
        default=default_watchlist,
        help="Ajoute librement d'autres tickers Yahoo Finance."
    )
    new_tkr = st.text_input("➕ Ajouter un ticker (Yahoo)")
    if new_tkr:
        selection = list(dict.fromkeys(selection + [new_tkr.strip().upper()]))

    days = st.slider("Historique (jours)", 5, 365, 30)
    view_mode = st.radio("Affichage", ["Vue Robot", "Vue Cartes"], horizontal=True)
    show_charts = st.checkbox("Mini-graph sous chaque carte", value=True)
    st.caption("Astuce : 7–30 jours = réactivité optimale.")

# =========================
#   HELPERS
# =========================
def _pick_period_interval(days:int):
    if days <= 7:   return ("7d", "15m")
    if days <= 30:  return (f"{days}d", "30m")
    if days <= 90:  return (f"{days}d", "60m")
    return (f"{days}d", "1d")

@st.cache_data(ttl=180, show_spinner=False)
def load_ohlcv(ticker:str, period_days:int) -> pd.DataFrame:
    period, interval = _pick_period_interval(period_days)
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval, prepost=True, actions=False, auto_adjust=False)
    if df is None or df.empty:
        df = tk.history(period=f"{period_days}d", interval="1d", prepost=True, actions=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    # Conserver uniquement OHLCV + Drop NA
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    return df[keep].dropna()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()

    out = df.copy()
    # RSI(14)
    delta = out["Close"].diff()
    gain = np.where(delta>0, delta, 0.0)
    loss = np.where(delta<0, -delta, 0.0)
    roll_up = pd.Series(gain, index=out.index).rolling(14).mean()
    roll_down = pd.Series(loss, index=out.index).rolling(14).mean()
    rs = roll_up / roll_down
    out["RSI"] = 100 - (100/(1+rs))

    # EMA20 / EMA50
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()

    # MACD (12-26)
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26

    # ATR(14)
    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift()).abs()
    low_close  = (out["Low"]  - out["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()

    return out

def _cross_today(prev_val, prev_ref, now_val, now_ref, up=True):
    if any(pd.isna(x) for x in [prev_val, prev_ref, now_val, now_ref]): return False
    return (prev_val < prev_ref and now_val > now_ref) if up else (prev_val > prev_ref and now_val < now_ref)

def classify_signal(df: pd.DataFrame):
    row_now, row_prev = df.iloc[-1], df.iloc[-2]
    price = float(row_now["Close"])
    rsi   = float(row_now["RSI"])
    macd_now, macd_prev = float(row_now["MACD"]), float(row_prev["MACD"])
    ema20_now, ema50_now = float(row_now["EMA20"]), float(row_now["EMA50"])
    cross_up   = (macd_prev < 0 and macd_now > 0)
    cross_down = (macd_prev > 0 and macd_now < 0)
    cross_above = _cross_today(row_prev["Close"], row_prev["EMA20"], row_now["Close"], row_now["EMA20"], up=True)
    cross_below = _cross_today(row_prev["Close"], row_prev["EMA20"], row_now["Close"], row_now["EMA20"], up=False)

    # ordre de priorité — Achat / Short > Cassures > Attente
    if (rsi < 30) or cross_up:
        label = "ACHAT"; reason = f"RSI {rsi:.1f} ou MACD↑"
    elif (rsi > 70) or cross_down:
        label = "SHORT"; reason = f"RSI {rsi:.1f} ou MACD↓"
    elif (price > ema20_now > ema50_now) and cross_above:
        label = "CASSURE+"; reason = "Prix>EMA20>EMA50 (cassure EMA20↑)"
    elif (price < ema20_now < ema50_now) and cross_below:
        label = "CASSURE-"; reason = "Prix<EMA20<EMA50 (cassure EMA20↓)"
    else:
        label = "ATTENTE"; reason = "Neutre"
    return label, reason, price

def tp_sl_from_atr(side:str, price:float, atr:float):
    # SL serré ≈ 0.8 ATR ; TP1 0.75 ATR ; TP2 1.5 ATR
    if np.isnan(atr) or atr <= 0:
        step = max(price*0.01, 0.5)  # fallback
        atr = step
    if side == "SHORT":
        tp1 = price - 0.75*atr
        tp2 = price - 1.50*atr
        sl  = price + 0.80*atr
    else: # ACHAT / Cassures
        tp1 = price + 0.75*atr
        tp2 = price + 1.50*atr
        sl  = price - 0.80*atr
    return round(tp1,2), round(tp2,2), round(sl,2)

def color_classes(label:str):
    return {
        "ACHAT": ("badge-buy","✅ Achat","--buy"),
        "SHORT": ("badge-sell","🚨 Short","--sell"),
        "CASSURE+": ("badge-plus","📈 Cassure+","--plus"),
        "CASSURE-": ("badge-minus","📉 Cassure-","--minus"),
        "ATTENTE": ("badge-wait","➖ Attente","--wait"),
    }[label]

# =========================
#   SCAN
# =========================
tickers = selection if selection else default_watchlist
rows = []

for t in tickers:
    df = load_ohlcv(t, days)
    if df.empty or len(df) < 50:
        rows.append({"Ticker": t, "Prix": None, "RSI": None, "Signal":"ATTENTE", "Raison":"Pas assez de données",
                     "TP1":None, "TP2":None, "SL":None, "ATR":None, "Data":df})
        continue
    dfi = add_indicators(df).dropna()
    label, reason, price = classify_signal(dfi)
    atr = float(dfi["ATR"].iloc[-1])
    side_for_tp = "SHORT" if label=="SHORT" else "ACHAT"
    tp1, tp2, sl = tp_sl_from_atr(side_for_tp, price, atr)
    rows.append({
        "Ticker": t, "Prix": round(price,2),
        "RSI": round(float(dfi["RSI"].iloc[-1]),1),
        "Signal": label, "Raison": reason,
        "TP1": tp1, "TP2": tp2, "SL": sl, "ATR": round(atr,2),
        "Data": dfi
    })

df_all = pd.DataFrame(rows)

# =========================
#   HEADER DIRECTION GLOBALE
# =========================
nb_buy  = int((df_all["Signal"]=="ACHAT").sum())
nb_sell = int((df_all["Signal"]=="SHORT").sum())
direction = "BUY" if nb_buy>nb_sell else "SELL" if nb_sell>nb_buy else "NEUTRE"
badge_cls = "badge-buy" if direction=="BUY" else "badge-sell" if direction=="SELL" else "badge-wait"

colA,colB,colC,colD,colE = st.columns([1.2,1,1,1,1.2])
with colA:
    st.markdown(f'<div class="card"><div class="{badge_cls}">📊 Direction : <b>{direction}</b></div>'
                f'<div class="small">Mise à jour : {datetime.utcnow().strftime("%H:%M UTC (%d-%m)")} </div></div>', unsafe_allow_html=True)
with colB: st.markdown(f'<div class="kpi"><div class="v">{len(df_all)}</div><div class="l">Tickers</div></div>', unsafe_allow_html=True)
with colC: st.markdown(f'<div class="kpi"><div class="v">{nb_buy}</div><div class="l">Achat</div></div>', unsafe_allow_html=True)
with colD: st.markdown(f'<div class="kpi"><div class="v">{nb_sell}</div><div class="l">Short</div></div>', unsafe_allow_html=True)
with colE:
    nb_plus = int((df_all["Signal"]=="CASSURE+").sum())
    nb_minus= int((df_all["Signal"]=="CASSURE-").sum())
    st.markdown(f'<div class="kpi"><div class="v">{nb_plus}/{nb_minus}</div><div class="l">Cassure + / -</div></div>', unsafe_allow_html=True)

st.divider()

# =========================
#   AFFICHAGE
# =========================
if view_mode == "Vue Robot":
    st.subheader("📋 Signaux — Vue Robot")

    def show_bucket(title:str, label:str):
        b = df_all[df_all["Signal"]==label][["Ticker","Prix","RSI","TP1","TP2","SL","Raison"]]
        cls, text, _ = color_classes(label)
        st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)
        if b.empty:
            st.write("—")
        else:
            st.dataframe(b.reset_index(drop=True), use_container_width=True, hide_index=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: show_bucket("Achat","ACHAT")
    with c2: show_bucket("Short","SHORT")
    with c3: show_bucket("Cassure+","CASSURE+")
    with c4: show_bucket("Cassure-","CASSURE-")

    with st.expander("👀 Voir la liste complète (inclut ATTENTE)"):
        st.dataframe(df_all.drop(columns=["Data"]).reset_index(drop=True), use_container_width=True, hide_index=True)

else:
    st.subheader("🧾 Vue Cartes (par action)")
    for _, r in df_all.iterrows():
        cls, text, var = color_classes(r["Signal"])
        st.markdown(f'<div class="card">'
                    f'<div class="{cls}" style="margin-bottom:.5rem">{text} — <b>{r["Ticker"]}</b></div>'
                    f'<div class="small">Prix: <b>{r["Prix"]}</b> • RSI(14): <b>{r["RSI"]}</b> • ATR: <b>{r["ATR"]}</b></div>'
                    f'<div class="small">🎯 TP1: <b>{r["TP1"]}</b> • TP2: <b>{r["TP2"]}</b> • 🛑 SL: <b>{r["SL"]}</b></div>'
                    f'<div class="small">📝 {r["Raison"]}</div>'
                    f'</div>', unsafe_allow_html=True)

        if show_charts and isinstance(r["Data"], pd.DataFrame) and not r["Data"].empty:
            d = r["Data"].reset_index().rename(columns={"index":"Date"})
            chart = alt.Chart(d).mark_line().encode(
                x=alt.X('Date:T', axis=alt.Axis(labels=False, ticks=False, title=None)),
                y=alt.Y('Close:Q', axis=alt.Axis(title=None))
            ).properties(width='container', height=110)
            st.altair_chart(chart, use_container_width=True)
        st.write("")

st.caption("⚠️ Outil éducatif. Rien ici n’est un conseil d’investissement. Les niveaux TP/SL sont indicatifs (ATR).")
