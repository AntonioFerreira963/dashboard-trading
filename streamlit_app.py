# streamlit_app.py — V7.2 (Lisibilité ++, contrastes forts, cartes propres)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# =========================
#   CONFIG & THEME
# =========================
st.set_page_config(page_title="Robot Trading — V7.2", layout="wide", page_icon="🤖")

# ---- UI Controls (avant CSS pour agir sur la taille globale)
with st.sidebar:
    st.header("⚙️ Paramètres")
    ui_scale = st.slider("Taille de l’interface", 90, 120, 100, help="Zoom de l’UI (en %)")
    compact = st.toggle("Mode compact (tables denses)", value=True)
    st.divider()

# ---- CSS haute lisibilité
st.markdown(f"""
<style>
html {{ zoom: {ui_scale/100}; }}
:root {{
  --bg:#0b0f1a;        /* fond app */
  --panel:#0f1629;     /* panneaux */
  --panel2:#131c34;    /* surcouche */
  --txt:#e6edf3;       /* texte clair */
  --muted:#a2b0c6;     /* texte secondaire */
  --grid:#23304f;      /* bordures */
  --buy:#10c77b;       /* vert achat */
  --sell:#ff4d4f;      /* rouge short */
  --plus:#8b5cf6;      /* violet cassure+ */
  --minus:#eab308;     /* jaune cassure- */
  --wait:#94a3b8;      /* gris attente */
}}
html, body, [data-testid="stAppViewContainer"] {{ background: var(--bg) !important; color: var(--txt); }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg,#0b1222,#0b0f1a); color: var(--txt); }}
.block-container{{padding-top:1.0rem; padding-bottom:2rem; }}
h1,h2,h3,h4{{ color: var(--txt); }}
.small {{ color: var(--muted); font-size:.88rem }}
.card {{
  background: var(--panel);
  border:1px solid var(--grid);
  border-radius:16px; padding:14px 16px;
  box-shadow: 0 6px 20px rgba(0,0,0,.35);
}}
.kpi {{
  background: var(--panel2);
  border:1px solid var(--grid);
  border-radius:14px; padding:10px 12px; text-align:center;
}}
.kpi .v{{ font-size:1.35rem; font-weight:800; color:#fff; }}
.kpi .l{{ font-size:.82rem; color:var(--muted); }}
.head {{
  padding:10px 12px; color:#fff; font-weight:700; border-radius:12px 12px 0 0;
}}
.head-buy  {{ background: linear-gradient(90deg, rgba(16,199,123,.95), rgba(16,199,123,.45)); }}
.head-sell {{ background: linear-gradient(90deg, rgba(255,77,79,.95), rgba(255,77,79,.45)); }}
.head-plus {{ background: linear-gradient(90deg, rgba(139,92,246,.95), rgba(139,92,246,.45)); }}
.head-minus{{ background: linear-gradient(90deg, rgba(234,179,8,.95), rgba(234,179,8,.45)); }}
.tbl {{ border-left:1px solid var(--grid); border-right:1px solid var(--grid); border-bottom:1px solid var(--grid);
        border-radius:0 0 12px 12px; padding:6px; background: #0f1527; }}
.badge {{
  display:inline-block; padding:.28rem .6rem; border-radius:999px; font-weight:700; border:1px solid rgba(255,255,255,.18)
}}
.badge-buy  {{ background:rgba(16,199,123,.18); color:var(--buy);  }}
.badge-sell {{ background:rgba(255,77,79,.18); color:var(--sell); }}
.badge-plus {{ background:rgba(139,92,246,.18); color:var(--plus); }}
.badge-minus{{ background:rgba(234,179,8,.18);  color:var(--minus);}}
.badge-wait {{ background:rgba(148,163,184,.18); color:var(--wait); }}
/* Densité tableau */
[data-testid="stDataFrame"] div[role="row"] div[role="gridcell"] {{
  padding: {'.25rem .35rem' if compact else '.5rem .6rem'} !important;
}}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Robot Trading — RSI / MACD / EMA (V7.2)")

with st.expander("ℹ️ Règles des signaux (rapide)", expanded=False):
    st.markdown("""
**ACHAT** : RSI < 30 **ou** MACD croise haussier.  
**SHORT** : RSI > 70 **ou** MACD croise baissier.  
**CASSURE+** : Prix > EMA20 > EMA50 **ET** franchit EMA20 aujourd’hui.  
**CASSURE-** : Prix < EMA20 < EMA50 **ET** casse EMA20 aujourd’hui.  
**TP1/TP2/SL** : basés sur **ATR(14)** → TP1 ≈ 0.75×ATR, TP2 ≈ 1.5×ATR, SL serré ≈ 0.8×ATR.
""")

# =========================
#   SIDEBAR (suite)
# =========================
with st.sidebar:
    default_watchlist = ["GC=F","NVDA","BABA","QQQ","IREN","BYND"]
    selection = st.multiselect("Watchlist (Yahoo)", sorted(set(default_watchlist)), default_watchlist)
    new_tkr = st.text_input("➕ Ajouter un ticker (Yahoo)")
    if new_tkr:
        selection = list(dict.fromkeys(selection + [new_tkr.strip().upper()]))

    days = st.slider("Historique (jours)", 5, 365, 30)
    view_mode = st.radio("Affichage", ["Vue Robot", "Vue Cartes"], horizontal=True)
    show_charts = st.checkbox("Mini-graph sous chaque carte", value=True)
    st.caption("Astuce : 7–30 jours = réactif.")

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
    if df is None or df.empty: return pd.DataFrame()
    df = df.rename(columns=str.title)
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

    # EMA
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()

    # MACD
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26

    # ATR(14)
    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift()).abs()
    low_close  = (out["Low"]  - out["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()
    return out.dropna()

def _cross_today(prev_val, prev_ref, now_val, now_ref, up=True):
    if any(pd.isna(x) for x in [prev_val, prev_ref, now_val, now_ref]): return False
    return (prev_val < prev_ref and now_val > now_ref) if up else (prev_val > prev_ref and now_val < now_ref)

def classify_signal(dfi: pd.DataFrame):
    row_now, row_prev = dfi.iloc[-1], dfi.iloc[-2]
    p = float(row_now["Close"])
    rsi = float(row_now["RSI"])
    macd_now, macd_prev = float(row_now["MACD"]), float(row_prev["MACD"])
    ema20_now, ema50_now = float(row_now["EMA20"]), float(row_now["EMA50"])
    cross_up   = (macd_prev < 0 and macd_now > 0)
    cross_down = (macd_prev > 0 and macd_now < 0)
    cross_above= _cross_today(row_prev["Close"], row_prev["EMA20"], row_now["Close"], row_now["EMA20"], True)
    cross_below= _cross_today(row_prev["Close"], row_prev["EMA20"], row_now["Close"], row_now["EMA20"], False)

    if (rsi < 30) or cross_up:
        return "ACHAT", f"RSI {rsi:.1f} ou MACD↑", p
    if (rsi > 70) or cross_down:
        return "SHORT", f"RSI {rsi:.1f} ou MACD↓", p
    if (p > ema20_now > ema50_now) and cross_above:
        return "CASSURE+", "Prix>EMA20>EMA50 (cassure EMA20↑)", p
    if (p < ema20_now < ema50_now) and cross_below:
        return "CASSURE-", "Prix<EMA20<EMA50 (cassure EMA20↓)", p
    return "ATTENTE", "Neutre", p

def tp_sl_from_atr(side:str, price:float, atr:float):
    if np.isnan(atr) or atr <= 0: atr = max(price*0.008, 0.3)
    mult = {"TP1":0.75, "TP2":1.50, "SL":0.80}
    if side == "SHORT":
        tp1, tp2, sl = price - mult["TP1"]*atr, price - mult["TP2"]*atr, price + mult["SL"]*atr
    else:
        tp1, tp2, sl = price + mult["TP1"]*atr, price + mult["TP2"]*atr, price - mult["SL"]*atr
    return round(tp1,2), round(tp2,2), round(sl,2)

def color_meta(label:str):
    meta = {
        "ACHAT": ("Achat","head-buy","badge-buy"),
        "SHORT": ("Short","head-sell","badge-sell"),
        "CASSURE+": ("Cassure+","head-plus","badge-plus"),
        "CASSURE-": ("Cassure-","head-minus","badge-minus"),
        "ATTENTE": ("Attente","head-wait","badge-wait")
    }
    return meta[label]

# =========================
#   SCAN
# =========================
tickers = selection or ["GC=F","NVDA","BABA","QQQ","IREN","BYND"]
rows = []
for t in tickers:
    df = load_ohlcv(t, days)
    if df.empty or len(df) < 50:
        rows.append({"Ticker":t,"Prix":None,"RSI":None,"Signal":"ATTENTE","Raison":"Pas assez de données",
                     "TP1":None,"TP2":None,"SL":None,"ATR":None,"Data":df})
        continue
    dfi = add_indicators(df)
    label, reason, price = classify_signal(dfi)
    atr = float(dfi["ATR"].iloc[-1])
    side = "SHORT" if label=="SHORT" else "ACHAT"
    tp1, tp2, sl = tp_sl_from_atr(side, price, atr)
    rows.append({"Ticker":t,"Prix":round(price,2),"RSI":round(float(dfi['RSI'].iloc[-1]),1),
                 "Signal":label,"Raison":reason,"TP1":tp1,"TP2":tp2,"SL":sl,"ATR":round(atr,2),"Data":dfi})

df_all = pd.DataFrame(rows)

# =========================
#   HEADER DIRECTION
# =========================
nb_buy  = int((df_all["Signal"]=="ACHAT").sum())
nb_sell = int((df_all["Signal"]=="SHORT").sum())
direction = "BUY" if nb_buy>nb_sell else "SELL" if nb_sell>nb_buy else "NEUTRE"
badge = "badge-buy" if direction=="BUY" else "badge-sell" if direction=="SELL" else "badge-wait"

cA,cB,cC,cD,cE = st.columns([1.4,1,1,1,1.2])
with cA:
    st.markdown(f"""
    <div class="card">
      <span class="{badge}">📊 Direction : <b>{direction}</b></span>
      <div class="small">Mise à jour : {datetime.utcnow().strftime("%H:%M UTC (%d-%m)")}</div>
    </div>""", unsafe_allow_html=True)
with cB: st.markdown(f'<div class="kpi"><div class="v">{len(df_all)}</div><div class="l">Tickers</div></div>', unsafe_allow_html=True)
with cC: st.markdown(f'<div class="kpi"><div class="v">{nb_buy}</div><div class="l">Achat</div></div>', unsafe_allow_html=True)
with cD: st.markdown(f'<div class="kpi"><div class="v">{nb_sell}</div><div class="l">Short</div></div>', unsafe_allow_html=True)
with cE:
    nb_plus = int((df_all["Signal"]=="CASSURE+").sum())
    nb_minus= int((df_all["Signal"]=="CASSURE-").sum())
    st.markdown(f'<div class="kpi"><div class="v">{nb_plus}/{nb_minus}</div><div class="l">Cassure + / -</div></div>', unsafe_allow_html=True)

st.markdown("### 🗂️ Signaux — Vue Robot" if view_mode=="Vue Robot" else "### 🗂️ Vue Cartes (par action)")

# =========================
#   AFFICHAGE
# =========================
def table_for(label:str):
    sub = df_all[df_all["Signal"]==label][["Ticker","Prix","RSI","TP1","TP2","SL","Raison","ATR"]]
    if sub.empty:
        st.write("—"); return
    st.dataframe(
        sub.reset_index(drop=True),
        use_container_width=True, hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Prix":   st.column_config.NumberColumn("Prix", format="%.2f", width="small"),
            "RSI":    st.column_config.NumberColumn("RSI", format="%.1f", width="small"),
            "TP1":    st.column_config.NumberColumn("TP1", format="%.2f", width="small"),
            "TP2":    st.column_config.NumberColumn("TP2", format="%.2f", width="small"),
            "SL":     st.column_config.NumberColumn("SL",  format="%.2f", width="small"),
            "ATR":    st.column_config.NumberColumn("ATR", format="%.2f", width="small"),
            "Raison": st.column_config.TextColumn("Raison", width="medium"),
        }
    )

if view_mode == "Vue Robot":
    c1,c2,c3,c4 = st.columns(4)

    def bucket(col, title, head_cls, label):
        with col:
            st.markdown(f'<div class="head {head_cls}">{title}</div>', unsafe_allow_html=True)
            st.markdown('<div class="tbl">', unsafe_allow_html=True)
            table_for(label)
            st.markdown('</div>', unsafe_allow_html=True)

    bucket(c1, "✅ Achat",   "head-buy",  "ACHAT")
    bucket(c2, "🚨 Short",   "head-sell", "SHORT")
    bucket(c3, "📈 Cassure+", "head-plus", "CASSURE+")
    bucket(c4, "📉 Cassure-", "head-minus","CASSURE-")

    with st.expander("👀 Liste complète (inclut ATTENTE)"):
        st.dataframe(df_all.drop(columns=["Data"]).reset_index(drop=True),
                     use_container_width=True, hide_index=True)

else:
    # Cartes individuelles très lisibles
    for _, r in df_all.iterrows():
        title, head_cls, badge_cls = color_meta(r["Signal"])
        st.markdown(f'<div class="card" style="padding:0;">'
                    f'<div class="head {head_cls}">{title} — {r["Ticker"]}</div>'
                    f'<div style="padding:12px 14px;">'
                    f'<span class="{badge_cls}" style="margin-right:.5rem">Prix {r["Prix"]}</span>'
                    f'<span class="badge badge-wait">RSI {r["RSI"]}</span> '
                    f'<div class="small" style="margin-top:.4rem">🎯 TP1 <b>{r["TP1"]}</b> • TP2 <b>{r["TP2"]}</b> • 🛑 SL <b>{r["SL"]}</b> • ATR <b>{r["ATR"]}</b></div>'
                    f'<div class="small">📝 {r["Raison"]}</div>'
                    f'</div></div>', unsafe_allow_html=True)

        if show_charts and isinstance(r["Data"], pd.DataFrame) and not r["Data"].empty:
            d = r["Data"].reset_index().rename(columns={"index":"Date"})
            chart = alt.Chart(d).mark_line().encode(
                x=alt.X('Date:T', axis=alt.Axis(labels=False, ticks=False, title=None)),
                y=alt.Y('Close:Q', axis=alt.Axis(title=None))
            ).properties(height=110)
            st.altair_chart(chart, use_container_width=True)
        st.write("")

st.caption("⚠️ Outil éducatif. Niveaux indicatifs (ATR).")
