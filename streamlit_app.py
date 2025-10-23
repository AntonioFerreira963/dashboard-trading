# streamlit_app.py — V5.2 Robot lisible (SL affiché + définitions claires)
import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Robot Trading – 4 colonnes", layout="wide")
st.title("🤖 Robot Trading — Signaux automatiques (RSI / MACD / EMA)")

with st.expander("ℹ️ Règles des signaux (cliquer pour voir)", expanded=False):
    st.markdown("""
**ACHAT** : RSI < 30 **ou** croisement MACD haussier (passe <0 → >0).  
**SHORT** : RSI > 70 **ou** croisement MACD baissier (passe >0 → <0).  
**CASSURE+** : Prix > EMA20 > EMA50 **et** passage **au-dessus** de l’EMA20 aujourd’hui.  
**CASSURE-** : Prix < EMA20 < EMA50 **et** passage **en-dessous** de l’EMA20 aujourd’hui.  
TP/SL auto = calculs indicatifs basés sur une volatilité simple (pas des conseils).
""")

# ============
#   SIDEBAR
# ============
with st.sidebar:
    st.header("⚙️ Paramètres")
    tickers_text = st.text_area(
        "Tickers Yahoo (séparés par virgule)",
        value="GC=F,RIOT,APLD,WULF,IREN,NVDA,BABA,SPY,QQQ,MSTR,TSLA",
        height=80
    )
    days = st.slider("Historique (jours)", 5, 365, 30)
    show_charts = st.checkbox("📊 Afficher les graphiques (bas de page)", value=True)
    st.caption("Astuce : si pas de graphe, essaie 30 jours ou 7 jours.")

# =====================
#  HELPERS
# =====================
def _pick_period_interval(days: int):
    if days <= 7: return ("7d", "15m")
    elif days <= 30: return (f"{days}d", "30m")
    elif days <= 90: return (f"{days}d", "60m")
    else: return (f"{days}d", "1d")

def _ensure_close_volume(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    low = {c.lower(): c for c in df.columns}
    close = df[low["close"]] if "close" in low else (
        df[low["adj close"]] if "adj close" in low else df.select_dtypes("number").iloc[:, -1]
    )
    volume = df[low["volume"]] if "volume" in low else pd.Series([None]*len(df), index=df.index)
    return pd.DataFrame({"close": close, "volume": volume}).dropna(subset=["close"])

@st.cache_data(ttl=300, show_spinner=False)
def load_series(ticker: str, period_days: int) -> pd.DataFrame:
    try:
        period, interval = _pick_period_interval(period_days)
        df = yf.Ticker(ticker).history(period=period, interval=interval, prepost=True, actions=False)
        if df is None or df.empty:
            df = yf.download(ticker, period=f"{period_days}d", interval="1d", progress=False, prepost=True)
        if df is None or df.empty: return pd.DataFrame()
        return _ensure_close_volume(df)
    except Exception:
        return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    return df

def _cross_today(price_prev, ema20_prev, price_now, ema20_now, direction="+"):
    if any(pd.isna(x) for x in [price_prev, ema20_prev, price_now, ema20_now]): return False
    if direction == "+":
        return price_prev < ema20_prev and price_now > ema20_now
    return price_prev > ema20_prev and price_now < ema20_now

def classify_signal(row_now, row_prev):
    p  = float(row_now["close"])
    rsi = float(row_now["RSI"])
    ema20_now, ema50_now = float(row_now["EMA20"]), float(row_now["EMA50"])
    macd_now, macd_prev = float(row_now["MACD"]), float(row_prev["MACD"])
    cross_up = macd_prev < 0 and macd_now > 0
    cross_dn = macd_prev > 0 and macd_now < 0

    cross_above = _cross_today(row_prev["close"], row_prev["EMA20"], row_now["close"], row_now["EMA20"], "+")
    cross_below = _cross_today(row_prev["close"], row_prev["EMA20"], row_now["close"], row_now["EMA20"], "-")

    if rsi < 30 or cross_up:
        return "ACHAT", f"RSI {rsi:.1f} ou MACD↑"
    if rsi > 70 or cross_dn:
        return "SHORT", f"RSI {rsi:.1f} ou MACD↓"
    if (p > ema20_now > ema50_now) and cross_above:
        return "CASSURE+", "Prix>EMA20>EMA50 (cassure au-dessus EMA20)"
    if (p < ema20_now < ema50_now) and cross_below:
        return "CASSURE-", "Prix<EMA20<EMA50 (cassure sous EMA20)"
    return "ATTENTE", "Neutre"

def calc_tp_sl(price, pct=0.02):
    tp = price * (1 + pct)
    sl = price * (1 - pct/1.3)
    return round(tp,2), round(sl,2)

# =========================
#   SCAN & TABLES
# =========================
tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]
rows = []
for t in tickers:
    df = load_series(t, days)
    if df.empty or len(df) < 50:
        rows.append({"Ticker": t, "Prix": None, "RSI": None, "Signal": "N/A", "Raison": "Pas de données", "TP": None, "SL": None})
        continue
    df = add_indicators(df).dropna()
    row_now, row_prev = df.iloc[-1], df.iloc[-2]
    sig, reason = classify_signal(row_now, row_prev)
    tp, sl = calc_tp_sl(float(row_now["close"]))
    rows.append({
        "Ticker": t, "Prix": round(float(row_now["close"]),2),
        "RSI": round(float(row_now["RSI"]),1),
        "Signal": sig, "Raison": reason, "TP": tp, "SL": sl
    })
df_res = pd.DataFrame(rows)

st.subheader("📋 Signaux — Vue Robot")

col1, col2, col3, col4 = st.columns(4)
def show_bucket(col, label, emoji):
    subset = df_res[df_res["Signal"] == label][["Ticker","Prix","RSI","TP","SL","Raison"]]
    col.markdown(f"### {emoji} {label}")
    if subset.empty: col.write("—")
    else: col.dataframe(subset.reset_index(drop=True), use_container_width=True)

show_bucket(col1, "ACHAT", "✅")
show_bucket(col2, "SHORT", "🚨")
show_bucket(col3, "CASSURE+", "📈")
show_bucket(col4, "CASSURE-", "📉")

st.caption("Signaux basés sur RSI, MACD et structure EMA20/EMA50. **TP/SL auto** = indicatif.")

# ==============================
#   GRAPHES (bas de page)
# ==============================
if show_charts:
    st.divider()
    st.subheader("📊 Graphiques (top 6 tickers)")
    to_plot = tickers[:6]
    for i in range(0, len(to_plot), 3):
        cols = st.columns(3)
        for j, t in enumerate(to_plot[i:i+3]):
            with cols[j]:
                ddf = load_series(t, days)
                if ddf.empty:
                    st.write(f"{t} : pas de données"); continue
                ddf = add_indicators(ddf)
                st.markdown(f"**{t}**")
                st.line_chart(ddf[["close"]])
                if "RSI" in ddf and ddf["RSI"].notna().any():
                    st.caption(f"RSI(14) : {ddf['RSI'].dropna().iloc[-1]:.1f}")
