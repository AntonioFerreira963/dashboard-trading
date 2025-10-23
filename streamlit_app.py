# streamlit_app.py — V5 Robot (lisible 4 colonnes + tableaux clairs)
import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Robot Trading – 4 colonnes", layout="wide")
st.title("🤖 Robot Trading — Signaux automatiques (RSI / MACD / EMA)")

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
#  HELPERS & CACHING
# =====================
def _pick_period_interval(days: int):
    if days <= 7: return ("7d", "15m")
    elif days <= 30: return (f"{days}d", "30m")
    elif days <= 90: return (f"{days}d", "60m")
    else: return (f"{days}d", "1d")

def _ensure_close_volume(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    low = {c.lower(): c for c in df.columns}
    close = df[low["close"]] if "close" in low else df.select_dtypes("number").iloc[:, -1]
    volume = df[low["volume"]] if "volume" in low else pd.Series([None]*len(df), index=df.index)
    return pd.DataFrame({"close": close, "volume": volume}).dropna(subset=["close"])

@st.cache_data(ttl=300)
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
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACDsig"] = df["MACD"].ewm(span=9).mean()
    return df

def classify_signal(row_now, row_prev):
    p  = float(row_now["close"])
    rsi = float(row_now["RSI"])
    ema20, ema50 = row_now["EMA20"], row_now["EMA50"]
    macd_now, macd_prev = row_now["MACD"], row_prev["MACD"]
    # règles
    if rsi < 30 or (macd_prev < 0 and macd_now > 0):
        return "ACHAT", f"RSI {rsi:.1f} / MACD↑"
    if rsi > 70 or (macd_prev > 0 and macd_now < 0):
        return "SHORT", f"RSI {rsi:.1f} / MACD↓"
    if p > ema20 > ema50: return "CASSURE+", "Cassure haussière EMA20/50"
    if p < ema20 < ema50: return "CASSURE-", "Cassure baissière EMA20/50"
    return "ATTENTE", "Neutre"

def calc_tp_sl(price, pct=0.02):
    return price*(1+pct), price*(1-pct/1.3)

# =========================
#   SCAN & CLASSIFICATION
# =========================
tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]
rows = []
for t in tickers:
    df = load_series(t, days)
    if df.empty or len(df)<50: 
        rows.append({"Ticker":t,"Signal":"N/A"}); continue
    df = add_indicators(df).dropna()
    row_now, row_prev = df.iloc[-1], df.iloc[-2]
    sig, reason = classify_signal(row_now, row_prev)
    tp, sl = calc_tp_sl(row_now["close"])
    rows.append({
        "Ticker":t, "Prix":round(row_now["close"],2),
        "RSI":round(row_now["RSI"],1),
        "Signal":sig, "Raison":reason,
        "TP":round(tp,2), "SL":round(sl,2)
    })

df_res = pd.DataFrame(rows)

# ==============================
#   TABLEAU 4 COLONNES
# ==============================
st.subheader("📋 Signaux — Vue Robot")

col1,col2,col3,col4 = st.columns(4)
def show(col,label,emoji):
    subset = df_res[df_res["Signal"]==label][["Ticker","Prix","RSI","TP","SL","Raison"]]
    col.markdown(f"### {emoji} {label}")
    if subset.empty: col.write("—")
    else: col.dataframe(subset, use_container_width=True)

show(col1,"ACHAT","✅")
show(col2,"SHORT","🚨")
show(col3,"CASSURE+","📈")
show(col4,"CASSURE-","📉")

# ==============================
#   GRAPHES
# ==============================
if show_charts:
    st.divider(); st.subheader("📊 Graphiques (top 6 tickers)")
    for i in range(0,len(tickers[:6]),3):
        cols = st.columns(3)
        for j,t in enumerate(tickers[i:i+3]):
            with cols[j]:
                df = load_series(t,days)
                if df.empty: st.write(f"{t}: pas de données"); continue
                df = add_indicators(df)
                st.markdown(f"**{t}**")
                st.line_chart(df[["close"]])
                st.caption(f"RSI(14) : {df['RSI'].iloc[-1]:.1f}")
