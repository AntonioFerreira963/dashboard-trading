# streamlit_app.py — V4 Robot (tableau 4 colonnes en haut + graphes en bas)
import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Robot Trading – 4 colonnes", layout="wide")
st.title("🤖 Robot Trading — ACHAT / SHORT / Cassures")

# ============
#   SIDEBAR
# ============
with st.sidebar:
    st.header("⚙️ Paramètres")
    # Liste par défaut : Or + miners BTC + tech + indices
    tickers_text = st.text_area(
        "Tickers Yahoo (séparés par virgule)",
        value="GC=F,RIOT,APLD,WULF,IREN,NVDA,BABA,SPY,QQQ,MSTR,TSLA",
        height=80
    )
    days = st.slider("Historique (jours)", 5, 365, 30)
    show_charts = st.checkbox("Afficher les graphiques (bas de page)", value=True)
    st.caption("Astuce : si pas de graphe, essaie 30 jours ou 7 jours.")

# =====================
#  HELPERS & CACHING
# =====================
def _pick_period_interval(days: int):
    if days <= 7:
        return ("7d", "15m")
    elif days <= 30:
        return (f"{days}d", "30m")
    elif days <= 90:
        return (f"{days}d", "60m")
    else:
        return (f"{days}d", "1d")

def _ensure_close_volume(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    low = {c.lower(): c for c in df.columns}
    # close
    if "close" in low:
        close = df[low["close"]]
    elif "adj close" in low:
        close = df[low["adj close"]]
    else:
        close = df.select_dtypes("number").iloc[:, -1]
    # volume
    volume = df[low["volume"]] if "volume" in low else pd.Series([None] * len(df), index=df.index)
    out = pd.DataFrame({"close": close, "volume": volume})
    return out.dropna(subset=["close"])

@st.cache_data(ttl=300, show_spinner=False)
def last_price(ticker: str):
    try:
        d = yf.Ticker(ticker).history(period="1d", interval="1m", prepost=True, actions=False)
        if d is None or d.empty:
            d = yf.Ticker(ticker).history(period="5d", interval="5m", prepost=True, actions=False)
        if d is None or d.empty:
            return None
        return float(d["Close"].iloc[-1])
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def load_series(ticker: str, period_days: int) -> pd.DataFrame:
    """Récupère une série fiable avec plusieurs fallback."""
    try:
        period, interval = _pick_period_interval(period_days)
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, prepost=True, actions=False, auto_adjust=False)
        if df is None or df.empty:
            df = tk.history(period=f"{period_days}d", interval="1d", prepost=True, actions=False, auto_adjust=False)
        if df is None or df.empty:
            df = yf.download(ticker, period="30d", interval="30m", progress=False, prepost=True)
        if df is None or df.empty:
            df = yf.download(ticker, period=f"{period_days}d", interval="1d", progress=False, prepost=True)
        if df is None or df.empty:
            return pd.DataFrame()
        return _ensure_close_volume(df)
    except Exception:
        return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # RSI(14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    # EMA
    df["EMA20"]  = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"]  = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACDsig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def classify_signal(row_now, row_prev):
    """
    Renvoie un dict:
      'label' in {'ACHAT','SHORT','ATTENTE','CASSURE+','CASSURE-'}
      'reason' (texte court)
    """
    p  = float(row_now["close"])
    rsi = float(row_now["RSI"]) if pd.notna(row_now["RSI"]) else None
    ema20_now  = float(row_now["EMA20"]) if pd.notna(row_now["EMA20"]) else None
    ema50_now  = float(row_now["EMA50"]) if pd.notna(row_now["EMA50"]) else None
    macd_now   = float(row_now["MACD"]) if pd.notna(row_now["MACD"]) else None
    macd_prev  = float(row_prev["MACD"]) if pd.notna(row_prev["MACD"]) else None
    em20_prev  = float(row_prev["EMA20"]) if pd.notna(row_prev["EMA20"]) else None
    price_prev = float(row_prev["close"])

    macd_cross_up   = macd_prev is not None and macd_now is not None and macd_prev < 0 and macd_now > 0
    macd_cross_down = macd_prev is not None and macd_now is not None and macd_prev > 0 and macd_now < 0

    cross_above_ema20 = (em20_prev is not None and ema20_now is not None
                         and price_prev < em20_prev and p > ema20_now)
    cross_below_ema20 = (em20_prev is not None and ema20_now is not None
                         and price_prev > em20_prev and p < ema20_now)

    cassure_plus  = (ema20_now is not None and ema50_now is not None
                     and p > ema20_now > ema50_now and cross_above_ema20)
    cassure_moins = (ema20_now is not None and ema50_now is not None
                     and p < ema20_now < ema50_now and cross_below_ema20)

    if rsi is not None:
        if rsi < 30 or macd_cross_up or (ema20_now and ema50_now and p > ema20_now > ema50_now and cross_above_ema20):
            return {"label": "ACHAT", "reason": f"RSI {rsi:.1f} / MACD↑ / EMA20>EMA50"}
        if rsi > 70 or macd_cross_down or (ema20_now and ema50_now and p < ema20_now < ema50_now and cross_below_ema20):
            return {"label": "SHORT", "reason": f"RSI {rsi:.1f} / MACD↓ / EMA20<EMA50"}

    if cassure_plus:
        return {"label": "CASSURE+", "reason": "Prix>EMA20>EMA50 (cassure)"}
    if cassure_moins:
        return {"label": "CASSURE-", "reason": "Prix<EMA20<EMA50 (cassure)"}

    return {"label": "ATTENTE", "reason": "Signal neutre"}

def calc_tp_sl(price, vol_pct=0.02):
    tp = price * (1 + vol_pct)
    sl = price * (1 - vol_pct / 1.3)
    return tp, sl

# =========================
#   SCAN & CLASSIFICATION
# =========================
tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]
results = []

for t in tickers:
    df = load_series(t, days)
    if df.empty or len(df) < 50:
        results.append({"Ticker": t, "Prix": None, "RSI": None, "Signal": "N/A", "Raison": "Pas de données"})
        continue

    df = add_indicators(df).dropna()
    if df.empty:
        results.append({"Ticker": t, "Prix": None, "RSI": None, "Signal": "N/A", "Raison": "Indicateurs indisponibles"})
        continue

    row_now  = df.iloc[-1]
    row_prev = df.iloc[-2]
    p = float(row_now["close"])
    rsi = float(row_now["RSI"])
    sig = classify_signal(row_now, row_prev)
    tp, sl = calc_tp_sl(p)

    results.append({
        "Ticker": t,
        "Prix": round(p, 2),
        "RSI": round(rsi, 1),
        "Signal": sig["label"],
        "Raison": sig["reason"],
        "TP_auto": round(tp, 2),
        "SL_auto": round(sl, 2)
    })

df_res = pd.DataFrame(results)

# ==============================
#   TABLEAU 4 COLONNES (TOP)
# ==============================
st.subheader("📋 Signals — Vue Robot (top)")
col_buy, col_short, col_c_up, col_c_down = st.columns(4)

def render_bucket(col, title, label):
    col.markdown(f"**{title}**")
    subset = df_res[df_res["Signal"] == label].copy()
    if subset.empty:
        col.write("—")
        return
    subset = subset[["Ticker", "Prix", "RSI", "TP_auto", "SL_auto", "Raison"]]
    col.dataframe(subset, use_container_width=True)

render_bucket(col_buy,   "✅ ACHETER", "ACHAT")
render_bucket(col_short, "🚨 SHORTER", "SHORT")
render_bucket(col_c_up,  "📈 CASSURE HAUSSIÈRE", "CASSURE+")
render_bucket(col_c_down,"📉 CASSURE BAISSIÈRE", "CASSURE-")

st.caption("Les signaux sont basés sur RSI, croisements MACD, et structure EMA20/EMA50.")

# ==============================
#   GRAPHES (BAS DE PAGE)
# ==============================
if show_charts:
    st.divider()
    st.subheader("📊 Graphiques")
    to_plot = tickers[:6]  # jusqu'à 6 graphes
    for i in range(0, len(to_plot), 3):
        cols = st.columns(3)
        for j, t in enumerate(to_plot[i:i+3]):
            with cols[j]:
                ddf = load_series(t, days)
                if ddf.empty:
                    st.write(f"{t} — pas de données")
                    continue
                ddf = add_indicators(ddf)
                st.markdown(f"**{t}**")
                st.line_chart(ddf[["close"]])
                if "RSI" in ddf and ddf["RSI"].notna().any():
                    st.caption(f"RSI(14) : {ddf['RSI'].dropna().iloc[-1]:.1f}")

st.caption("⚠️ Outil éducatif. Les signaux ne sont pas des conseils d'investissement.")
