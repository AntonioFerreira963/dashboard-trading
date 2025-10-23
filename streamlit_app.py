# streamlit_app.py — V3 robuste
import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Dashboard Trading", layout="wide")
st.title("📈 Dashboard rapide — XAUUSD + 2 actions")

# --- Barre latérale ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    xau_ticker = st.text_input("Symbole Or (Yahoo)", value="GC=F")  # alternative: XAUUSD=X
    stock1 = st.text_input("Action 1", value="BABA")
    stock2 = st.text_input("Action 2", value="NVDA")
    days = st.slider("Historique (jours)", 5, 365, 30)
    st.caption("Astuce : si pas de graphe, mets 30 jours ou 7 jours.")

# --- Aides ---
def _pick_period_interval(days:int):
    # Choix compatible Yahoo (certains couples vident le résultat)
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
    # harmonise les noms
    low = {c.lower(): c for c in df.columns}
    if "close" in low:
        close = df[low["close"]]
    elif "adj close" in low:
        close = df[low["adj close"]]
    else:
        close = df.select_dtypes("number").iloc[:, -1]
    volume = df[low["volume"]] if "volume" in low else pd.Series([None]*len(df), index=df.index)
    out = pd.DataFrame({"close": close, "volume": volume})
    return out.dropna(subset=["close"])

@st.cache_data(ttl=300, show_spinner=False)
def last_price(ticker:str):
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
def load_series(ticker:str, period_days:int) -> pd.DataFrame:
    """Récupère une série fiable avec plusieurs fallback."""
    try:
        # 1) Ticker().history (souvent plus stable que download)
        period, interval = _pick_period_interval(period_days)
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, prepost=True, actions=False, auto_adjust=False)
        if df is None or df.empty:
            # 2) Quotidien (toujours dispo)
            df = tk.history(period=f"{period_days}d", interval="1d", prepost=True, actions=False, auto_adjust=False)
        if df is None or df.empty:
            # 3) download (autre backend)
            df = yf.download(ticker, period="30d", interval="30m", progress=False, prepost=True)
        if df is None or df.empty:
            # 4) download quotidien
            df = yf.download(ticker, period=f"{period_days}d", interval="1d", progress=False, prepost=True)
        if df is None or df.empty:
            return pd.DataFrame()
        return _ensure_close_volume(df)
    except Exception:
        return pd.DataFrame()

# --- Ligne 1 : prix instantanés ---
c1, c2, c3 = st.columns(3)
for col, tkr, label in [(c1, xau_ticker, "Or (XAUUSD)"),
                        (c2, stock1, "Action 1"),
                        (c3, stock2, "Action 2")]:
    with col:
        p = last_price(tkr)
        if p is None:
            st.error(f"{label} — {tkr}: symbole introuvable")
        else:
            st.metric(f"{label} — {tkr}", f"{p:,.2f}")

st.divider()

# --- RSI + graphiques ---
def add_rsi(df, window=14):
    if df.empty: 
        return df
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df
def calc_tp_sl(price, vol_pct=0.02):
    """Calcule TP et SL simples à partir d'un prix et d'une volatilité approx."""
    tp = price * (1 + vol_pct)
    sl = price * (1 - vol_pct/1.3)  # SL un peu plus serré
    return tp, sl

def chart_block(title, ticker):
    st.subheader(f"{title} — {ticker}")
    df = load_series(ticker, days)
    if df.empty:
        st.warning("Pas de données (essaie 30 jours ou 7 jours, ou un autre ticker).")
        return

    df = add_rsi(df)
    st.line_chart(df[["close"]])

    price = df["close"].iloc[-1]
    rsi = df["RSI"].dropna().iloc[-1] if "RSI" in df and df["RSI"].notna().any() else None

    if rsi is not None:
        st.caption(f"RSI(14) : {rsi:.1f} | Dernier prix : {price:,.2f}")

        # Badge visuel
        if rsi < 30:
            st.success("✅ Signal rapide : **ACHAT POTENTIEL** (RSI < 30)")
        elif rsi > 70:
            st.error("🚨 Signal rapide : **SHORT/VENTE POTENTIEL** (RSI > 70)")
        else:
            st.info("➖ Signal rapide : **ATTENTE** (RSI neutre)")

        # TP / SL auto
        tp, sl = calc_tp_sl(price)
        st.write(f"🎯 TP auto ≈ {tp:,.2f} | 🛑 SL auto ≈ {sl:,.2f}")
    else:
        st.caption("RSI indisponible pour cet intervalle.")


colA, colB, colC = st.columns(3)
with colA: chart_block("Or", xau_ticker)
with colB: chart_block("Action 1", stock1)
with colC: chart_block("Action 2", stock2)

st.caption("⚠️ Indicateurs simplifiés à but informatif.")
