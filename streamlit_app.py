# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Dashboard Trading", layout="wide")

st.title("📈 Dashboard rapide — XAUUSD + 2 actions")

# --- Paramètres par défaut (modifiables dans la barre latérale) ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    xau_ticker = st.text_input("Symbole Or (Yahoo)", value="GC=F")  # alternative: GC=F
    stock1 = st.text_input("Action 1", value="BABA")
    stock2 = st.text_input("Action 2", value="NVDA")
    days = st.slider("Historique (jours)", 5, 365, 60)
    st.caption("Astuce : tape le symbole exact Yahoo Finance.")

def _ensure_close_and_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les colonnes 'close' et 'volume' quel que soit le format Yahoo."""
    if df is None or df.empty:
        return pd.DataFrame()

    cols = {c.lower(): c for c in df.columns}

    # close
    if "close" in cols:
        close_series = df[cols["close"]]
    elif "adj close" in cols:
        close_series = df[cols["adj close"]]
    else:
        # dernier fallback : dernière colonne numérique
        close_series = df.select_dtypes("number").iloc[:, -1]

    # volume
    if "volume" in cols:
        volume_series = df[cols["volume"]]
    else:
        volume_series = pd.Series([None] * len(df), index=df.index)

    out = pd.DataFrame({"close": close_series, "volume": volume_series})
    return out


def load_series(ticker, period_days):
    try:
        # Essai 1 : intervalle 1h
        df = yf.download(ticker, period=f"{period_days}d", interval="1h", progress=False)
        # Essai 2 : si vide → intervalle 30m (max 30 jours)
        if df is None or df.empty:
            df = yf.download(ticker, period=f"{min(period_days,30)}d", interval="30m", progress=False)
        # Essai 3 : si encore vide → intervalle 1d (quotidien, plus fiable)
        if df is None or df.empty:
            df = yf.download(ticker, period=f"{period_days}d", interval="1d", progress=False)

        if df is None or df.empty:
            return pd.DataFrame()

        df = _ensure_close_and_volume(df)
        return df
    except Exception:
        return pd.DataFrame()




def last_price(ticker):
    data = yf.Ticker(ticker).history(period="1d", interval="1m")
    if data.empty:
        return None
    return float(data["Close"].iloc[-1])

# --- Ligne 1 : Prix instantanés ---
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

# --- Ligne 2 : Graphiques horaires + RSI simple ---
def add_rsi(df, window=14):
    if df.empty: 
        return df
    delta = df["close"].diff()
    gain = (delta.clip(lower=0)).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def chart_block(title, ticker):
    st.subheader(f"{title} — {ticker}")
    df = load_series(ticker, days)
    if df.empty:
        st.warning("Pas de données. Vérifie le symbole.")
        return
    df = add_rsi(df)
    st.line_chart(df[["close"]])
    st.caption(f"RSI(14) dernière valeur : {df['RSI'].iloc[-1]:.1f}")
    # Signal très simple (indicatif)
    rsi = df["RSI"].iloc[-1]
    if rsi < 30:
        st.success("Signal rapide : **ACHAT POTENTIEL** (RSI < 30)")
    elif rsi > 70:
        st.error("Signal rapide : **VENTE/SHORT POTENTIEL** (RSI > 70)")
    else:
        st.info("Signal rapide : **ATTENTE** (RSI neutre)")

colA, colB, colC = st.columns(3)
with colA: chart_block("Or", xau_ticker)
with colB: chart_block("Action 1", stock1)
with colC: chart_block("Action 2", stock2)

st.caption("⚠️ Indicateurs simplifiés à but informatif.")

