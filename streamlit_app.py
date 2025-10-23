# streamlit_app.py — V6 (watchlist dans la sidebar + Robot/Tableau + Cartes + graphs)
import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Robot Trading – V6", layout="wide")
st.title("🤖 Robot Trading — RSI / MACD / EMA (V6)")

with st.expander("ℹ️ Règles des signaux", expanded=False):
    st.markdown("""
**ACHAT** : RSI < 30 **ou** MACD croise haussier (négatif → positif).  
**SHORT** : RSI > 70 **ou** MACD croise baissier (positif → négatif).  
**CASSURE+** : Prix > EMA20 > EMA50 **et** passage **au-dessus** de l’EMA20 aujourd’hui.  
**CASSURE-** : Prix < EMA20 < EMA50 **et** passage **en-dessous** de l’EMA20 aujourd’hui.  
**TP/SL auto** : calculs indicatifs basés sur une volatilité simple (p.ex. ±2%).  
""")

# ============
#   SIDEBAR
# ============
with st.sidebar:
    st.header("⚙️ Paramètres")

    default_watchlist = [
        "GC=F","RIOT","APLD","WULF","IREN",
        "NVDA","BABA","SPY","QQQ","MSTR","TSLA"
    ]

    selection = st.multiselect(
        "Sélectionne tes tickers (recherche possible)",
        options=sorted(default_watchlist),
        default=["GC=F","RIOT","APLD","WULF","IREN","NVDA"]
    )

    new_tkr = st.text_input("➕ Ajouter un ticker (Yahoo)")
    if new_tkr:
        selection = list(dict.fromkeys(selection + [new_tkr.upper()]))

    days = st.slider("Historique (jours)", 5, 365, 30)
    view_mode = st.radio("Affichage", ["Tableau", "Cartes"], horizontal=True)
    show_charts = st.checkbox("📊 Graphiques (bas de page)", value=True)
    st.caption("Astuce : si pas de graphe, essaie 30 jours ou 7 jours.")

# =====================
#  HELPERS & CACHING
# =====================
def _pick_period_interval(days: int):
    if days <= 7:   return ("7d", "15m")
    if days <= 30:  return (f"{days}d", "30m")
    if days <= 90:  return (f"{days}d", "60m")
    return (f"{days}d", "1d")

def _ensure_close_volume(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    low = {c.lower(): c for c in df.columns}
    if "close" in low: close = df[low["close"]]
    elif "adj close" in low: close = df[low["adj close"]]
    else: close = df.select_dtypes("number").iloc[:, -1]
    volume = df[low["volume"]] if "volume" in low else pd.Series([None]*len(df), index=df.index)
    return pd.DataFrame({"close": close, "volume": volume}).dropna(subset=["close"])

@st.cache_data(ttl=300, show_spinner=False)
def load_series(ticker: str, period_days: int) -> pd.DataFrame:
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
    df["EMA20"]  = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"]  = df["close"].ewm(span=50, adjust=False).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    return df

def _cross_today(price_prev, ema20_prev, price_now, ema20_now, up=True):
    if any(pd.isna(x) for x in [price_prev, ema20_prev, price_now, ema20_now]): return False
    return (price_prev < ema20_prev and price_now > ema20_now) if up else (price_prev > ema20_prev and price_now < ema20_now)

def classify_signal(row_now, row_prev):
    p  = float(row_now["close"])
    rsi = float(row_now["RSI"])
    ema20_now, ema50_now = float(row_now["EMA20"]), float(row_now["EMA50"])
    macd_now, macd_prev   = float(row_now["MACD"]), float(row_prev["MACD"])
    cross_up  = (macd_prev < 0 and macd_now > 0)
    cross_down= (macd_prev > 0 and macd_now < 0)
    cross_above = _cross_today(row_prev["close"], row_prev["EMA20"], row_now["close"], row_now["EMA20"], up=True)
    cross_below = _cross_today(row_prev["close"], row_prev["EMA20"], row_now["close"], row_now["EMA20"], up=False)

    if rsi < 30 or cross_up:
        return "ACHAT",  f"RSI {rsi:.1f} ou MACD↑"
    if rsi > 70 or cross_down:
        return "SHORT",  f"RSI {rsi:.1f} ou MACD↓"
    if (p > ema20_now > ema50_now) and cross_above:
        return "CASSURE+", "Prix>EMA20>EMA50 (cassure EMA20↑)"
    if (p < ema20_now < ema50_now) and cross_below:
        return "CASSURE-", "Prix<EMA20<EMA50 (cassure EMA20↓)"
    return "ATTENTE", "Neutre"

def calc_tp_sl(price, pct=0.02):
    tp = price * (1 + pct)
    sl = price * (1 - pct/1.3)
    return round(tp,2), round(sl,2)

# =========================
#   SCAN (garde tout, même ATTENTE)
# =========================
tickers = selection if selection else default_watchlist
rows = []
for t in tickers:
    df = load_series(t, days)
    if df.empty or len(df) < 50:
        rows.append({"Ticker": t, "Prix": None, "RSI": None, "Signal": "ATTENTE",
                     "Raison": "Pas assez de données", "TP": None, "SL": None})
        continue
    df = add_indicators(df).dropna()
    row_now, row_prev = df.iloc[-1], df.iloc[-2]
    sig, reason = classify_signal(row_now, row_prev)
    tp, sl = calc_tp_sl(float(row_now["close"]))
    rows.append({
        "Ticker": t,
        "Prix": round(float(row_now["close"]),2),
        "RSI": round(float(row_now["RSI"]),1),
        "Signal": sig,
        "Raison": reason,
        "TP": tp,
        "SL": sl
    })

df_all = pd.DataFrame(rows)

# --- Résumé des comptes ---
c1,c2,c3,c4,c5 = st.columns(5)
total = len(df_all)
c1.metric("Tickers scannés", total)
c2.metric("✅ Achat", int((df_all["Signal"]=="ACHAT").sum()))
c3.metric("🚨 Short", int((df_all["Signal"]=="SHORT").sum()))
c4.metric("📈 Cassure+", int((df_all["Signal"]=="CASSURE+").sum()))
c5.metric("📉 Cassure-", int((df_all["Signal"]=="CASSURE-").sum()))

# ==============================
#   AFFICHAGE
# ==============================
if view_mode == "Tableau":
    st.subheader("📋 Signaux — Vue Robot")

    col1,col2,col3,col4 = st.columns(4)

    def show_bucket(col, label, emoji):
        subset = df_all[df_all["Signal"]==label][["Ticker","Prix","RSI","TP","SL","Raison"]]
        col.markdown(f"### {emoji} {label}")
        if subset.empty: col.write("—")
        else: col.dataframe(subset.reset_index(drop=True), use_container_width=True)

    show_bucket(col1, "ACHAT", "✅")
    show_bucket(col2, "SHORT", "🚨")
    show_bucket(col3, "CASSURE+", "📈")
    show_bucket(col4, "CASSURE-", "📉")

    with st.expander("👀 Voir aussi la liste complète (y compris ATTENTE)", expanded=False):
        st.dataframe(df_all[["Ticker","Prix","RSI","Signal","TP","SL","Raison"]].reset_index(drop=True),
                     use_container_width=True)

else:
    st.subheader("🧾 Vue Cartes (par action)")
    for _, r in df_all.iterrows():
        with st.container():
            cA,cB,cC,cD = st.columns([2,1,1,2])
            cA.markdown(f"**{r['Ticker']}**")
            cB.metric("Prix", f"{r['Prix'] if pd.notna(r['Prix']) else '-'}")
            cC.metric("RSI(14)", f"{r['RSI'] if pd.notna(r['RSI']) else '-'}")
            badge = {"ACHAT":"✅ ACHAT","SHORT":"🚨 SHORT","CASSURE+":"📈 CASSURE+",
                     "CASSURE-":"📉 CASSURE-","ATTENTE":"➖ ATTENTE"}[r["Signal"]]
            cD.markdown(f"**{badge}**  \n{r['Raison']}")
            st.caption(f"🎯 TP: {r['TP'] if pd.notna(r['TP']) else '-'} | 🛑 SL: {r['SL'] if pd.notna(r['SL']) else '-'}")
            st.divider()

# ==============================
#   GRAPHES (bas de page)
# ==============================
if show_charts:
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

st.caption("⚠️ Outil éducatif. Rien ici n'est un conseil d'investissement.")
