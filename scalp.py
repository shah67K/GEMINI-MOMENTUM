import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Open Momentum Signal Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp {
    background:#0b0f14;
    color:#eeeeee;
}

.block-container {
    padding-top:1rem;
    padding-bottom:1rem;
    max-width:1500px;
}

h1 {
    font-size:26px !important;
    margin-bottom:5px !important;
}

h2 {
    font-size:19px !important;
}

h3 {
    font-size:16px !important;
}

div[data-testid="stMetric"] {
    background:#121820;
    border:1px solid #26303a;
    padding:8px;
    border-radius:8px;
}

div[data-testid="stMetricLabel"] {
    font-size:11px !important;
}

div[data-testid="stMetricValue"] {
    font-size:20px !important;
}

.small {
    font-size:12px;
    color:#aeb8c2;
}

.signal-box {
    background:#111820;
    border:1px solid #34404c;
    border-radius:10px;
    padding:12px;
    margin-bottom:8px;
}

.go {
    background:#12351f;
    border:1px solid #2d9d52;
}

.wait {
    background:#191d21;
    border:1px solid #505861;
}

.danger {
    background:#351719;
    border:1px solid #b94a50;
}

.big {
    font-size:25px;
    font-weight:700;
}

.label {
    font-size:10px;
    color:#8e9aa6;
    text-transform:uppercase;
}

.value {
    font-size:17px;
    font-weight:600;
}

hr {
    border-color:#27313b;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SETTINGS
# =========================================================

REFRESH_DEFAULT = 15

SYMBOLS = {
    "NIFTY": "^NSEI",
    "BANK NIFTY": "^NSEBANK"
}

LOG_FILE = "open_momentum_signals.csv"

# =========================================================
# DATA
# =========================================================

@st.cache_data(ttl=15)
def get_data(symbol):

    try:
        df = yf.download(
            symbol,
            period="5d",
            interval="5m",
            progress=False,
            auto_adjust=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        needed = ["Open", "High", "Low", "Close", "Volume"]

        for c in needed:
            if c not in df.columns:
                return pd.DataFrame()

        df = df[needed].copy()

        df.index = pd.to_datetime(df.index)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")

        df = df.sort_index()

        # -------------------------------------------------
        # REMOVE CURRENT INCOMPLETE 5-MIN CANDLE
        # -------------------------------------------------

        now = pd.Timestamp.now(tz="Asia/Kolkata")

        current_bucket = now.floor("5min")

        df = df[df.index < current_bucket]

        df = df.dropna()

        return df

    except Exception as e:
        st.error(f"Data error: {e}")
        return pd.DataFrame()


# =========================================================
# INDICATORS
# =========================================================

def atr(df, period=14):

    prev_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def supertrend(df, period=7, multiplier=2):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    atr_val = atr(df, period)

    hl2 = (high + low) / 2

    upper = hl2 + multiplier * atr_val
    lower = hl2 - multiplier * atr_val

    final_upper = upper.copy()
    final_lower = lower.copy()

    direction = pd.Series(index=df.index, dtype=int)

    direction.iloc[0] = 1

    for i in range(1, len(df)):

        if (
            upper.iloc[i] < final_upper.iloc[i - 1]
            or close.iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (
            lower.iloc[i] > final_lower.iloc[i - 1]
            or close.iloc[i - 1] < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if direction.iloc[i - 1] == -1:

            if close.iloc[i] > final_upper.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1

        else:

            if close.iloc[i] < final_lower.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1

    st_line = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):

        if direction.iloc[i] == 1:
            st_line.iloc[i] = final_lower.iloc[i]
        else:
            st_line.iloc[i] = final_upper.iloc[i]

    return st_line, direction


def add_indicators(df):

    df = df.copy()

    df["ATR"] = atr(df, 14)

    df["ST"], df["ST_DIR"] = supertrend(
        df,
        period=7,
        multiplier=2
    )

    # -------------------------------------------------
    # MACD
    # -------------------------------------------------

    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_HIST"] = (
        df["MACD"] -
        df["MACD_SIGNAL"]
    )

    # -------------------------------------------------
    # DAILY PIVOT
    # -------------------------------------------------

    df["DATE"] = df.index.date

    daily = df.groupby("DATE").agg({
        "High": "max",
        "Low": "min",
        "Close": "last"
    })

    daily["PIVOT"] = (
        daily["High"] +
        daily["Low"] +
        daily["Close"]
    ) / 3

    daily["R1"] = (
        2 * daily["PIVOT"] -
        daily["Low"]
    )

    daily["S1"] = (
        2 * daily["PIVOT"] -
        daily["High"]
    )

    daily["R2"] = (
        daily["PIVOT"] +
        daily["High"] -
        daily["Low"]
    )

    daily["S2"] = (
        daily["PIVOT"] -
        daily["High"] +
        daily["Low"]
    )

    df["PIVOT"] = df["DATE"].map(daily["PIVOT"])
    df["R1"] = df["DATE"].map(daily["R1"])
    df["S1"] = df["DATE"].map(daily["S1"])
    df["R2"] = df["DATE"].map(daily["R2"])
    df["S2"] = df["DATE"].map(daily["S2"])

    # -------------------------------------------------
    # CANDLE MOMENTUM
    # -------------------------------------------------

    df["BODY"] = (df["Close"] - df["Open"]).abs()
    df["RANGE"] = df["High"] - df["Low"]

    df["BODY_PCT"] = np.where(
        df["RANGE"] > 0,
        df["BODY"] / df["RANGE"],
        0
    )

    df["UP_CANDLE"] = df["Close"] > df["Open"]
    df["DOWN_CANDLE"] = df["Close"] < df["Open"]

    # Previous highs/lows
    df["PREV_HIGH"] = df["High"].shift(1)
    df["PREV_LOW"] = df["Low"].shift(1)

    df["HIGH_3"] = df["High"].shift(1).rolling(3).max()
    df["LOW_3"] = df["Low"].shift(1).rolling(3).min()

    return df


# =========================================================
# SIGNAL ENGINE
# =========================================================

def generate_signal(df, i):

    if i < 30:
        return None

    r = df.iloc[i]
    p = df.iloc[i - 1]

    price = float(r["Close"])
    atr_val = float(r["ATR"])

    if not np.isfinite(atr_val) or atr_val <= 0:
        return None

    st_dir = int(r["ST_DIR"])
    prev_dir = int(p["ST_DIR"])

    macd = float(r["MACD"])
    macd_sig = float(r["MACD_SIGNAL"])
    hist = float(r["MACD_HIST"])
    prev_hist = float(p["MACD_HIST"])

    pivot = float(r["PIVOT"])

    bullish_candle = bool(r["UP_CANDLE"])
    bearish_candle = bool(r["DOWN_CANDLE"])

    # -------------------------------------------------
    # PRIMARY SUPERTREND EVENT
    # -------------------------------------------------

    bullish_flip = (
        st_dir == 1 and
        prev_dir == -1
    )

    bearish_flip = (
        st_dir == -1 and
        prev_dir == 1
    )

    # -------------------------------------------------
    # EARLY MOMENTUM
    # Price can move before ST officially flips.
    # This is deliberately OPEN / LIGHT.
    # -------------------------------------------------

    bullish_break = (
        price > float(r["HIGH_3"])
    )

    bearish_break = (
        price < float(r["LOW_3"])
    )

    bullish_macd = (
        macd > macd_sig or
        hist > prev_hist
    )

    bearish_macd = (
        macd < macd_sig or
        hist < prev_hist
    )

    bullish_price_action = (
        bullish_candle and
        float(r["BODY_PCT"]) >= 0.45 and
        float(r["RANGE"]) >= atr_val * 0.30
    )

    bearish_price_action = (
        bearish_candle and
        float(r["BODY_PCT"]) >= 0.45 and
        float(r["RANGE"]) >= atr_val * 0.30
    )

    # -------------------------------------------------
    # PIVOT CONTEXT
    # NOT MANDATORY
    # -------------------------------------------------

    pivot_bull = price >= pivot
    pivot_bear = price <= pivot

    # =================================================
    # CE LOGIC
    # =================================================

    ce_score = 0
    ce_reasons = []

    # Supertrend is the main engine
    if st_dir == 1:
        ce_score += 2
        ce_reasons.append("ST bullish")

    if bullish_flip:
        ce_score += 2
        ce_reasons.append("ST 7,2 flip")

    if bullish_break:
        ce_score += 1
        ce_reasons.append("price breakout")

    if bullish_macd:
        ce_score += 1
        ce_reasons.append("MACD support")

    if bullish_price_action:
        ce_score += 1
        ce_reasons.append("bullish candle")

    if pivot_bull:
        ce_score += 1
        ce_reasons.append("above Pivot")

    # =================================================
    # PE LOGIC
    # =================================================

    pe_score = 0
    pe_reasons = []

    if st_dir == -1:
        pe_score += 2
        pe_reasons.append("ST bearish")

    if bearish_flip:
        pe_score += 2
        pe_reasons.append("ST 7,2 flip")

    if bearish_break:
        pe_score += 1
        pe_reasons.append("price breakdown")

    if bearish_macd:
        pe_score += 1
        pe_reasons.append("MACD support")

    if bearish_price_action:
        pe_score += 1
        pe_reasons.append("bearish candle")

    if pivot_bear:
        pe_score += 1
        pe_reasons.append("below Pivot")

    # =================================================
    # SIGNAL DECISION
    # =================================================

    signal = None

    # Direct ST flip gets priority.
    if bullish_flip and bullish_candle:
        signal = "CE"

    elif bearish_flip and bearish_candle:
        signal = "PE"

    # Early momentum path
    elif (
        st_dir == 1 and
        bullish_break and
        (bullish_macd or bullish_price_action)
    ):
        signal = "CE"

    elif (
        st_dir == -1 and
        bearish_break and
        (bearish_macd or bearish_price_action)
    ):
        signal = "PE"

    # Continuation path
    elif (
        st_dir == 1 and
        bullish_price_action and
        bullish_macd and
        price > float(p["Close"])
    ):
        signal = "CE"

    elif (
        st_dir == -1 and
        bearish_price_action and
        bearish_macd and
        price < float(p["Close"])
    ):
        signal = "PE"

    if signal is None:
        return None

    # =================================================
    # LEVELS
    # =================================================

    # Volatility-aware but practical.
    sl_distance = max(
        atr_val * 0.70,
        12.0
    )

    sl_distance = min(
        sl_distance,
        80.0
    )

    t1_distance = max(
        atr_val * 1.00,
        sl_distance * 1.15,
        20.0
    )

    t2_distance = max(
        atr_val * 1.70,
        sl_distance * 1.70,
        40.0
    )

    if signal == "CE":

        entry = price
        sl = entry - sl_distance
        t1 = entry + t1_distance
        t2 = entry + t2_distance

        score = ce_score
        reasons = ce_reasons

        setup = (
            "ST FLIP" if bullish_flip
            else "EARLY MOMENTUM" if bullish_break
            else "CONTINUATION"
        )

    else:

        entry = price
        sl = entry + sl_distance
        t1 = entry - t1_distance
        t2 = entry - t2_distance

        score = pe_score
        reasons = pe_reasons

        setup = (
            "ST FLIP" if bearish_flip
            else "EARLY MOMENTUM" if bearish_break
            else "CONTINUATION"
        )

    # =================================================
    # SIGNAL QUALITY
    # =================================================

    if score >= 6:
        quality = "STRONG"

    elif score >= 4:
        quality = "ACTIONABLE"

    else:
        quality = "EARLY"

    reason = " + ".join(reasons[:5])

    return {
        "Time": df.index[i],
        "Signal": "GO",
        "CE_PE": signal,
        "Setup": setup,
        "Quality": quality,
        "Entry": round(entry, 2),
        "SL": round(sl, 2),
        "T1": round(t1, 2),
        "T2": round(t2, 2),
        "Reason": reason,
        "ATR": round(atr_val, 2),
        "ST_DIR": st_dir
    }


# =========================================================
# TODAY SIGNAL SCAN
# =========================================================

def scan_today(df):

    if df.empty:
        return []

    today = df.index[-1].date()

    today_df = df[
        df.index.date == today
    ].copy()

    signals = []

    last_direction = None
    last_signal_time = None

    for i in range(30, len(today_df)):

        sig = generate_signal(
            today_df,
            i
        )

        if sig is None:
            continue

        # ------------------------------------------------
        # Avoid repeated same-direction signals too close
        # ------------------------------------------------

        if (
            last_direction == sig["CE_PE"]
            and last_signal_time is not None
        ):

            minutes = (
                sig["Time"] -
                last_signal_time
            ).total_seconds() / 60

            if minutes < 20:
                continue

        signals.append(sig)

        last_direction = sig["CE_PE"]
        last_signal_time = sig["Time"]

    return signals


# =========================================================
# SIGNAL RESULT
# =========================================================

def evaluate_signal(signal, df):

    signal_time = signal["Time"]

    future = df[
        df.index > signal_time
    ].copy()

    if future.empty:
        return {
            "T1": "OPEN",
            "T2": "OPEN",
            "Favourable": None,
            "Points": None
        }

    entry = signal["Entry"]
    sl = signal["SL"]
    t1 = signal["T1"]
    t2 = signal["T2"]
    direction = signal["CE_PE"]

    t1_hit = False
    t2_hit = False
    sl_hit = False

    favourable = 0.0

    for _, row in future.iterrows():

        high = float(row["High"])
        low = float(row["Low"])

        if direction == "CE":

            favourable = max(
                favourable,
                high - entry
            )

            hit_t1 = high >= t1
            hit_t2 = high >= t2
            hit_sl = low <= sl

        else:

            favourable = max(
                favourable,
                entry - low
            )

            hit_t1 = low <= t1
            hit_t2 = low <= t2
            hit_sl = high >= sl

        # ----------------------------------------------
        # T1
        # ----------------------------------------------

        if not t1_hit and hit_t1:
            t1_hit = True

        # ----------------------------------------------
        # T2
        # ----------------------------------------------

        if not t2_hit and hit_t2:
            t2_hit = True

        # ----------------------------------------------
        # SL
        # ----------------------------------------------

        if not sl_hit and hit_sl:
            sl_hit = True

        if t2_hit:
            break

    if t1_hit:
        t1_result = "HIT"
    elif sl_hit:
        t1_result = "SL"
    else:
        t1_result = "OPEN"

    if t2_hit:
        t2_result = "HIT"
    elif sl_hit:
        t2_result = "SL"
    else:
        t2_result = "OPEN"

    if direction == "CE":
        points = (
            t2 - entry
            if t2_hit
            else t1 - entry
            if t1_hit
            else - (entry - sl)
            if sl_hit
            else None
        )
    else:
        points = (
            entry - t2
            if t2_hit
            else entry - t1
            if t1_hit
            else - (sl - entry)
            if sl_hit
            else None
        )

    return {
        "T1": t1_result,
        "T2": t2_result,
        "Favourable": round(favourable, 2),
        "Points": (
            round(points, 2)
            if points is not None
            else None
        )
    }


# =========================================================
# SAVE LOG
# =========================================================

def save_signals(symbol_name, signals):

    if not signals:
        return

    rows = []

    for s in signals:

        row = s.copy()

        row["Instrument"] = symbol_name

        row["Time"] = (
            pd.Timestamp(
                row["Time"]
            ).strftime("%Y-%m-%d %H:%M")
        )

        rows.append(row)

    new_df = pd.DataFrame(rows)

    if os.path.exists(LOG_FILE):

        try:
            old = pd.read_csv(LOG_FILE)
            combined = pd.concat(
                [old, new_df],
                ignore_index=True
            )
        except:
            combined = new_df

    else:
        combined = new_df

    # Remove duplicates
    if "Instrument" in combined.columns and "Time" in combined.columns:

        combined = combined.drop_duplicates(
            subset=[
                "Instrument",
                "Time",
                "CE_PE",
                "Setup"
            ],
            keep="last"
        )

    combined.to_csv(
        LOG_FILE,
        index=False
    )


# =========================================================
# MAIN UI
# =========================================================

st.title("⚡ OPEN MOMENTUM SIGNAL ENGINE")

st.caption(
    "Supertrend 7,2 primary trigger • Open signal logic • "
    "5-minute completed candles • Manual trading only"
)

# =========================================================
# CONTROLS
# =========================================================

c1, c2, c3 = st.columns([2, 2, 2])

with c1:
    refresh = st.slider(
        "Auto Refresh (sec)",
        1,
        60,
        REFRESH_DEFAULT
    )

with c2:
    if st.button("🔄 REFRESH NOW", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with c3:
    st.write("")
    st.caption(
        f"Last update: "
        f"{datetime.now().strftime('%H:%M:%S')}"
    )

st.divider()

# =========================================================
# TABS
# =========================================================

tab_nifty, tab_bank = st.tabs(
    ["📈 NIFTY", "🏦 BANK NIFTY"]
)


def render_market(name, symbol):

    df = get_data(symbol)

    if df.empty:

        st.error(
            "Market data available nahi hai."
        )

        return

    df = add_indicators(df)

    # -----------------------------------------------------
    # TODAY
    # -----------------------------------------------------

    today = df.index[-1].date()

    today_df = df[
        df.index.date == today
    ].copy()

    signals = scan_today(df)

    # Save signals
    save_signals(
        name,
        signals
    )

    # -----------------------------------------------------
    # LATEST STATE
    # -----------------------------------------------------

    last = df.iloc[-1]

    current_price = float(last["Close"])
    st_direction = int(last["ST_DIR"])

    if st_direction == 1:
        trend = "BULLISH"
    else:
        trend = "BEARISH"

    # -----------------------------------------------------
    # MOST RECENT SIGNAL
    # -----------------------------------------------------

    latest_signal = (
        signals[-1]
        if signals
        else None
    )

    # =====================================================
    # MAIN DECISION PANEL
    # =====================================================

    st.subheader(
        f"{name} — MAIN DECISION"
    )

    if latest_signal is not None:

        result = evaluate_signal(
            latest_signal,
            df
        )

        box_class = "go"

        st.markdown(
            f"""
            <div class="signal-box {box_class}">
                <div class="big">
                    🟢 {latest_signal["Signal"]}
                    &nbsp;&nbsp;
                    {latest_signal["CE_PE"]}
                    &nbsp;&nbsp;
                    {latest_signal["Quality"]}
                </div>
                <hr>
            </div>
            """,
            unsafe_allow_html=True
        )

        a, b, c, d, e, f = st.columns(6)

        with a:
            st.metric(
                "ENTRY",
                latest_signal["Entry"]
            )

        with b:
            st.metric(
                "SL",
                latest_signal["SL"]
            )

        with c:
            st.metric(
                "TARGET 1",
                latest_signal["T1"]
            )

        with d:
            st.metric(
                "TARGET 2",
                latest_signal["T2"]
            )

        with e:
            st.metric(
                "FAVOURABLE",
                (
                    result["Favourable"]
                    if result["Favourable"] is not None
                    else "-"
                )
            )

        with f:
            st.metric(
                "POINTS",
                (
                    result["Points"]
                    if result["Points"] is not None
                    else "-"
                )
            )

        st.markdown(
            f"""
            <div class="signal-box go">
                <b>SETUP:</b> {latest_signal["Setup"]}<br>
                <b>REASON:</b> {latest_signal["Reason"]}<br>
                <b>SIGNAL TIME:</b>
                {latest_signal["Time"].strftime("%H:%M")}<br>
                <b>SIGNAL VALID:</b>
                Completed 5-min candle<br>
                <b>ATR:</b> {latest_signal["ATR"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="signal-box wait">
                <div class="big">
                    ⚪ NO ACTIONABLE SIGNAL YET
                </div>
                <hr>
                Dashboard market ko continuously scan kar raha hai.
                <br>
                <span class="small">
                Signal ke liye Entry + SL + T1 + T2 available
                hona mandatory hai.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # MARKET STATE
    # =====================================================

    st.subheader("Market State")

    a, b, c, d, e, f = st.columns(6)

    with a:
        st.metric(
            "PRICE",
            round(current_price, 2)
        )

    with b:
        st.metric(
            "SUPERTREND",
            trend
        )

    with c:
        st.metric(
            "ST VALUE",
            round(float(last["ST"]), 2)
        )

    with d:
        st.metric(
            "ATR",
            round(float(last["ATR"]), 2)
        )

    with e:
        st.metric(
            "MACD",
            round(float(last["MACD"]), 2)
        )

    with f:
        st.metric(
            "PIVOT",
            round(float(last["PIVOT"]), 2)
        )

    # =====================================================
    # IMPORTANT LEVELS
    # =====================================================

    st.subheader("Important Levels")

    a, b, c, d, e, f = st.columns(6)

    with a:
        st.metric(
            "Pivot",
            round(float(last["PIVOT"]), 2)
        )

    with b:
        st.metric(
            "R1",
            round(float(last["R1"]), 2)
        )

    with c:
        st.metric(
            "R2",
            round(float(last["R2"]), 2)
        )

    with d:
        st.metric(
            "S1",
            round(float(last["S1"]), 2)
        )

    with e:
        st.metric(
            "S2",
            round(float(last["S2"]), 2)
        )

    with f:
        st.metric(
            "Day High",
            round(float(today_df["High"].max()), 2)
        )

    # =====================================================
    # TODAY'S SIGNALS
    # =====================================================

    st.subheader(
        f"Today's Signals — {len(signals)}"
    )

    if signals:

        display_rows = []

        for s in signals:

            res = evaluate_signal(
                s,
                df
            )

            display_rows.append({
                "Time":
                    s["Time"].strftime("%H:%M"),

                "Setup":
                    s["Setup"],

                "CE/PE":
                    s["CE_PE"],

                "Entry":
                    s["Entry"],

                "SL":
                    s["SL"],

                "T1":
                    s["T1"],

                "T1 Result":
                    res["T1"],

                "T2":
                    s["T2"],

                "T2 Result":
                    res["T2"],

                "Points":
                    res["Points"]
                    if res["Points"] is not None
                    else "-",

                "Favourable":
                    res["Favourable"]
                    if res["Favourable"] is not None
                    else "-"
            })

        log_df = pd.DataFrame(
            display_rows
        )

        st.dataframe(
            log_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Aaj abhi koi actionable signal record nahi hua."
        )

    # =====================================================
    # RECENT CANDLES / ENGINE OBSERVATION
    # =====================================================

    st.subheader(
        "Engine Observation — Last 8 Completed Candles"
    )

    obs = today_df.tail(8).copy()

    obs_display = pd.DataFrame({
        "Time":
            obs.index.strftime("%H:%M"),

        "Close":
            obs["Close"].round(2),

        "ST":
            np.where(
                obs["ST_DIR"] == 1,
                "BULLISH",
                "BEARISH"
            ),

        "MACD":
            obs["MACD"].round(2),

        "Hist":
            obs["MACD_HIST"].round(2),

        "ATR":
            obs["ATR"].round(2)
    })

    st.dataframe(
        obs_display,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # STATUS
    # =====================================================

    st.caption(
        f"Completed candles scanned today: "
        f"{len(today_df)} | "
        f"Signals found: {len(signals)} | "
        f"Data: yfinance 5-minute"
    )


# =========================================================
# RENDER TABS
# =========================================================

with tab_nifty:
    render_market(
        "NIFTY",
        SYMBOLS["NIFTY"]
    )

with tab_bank:
    render_market(
        "BANK NIFTY",
        SYMBOLS["BANK NIFTY"]
    )


# =========================================================
# AUTO REFRESH
# =========================================================

time.sleep(refresh)
st.cache_data.clear()
st.rerun()
