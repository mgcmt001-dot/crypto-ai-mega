import streamlit as st
import requests
import pandas as pd
import numpy as np

# ============ Streamlit 基本设置 ============
st.set_page_config(
    page_title="主流币短线波动多空终端（升级版）",
    layout="wide"
)

st.title("主流币 1–2 天短线波动多空终端（OKX · 升级版）")
st.caption("仅供量化研究与教学使用，不构成任何投资建议。请理性使用杠杆。")

BASE_URL = "https://www.okx.com"


# ============ 数据 & 技术指标函数 ============

@st.cache_data(show_spinner=False)
def fetch_okx_candles(inst_id: str, bar: str = "1H", limit: int = 500) -> pd.DataFrame:
    """
    从 OKX 获取 K 线数据
    inst_id: 'BTC-USDT-SWAP' 等
    bar: '1H','4H','1D'...
    """
    url = f"{BASE_URL}/api/v5/market/candles"
    params = {"instId": inst_id, "bar": bar, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "0":
        raise ValueError(f"OKX API error: {data.get('msg')}")
    raw = data.get("data", [])

    df = pd.DataFrame(
        raw,
        columns=[
            "ts", "open", "high", "low", "close", "vol",
            "volCcy", "volCcyQuote", "confirm"
        ]
    )
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("ts").reset_index(drop=True).set_index("ts")
    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr_val


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    df["ema_fast"] = ema(close, 20)
    df["ema_slow"] = ema(close, 60)
    df["rsi"] = rsi(close, 14)
    df["atr"] = atr(df, 14)
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger_bands(close, 20, 2.0)

    # 为风格剖面预先计算两个因子
    df["trend_strength"] = (
        (df["ema_fast"] - df["ema_slow"]).abs() / (df["atr"] + 1e-9)
    )
    df["bb_width"] = (
        (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-9)
    )

    return df


# ============ 市场状态识别 & 信号生成 ============

def classify_regime(row: pd.Series) -> str:
    """
    基于单根K线的指标，判断市场状态：
    - 'trend'          : 趋势市
    - 'squeeze'        : 压缩待爆发
    - 'mean_reversion' : 震荡均值回归
    """
    if (
        np.isnan(row["atr"]) or row["atr"] <= 0
        or np.isnan(row["trend_strength"])
        or np.isnan(row["bb_width"]) or row["bb_mid"] <= 0
    ):
        return "unknown"

    ts = row["trend_strength"]
    bbw = row["bb_width"]

    # 这些阈值是经验值，可按回测结果微调
    if bbw < 0.02:
        return "squeeze"
    elif ts > 1.5 and bbw > 0.02:
        return "trend"
    else:
        return "mean_reversion"


def gen_short_term_signal(
    df: pd.DataFrame,
    lookback_breakout: int = 24,
    max_hold_trend: int = 48,
    max_hold_meanrev: int = 24,
):
    """
    对整段历史生成信号（用于回测），并对最新一根给出当前建议。
    返回：
      signals_df: 每根K线的信号信息
      latest_signal: 最新一根K线的信号 dict
    """
    df = df.copy()
    n = len(df)
    cols = [
        "regime", "side", "signal_type", "reason",
        "entry_price", "sl", "tp", "max_hold_bars"
    ]
    signals = pd.DataFrame(index=df.index, columns=cols)
    signals.iloc[:] = np.nan

    for i in range(lookback_breakout, n):
        row = df.iloc[i]
        idx = df.index[i]
        regime = classify_regime(row)

        signals.at[idx, "regime"] = regime

        hist = df.iloc[i - lookback_breakout:i]
        high_lookback = hist["high"].max()
        low_lookback = hist["low"].min()

        side = "flat"
        sig_type = "none"
        reason = ""
        entry = row["close"]
        atr_val = row["atr"]
        sl = np.nan
        tp = np.nan
        max_hold = np.nan

        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(row["rsi"]):
            signals.at[idx, "side"] = "flat"
            signals.at[idx, "signal_type"] = "none"
            signals.at[idx, "reason"] = "指标不完整，自动观望"
            continue

        # ===== 趋势 / 压缩：趋势突破策略 =====
        if regime in ["trend", "squeeze"]:
            # 多头突破
            if (
                entry > high_lookback
                and entry > row["ema_fast"]
                and row["rsi"] > 55
            ):
                side = "long"
                sig_type = "breakout_trend"
                reason = "价格突破近24根高点 + 趋势向上 + RSI偏强"
                sl_mult = 1.8 if regime == "trend" else 1.5
                sl = entry - sl_mult * atr_val
                tp = entry + 2 * (entry - sl)  # R:R=1:2
                max_hold = max_hold_trend

            # 空头突破
            elif (
                entry < low_lookback
               
