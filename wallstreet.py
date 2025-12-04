import streamlit as st
import requests
import pandas as pd
import numpy as np

# ============ Streamlit 基本设置 ============
st.set_page_config(
    page_title="主流币短线波动多空终端（多周期·信号打分）",
    layout="wide"
)

st.title("主流币 1–2 天短线波动多空终端 · 多周期信号打分版")
st.caption("仅供量化研究与教学使用，不构成任何投资建议。请理性使用杠杆。")

BASE_URL = "https://www.okx.com"


# ============ 工具函数 & 技术指标 ============

@st.cache_data(show_spinner=False)
def fetch_okx_candles(inst_id: str, bar: str = "1H", limit: int = 500) -> pd.DataFrame:
    """
    从 OKX 获取 K 线数据
    inst_id: 'BTC-USDT-SWAP' 等
    bar: '1H','4H','1D','15m'...
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

    # 风格因子
    df["trend_strength"] = (df["ema_fast"] - df["ema_slow"]).abs() / (df["atr"] + 1e-9)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-9)

    return df


# ============ Regime 识别 & 信号生成 ============

def classify_regime(row: pd.Series) -> str:
    """
    市场状态：
    - 'trend'          : 趋势市
    - 'squeeze'        : 压缩待爆发
    - 'mean_reversion' : 震荡均值回归
    """
    if (
        np.isnan(row.get("atr", np.nan)) or row["atr"] <= 0
        or np.isnan(row.get("trend_strength", np.nan))
        or np.isnan(row.get("bb_width", np.nan))
        or np.isnan(row.get("bb_mid", np.nan)) or row["bb_mid"] <= 0
    ):
        return "unknown"

    ts = row["trend_strength"]
    bbw = row["bb_width"]

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
    基于 1H 生成整段历史信号 & 最新信号
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
                tp = entry + 2 * (entry - sl)
                max_hold = max_hold_trend

            # 空头突破
            elif (
                entry < low_lookback
                and entry < row["ema_fast"]
                and row["rsi"] < 45
            ):
                side = "short"
                sig_type = "breakout_trend"
                reason = "价格跌破近24根低点 + 趋势向下 + RSI偏弱"
                sl_mult = 1.8 if regime == "trend" else 1.5
                sl = entry + sl_mult * atr_val
                tp = entry - 2 * (sl - entry)
                max_hold = max_hold_trend

        # ===== 震荡：均值回归策略 =====
        if side == "flat" and regime == "mean_reversion":
            prev_row = df.iloc[i - 1]
            ret_1 = (entry - prev_row["close"]) / (prev_row["close"] + 1e-9)

            # 超跌反弹
            if (
                entry < row["bb_lower"]
                and row["rsi"] < 30
                and ret_1 < -1.0 * atr_val / (prev_row["close"] + 1e-9)
            ):
                side = "long"
                sig_type = "mean_reversion"
                reason = "价格击穿布林下轨 + RSI超卖 + 急跌，博反弹"
                sl = entry - 1.2 * atr_val
                tp = entry + 1.8 * atr_val
                max_hold = max_hold_meanrev

            # 超涨回落
            elif (
                entry > row["bb_upper"]
                and row["rsi"] > 70
                and ret_1 > 1.0 * atr_val / (prev_row["close"] + 1e-9)
            ):
                side = "short"
                sig_type = "mean_reversion"
                reason = "价格突破布林上轨 + RSI超买 + 急涨，博回调"
                sl = entry + 1.2 * atr_val
                tp = entry - 1.8 * atr_val
                max_hold = max_hold_meanrev

        signals.at[idx, "side"] = side
        signals.at[idx, "signal_type"] = sig_type
        signals.at[idx, "reason"] = reason
        signals.at[idx, "entry_price"] = entry
        signals.at[idx, "sl"] = sl
        signals.at[idx, "tp"] = tp
        signals.at[idx, "max_hold_bars"] = max_hold

    latest_idx = df.index[-1]
    latest_row = signals.loc[latest_idx].to_dict()
    return signals, latest_row


# ============ 回测 & 统计 ============

def backtest_short_term(df: pd.DataFrame, signals_df: pd.DataFrame):
    trades = []
    in_pos = False
    direction = None
    entry_price = None
    sl = None
    tp = None
    entry_idx = None
    max_hold = None
    bars_held = 0
    signal_type = None
    entry_regime = None

    idx_list = df.index
    n = len(idx_list)

    for i in range(n):
        t = idx_list[i]
        row = df.iloc[i]
        sig = signals_df.loc[t]

        if not in_pos:
            side = sig.get("side", "flat")
            if side in ["long", "short"]:
                if np.isnan(sig["sl"]) or np.isnan(sig["tp"]):
                    continue
                in_pos = True
                direction = side
                entry_price = sig["entry_price"]
                sl = sig["sl"]
                tp = sig["tp"]
                entry_idx = t
                max_hold = int(sig["max_hold_bars"])
                bars_held = 0
                signal_type = sig.get("signal_type", "unknown")
                entry_regime = sig.get("regime", "unknown")
        else:
            bars_held += 1
            high = row["high"]
            low = row["low"]
            exit_reason = None
            exit_price = None

            if direction == "long":
                hit_sl = low <= sl
                hit_tp = high >= tp
            else:
                hit_sl = high >= sl
                hit_tp = low <= tp

            if hit_sl:
                exit_price = sl
                exit_reason = "SL"
            elif hit_tp:
                exit_price = tp
                exit_reason = "TP"
            elif bars_held >= max_hold:
                exit_price = row["close"]
                exit_reason = "TIME"

            if exit_reason is not None:
                risk = abs(entry_price - sl)
                if risk == 0:
                    r = 0.0
                else:
                    if direction == "long":
                        pnl = exit_price - entry_price
                    else:
                        pnl = entry_price - exit_price
                    r = pnl / risk

                trades.append({
                    "entry_time": entry_idx,
                    "exit_time": t,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "r": r,
                    "exit_reason": exit_reason,
                    "holding_bars": bars_held,
                    "signal_type": signal_type,
                    "entry_regime": entry_regime
                })
                in_pos = False
                direction = None

    if not trades:
        trades_df = pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "r",
            "exit_reason", "holding_bars",
            "signal_type", "entry_regime"
        ])
    else:
        trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        equity_curve = None
    else:
        eq = (1 + trades_df["r"] * 0.01).cumprod()
        equity_curve = pd.Series(eq.values, index=trades_df["exit_time"])

    return trades_df, equity_curve


def summarize_trades(trades_df: pd.DataFrame):
    if trades_df.empty:
        return {}
    total = len(trades_df)
    wins = (trades_df["r"] > 0).sum()
    losses = (trades_df["r"] < 0).sum()
    win_rate = wins / total if total > 0 else 0.0
    avg_r = trades_df["r"].mean()
    avg_win = trades_df.loc[trades_df["r"] > 0, "r"].mean() if wins > 0 else 0.0
    avg_loss = trades_df.loc[trades_df["r"] < 0, "r"].mean() if losses > 0 else 0.0

    eq = 1 + trades_df["r"].cumsum()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    return {
        "total_trades": int(total),
        "win_rate": float(win_rate),
        "avg_r": float(avg_r),
        "avg_win_r": float(avg_win),
        "avg_loss_r": float(avg_loss),
        "max_drawdown": float(max_dd)
    }


def summarize_trades_by_type(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=[
            "signal_type", "total_trades", "win_rate",
            "avg_r", "avg_win_r", "avg_loss_r", "max_drawdown"
        ])

    rows = []
    for sig_type, sub in trades_df.groupby("signal_type"):
        stats = summarize_trades(sub)
        rows.append({"signal_type": sig_type, **stats})
    return pd.DataFrame(rows)


# ============ 资产级打分（0–100） ============

def compute_asset_score(stats: dict, trades_df: pd.DataFrame) -> float:
    if not stats or stats.get("total_trades", 0) == 0:
        return 0.0

    total = stats["total_trades"]
    win_rate = stats["win_rate"]
    avg_r = stats["avg_r"]
    max_dd = stats["max_drawdown"]

    if len(trades_df) >= 5:
        recent = trades_df.tail(20)
        recent_avg_r = recent["r"].mean()
    else:
        recent_avg_r = 0.0

    win_score = win_rate
    r_score = 0.5 + 0.5 * np.tanh(avg_r / 1.0)
    recent_score = 0.5 + 0.5 * np.tanh(recent_avg_r / 1.0)
    dd_score = 0.5 + 0.5 * np.tanh(-max_dd / 0.3)

    min_trades = 20
    size_factor = min(1.0, total / min_trades)

    raw = (
        0.4 * win_score +
        0.25 * r_score +
        0.2 * recent_score +
        0.15 * dd_score
    )
    score_0_1 = raw * size_factor
    return float(score_0_1 * 100)


# ============ 仓位建议 & 多周期微观趋势 ============

def position_sizing(
    capital_usdt: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
):
    if capital_usdt <= 0 or entry_price <= 0 or np.isnan(stop_price):
        return 0.0, 0.0
    risk_amt = capital_usdt * risk_pct
    stop_dist = abs(entry_price - stop_price)
    if stop_dist <= 0:
        return 0.0, 0.0
    qty = risk_amt / stop_dist
    notional = qty * entry_price
    return qty, notional


def compute_micro_trend(df_15m: pd.DataFrame):
    """
    基于 15m 周期给一个“微观趋势”判断:
    - 上涨 micro_trend: up
    - 下跌 micro_trend: down
    - neutral: 中性
    并返回一个简单得分（-2~+2 左右）
    """
    if df_15m is None or df_15m.empty:
        return "unknown", 0.0

    row = df_15m.iloc[-1]
    score = 0.0

    if not np.isnan(row.get("ema_fast", np.nan)) and not np.isnan(row.get("ema_slow", np.nan)):
        if row["ema_fast"] > row["ema_slow"]:
            score += 1.0
        elif row["ema_fast"] < row["ema_slow"]:
            score -= 1.0

    if not np.isnan(row.get("rsi", np.nan)):
        if row["rsi"] > 55:
            score += 1.0
        elif row["rsi"] < 45:
            score -= 1.0

    if not np.isnan(row.get("bb_mid", np.nan)) and row["bb_mid"] > 0:
        if row["close"] > row["bb_mid"]:
            score += 0.5
        elif row["close"] < row["bb_mid"]:
            score -= 0.5

    if score > 0.5:
        return "up", score
    elif score < -0.5:
        return "down", score
    else:
        return "neutral", score


def compute_signal_quality(row_dict: dict) -> float:
    """
    0-100 信号质量分：
    - 策略类型 + Regime -> 基础分
    - 资产历史打分 -> 加减分
    - 15m 微观趋势是否与 1H 方向共振 -> 加减分
    """
    side = row_dict.get("side", "flat")
    if side not in ["long", "short"]:
        return 0.0

    signal_type = row_dict.get("signal_type", "none")
    regime = row_dict.get("latest_regime", "unknown")
    asset_score = row_dict.get("asset_score", 50.0)
    micro_dir = row_dict.get("micro_trend_dir", "neutral")
    micro_score = row_dict.get("micro_trend_score", 0.0)

    # 1）基础分：趋势突破 > 均值回归
    if signal_type == "breakout_trend" and regime in ["trend", "squeeze"]:
        base = 70.0
    elif signal_type == "breakout_trend":
        base = 60.0
    elif signal_type == "mean_reversion":
        base = 55.0
    else:
        base = 50.0

    # 2）资产历史评分（以 50 为中性）
    asset_term = 0.3 * (asset_score - 50.0)  # ±15 区间

    # 3）15m 微观趋势共振
    micro_term = 5 * np.tanh(micro_score)  # -5 ~ 5 附近
    # 若方向与微观趋势明显反向，额外扣分
    if side == "long" and micro_dir == "down":
        micro_term -= 10
    if side == "short" and micro_dir == "up":
        micro_term -= 10

    raw = base + asset_term + micro_term
    return float(np.clip(raw, 0.0, 100.0))


# ============ Sidebar 参数 ============

st.sidebar.header("参数设置（组合 & 风险）")

DEFAULT_UNIVERSE = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "XRP-USDT-SWAP",
    "BNB-USDT-SWAP",
    "LTC-USDT-SWAP",
    "LINK-USDT-SWAP",
]

universe = st.sidebar.multiselect(
    "交易标的（OKX 永续合约 instId）",
    options=DEFAULT_UNIVERSE,
    default=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
)

capital = st.sidebar.number_input(
    "账户资金（USDT）", min_value=100.0, value=10000.0, step=100.0
)
single_risk_pct = st.sidebar.slider(
    "单笔风险占资金比例（%）", min_value=0.2, max_value=3.0, value=1.0, step=0.1
) / 100.0
portfolio_risk_pct = st.sidebar.slider(
    "组合层面同时承受的总风险上限（%）",
    min_value=0.5, max_value=5.0, value=2.0, step=0.5
) / 100.0

recent_n = st.sidebar.slider(
    "最近 N 笔信号用于盈亏直方图", min_value=10, max_value=200, value=50, step=10
)

st.sidebar.markdown("---")
st.sidebar.caption("数据来源：OKX 公共API；本工具不保证数据完整与实时，实盘前请务必自测。")

if not universe:
    st.warning("请在左侧选择至少一个交易标的。")
    st.stop()


# ============ 主逻辑：多周期 + 回测 + 打分 ============

st.subheader("一、跨品种多周期信号 & 资产级打分（0–100）")

rows = []

for inst in universe:
    with st.spinner(f"获取 {inst} 1H & 15m K 线数据并回测策略..."):
        try:
            df_1h = fetch_okx_candles(inst, "1H", limit=500)
        except Exception as e:
            st.error(f"{inst} 1H 数据获取失败: {e}")
            continue

        try:
            df_15m = fetch_okx_candles(inst, "15m", limit=200)
        except Exception:
            df_15m = None

    df_1h_ind = add_indicators(df_1h)
    signals_df, latest_sig = gen_short_term_signal(df_1h_ind)

    regimes = [classify_regime(r) for _, r in df_1h_ind.iterrows()]
    df_1h_ind["regime"] = regimes

    trades_df, equity_curve = backtest_short_term(df_1h_ind, signals_df)
    stats = summarize_trades(trades_df)
    asset_score = compute_asset_score(stats, trades_df)

    latest_price = df_1h_ind["close"].iloc[-1]
    latest_regime = df_1h_ind["regime"].iloc[-1]

    # 当前信号 & 仓位建议（单笔风险）
    side = latest_sig.get("side", "flat")
    entry = latest_sig.get("entry_price", np.nan)
    sl = latest_sig.get("sl", np.nan)
    tp = latest_sig.get("tp", np.nan)
    qty, notion = position_sizing(
        capital_usdt=capital,
        risk_pct=single_risk_pct,
        entry_price=entry if not np.isnan(entry) else latest_price,
        stop_price=sl,
    )

    # 15m 微观趋势
    if df_15m is not None and not df_15m.empty:
        df_15m_ind = add_indicators(df_15m)
        micro_trend_dir, micro_trend_score = compute_micro_trend(df_15m_ind)
    else:
        micro_trend_dir, micro_trend_score = "unknown", 0.0

    # 信号优先级（粗）：趋势突破 > 均值回归 > 无
    if side in ["long", "short"]:
        if latest_sig.get("signal_type") == "breakout_trend" and latest_regime in [
            "trend",
            "squeeze",
        ]:
            base_priority = 2
        elif latest_sig.get("signal_type") == "mean_reversion":
            base_priority = 1
        else:
            base_priority = 0
    else:
        base_priority = -1

    row = {
        "inst": inst,
        "price": latest_price,
        "latest_regime": latest_regime,
        "side": side,
        "signal_type": latest_sig.get("signal_type", "none"),
        "reason": latest_sig.get("reason", ""),
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "suggest_qty_single": qty,
        "suggest_notional_single": notion,
        "signal_priority": base_priority,
        "asset_score": asset_score,
        "stats": stats,
        "trades_df": trades_df,
        "equity_curve": equity_curve,
        "df_1h": df_1h_ind,
        "signals_df": signals_df,
        "micro_trend_dir": micro_trend_dir,
        "micro_trend_score": micro_trend_score,
    }
    rows.append(row)

if not rows:
    st.error("所有标的的数据都获取失败，请检查网络或 instId 是否正确。")
    st.stop()

# === 计算信号质量分（0-100），再做组合分配 ===
for r in rows:
    r["signal_quality"] = compute_signal_quality(r)

signaled = [r for r in rows if r["side"] in ["long", "short"]]
signaled_sorted = sorted(signaled, key=lambda x: x["signal_quality"], reverse=True)

if signaled_sorted:
    max_positions = min(3, len(signaled_sorted))
    top_assets = signaled_sorted[:max_positions]
    per_position_risk_pct = portfolio_risk_pct / max_positions
else:
    top_assets = []
    per_position_risk_pct = 0.0

# 用组合风险重算“组合建议仓位”
for r in rows:
    if r in top_assets and r["side"] in ["long", "short"]:
        entry = r["entry"]
        sl = r["sl"]
        qty, notion = position_sizing(
            capital_usdt=capital,
            risk_pct=per_position_risk_pct,
            entry_price=entry if not np.isnan(entry) else r["price"],
            stop_price=sl,
        )
        r["portfolio_qty"] = qty
        r["portfolio_notional"] = notion
    else:
        r["portfolio_qty"] = 0.0
        r["portfolio_notional"] = 0.0

# === 展示：跨品种多周期概览 ===
display_rows = []
for r in rows:
    side_zh = {"long": "做多", "short": "做空", "flat": "观望"}.get(r["side"], "观望")
    direction_en = r["side"] if r["side"] in ["long", "short"] else "flat"

    regime_zh = {
        "trend": "趋势市",
        "squeeze": "压缩待爆发",
        "mean_reversion": "震荡均值回归",
        "unknown": "待定",
    }.get(r["latest_regime"], "待定")

    s = r["stats"]
    if s:
        win_str = f"{s['win_rate']*100:.1f}%"
        avg_r_str = f"{s['avg_r']:.2f}"
        dd_str = f"{s['max_drawdown']*100:.1f}%"
        total_trades = s["total_trades"]
    else:
        win_str = avg_r_str = dd_str = ""
        total_trades = 0

    micro_zh = {
        "up": "15m 上涨",
        "down": "15m 下跌",
        "neutral": "15m 中性",
        "unknown": "15m 未知",
    }.get(r["micro_trend_dir"], "15m 未知")

    display_rows.append(
        {
            "品种": r["inst"],
            "最新价格": f"{r['price']:.2f}",
            "市场状态(1H)": regime_zh,
            "多空方向": side_zh,
            "Direction_EN": direction_en,   # 明确: long / short / flat
            "信号类型": {
                "breakout_trend": "趋势突破",
                "mean_reversion": "均值回归",
                "none": "无",
            }.get(r["signal_type"], "无"),
            "15m 微观趋势": micro_zh,
            "信号质量(0-100)": f"{r['signal_quality']:.1f}",
            "资产历史评分(0-100)": f"{r['asset_score']:.1f}",
            "历史交易数": total_trades,
            "历史胜率": win_str,
            "平均R": avg_r_str,
            "最大回撤(≈)": dd_str,
            "组合建议币数": f"{r['portfolio_qty']:.4f}"
            if r["portfolio_qty"] > 0
            else "",
            "组合名义价值USDT": f"{r['portfolio_notional']:.2f}"
            if r["portfolio_notional"] > 0
            else "",
        }
    )

summary_df = pd.DataFrame(display_rows).sort_values(
    "信号质量(0-100)", ascending=False
)
st.dataframe(summary_df, use_container_width=True)

st.markdown(
    """
**解读要点：**

- `多空方向`：中文语义；  
- `Direction_EN`：明确给出 `long / short / flat`，方便程序化对接；  
- `信号质量(0-100)` =
  - 策略类型（趋势突破 > 均值回归）；  
  - 该币在本策略下的历史表现（资产评分）；  
  - 15m 与 1H 的多周期共振（同向加分，反向减分）；  
- `组合建议币数`：已经考虑组合总风险上限（例如 2%），只把风险分配给**质量最高的前 1–3 个信号**。
"""
)

# ============ 二、单一标的：策略拆分 + 回测细节 ============

st.subheader("二、单一标的拆解：趋势 vs 均值回归 + 资金曲线")

inst_options = [r["inst"] for r in rows]
chosen_inst = st.selectbox("选择查看详细回测的标的", options=inst_options)

chosen = next(r for r in rows if r["inst"] == chosen_inst)
df_chosen = chosen["df_1h"]
signals_chosen = chosen["signals_df"]
trades_chosen = chosen["trades_df"]
equity_chosen = chosen["equity_curve"]

stats = summarize_trades(trades_chosen)
stats_type = summarize_trades_by_type(trades_chosen)

col1, col2, col3, col4, col5 = st.columns(5)
if stats:
    col1.metric("总交易次数", stats["total_trades"])
    col2.metric("整体胜率", f"{stats['win_rate']*100:.1f}%")
    col3.metric("整体平均R", f"{stats['avg_r']:.2f}")
    col4.metric("整体平均盈利R", f"{stats['avg_win_r']:.2f}")
    col5.metric("整体最大回撤(≈)", f"{stats['max_drawdown']*100:.1f}%")
else:
    st.info("当前策略参数下暂无交易记录。")

st.markdown("**按策略类型拆分表现：**")
if not stats_type.empty:
    mapping = {
        "breakout_trend": "趋势突破",
        "mean_reversion": "均值回归",
        "none": "无信号",
        "unknown": "其他",
    }
    stats_type["策略类型"] = stats_type["signal_type"].map(mapping).fillna("其他")
    display_type = stats_type[
        [
            "策略类型",
            "total_trades",
            "win_rate",
            "avg_r",
            "avg_win_r",
            "max_drawdown",
        ]
    ].copy()
    display_type.columns = ["策略类型", "交易数", "胜率", "平均R", "平均盈利R", "最大回撤(≈)"]
    display_type["胜率"] = display_type["胜率"].apply(lambda x: f"{x*100:.1f}%")
    display_type["平均R"] = display_type["平均R"].map(lambda x: f"{x:.2f}")
    display_type["平均盈利R"] = display_type["平均盈利R"].map(lambda x: f"{x:.2f}")
    display_type["最大回撤(≈)"] = display_type["最大回撤(≈)"].map(lambda x: f"{x*100:.1f}%")
    st.dataframe(display_type, use_container_width=True)
else:
    st.info("尚无足够交易数据拆分到具体策略类型。")

st.markdown("**机械执行净值曲线（假设每笔风险≈1%资金）：**")
if equity_chosen is not None and not equity_chosen.empty:
    st.line_chart(equity_chosen)
else:
    st.info("暂无足够交易记录绘制资金曲线。")

st.markdown(f"**最近 {min(recent_n, len(trades_chosen))} 笔信号的盈亏分布（单位：R）**")
if not trades_chosen.empty:
    recent_trades = trades_chosen.tail(recent_n)
    st.bar_chart(recent_trades["r"])
    st.write("最近部分交易明细：")
    st.dataframe(recent_trades.tail(30), use_container_width=True)
else:
    st.info("暂无交易记录，无法绘制盈亏分布。")

# ============ 三、1H 风格剖面 ============

st.subheader("三、1H 风格剖面：趋势 / 压缩 / 震荡 占比")

regime_counts = chosen["df_1h"]["regime"].value_counts(normalize=True)
for regime in ["trend", "squeeze", "mean_reversion", "unknown"]:
    if regime not in regime_counts:
        regime_counts[regime] = 0.0
regime_counts = regime_counts[["trend", "squeeze", "mean_reversion", "unknown"]]

regime_zh_map = {
    "trend": "趋势市",
    "squeeze": "压缩待爆发",
    "mean_reversion": "震荡均值回归",
    "unknown": "待定",
}

style_df = pd.DataFrame(
    {
        "市场状态": [regime_zh_map[k] for k in regime_counts.index],
        "历史占比": [f"{v*100:.1f}%" for v in regime_counts.values],
    }
)
st.dataframe(style_df, use_container_width=True)

latest_row = chosen["df_1h"].iloc[-1]
st.markdown(
    f"""
**当前1H K线的关键因子：**

- 趋势强度 trend_strength：{latest_row['trend_strength']:.2f}  
- 布林带宽度 bb_width：{latest_row['bb_width']:.4f}  
- RSI(14)：{latest_row['rsi']:.1f}  
- ATR(14)：{latest_row['atr']:.2f}  
- 当前1H市场状态：**{regime_zh_map.get(latest_row['regime'], '待定')}**
"""
)

st.markdown(
    """
从交易风格角度来说：

- 趋势市占比较高 + 趋势突破策略表现好 → 这个币适合“顺势波段”；  
- 震荡市占比较高 + 均值回归策略表现更优 → 适合“短线高抛低吸”；  
- 你完全可以据此做“择币 + 择时”：  
  - 在今日信号质量最高、且风格匹配你的策略偏好的那几个标的上集中火力。
"""
)

# ============ 风险提示 ============

st.warning(
    """
⚠️ 风险提示：

- 本工具只是把“多周期 + 策略拆分 + 回测 + 组合风险”做了系统化呈现，
  并不能消灭风险，只是帮你**有意识地承担风险**；
- `Direction_EN` / `多空方向` 只是模型信号，不是保证金催收单；
- 真正决定收益的，是你是否能在连续亏损时依然保持纪律，
  以及在收益上来时，是否懂得控制贪婪、分批止盈。
"""
)
