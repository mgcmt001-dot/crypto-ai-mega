import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ============ Streamlit 基本设置 ============
st.set_page_config(
    page_title="Crypto 短线波动多空终端",
    layout="wide"
)

st.title("主流币 1–2 天短线波动策略终端（OKX）")
st.caption("仅供量化研究与教学使用，不构成任何投资建议。请理性使用杠杆。")

BASE_URL = "https://www.okx.com"

# ============ 工具函数 ============

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
    return df


def classify_regime(row: pd.Series) -> str:
    """
    基于单根K线的指标，判断市场状态：
    - 'trend'          : 趋势市
    - 'squeeze'        : 压缩待爆发
    - 'mean_reversion' : 震荡均值回归
    """
    if np.isnan(row["atr"]) or row["atr"] <= 0 \
       or np.isnan(row["ema_fast"]) or np.isnan(row["ema_slow"]) \
       or np.isnan(row["bb_mid"]) or row["bb_mid"] <= 0 \
       or np.isnan(row["bb_upper"]) or np.isnan(row["bb_lower"]):
        return "unknown"

    ts = abs(row["ema_fast"] - row["ema_slow"]) / (row["atr"] + 1e-9)
    bbw = (row["bb_upper"] - row["bb_lower"]) / (row["bb_mid"] + 1e-9)

    # 这些阈值是经验值，可按回测结果微调
    if bbw < 0.02:
        return "squeeze"
    elif ts > 1.5 and bbw > 0.02:
        return "trend"
    else:
        return "mean_reversion"


def gen_short_term_signal(df: pd.DataFrame,
                          lookback_breakout: int = 24,
                          max_hold_trend: int = 48,
                          max_hold_meanrev: int = 24):
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
        "entry_price", "sl", "tp",
        "max_hold_bars"
    ]
    signals = pd.DataFrame(index=df.index, columns=cols)
    signals.iloc[:] = np.nan

    for i in range(lookback_breakout, n):
        row = df.iloc[i]
        idx = df.index[i]
        regime = classify_regime(row)

        signals.at[idx, "regime"] = regime

        # 历史窗口（不含当前）
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
            # 指标不完整，直接观望
            signals.at[idx, "side"] = "flat"
            continue

        # ===== 趋势 / 压缩：突破顺势 =====
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

        # ===== 震荡：均值回归 =====
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

    # 最新信号
    latest_idx = df.index[-1]
    latest_row = signals.loc[latest_idx].to_dict()
    return signals, latest_row


def backtest_short_term(df: pd.DataFrame, signals_df: pd.DataFrame):
    """
    基于 gen_short_term_signal 生成的信号做简单回测。
    - 在信号出现的那根K线收盘价开仓
    - 后续K线内检查 high/low 是否触及止盈/止损/超时
    - 单位为 R（止损距离为1R）
    """
    trades = []
    in_pos = False
    direction = None
    entry_price = None
    sl = None
    tp = None
    entry_idx = None
    max_hold = None
    bars_held = 0

    idx_list = df.index
    n = len(df)

    for i in range(len(idx_list)):
        t = idx_list[i]
        row = df.iloc[i]
        sig = signals_df.loc[t]

        if not in_pos:
            # 看是否有新信号
            if sig.get("side", "flat") in ["long", "short"]:
                if np.isnan(sig["sl"]) or np.isnan(sig["tp"]):
                    continue
                in_pos = True
                direction = sig["side"]
                entry_price = sig["entry_price"]
                sl = sig["sl"]
                tp = sig["tp"]
                entry_idx = t
                max_hold = int(sig["max_hold_bars"])
                bars_held = 0
        else:
            # 持仓中：从下一根开始检查止盈止损
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
                    "holding_bars": bars_held
                })
                in_pos = False
                direction = None

    if not trades:
        trades_df = pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "r",
            "exit_reason", "holding_bars"
        ])
    else:
        trades_df = pd.DataFrame(trades)

    # 资金曲线（假设每笔风险=资金1%，R近似为1%）
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


def position_sizing(capital_usdt: float,
                    risk_pct: float,
                    entry_price: float,
                    stop_price: float):
    """
    现货/单向合约的建议币数
    """
    if capital_usdt <= 0 or entry_price <= 0 or np.isnan(stop_price):
        return 0.0, 0.0
    risk_amt = capital_usdt * risk_pct
    stop_dist = abs(entry_price - stop_price)
    if stop_dist <= 0:
        return 0.0, 0.0
    qty = risk_amt / stop_dist
    notional = qty * entry_price
    return qty, notional


# ============ Sidebar 参数 ============

st.sidebar.header("参数设置")

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
    "选择交易标的（OKX 永续合约 instId）",
    options=DEFAULT_UNIVERSE,
    default=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
)

capital = st.sidebar.number_input(
    "账户资金（USDT）", min_value=100.0, value=10000.0, step=100.0
)
risk_pct = st.sidebar.slider(
    "每笔风险占资金比例（%）", min_value=0.2, max_value=3.0, value=1.0, step=0.1
) / 100.0

recent_n = st.sidebar.slider(
    "最近 N 笔信号用于盈亏直方图", min_value=10, max_value=200, value=50, step=10
)

st.sidebar.markdown("---")
st.sidebar.caption("数据来源：OKX 公共API\n本工具不保证数据完整与实时，实盘前请务必自测。")

if not universe:
    st.warning("请在左侧选择至少一个交易标的。")
    st.stop()

# ============ 主逻辑：批量获取数据 & 生成信号 ============

st.subheader("一、当前短线信号总览")

rows = []

for inst in universe:
    with st.spinner(f"获取 {inst} 1H K 线数据..."):
        try:
            df = fetch_okx_candles(inst, "1H", limit=500)
        except Exception as e:
            st.error(f"{inst} 数据获取失败: {e}")
            continue

    df_ind = add_indicators(df)
    signals_df, latest_sig = gen_short_term_signal(df_ind)

    latest_price = df_ind["close"].iloc[-1]
    latest_regime = classify_regime(df_ind.iloc[-1])

    # 仓位建议
    side = latest_sig.get("side", "flat")
    entry = latest_sig.get("entry_price", np.nan)
    sl = latest_sig.get("sl", np.nan)
    tp = latest_sig.get("tp", np.nan)
    qty, notion = position_sizing(
        capital_usdt=capital,
        risk_pct=risk_pct,
        entry_price=entry if not np.isnan(entry) else latest_price,
        stop_price=sl
    )

    # 简单评分：根据 regime 和信号类型，给一个直观评级
    if side in ["long", "short"]:
        if latest_sig.get("signal_type") == "breakout_trend" and latest_regime in ["trend", "squeeze"]:
            edge_score = 2  # 趋势突破，优先级高
        elif latest_sig.get("signal_type") == "mean_reversion":
            edge_score = 1  # 均值回归，次优先
        else:
            edge_score = 0
    else:
        edge_score = -1  # 无信号

    rows.append({
        "inst": inst,
        "price": latest_price,
        "regime": latest_regime,
        "side": side,
        "signal_type": latest_sig.get("signal_type", "none"),
        "reason": latest_sig.get("reason", ""),
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "suggest_qty": qty,
        "suggest_notional": notion,
        "edge_score": edge_score,
        "df": df_ind,
        "signals_df": signals_df
    })

if not rows:
    st.error("所有标的的数据都获取失败，请检查网络或 instId 是否正确。")
    st.stop()

# 把交易信息转成 DataFrame 方便展示
display_rows = []
for r in rows:
    side_zh = {"long": "做多", "short": "做空", "flat": "观望"}.get(r["side"], "观望")
    regime_zh = {
        "trend": "趋势市",
        "squeeze": "压缩待爆发",
        "mean_reversion": "震荡均值回归",
        "unknown": "待定"
    }.get(r["regime"], "待定")

    display_rows.append({
        "品种": r["inst"],
        "最新价格": f"{r['price']:.2f}",
        "市场状态": regime_zh,
        "当前信号": side_zh,
        "信号类型": {
            "breakout_trend": "趋势突破",
            "mean_reversion": "均值回归",
            "none": "无"
        }.get(r["signal_type"], "无"),
        "信号说明": r["reason"],
        "计划进场价": f"{r['entry']:.2f}" if not np.isnan(r["entry"]) else "",
        "止损价": f"{r['sl']:.2f}" if not np.isnan(r["sl"]) else "",
        "止盈价": f"{r['tp']:.2f}" if not np.isnan(r["tp"]) else "",
        "建议币数": f"{r['suggest_qty']:.4f}" if r["suggest_qty"] > 0 else "",
        "名义价值USDT": f"{r['suggest_notional']:.2f}" if r["suggest_notional"] > 0 else "",
        "信号优先级": r["edge_score"]
    })

summary_df = pd.DataFrame(display_rows).sort_values("信号优先级", ascending=False)
st.dataframe(summary_df, use_container_width=True)

st.markdown("""
**阅读建议：**

- 若你只能关注少数几个标的，可以优先看“信号优先级”高的（趋势突破 > 均值回归 > 无信号）；
- 市场状态为“趋势市 / 压缩待爆发”且有趋势突破信号，是典型 1–2 天波段机会；
- 市场状态为“震荡均值回归”且有均值回归信号，适合小仓位、日内或 1 天内博差价。
""")

# ============ 单一标的详细回测与分布 ============

st.subheader("二、单一标的详细回测与盈亏分布")

inst_options = [r["inst"] for r in rows]
chosen_inst = st.selectbox("选择查看详细回测的标的", options=inst_options)

chosen = next(r for r in rows if r["inst"] == chosen_inst)
df_chosen = chosen["df"]
signals_chosen = chosen["signals_df"]

with st.spinner(f"正在对 {chosen_inst} 策略做回测..."):
    trades_df, equity_curve = backtest_short_term(df_chosen, signals_chosen)

stats = summarize_trades(trades_df)

col1, col2, col3, col4, col5 = st.columns(5)
if stats:
    col1.metric("总交易次数", stats["total_trades"])
    col2.metric("胜率", f"{stats['win_rate']*100:.1f}%")
    col3.metric("单笔平均R", f"{stats['avg_r']:.2f}")
    col4.metric("平均盈利R", f"{stats['avg_win_r']:.2f}")
    col5.metric("最大回撤(≈)", f"{stats['max_drawdown']*100:.1f}%")
else:
    st.info("当前策略参数下暂无交易记录。")

st.markdown("**机械执行净值曲线（假设每笔风险≈1%资金）：**")
if equity_curve is not None and not equity_curve.empty:
    st.line_chart(equity_curve)
else:
    st.info("暂无足够交易记录绘制资金曲线。")

st.markdown(f"**最近 {min(recent_n, len(trades_df))} 笔信号的盈亏分布（单位：R）**")
if not trades_df.empty:
    recent_trades = trades_df.tail(recent_n)
    st.bar_chart(recent_trades["r"])
    st.write("最近部分交易明细：")
    st.dataframe(recent_trades.tail(30), use_container_width=True)
else:
    st.info("暂无交易记录，无法绘制盈亏分布。")

st.markdown("""
> 提醒：  
> - 回测未计入手续费、滑点、资金费率；  
> - 在极端行情下，实际滑点可能显著放大真实亏损；  
> - 若最近 N 笔信号连续负R，说明当前市场与策略风格“脱节”，此时减仓或暂停，比一味加码更专业。
""")

# ============ 风险提示 ============

st.warning("""
⚠️ 风险提示：

- 本策略基于历史 1H K 线的统计结构构建，**历史不代表未来**；
- 策略只回答：“在某种行情结构下，怎样的进出场在历史上有统计优势”，
  不保证未来每一次信号都是盈利；
- 真正的优势来自：长期执行 + 严格风控 + 减少主观干扰；
- 请务必根据自身资金规模、风险承受能力、交易经验，谨慎使用本策略。
""")
