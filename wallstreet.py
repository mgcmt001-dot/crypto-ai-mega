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


# ============ 回测 ============

def backtest_short_term(df: pd.DataFrame, signals_df: pd.DataFrame):
    """
    基于 gen_short_term_signal 生成的信号做简单回测。
    - 在信号出现的那根K线收盘价开仓
    - 后续K线内检查 high/low 是否触及止盈/止损/超时
    - 单位为 R（止损距离为 1R）
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
    signal_type = None
    entry_regime = None

    idx_list = df.index
    n = len(idx_list)

    for i in range(n):
        t = idx_list[i]
        row = df.iloc[i]
        sig = signals_df.loc[t]

        if not in_pos:
            # 新信号
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
            # 持仓中
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

    # 资金曲线（假设每笔风险=资金1%，R≈1%）
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
    """
    拆分到不同 signal_type 看表现：
    - breakout_trend
    - mean_reversion
    """
    if trades_df.empty:
        return pd.DataFrame(columns=[
            "signal_type", "total_trades", "win_rate",
            "avg_r", "avg_win_r", "avg_loss_r", "max_drawdown"
        ])

    rows = []
    for sig_type, sub in trades_df.groupby("signal_type"):
        stats = summarize_trades(sub)
        rows.append({
            "signal_type": sig_type,
            **stats
        })
    return pd.DataFrame(rows)


# ============ 资产级综合打分（0–100） ============

def compute_asset_score(stats: dict, trades_df: pd.DataFrame) -> float:
    """
    按机构视角给一个 0–100 的综合评分：
    - 胜率（越高越好）
    - 平均R（>0最好）
    - 最大回撤（越小越好）
    - 最近20笔平均R（衡量当前“状态”）
    - 交易样本数量（样本太少自动打折）
    """
    if not stats or "total_trades" not in stats or stats["total_trades"] == 0:
        return 0.0

    total = stats["total_trades"]
    win_rate = stats["win_rate"]        # 0 ~ 1
    avg_r = stats["avg_r"]              # 可正可负
    max_dd = stats["max_drawdown"]      # 负数为主

    # 最近表现
    if len(trades_df) >= 5:
        recent = trades_df.tail(20)
        recent_avg_r = recent["r"].mean()
    else:
        recent_avg_r = 0.0

    # 将各维度映射到 0~1 区间（用 tanh 做平滑）
    win_score = win_rate  # 已是0~1
    r_score = 0.5 + 0.5 * np.tanh(avg_r / 1.0)          # 大致 (-∞,∞)->(0,1)
    recent_score = 0.5 + 0.5 * np.tanh(recent_avg_r / 1.0)
    dd_score = 0.5 + 0.5 * np.tanh(-max_dd / 0.3)       # 回撤越大，得分越低

    # 样本数量惩罚：小样本打折
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


# ============ 仓位建议 ============

def position_sizing(
    capital_usdt: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
):
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
st.sidebar.caption("数据来源：OKX 公共API\n本工具不保证数据完整与实时，实盘前请务必自测。")

if not universe:
    st.warning("请在左侧选择至少一个交易标的。")
    st.stop()


# ============ 主逻辑：批量获取数据 & 回测 & 打分 ============

st.subheader("一、跨品种信号 & 资产级打分（0–100）")

rows = []

for inst in universe:
    with st.spinner(f"获取 {inst} 1H K 线数据并回测策略..."):
        try:
            df = fetch_okx_candles(inst, "1H", limit=500)
        except Exception as e:
            st.error(f"{inst} 数据获取失败: {e}")
            continue

    df_ind = add_indicators(df)
    signals_df, latest_sig = gen_short_term_signal(df_ind)

    # 加一列 regime 到 df 方便后面分析
    regimes = []
    for _, r in df_ind.iterrows():
        regimes.append(classify_regime(r))
    df_ind["regime"] = regimes

    trades_df, equity_curve = backtest_short_term(df_ind, signals_df)
    stats = summarize_trades(trades_df)
    asset_score = compute_asset_score(stats, trades_df)

    latest_price = df_ind["close"].iloc[-1]
    latest_regime = df_ind["regime"].iloc[-1]

    # 当前信号与仓位建议
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

    # 信号优先级：趋势突破 > 均值回归 > 无
    if side in ["long", "short"]:
        if latest_sig.get("signal_type") == "breakout_trend" and latest_regime in [
            "trend",
            "squeeze",
        ]:
            edge_score = 2
        elif latest_sig.get("signal_type") == "mean_reversion":
            edge_score = 1
        else:
            edge_score = 0
    else:
        edge_score = -1

    rows.append(
        {
            "inst": inst,
            "price": latest_price,
            "latest_regime": latest_regime,
            "side": side,
            "signal_type": latest_sig.get("signal_type", "none"),
            "reason": latest_sig.get("reason", ""),
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "suggest_qty": qty,
            "suggest_notional": notion,
            "signal_priority": edge_score,
            "asset_score": asset_score,
            "stats": stats,
            "trades_df": trades_df,
            "equity_curve": equity_curve,
            "df": df_ind,
            "signals_df": signals_df,
        }
    )

if not rows:
    st.error("所有标的的数据都获取失败，请检查网络或 instId 是否正确。")
    st.stop()

# === 组合层面：根据资产级评分 + 信号强度分配风险 ===
# 策略：只对有信号的品种分配风险，且总风险不超过 portfolio_risk_pct
signaled = [r for r in rows if r["side"] in ["long", "short"]]
signaled_sorted = sorted(signaled, key=lambda x: (x["signal_priority"], x["asset_score"]), reverse=True)

if signaled_sorted:
    # 最高优先级的前 K 个标的参与组合（简单做法：最多 3 个）
    max_positions = min(3, len(signaled_sorted))
    top_assets = signaled_sorted[:max_positions]

    # 若只有1个，就给它全部组合风险；多个则平均分配组合风险
    per_position_risk_pct = portfolio_risk_pct / max_positions
else:
    top_assets = []
    per_position_risk_pct = 0.0

# 用组合风险重算“建议币数”（比刚才单品种固定风险更贴近组合管理）
for r in rows:
    if r in top_assets and r["side"] in ["long", "short"]:
        # 组合位置：使用组合风险；其余保持0仓
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

# === 展示：跨品种概览 ===
display_rows = []
for r in rows:
    side_zh = {"long": "做多", "short": "做空", "flat": "观望"}.get(r["side"], "观望")
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

    display_rows.append(
        {
            "品种": r["inst"],
            "最新价格": f"{r['price']:.2f}",
            "市场状态": regime_zh,
            "当前信号": side_zh,
            "信号类型": {
                "breakout_trend": "趋势突破",
                "mean_reversion": "均值回归",
                "none": "无",
            }.get(r["signal_type"], "无"),
            "信号说明": r["reason"],
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
            "信号优先级": r["signal_priority"],
        }
    )

summary_df = pd.DataFrame(display_rows).sort_values(
    ["信号优先级", "资产历史评分(0-100)"], ascending=False
)
st.dataframe(summary_df, use_container_width=True)

st.markdown(
    """
**如何解读这张表：**

- **资产历史评分(0–100)**：综合过去一段时间的胜率、平均R、最大回撤、最近20笔表现；
- **信号优先级**：趋势突破 > 均值回归 > 无信号；
- **组合建议币数**：已经考虑了 *组合层面风险上限*，例如你把组合总风险定在 2%，
  而有 2 个币入选组合，则每个币约 1% 风险；
- 实战的时候，你可以优先关注：
  - 资产评分高
  - 当前有信号
  - 信号优先级高  
  的那 1~3 个币种。
"""
)


# ============ 二、单一标的：策略分解 + 回测细节 ============

st.subheader("二、单一标的策略拆分表现（趋势突破 vs 均值回归）")

inst_options = [r["inst"] for r in rows]
chosen_inst = st.selectbox("选择查看详细回测的标的", options=inst_options)

chosen = next(r for r in rows if r["inst"] == chosen_inst)
df_chosen = chosen["df"]
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
    # 显示成更友好的中文
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


# ============ 三、风格剖面：这个币擅长什么行情？ ============

st.subheader("三、风格剖面：趋势 / 压缩 / 震荡 占比")

regime_counts = chosen["df"]["regime"].value_counts(normalize=True)
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

latest_row = chosen["df"].iloc[-1]
st.markdown(
    f"""
**当前K线的风格因子：**

- 趋势强度 trend_strength：{latest_row['trend_strength']:.2f}  
- 布林带宽度 bb_width：{latest_row['bb_width']:.4f}  
- RSI(14)：{latest_row['rsi']:.1f}  
- ATR(14)：{latest_row['atr']:.2f}  
- 当前市场状态：**{regime_zh_map.get(latest_row['regime'], '待定')}**
"""
)

st.markdown(
    """
从专业角度看：

- 这个币若“趋势市占比”长期偏高，说明它更适合趋势突破策略；
- 若“震荡均值回归”占比较高，均值回归策略贡献可能更大；
- 你也可以据此做 **“择币”**：  
  把擅长趋势的币放在趋势组合，把适合震荡的币放在套利/盘整组合，而不是一套打法打遍所有标的。
"""
)


# ============ 风险提示 ============

st.warning(
    """
⚠️ 风险提示（升级版同样适用）：

- 本策略基于 1H 历史数据做统计，**历史不代表未来**；
- 资产级评分只是在当前数据下给出的“统计信心”，不是“稳赚评级”；
- 组合层面风险控制虽然更接近机构方法，但依然没有覆盖所有极端情况（如闪崩、插针、交易所故障等）；
- 真正专业的做法是：先用极小资金验证一段时间，  
  熟悉策略的节奏、回撤特征、情绪考验，再考虑逐步加大仓位。
"""
)
