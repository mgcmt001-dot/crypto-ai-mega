import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from math import floor

# =============================
# Streamlit 基本设置
# =============================
st.set_page_config(
    page_title="OKX 多因子量化终端",
    layout="wide"
)

st.title("OKX 多因子多周期量化分析终端")
st.caption("⚠️ 本工具仅用于量化研究与教育，不构成任何投资建议。加密资产高风险，请谨慎使用杠杆。")

# =============================
# OKX 数据抓取部分
# =============================
BASE_URL = "https://www.okx.com"


@st.cache_data(show_spinner=False)
def fetch_okx_candles(inst_id: str, bar: str = "4H", limit: int = 500) -> pd.DataFrame:
    """
    从 OKX 获取 K 线数据
    inst_id: 如 'BTC-USDT-SWAP' 或 'BTC-USDT'
    bar: '1H','4H','1D' 等
    limit: 条数，OKX 单次最大通常在 300 左右，具体以文档为准
    """
    url = f"{BASE_URL}/api/v5/market/candles"
    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "0":
        raise ValueError(f"OKX API error: {data.get('msg')}")
    raw = data.get("data", [])

    # 返回是最新在前，需倒序
    df = pd.DataFrame(
        raw,
        columns=[
            "ts", "open", "high", "low", "close", "vol",
            "volCcy", "volCcyQuote", "confirm"
        ]
    )
    # 类型转换
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("ts").reset_index(drop=True)
    df = df.set_index("ts")
    return df


# =============================
# 技术指标函数
# =============================

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


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始 OHLCV 基础上添加常用技术指标
    """
    df = df.copy()
    close = df["close"]

    df["ema_fast"] = ema(close, 12)
    df["ema_med"] = ema(close, 50)
    df["ema_slow"] = ema(close, 200)

    df["rsi"] = rsi(close, 14)
    df["atr"] = atr(df, 14)

    df["macd"], df["macd_signal"], df["macd_hist"] = macd(close)

    df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger_bands(close, 20, 2)

    # 波动率（简单用 ATR/Close 表示）
    df["volatility"] = df["atr"] / (df["close"] + 1e-9)

    return df


# =============================
# 因子打分：趋势 / 反转 / 波动率
# =============================

def clamp(x, min_v=-1.0, max_v=1.0):
    return max(min_v, min(max_v, x))


def compute_style_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：已计算指标的 df
    输出：三个风格因子 + 综合多空评分（-1~1）
    """
    df = df.copy()
    factors = pd.DataFrame(index=df.index)

    # ---- 趋势因子：EMA 排列 + MACD + 价格相对均线
    trend_score = []

    for idx, row in df.iterrows():
        score = 0.0
        w_sum = 0.0

        # EMA 排列：fast > med > slow 强多头
        if not np.isnan(row["ema_fast"]) and not np.isnan(row["ema_med"]) and not np.isnan(row["ema_slow"]):
            if row["ema_fast"] > row["ema_med"] > row["ema_slow"]:
                score += 0.6
            elif row["ema_fast"] < row["ema_med"] < row["ema_slow"]:
                score -= 0.6
            w_sum += 0.6

        # MACD 大于 0 倾向多；小于 0 倾向空
        if not np.isnan(row["macd"]):
            macd_norm = np.tanh(row["macd"] / (row["close"] * 0.01 + 1e-9))
            score += 0.3 * macd_norm
            w_sum += 0.3

        # 价格相对 200EMA 的偏离（趋势状态）
        if not np.isnan(row["ema_slow"]):
            dev = (row["close"] - row["ema_slow"]) / (row["ema_slow"] + 1e-9)
            dev_norm = np.tanh(dev * 3)
            score += 0.3 * dev_norm
            w_sum += 0.3

        if w_sum > 0:
            score = score / w_sum
        trend_score.append(clamp(score))

    factors["trend_factor"] = trend_score

    # ---- 反转因子：RSI + 价格相对布林带
    reversal_score = []
    for idx, row in df.iterrows():
        score = 0.0
        w_sum = 0.0

        # RSI：<30 认为有反弹潜力（看多反转）；>70 有回调潜力（看空反转）
        if not np.isnan(row["rsi"]):
            if row["rsi"] < 30:
                # 超卖越深，反转越强
                score += (30 - row["rsi"]) / 30.0  # 0~1
            elif row["rsi"] > 70:
                score -= (row["rsi"] - 70) / 30.0
            w_sum += 1.0

        # 布林带：跌破下轨 -> 看多反转；突破上轨 -> 看空反转
        if not np.isnan(row["bb_lower"]) and not np.isnan(row["bb_upper"]):
            if row["close"] < row["bb_lower"]:
                score += 0.7
                w_sum += 0.7
            elif row["close"] > row["bb_upper"]:
                score -= 0.7
                w_sum += 0.7

        if w_sum > 0:
            score = score / w_sum
        reversal_score.append(clamp(score))

    factors["reversal_factor"] = reversal_score

    # ---- 波动率因子：适中最好，过高或过低都扣分
    # 先算波动率分布的中位数和 IQR
    vol = df["volatility"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vol) > 0:
        med = vol.median()
        iqr = vol.quantile(0.75) - vol.quantile(0.25)
        if iqr == 0:
            iqr = med if med != 0 else 1.0
    else:
        med, iqr = 0.0, 1.0

    vol_scores = []
    for v in df["volatility"]:
        if np.isnan(v):
            vol_scores.append(0.0)
            continue
        z = (v - med) / (iqr + 1e-9)
        # |z| < 0.5 -> 适中，得分接近 1
        # |z| > 2   -> 过低/过高，得分接近 -1
        score = 1.0 - min(abs(z) / 2.0, 2.0)  # 1 -> -1
        vol_scores.append(clamp(score, -1, 1))
    factors["volatility_factor"] = vol_scores

    # ---- 综合多空评分：趋势为主，反转为次，波动率作风险调节
    # 趋势 0.5，反转 0.3，波动率 0.2
    factors["raw_score"] = (
        0.5 * factors["trend_factor"] +
        0.3 * factors["reversal_factor"] +
        0.2 * factors["volatility_factor"]
    )

    # 波动率作为“置信度”调节：vol_factor 为负时，降低绝对信号强度
    factors["score"] = factors["raw_score"] * (0.5 + 0.5 * factors["volatility_factor"])

    return factors


# =============================
# 信号 & 止盈止损逻辑
# =============================

def generate_trade_plan(latest_row: pd.Series,
                        factor_row: pd.Series,
                        direction_hint: str,
                        horizon: str,
                        risk_reward: float = 2.0):
    """
    针对一个周期，基于最新价格 + ATR，给出方向 & 止盈止损。
    direction_hint: 'long' / 'short' / 'flat'
    horizon: '超短线','短线','中线','波段','趋势'
    """
    price = latest_row["close"]
    atr_val = latest_row["atr"]
    if np.isnan(atr_val) or atr_val <= 0:
        return {"direction": "观望", "entry": np.nan, "sl": np.nan, "tp": np.nan}

    # 不同周期给不同的 ATR 止损宽度
    if horizon == "超短线":
        sl_mult = 0.8
    elif horizon == "短线":
        sl_mult = 1.2
    elif horizon == "中线":
        sl_mult = 1.8
    elif horizon == "波段":
        sl_mult = 2.2
    else:  # 趋势
        sl_mult = 2.8

    # 若因子评分绝对值太小，则观望
    score = factor_row["score"] if "score" in factor_row else 0.0
    if abs(score) < 0.25:
        return {"direction": "观望", "entry": price, "sl": np.nan, "tp": np.nan}

    # 合并 direction_hint 与 score
    # 若二者方向冲突，适当减弱
    dir_from_score = "long" if score > 0 else "short"
    effective_dir = dir_from_score
    if direction_hint != "flat" and direction_hint != dir_from_score:
        # 冲突则观望
        return {"direction": "观望", "entry": price, "sl": np.nan, "tp": np.nan}

    if effective_dir == "long":
        sl = price - sl_mult * atr_val
        tp = price + sl_mult * atr_val * risk_reward
        direction_zh = "做多"
    else:
        sl = price + sl_mult * atr_val
        tp = price - sl_mult * atr_val * risk_reward
        direction_zh = "做空"

    return {
        "direction": direction_zh,
        "entry": price,
        "sl": sl,
        "tp": tp,
        "score": score,
        "atr": atr_val
    }


# =============================
# 简单回测逻辑（基于单一周期）
# =============================

def backtest_factor_strategy(df: pd.DataFrame,
                             factors: pd.DataFrame,
                             score_open_threshold: float = 0.4,
                             atr_sl_mult: float = 1.2,
                             rr: float = 2.0,
                             max_hold_bars: int = 60):
    """
    非高频、非精准撮合，仅用于评估策略大致胜率 / 期望。
    模型：
      - 当 score > threshold 开多
      - 当 score < -threshold 开空
      - 止损：ATR * atr_sl_mult
      - 止盈：收益:R 风险:1
      - 同时触发止盈止损时，保守地认为先止损
      - 不考虑手续费和滑点（可自行扩展）
    返回：
      trades_df, equity_curve (Series starting from 1.0)
    """
    df_bt = df.copy()
    df_bt["score"] = factors["score"]

    trades = []
    equity = [1.0]
    equity_times = [df_bt.index[0]]

    in_pos = False
    direction = None
    entry_price = None
    sl = None
    tp = None
    entry_time = None

    for i in range(len(df_bt)):
        row = df_bt.iloc[i]
        time = df_bt.index[i]
        price = row["close"]
        score = row["score"]
        atr_val = row["atr"]

        if not in_pos:
            if np.isnan(score) or np.isnan(atr_val):
                continue
            # 开仓逻辑
            if score > score_open_threshold:
                direction = "long"
            elif score < -score_open_threshold:
                direction = "short"
            else:
                direction = None

            if direction is not None:
                in_pos = True
                entry_price = price
                entry_time = time
                sl_dist = atr_sl_mult * atr_val
                if direction == "long":
                    sl = entry_price - sl_dist
                    tp = entry_price + sl_dist * rr
                else:
                    sl = entry_price + sl_dist
                    tp = entry_price - sl_dist * rr
                bars_held = 0
        else:
            # 在持仓中，检查止盈止损 or 超时
            bars_held += 1
            high = row["high"]
            low = row["low"]
            exit_reason = None

            if direction == "long":
                hit_sl = low <= sl
                hit_tp = high >= tp
            else:
                hit_sl = high >= sl
                hit_tp = low <= tp

            # 保守处理：同一根 K 线同时触发，先算止损
            if hit_sl:
                exit_price = sl
                exit_time = time
                exit_reason = "SL"
            elif hit_tp:
                exit_price = tp
                exit_time = time
                exit_reason = "TP"
            elif bars_held >= max_hold_bars:
                exit_price = price
                exit_time = time
                exit_reason = "TIME"

            if exit_reason is not None:
                # 用 R 来衡量盈亏：1R = 止损距离
                risk_per_unit = abs(entry_price - sl)
                if risk_per_unit == 0:
                    r = 0.0
                else:
                    if direction == "long":
                        pnl = exit_price - entry_price
                    else:
                        pnl = entry_price - exit_price
                    r = pnl / risk_per_unit

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "r": r,
                    "exit_reason": exit_reason,
                    "holding_bars": bars_held
                })

                # 假设每笔风险=1%资金，则资金变化：equity *= (1 + r * 0.01)
                # 为简化，这里用每笔风险固定为资金的 1 单位，R 直接累计
                equity.append(equity[-1] * (1 + r * 0.01))
                equity_times.append(exit_time)

                in_pos = False
                direction = None
                entry_price = None
                sl = None
                tp = None

    if len(equity) == 1:
        # 没有任何交易
        equity_curve = pd.Series([1.0, 1.0], index=[df_bt.index[0], df_bt.index[-1]])
        trades_df = pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction", "entry_price",
            "exit_price", "r", "exit_reason", "holding_bars"
        ])
    else:
        equity_curve = pd.Series(equity, index=equity_times)
        trades_df = pd.DataFrame(trades)

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

    # 简单最大回撤（基于 equity=1+累计R 的近似）
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


# =============================
# 仓位建议（基于资金 & 止损距离）
# =============================

def position_sizing(capital_usdt: float,
                    risk_pct: float,
                    entry_price: float,
                    stop_price: float):
    """
    根据资金 & 风险比例 & 止损价格，给出现货/单向合约的建议币数
    """
    if capital_usdt <= 0 or entry_price <= 0 or np.isnan(stop_price):
        return 0.0, 0.0

    risk_amt = capital_usdt * risk_pct
    stop_dist = abs(entry_price - stop_price)
    if stop_dist <= 0:
        return 0.0, 0.0

    qty = risk_amt / stop_dist
    # 对于永续合约，一张合约代表的面值需参考 OKX 具体设置，此处给出币数量级
    notional = qty * entry_price
    return qty, notional


# =============================
# UI 布局
# =============================

# ---- Sidebar 参数 ----
st.sidebar.header("参数设置")

default_inst = "BTC-USDT-SWAP"
inst_id = st.sidebar.text_input("交易对（OKX instId）", value=default_inst)

timeframes = ["1H", "4H", "1D"]
main_tf = st.sidebar.selectbox("主回测周期", options=timeframes, index=1)

capital = st.sidebar.number_input("账户资金（USDT）", min_value=100.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("每笔风险占资金比例", min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100.0

score_open_th = st.sidebar.slider("开仓信号阈值 |score| >", min_value=0.2, max_value=0.8, value=0.4, step=0.05)
recent_n = st.sidebar.slider("最近 N 笔信号用于直方图", min_value=10, max_value=200, value=50, step=10)

st.sidebar.markdown("---")
st.sidebar.markdown("数据来自 OKX 公共 API，实际结果可能受接口限制及网络环境影响。")

# ---- 主页面：数据抓取 ----
with st.spinner("从 OKX 获取数据中..."):
    try:
        df_1h = fetch_okx_candles(inst_id, "1H", limit=600)
        df_4h = fetch_okx_candles(inst_id, "4H", limit=600)
        df_1d = fetch_okx_candles(inst_id, "1D", limit=600)
    except Exception as e:
        st.error(f"获取 OKX 数据失败：{e}")
        st.stop()

# 添加指标 & 因子
df_1h_ind = add_indicators(df_1h)
fac_1h = compute_style_factors(df_1h_ind)

df_4h_ind = add_indicators(df_4h)
fac_4h = compute_style_factors(df_4h_ind)

df_1d_ind = add_indicators(df_1d)
fac_1d = compute_style_factors(df_1d_ind)

latest_1h = df_1h_ind.iloc[-1]
latest_4h = df_4h_ind.iloc[-1]
latest_1d = df_1d_ind.iloc[-1]

fac_last_1h = fac_1h.iloc[-1]
fac_last_4h = fac_4h.iloc[-1]
fac_last_1d = fac_1d.iloc[-1]

current_price = latest_1h["close"]
st.subheader(f"{inst_id} 当前价格：{current_price:.2f} USDT（基于 1H 最新收盘）")

# =============================
# 顶层方向判定：多周期逻辑
# =============================

def dir_from_score(score):
    if score > 0.15:
        return "long"
    elif score < -0.15:
        return "short"
    else:
        return "flat"


dir_1h = dir_from_score(fac_last_1h["score"])
dir_4h = dir_from_score(fac_last_4h["score"])
dir_1d = dir_from_score(fac_last_1d["score"])

# 组合方向（简单多数表决，日线权重最高）
vote = {"long": 0, "short": 0, "flat": 0}
vote[dir_1h] += 1
vote[dir_4h] += 2
vote[dir_1d] += 3
overall_dir = max(vote, key=vote.get)

overall_dir_zh = {
    "long": "整体偏多",
    "short": "整体偏空",
    "flat": "整体观望"
}[overall_dir]

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("1H 多空评分", f"{fac_last_1h['score']:.2f}")
    st.write(f"方向：{dir_1h}")
with col_b:
    st.metric("4H 多空评分", f"{fac_last_4h['score']:.2f}")
    st.write(f"方向：{dir_4h}")
with col_c:
    st.metric("1D 多空评分", f"{fac_last_1d['score']:.2f}")
    st.write(f"方向：{dir_1d}")

st.info(f"多周期综合判断：**{overall_dir_zh}**（1D 权重最高，其次 4H，再是 1H）")

# =============================
# 各周期交易计划：方向 + 止盈止损 + 仓位建议
# =============================

st.subheader("多周期交易计划（超短线 / 短线 / 中线 / 波段 / 趋势）")

plans = []

# 定义 5 个维度映射到不同时间框架
horizon_map = [
    ("超短线", df_1h_ind, fac_1h),
    ("短线", df_4h_ind, fac_4h),
    ("中线", df_1d_ind, fac_1d),
    ("波段", df_1d_ind, fac_1d),
    ("趋势", df_1d_ind, fac_1d),
]

for horizon, df_use, fac_use in horizon_map:
    latest_row = df_use.iloc[-1]
    fac_row = fac_use.iloc[-1]
    # 方向提示：趋势型维度更多参考 1D / 4H 综合方向
    dir_hint = overall_dir
    plan = generate_trade_plan(latest_row, fac_row, dir_hint, horizon, risk_reward=2.0)

    # 计算建议仓位
    qty, notion = position_sizing(
        capital_usdt=capital,
        risk_pct=risk_pct,
        entry_price=plan.get("entry", np.nan),
        stop_price=plan.get("sl", np.nan)
    )

    plan["horizon"] = horizon
    plan["suggest_qty"] = qty
    plan["suggest_notional"] = notion
    plans.append(plan)

plans_df = pd.DataFrame(plans)[
    ["horizon", "direction", "score", "entry", "sl", "tp", "atr", "suggest_qty", "suggest_notional"]
]

# 格式化展示
def fmt_row(row):
    return {
        "周期": row["horizon"],
        "方向": row["direction"],
        "多空评分": f"{row['score']:.2f}" if not pd.isna(row["score"]) else "",
        "计划进场价": f"{row['entry']:.2f}" if not pd.isna(row["entry"]) else "",
        "止损价": f"{row['sl']:.2f}" if not pd.isna(row["sl"]) else "",
        "止盈价": f"{row['tp']:.2f}" if not pd.isna(row["tp"]) else "",
        "当前 ATR": f"{row['atr']:.2f}" if not pd.isna(row["atr"]) else "",
        "建议币数": f"{row['suggest_qty']:.4f}" if row["suggest_qty"] > 0 else "",
        "对应名义价值 USDT": f"{row['suggest_notional']:.2f}" if row["suggest_notional"] > 0 else ""
    }

plans_fmt = pd.DataFrame([fmt_row(r) for _, r in plans_df.iterrows()])
st.dataframe(plans_fmt, use_container_width=True)

st.markdown("""
**解读要点：**

- 若某周期显示“观望”，说明当前因子信号不够强，或与上级周期方向冲突；
- 建议币数是基于你设置的资金与单笔风险比例计算的，  
  例如：资金 10,000 USDT、风险 1%、止损距离 100 USDT，则最大亏损 100 USDT，仓位约 1 枚；
- 趋势周期止损更宽（更大 ATR 倍数），适合低杠杆、低频持有。
""")

# =============================
# 风格剖面：趋势 / 反转 / 波动率因子评分
# =============================

st.subheader("多因子风格剖面（趋势 / 反转 / 波动率）")

style_profile = pd.DataFrame({
    "因子": ["趋势因子", "反转因子", "波动率因子"],
    "1H": [
        fac_last_1h["trend_factor"],
        fac_last_1h["reversal_factor"],
        fac_last_1h["volatility_factor"],
    ],
    "4H": [
        fac_last_4h["trend_factor"],
        fac_last_4h["reversal_factor"],
        fac_last_4h["volatility_factor"],
    ],
    "1D": [
        fac_last_1d["trend_factor"],
        fac_last_1d["reversal_factor"],
        fac_last_1d["volatility_factor"],
    ],
})

st.dataframe(style_profile.set_index("因子"), use_container_width=True)
st.markdown("""
- 因子区间：-1 ~ 1；  
  - 趋势因子 > 0 表示偏多头趋势，< 0 表示偏空头；  
  - 反转因子 > 0 表示更偏向“多头反转”（低位反弹），< 0 偏向“空头反转”；  
  - 波动率因子接近 1，表示波动处于“健康区间”，策略信号可靠性更高。
""")

# =============================
# 回测：最近约 3 个月（取主周期 K 线历史）
# =============================

st.subheader("简单历史回测：如果机械执行这套模型，过去一段时间表现如何？")

if main_tf == "1H":
    df_main = df_1h_ind
    fac_main = fac_1h
elif main_tf == "4H":
    df_main = df_4h_ind
    fac_main = fac_4h
else:
    df_main = df_1d_ind
    fac_main = fac_1d

st.write(f"当前主回测周期：**{main_tf}**，K 线条数：{len(df_main)}（约略对应最近几个月数据，具体取决于 OKX 接口限制）")

trades_df, equity_curve = backtest_factor_strategy(
    df_main,
    fac_main,
    score_open_threshold=score_open_th,
    atr_sl_mult=1.2,
    rr=2.0,
    max_hold_bars=60
)

summary = summarize_trades(trades_df)

col1, col2, col3, col4, col5 = st.columns(5)
if summary:
    col1.metric("总交易次数", summary["total_trades"])
    col2.metric("胜率", f"{summary['win_rate']*100:.1f}%")
    col3.metric("单笔平均 R 值", f"{summary['avg_r']:.2f}")
    col4.metric("平均盈利 R", f"{summary['avg_win_r']:.2f}")
    col5.metric("最大回撤（基于R近似）", f"{summary['max_drawdown']*100:.1f}%")
else:
    st.info("当前参数下暂无足够历史信号用于回测。")

# 资金曲线
st.markdown("**机械执行净值曲线（假设每笔风险为资金 1 单位，R 换算为约 1% 波动）：**")
st.line_chart(equity_curve)

# 最近 N 次信号的盈亏直方图
st.markdown(f"**最近 {min(recent_n, len(trades_df))} 笔信号的盈亏分布（单位：R）**")
if not trades_df.empty:
    recent_trades = trades_df.tail(recent_n)
    hist_data = recent_trades["r"]
    st.bar_chart(hist_data, height=200)
    st.write("最近几笔交易明细：")
    st.dataframe(recent_trades.tail(20), use_container_width=True)
else:
    st.info("暂无交易记录，无法绘制盈亏直方图。")

st.markdown("""
> 回测说明：  
> - 为简单起见，我们假设：
>   - 不计交易手续费和滑点；
>   - 止盈止损在 K 线内按“先触发止损”处理（保守估计）；  
>   - 每次开仓使用相同的风险规模（R 统一口径）。
> - 若你要用于实盘，请在此基础上进一步：
>   - 加入手续费滑点模型；  
>   - 增加多品种、多周期组合回测；  
>   - 做真实资金曲线与风控模拟（如最大回撤、卡玛比等）。
""")

# =============================
# 风控提示
# =============================

st.warning("""
**重要风险提示：**

- 本模型是“技术因子 + 简单规则 + 粗粒度回测”，
  并非高频做市或复杂机器学习策略；
- 历史表现不代表未来收益，市场可能在回测期外进入完全不同的结构；
- 仓位建议只基于“单笔最大亏多少”这一维度，
  没有考虑你的总品种数量、组合相关性等问题；
- 真正的专业交易，会把“是否出手”看得比“出手方向”更重要。  
  当信号不够强，**不交易就是最好的交易**。
""")
