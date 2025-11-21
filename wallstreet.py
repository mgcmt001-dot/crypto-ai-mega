import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import time
from datetime import datetime, timezone
import ta

# ==========================
# Streamlit 全局配置
# ==========================
st.set_page_config(
    page_title="量化炒币分析助手 - OKX 多因子模型",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# 常量配置
# ==========================
OKX_BASE_URL = "https://www.okx.com"

TF_LABELS = {
    "15m": "15分钟",
    "1H": "1小时",
    "4H": "4小时",
    "1D": "1天"
}

# 为了尽量覆盖过去 3 个月：
# 15m: 90天≈8640根，取9000略多
# 1H: 90天≈2160根
# 4H: 90天≈540根
# 1D: 90天≈90根，但最多300根
MAX_CANDLES_BY_TF = {
    "15m": 9000,
    "1H": 2160,
    "4H": 540,
    "1D": 300,
}

# 多周期权重（针对短线交易偏好，短周期权重大）
TF_WEIGHTS = {
    "15m": 0.4,
    "1H": 0.3,
    "4H": 0.2,
    "1D": 0.1,
}

# 默认参数
DEFAULT_INST_ID = "BTC-USDT"
DEFAULT_CAPITAL = 10000.0  # 美元
DEFAULT_RISK_PCT = 1.0    # 单笔风险占比
ATR_MULTIPLIER = 2.5      # 止损 ATR 倍数
TAKE_PROFIT_R_MULTIPLE = 2.0  # 默认 2R 止盈
DEFAULT_LONG_THRESHOLD = 25.0
DEFAULT_SHORT_THRESHOLD = -25.0

# ==========================
# 工具函数：抓取数据
# ==========================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_okx_candles(inst_id: str, bar: str, max_candles: int) -> pd.DataFrame:
    """
    从 OKX 抓取 K 线数据（自动拼接多页，尽量覆盖 max_candles 根K线）
    使用 /market/candles + /market/history-candles
    """
    all_rows = []

    # 最新 300 根
    url_recent = f"{OKX_BASE_URL}/api/v5/market/candles"
    params = {"instId": inst_id, "bar": bar, "limit": 300}

    try:
        resp = requests.get(url_recent, params=params, timeout=10)
        j = resp.json()
    except Exception as e:
        st.error(f"请求 OKX 失败: {e}")
        return pd.DataFrame()

    if j.get("code") != "0":
        st.error(f"OKX API 返回错误: {j.get('msg')}")
        return pd.DataFrame()

    rows = j.get("data", [])
    if not rows:
        return pd.DataFrame()

    all_rows.extend(rows)

    # 更早历史
    url_hist = f"{OKX_BASE_URL}/api/v5/market/history-candles"

    while len(all_rows) < max_candles:
        oldest_ts = rows[-1][0]  # 毫秒时间戳字符串
        params_hist = {
            "instId": inst_id,
            "bar": bar,
            "before": oldest_ts,
            "limit": 300
        }
        try:
            resp = requests.get(url_hist, params=params_hist, timeout=10)
            j = resp.json()
        except Exception as e:
            st.warning(f"继续抓取历史K线失败: {e}")
            break

        if j.get("code") != "0":
            st.warning(f"OKX 历史K线接口错误: {j.get('msg')}")
            break

        rows = j.get("data", [])
        if not rows:
            break

        all_rows.extend(rows)
        time.sleep(0.2)  # 简单防止频率过高

    if not all_rows:
        return pd.DataFrame()

    cols = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]
    df = pd.DataFrame(all_rows, columns=cols)

    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)

    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    df = df.drop_duplicates(subset="ts")
    df = df.sort_values("ts").set_index("ts")

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fear_greed_index():
    """
    贪婪与恐惧指数（来自 alternative.me）
    返回 (value:int 0~100, classification:str, timestamp:datetime)
    """
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json().get("data", [])
        if not data:
            return None, None, None
        d = data[0]
        value = int(d["value"])
        classification = d["value_classification"]
        ts = pd.to_datetime(int(d["timestamp"]), unit="s", utc=True)
        return value, classification, ts
    except Exception as e:
        st.warning(f"贪婪与恐惧指数获取失败: {e}")
        return None, None, None


# ==========================
# 技术指标 & 因子计算
# ==========================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在原始K线基础上添加常用技术指标"""
    if df.empty:
        return df

    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA
    ema_fast_ind = ta.trend.EMAIndicator(close=close, window=20)
    ema_slow_ind = ta.trend.EMAIndicator(close=close, window=50)
    df["ema_fast"] = ema_fast_ind.ema_indicator()
    df["ema_slow"] = ema_slow_ind.ema_indicator()

    # RSI
    rsi_ind = ta.momentum.RSIIndicator(close=close, window=14)
    df["rsi"] = rsi_ind.rsi()

    # MACD
    macd_ind = ta.trend.MACD(close=close)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()

    # ATR
    atr_ind = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    )
    df["atr"] = atr_ind.average_true_range()

    # 布林带
    bb_ind = ta.volatility.BollingerBands(
        close=close, window=20, window_dev=2
    )
    df["bb_high"] = bb_ind.bollinger_hband()
    df["bb_low"] = bb_ind.bollinger_lband()
    df["bb_mid"] = bb_ind.bollinger_mavg()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / close

    # ADX
    adx_ind = ta.trend.ADXIndicator(
        high=high, low=low, close=close, window=14
    )
    df["adx"] = adx_ind.adx()

    return df


def add_factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于指标构建三大风格因子 + 综合多空得分：
    - 趋势因子：EMA 斜率 + ADX 强度
    - 反转因子：RSI 偏离 + 布林带位置
    - 波动率因子：ATR% + 布林带宽度相对历史水平
    输出:
      trend_score, reversal_score, vol_score, signal_score
      范围大致在 [-100, 100]
    """
    if df.empty:
        return df

    df = df.copy()

    # 趋势：EMA 斜率 * ADX 强度
    df["ema_slope"] = (df["ema_fast"] - df["ema_slow"]) / df["ema_slow"]
    df["ema_slope"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df["ema_slope"].fillna(0, inplace=True)

    df["adx_norm"] = df["adx"] / 25.0  # ADX>25 视为趋势较强
    df["adx_norm"].fillna(0, inplace=True)

    trend_raw = df["ema_slope"] * df["adx_norm"] * 10.0
    df["trend_score"] = 50.0 * np.tanh(trend_raw)

    # 反转：RSI 偏离 + 布林带位置
    rsi = df["rsi"].copy()
    rsi.fillna(50.0, inplace=True)
    rsi_dev = (50.0 - rsi) / 15.0  # Oversold => 正值，Overbought => 负值

    denom = (df["bb_high"] - df["bb_low"]).replace(0, np.nan)
    bb_pos = (df["close"] - df["bb_low"]) / denom
    bb_pos = bb_pos.clip(0.0, 1.0).fillna(0.5)  # 中轨附近记作0.5

    rev_raw = rsi_dev + (0.5 - bb_pos)  # 底部偏多 => 正值
    df["reversal_score"] = 50.0 * np.tanh(rev_raw)

    # 波动率因子：ATR% + BB 宽度相对过去中位数
    vol_raw = (df["atr"] / df["close"]).fillna(0) + df["bb_width"].fillna(0)
    median_vol = vol_raw.rolling(200, min_periods=50).median()
    vol_ratio = vol_raw / median_vol.replace(0, np.nan)
    vol_ratio.fillna(1.0, inplace=True)

    df["vol_score"] = 50.0 * (vol_ratio - 1.0)
    df["vol_score"] = df["vol_score"].clip(-50.0, 50.0)

    # 综合多空评分：趋势为主，反转为辅
    df["signal_score"] = df["trend_score"] * 0.7 + df["reversal_score"] * 0.3
    df["signal_score"] = df["signal_score"].clip(-100.0, 100.0)

    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_data_with_factors(inst_id: str, bar: str) -> pd.DataFrame:
    """整体封装：抓 K 线 + 指标 + 因子"""
    max_candles = MAX_CANDLES_BY_TF.get(bar, 300)
    df = fetch_okx_candles(inst_id, bar, max_candles=max_candles)
    if df.empty:
        return df
    df = add_indicators(df)
    df = add_factor_scores(df)
    return df


def get_last_snapshot(df: pd.DataFrame) -> dict:
    """获取最新一根K线对应的因子快照"""
    d = df.dropna().iloc[-1]
    return {
        "close": float(d["close"]),
        "atr": float(d["atr"]),
        "trend_score": float(d["trend_score"]),
        "reversal_score": float(d["reversal_score"]),
        "vol_score": float(d["vol_score"]),
        "signal_score": float(d["signal_score"]),
        "rsi": float(d["rsi"]),
        "adx": float(d["adx"]),
        "time": df.dropna().index[-1],
    }


@st.cache_data(ttl=600, show_spinner=False)
def multi_tf_analysis(inst_id: str):
    """
    多周期分析：15m / 1H / 4H / 1D
    返回：
      per_tf: {tf: snapshot}
      agg: 聚合因子结果
    """
    per_tf = {}
    agg = {"trend": 0.0, "reversal": 0.0, "vol": 0.0, "signal": 0.0}
    used_weights = 0.0

    for tf, w in TF_WEIGHTS.items():
        df = load_data_with_factors(inst_id, tf)
        if df.empty or df.dropna().empty:
            continue
        snap = get_last_snapshot(df)
        per_tf[tf] = snap
        agg["trend"] += snap["trend_score"] * w
        agg["reversal"] += snap["reversal_score"] * w
        # 波动率风格看的是绝对大小
        agg["vol"] += abs(snap["vol_score"]) * w
        agg["signal"] += snap["signal_score"] * w
        used_weights += w

    if used_weights > 0:
        for k in agg:
            agg[k] /= used_weights

    return per_tf, agg


# ==========================
# 回测引擎（简化）
# ==========================

def backtest_strategy(
    df: pd.DataFrame,
    long_th: float = DEFAULT_LONG_THRESHOLD,
    short_th: float = DEFAULT_SHORT_THRESHOLD,
    atr_mult: float = ATR_MULTIPLIER,
    tp_r_mult: float = TAKE_PROFIT_R_MULTIPLE,
):
    """
    简单规则：
      - 使用 signal_score 作为信号
      - signal_score >= long_th => 下一根K线开盘做多
      - signal_score <= short_th => 下一根K线开盘做空
      - 止损：ATR * atr_mult
      - 止盈：距离= R * tp_r_mult
      - 信号衰减（多头时 signal_score <= 0 / 空头时 >=0）则平仓
    忽略手续费与滑点，仅用于研究胜率和风格。
    """
    df_bt = df.dropna(subset=["signal_score", "atr"]).copy()
    if df_bt.shape[0] < 60:
        return [], pd.Series(dtype=float)

    trades = []
    equity_curve = []
    equity = 1.0

    idx = df_bt.index
    pos = None  # 当前持仓

    for i in range(1, len(df_bt)):
        t = idx[i]
        prev_t = idx[i - 1]
        row = df_bt.iloc[i]
        prev = df_bt.iloc[i - 1]

        open_price = row["open"]
        high = row["high"]
        low = row["low"]
        signal_prev = prev["signal_score"]
        atr_prev = prev["atr"]

        # 先检查已有持仓是否需要平仓
        if pos is not None:
            exit_reason = None
            exit_price = None

            if pos["direction"] == "long":
                stop = pos["stop"]
                target = pos["target"]

                # 保守处理：同一根K线中若同时到达止损和止盈，认为先止损
                if low <= stop:
                    exit_price = stop
                    exit_reason = "stop"
                elif high >= target:
                    exit_price = target
                    exit_reason = "target"
                elif signal_prev <= 0:
                    exit_price = open_price
                    exit_reason = "signal_fade"

            else:  # short
                stop = pos["stop"]
                target = pos["target"]
                if high >= stop:
                    exit_price = stop
                    exit_reason = "stop"
                elif low <= target:
                    exit_price = target
                    exit_reason = "target"
                elif signal_prev >= 0:
                    exit_price = open_price
                    exit_reason = "signal_fade"

            if exit_price is not None:
                if pos["direction"] == "long":
                    ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                else:
                    ret = (pos["entry_price"] - exit_price) / pos["entry_price"]

                equity *= (1.0 + ret)

                trades.append(
                    {
                        "entry_time": pos["entry_time"],
                        "exit_time": t,
                        "direction": pos["direction"],
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "pnl_pct": ret * 100.0,
                        "reason": exit_reason,
                    }
                )
                equity_curve.append({"time": t, "equity": equity})
                pos = None

        # 再看是否需要开新仓
        if pos is None:
            if signal_prev >= long_th:
                stop = open_price - atr_mult * atr_prev
                target = open_price + atr_mult * tp_r_mult * atr_prev
                pos = {
                    "direction": "long",
                    "entry_time": t,
                    "entry_price": open_price,
                    "stop": stop,
                    "target": target,
                }
            elif signal_prev <= short_th:
                stop = open_price + atr_mult * atr_prev
                target = open_price - atr_mult * tp_r_mult * atr_prev
                pos = {
                    "direction": "short",
                    "entry_time": t,
                    "entry_price": open_price,
                    "stop": stop,
                    "target": target,
                }

    if equity_curve:
        eq_series = pd.Series(
            [e["equity"] for e in equity_curve],
            index=[e["time"] for e in equity_curve],
        ).sort_index()
    else:
        eq_series = pd.Series(dtype=float)

    return trades, eq_series


def summarize_trades(trades):
    """从交易列表中提取关键统计指标"""
    if not trades:
        return None

    df_tr = pd.DataFrame(trades)
    n = len(df_tr)
    wins = (df_tr["pnl_pct"] > 0).sum()
    losses = (df_tr["pnl_pct"] <= 0).sum()
    win_rate = wins / n if n > 0 else np.nan

    avg_pnl = df_tr["pnl_pct"].mean()
    avg_win = df_tr.loc[df_tr["pnl_pct"] > 0, "pnl_pct"].mean()
    avg_loss = df_tr.loc[df_tr["pnl_pct"] <= 0, "pnl_pct"].mean()

    total_win = df_tr.loc[df_tr["pnl_pct"] > 0, "pnl_pct"].sum()
    total_loss = df_tr.loc[df_tr["pnl_pct"] <= 0, "pnl_pct"].sum()
    profit_factor = (
        total_win / abs(total_loss) if losses > 0 and total_loss != 0 else np.nan
    )

    # 以单笔收益序列近似计算净值 & 最大回撤
    eq = (1 + df_tr["pnl_pct"] / 100.0).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min() * 100.0 if len(dd) > 0 else np.nan

    # Kelly 估计
    if not np.isnan(avg_win) and not np.isnan(avg_loss) and avg_loss < 0:
        R = avg_win / abs(avg_loss)
        kelly = win_rate - (1 - win_rate) / max(R, 1e-6)
    else:
        kelly = np.nan

    return {
        "n_trades": n,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "kelly": kelly,
        "df_trades": df_tr,
    }


# ==========================
# 仓位管理
# ==========================

def calc_position_size(
    capital_usd: float,
    risk_pct: float,
    price: float,
    atr: float,
    atr_mult: float = ATR_MULTIPLIER,
    kelly: float | None = None,
):
    """
    基于 ATR 止损距离 + 风险占比 + Kelly 调整建议仓位（币数）
    """
    if np.isnan(price) or np.isnan(atr) or atr <= 0:
        return 0.0, 0.0, 0.0

    stop_dist = atr_mult * atr
    stop_pct = stop_dist / price

    if stop_pct <= 0:
        return 0.0, 0.0, 0.0

    base_risk_pct = risk_pct / 100.0

    # 若有 Kelly 估计，则做一个柔和调节
    kelly_adj = 1.0
    if kelly is not None and not np.isnan(kelly):
        # 理论 Kelly*f, 但我们限制在 [0.25, 1.5] 之间
        kelly_adj = float(np.clip(1.0 + kelly, 0.25, 1.5))

    effective_risk_pct = base_risk_pct * kelly_adj
    effective_risk_pct = float(np.clip(effective_risk_pct, 0.001, 0.05))  # 0.1% ~ 5%

    risk_capital = capital_usd * effective_risk_pct
    position_notional = risk_capital / stop_pct
    position_notional = min(position_notional, capital_usd)  # 不超过总资金

    coins = position_notional / price

    return coins, effective_risk_pct * 100.0, stop_pct * 100.0


# ==========================
# 可视化组件
# ==========================

def plot_price_chart(df: pd.DataFrame, title: str):
    """K线 + EMA + 布林带"""
    if df.empty:
        return go.Figure()

    df_plot = df.tail(300).copy()
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot["open"],
            high=df_plot["high"],
            low=df_plot["low"],
            close=df_plot["close"],
            name="K线",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["ema_fast"],
            name="EMA20",
            line=dict(color="#42a5f5", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["ema_slow"],
            name="EMA50",
            line=dict(color="#ab47bc", width=1.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["bb_high"],
            name="Bollinger 上轨",
            line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["bb_low"],
            name="Bollinger 下轨",
            line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(200,200,200,0.1)",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="价格",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def plot_style_radar(agg_style: dict):
    """多因子风格剖面雷达图"""
    trend = agg_style.get("trend", 0.0)
    rev = agg_style.get("reversal", 0.0)
    vol = agg_style.get("vol", 0.0)

    # 映射到 0~100
    def norm(x):
        return float(np.clip((abs(x) / 100.0) * 100.0, 0, 100))

    r_vals = [norm(trend), norm(rev), norm(vol)]
    categories = ["趋势因子", "反转因子", "波动率因子"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=r_vals + [r_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="风格剖面",
            line=dict(color="#42a5f5"),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
            )
        ),
        showlegend=False,
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


# ==========================
# Streamlit 主应用
# ==========================

def main():
    st.title("💹 量化炒币分析助手（OKX 多周期多因子模型）")

    st.markdown(
        """
**说明：**

- 数据源：OKX 公共行情接口（无需 API Key），15m / 1H / 4H / 1D 多周期。
- 模型：趋势因子 + 反转因子 + 波动率因子，多周期加权，输出综合多空评分（-100 ~ +100）。
- 功能：
  - 多周期 K 线图 + 指标
  - 多空方向建议 + 止盈止损参考
  - 回测胜率 / 最近 N 笔信号盈亏直方图
  - 过去约 3 个月机械执行的净值曲线（受 API 历史长度限制）
  - 基于资金规模 + 波动率的建议仓位（币数）
  - 贪婪与恐惧指数（情绪辅助）

> **风险提示：** 仅供量化研究与教学，实盘请自担风险，并做好仓位与止损。
"""
    )

    # 侧边栏参数
    st.sidebar.header("参数设置")

    inst_id = st.sidebar.text_input("交易对（OKX instId，例如 BTC-USDT）", DEFAULT_INST_ID)

    tf_choice = st.sidebar.selectbox(
        "回测与信号主周期",
        options=list(TF_LABELS.keys()),
        format_func=lambda x: TF_LABELS.get(x, x),
        index=1,  # 默认 1H
    )

    capital = st.sidebar.number_input(
        "账户资金规模（USD）", min_value=100.0, value=DEFAULT_CAPITAL, step=100.0
    )

    risk_pct = st.sidebar.slider(
        "单笔最大风险占比（%）", min_value=0.1, max_value=5.0, value=DEFAULT_RISK_PCT, step=0.1
    )

    n_signals = st.sidebar.slider(
        "最近 N 笔交易用于盈亏分布统计", min_value=20, max_value=200, value=100, step=10
    )

    long_th = st.sidebar.slider(
        "做多信号阈值（signal_score ≥）", min_value=5.0, max_value=60.0, value=DEFAULT_LONG_THRESHOLD, step=5.0
    )
    short_th = -st.sidebar.slider(
        "做空信号阈值（signal_score ≤ -X）", min_value=5.0, max_value=60.0, value=abs(DEFAULT_SHORT_THRESHOLD), step=5.0
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "建议：短线可用 15m/1H，波段用 4H，趋势用 1D；阈值越高信号越少但质量通常更高。"
    )

    # ================== 多周期分析 ==================
    st.subheader("📊 多周期因子分析")

    per_tf, agg = multi_tf_analysis(inst_id)

    if not per_tf:
        st.error("无法获取该交易对的数据，请检查 instId 是否正确（例如 BTC-USDT）。")
        return

    # 贪婪恐惧指数
    fng_value, fng_class, fng_ts = fetch_fear_greed_index()

    col1, col2 = st.columns([2, 1])

    with col1:
        # 显示主周期K线
        df_main = load_data_with_factors(inst_id, tf_choice)
        if df_main.empty:
            st.error("主周期数据为空。")
            return
        last_price = float(df_main["close"].iloc[-1])
        fig_price = plot_price_chart(
            df_main, f"{inst_id} - {TF_LABELS.get(tf_choice, tf_choice)} K线 & 指标"
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        st.markdown("**多周期综合因子风格剖面**")
        fig_radar = plot_style_radar(agg)
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("**当前多周期综合多空评分**")
        signal_score_agg = agg.get("signal", 0.0)
        trend_agg = agg.get("trend", 0.0)
        rev_agg = agg.get("reversal", 0.0)
        vol_agg = agg.get("vol", 0.0)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "综合多空评分 (−100~+100)",
                f"{signal_score_agg: .1f}",
            )
            st.metric("趋势因子", f"{trend_agg: .1f}")
        with col_b:
            st.metric("反转因子", f"{rev_agg: .1f}")
            st.metric("波动率因子", f"{vol_agg: .1f}")

        st.markdown("---")

        if fng_value is not None:
            st.markdown("**市场情绪：贪婪与恐惧指数**")
            st.metric("Fear & Greed Index", f"{fng_value} / 100", fng_class)
            if fng_value >= 75:
                st.caption("情绪极度贪婪：追高风险加大，留意风险，减仓或收紧止损更稳妥。")
            elif fng_value <= 25:
                st.caption("情绪极度恐惧：容易出现恐慌抛售后的反弹，适合逢低小仓布局，但仍需谨慎。")
            else:
                st.caption("情绪中性偏温和，模型信号可靠性相对较高。")
        else:
            st.caption("暂时无法获取贪婪与恐惧指数。")

    # ================== 多周期详情表 ==================
    st.markdown("### 🧭 各周期因子评分一览")

    tf_table_rows = []
    for tf, snap in per_tf.items():
        tf_table_rows.append(
            {
                "周期": TF_LABELS.get(tf, tf),
                "最新时间": snap["time"].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "价格": round(snap["close"], 4),
                "Trend 趋势因子": round(snap["trend_score"], 1),
                "Reversal 反转因子": round(snap["reversal_score"], 1),
                "Vol 波动率因子": round(snap["vol_score"], 1),
                "综合信号": round(snap["signal_score"], 1),
                "RSI": round(snap["rsi"], 1),
                "ADX": round(snap["adx"], 1),
            }
        )
    st.dataframe(pd.DataFrame(tf_table_rows).set_index("周期"))

    # ================== 当前交易建议 ==================
    st.subheader("🎯 当前模型多空方向 & 止盈止损参考")

    # 使用主周期数据 + 最新因子
    df_main = df_main.dropna(subset=["signal_score", "atr"])
    last_row = df_main.iloc[-1]
    last_signal = float(last_row["signal_score"])
    last_atr = float(last_row["atr"])
    last_time = df_main.index[-1]

    direction = "观望"
    bias_text = ""
    if last_signal >= long_th:
        direction = "偏多（做多优先）"
        if last_signal >= (long_th + 20):
            bias_text = "多头趋势+动能都较强，适合顺势做多，但注意追高风险。"
        else:
            bias_text = "多头信号有效，但强度一般，可考虑分批建仓。"
    elif last_signal <= short_th:
        direction = "偏空（做空/做空对冲）"
        if last_signal <= (short_th - 20):
            bias_text = "空头趋势明显，反弹多为离场/加空机会。"
        else:
            bias_text = "空头信号有效，但力度一般，适合轻仓试空或做对冲。"
    else:
        direction = "观望（信号不明确）"
        bias_text = "多空力量暂时均衡，不宜激进建仓，可等待更极端的信号。"

    col_signal, col_pos = st.columns(2)

    with col_signal:
        st.markdown(
            f"""
- 当前时间（主周期）: **{last_time.strftime('%Y-%m-%d %H:%M:%S %Z')}**
- 最新价格: **{last_price:.4f}**
- 当前 signal_score: **{last_signal:.1f}**
- 模型方向判断：**{direction}**
"""
        )
        st.caption(bias_text)

        if last_atr > 0:
            stop_long = last_price - ATR_MULTIPLIER * last_atr
            tp_long = last_price + ATR_MULTIPLIER * TAKE_PROFIT_R_MULTIPLE * last_atr
            stop_short = last_price + ATR_MULTIPLIER * last_atr
            tp_short = last_price - ATR_MULTIPLIER * TAKE_PROFIT_R_MULTIPLE * last_atr

            st.markdown("**参考止盈止损（基于 ATR 波动）:**")
            st.markdown(
                f"""
- 若做多：建议止损约 **{stop_long:.4f}**，参考止盈约 **{tp_long:.4f}**
- 若做空：建议止损约 **{stop_short:.4f}**，参考止盈约 **{tp_short:.4f}**
"""
            )
        else:
            st.caption("ATR 为 0，无法给出合理的止损/止盈价格。")

    # ================== 回测 + 仓位建议 ==================
    st.subheader("📈 简单因子打分回测（近约 3 个月）")

    trades, eq_series = backtest_strategy(
        df_main, long_th=long_th, short_th=short_th
    )
    stats = summarize_trades(trades)

    if not trades or stats is None:
        st.warning("历史数据不足以回测，或当前参数下没有产生足够的信号。")
        return

    df_tr = stats["df_trades"]

    with col_pos:
        # 仓位建议（使用主周期 ATR）
        coins, eff_risk_pct, stop_pct = calc_position_size(
            capital_usd=capital,
            risk_pct=risk_pct,
            price=last_price,
            atr=last_atr,
            atr_mult=ATR_MULTIPLIER,
            kelly=stats["kelly"],
        )

        st.markdown("**建议仓位（基于波动率 + 风险控制）**")
        st.markdown(
            f"""
- 账户资金：**{capital:.2f} USD**
- 有效单笔风险占比（结合 Kelly 微调）：**{eff_risk_pct:.2f}%**
- 对应价格跌幅/涨幅止损距离约：**{stop_pct:.2f}%**
- 建议单次仓位：**{coins:.4f} {inst_id.split('-')[0]}** （约 {coins * last_price:.2f} USD）
"""
        )
        st.caption("说明：若波动率增大，模型会自动降低建议仓位规模，以控制每笔最大损失。")

    # 关键回测指标
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**回测概览（主周期）**")
        st.metric("交易笔数", stats["n_trades"])
        st.metric("胜率", f"{stats['win_rate'] * 100: .1f}%")
        st.metric("平均每笔收益", f"{stats['avg_pnl']: .2f}%")

    with col_r:
        st.metric("Profit Factor", f"{stats['profit_factor']: .2f}")
        st.metric("最大回撤（近似）", f"{stats['max_dd']: .1f}%")
        if stats["kelly"] is not None and not np.isnan(stats["kelly"]):
            st.metric("Kelly 估计 (理论仓位比例)", f"{stats['kelly'] * 100: .1f}%")
        else:
            st.metric("Kelly 估计", "数据不足")

    # 净值曲线
    if not eq_series.empty:
        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=eq_series.index,
                y=eq_series.values,
                mode="lines",
                name="净值",
                line=dict(color="#42a5f5"),
            )
        )
        fig_eq.update_layout(
            title="如果过去都机械执行模型信号，净值曲线大致会长这样（初始净值=1.0）",
            xaxis_title="时间",
            yaxis_title="净值",
            template="plotly_dark",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # 最近 N 笔盈亏分布直方图
    st.markdown("### 📉 最近 N 笔信号盈亏分布")

    df_recent_tr = df_tr.tail(n_signals)
    fig_hist = px.histogram(
        df_recent_tr,
        x="pnl_pct",
        nbins=20,
        title=f"最近 {len(df_recent_tr)} 笔交易盈亏分布（单位：%）",
    )
    fig_hist.update_layout(
        template="plotly_dark",
        xaxis_title="单笔收益率 (%)",
        yaxis_title="次数",
        bargap=0.05,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.caption(
        "观察直方图的偏度和长尾，可以直观感受这套因子模型偏向“高胜率小盈亏”还是“低胜率大盈亏”。"
    )

    # 展示回测交易表（可选）
    with st.expander("查看完整回测交易明细"):
        st.dataframe(
            df_tr[
                [
                    "entry_time",
                    "exit_time",
                    "direction",
                    "entry_price",
                    "exit_price",
                    "pnl_pct",
                    "reason",
                ]
            ].sort_values("entry_time"),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
