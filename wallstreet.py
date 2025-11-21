import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import talib
import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# =========================
# 全局配置
# =========================

DEFAULT_PAIRS = [
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "XRP-USDT",
    "ADA-USDT",
    "DOGE-USDT"
]

TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# 多周期权重：越长周期权重越大
TF_WEIGHTS = {
    "15m": 0.1,   # 短线噪音多，权重较低
    "1h": 0.2,
    "4h": 0.3,    # 波段核心周期
    "1d": 0.4     # 趋势中枢
}

MAX_LIMIT = 1500           # 单次从 OKX 拉取的最大K线数量
FEE_RATE = 0.0005          # 模拟交易手续费（单边 0.05%）
MIN_BARS_FOR_FACTORS = 60  # 起码要有这么多K线才谈得上因子


# =========================
# 工具函数：OKX 数据获取
# =========================

def tf_to_okx_bar(tf: str) -> str:
    """将自定义周期转成 OKX bar 参数"""
    # OKX bar: 1m, 5m, 15m, 1H, 4H, 1D, ...
    if tf.endswith("m"):   # 分钟
        return tf
    if tf.endswith("h"):   # 小时
        return tf[:-1] + "H"
    if tf.endswith("d"):   # 日
        return tf[:-1] + "D"
    return tf


def estimate_bars(tf: str, days: int) -> int:
    """估算回测期需要多少根K线，最多不超过 MAX_LIMIT"""
    if tf.endswith("m"):
        minutes = int(tf[:-1])
        bars_per_day = 24 * 60 // minutes
    elif tf.endswith("h"):
        hours = int(tf[:-1])
        bars_per_day = 24 // hours
    elif tf.endswith("d"):
        bars_per_day = 1
    else:
        bars_per_day = 24
    return min(MAX_LIMIT, bars_per_day * days + 100)


@st.cache_data(ttl=180)
def fetch_okx_klines(inst_id: str, tf: str, limit: int = 500) -> pd.DataFrame | None:
    """
    从 OKX 公共 REST 接口拉取 K 线数据
    inst_id 例：BTC-USDT
    tf      例：15m / 1h / 4h / 1d
    """
    url = "https://www.okx.com/api/v5/market/candles"
    params = {
        "instId": inst_id,
        "bar": tf_to_okx_bar(tf),
        "limit": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10)
    except Exception as e:
        st.error(f"请求 OKX 失败：{e}")
        return None

    if r.status_code != 200:
        st.error(f"OKX HTTP 错误：{r.status_code}")
        return None

    js = r.json()
    if js.get("code") != "0":
        st.error(f"OKX API 错误：{js.get('msg')}")
        return None

    data = js.get("data", [])
    if not data:
        st.warning("OKX 返回空数据")
        return None

    cols = [
        "ts", "open", "high", "low",
        "close", "volume", "volCcy",
        "volCcyQuote", "confirm"
    ]
    df = pd.DataFrame(data, columns=cols)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")

    float_cols = ["open", "high", "low", "close", "volume"]
    for c in float_cols:
        df[c] = df[c].astype(float)

    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    return df


# =========================
# 市场情绪 & 全市场指数
# =========================

@st.cache_data(ttl=600)
def fetch_fear_greed():
    """贪婪与恐惧指数（alternative.me）"""
    url = "https://api.alternative.me/fng/"
    try:
        r = requests.get(url, timeout=10)
        js = r.json()
        d = js["data"][0]
        return {
            "value": int(d["value"]),
            "classification": d["value_classification"],
            "timestamp": datetime.fromtimestamp(int(d["timestamp"]))
        }
    except Exception as e:
        st.warning(f"贪婪与恐惧指数获取失败：{e}")
        return None


@st.cache_data(ttl=600)
def fetch_global_market():
    """全市场指标（用 CoinGecko 免费 API 代替 CMC，效果类似）"""
    url = "https://api.coingecko.com/api/v3/global"
    try:
        r = requests.get(url, timeout=10)
        js = r.json()["data"]
        mcap = js["total_market_cap"]["usd"]
        vol = js["total_volume"]["usd"]
        btc_dom = js["market_cap_percentage"]["btc"]
        mcap_chg = js["market_cap_change_percentage_24h_usd"]
        return {
            "mcap": mcap,
            "volume": vol,
            "btc_dom": btc_dom,
            "mcap_change_24h": mcap_chg,
            "active_coins": js.get("active_cryptocurrencies")
        }
    except Exception as e:
        st.warning(f"全市场指数获取失败：{e}")
        return None


# =========================
# 多因子计算
# =========================

def compute_factor_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    对单一周期K线计算因子时间序列：
    - 趋势因子：EMA 斜率 + MACD + ADX
    - 反转因子：RSI + Bollinger 位置
    - 波动率因子：近 20 根收益波动 vs 历史中位数
    - 综合打分：[-100, 100]
    """
    if df is None or len(df) < MIN_BARS_FOR_FACTORS:
        return pd.DataFrame(index=df.index if df is not None else None)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    rsi = talib.RSI(close, timeperiod=14)
    adx = talib.ADX(high, low, close, timeperiod=14)
    ema_fast = talib.EMA(close, timeperiod=20)
    ema_slow = talib.EMA(close, timeperiod=50)
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    atr = talib.ATR(high, low, close, timeperiod=14)
    bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

    ret = pd.Series(close, index=df.index).pct_change()
    vol20 = ret.rolling(20).std()

    fac = pd.DataFrame(index=df.index)
    fac["rsi"] = rsi
    fac["adx"] = adx
    fac["ema_fast"] = ema_fast
    fac["ema_slow"] = ema_slow
    fac["macd"] = macd
    fac["macd_signal"] = macd_signal
    fac["macd_hist"] = macd_hist
    fac["atr"] = atr
    fac["bb_upper"] = bb_upper
    fac["bb_mid"] = bb_mid
    fac["bb_lower"] = bb_lower
    fac["volatility"] = vol20

    fac["ema_slope"] = (fac["ema_fast"] - fac["ema_slow"]) / fac["ema_slow"]
    fac["bb_position"] = (df["close"] - fac["bb_lower"]) / (fac["bb_upper"] - fac["bb_lower"])
    fac["bb_position"] = fac["bb_position"].clip(0, 1)

    # 趋势因子：EMA斜率 + MACD + ADX
    trend_raw = np.zeros(len(df))

    # EMA斜率：趋势越陡，越接近 +-1
    trend_raw += np.tanh(fac["ema_slope"].fillna(0) * 50)

    # MACD 动量：按波动标准化
    macd_std = fac["macd_hist"].rolling(50).std()
    macd_norm = fac["macd_hist"] / (macd_std + 1e-8)
    trend_raw += np.tanh(macd_norm.fillna(0))

    # ADX：趋势强度，>20 认为有趋势
    adx_comp = (fac["adx"] - 20) / 25
    adx_comp[fac["adx"] < 20] = 0
    trend_raw += adx_comp.fillna(0)

    fac["trend_score"] = (trend_raw * 20).clip(-50, 50)

    # 反转因子：RSI + Bollinger 位置
    reversal_raw = np.zeros(len(df))
    reversal_raw += (50 - fac["rsi"]) / 25.0              # RSI < 50 → 正分（偏多反转）
    reversal_raw += (0.5 - fac["bb_position"]) * 2.0      # 接近下轨 → 正分
    fac["reversal_score"] = (reversal_raw * 20).clip(-50, 50)

    # 波动率因子：当前波动 vs 历史中位数
    base_vol = fac["volatility"].rolling(100).median()
    vol_ratio = fac["volatility"] / (base_vol + 1e-8)
    fac["volatility_score"] = ((vol_ratio - 1.0) * 30).clip(-50, 50)

    # 综合评分：趋势 50% + 反转 30% + 波动率方向性加权 20%
    comp = (
        0.5 * fac["trend_score"] +
        0.3 * fac["reversal_score"] +
        0.2 * np.sign(fac["trend_score"]) * fac["volatility_score"].abs()
    )
    fac["composite_score"] = comp.clip(-100, 100)

    return fac


def get_latest_factors_all_timeframes(dfs: dict) -> pd.DataFrame:
    """对每个周期提取最新一条因子值，组成一个表"""
    rows = []
    for tf, df in dfs.items():
        fac = compute_factor_series(df)
        if fac is None or fac.empty:
            continue
        last = fac.iloc[-1]
        rows.append({
            "timeframe": tf,
            "price": df["close"].iloc[-1],
            "trend_score": last["trend_score"],
            "reversal_score": last["reversal_score"],
            "volatility_score": last["volatility_score"],
            "composite_score": last["composite_score"],
            "rsi": last["rsi"],
            "adx": last["adx"],
            "atr": last["atr"],
            "bb_position": last["bb_position"]
        })
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).set_index("timeframe")
    return table


def aggregate_score(factor_table: pd.DataFrame, weights: dict) -> float:
    """按周期权重对综合评分进行加权平均"""
    if factor_table is None or factor_table.empty:
        return 0.0
    s, w_sum = 0.0, 0.0
    for tf, w in weights.items():
        if tf in factor_table.index:
            s += factor_table.loc[tf, "composite_score"] * w
            w_sum += w
    if w_sum == 0:
        return 0.0
    return float(s / w_sum)


def score_to_bias(score: float, long_thr: float, short_thr: float) -> str:
    """把综合分数转成简单多空意见"""
    if score >= long_thr:
        return "偏多"
    if score <= short_thr:
        return "偏空"
    return "震荡/观望"


# =========================
# 实时信号 & 仓位建议
# =========================

def generate_realtime_signal(
    inst_id: str,
    dfs: dict,
    main_tf: str,
    capital: float,
    long_thr: float,
    short_thr: float,
    sl_mult: float,
    tp_mult: float,
    risk_frac: float
):
    """多周期综合信号 + 主周期仓位建议"""
    factor_table = get_latest_factors_all_timeframes(dfs)
    agg_score = aggregate_score(factor_table, TF_WEIGHTS)

    main_df = dfs[main_tf]
    main_fac = compute_factor_series(main_df)
    if main_fac is None or main_fac.empty:
        return {
            "direction": None,
            "agg_score": agg_score,
            "factor_table": factor_table,
            "main_factors": None
        }

    last_fac = main_fac.iloc[-1]
    price = float(main_df["close"].iloc[-1])
    atr = float(last_fac["atr"])

    direction = None
    if agg_score >= long_thr:
        direction = "long"
    elif agg_score <= short_thr:
        direction = "short"

    if direction is None or np.isnan(atr) or atr <= 0:
        return {
            "direction": None,
            "agg_score": agg_score,
            "factor_table": factor_table,
            "main_factors": last_fac,
            "price": price,
            "position_size": 0.0,
            "stop_loss": None,
            "take_profit": None
        }

    # 价位 & 止盈止损
    if direction == "long":
        stop_loss = price - sl_mult * atr
        take_profit = price + tp_mult * atr
    else:
        stop_loss = price + sl_mult * atr
        take_profit = price - tp_mult * atr

    # 仓位：按风险金额 = capital * risk_frac
    risk_amount = capital * risk_frac
    unit_risk = abs(price - stop_loss)
    if unit_risk <= 0:
        size = 0.0
    else:
        size = risk_amount / unit_risk

    base = inst_id.split("-")[0]
    if base == "BTC":
        size = round(size, 4)
    elif base in ["ETH", "SOL"]:
        size = round(size, 3)
    else:
        size = round(size, 0)

    return {
        "direction": direction,
        "agg_score": agg_score,
        "factor_table": factor_table,
        "main_factors": last_fac,
        "price": price,
        "position_size": size,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }


# =========================
# 回测引擎（主周期）
# =========================

def backtest_on_dataframe(
    df: pd.DataFrame,
    long_thr: float,
    short_thr: float,
    sl_mult: float,
    tp_mult: float,
    init_capital: float,
    risk_frac: float,
    max_holding_bars: int = 40
):
    """
    在指定周期 df 上做回测：
    - 依据 composite_score 触发开仓
    - 按 ATR 设置止盈止损
    - 使用下一根K线的高低价判断是否触及止损/止盈
    - 单次最多一笔仓位
    """
    fac = compute_factor_series(df)
    if fac is None or fac.empty:
        return None, None

    # 找到第一个因子齐全的 index
    valid = ~fac["composite_score"].isna()
    if not valid.any():
        return None, None
    start_idx = np.where(valid.values)[0][0]

    capital = init_capital
    equity_list = [capital]
    equity_index = [df.index[start_idx]]

    trades = []
    position = None  # 当前持仓

    for i in range(start_idx, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]
        frow = fac.iloc[i]

        if position is None:
            score = frow["composite_score"]
            direction = None
            if score >= long_thr:
                direction = "long"
            elif score <= short_thr:
                direction = "short"

            if direction is not None and not np.isnan(frow["atr"]) and frow["atr"] > 0:
                entry_price = float(row["close"])
                atr = float(frow["atr"])

                if direction == "long":
                    sl = entry_price - sl_mult * atr
                    tp = entry_price + tp_mult * atr
                else:
                    sl = entry_price + sl_mult * atr
                    tp = entry_price - tp_mult * atr

                unit_risk = abs(entry_price - sl)
                if unit_risk <= 0:
                    equity_list.append(capital)
                    equity_index.append(nxt.name)
                    continue

                risk_amount = capital * risk_frac
                size = risk_amount / unit_risk

                position = {
                    "entry_time": row.name,
                    "entry_price": entry_price,
                    "direction": direction,
                    "sl": sl,
                    "tp": tp,
                    "size": size,
                    "entry_idx": i
                }

        else:
            # 检查平仓
            exit_price = None
            reason = None
            high = float(nxt["high"])
            low = float(nxt["low"])

            if position["direction"] == "long":
                if low <= position["sl"]:
                    exit_price = position["sl"]
                    reason = "stop"
                elif high >= position["tp"]:
                    exit_price = position["tp"]
                    reason = "take_profit"
            else:  # short
                if high >= position["sl"]:
                    exit_price = position["sl"]
                    reason = "stop"
                elif low <= position["tp"]:
                    exit_price = position["tp"]
                    reason = "take_profit"

            # 时间止盈：超出最大持仓K线数
            if exit_price is None and (i + 1 - position["entry_idx"] >= max_holding_bars):
                exit_price = float(nxt["close"])
                reason = "time_exit"

            if exit_price is not None:
                if position["direction"] == "long":
                    gross = (exit_price - position["entry_price"]) * position["size"]
                else:
                    gross = (position["entry_price"] - exit_price) * position["size"]

                notional = position["entry_price"] * position["size"]
                fees = notional * FEE_RATE * 2
                pnl = gross - fees
                gross_exposure = notional
                ret_pct = pnl / (gross_exposure + 1e-8) * 100

                capital += pnl
                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": nxt.name,
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "size": position["size"],
                    "pnl": pnl,
                    "return_pct": ret_pct,
                    "reason": reason
                })
                position = None

        equity_list.append(capital)
        equity_index.append(nxt.name)

    equity_series = pd.Series(equity_list, index=equity_index)
    trades_df = pd.DataFrame(trades)
    return equity_series, trades_df


def compute_trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {}

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    win_rate = len(wins) / len(trades) * 100
    avg_pnl = trades["pnl"].mean()
    avg_ret = trades["return_pct"].mean()
    total_pnl = trades["pnl"].sum()

    cum = trades["pnl"].cumsum()
    peak = cum.cummax()
    drawdown = cum - peak
    max_dd = -drawdown.min() if len(drawdown) > 0 else 0.0

    # 把每笔当作“独立样本”，粗略年化夏普
    sharpe = 0.0
    if trades["return_pct"].std() > 0:
        sharpe = (trades["return_pct"].mean() /
                  trades["return_pct"].std()) * np.sqrt(252)

    return {
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_ret": avg_ret,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "n_trades": len(trades)
    }


# =========================
# Streamlit 页面布局
# =========================

st.set_page_config(
    page_title="📈 华尔街级加密量化分析助手 · 升级版",
    layout="wide"
)

st.title("📈 华尔街级加密量化分析助手 · 多周期因子 + 回测升级版")
st.caption("实时 OKX 行情 · 多因子多周期模型 · 机械回测 · 无实盘下单（纯分析模式）")

# 侧边栏：策略配置
st.sidebar.header("🔧 策略配置")

selected_pair = st.sidebar.selectbox(
    "选择交易对（OKX 现货）",
    DEFAULT_PAIRS,
    index=0
)

main_timeframe = st.sidebar.selectbox(
    "主交易周期（用于仓位 & 回测）",
    TIMEFRAMES,
    index=2  # 默认 4h
)

capital_input = st.sidebar.number_input(
    "账户资金规模 (USD)",
    min_value=100.0,
    max_value=1_000_000.0,
    value=10_000.0,
    step=1_000.0
)

risk_fraction = st.sidebar.slider(
    "单笔最大风险占比",
    min_value=0.005,
    max_value=0.05,
    value=0.02,
    step=0.005,
    format="%.3f"
)

long_threshold = st.sidebar.slider(
    "做多信号阈值",
    min_value=10,
    max_value=80,
    value=30,
    step=5
)

short_threshold = st.sidebar.slider(
    "做空信号阈值",
    min_value=-80,
    max_value=-10,
    value=-30,
    step=5
)

atr_sl_mult = st.sidebar.slider(
    "ATR 止损倍数",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.1
)

atr_tp_mult = st.sidebar.slider(
    "ATR 止盈倍数",
    min_value=0.5,
    max_value=8.0,
    value=3.0,
    step=0.1
)

backtest_days = st.sidebar.slider(
    "回测区间（按主周期，近多少天）",
    min_value=30,
    max_value=365,
    value=90,
    step=15
)

max_holding_bars = st.sidebar.slider(
    "最大持仓K线数（时间止盈）",
    min_value=5,
    max_value=200,
    value=40,
    step=5
)

n_hist_trades = st.sidebar.slider(
    "最近 N 笔交易用于盈亏分布",
    min_value=20,
    max_value=300,
    value=100,
    step=10
)

st.sidebar.markdown("---")
st.sidebar.caption("本工具仅做量化分析与回测示范，不构成任何投资建议。")


# =========================
# 数据获取
# =========================

st.info(f"正在从 OKX 获取 {selected_pair} 的多周期行情数据……")

dfs = {}
for tf in TIMEFRAMES:
    if tf == main_timeframe:
        limit = estimate_bars(tf, backtest_days)
    else:
        limit = 400
    dfs[tf] = fetch_okx_klines(selected_pair, tf, limit=limit)

if any((df is None or df.empty) for df in dfs.values()):
    st.error("❌ 部分周期数据获取失败，请稍后重试或检查网络。")
    st.stop()

# 主周期 DataFrame
main_df = dfs[main_timeframe]

# 贪婪恐惧 & 全市场
fg = fetch_fear_greed()
global_mkt = fetch_global_market()


# =========================
# 上半部分：K线 + 实时信号
# =========================

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader(f"📊 {selected_pair} · {main_timeframe} 主周期 K 线 & 指标")

    fac_main = compute_factor_series(main_df)
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=main_df.index,
        open=main_df["open"],
        high=main_df["high"],
        low=main_df["low"],
        close=main_df["close"],
        name=f"{main_timeframe} K 线",
        increasing_line_color="green",
        decreasing_line_color="red",
        showlegend=True
    ))

    if not fac_main.empty:
        fig.add_trace(go.Scatter(
            x=main_df.index,
            y=fac_main["ema_fast"],
            name="EMA 20",
            line=dict(color="deepskyblue", width=1.2)
        ))
        fig.add_trace(go.Scatter(
            x=main_df.index,
            y=fac_main["ema_slow"],
            name="EMA 50",
            line=dict(color="orange", width=1.2)
        ))

        last_atr = fac_main["atr"].iloc[-1]
        upper_band = main_df["close"] + last_atr * 2
        lower_band = main_df["close"] - last_atr * 2

        fig.add_trace(go.Scatter(
            x=main_df.index,
            y=upper_band,
            name="ATR 上轨",
            line=dict(color="gray", dash="dot"),
            opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=main_df.index,
            y=lower_band,
            name="ATR 下轨",
            line=dict(color="gray", dash="dot"),
            opacity=0.5
        ))

    fig.update_layout(
        height=550,
        xaxis_title="时间",
        yaxis_title="价格 (USDT)",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🎯 多周期综合信号 & 仓位建议")

    signal_info = generate_realtime_signal(
        selected_pair,
        dfs,
        main_timeframe,
        capital_input,
        long_threshold,
        short_threshold,
        atr_sl_mult,
        atr_tp_mult,
        risk_fraction
    )

    direction = signal_info["direction"]
    agg_score = signal_info["agg_score"]
    price = signal_info.get("price", np.nan)
    size = signal_info.get("position_size", 0.0)
    sl = signal_info.get("stop_loss", None)
    tp = signal_info.get("take_profit", None)
    factor_table = signal_info["factor_table"]
    main_factors = signal_info["main_factors"]

    if direction:
        dir_cn = "做多" if direction == "long" else "做空"
        st.success(f"当前多周期综合信号：**{dir_cn} {selected_pair}**")
        st.metric("多周期综合评分", f"{agg_score:.1f}")
        st.metric("当前价格", f"{price:.4f} USDT")
        st.metric("建议仓位规模", f"{size:.6f} {selected_pair.split('-')[0]}")
        if sl and tp:
            st.metric("止损价", f"{sl:.4f} USDT")
            st.metric("止盈价", f"{tp:.4f} USDT")
    else:
        st.warning("当前无强信号（多周期偏向中性 / 震荡），建议观望或缩小仓位。")
        st.metric("多周期综合评分", f"{agg_score:.1f}")

    if not factor_table.empty:
        st.markdown("**🧬 多周期风格剖面（短线 / 中线 / 波段 / 趋势）**")

        table = factor_table.copy()
        table["bias"] = table["composite_score"].apply(
            lambda s: score_to_bias(s, long_threshold, short_threshold)
        )
        table = table[[
            "price", "trend_score", "reversal_score",
            "volatility_score", "composite_score", "rsi", "adx", "bias"
        ]]

        st.dataframe(
            table.style.format(
                {
                    "price": "{:.4f}",
                    "trend_score": "{:.1f}",
                    "reversal_score": "{:.1f}",
                    "volatility_score": "{:.1f}",
                    "composite_score": "{:.1f}",
                    "rsi": "{:.1f}",
                    "adx": "{:.1f}"
                }
            ),
            use_container_width=True
        )

        # 汇总风格雷达图（按权重求和）
        agg_trend = sum(
            factor_table.loc[tf, "trend_score"] * w
            for tf, w in TF_WEIGHTS.items()
            if tf in factor_table.index
        )
        agg_reversal = sum(
            factor_table.loc[tf, "reversal_score"] * w
            for tf, w in TF_WEIGHTS.items()
            if tf in factor_table.index
        )
        agg_vol = sum(
            factor_table.loc[tf, "volatility_score"] * w
            for tf, w in TF_WEIGHTS.items()
            if tf in factor_table.index
        )

        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=[agg_trend, agg_reversal, agg_vol],
            theta=["趋势因子", "反转因子", "波动率因子"],
            fill="toself",
            name="加权风格",
            line=dict(color="cyan")
        ))
        radar_fig.update_layout(
            title="多因子风格剖面（加权）",
            polar=dict(
                radialaxis=dict(visible=True, range=[-60, 60])
            ),
            showlegend=False,
            height=320,
            template="plotly_dark"
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    if main_factors is not None:
        st.markdown("**📌 解析当前主周期信号逻辑（像首席分析师一样解释给自己听）**")
        explain_lines = []
        explain_lines.append(
            f"- 趋势因子：ADX ≈ {main_factors['adx']:.1f}，EMA20/50 斜率 {main_factors['ema_slope'] * 100:.2f}%"
        )
        explain_lines.append(
            f"- 反转因子：RSI ≈ {main_factors['rsi']:.1f}，价格位于布林带 {main_factors['bb_position'] * 100:.1f}% 位置"
        )
        explain_lines.append(
            f"- 波动率因子：近 20 根收益波动率 ≈ {main_factors['volatility'] * 100:.2f}%"
        )
        st.markdown("\n".join(explain_lines))


# =========================
# 中部：回测 & 盈亏分布
# =========================

st.markdown("---")
st.subheader(f"📈 机械执行回测：过去 {backtest_days} 天（主周期 {main_timeframe}）")

# 剪切主周期数据到指定天数
cutoff = main_df.index[-1] - timedelta(days=backtest_days)
bt_df = main_df[main_df.index >= cutoff]

if len(bt_df) < MIN_BARS_FOR_FACTORS + 10:
    st.warning("主周期数据长度不足，无法进行有效回测。请尝试缩短回测区间或选择更长周期。")
else:
    with st.spinner("正在运行历史回测引擎（只算不下单）……"):
        equity, trades = backtest_on_dataframe(
            bt_df,
            long_threshold,
            short_threshold,
            atr_sl_mult,
            atr_tp_mult,
            capital_input,
            risk_fraction,
            max_holding_bars=max_holding_bars
        )

    if equity is None or trades is None or trades.empty:
        st.warning("回测期间没有产生有效交易（可能阈值过高或市场极度震荡）。")
    else:
        stats = compute_trade_stats(trades)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总交易笔数", f"{stats['n_trades']}")
        with col2:
            st.metric("胜率", f"{stats['win_rate']:.1f}%")
        with col3:
            st.metric("累计收益", f"{stats['total_pnl']:.2f} USDT")
        with col4:
            st.metric("最大回撤", f"{stats['max_drawdown']:.2f} USDT")

        col5, col6 = st.columns(2)
        with col5:
            st.metric("单笔平均收益", f"{stats['avg_pnl']:.2f} USDT")
        with col6:
            st.metric("单笔平均收益率", f"{stats['avg_ret']:.2f}%")

        # 净值曲线
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="模拟净值",
            line=dict(color="gold", width=2)
        ))
        fig_eq.add_hline(
            y=capital_input,
            line=dict(color="gray", dash="dash"),
            annotation_text="初始资金",
            annotation_position="bottom right"
        )
        fig_eq.update_layout(
            title="如果过去这段时间全部机械执行，会长成怎样的净值曲线？",
            xaxis_title="时间",
            yaxis_title="账户权益 (USDT)",
            height=400,
            template="plotly_dark"
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # 最近 N 笔交易盈亏分布
        st.subheader(f"📊 最近 {n_hist_trades} 笔信号的盈亏分布")
        trades_hist = trades.tail(n_hist_trades)
        fig_hist = px.histogram(
            trades_hist,
            x="pnl",
            nbins=20,
            title="盈亏分布直方图",
            color_discrete_sequence=["#00FF99"]
        )
        fig_hist.add_vline(
            x=trades_hist["pnl"].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text="平均值"
        )
        fig_hist.add_vline(
            x=0,
            line_dash="dot",
            line_color="white",
            annotation_text="盈亏平衡"
        )
        fig_hist.update_layout(
            xaxis_title="单笔盈亏 (USDT)",
            yaxis_title="频数",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # 交易明细表（可选）
        with st.expander("查看详细交易记录（开平仓时间、方向、盈亏等）"):
            show_cols = [
                "entry_time", "exit_time", "direction",
                "entry_price", "exit_price", "size",
                "pnl", "return_pct", "reason"
            ]
            st.dataframe(
                trades[show_cols].sort_values("entry_time"),
                use_container_width=True
            )


# =========================
# 情绪 & 全市场指数
# =========================

st.markdown("---")
st.subheader("🧠 市场情绪 & 全市场环境")

col_a, col_b = st.columns([1, 2])

with col_a:
    if fg:
        color = "green" if fg["value"] >= 70 else "red" if fg["value"] <= 30 else "yellow"
        st.markdown(
            f"""
            <div style="text-align:center; padding:18px; background-color:{color}20;
                        border-radius:10px; border:1px solid {color}">
                <h4 style="color:{color}; margin-bottom:0;">贪婪与恐惧指数</h4>
                <h2 style="color:{color}; margin:4px 0;">{fg['value']}</h2>
                <p style="color:white; margin:0;">{fg['classification']}</p>
                <small style="color:lightgray;">
                    更新时间：{fg['timestamp'].strftime('%Y-%m-%d %H:%M')}
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("暂时无法获取贪婪与恐惧指数。")

    if global_mkt:
        st.markdown("---")
        st.markdown("**🌍 全市场概览（来自 CoinGecko）**")

        mcap = global_mkt["mcap"]
        vol = global_mkt["volume"]
        btc_dom = global_mkt["btc_dom"]
        chg = global_mkt["mcap_change_24h"]

        def fmt(num):
            if num >= 1e12:
                return f"{num / 1e12:.2f} 万亿"
            if num >= 1e9:
                return f"{num / 1e9:.2f} 十亿"
            if num >= 1e6:
                return f"{num / 1e6:.2f} 百万"
            return f"{num:.0f}"

        st.metric("加密总市值", f"{fmt(mcap)} USD", f"{chg:+.2f}%/24h")
        st.metric("24h 总成交额", f"{fmt(vol)} USD")
        st.metric("BTC 主导率", f"{btc_dom:.2f}%")

with col_b:
    st.markdown("""
    **如何把情绪与因子结合？**

    - 当 **多周期综合评分 > 做多阈值** 且 **贪婪指数 > 70**：  
      → 技术面偏多 + 情绪极度贪婪，适合**控制仓位、严格止盈**，防御“最后一冲”。

    - 当 **综合评分 < 做空阈值** 且 **贪婪指数 < 20**：  
      → 技术面偏空 + 情绪极度恐惧，容易出现**情绪底 / 左侧机会**，可以用分批建仓 + 更宽止损。

    - 若 **BTC 主导率上升 & 总市值下跌**：  
      → 资金回流 BTC、防御环境，山寨风险更大；模型信号建议对小币更保守。

    量化的意义，不是预测每一根 K 线，而是：  
    在任何时刻，**清楚自己站在什么环境、持有什么风格、承担多大风险**。
    """)

# =========================
# 页脚
# =========================

st.markdown("---")
st.caption("""
💡 免责声明：本应用仅用于量化研究与教学示范，不构成任何投资建议。  
加密市场波动极大，请务必控制仓位、严格止损，对自己的资金负责。

你现在已经拥有了一套「华尔街级」的多因子决策终端的雏形：
- 多周期一致性 → 决定方向与节奏  
- 因子风格剖面 → 告诉你是在做趋势还是做反转  
- 机械回测 → 把感觉变成统计  
- 情绪指标 → 防止在极端情绪中失去理性  

接下来可以继续玩的升级方向（依旧保持**不接实盘**）：
- 加入多币种「组合回测」，看如果同时按模型交易 BTC+ETH，会怎样；
- 加入「参数扫描 / 网格搜索」，自动找出某段时间内表现最好的阈值组合；
- 加一个「策略对比面板」，把两套不同参数的净值曲线叠在一起。

如果你想，我可以下一步帮你：
- 把回测部分抽成独立模块，便于以后接多种策略；
- 或者给你加一个「参数扫描页面」，一键看哪种风格最适合当前市场。
""")
