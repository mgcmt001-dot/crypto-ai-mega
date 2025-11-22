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

# 所有分析周期
TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# 固定主周期用于：K线 & 回测
MAIN_TIMEFRAME = "4h"

# 多周期权重：越长周期权重越大
TF_WEIGHTS = {
    "15m": 0.1,
    "1h": 0.2,
    "4h": 0.3,
    "1d": 0.4
}

MAX_LIMIT = 1500
FEE_RATE = 0.0005           # 手续费假设（单边 0.05%）
MIN_BARS_FOR_FACTORS = 60   # 起码要有这么多K线才算有因子
INIT_CAPITAL = 10000.0      # 回测虚拟初始资金（页面不展示）

# ==== 固定策略参数（更稳妥，不激进） ====
# 综合评分 ≥ LONG_THRESHOLD → 多头；≤ SHORT_THRESHOLD → 空头
LONG_THRESHOLD = 30
SHORT_THRESHOLD = -30

# 止盈止损 ATR 倍数
ATR_SL_MULT = 2.0
ATR_TP_MULT = 3.0

# 回测：单笔风险占比、回测天数、最大持仓K线数、盈亏分布显示最近 N 笔
RISK_FRACTION = 0.02
BACKTEST_DAYS = 90
MAX_HOLDING_BARS = 40
N_HIST_TRADES = 100

# 本周期最近 N 根涨跌幅
PERIOD_RET_LOOKBACK = 20
# “本月高低点百分位”的窗口（用近 30 天近似）
MONTH_WINDOW_DAYS = 30

# 用于经验概率分析：每个周期前瞻多少根 K 线
PROB_HORIZON_BARS = {
    "15m": 8,   # 约 2 小时
    "1h": 6,    # 约 6 小时
    "4h": 6,    # 约 1 天
    "1d": 3     # 约 3 天
}

# 时间框架说明（卡片用）
TF_DESC = {
    "15m": "超短线",
    "1h": "日内",
    "4h": "波段",
    "1d": "趋势"
}


# =========================
# 工具函数：OKX 数据获取
# =========================

def tf_to_okx_bar(tf: str) -> str:
    """将自定义周期转成 OKX bar 参数"""
    if tf.endswith("m"):
        return tf
    if tf.endswith("h"):
        return tf[:-1] + "H"
    if tf.endswith("d"):
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
def fetch_okx_klines(inst_id: str, tf: str, limit: int = 500):
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
    """全市场指标（用 CoinGecko 免费 API 代替 CMC）"""
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
# 多因子计算 & 新增统计方法
# =========================

def compute_factor_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    对单一周期K线计算因子时间序列：
    - 趋势因子：EMA 斜率 + MACD + ADX
    - 反转因子：RSI + Bollinger 位置
    - 波动率因子：近 20 根收益波动
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
    macd, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    atr = talib.ATR(high, low, close, timeperiod=14)
    bb_upper, bb_mid, bb_lower = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2
    )

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

    trend_raw += np.tanh(fac["ema_slope"].fillna(0) * 50)

    macd_std = fac["macd_hist"].rolling(50).std()
    macd_norm = fac["macd_hist"] / (macd_std + 1e-8)
    trend_raw += np.tanh(macd_norm.fillna(0))

    adx_comp = (fac["adx"] - 20) / 25
    adx_comp[fac["adx"] < 20] = 0
    trend_raw += adx_comp.fillna(0)

    fac["trend_score"] = (trend_raw * 20).clip(-50, 50)

    # 反转因子
    reversal_raw = np.zeros(len(df))
    reversal_raw += (50 - fac["rsi"]) / 25.0
    reversal_raw += (0.5 - fac["bb_position"]) * 2.0
    fac["reversal_score"] = (reversal_raw * 20).clip(-50, 50)

    # 波动率因子
    base_vol = fac["volatility"].rolling(100).median()
    vol_ratio = fac["volatility"] / (base_vol + 1e-8)
    fac["volatility_score"] = ((vol_ratio - 1.0) * 30).clip(-50, 50)

    comp = (
        0.5 * fac["trend_score"] +
        0.3 * fac["reversal_score"] +
        0.2 * np.sign(fac["trend_score"]) * fac["volatility_score"].abs()
    )
    fac["composite_score"] = comp.clip(-100, 100)

    return fac


def score_to_bias(score: float, long_thr: float, short_thr: float) -> str:
    if score >= long_thr:
        return "偏多"
    if score <= short_thr:
        return "偏空"
    return "震荡/观望"


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


def compute_forward_prob_stats(df: pd.DataFrame, fac: pd.DataFrame,
                               horizon: int, score_window: float = 10.0,
                               min_samples: int = 40):
    """
    经验概率模块：
    给定因子序列 & K线，估计：
    - 当前得分附近（±score_window）在历史上的样本
    - 未来 horizon 根 K 线的经验上涨概率 / 期望收益等
    """
    if df is None or fac is None:
        return None
    if len(df) <= horizon + 5:
        return None
    if "composite_score" not in fac.columns:
        return None

    scores = fac["composite_score"]
    closes = df["close"]

    # 未来 horizon 根的收益：对前 len-horizon 根有效
    fwd_ret = (closes.shift(-horizon) / closes - 1).iloc[:-horizon]
    scores_hist = scores.iloc[:-horizon]

    mask = scores_hist.notna() & fwd_ret.notna()
    scores_hist = scores_hist[mask]
    fwd_ret = fwd_ret[mask]

    if len(fwd_ret) < min_samples:
        return None

    score_now = scores.iloc[-1]
    if pd.isna(score_now):
        return None

    similar = scores_hist.between(score_now - score_window, score_now + score_window)
    if similar.sum() >= min_samples:
        rets = fwd_ret[similar]
        n = similar.sum()
    else:
        rets = fwd_ret
        n = len(fwd_ret)

    if n == 0:
        return None

    prob_up = (rets > 0).mean()
    exp_ret = rets.mean()
    worst_10 = rets.quantile(0.1)
    best_10 = rets.quantile(0.9)

    return {
        "prob_up": float(prob_up),
        "exp_ret": float(exp_ret),
        "worst_10": float(worst_10),
        "best_10": float(best_10),
        "n_samples": int(n)
    }


def build_multi_tf_signals(
    inst_id: str,
    dfs: dict,
    long_thr: float,
    short_thr: float,
    sl_mult: float,
    tp_mult: float
) -> pd.DataFrame:
    """
    对每个周期，独立给出：
    - 因子得分
    - 多空方向
    - 止盈止损点位
    - 本周期最近 N 根的涨跌幅
    - 当前价格在近 MONTH_WINDOW_DAYS 天高低点区间的百分位
    - 当前综合得分在本周期历史中的分位数（动态评估）
    - 基于历史的经验上涨概率 & 期望收益
    """
    rows = []
    for tf, df in dfs.items():
        fac = compute_factor_series(df)
        if fac is None or fac.empty:
            continue
        last = fac.iloc[-1]
        price = float(df["close"].iloc[-1])
        atr = float(last["atr"]) if not np.isnan(last["atr"]) else None
        score = float(last["composite_score"])

        # 方向 & 止盈止损（保守阈值）
        direction = None
        sl = None
        tp = None
        if atr is not None and atr > 0:
            if score >= long_thr:
                direction = "多"
                sl = price - sl_mult * atr
                tp = price + tp_mult * atr
            elif score <= short_thr:
                direction = "空"
                sl = price + sl_mult * atr
                tp = price - tp_mult * atr

        # 近 N 根K线的累计涨跌幅
        if len(df) > PERIOD_RET_LOOKBACK:
            period_ret = df["close"].iloc[-1] / df["close"].iloc[-PERIOD_RET_LOOKBACK] - 1
        else:
            period_ret = np.nan

        # “本月高低点百分位”：近 MONTH_WINDOW_DAYS 天（不够则用全样本）
        if len(df) > 5:
            cutoff = df.index[-1] - timedelta(days=MONTH_WINDOW_DAYS)
            df_win = df[df.index >= cutoff]
            if len(df_win) < 5:
                df_win = df
            hi = df_win["high"].max()
            lo = df_win["low"].min()
            last_close = df_win["close"].iloc[-1]
            if hi > lo:
                month_pct = (last_close - lo) / (hi - lo)
            else:
                month_pct = np.nan
        else:
            month_pct = np.nan

        # 动态评分分位数
        hist_scores = fac["composite_score"].dropna()
        if len(hist_scores) >= 60:
            q25, q50, q75 = hist_scores.quantile([0.25, 0.5, 0.75])
            score_pct = (hist_scores < score).mean()  # 当前得分在历史中的经验分位
        else:
            q25 = q75 = score_pct = np.nan

        # 经验概率统计
        horizon = PROB_HORIZON_BARS.get(tf, 5)
        prob_stats = compute_forward_prob_stats(df, fac, horizon=horizon)
        if prob_stats is not None:
            prob_up = prob_stats["prob_up"]
            exp_ret = prob_stats["exp_ret"]
            n_samples = prob_stats["n_samples"]
        else:
            prob_up = exp_ret = np.nan
            n_samples = 0

        rows.append({
            "timeframe": tf,
            "price": price,
            "trend_score": last["trend_score"],
            "reversal_score": last["reversal_score"],
            "volatility_score": last["volatility_score"],
            "composite_score": score,
            "rsi": last["rsi"],
            "adx": last["adx"],
            "atr": atr,
            "bb_position": last["bb_position"],
            "direction": direction,
            "stop_loss": sl,
            "take_profit": tp,
            "period_return": period_ret,
            "month_percentile": month_pct,
            "score_percentile": score_pct,
            "dyn_long_thr": q75,
            "dyn_short_thr": q25,
            "prob_up": prob_up,
            "exp_ret": exp_ret,
            "prob_n": n_samples
        })

    if not rows:
        return pd.DataFrame()

    df_tf = pd.DataFrame(rows).set_index("timeframe")
    df_tf = df_tf.reindex([tf for tf in TIMEFRAMES if tf in df_tf.index])

    # 把 None 转成 NaN，避免格式化时报 TypeError
    df_tf["stop_loss"] = pd.to_numeric(df_tf["stop_loss"], errors="coerce")
    df_tf["take_profit"] = pd.to_numeric(df_tf["take_profit"], errors="coerce")

    return df_tf


def detect_regime_main(fac: pd.DataFrame):
    """
    基于主周期因子（4h），检测当前市场状态 Regime：
    - 趋势市 / 低波震荡市 / 高波动混乱市 / 过渡阶段
    """
    if fac is None or fac.empty:
        return "未知", "因子数据不足，暂时无法判断市场状态。"

    last = fac.iloc[-1]
    trend = last.get("trend_score", np.nan)
    adx = last.get("adx", np.nan)
    vol_score = last.get("volatility_score", np.nan)

    if pd.isna(trend) or pd.isna(adx) or pd.isna(vol_score):
        return "未知", "关键因子存在缺失值，暂时无法判断市场状态。"

    at = abs(trend)
    av = abs(vol_score)

    # 简单规则：可以后续再细化
    if at > 20 and adx > 25:
        regime = "趋势市"
        comment = "当前 4h 呈明显单边趋势，顺势信号的统计优势通常更好，逆势博弈需要格外谨慎。"
    elif at < 10 and adx < 18 and av < 10:
        regime = "低波震荡市"
        comment = "当前 4h 趋势不强、波动有限，更偏向箱体震荡环境，均值回归/区间交易相对占优。"
    elif av > 20:
        regime = "高波动混乱市"
        comment = "当前 4h 波动率显著放大，方向噪声较多，建议控制杠杆和单笔风险，等待结构更清晰再重仓。"
    else:
        regime = "过渡阶段"
        comment = "当前 4h 介于趋势与震荡之间，处于方向酝酿期，信号可靠性有限，可适度减仓观望。"

    return regime, comment


def build_card_comment(tf: str, row: pd.Series, tf_signals: pd.DataFrame,
                       long_thr: float, short_thr: float) -> list:
    """
    为单个周期卡片生成“有逻辑的分析语句”，
    结合：
    - 本周期因子
    - 与 4h、1d 的多空关系
    - 近 N 根涨跌 + 本月百分位
    - 评分在历史中的分位 & 经验上涨概率
    """
    lines = []

    direction = row["direction"]  # "多" / "空" / None
    score = row["composite_score"]
    trend = row["trend_score"]
    rsi = row["rsi"]
    adx = row["adx"]
    vol_score = row["volatility_score"]
    period_ret = row.get("period_return", np.nan)
    month_pct = row.get("month_percentile", np.nan)
    score_pct = row.get("score_percentile", np.nan)
    prob_up = row.get("prob_up", np.nan)
    exp_ret = row.get("exp_ret", np.nan)

    dir_4h = tf_signals.loc[MAIN_TIMEFRAME, "direction"] if MAIN_TIMEFRAME in tf_signals.index else None
    dir_1d = tf_signals.loc["1d", "direction"] if "1d" in tf_signals.index else None

    # 1）评分所在区间：多 / 空 / 中性
    if pd.notna(score):
        if score >= long_thr:
            lines.append("综合评分偏多，模型在本周期明确倾向多头。")
        elif score <= short_thr:
            lines.append("综合评分偏空，模型在本周期明确倾向空头。")
        else:
            lines.append("综合评分位于中性区间，多空力量大致均衡。")

    # 2）本周期在多周期结构中的角色
    if tf in ["15m", "1h"]:
        if direction in ["多", "空"]:
            if dir_4h == direction and dir_1d == direction:
                lines.append("短周期与 4h、日线同向，是顺大趋势的短线机会。")
            elif dir_4h == direction and (dir_1d is None or pd.isna(dir_1d)):
                lines.append("短周期与 4h 同向，日线中性，适合做波段内部的跟随。")
            elif dir_4h not in [None, direction] and not pd.isna(dir_4h):
                lines.append(f"短周期方向与 4h 相反，更像是{dir_4h}势中的反弹/回调，持仓周期不宜过长。")
            else:
                lines.append("短周期信号相对独立，需结合 4h 与日线综合判断。")
        else:
            if dir_4h in ["多", "空"]:
                lines.append(f"当前短周期无明确信号，但 4h 偏{dir_4h}，可等待短周期与其共振。")

    elif tf == "4h":
        if direction in ["多", "空"] and dir_1d == direction:
            lines.append("4 小时与日线同向，是当前主要趋势方向，适合按该方向做波段主线。")
        elif direction in ["多", "空"] and dir_1d not in [None, direction] and not pd.isna(dir_1d):
            lines.append("4 小时与日线相反，可能处于日线趋势中的中级反弹/中级回调。")
        elif direction is None and dir_1d in ["多", "空"]:
            lines.append(f"4 小时震荡，但日线偏{dir_1d}，更适合等待 4h 方向与日线统一。")
        else:
            lines.append("4 小时与日线都偏中性，更接近箱体震荡环境。")

    elif tf == "1d":
        if pd.notna(trend) and pd.notna(adx):
            if trend > 15 and adx > 25:
                lines.append("日线处于明显上升趋势，趋势因子和 ADX 同时支持多头。")
            elif trend < -15 and adx > 25:
                lines.append("日线处于明显下降趋势，趋势因子和 ADX 同时偏空。")
            elif abs(trend) < 10 and adx < 20:
                lines.append("日线趋势不明显，偏震荡市，大级别不适合追涨杀跌。")
            else:
                lines.append("日线处于趋势与震荡之间的过渡阶段，方向感一般。")

    # 3）近 N 根涨跌幅
    if pd.notna(period_ret):
        if period_ret > 0.1:
            lines.append(f"最近 {PERIOD_RET_LOOKBACK} 根累计涨 {period_ret:.1%}，上升动能较强。")
        elif period_ret < -0.1:
            lines.append(f"最近 {PERIOD_RET_LOOKBACK} 根累计跌 {period_ret:.1%}，处于连续回落之后。")

    # 4）价格在本月高低点区间的位置
    if pd.notna(month_pct):
        if month_pct > 0.8:
            lines.append("当前价接近近期高位，追高风险上升。")
        elif month_pct < 0.2:
            lines.append("当前价接近近期低位，左侧布局意愿会增强。")

    # 5）评分在本周期历史中的分位数
    if pd.notna(score_pct):
        lines.append(f"当前综合评分约高于本周期历史 {score_pct:.1%} 的时间，属于{'偏极端' if score_pct > 0.8 or score_pct < 0.2 else '中等偏'}水平。")

    # 6）经验上涨概率 & 期望收益
    if pd.notna(prob_up) and pd.notna(exp_ret):
        if prob_up > 0.55:
            lines.append(f"在历史上与当前得分相近的情形中，未来同周期约 {prob_up:.1%} 的概率上涨，平均收益约 {exp_ret:.2%}。")
        elif prob_up < 0.45:
            lines.append(f"在历史上与当前得分相近的情形中，未来同周期上涨概率仅约 {prob_up:.1%}，平均收益约 {exp_ret:.2%}。")
        else:
            lines.append(f"历史上相似得分后的涨跌概率接近五五开，平均收益约 {exp_ret:.2%}，统计优势并不明显。")

    # 7）技术细节：RSI / ADX / 波动率
    if pd.notna(rsi):
        if rsi < 30:
            lines.append("RSI 已进入超卖区域，可能存在反弹博弈机会。")
        elif rsi > 70:
            lines.append("RSI 已进入超买区域，短线回调压力增加。")

    if pd.notna(adx):
        if adx > 30:
            lines.append("ADX 偏高，当前处在单边趋势阶段，适合顺势。")
        elif adx < 18:
            lines.append("ADX 偏低，趋势不强，假突破概率较高。")

    if pd.notna(vol_score):
        if vol_score > 10:
            lines.append("波动率放大，收益与回撤都会放大。")
        elif vol_score < -10:
            lines.append("波动率收缩，可能在为下次行情蓄力。")

    if not lines:
        lines.append("当前周期各项因子信号偏弱，暂无明显优势方向。")

    return lines


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
    fac = compute_factor_series(df)
    if fac is None or fac.empty:
        return None, None

    valid = ~fac["composite_score"].isna()
    if not valid.any():
        return None, None
    start_idx = np.where(valid.values)[0][0]

    capital = init_capital
    equity_list = [capital]
    equity_index = [df.index[start_idx]]

    trades = []
    position = None

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
    win_rate = len(wins) / len(trades) * 100
    avg_pnl = trades["pnl"].mean()
    avg_ret = trades["return_pct"].mean()
    total_pnl = trades["pnl"].sum()

    cum = trades["pnl"].cumsum()
    peak = cum.cummax()
    drawdown = cum - peak
    max_dd = -drawdown.min() if len(drawdown) > 0 else 0.0

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
    page_title="📈 华尔街级加密量化分析助手 · 多周期因子版",
    layout="wide"
)

st.title("📈 华尔街级加密量化分析助手 · 多周期因子 & 回测")
st.caption("实时 OKX 行情 · 多周期因子模型 · 统计分析 · 纯分析，不接实盘")

# 侧边栏：只保留币种选择
st.sidebar.header("⚙️ 设置")
selected_pair = st.sidebar.selectbox(
    "选择交易对（OKX 现货）",
    DEFAULT_PAIRS,
    index=0
)
st.sidebar.markdown("---")
st.sidebar.caption("本工具仅作量化分析与回测示范，不涉及真实资金与下单。")

# =========================
# 数据获取 + 显示“抓取时间”（北京时间）
# =========================

fetch_time_utc = pd.Timestamp.now(tz="UTC")

status = st.empty()
status.info(f"正在从 OKX 获取 {selected_pair} 的多周期行情数据……")

dfs = {}
for tf in TIMEFRAMES:
    if tf == MAIN_TIMEFRAME:
        limit = estimate_bars(tf, BACKTEST_DAYS)
    else:
        limit = 400
    dfs[tf] = fetch_okx_klines(selected_pair, tf, limit=limit)

if any((df is None or df.empty) for df in dfs.values()):
    status.error("部分周期数据获取失败，请稍后重试或检查网络。")
    st.error("❌ 数据获取失败。")
    st.stop()

main_df = dfs[MAIN_TIMEFRAME]

try:
    bj_fetch = fetch_time_utc.tz_convert("Asia/Shanghai")
    fetch_str = bj_fetch.strftime("%Y年%m月%d日 %H:%M:%S")

    last_ts = main_df.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    bj_kline = last_ts.tz_convert("Asia/Shanghai")
    kline_str = bj_kline.strftime("%Y年%m月%d日 %H:%M:%S")

    status.success(
        f"已从 OKX 获取 {selected_pair} 多周期数据。"
        f" 抓取时间：{fetch_str}（北京时间），"
        f"最新 {MAIN_TIMEFRAME} K 线时间：{kline_str}（北京时间）"
    )
except Exception:
    status.success(f"已从 OKX 获取 {selected_pair} 多周期数据。")

fg = fetch_fear_greed()
global_mkt = fetch_global_market()

# 预先计算多周期信号 & 主周期因子
tf_signals = build_multi_tf_signals(
    selected_pair, dfs,
    LONG_THRESHOLD, SHORT_THRESHOLD,
    ATR_SL_MULT, ATR_TP_MULT
)
fac_main = compute_factor_series(main_df)

# 主周期 Regime 检测
regime_label, regime_comment = detect_regime_main(fac_main)

# =========================
# 顶部：多周期综述 + Regime
# =========================

st.subheader("🎯 多周期核心信号总览")

if tf_signals.empty:
    st.warning("因子数据不足，暂时无法生成多周期信号。")
else:
    overall_score = aggregate_score(tf_signals, TF_WEIGHTS)
    overall_bias = score_to_bias(overall_score, LONG_THRESHOLD, SHORT_THRESHOLD)

    col_top1, col_top2 = st.columns([1, 2])
    with col_top1:
        st.metric("多周期综合评分（加权）", f"{overall_score:.1f}", overall_bias)
    with col_top2:
        color = "#16c784" if regime_label == "趋势市" else "#ea3943" if regime_label == "高波动混乱市" else "#f0ad4e"
        st.markdown(
            f"""
            <div style="border-radius:8px; border:1px solid {color}; padding:8px; background-color:#050505;">
                <span style="color:{color}; font-weight:bold;">当前 4h 市场状态：{regime_label}</span><br>
                <span style="color:#dddddd; font-size:12px;">{regime_comment}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    available_tfs = [tf for tf in TIMEFRAMES if tf in tf_signals.index]
    cols = st.columns(len(available_tfs))

    for col, tf in zip(cols, available_tfs):
        row = tf_signals.loc[tf]
        direction = row["direction"] if pd.notna(row["direction"]) else "观望"
        price = row["price"]

        if direction == "多":
            color = "#16c784"
            dir_text = "多头"
        elif direction == "空":
            color = "#ea3943"
            dir_text = "空头"
        else:
            color = "#999999"
            dir_text = "观望"

        sl = row["stop_loss"]
        tp = row["take_profit"]
        sl_str = f"{sl:.4f}" if pd.notna(sl) else "—"
        tp_str = f"{tp:.4f}" if pd.notna(tp) else "—"

        comment_lines = build_card_comment(tf, row, tf_signals, LONG_THRESHOLD, SHORT_THRESHOLD)
        explain_html = "<br>".join(comment_lines)

        with col:
            st.markdown(
                f"""
                <div style="border-radius:10px; border:1px solid {color};
                            padding:10px; background-color:#050505;">
                    <div style="color:{color}; font-weight:bold; font-size:16px; margin-bottom:4px;">
                        {tf} · {TF_DESC.get(tf, "")}
                    </div>
                    <div style="font-size:13px; color:white; margin-bottom:4px;">
                        方向：<b style="color:{color};">{dir_text}</b>
                        &nbsp;|&nbsp; 价格：{price:.4f}
                    </div>
                    <div style="font-size:12px; color:lightgray; margin-bottom:4px;">
                        止损：{sl_str} · 止盈：{tp_str}
                    </div>
                    <div style="font-size:11px; color:#cccccc;">
                        {explain_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================
# 多周期详细指标 & 风格剖面
# =========================

st.subheader("📊 多周期详细指标 & 统计风格剖面")

if tf_signals.empty:
    st.info("暂无多周期详细指标。")
else:
    table = tf_signals.copy()
    table["方向"] = table["direction"].fillna("观望")

    table_show = table[[
        "price", "trend_score", "reversal_score", "volatility_score",
        "composite_score", "rsi", "adx", "方向",
        "stop_loss", "take_profit",
        "period_return", "month_percentile",
        "score_percentile", "prob_up", "exp_ret"
    ]]

    ret_col = f"近{PERIOD_RET_LOOKBACK}根涨跌幅"
    month_col = "本月高低点百分位"
    score_pct_col = "当前评分分位"
    prob_up_col = "经验上涨概率"
    exp_ret_col = "经验期望收益"

    table_show = table_show.rename(columns={
        "period_return": ret_col,
        "month_percentile": month_col,
        "score_percentile": score_pct_col,
        "prob_up": prob_up_col,
        "exp_ret": exp_ret_col
    })

    fmt_dict = {
        "price": "{:.4f}",
        "trend_score": "{:.1f}",
        "reversal_score": "{:.1f}",
        "volatility_score": "{:.1f}",
        "composite_score": "{:.1f}",
        "rsi": "{:.1f}",
        "adx": "{:.1f}",
        "stop_loss": "{:.4f}",
        "take_profit": "{:.4f}",
        ret_col: "{:.2%}",
        month_col: "{:.1%}",
        score_pct_col: "{:.1%}",
        prob_up_col: "{:.1%}",
        exp_ret_col: "{:.2%}"
    }

    st.dataframe(
        table_show.style.format(fmt_dict, na_rep="—"),
        use_container_width=True
    )

    # 多因子风格剖面（加权雷达图）
    agg_trend = sum(
        tf_signals.loc[tf, "trend_score"] * w
        for tf, w in TF_WEIGHTS.items()
        if tf in tf_signals.index and pd.notna(tf_signals.loc[tf, "trend_score"])
    )
    agg_reversal = sum(
        tf_signals.loc[tf, "reversal_score"] * w
        for tf, w in TF_WEIGHTS.items()
        if tf in tf_signals.index and pd.notna(tf_signals.loc[tf, "reversal_score"])
    )
    agg_vol = sum(
        tf_signals.loc[tf, "volatility_score"] * w
        for tf, w in TF_WEIGHTS.items()
        if tf in tf_signals.index and pd.notna(tf_signals.loc[tf, "volatility_score"])
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

# =========================
# 中部：K线图
# =========================

st.markdown("---")
st.subheader(f"📊 {selected_pair} · {MAIN_TIMEFRAME} K 线 & 技术结构")

fig_k = go.Figure()

fig_k.add_trace(go.Candlestick(
    x=main_df.index,
    open=main_df["open"],
    high=main_df["high"],
    low=main_df["low"],
    close=main_df["close"],
    name=f"{MAIN_TIMEFRAME} K 线",
    increasing_line_color="green",
    decreasing_line_color="red",
    showlegend=True
))

if not fac_main.empty:
    fig_k.add_trace(go.Scatter(
        x=main_df.index,
        y=fac_main["ema_fast"],
        name="EMA 20",
        line=dict(color="deepskyblue", width=1.2)
    ))
    fig_k.add_trace(go.Scatter(
        x=main_df.index,
        y=fac_main["ema_slow"],
        name="EMA 50",
        line=dict(color="orange", width=1.2)
    ))

    if not fac_main["atr"].empty and pd.notna(fac_main["atr"].iloc[-1]):
        last_atr = fac_main["atr"].iloc[-1]
        upper_band = main_df["close"] + last_atr * 2
        lower_band = main_df["close"] - last_atr * 2

        fig_k.add_trace(go.Scatter(
            x=main_df.index,
            y=upper_band,
            name="ATR 上轨",
            line=dict(color="gray", dash="dot"),
            opacity=0.5
        ))
        fig_k.add_trace(go.Scatter(
            x=main_df.index,
            y=lower_band,
            name="ATR 下轨",
            line=dict(color="gray", dash="dot"),
            opacity=0.5
        ))

fig_k.update_layout(
    height=550,
    xaxis_title="时间",
    yaxis_title="价格 (USDT)",
    template="plotly_dark"
)

st.plotly_chart(fig_k, use_container_width=True)

# =========================
# 回测 & 盈亏分布（主周期 4h）
# =========================

st.markdown("---")
st.subheader(f"📈 机械执行回测：过去 {BACKTEST_DAYS} 天（主周期 {MAIN_TIMEFRAME}）")

cutoff = main_df.index[-1] - timedelta(days=BACKTEST_DAYS)
bt_df = main_df[main_df.index >= cutoff]

if len(bt_df) < MIN_BARS_FOR_FACTORS + 10:
    st.warning("主周期数据长度不足，无法进行有效回测。")
else:
    with st.spinner("正在运行历史回测引擎（纯模拟、不接实盘）……"):
        equity, trades = backtest_on_dataframe(
            bt_df,
            LONG_THRESHOLD,
            SHORT_THRESHOLD,
            ATR_SL_MULT,
            ATR_TP_MULT,
            INIT_CAPITAL,
            RISK_FRACTION,
            max_holding_bars=MAX_HOLDING_BARS
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
            st.metric("最大回撤（按累计PnL）", f"{stats['max_drawdown']:.2f} USDT")

        col5, col6 = st.columns(2)
        with col5:
            st.metric("单笔平均收益", f"{stats['avg_pnl']:.2f} USDT")
        with col6:
            st.metric("单笔平均收益率", f"{stats['avg_ret']:.2f}%")

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="模拟净值",
            line=dict(color="gold", width=2)
        ))
        fig_eq.add_hline(
            y=INIT_CAPITAL,
            line=dict(color="gray", dash="dash"),
            annotation_text="初始资金（虚拟）",
            annotation_position="bottom right"
        )
        fig_eq.update_layout(
            title="若过去区间全部机械执行，会长成怎样的净值曲线？",
            xaxis_title="时间",
            yaxis_title="账户权益 (USDT)",
            height=400,
            template="plotly_dark"
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        st.subheader(f"📊 最近 {N_HIST_TRADES} 笔信号的盈亏分布")
        trades_hist = trades.tail(N_HIST_TRADES)
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
        st.markdown("**🌍 全市场概览（CoinGecko）**")

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
    - 当 **多周期综合评分偏多** 且 情绪偏贪婪时：技术多头 + 情绪乐观，适合严格止盈、控制仓位。
    - 当 **多周期综合评分偏空** 且 情绪极度恐惧时：技术空头 + 情绪冰点，容易出现情绪底，适合分批布局而非重仓梭哈。
    - BTC 主导率上升且总市值回落时：资金偏防御，山寨币相对更危险。
    """)

# =========================
# 页脚
# =========================

st.markdown("---")
st.caption("""
本应用为量化分析与回测工具，不构成任何投资建议。  
模型基于历史数据与技术因子，无法保证未来表现。  
加密货币波动性极高，请谨慎决策，严格止损。
""")
