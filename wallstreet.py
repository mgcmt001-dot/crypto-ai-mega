import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================================
# 0. 全局配置：OKX（无代理，适配 share.streamlit.io）
# ============================================================

EXCHANGE_ID = "okx"

OKX_CONFIG = {
    "enableRateLimit": True,
    "timeout": 20000,
    "options": {
        "defaultType": "spot",   # 现货；如果想改永续，可以改为 "swap"
    },
}


# ============================================================
# 1. 页面与样式
# ============================================================

st.set_page_config(
    page_title="WallStreet Alpha Desk – OKX Edition",
    page_icon="🦅",
    layout="wide",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

    .stApp {
        background-color: #050712;
        color: #e5e7eb;
        font-family: 'Noto Sans SC', sans-serif;
    }
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    section[data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1f2937;
    }
    .quant-card {
        background: radial-gradient(circle at top left, #111827 0, #020617 55%);
        border-radius: 10px;
        border: 1px solid #1f2937;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.55);
    }
    .quant-header {
        display:flex;
        justify-content:space-between;
        align-items:baseline;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 6px;
        margin-bottom: 8px;
    }
    .quant-title {
        font-size: 15px;
        font-weight: 700;
        color:#fde68a;
    }
    .quant-tag {
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing:0.08em;
    }
    .tag-bull { background: rgba(34,197,94,0.14); color:#4ade80; border:1px solid rgba(34,197,94,0.7); }
    .tag-bear { background: rgba(248,113,113,0.14); color:#fb7185; border:1px solid rgba(248,113,113,0.7); }
    .tag-neutral { background: rgba(148,163,184,0.16); color:#e5e7eb; border:1px solid rgba(148,163,184,0.6); }

    .logic-list { font-size: 13px; line-height:1.55; color:#e5e7eb; }
    .logic-item { display:flex; margin-bottom:3px; }
    .logic-bullet { color:#facc15; margin-right:6px; }

    .plan-box {
        margin-top: 8px;
        border-radius: 8px;
        padding: 9px 11px;
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(15,23,42,0.75));
        border: 1px dashed rgba(148,163,184,0.7);
        font-size: 12px;
    }
    .plan-row {
        display:flex;
        justify-content:space-between;
        margin-bottom:2px;
    }
    .plan-label { color:#9ca3af; }
    .plan-value { font-family:'JetBrains Mono',monospace; font-weight:600; }

    .bull { color:#4ade80; }
    .bear { color:#fb7185; }

    .backtest-box {
        margin-top:8px;
        border-radius:8px;
        padding:8px 10px;
        background:rgba(15,23,42,0.9);
        border:1px solid rgba(56,189,248,0.5);
        font-size:12px;
    }

    .summary-panel {
        margin-top:16px;
        padding:16px;
        border-radius: 10px;
        border:1px solid rgba(96,165,250,0.65);
        background: radial-gradient(circle at top left, rgba(37,99,235,0.25), rgba(15,23,42,0.96));
    }
    .summary-title {
        font-size: 13px;
        text-transform: uppercase;
        color:#bfdbfe;
        letter-spacing:0.12em;
        margin-bottom: 4px;
    }
    .summary-text {
        font-size: 19px;
        font-weight: 700;
        color:#e5f0ff;
        margin-bottom: 6px;
    }
    .summary-sub {
        font-size: 12px;
        color:#9ca3af;
    }

    .risk-note {
        font-size: 12px;
        color:#9ca3af;
        border-left: 3px solid #4b5563;
        padding-left: 8px;
        margin-top: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 2. 数据结构
# ============================================================

TF_LABELS = {
    "1m": "超短线 / 剥头皮 (1m)",
    "5m": "超短线 / 高频 (5m)",
    "15m": "短线 / 日内驱动 (15m)",
    "1h": "中线 / 短波段 (1h)",
    "4h": "波段 (4h)",
    "1d": "趋势级别 (1d)",
}

@dataclass
class SignalExplanation:
    timeframe: str
    regime: str
    bias: str      # “偏多 / 偏空 / 观望”
    conviction: float  # 0–100
    long_score: float
    short_score: float
    reasons: List[str] = field(default_factory=list)

    entry_hint: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    reward_risk_1: Optional[float] = None
    reward_risk_2: Optional[float] = None

    # 简单“历史回测”统计
    bt_trades: int = 0
    bt_winrate: Optional[float] = None
    bt_avg_rr: Optional[float] = None


# ============================================================
# 3. 数据引擎：OKX + 指标
# ============================================================

class OKXDataEngine:
    def __init__(self, config: Dict):
        exchange_class = getattr(ccxt, EXCHANGE_ID)
        self.exchange = exchange_class(config)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 800) -> Optional[pd.DataFrame]:
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            st.error(f"从 OKX 获取 {symbol} {timeframe} 数据失败: {e}")
            return None

        if not raw:
            return None

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]

        # --- 趋势 ---
        df["EMA_10"] = ta.ema(close, length=10)
        df["EMA_20"] = ta.ema(close, length=20)
        df["EMA_50"] = ta.ema(close, length=50)
        df["EMA_100"] = ta.ema(close, length=100)
        df["EMA_200"] = ta.ema(close, length=200)

        # --- 动能 ---
        df["RSI_14"] = ta.rsi(close, length=14)
        stoch_rsi = ta.stochrsi(close, length=14)
        if stoch_rsi is not None and not stoch_rsi.empty:
            df["STOCHRSI_K"] = stoch_rsi.iloc[:, 0]
            df["STOCHRSI_D"] = stoch_rsi.iloc[:, 1]

        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df["MACD"] = macd.iloc[:, 0]
            df["MACD_SIGNAL"] = macd.iloc[:, 1]
            df["MACD_HIST"] = macd.iloc[:, 2]

        # --- 波动率 ---
        df["ATR_14"] = ta.atr(high, low, close, length=14)
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None and not bb.empty:
            df["BB_LOWER"] = bb.iloc[:, 0]
            df["BB_MID"] = bb.iloc[:, 1]
            df["BB_UPPER"] = bb.iloc[:, 2]
            df["BB_WIDTH"] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]

        # --- 趋势强度 ---
        adx = ta.adx(high, low, close, length=14)
        if adx is not None and not adx.empty:
            df["ADX_14"] = adx.iloc[:, 0]
            df["+DI_14"] = adx.iloc[:, 1]
            df["-DI_14"] = adx.iloc[:, 2]

        # --- Supertrend ---
        try:
            st_df = ta.supertrend(high, low, close, length=10, multiplier=3.0)
            if st_df is not None and not st_df.empty:
                df["SUPERT"] = st_df.iloc[:, 0]
                df["SUPERT_DIR"] = st_df.iloc[:, 1]
        except Exception:
            pass

        # --- 资金流 ---
        df["MFI_14"] = ta.mfi(high, low, close, vol, length=14)
        df["OBV"] = ta.obv(close, vol)
        df["OBV_MA"] = ta.ema(df["OBV"], length=20)
        df["VOL_MA_20"] = ta.sma(vol, length=20)

        return df.dropna().copy()


# ============================================================
# 4. 单周期分析 + 简单回测
# ============================================================

class SingleFrameAnalyst:
    def __init__(self, df: pd.DataFrame, tf: str):
        self.df = df
        self.tf = tf
        self.label = TF_LABELS.get(tf, tf)

    def analyze(self) -> SignalExplanation:
        d = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        price = d["close"]
        ema10, ema20, ema50, ema100, ema200 = (
            d["EMA_10"],
            d["EMA_20"],
            d["EMA_50"],
            d["EMA_100"],
            d["EMA_200"],
        )
        rsi = d.get("RSI_14", np.nan)
        st_k = d.get("STOCHRSI_K", np.nan)
        st_d = d.get("STOCHRSI_D", np.nan)
        macd = d.get("MACD", np.nan)
        macd_sig = d.get("MACD_SIGNAL", np.nan)
        macd_hist = d.get("MACD_HIST", np.nan)
        atr = d.get("ATR_14", np.nan)
        bb_width = d.get("BB_WIDTH", np.nan)
        adx = d.get("ADX_14", np.nan)
        plus_di = d.get("+DI_14", np.nan)
        minus_di = d.get("-DI_14", np.nan)
        supert_dir = d.get("SUPERT_DIR", np.nan)
        mfi = d.get("MFI_14", np.nan)
        vol = d["volume"]
        vol_ma = d.get("VOL_MA_20", np.nan)
        obv = d.get("OBV", np.nan)
        obv_ma = d.get("OBV_MA", np.nan)

        long_score = 0.0
        short_score = 0.0
        reasons: List[str] = []
        regime = "中性结构"

        # === 1. 趋势结构 ===
        if price > ema20 > ema50 > ema100:
            reasons.append("趋势结构：价格强势站在 EMA 梯队上方，多头主导。")
            long_score += 3.0
        elif price < ema20 < ema50 < ema100:
            reasons.append("趋势结构：价格长时间压在 EMA 梯队下方，空头主导。")
            short_score += 3.0
        else:
            reasons.append("趋势结构：均线纠缠，方向不纯，更偏向震荡。")

        if not math.isnan(adx):
            if adx >= 25:
                regime = "趋势主导"
                if plus_di > minus_di:
                    long_score += 1.5
                else:
                    short_score += 1.5
                reasons.append(f"ADX ≈ {adx:.1f}，说明市场确实在走趋势，此时顺势更占优势。")
            elif adx <= 15:
                regime = "震荡为主"
                reasons.append(f"ADX ≈ {adx:.1f}，动能不足，容易上下扫止损。")
            else:
                reasons.append(f"ADX ≈ {adx:.1f}，趋势处在酝酿阶段。")

        if not math.isnan(supert_dir):
            if supert_dir > 0:
                long_score += 1.0
                reasons.append("Supertrend 在价格下方，为多头提供“底托”。")
            elif supert_dir < 0:
                short_score += 1.0
                reasons.append("Supertrend 在价格上方，对多头形成“天花板”。")

        # === 2. 动能/反转 ===
        if not math.isnan(rsi):
            if rsi > 70:
                reasons.append(f"RSI ≈ {rsi:.1f}，已明显超买，追多性价比不高。")
                short_score += 1.0
            elif rsi < 30:
                reasons.append(f"RSI ≈ {rsi:.1f}，已明显超卖，存在情绪修复空间。")
                long_score += 1.0

        if not math.isnan(st_k) and not math.isnan(st_d):
            if st_k < 0.2 and st_d < 0.2 and st_k > st_d:
                reasons.append("StochRSI：低位金叉，短线多头反击信号。")
                long_score += 1.0
            elif st_k > 0.8 and st_d > 0.8 and st_k < st_d:
                reasons.append("StochRSI：高位死叉，短线多头乏力。")
                short_score += 1.0

        if not math.isnan(macd) and not math.isnan(macd_sig) and not math.isnan(macd_hist):
            if macd > macd_sig and macd_hist > prev.get("MACD_HIST", 0):
                reasons.append("MACD 多头动能柱放大，资金正在加速推动上涨。")
                long_score += 1.5
            elif macd < macd_sig and macd_hist < prev.get("MACD_HIST", 0):
                reasons.append("MACD 空头动能柱放大，反弹更像离场机而非起涨点。")
                short_score += 1.5

        # === 3. 波动率 ===
        if not math.isnan(bb_width):
            if bb_width < 0.03:
                reasons.append(f"布林带带宽 {bb_width*100:.1f}% 极度收缩，大行情前的“屏息期”。")
            elif bb_width > 0.08:
                reasons.append(f"布林带带宽 {bb_width*100:.1f}% 已较高，短线波动剧烈。")

        # === 4. 资金流 ===
        if not math.isnan(mfi):
            if mfi > 80:
                reasons.append(f"MFI ≈ {mfi:.1f}，资金高度拥挤在多头一侧，边际买盘可能放缓。")
                short_score += 0.5
            elif mfi < 20:
                reasons.append(f"MFI ≈ {mfi:.1f}，资金极度撤离后，更容易对利好产生放大量反应。")
                long_score += 0.5

        if not math.isnan(obv) and not math.isnan(obv_ma):
            if obv > obv_ma:
                reasons.append("OBV 高于均线，量价齐升，资金净流入明显。")
                long_score += 0.5
            elif obv < obv_ma:
                reasons.append("OBV 低于均线，价格上行缺乏资金配合。")
                short_score += 0.5

        # === 5. 综合方向 ===
        net_score = long_score - short_score
        conviction = min(100.0, abs(net_score) * 10.0)

        if net_score >= 2.0:
            bias = "偏多 / 顺势做多优先"
        elif net_score <= -2.0:
            bias = "偏空 / 反弹做空优先"
        else:
            bias = "震荡 / 观望为主"

        # === 6. 止盈止损 ===
        entry_hint = price
        stop_loss = None
        tp1 = None
        tp2 = None
        rr1 = None
        rr2 = None

        lookback = 30
        recent = self.df.iloc[-lookback:]
        recent_low = recent["low"].min()
        recent_high = recent["high"].max()

        if not math.isnan(atr) and atr > 0:
            if net_score >= 2.0:
                sl_1 = price - 1.5 * atr
                sl_2 = recent_low
                stop_loss = min(sl_1, sl_2)
                risk = max(price - stop_loss, 1e-8)
                tp1 = price + 2.0 * risk
                tp2 = price + 3.5 * risk
                rr1 = 2.0
                rr2 = 3.5
                reasons.append("多头止损压在结构低点与 1.5 ATR 更深处，让市场证明你真的错了才退出。")
            elif net_score <= -2.0:
                sl_1 = price + 1.5 * atr
                sl_2 = recent_high
                stop_loss = max(sl_1, sl_2)
                risk = max(stop_loss - price, 1e-8)
                tp1 = price - 2.0 * risk
                tp2 = price - 3.5 * risk
                rr1 = 2.0
                rr2 = 3.5
                reasons.append("空头止损顶在结构高点与 1.5 ATR 之上，只在真正反转时离场。")
        else:
            reasons.append("ATR 数据异常，本周期只建议做方向参考，不建议机械挂单。")

        # === 7. 简单因子回测 ===
        bt_trades, bt_winrate, bt_avg_rr = self._simple_backtest()

        return SignalExplanation(
            timeframe=self.label,
            regime=regime,
            bias=bias,
            conviction=conviction,
            long_score=long_score,
            short_score=short_score,
            reasons=reasons,
            entry_hint=entry_hint,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            reward_risk_1=rr1,
            reward_risk_2=rr2,
            bt_trades=bt_trades,
            bt_winrate=bt_winrate,
            bt_avg_rr=bt_avg_rr,
        )

    def _simple_backtest(self, lookback: int = 200) -> Tuple[int, Optional[float], Optional[float]]:
        df = self.df.tail(lookback).copy()
        if len(df) < 80:
            return 0, None, None

        results = []
        for i in range(30, len(df) - 3):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            price = row["close"]
            ema20 = row["EMA_20"]
            ema50 = row["EMA_50"]
            ema100 = row["EMA_100"]
            rsi = row.get("RSI_14", np.nan)
            adx = row.get("ADX_14", np.nan)
            plus_di = row.get("+DI_14", np.nan)
            minus_di = row.get("-DI_14", np.nan)
            macd = row.get("MACD", np.nan)
            macd_sig = row.get("MACD_SIGNAL", np.nan)
            macd_hist = row.get("MACD_HIST", np.nan)
            atr = row.get("ATR_14", np.nan)

            if math.isnan(atr) or atr <= 0:
                continue

            long_s = 0.0
            short_s = 0.0

            if price > ema20 > ema50 > ema100:
                long_s += 2.5
            elif price < ema20 < ema50 < ema100:
                short_s += 2.5

            if not math.isnan(adx) and adx >= 25:
                if plus_di > minus_di:
                    long_s += 1.0
                else:
                    short_s += 1.0

            if not math.isnan(rsi):
                if rsi < 30:
                    long_s += 1.0
                elif rsi > 70:
                    short_s += 1.0

            if not (math.isnan(macd) or math.isnan(macd_sig) or math.isnan(macd_hist)):
                if macd > macd_sig and macd_hist > prev.get("MACD_HIST", 0):
                    long_s += 1.0
                elif macd < macd_sig and macd_hist < prev.get("MACD_HIST", 0):
                    short_s += 1.0

            net = long_s - short_s
            if net >= 2.0:
                entry = price
                sl = entry - 1.5 * atr
                risk = entry - sl
                tp = entry + 2.0 * risk
                outcome_rr = self._simulate_trade(df.iloc[i+1:i+4], "long", entry, sl, tp)
                results.append(outcome_rr)
            elif net <= -2.0:
                entry = price
                sl = entry + 1.5 * atr
                risk = sl - entry
                tp = entry - 2.0 * risk
                outcome_rr = self._simulate_trade(df.iloc[i+1:i+4], "short", entry, sl, tp)
                results.append(outcome_rr)

        if not results:
            return 0, None, None

        wins = sum(1 for r in results if r > 0)
        winrate = wins / len(results)
        avg_rr = sum(results) / len(results)
        return len(results), winrate, avg_rr

    @staticmethod
    def _simulate_trade(subdf: pd.DataFrame, direction: str, entry: float, sl: float, tp: float) -> float:
        if direction == "long":
            risk = entry - sl
            for _, r in subdf.iterrows():
                if r["low"] <= sl:
                    return -1.0
                if r["high"] >= tp:
                    return 2.0
            final = subdf.iloc[-1]["close"]
            return (final - entry) / risk
        else:
            risk = sl - entry
            for _, r in subdf.iterrows():
                if r["high"] >= sl:
                    return -1.0
                if r["low"] <= tp:
                    return 2.0
            final = subdf.iloc[-1]["close"]
            return (entry - final) / risk


# ============================================================
# 5. 多周期综合
# ============================================================

class MultiFrameChiefAnalyst:
    def __init__(self, signals: Dict[str, SignalExplanation]):
        self.signals = signals

    def synthesize(self) -> Tuple[str, str, float]:
        weights = {
            "1m": 0.5,
            "5m": 0.8,
            "15m": 1.0,
            "1h": 1.5,
            "4h": 2.0,
            "1d": 2.5,
        }

        bull_power = 0.0
        bear_power = 0.0
        fragments = []

        for tf, sig in self.signals.items():
            if sig is None:
                continue
            w = weights.get(tf, 1.0)
            net = sig.long_score - sig.short_score
            if net > 0:
                bull_power += net * w
            elif net < 0:
                bear_power += -net * w

            direction = "偏多" if net > 1 else "偏空" if net < -1 else "震荡"
            fragments.append(
                f"{sig.timeframe}：{direction} (多 {sig.long_score:.1f} / 空 {sig.short_score:.1f} · 权重 {w:.1f})"
            )

        if bull_power == 0 and bear_power == 0:
            return "所有周期都在犹豫，市场暂时没有给出可交易级别的信号。", "NEUTRAL", 5.0

        total = bull_power + bear_power
        bull_ratio = bull_power / total
        conviction = min(100.0, total * 7.0)

        if bull_ratio > 0.7 and bull_power > 6:
            stance = "STRONG_BULL"
            main = "从超短线到趋势，大部分时间尺度都支持多头，这是可以主动拥抱的趋势结构。"
        elif bull_ratio > 0.55 and bull_power > bear_power:
            stance = "BULL"
            main = "整体略偏多：更适合在回调中做多，而不是在高位盲目追多。"
        elif bull_ratio < 0.3 and bear_power > 6:
            stance = "STRONG_BEAR"
            main = "多周期共振偏空：反弹更像是减仓或做空的机会。"
        elif bull_ratio < 0.45 and bear_power > bull_power:
            stance = "BEAR"
            main = "整体略偏空：多头每一次上攻都显得有气无力。"
        else:
            stance = "NEUTRAL"
            main = "各周期之间意见分裂，缺乏统一方向，仓位与杠杆都该收缩。"

        detail = " | ".join(fragments)
        return main + " 细分维度：" + detail, stance, conviction


# ============================================================
# 6. UI 渲染（已统一修正所有 div 结构）
# ============================================================

def render_signal_card(sig: Optional[SignalExplanation]):
    if sig is None:
        st.markdown("<div class='quant-card'>该周期数据不足，暂不输出观点。</div>", unsafe_allow_html=True)
        return

    # 标签颜色
    if "多" in sig.bias:
        tag_class = "tag-bull"
    elif "空" in sig.bias:
        tag_class = "tag-bear"
    else:
        tag_class = "tag-neutral"

    # 头部 + 开启 logic-list 容器
    header = f"""
    <div class="quant-card">
      <div class="quant-header">
        <div class="quant-title">{sig.timeframe}</div>
        <div class="quant-tag {tag_class}">{sig.bias} · 信心 {sig.conviction:.0f}/100</div>
      </div>
      <div class="logic-list">
    """

    # 逻辑点
    logic_html = "".join(
        f"<div class='logic-item'><div class='logic-bullet'>•</div><div>{r}</div></div>"
        for r in sig.reasons
    )

    # 止盈止损块
    if sig.stop_loss is not None and sig.take_profit_1 is not None:
        dir_word = "做多" if sig.long_score > sig.short_score else "做空"
        dir_class = "bull" if dir_word == "做多" else "bear"
        rr1 = f"{sig.reward_risk_1:.1f}R" if sig.reward_risk_1 else "—"
        rr2 = f"{sig.reward_risk_2:.1f}R" if sig.reward_risk_2 else "—"

        plan_html = f"""
        <div class="plan-box">
            <div class="plan-row">
                <span class="plan-label">执行方向</span>
                <span class="plan-value {dir_class}">{dir_word}</span>
            </div>
            <div class="plan-row">
                <span class="plan-label">战术入场</span>
                <span class="plan-value">${sig.entry_hint:,.4f}</span>
            </div>
            <div class="plan-row">
                <span class="plan-label">防守止损</span>
                <span class="plan-value bear">${sig.stop_loss:,.4f}</span>
            </div>
            <div class="plan-row">
                <span class="plan-label">止盈一档</span>
                <span class="plan-value bull">${sig.take_profit_1:,.4f} · {rr1}</span>
            </div>
            <div class="plan-row">
                <span class="plan-label">止盈二档</span>
                <span class="plan-value bull">${sig.take_profit_2:,.4f} · {rr2}</span>
            </div>
        </div>
        """
    else:
        plan_html = "<div class='plan-box'>本周期仅给出方向性参考，不建议机械挂单。</div>"

    # 回测块（关键：这是普通 HTML 字符串，不是 markdown 代码块）
    if sig.bt_trades > 0 and sig.bt_winrate is not None:
        win = sig.bt_winrate * 100
        rr = sig.bt_avg_rr
        bt_html = f"""
        <div class="backtest-box">
            历史回测（最近 {sig.bt_trades} 笔模拟信号）：<br/>
            · 胜率约：<b>{win:.1f}%</b> · 平均每笔期望：<b>{rr:.2f}R</b><br/>
            <span style="color:#9ca3af;">这不是预测未来，而是在告诉你：这套打分在过去<b>大致有统计优势</b>。</span>
        </div>
        """
    else:
        bt_html = ""

    # 只关闭 logic-list 和 quant-card 这两层
    tail = "</div></div>"

    # 整体一次性输出，开启 unsafe_allow_html，防止当成“代码”渲染
    st.markdown(header + logic_html + plan_html + bt_html + tail, unsafe_allow_html=True)
        return

    if "多" in sig.bias:
        tag_class = "tag-bull"
    elif "空" in sig.bias:
        tag_class = "tag-bear"
    else:
        tag_class = "tag-neutral"

    # header + logic-list 容器
    header = f"""
    <div class="quant-card">
      <div class="quant-header">
        <div class="quant-title">{sig.timeframe}</div>
        <div class="quant-tag {tag_class}">{sig.bias} · 信心 {sig.conviction:.0f}/100</div>
      </div>
      <div class="logic-list">
    """

    logic_html = "".join(
        f"<div class='logic-item'><div class='logic-bullet'>•</div><div>{r}</div></div>"
        for r in sig.reasons
    )

    # 止盈止损
    if sig.stop_loss is not None and sig.take_profit_1 is not None:
        dir_word = "做多" if sig.long_score > sig.short_score else "做空"
        dir_class = "bull" if dir_word == "做多" else "bear"
        rr1 = f"{sig.reward_risk_1:.1f}R" if sig.reward_risk_1 else "—"
        rr2 = f"{sig.reward_risk_2:.1f}R" if sig.reward_risk_2 else "—"

        plan_html = f"""
        <div class="plan-box">
            <div class="plan-row">
                <span class="plan-label">执行方向</span>
                <span class="plan-value {dir_class}">{dir_word}</span>
            </div>
            <div class="plan-row">
                <span class="plan-label">战术入场</span>
                <span class="plan-value">${sig.entry_hint:,.4f}</span>
            </div>
            <div class="plan-row">
                <span class="plan-label">防守止损</span>
                <span class="plan-value bear">${sig.stop_loss:,.4f}</span>
            </div>
            <div class="plan-row">
                <span class="plan-label">止盈一档</span>
                <span class="plan-value bull">${sig.take_profit_1:,.4f} · {rr1}</span>
            </div>
            <div class="plan-row">
                <span class="plan-label">止盈二档</span>
                <span class="plan-value bull">${sig.take_profit_2:,.4f} · {rr2}</span>
            </div>
        </div>
        """
    else:
        plan_html = "<div class='plan-box'>本周期仅给出方向性参考，不建议机械挂单。</div>"

    # 回测块：自身 div 自洽，不影响外部计数
    if sig.bt_trades > 0 and sig.bt_winrate is not None:
        win = sig.bt_winrate * 100
        rr = sig.bt_avg_rr
        bt_html = f"""
        <div class="backtest-box">
            历史回测（最近 {sig.bt_trades} 笔模拟信号）：<br/>
            · 胜率约：<b>{win:.1f}%</b> · 平均每笔期望：<b>{rr:.2f}R</b><br/>
            <span style="color:#9ca3af;">这不是预测未来，而是在告诉你：这套打分在过去<b>大致有统计优势</b>。</span>
        </div>
        """
    else:
        bt_html = ""

    # 这里只需要关掉 logic-list 和 quant-card 各一个 div
    tail = "</div></div>"

    st.markdown(header + logic_html + plan_html + bt_html + tail, unsafe_allow_html=True)


# ============================================================
# 7. 仓位建议
# ============================================================

def compute_position(
    equity_usdt: float,
    risk_pct: float,
    entry: float,
    stop: float,
    contract_mult: float = 1.0,
) -> Tuple[float, float]:
    if equity_usdt <= 0 or risk_pct <= 0 or entry <= 0 or stop <= 0 or entry == stop:
        return 0.0, 0.0
    max_loss = equity_usdt * (risk_pct / 100.0)
    per_unit_loss = abs(entry - stop) * contract_mult
    if per_unit_loss <= 0:
        return 0.0, 0.0
    size = max_loss / per_unit_loss
    return size, max_loss


# ============================================================
# 8. 主程序
# ============================================================

def main():
    st.title("🦅 WallStreet Alpha Desk – OKX Edition")
    st.caption("数据源：OKX 公共行情 · 无代理 · 仅供量化研究与教育，不构成投资建议。")

    with st.sidebar:
        st.subheader("📡 市场选择")

        COINS = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "OKB/USDT",
            "DOGE/USDT", "PEPE/USDT", "WIF/USDT", "SHIB/USDT",
            "SUI/USDT", "APT/USDT", "ORDI/USDT",
            "XRP/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT",
            "NEAR/USDT", "ARB/USDT", "OP/USDT",
        ]
        symbol = st.selectbox("选择标的 (OKX 现货)", COINS, index=0)

        tfs_all = ["1m", "5m", "15m", "1h", "4h", "1d"]
        enabled_tfs = st.multiselect(
            "启用的周期",
            options=tfs_all,
            default=["1m", "5m", "15m", "1h", "4h", "1d"],
        )

        st.markdown("---")
        st.subheader("💰 资金 & 风险参数")

        equity = st.number_input("账户总资金 (USDT)", min_value=100.0, value=10000.0, step=100.0)
        risk_pct = st.slider("单笔最大风险占比 (%)", 0.1, 5.0, 1.0, 0.1)

    engine = OKXDataEngine(OKX_CONFIG)
    try:
        ticker = engine.exchange.fetch_ticker(symbol)
    except Exception as e:
        st.error(f"无法连接 OKX，请检查网络或 IP 限制。\n{e}")
        return

    last = ticker.get("last", None)
    pct = ticker.get("percentage", None) or 0
    if last is None:
        st.error("Ticker 数据异常。")
        return

    col1, col2 = st.columns([2, 3])
    with col1:
        color = "#4ade80" if pct >= 0 else "#fb7185"
        st.markdown(
            f"""
        <div style="padding:14px 16px;border-radius:10px;
                    background:linear-gradient(135deg,#020617,#0b1120);
                    border:1px solid #1f2937;">
            <div style="font-size:13px;color:#9ca3af;">{symbol}</div>
            <div style="font-size:28px;font-weight:700;color:#e5e7eb;">${last:,.4f}</div>
            <div style="font-size:13px;color:{color};">24h 变动：{pct:.2f}%</div>
            <div style="font-size:11px;color:#6b7280;margin-top:4px;">
                北京时间：{(datetime.utcnow() + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div class="risk-note">
        · 我们关注的不是下一根K线的方向，而是：<b>现在这个方向，是否值得你冒一点可控的风险。</b><br/>
        · 多周期信号，会告诉你：短线在吵什么、趋势在偏向哪里、资金实际站在哪一边。<br/>
        · 真正的职业交易，本质是：<b>用严谨的风险控制，长期重复一个有统计优势的行为。</b>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # 多周期分析
    st.markdown("### 🧠 多周期量化评估")

    signals: Dict[str, Optional[SignalExplanation]] = {}
    data_cache: Dict[str, Optional[pd.DataFrame]] = {}

    prog = st.progress(0.0)
    for i, tf in enumerate(enabled_tfs):
        with st.spinner(f"拉取 {symbol} · {tf} 数据 & 计算指标中..."):
            df = engine.fetch_ohlcv(symbol, tf, limit=600)
            data_cache[tf] = df
            if df is None or len(df) < 80:
                signals[tf] = None
            else:
                analyst = SingleFrameAnalyst(df, tf)
                signals[tf] = analyst.analyze()
        prog.progress((i + 1) / max(len(enabled_tfs), 1))
    prog.empty()

    c_short, c_long = st.columns(2)
    with c_short:
        st.subheader("🎯 超短线 / 短线视角")
        for tf in ["1m", "5m", "15m"]:
            if tf in enabled_tfs:
                render_signal_card(signals.get(tf))
    with c_long:
        st.subheader("🌊 中线 / 波段 / 趋势")
        for tf in ["1h", "4h", "1d"]:
            if tf in enabled_tfs:
                render_signal_card(signals.get(tf))

    # 多周期统一裁决
    chief = MultiFrameChiefAnalyst(signals)
    summary, stance, global_conviction = chief.synthesize()

    st.markdown("### 🏛 首席分析师 · 统一结论")

    color_map = {
        "STRONG_BULL": "#4ade80",
        "BULL": "#22c55e",
        "NEUTRAL": "#e5e7eb",
        "BEAR": "#fb923c",
        "STRONG_BEAR": "#fb7185",
    }
    s_color = color_map.get(stance, "#e5e7eb")

    st.markdown(
        f"""
    <div class="summary-panel" style="border-color:{s_color}99;">
        <div class="summary-title">GLOBAL VIEW</div>
        <div class="summary-text" style="color:{s_color};">{summary}</div>
        <div class="summary-sub">
            立场：<b>{stance}</b> · 模型综合置信度：<b>{global_conviction:.0f}/100</b><br/>
            如果你今天只能在「做多 / 做空 / 空仓」里选一个——<br/>
            这是把所有周期交易员锁在会议室里吵完一整天之后，<b>他们勉强达成的共识。</b>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 仓位建议
    st.markdown("### 📦 仓位与执行建议")

    main_sig = None
    for key in ["1h", "4h", "15m", "1d"]:
        if key in enabled_tfs and signals.get(key) is not None:
            main_sig = signals[key]
            break

    if main_sig is None or main_sig.stop_loss is None:
        st.info("当前没有带有效止损的主操作周期信号，仅建议观望或轻仓试探。")
    else:
        entry = main_sig.entry_hint
        stop = main_sig.stop_loss
        size, max_loss = compute_position(equity, risk_pct, entry, stop, contract_mult=1.0)

        dir_word = "做多" if main_sig.long_score > main_sig.short_score else "做空"
        dir_color = "#4ade80" if dir_word == "做多" else "#fb7185"

        st.markdown(
            f"""
        <div class="quant-card">
            <div class="quant-header">
                <div class="quant-title">基于「{main_sig.timeframe}」信号的执行模板</div>
                <div class="quant-tag" style="border-color:{dir_color};color:{dir_color};">{dir_word}</div>
            </div>
            <div style="font-size:13px;line-height:1.6;">
                · 当前统计意义上性价比最高的一侧是：<b style="color:{dir_color};">{dir_word}</b><br/>
                · 入场参考：<b>${entry:,.4f}</b> · 止损保护：<b>${stop:,.4f}</b><br/>
                · 以你账户 <b>{equity:,.0f} USDT</b>，单笔愿意承担 <b>{risk_pct:.1f}%</b> 风险：<br/>
                &nbsp;&nbsp;⇒ 理论最大亏损 ≈ <b>{max_loss:,.2f} USDT</b><br/>
                &nbsp;&nbsp;⇒ 在当前止损距离下，<b>建议仓位 ≈ {size:,.4f} 币</b>（1x 杠杆等效）。<br/><br/>
                这套仓位，不是让你去梭哈方向，而是：<br/>
                · 把亏损<b>锁在你心理能接受的区间</b>；<br/>
                · 同时保留足够体量，让<b>正确的那几次信号，足以改变你的净值曲线。</b>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # 图表
    st.markdown("### 📈 价格行为与关键均线")

    chart_tf = "1h" if "1h" in enabled_tfs else (enabled_tfs[-1] if enabled_tfs else "1h")
    df_chart = data_cache.get(chart_tf)
    if df_chart is not None:
        dff = df_chart.tail(200)
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=dff.index,
                open=dff["open"],
                high=dff["high"],
                low=dff["low"],
                close=dff["close"],
                increasing_line_color="#4ade80",
                decreasing_line_color="#fb7185",
                name="Price",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dff.index,
                y=dff["EMA_20"],
                line=dict(color="#60a5fa", width=1.3),
                name="EMA 20",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dff.index,
                y=dff["EMA_50"],
                line=dict(color="#fbbf24", width=1.1),
                name="EMA 50",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dff.index,
                y=dff["EMA_200"],
                line=dict(color="#9ca3af", width=1.0, dash="dot"),
                name="EMA 200",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=10, r=10, t=30, b=20),
            paper_bgcolor="rgba(5,7,17,1)",
            plot_bgcolor="rgba(5,7,17,1)",
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
<div class="risk-note">
这个终端的意义，不是替你做决定，而是把<b>专业交易员的思考路径</b>摆在你面前：<br/>
趋势、动能、波动率、资金流、多周期、风险预算……<br/>
当你开始用这些东西来约束自己，而不是用情绪来驱动仓位时，<br/>
你就已经在向“首席分析师”的那一侧靠近了。
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

