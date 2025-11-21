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
# 0. 全局配置：OKX（无代理）
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
# 1. 样式
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
        background: radial-gradient(circle at top left, #111829 0, #0b1120 55%);
        border-radius: 10px;
        border: 1px solid #1f2937;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.5);
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
        background: linear-gradient(135deg, rgba(15,23,42,0.95) 0, rgba(15,23,42,0.7) 55%);
        border:1px solid rgba(148,163,184,0.6); 
        font-size: 12px;
    }
    .plan-row {
        display:flex; justify-content:space-between; margin-bottom:2px; }
    .plan-label { color:#9ca3af; }
    .plan-value { font-family:'JetBrains Mono',monospace; font-weight:600; }

    .bull { color:#4ade80; }
    .bear { color:#fb7185; }

    .backtest-box {
        margin-top:8px;
        border-radius: 8px;
        padding:8px 10px;
        background:rgba(15,23,42,0.9);
        border:1px solid rgba(56,189,248,0.5);
        font-size:12px;
    }
    .summary-panel {
        margin-top:16px;
        padding:16px;
        border-radius: 10px;
        border:1px solid rgba(96,165,250,0.6);
        background: radial-gradient(circle at top left, rgba(37,99,235,0.25), rgba(15,23,42,0.96);
    }
    .summary-text {
        font-size: 19px;
        font-weight: 700;
        color:#e5f0ff;
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
    bt_trades: int = 0
    bt_winrate: Optional[float] = None
    bt_avg_rr: Optional[float] = None

# ============================================================
# 3. 数据引擎：OKX + 指标
# ============================================================

class OKXDataEngine:
    def __init__(self, config):
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

        df["RSI_14"] = ta.rsi(close, length=14)
        stoch_rsi = ta.stochrsi(close, length=14)
        if stoch_rsi is not None and not stoch_rsi.empty:
            df["STOCHRSI_K"] = stoch_rsi.iloc[:, 0]
            df["STOCHRSI_D"] = stoch_rsi.iloc[:, 1]
            df["STOCHRSI_HIST"] = stoch_rsi.iloc[:, 2]

        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df["MACD"] = macd.iloc[:, 0]
            df["MACD_SIGNAL"] = macd.iloc[:, 1]
            df["MACD_HIST"] = macd.iloc[:, 2]

        adx = ta.adx(high, low, close, length=14)
        if adx is not None and not adx.empty:
            df["ADX_14"] = adx.iloc[:, 0]
            df["+DI_14"] = adx.iloc[:, 1]
            df["+DI_14"] = adx.iloc[:, 2]

        atr = ta.atr(high, low, close, length=14)
        if atr is not None and not atr.empty:
            df["ATR_14"] = atr.iloc[:, 0]
            df["ATR_HIST"] = atr.iloc[:, 1]

        bb = ta.bbands(close, length=20, std=2)
        if bb is not None and not bb.empty:
            df["BB_LOWER"] = bb.iloc[:, 0]
            df["BB_MID"] = bb.iloc[:, 1]
            df["BB_UPPER"] = bb.iloc[:, 2]
            df["BB_WIDTH"] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]

        adx = ta.adx(high, low, close, length=14)
        if adx is not None and not adx.empty:
            df["ADX_14"] = adx.iloc[:, 0]
            df["+DI_14"] = adx.iloc[:, 1]
            df["+DI_14"] = adx.iloc[:, 2]

        mfi = ta.mfi(high, low, close, vol, length=14)
        if mfi is not None and not mfi.empty:
            df["MFI_14"] = mfi.iloc[:, 0]
            df["MFI_MA"] = mfi.iloc[:, 1]
            df["OBV"] = mfi.iloc[:, 2]
            df["OBV_MA"] = mfi.iloc[:, 3]

        return df.dropna().copy()

# ============================================================
# 4. 单周期分析 + 回测
# ============================================================

class SingleFrameAnalyst:
    def __init__(self, df: pd.DataFrame, tf: str):
        self.df = df
        self.tf = tf

    def analyze(self) -> SignalExplanation:
        d = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        price = d.get("close", None)
        ema20 = d.get("EMA_20", np.nan)
        ema50 = d.get("EMA_50", np.nan)
        ema100 = d.get("EMA_100", np.nan)
        rsi = d.get("RSI_14", np.nan)
        stoch_k = d.get("STOCHRSI_K", np.nan)
        stoch_d = d.get("STOCHRSI_D", np.nan)
        macd = d.get("MACD", np.nan)
        macd_sig = d.get("MACD_SIGNAL", np.nan)
        macd_hist = d.get("MACD_HIST", np.nan)
        atr = d.get("ATR_14", np.nan)
        bb_width = d.get("BB_WIDTH", np.nan)
        adx = d.get("ADX_14", np.nan)
        plus_di = d.get("+DI_14", np.nan)
        minus_di = d.get("-DI_14", np.nan)

        long_score = 0.0
        short_score = 0.0
        reasons: List[str] = []
        regime = "neutral"

        if price > ema20 > ema50 > ema100:
            reasons.append("price ≈ {} ≈ EMA 10 ≈ EMA 20 ≈ EMA 50 ≈ EMA 100".format(price))
            long_score += 1.0
        elif price < ema20 < ema50 < ema100:
            reasons.append("price ≈ {} ≈ EMA 10 ≈ EMA 20 ≈ EMA 50 ≈ EMA 100".format(price))
            short_score += 1.0
        else:
            reasons.append("price ≈ {} ≈ EMA 10 ≈ EMA 20 ≈ EMA 50 ≈ EMA 100".format(price))
            short_score += 1.0

        for reason in reasons:
            reasons.append(reason)

        return SignalExplanation(
            timeframe=self.tf,
            regime=regime,
            bias="neutral",
            conviction=conviction,
            long_score=long_score,
            short_score=short_score,
            reasons=reasons,
            entry_hint=None,
            stop_loss=None,
            take_profit_1=None,
            take_profit_2=None,
            reward_risk_1=None,
            reward_risk_2=None,
            bt_trades=0,
            bt_winrate=None,
            bt_avg_rr=None
        )

    def _simple_backtest(self):
        results = []
        for i in range(30, len(self.df) - 3):
            row = self.df.iloc[i]
            prev = self.df.iloc[i - 1]
            outcome = self.analyze(row)
            results.append(outcome)
        return len(results), sum(results) / len(results), sum(results) / len(results)

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

            direction = "bull" if net > 1 else "bear" if net < -1 else "neutral"
            fragments.append(
                f"{sig.timeframe}: {direction} (多 {sig.long_score:.1f} / 空 {sig.short_score:.1f} · 权重 {w:.1f})"
            )

        total = bull_power + bear_power
        bull_ratio = bull_power / total
        conviction = min(100.0, total * 7.0)

        if bull_ratio > 0.7 and bull_power > 6:
            stance = "STRONG_BULL"
            main = "从超短线到趋势，大部分时间尺度都支持多头，这是可以主动拥抱的趋势结构。"
        elif bull_ratio > 0.55 and bull_power > bear_power:
            stance = "BULL"
            main = "整体略偏多：更适合在回调中做多，而不是在高位盲目追多。"
        elif bull_ratio < 0.3 and bear_power > bull_power:
            stance = "STRONG_BEAR"
            main = "多周期共振偏空：反弹更像是减仓或做空的机会。"
        else:
            stance = "NEUTRAL"
            main = "各周期之间意见分裂，缺乏统一方向，仓位与杠杆都该收缩。"

        detail = " | ".join(fragments)
        return main + " 细分维度：" + detail, stance, conviction

# ============================================================
# 6. 渲染卡片（关键修复点）
# ============================================================

def render_signal_card(sig: Optional[SignalExplanation]):
    if sig is None:
        st.markdown("<div class='quant-card'>该周期数据不足，暂不输出观点。</div>", unsafe_allow_html=True)
        return

    if "多" in sig.bias:
        tag_class = "tag-bull"
    elif "空" in sig.bias:
        tag_class = "tag-bear"
    else:
        tag_class = "tag-neutral"

    header = f"""
    <div class='quant-card'>
      <div class='quant-header'>
        <div class='quant-title'>{sig.timeframe}</div>
        <div class='quant-tag {tag_class}'>{sig.bias} · 信心 {sig.conviction:.0f}/100</div>
      </div>
      <div style='font-size:13px;line-height:1.6%;color:#e5e7eb;'>{header}</div>
    </div>"""

    logic_html = "".join(
        f"<div class='logic-item'><div class='logic-bullet'>•</div><div>{r}</div></div>" for r in sig.reasons
    )

    if sig.stop_loss is not None and sig.take_profit_1 is not None:
        dir_word = "做多" if sig.long_score > sig.short_score else "做空"
        dir_class = "bull" if dir_word == "做多" else "bear"
        rr1 = f"{sig.reward_risk_1:.1f}R" if sig.reward_risk_1 else "—"
        rr2 = f"{sig.reward_risk_2:.1f}R" if sig.reward_risk_2 else "—"

        plan_html = f"""
        <div class='plan-box'>
          <div class='plan-row'>
            <span class='plan-label'>执行方向</span>
            <span class='{dir_class}'>{dir_word}</span>
          </div>
          <div class='plan-row'>
            <span class='plan-label'>战术入场</span>
            <span class='plan-value ${dir_class}'>${sig.entry_hint:,.4f}</span>
          </div>
          <div class='plan-row'>
            <span class='plan-label'>防守止损</span>
            <span class='plan-value bear'>${sig.stop_loss:,.4f}</span>
          </div>
          <div class='plan-row'>
            <span class='plan-label'>止盈一档</span>
            <span class='plan-value bull'>${sig.take_profit_1:,.4f} · {rr1}</span>
          </div>
          <div class='plan-row'>
            <span class='plan-label'>止盈二档</span>
            <span class='plan-value bull'>${sig.take_profit_2:,.4f} · {rr2}</span>
          </div>
        </div>
        """

    tail = "</div></div>"

    html = f"""
    <div class='quant-card'>
      <div class='quant-header'>
        <div class='quant-title'>{sig.timeframe}</div>
        <div class='quant-tag {tag_class}'>{sig.bias} · 信心 {sig.conviction:.0f}/100</div>
      </div>
      <div style='font-size:13px;line-height:1.55; color:#e5e7eb;'>{header}</div>
      <div style='margin-top:16px; padding:16px;border-radius:10px;border:1px solid #1f2937;'>{logic_html}</div>
      <div style='margin-top:8px;padding:8px 10px;border-radius: 8px;background:linear-gradient(135deg,#020617,#0b1120);border:1px solid #1f2937;'>{plan_html}</div>
      <div style='margin-top:8px;padding:8px 10px;border-radius: 8px;background:linear-gradient(circle at top left, #111829 0, #0b1120 55%);border:1px solid #1f2937;'>{tail}</div>
    </div>
    """,

    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# 7. 仓位
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
            default=tfs_all,
        )

        st.markdown("")
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
        <div style='padding:14px 16px;border-radius:10px;
                    background:linear-gradient(135deg,#020617,#0b1120);
                    border:1px solid #1f2937;
                    font-size:13px;line-height:1.6%;
                    color:#e5e7eb;
                    {'':width:13px;height:15px;line-height:1.55; color:#e5e7eb;}"""
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div style='font-size:13px;line-height:1.55; color:#e5e7eb;'>{'':text-align:center;font-size:13px;}"""
            unsafe_allow_html=True,
        )

    st.markdown("### 🧠 多周期量化评估")

    signals: Dict[str, Optional[SignalExplanation]] = {}
    data_cache: Dict[str, Optional[pd.DataFrame]] = {}

    prog = st.progress(0.0)
    for i, tf in enumerate(enabled_tfs):
        with st.spinner(f"拉取 {symbol} {tf} 数据并计算指标中..."):
            df = engine.exchange.fetch_ohlcv(symbol, tf, limit=600)
            data_cache[tf] = df
            if df is None or len(df) < 80:
                signals[tf] = None
            else:
                analyst = SingleFrameAnalyst(df, tf)
                signals[tf] = analyst.analyze()
        prog.progress((i + 1) / max(len(enabled_tfs), 1))
    prog.empty()

    c_short, c_long = st.columns([2, 3])
    with c_short:
        st.subheader("🎯 超短线 / 短线视角")
        for tf in ["1m", "5m", "15m"]:
            if tf in enabled_tfs:
                render_signal_card(signals.get(tf))
    with c_long:
        st.subheader("🌊 中线 / 波段 / 趋势视角")
        for tf in ["1h", "4h", "1d"]:
            if tf in enabled_tfs:
                render_signal_card(signals.get(tf))

    st.markdown("### 🏛 多周期综合")
    chief = MultiFrameChiefAnalyst(signals)
    summary, stance, global_conviction = chief.synthesize()

    st.markdown("### 🏛 首席分析师 · 统一结论")

    color_map = {
        "STRONG_BULL": "#4ade80",
        "BULL": "#22c55e",
        "NEUTRAL": "#e5e7eb",
        "BEAR": "#fb7185",
        "STRONG_BEAR": "#fb923c",
    }
    s_color = color_map.get(stance, "#e5e7eb")

    st.markdown(f"""
    <div class='summary-panel' style='border-color:{s_color}99;'>{'':background:radial-gradient(circle at top left, #020617 0, #0b1120 55%) ;border:1px solid #1f2937;'>{'':font-size:13px;line-height:1.55; color:#e5e7eb;'>{'':font-size:13px;line-height:1.55; color:#e5e7eb;'>{'':summary-title}{summary}</div>
    ''", unsafe_allow_html=True)

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
        size, max_loss = compute_position(equity_usdt, risk_pct, entry, stop, contract_mult=1.0)

        dir_word = "做多" if main_sig.long_score > main_sig.short_score else "做空"
        dir_class = "bull" if dir_word == "做多" else "bear"
        rr1 = f"{main_sig.reward_risk_1:.1f}R" if main_sig.reward_risk_1 else "—"
        rr2 = f"{main_sig.reward_risk_2:.1f}R" if main_sig.reward_risk_2 else "—"

        plan_html = f"""
        <div class='quant-card'>
          <div class='quant-header'>
            <div class='quant-title'>{main_sig.timeframe}</div>
            <div class='quant-tag {dir_class}'>{dir_word}</div>
          </div>
          <div style='font-size:13px;line-height:1.6%; color:#e5e7eb;'>{header}</div>
          <div style='margin-top:8px;padding:8px 10px;border-radius: 8px;background:linear-gradient(circle at top left, #111829 0, #0b1120 55%) ;border:1px solid #1f2937;'>{logic_html}</div>
          <div style='margin-top:8px;padding:8px 10px;border-radius: 8px;background:linear-gradient(circle at top left, #111829 0, #0b1120 55%) ;border:1px solid #1f2937;'>{plan_html}</div>
        </div>
        """,

        st.markdown(plan_html, unsafe_allow_html=True)

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
                y=dff["EMA_100"],
                line=dict(color="#9ca3af", width=1.0, dash="dot"),
                name="EMA 100",
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
                y=1,
                xanchor="right",
                x=1,
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
<div class='risk-note'>
当你开始用固定的风险、固定的价格控制去执行这些信号时，
你就已经从“赌徒”这边慢慢往“首席分析师”那边靠近了。
</div>""",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
