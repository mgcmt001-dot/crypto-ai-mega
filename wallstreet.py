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
        # 你也可以改成 "swap" 用永续合约
        "defaultType": "spot",
    },
}


# ============================================================
# 1. 数据结构
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
    bias: str              # “偏多 / 偏空 / 震荡 / 观望”
    conviction: float      # 0–100
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
# 2. 数据引擎：OKX + 指标
# ============================================================

class OKXDataEngine:
    def __init__(self, config: Dict):
        exchange_class = getattr(ccxt, EXCHANGE_ID)
        self.exchange = exchange_class(config)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 800) -> Optional[pd.DataFrame]:
        """
        从 OKX 拉取 K 线数据，并计算一整套技术指标。
        """
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

        # --- 均线体系 ---
        df["EMA_10"] = ta.ema(close, length=10)
        df["EMA_20"] = ta.ema(close, length=20)
        df["EMA_50"] = ta.ema(close, length=50)
        df["EMA_100"] = ta.ema(close, length=100)
        df["EMA_200"] = ta.ema(close, length=200)
        df["SMA_20"] = ta.sma(close, length=20)
        df["SMA_50"] = ta.sma(close, length=50)

        # --- 振荡 & 动能 ---
        df["RSI_14"] = ta.rsi(close, length=14)

        stoch = ta.stoch(high, low, close, k=14, d=3)
        if stoch is not None and not stoch.empty:
            df["STOCH_K"] = stoch.iloc[:, 0]
            df["STOCH_D"] = stoch.iloc[:, 1]

        stoch_rsi = ta.stochrsi(close, length=14)
        if stoch_rsi is not None and not stoch_rsi.empty:
            df["STOCHRSI_K"] = stoch_rsi.iloc[:, 0]
            df["STOCHRSI_D"] = stoch_rsi.iloc[:, 1]

        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df["MACD"] = macd.iloc[:, 0]
            df["MACD_SIGNAL"] = macd.iloc[:, 1]
            df["MACD_HIST"] = macd.iloc[:, 2]

        # --- 波动率 & 布林 ---
        df["ATR_14"] = ta.atr(high, low, close, length=14)
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None and not bb.empty:
            df["BB_LOWER"] = bb.iloc[:, 0]
            df["BB_MID"] = bb.iloc[:, 1]
            df["BB_UPPER"] = bb.iloc[:, 2]
            df["BB_WIDTH"] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]

        # --- 趋势强度 ADX/DI ---
        adx = ta.adx(high, low, close, length=14)
        if adx is not None and not adx.empty:
            df["ADX_14"] = adx.iloc[:, 0]
            df["+DI_14"] = adx.iloc[:, 1]
            df["-DI_14"] = adx.iloc[:, 2]

        # --- Supertrend（pandas_ta 内置） ---
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
# 3. 单周期分析 + 简单回测
# ============================================================

class SingleFrameAnalyst:
    """
    单一周期的“量化交易员”：
    - 读取一整个 DataFrame
    - 用一堆指标给当前 K 线多空打分
    - 给出方向 + 止盈止损建议
    - 做一个非常简化的 histórico 信号回测（胜率 + 平均R）
    """

    def __init__(self, df: pd.DataFrame, tf: str):
        self.df = df
        self.tf = tf
        self.label = TF_LABELS.get(tf, tf)

    def analyze(self) -> SignalExplanation:
        d = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        price = d["close"]
        ema10, ema20, ema50, ema100, ema200 = (
            d["EMA_10"], d["EMA_20"], d["EMA_50"], d["EMA_100"], d["EMA_200"]
        )
        sma20 = d.get("SMA_20", np.nan)
        sma50 = d.get("SMA_50", np.nan)

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

        # ========== 1. 趋势结构：价格 vs EMA / SMA 梯队 ==========
        if price > ema20 > ema50 > ema100:
            reasons.append("趋势结构：价格强势踏在 EMA20/50/100 之上，多头主导，中短期抬升节奏良好。")
            long_score += 3.0
        elif price < ema20 < ema50 < ema100:
            reasons.append("趋势结构：价格长期压在 EMA20/50/100 之下，空头主导，反弹大多是逃命波。")
            short_score += 3.0
        else:
            reasons.append("趋势结构：均线相互纠缠，趋势不纯，更像是多空拉锯的中性区间。")

        # 大级别慢均线位置：价格相对 EMA200 / SMA50 的大势判断
        if price > ema200:
            reasons.append("长期均线：价格整体运行在 EMA200 之上，长期结构偏多。")
            long_score += 1.0
        elif price < ema200:
            reasons.append("长期均线：价格整体运行在 EMA200 之下，长期结构偏空。")
            short_score += 1.0

        if price > sma50 > sma20:
            reasons.append("中期均线：SMA20 在 SMA50 下方，短线节奏略显急促，多头仍占优但存在回踩需求。")
        elif price < sma50 < sma20:
            reasons.append("中期均线：SMA20 在 SMA50 上方但价格已跌破，说明短期多头有被反杀的风险。")

        # ========== 2. ADX / DI：趋势强度 + 谁在主导 ==========
        if not math.isnan(adx):
            if adx >= 25:
                regime = "趋势主导"
                if plus_di > minus_di:
                    long_score += 1.5
                    reasons.append(f"ADX ≈ {adx:.1f}，趋势强度已成型，且 +DI > -DI，多头趋势占上风。")
                else:
                    short_score += 1.5
                    reasons.append(f"ADX ≈ {adx:.1f}，趋势强度已成型，且 -DI > +DI，空头趋势占上风。")
            elif adx <= 15:
                regime = "震荡为主"
                reasons.append(f"ADX ≈ {adx:.1f}，动能偏弱，目前更像是区间博弈而非趋势单边。")
            else:
                reasons.append(f"ADX ≈ {adx:.1f}，趋势刚起步或处于过渡阶段，还没完全站队。")

        # ========== 3. Supertrend 作为“趋势护盾” ==========
        if not math.isnan(supert_dir):
            if supert_dir > 0:
                long_score += 1.0
                reasons.append("Supertrend 当前在价格下方，相当于给多头提供了一个动态抬升的防守位。")
            elif supert_dir < 0:
                short_score += 1.0
                reasons.append("Supertrend 当前压在价格上方，对多头形成天花板，反弹容易被压制。")

        # ========== 4. RSI / StochRSI：情绪极端 & 拐点线索 ==========
        if not math.isnan(rsi):
            if rsi > 70:
                reasons.append(f"RSI ≈ {rsi:.1f}，情绪已经偏热，继续追高需要非常坚实的资金接力。")
                short_score += 1.0
            elif rsi < 30:
                reasons.append(f"RSI ≈ {rsi:.1f}，情绪极度悲观，往往离情绪修复不远。")
                long_score += 1.0

        if not math.isnan(st_k) and not math.isnan(st_d):
            if st_k < 0.2 and st_d < 0.2 and st_k > st_d:
                reasons.append("StochRSI：在深度超卖区出现金叉，短线多头有“反扑权”。")
                long_score += 1.0
            elif st_k > 0.8 and st_d > 0.8 and st_k < st_d:
                reasons.append("StochRSI：在高位死叉，资金在高位开始兑现，短线向上空间有限。")
                short_score += 1.0

        # ========== 5. MACD：中期动能的增减 ==========
        if not math.isnan(macd) and not math.isnan(macd_sig) and not math.isnan(macd_hist):
            prev_hist = prev.get("MACD_HIST", 0.0)
            if macd > macd_sig and macd_hist > prev_hist:
                reasons.append("MACD：多头柱放大且线在信号线上方，中期上涨动能在积累。")
                long_score += 1.5
            elif macd < macd_sig and macd_hist < prev_hist:
                reasons.append("MACD：空头柱放大且线在信号线下方，中期下跌动能在积累。")
                short_score += 1.5

        # ========== 6. 波动率 & 布林带状态 ==========
        if not math.isnan(bb_width):
            if bb_width < 0.03:
                reasons.append(f"布林带带宽仅 {bb_width*100:.1f}%：波动极度收缩，大行情往往从这种“闷局”后突然爆发。")
            elif bb_width > 0.08:
                reasons.append(f"布林带带宽约 {bb_width*100:.1f}%：波动已经被彻底点燃，追单容易被剧烈回撤洗出去。")

        # ========== 7. 资金流：MFI / OBV / 成交量 ==========
        if not math.isnan(mfi):
            if mfi > 80:
                reasons.append(f"MFI ≈ {mfi:.1f}，买盘极度拥挤，任何利空都可能触发多头集体减仓。")
                short_score += 0.5
            elif mfi < 20:
                reasons.append(f"MFI ≈ {mfi:.1f}，资金极度悲观，稍有利好就可能点燃一轮报复性反弹。")
                long_score += 0.5

        if not math.isnan(obv) and not math.isnan(obv_ma):
            if obv > obv_ma:
                reasons.append("OBV 在其均线上方，量价同向上行，说明有“真金白银”在推这波行情。")
                long_score += 0.5
            elif obv < obv_ma:
                reasons.append("OBV 在其均线下方，价格的每一次拉升都更像是“无量空拉”。")
                short_score += 0.5

        if not math.isnan(vol_ma):
            if vol > 1.5 * vol_ma:
                reasons.append("当前成交量明显高于近 20 根均量，这个价位附近多空正在认真表态。")
            elif vol < 0.6 * vol_ma:
                reasons.append("成交量显著低于均值，这一波波动更像是“假动作”和“试探”。")

        # ========== 8. 多空方向综合 ==========
        net_score = long_score - short_score
        conviction = min(100.0, abs(net_score) * 9.0)  # 放大一点差值

        if net_score >= 2.5:
            bias = "偏多 / 顺势做多优先"
        elif net_score <= -2.5:
            bias = "偏空 / 反弹做空优先"
        elif -1.5 < net_score < 1.5:
            bias = "震荡 / 观望为主"
        else:
            bias = "轻微倾向，但不足以重仓下注"

        # ========== 9. 根据 ATR + 近期结构给出止盈止损 ==========
        entry_hint = price
        stop_loss = None
        tp1 = None
        tp2 = None
        rr1 = None
        rr2 = None

        lookback = 40
        recent = self.df.iloc[-lookback:]
        recent_low = recent["low"].min()
        recent_high = recent["high"].max()

        if not math.isnan(atr) and atr > 0:
            if net_score >= 2.5:
                # 做多：止损压在结构低点 / 1.5 ATR 之下
                sl_1 = price - 1.5 * atr
                sl_2 = recent_low
                stop_loss = min(sl_1, sl_2)
                risk = max(price - stop_loss, 1e-8)
                tp1 = price + 2.0 * risk
                tp2 = price + 3.5 * risk
                rr1 = 2.0
                rr2 = 3.5
                reasons.append("止损放在 1.5 ATR 与近期结构低点更深处，只在真正确认错了才认输。")
            elif net_score <= -2.5:
                # 做空：止损顶在结构高点 / 1.5 ATR 之上
                sl_1 = price + 1.5 * atr
                sl_2 = recent_high
                stop_loss = max(sl_1, sl_2)
                risk = max(stop_loss - price, 1e-8)
                tp1 = price - 2.0 * risk
                tp2 = price - 3.5 * risk
                rr1 = 2.0
                rr2 = 3.5
                reasons.append("止损顶在 1.5 ATR 与近期结构高点更高处，只在行情真空头反转时离场。")
            else:
                reasons.append("当前周期多空打分不够极端，本周期仅做参考，不给机械挂单的止损止盈。")
        else:
            reasons.append("ATR 数据异常，本周期仅给出方向性结论，不做具体点位管理。")

        # ========== 10. 简单历史回测：这套打分是否“有点用”？ ==========
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

    # ---------- 简化版回测 ----------
    def _simple_backtest(self, lookback: int = 220) -> Tuple[int, Optional[float], Optional[float]]:
        """
        思路：
        - 回到历史数据里，每一根 K 线都重新算一次 long_score / short_score
        - 当 net_score >= 2.5 或 <= -2.5 时，视为一个信号
        - 用接下来的 3 根 K 线，模拟简单 RR 结果
        - 统计最近 N 笔信号的胜率与平均 R
        """
        df = self.df.tail(lookback).copy()
        if len(df) < 120:
            return 0, None, None

        results: List[float] = []

        for i in range(40, len(df) - 3):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            price = row["close"]
            ema20 = row["EMA_20"]
            ema50 = row["EMA_50"]
            ema100 = row["EMA_100"]
            ema200 = row["EMA_200"]
            rsi = row.get("RSI_14", np.nan)
            adx = row.get("ADX_14", np.nan)
            plus_di = row.get("+DI_14", np.nan)
            minus_di = row.get("-DI_14", np.nan)
            macd = row.get("MACD", np.nan)
            macd_sig = row.get("MACD_SIGNAL", np.nan)
            macd_hist = row.get("MACD_HIST", np.nan)
            atr = row.get("ATR_14", np.nan)
            supert_dir = row.get("SUPERT_DIR", np.nan)

            if math.isnan(atr) or atr <= 0:
                continue

            long_s = 0.0
            short_s = 0.0

            # 同一套打分逻辑（简化版）
            if price > ema20 > ema50 > ema100:
                long_s += 2.0
            elif price < ema20 < ema50 < ema100:
                short_s += 2.0

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
                prev_hist = prev.get("MACD_HIST", 0.0)
                if macd > macd_sig and macd_hist > prev_hist:
                    long_s += 1.0
                elif macd < macd_sig and macd_hist < prev_hist:
                    short_s += 1.0

            if not math.isnan(supert_dir):
                if supert_dir > 0:
                    long_s += 0.5
                elif supert_dir < 0:
                    short_s += 0.5

            net = long_s - short_s

            # 多头信号
            if net >= 2.5:
                entry = price
                sl = entry - 1.5 * atr
                risk = entry - sl
                tp = entry + 2.0 * risk
                outcome_rr = self._simulate_trade(df.iloc[i+1:i+4], "long", entry, sl, tp)
                results.append(outcome_rr)
            # 空头信号
            elif net <= -2.5:
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
        """
        非严格回测，只是用“最多走 3 根 K 线”的窗口，看：
        - 先触发止盈？则记 +2R
        - 先触发止损？则记 -1R
        - 都没触发？按最后收盘价换算成 R
        """
        if len(subdf) == 0:
            return 0.0

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
# 5. 多周期综合：像把一桌交易员关在会议室吵完
# ============================================================

class MultiFrameChiefAnalyst:
    def __init__(self, signals: Dict[str, Optional[SignalExplanation]]):
        self.signals = signals

    def synthesize(self) -> Tuple[str, str, float]:
        """
        把各个周期的多空打分加权合并，给出：
        - 一句话总结
        - 立场（BULL / BEAR / STRONG_BULL / STRONG_BEAR / NEUTRAL）
        - 综合置信度
        """
        weights = {"1m": 0.5, "5m": 0.7, "15m": 1.0, "1h": 1.8, "4h": 2.3, "1d": 2.8}
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

            if net > 1.5:
                direction = "偏多"
            elif net < -1.5:
                direction = "偏空"
            else:
                direction = "震荡"
            fragments.append(f"{sig.timeframe}：{direction}")

        if bull_power == 0 and bear_power == 0:
            return "所有周期都在犹豫，市场暂时没有给出可交易级别的清晰信号。", "NEUTRAL", 5.0

        total = bull_power + bear_power
        bull_ratio = bull_power / total
        conviction = min(100.0, total * 6.5)

        if bull_ratio > 0.72 and bull_power > 7:
            stance = "STRONG_BULL"
            main = "多头从超短线到趋势几乎全面占优，这是可以主动拥抱的多头环境。"
        elif bull_ratio > 0.55 and bull_power > bear_power:
            stance = "BULL"
            main = "整体略偏多，更适合在回调中接多，而不是在局部极端价位去追高。"
        elif bull_ratio < 0.28 and bear_power > 7:
            stance = "STRONG_BEAR"
            main = "多周期共振偏空，反弹更像是空头加仓或多头减仓的机会。"
        elif bull_ratio < 0.45 and bear_power > bull_power:
            stance = "BEAR"
            main = "整体略偏空，做空比做多更有胜率，但需要尊重反弹的杀伤力。"
        else:
            stance = "NEUTRAL"
            main = "各周期信号分裂，没有统一方向，这种时候仓位和杠杆都应该收缩。"

        detail = " | ".join(fragments)
        return main + " 细分维度：" + detail, stance, conviction


# ============================================================
# 6. 仓位管理：根据风险预算倒推币数
# ============================================================

def compute_position(
    equity_usdt: float,
    risk_pct: float,
    entry: float,
    stop: float,
    contract_mult: float = 1.0,
) -> Tuple[float, float]:
    """
    根据：
    - 账户总资金 equity_usdt
    - 单笔愿意亏损的百分比 risk_pct
    - 入场价 entry
    - 止损价 stop
    计算：
    - 建议持仓数量（币数或合约张数）
    - 对应的最大亏损金额
    """
    if equity_usdt <= 0 or risk_pct <= 0 or entry <= 0 or stop <= 0 or entry == stop:
        return 0.0, 0.0
    max_loss = equity_usdt * (risk_pct / 100.0)
    per_unit_loss = abs(entry - stop) * contract_mult
    if per_unit_loss <= 0:
        return 0.0, 0.0
    size = max_loss / per_unit_loss
    return size, max_loss


# ============================================================
# 7. UI 渲染（全部用 Markdown，杜绝 HTML 变代码）
# ============================================================

def render_signal_block(sig: Optional[SignalExplanation]):
    if sig is None:
        st.info("该周期数据不足，暂不输出观点。")
        return

    st.markdown(f"#### {sig.timeframe}")

    # 标题行：方向 + 置信度 + 市场状态
    st.markdown(
        f"- **方向**：{sig.bias}  \n"
        f"- **模型置信度**：`{sig.conviction:.0f} / 100`  \n"
        f"- **当前结构**：{sig.regime}"
    )

    st.markdown("**这套模型是怎么想的？（核心理由）**")
    for r in sig.reasons:
        st.markdown(f"- {r}")

    # 止盈止损
    if sig.stop_loss is not None and sig.take_profit_1 is not None:
        st.markdown("**执行参数建议（仅作研究示例，不构成投资建议）**")
        dir_word = "做多" if sig.long_score > sig.short_score else "做空"
        rr1 = f"{sig.reward_risk_1:.1f}R" if sig.reward_risk_1 else "—"
        rr2 = f"{sig.reward_risk_2:.1f}R" if sig.reward_risk_2 else "—"

        st.markdown(
            f"- 执行方向：**{dir_word}**  \n"
            f"- 参考入场：`{sig.entry_hint:,.4f}`  \n"
            f"- 防守止损：`{sig.stop_loss:,.4f}`  \n"
            f"- 止盈一档：`{sig.take_profit_1:,.4f}`（约 {rr1}）  \n"
            f"- 止盈二档：`{sig.take_profit_2:,.4f}`（约 {rr2}）"
        )
    else:
        st.markdown(
            "> 当前周期打分虽有倾向，但不足以支撑完整挂单计划：仅作方向性参考，"
            "不建议机械地设置止盈止损。"
        )

    # 回测表现
    if sig.bt_trades > 0 and sig.bt_winrate is not None:
        st.markdown("**历史简单回测（因子打分在本周期的表现）**")
        st.markdown(
            f"- 统计样本：最近 **{sig.bt_trades}** 笔模拟信号  \n"
            f"- 单笔胜率约：**{sig.bt_winrate * 100:.1f}%**  \n"
            f"- 单笔平均期望：**{sig.bt_avg_rr:.2f}R**  \n"
            f"> 这并不是对未来的承诺，而是在告诉你：\n"
            f"> 在过去的数据里，这种多空打分**大致是有一点统计优势的**。"
        )
    else:
        st.markdown("> 历史样本不足，本周期不展示回测统计。")

    st.markdown("---")


# ============================================================
# 8. 主程序：整合一切
# ============================================================

def main():
    st.title("🦅 WallStreet Alpha Desk – OKX 多周期量化终端")
    st.caption("数据源：OKX 公共行情 · 无代理直连 · 仅供量化研究与教育，不构成任何投资建议。")

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.subheader("📡 市场与周期")

        COINS = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "OKB/USDT",
            "DOGE/USDT", "WIF/USDT", "PEPE/USDT", "SHIB/USDT",
            "SUI/USDT", "APT/USDT", "ORDI/USDT",
            "XRP/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT",
            "NEAR/USDT", "ARB/USDT", "OP/USDT",
        ]
        symbol = st.selectbox("选择标的 (OKX 现货)", COINS, index=0)

        all_tfs = ["1m", "5m", "15m", "1h", "4h", "1d"]
        enabled_tfs = st.multiselect(
            "启用的周期（建议全选）",
            options=all_tfs,
            default=all_tfs,
        )

        st.subheader("💰 资金 & 风险参数")
        equity = st.number_input("账户总资金 (USDT)", min_value=100.0, value=10000.0, step=100.0)
        risk_pct = st.slider("单笔最大风险占比 (%)", 0.1, 5.0, 1.0, 0.1)

        st.markdown(
            "> 职业交易员不会问“这次能赚多少”，\n"
            "> 而是先问：“**如果错了，我愿意为这个观点付出多少学费？**”"
        )

    # ---------- Ticker 信息 ----------
    engine = OKXDataEngine(OKX_CONFIG)
    try:
        ticker = engine.exchange.fetch_ticker(symbol)
    except Exception as e:
        st.error(f"无法连接 OKX，请检查网络或 IP 限制。\n{e}")
        return

    last = ticker.get("last", None)
    pct = ticker.get("percentage", None) or 0.0
    if last is None:
        st.error("Ticker 数据异常。")
        return

    col_price, col_note = st.columns([2, 3])
    with col_price:
        st.markdown(f"### 当前行情：{symbol}")
        st.markdown(
            f"- 最新价：**{last:,.4f}** USDT  \n"
            f"- 24h 变动：**{pct:+.2f}%**  \n"
            f"- 北京时间：`{(datetime.utcnow() + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')}`"
        )

    with col_note:
        st.markdown("### 模型立场说明")
        st.markdown(
            "- 这不是一个“预测下一根 K 线”的玩具，而是一套**把主观观点量化**的框架。  \n"
            "- 它会同时看多周期、多因子，给出：\n"
            "  - 哪一边更值得你付出风险预算（多 / 空 / 观望）；\n"
            "  - 如果你愿意下注，止损应该放在哪、止盈应该往哪看；\n"
            "  - 回头复盘时，这种打法在过去究竟是赚是亏。"
        )

    # ---------- 多周期数据 & 分析 ----------
    st.markdown("## 🧠 多周期量化评估")

    signals: Dict[str, Optional[SignalExplanation]] = {}
    data_cache: Dict[str, Optional[pd.DataFrame]] = {}

    if not enabled_tfs:
        st.warning("请至少选择一个周期。")
        return

    prog = st.progress(0.0)
    for i, tf in enumerate(enabled_tfs):
        with st.spinner(f"拉取 {symbol} · {tf} 数据 & 计算指标中..."):
            df = engine.fetch_ohlcv(symbol, tf, limit=700)
            data_cache[tf] = df
            if df is None or len(df) < 120:
                signals[tf] = None
            else:
                analyst = SingleFrameAnalyst(df, tf)
                signals[tf] = analyst.analyze()
        prog.progress((i + 1) / max(len(enabled_tfs), 1))
    prog.empty()

    col_short, col_long = st.columns(2)

    with col_short:
        st.markdown("### 🎯 短线/超短线视角")
        for tf in ["1m", "5m", "15m"]:
            if tf in enabled_tfs:
                render_signal_block(signals.get(tf))

    with col_long:
        st.markdown("### 🌊 中线 / 波段 / 趋势视角")
        for tf in ["1h", "4h", "1d"]:
            if tf in enabled_tfs:
                render_signal_block(signals.get(tf))

    # ---------- 首席分析师统一裁决 ----------
    st.markdown("## 🏛 首席分析师 · 统一结论")

    chief = MultiFrameChiefAnalyst(signals)
    summary, stance, global_conviction = chief.synthesize()

    color_map = {
        "STRONG_BULL": "🟢",
        "BULL": "🟩",
        "NEUTRAL": "⚪",
        "BEAR": "🟥",
        "STRONG_BEAR": "🔴",
    }
    emoji = color_map.get(stance, "⚪")

    st.markdown(
        f"**{emoji} 总体立场：{stance} · 模型综合置信度：`{global_conviction:.0f} / 100`**  \n\n"
        f"{summary}"
    )

    st.markdown(
        "> 把所有时间尺度的交易员关在一个会议室里吵三小时，\n"
        "> 你现在看到的，就是他们“勉强达成一致”后的会议纪要。"
    )

    # ---------- 仓位与执行建议 ----------
    st.markdown("## 📦 仓位与执行模板（示意）")

    # 选择一个“主操作周期”作为执行参考：优先 1h / 4h / 15m / 1d
    main_sig = None
    for key in ["1h", "4h", "15m", "1d", "5m"]:
        if key in enabled_tfs and signals.get(key) is not None:
            sig = signals[key]
            if sig is not None and sig.stop_loss is not None:
                main_sig = sig
                break

    if main_sig is None or main_sig.stop_loss is None:
        st.info(
            "当前没有找到【既有方向又设定了止损】的主周期信号。\n\n"
            "这通常意味着：\n"
            "- 各周期意见分裂、力度不够；\n"
            "- 或者波动结构不支持合理的止损点位。\n\n"
            "在这种市场状态下，**观望本身就是一种非常职业的选择**。"
        )
    else:
        entry = main_sig.entry_hint
        stop = main_sig.stop_loss
        size, max_loss = compute_position(equity, risk_pct, entry, stop, contract_mult=1.0)

        dir_word = "做多" if main_sig.long_score > main_sig.short_score else "做空"
        rr1 = f"{main_sig.reward_risk_1:.1f}R" if main_sig.reward_risk_1 else "—"
        rr2 = f"{main_sig.reward_risk_2:.1f}R" if main_sig.reward_risk_2 else "—"

        st.markdown(f"### 当前主操作周期：**{main_sig.timeframe}** · 建议执行方向：**{dir_word}**")
        st.markdown(
            f"- 参考入场价：`{entry:,.4f}`  \n"
            f"- 防守止损：`{stop:,.4f}`  \n"
            f"- 止盈一档：`{main_sig.take_profit_1:,.4f}`（约 {rr1}）  \n"
            f"- 止盈二档：`{main_sig.take_profit_2:,.4f}`（约 {rr2}）"
        )

        st.markdown("#### 基于你的资金，模型建议的仓位是？")
        st.markdown(
            f"- 账户总资金：**{equity:,.0f} USDT**  \n"
            f"- 单笔愿意承受的最大回撤：**{risk_pct:.1f}% ≈ {max_loss:,.2f} USDT**  \n"
            f"- 在当前入场与止损距离下：  \n"
            f"  - **建议仓位 ≈ `{size:,.4f}` 币（1x 杠杆等效）**  \n"
        )

        st.markdown(
            "#### 这套仓位逻辑，背后真正的含义\n"
            "- 你不是在问“这次能赚多少”，而是在设计一个**统一的亏损上限**：\n"
            f"  - 不管行情多吓人，这一笔最多亏大约 **{risk_pct:.1f}%**，你睡得着觉。\n"
            "- 在这个前提下，让止损**放在“行情真的证明你错了”的位置**，\n"
            "  而不是放在“你情绪上受不了的地方”。\n"
            "- 只要你用同一套风险预算，去执行一批有统计优势的信号，\n"
            "  盈亏曲线自然会从“过山车”变成**相对平滑的权益曲线**。"
        )

    # ---------- 图表：价格 + 关键均线 ----------
    st.markdown("## 📈 价格行为与关键均线（用于肉眼 sanity check）")

    chart_tf = "1h" if "1h" in data_cache else (enabled_tfs[-1] if enabled_tfs else "1h")
    df_chart = data_cache.get(chart_tf)

    if df_chart is not None:
        dff = df_chart.tail(220)
        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=dff.index,
                open=dff["open"],
                high=dff["high"],
                low=dff["low"],
                close=dff["close"],
                increasing_line_color="#16a34a",
                decreasing_line_color="#dc2626",
                name="Price",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dff.index,
                y=dff["EMA_20"],
                line=dict(color="#60a5fa", width=1.2),
                name="EMA 20",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dff.index,
                y=dff["EMA_50"],
                line=dict(color="#fbbf24", width=1.0),
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
            height=480,
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
    else:
        st.info("图表数据不足，无法绘制 K 线。")

    st.markdown(
        "> 交易这件事，本质上就是：\n"
        "> 在一套有正期望的规则上，用**可控的风险**，\n"
        "> 对市场反复地、机械地敲同一种钉子。"
    )


if __name__ == "__main__":
    main()
