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
# 0. 全局配置：OKX（直连模式，适配 Streamlit Cloud）
# ============================================================

EXCHANGE_ID = "okx"

# 核心配置：开启速率限制，不使用代理（云端直连）
OKX_CONFIG = {
    "enableRateLimit": True,
    "timeout": 20000,
    "options": {
        "defaultType": "spot",     # 默认为现货，如需合约可改为 'swap' 但需处理 symbol 格式
    },
}


# ============================================================
# 1. 页面与专业级 UI 样式 (Bloomberg Terminal 风格)
# ============================================================

st.set_page_config(
    page_title="WallStreet Alpha Desk – OKX Edition",
    page_icon="🦅",
    layout="wide",
)

# 注入 CSS：为了防止 Markdown 解析错误，所有 CSS 压缩在 style 标签内
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');
    .stApp { background-color: #050712; color: #e5e7eb; font-family: 'Noto Sans SC', sans-serif; }
    h1, h2, h3, h4 { font-weight: 700; letter-spacing: 0.02em; color: #f3f4f6; }
    
    /* 侧边栏样式 */
    section[data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #1e293b; }
    
    /* 核心卡片容器 */
    .quant-card {
        background: radial-gradient(circle at top left, #1e293b 0%, #0f172a 60%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
    }
    .quant-card:hover { border-color: #475569; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4); }
    
    /* 头部信息 */
    .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #334155; }
    .card-title { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 16px; color: #fcd34d; }
    .card-score { font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 4px; }
    
    /* 信号标签颜色 */
    .bull-bg { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .bear-bg { background-color: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.4); }
    .neutral-bg { background-color: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.4); }
    
    /* 逻辑列表 */
    .logic-ul { list-style-type: none; padding: 0; margin: 0; font-size: 13px; line-height: 1.6; color: #cbd5e1; }
    .logic-li { margin-bottom: 4px; display: flex; align-items: flex-start; }
    .logic-icon { margin-right: 6px; color: #64748b; min-width: 12px; }
    
    /* 交易计划盒子 */
    .plan-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; background: #0f172a; padding: 10px; border-radius: 6px; border: 1px dashed #334155; }
    .plan-item { display: flex; flex-direction: column; }
    .plan-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; }
    .plan-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; }
    .val-green { color: #34d399; }
    .val-red { color: #fb7185; }
    
    /* 回测统计 */
    .bt-stat { margin-top: 10px; padding-top: 8px; border-top: 1px solid #334155; font-size: 12px; color: #94a3b8; display: flex; justify-content: space-between; }
    .bt-val { color: #f1f5f9; font-weight: 600; }
    
    /* 首席总结框 */
    .chief-box { background: linear-gradient(145deg, #1e1b4b, #0f172a); border: 1px solid #4f46e5; border-radius: 8px; padding: 20px; margin-top: 20px; }
    .chief-title { color: #818cf8; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }
    .chief-content { font-size: 16px; font-weight: 600; color: #e0e7ff; line-height: 1.6; }
    .chief-sub { font-size: 13px; color: #a5b4fc; margin-top: 8px; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 2. 核心数据结构 (Data Classes)
# ============================================================

@dataclass
class SignalResult:
    timeframe: str
    bias: str               # "BULL", "BEAR", "NEUTRAL"
    score: float            # -10.0 to +10.0
    confidence: float       # 0 to 100
    reasons: List[str]      # 逻辑依据列表
    
    # 交易计划
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    rr_ratio: float         # 盈亏比
    
    # 历史回测数据 (Backtest)
    bt_win_rate: float      # 0.0 to 1.0
    bt_total_trades: int
    bt_expectancy: float    # 每笔交易平均R值

# ============================================================
# 3. 数据引擎 (OKX Data Engine)
# ============================================================

class OKXDataEngine:
    def __init__(self):
        self.exchange = ccxt.okx(OKX_CONFIG)
        
    def get_market_price(self, symbol: str) -> Tuple[float, float]:
        """获取最新价格和24h涨跌幅"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last'], ticker['percentage']
        except Exception:
            return 0.0, 0.0

    def fetch_candles(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        获取K线数据并清洗。
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            st.error(f"数据拉取失败 [{timeframe}]: {str(e)}")
            return pd.DataFrame()

# ============================================================
# 4. 华尔街级分析核心 (The Alpha Brain)
# ============================================================

class AlphaAnalyst:
    """
    这是系统的核心大脑。
    包含：指标计算、多因子打分模型、动态止损算法、以及向量化回测引擎。
    """
    
    def __init__(self, df: pd.DataFrame, timeframe: str):
        self.df = df.copy()
        self.tf = timeframe
        self.label = self._format_tf(timeframe)
        self._calculate_indicators()
        
    def _format_tf(self, tf):
        mapping = {'1m': 'SCALPING (1m)', '5m': 'MOMENTUM (5m)', '15m': 'DAYTRADE (15m)', 
                   '1h': 'SWING (1h)', '4h': 'POSITION (4h)', '1d': 'TREND (1d)'}
        return mapping.get(tf, tf)

    def _calculate_indicators(self):
        """计算全套技术指标"""
        # 1. 趋势系
        self.df['EMA_20'] = ta.ema(self.df['close'], length=20)
        self.df['EMA_50'] = ta.ema(self.df['close'], length=50)
        self.df['EMA_200'] = ta.ema(self.df['close'], length=200)
        
        # 2. 趋势强度
        adx = ta.adx(self.df['high'], self.df['low'], self.df['close'], length=14)
        self.df['ADX'] = adx['ADX_14']
        
        # 3. 动能系
        self.df['RSI'] = ta.rsi(self.df['close'], length=14)
        
        # 4. 波动率 (用于止损)
        self.df['ATR'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=14)
        
        # 5. 资金流/量价
        self.df['OBV'] = ta.obv(self.df['close'], self.df['volume'])
        self.df['OBV_MA'] = ta.ema(self.df['OBV'], length=20)
        
        # 6. 布林带 (用于回归/突破)
        bb = ta.bbands(self.df['close'], length=20, std=2)
        self.df['BB_UP'] = bb['BBU_20_2.0']
        self.df['BB_LOW'] = bb['BBL_20_2.0']
        self.df['BB_W'] = bb['BBB_20_2.0'] # Bandwidth

        self.df.dropna(inplace=True)

    def analyze_signal(self) -> SignalResult:
        """执行当前K线的深度分析"""
        current = self.df.iloc[-1]
        prev = self.df.iloc[-2]
        
        score = 0.0
        reasons = []
        
        # --- 因子 1: EMA 均线排列 (趋势权重: 40%) ---
        if current['close'] > current['EMA_20'] > current['EMA_50']:
            score += 3.0
            reasons.append("多头排列：价格站稳 EMA20/50 之上，趋势向上。")
        elif current['close'] < current['EMA_20'] < current['EMA_50']:
            score -= 3.0
            reasons.append("空头排列：价格被 EMA20/50 压制，趋势向下。")
        else:
            reasons.append("均线纠缠：EMA 系统暂无明确方向，处于震荡或转折期。")
            
        # --- 因子 2: 趋势强度 ADX (过滤权重: 10%) ---
        if current['ADX'] > 25:
            score *= 1.2 # 趋势强劲时，放大当前信号权重
            reasons.append(f"ADX ({current['ADX']:.1f}) 显示当前趋势动能强劲，顺势交易胜率更高。")
        else:
            score *= 0.8 # 震荡市，缩小信号权重
            reasons.append(f"ADX ({current['ADX']:.1f}) 偏弱，市场处于无序震荡，需警惕假突破。")

        # --- 因子 3: RSI 动能与背离 (反转权重: 30%) ---
        if current['RSI'] > 70:
            score -= 1.5
            reasons.append(f"RSI ({current['RSI']:.1f}) 进入超买区，短期获利盘可能回吐。")
        elif current['RSI'] < 30:
            score += 1.5
            reasons.append(f"RSI ({current['RSI']:.1f}) 进入超卖区，技术性反弹概率增加。")
        
        # --- 因子 4: 资金流 OBV (确认权重: 20%) ---
        if current['OBV'] > current['OBV_MA']:
            score += 1.0
            reasons.append("资金流：OBV 位于均线上方，买盘量能健康。")
        else:
            score -= 1.0
            reasons.append("资金流：OBV 位于均线下方，上涨缺乏量能支撑。")

        # --- 综合裁决 ---
        confidence = min(abs(score) * 10, 100)
        bias = "NEUTRAL"
        if score >= 2.0: bias = "BULL"
        elif score <= -2.0: bias = "BEAR"
        
        # --- 动态交易计划 (ATR Based) ---
        atr = current['ATR']
        price = current['close']
        
        if bias == "BULL":
            # 多头：止损放在当前价格下方 1.5 - 2 倍 ATR
            sl = price - (2.0 * atr)
            risk = price - sl
            tp1 = price + (1.5 * risk)
            tp2 = price + (3.0 * risk)
        elif bias == "BEAR":
            # 空头：止损放在当前价格上方 1.5 - 2 倍 ATR
            sl = price + (2.0 * atr)
            risk = sl - price
            tp1 = price - (1.5 * risk)
            tp2 = price - (3.0 * risk)
        else:
            # 震荡：收窄止损
            sl = price * 0.99
            tp1 = price * 1.01
            tp2 = price * 1.02
            
        rr = 0.0
        if bias != "NEUTRAL" and abs(price - sl) > 0:
            rr = abs(tp1 - price) / abs(price - sl)

        # --- 实时回测 (Simulation) ---
        win_rate, trades, expectancy = self._run_backtest_logic()

        return SignalResult(
            timeframe=self.label,
            bias=bias,
            score=score,
            confidence=confidence,
            reasons=reasons,
            entry_price=price,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            rr_ratio=rr,
            bt_win_rate=win_rate,
            bt_total_trades=trades,
            bt_expectancy=expectancy
        )

    def _run_backtest_logic(self) -> Tuple[float, int, float]:
        """
        在当前 K 线图的历史数据上，运行完全相同的打分逻辑。
        这能告诉用户：'如果过去 500 根 K 线你都听我的，结果会怎样。'
        """
        wins = 0
        total = 0
        total_r = 0.0
        
        # 简单模拟：只看最近 200 根，避免计算太慢
        lookback = 200
        if len(self.df) < lookback + 50: return 0.0, 0, 0.0
        
        subset = self.df.iloc[-(lookback+20):-1] # 留最后几根没走完的不测
        
        for i in range(50, len(subset)-10):
            row = subset.iloc[i]
            
            # 简化的逻辑复刻 (为了速度)
            s = 0
            if row['close'] > row['EMA_20'] > row['EMA_50']: s += 3
            elif row['close'] < row['EMA_20'] < row['EMA_50']: s -= 3
            
            if row['RSI'] < 30: s += 1.5
            elif row['RSI'] > 70: s -= 1.5
            
            # 模拟交易结果
            outcome_r = 0
            entry = row['close']
            atr = row['ATR']
            
            if s >= 2.5: # 模拟做多
                sl = entry - 2.0 * atr
                tp = entry + 1.5 * (entry - sl)
                # 往后看 10 根 K 线
                future = subset.iloc[i+1:i+11]
                for _, f in future.iterrows():
                    if f['low'] <= sl: 
                        outcome_r = -1.0; break
                    if f['high'] >= tp:
                        outcome_r = 1.5; break
                total += 1
                if outcome_r > 0: wins += 1
                total_r += outcome_r
                
            elif s <= -2.5: # 模拟做空
                sl = entry + 2.0 * atr
                tp = entry - 1.5 * (sl - entry)
                future = subset.iloc[i+1:i+11]
                for _, f in future.iterrows():
                    if f['high'] >= sl:
                        outcome_r = -1.0; break
                    if f['low'] <= tp:
                        outcome_r = 1.5; break
                total += 1
                if outcome_r > 0: wins += 1
                total_r += outcome_r
                
        if total == 0: return 0.0, 0, 0.0
        return wins / total, total, total_r / total

# ============================================================
# 5. 首席分析师综合逻辑 (Synthesis)
# ============================================================

class ChiefAnalyst:
    @staticmethod
    def summarize(signals: List[SignalResult]) -> Tuple[str, str]:
        """汇总所有周期，给出最终结论"""
        bull_power = sum(s.confidence for s in signals if s.bias == "BULL")
        bear_power = sum(s.confidence for s in signals if s.bias == "BEAR")
        
        diff = bull_power - bear_power
        
        if diff > 150:
            title = "STRONG BUY / 强力做多结构"
            desc = "从短线到中长线，市场呈现完美的多头共振。资金、趋势、动能完全一致。建议激进做多，利用回调加仓。"
        elif diff > 50:
            title = "BUY / 震荡偏多"
            desc = "整体结构偏向多头，但可能存在短周期的回调压力或长周期的压制。建议逢低买入，避免追高。"
        elif diff < -150:
            title = "STRONG SELL / 强力做空结构"
            desc = "空头完全主导市场，多周期均线反压，资金持续流出。任何反弹都是做空的机会。"
        elif diff < -50:
            title = "SELL / 震荡偏空"
            desc = "市场重心下移，空头占优。建议在阻力位布局空单，设好防守。"
        else:
            title = "NEUTRAL / 激烈博弈"
            desc = "多空力量在不同周期打架（例如短线涨、长线跌）。此时市场缺乏方向，建议空仓观望或仅做超短线剥头皮。"
            
        return title, desc

# ============================================================
# 6. UI 渲染组件 (关键：解决 HTML 乱码的终极方案)
# ============================================================

def render_signal_card(res: SignalResult):
    """
    渲染单个周期的分析卡片。
    关键技术：使用 List Join 拼接 HTML，严禁换行符，确保 Streamlit 完美渲染。
    """
    if res.bias == "BULL":
        color_class = "bull-bg"
        icon = "🟢"
        score_txt = f"+{res.score:.1f}"
        bias_txt = "偏多 BULLISH"
    elif res.bias == "BEAR":
        color_class = "bear-bg"
        icon = "🔴"
        score_txt = f"{res.score:.1f}"
        bias_txt = "偏空 BEARISH"
    else:
        color_class = "neutral-bg"
        icon = "⚪"
        score_txt = f"{res.score:.1f}"
        bias_txt = "观望 NEUTRAL"

    # 构建逻辑列表 HTML
    logic_items = ""
    for reason in res.reasons:
        logic_items += f"<li class='logic-li'><span class='logic-icon'>›</span><span>{reason}</span></li>"
    
    # 构建回测数据 HTML
    win_rate_pct = res.bt_win_rate * 100
    expectancy_color = "#34d399" if res.bt_expectancy > 0 else "#fb7185"
    
    # ！！！核心修复：单行拼接，无缩进！！！
    html_parts = [
        "<div class='quant-card'>",
        "<div class='card-header'>",
        f"<div class='card-title'>{res.timeframe}</div>",
        f"<div class='card-score {color_class}'>{icon} {bias_txt} (Score: {score_txt})</div>",
        "</div>",
        f"<ul class='logic-ul'>{logic_items}</ul>",
        "<div class='plan-grid'>",
        f"<div class='plan-item'><span class='plan-label'>ENTRY</span><span class='plan-val'>${res.entry_price:,.2f}</span></div>",
        f"<div class='plan-item'><span class='plan-label'>STOP LOSS</span><span class='plan-val val-red'>${res.stop_loss:,.2f}</span></div>",
        f"<div class='plan-item'><span class='plan-label'>TARGET 1</span><span class='plan-val val-green'>${res.take_profit_1:,.2f}</span></div>",
        f"<div class='plan-item'><span class='plan-label'>RISK/REWARD</span><span class='plan-val'>{res.rr_ratio:.2f}R</span></div>",
        "</div>",
        "<div class='bt-stat'>",
        f"<span>因子回测 (近{res.bt_total_trades}笔)</span>",
        f"<span>胜率: <b style='color:#e2e8f0'>{win_rate_pct:.1f}%</b> &nbsp;|&nbsp; 期望值: <b style='color:{expectancy_color}'>{res.bt_expectancy:+.2f}R</b></span>",
        "</div>",
        "</div>"
    ]
    
    st.markdown("".join(html_parts), unsafe_allow_html=True)

def render_position_calculator(equity, risk_pct, entry, stop):
    """渲染仓位计算器"""
    if entry == 0 or stop == 0 or entry == stop:
        return
        
    risk_amt = equity * (risk_pct / 100)
    price_diff = abs(entry - stop)
    position_size = risk_amt / price_diff
    
    # 杠杆建议（简化版：名义价值/本金）
    notional = position_size * entry
    lev = notional / equity
    
    # 单行 HTML 拼接
    html = "".join([
        "<div class='quant-card' style='border-color: #6366f1; background: rgba(99, 102, 241, 0.05);'>",
        "<div class='card-title' style='color:#818cf8; margin-bottom:8px;'>📦 机构级仓位风控建议 (Position Sizing)</div>",
        "<div style='font-size:14px; color:#cbd5e1; line-height:1.6;'>",
        f"基于您 <b>${equity:,.0f}</b> 的本金，单笔风险限制在 <b>{risk_pct}% (${risk_amt:.1f})</b>：<br/>",
        f"建议开仓数量：<b style='color:#fff; font-size:18px;'>{position_size:.4f} 币</b><br/>",
        f"<span style='font-size:12px; color:#94a3b8'>(隐含杠杆率约为 {lev:.1f}x · 止损即亏损 ${risk_amt:.1f})</span>",
        "</div></div>"
    ])
    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# 7. 主程序 (Main Entry)
# ============================================================

def main():
    # --- 侧边栏配置 ---
    with st.sidebar:
        st.markdown("### 📡 ALPHA DESK SETUP")
        symbol = st.selectbox("选择标的 (Spot)", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "AVAX/USDT"], index=0)
        
        st.markdown("---")
        st.markdown("### 💰 资金管理 (Risk Mgmt)")
        equity = st.number_input("账户总权益 (USDT)", value=10000.0, step=1000.0)
        risk = st.slider("单笔最大风险 (%)", 0.5, 5.0, 2.0, 0.5)
        
        st.info("数据来源：OKX Public API\n模式：Direct Connect (No Proxy)\n延迟：实时")

    # --- 头部行情 ---
    engine = OKXDataEngine()
    price, pct = engine.get_market_price(symbol)
    
    color = "#34d399" if pct >= 0 else "#fb7185"
    utc_now = datetime.utcnow().strftime("%H:%M:%S UTC")
    
    # 头部横幅 HTML
    st.markdown("".join([
        "<div style='display:flex; align-items:baseline; gap:12px; margin-bottom:20px;'>",
        f"<h1 style='margin:0; font-size:32px;'>{symbol}</h1>",
        f"<span style='font-size:24px; font-family:monospace; font-weight:700; color:#f8fafc'>${price:,.2f}</span>",
        f"<span style='font-size:16px; color:{color}; font-weight:600'>{pct:+.2f}%</span>",
        f"<span style='margin-left:auto; font-size:12px; color:#64748b'>MARKET OPEN · {utc_now}</span>",
        "</div>"
    ]), unsafe_allow_html=True)

    # --- 核心分析循环 ---
    timeframes = ['15m', '1h', '4h', '1d']
    results = []
    
    progress = st.progress(0)
    
    # 布局：左侧分析卡片，右侧总结与图表
    col_left, col_right = st.columns([0.55, 0.45])
    
    with col_left:
        st.markdown("### 🔬 Multi-Timeframe Analysis")
        for i, tf in enumerate(timeframes):
            df = engine.fetch_candles(symbol, tf)
            if not df.empty and len(df) > 50:
                analyst = AlphaAnalyst(df, tf)
                res = analyst.analyze_signal()
                results.append(res)
                render_signal_card(res)
            progress.progress((i + 1) / len(timeframes))
            
    progress.empty()

    with col_right:
        # 1. 首席分析师总结
        if results:
            g_title, g_desc = ChiefAnalyst.summarize(results)
            st.markdown("".join([
                "<div class='chief-box'>",
                f"<div class='chief-title'>🏛 CHIEF ANALYST VERDICT</div>",
                f"<div class='chief-content'>{g_title}</div>",
                f"<div style='margin-top:8px; font-size:14px; color:#cbd5e1;'>{g_desc}</div>",
                "<div class='chief-sub'>* 基于多周期因子加权的一致性评估</div>",
                "</div>"
            ]), unsafe_allow_html=True)
            
            # 2. 仓位计算
            # 选取 1H 或 4H 的信号作为主交易参考
            ref_signal = next((r for r in results if r.timeframe.startswith('SWING') or r.timeframe.startswith('POSITION')), results[0])
            st.markdown("### 🛡️ Position Sizing")
            render_position_calculator(equity, risk, ref_signal.entry_price, ref_signal.stop_loss)

        # 3. 交互式图表 (1H)
        st.markdown("### 📈 Market Structure (1H)")
        chart_df = engine.fetch_candles(symbol, '1h', limit=200)
        if not chart_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name='OHLC'))
            # 添加 EMA
            ema20 = ta.ema(chart_df['close'], 20)
            ema50 = ta.ema(chart_df['close'], 50)
            fig.add_trace(go.Scatter(x=chart_df.index, y=ema20, line=dict(color='#fbbf24', width=1), name='EMA 20'))
            fig.add_trace(go.Scatter(x=chart_df.index, y=ema50, line=dict(color='#60a5fa', width=1), name='EMA 50'))
            
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                xaxis_rangeslider_visible=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#1e293b')
            )
            st.plotly_chart(fig, use_container_width=True)

    # 底部免责
    st.markdown("---")
    st.markdown("".join([
        "<div style='text-align:center; color:#475569; font-size:12px;'>",
        "WALLSTREET ALPHA DESK © 2025 • QUANTITATIVE RESEARCH ONLY • NOT FINANCIAL ADVICE",
        "</div>"
    ]), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
