import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import MACD, EMAIndicator, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import concurrent.futures
import time

# ==========================================
# 1. 系统配置与黑客帝国风UI
# ==========================================
st.set_page_config(page_title="Titan Alpha V3 | God Mode", layout="wide", page_icon="👁️")

st.markdown("""
<style>
    /* 全局暗黑风格 */
    .stApp { background-color: #050505; }
    
    /* 卡片容器 */
    .css-1r6slb0 { background-color: #111; border: 1px solid #333; }
    
    /* 信号卡片 */
    .signal-card {
        background-color: #121212;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .signal-card:hover { transform: scale(1.02); border-color: #555; }
    
    /* 颜色定义 */
    .bull { color: #00ff88; font-weight: bold; text-shadow: 0 0 10px rgba(0, 255, 136, 0.3); }
    .bear { color: #ff3355; font-weight: bold; text-shadow: 0 0 10px rgba(255, 51, 85, 0.3); }
    .neutral { color: #888; }
    
    /* 标题 */
    h1, h2, h3 { font-family: 'JetBrains Mono', monospace; color: #eee; }
    .metric-label { font-size: 0.8em; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 1.2em; color: #fff; font-weight: 500; }
    
    /* 分隔线 */
    hr { border-color: #333; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 高性能数据核心 (Parallel Fetching)
# ==========================================

class DataEngine:
    def __init__(self):
        self.exchange = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

    @st.cache_data(ttl=3600)
    def get_symbols(_self):
        try:
            mkts = _self.exchange.load_markets()
            return [k for k in mkts.keys() if 'USDT' in k and ':' in k]
        except: return ["BTC/USDT:USDT", "ETH/USDT:USDT"]

    def fetch_all_timeframes(self, symbol):
        """并发抓取4个周期的数据，极速响应"""
        timeframes = ['15m', '1h', '4h', '1d']
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_tf = {executor.submit(self._fetch_single, symbol, tf): tf for tf in timeframes}
            for future in concurrent.futures.as_completed(future_to_tf):
                tf = future_to_tf[future]
                try:
                    results[tf] = future.result()
                except Exception as e:
                    results[tf] = pd.DataFrame()
        return results

    def _fetch_single(self, symbol, tf):
        # 获取更多数据以保证指标稳定
        ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=300)
        if not ohlcv: return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        return df.astype(float)

# ==========================================
# 3. 华尔街分析逻辑 (Deep Analytics)
# ==========================================

class Analyst:
    def __init__(self, df):
        self.df = df
        
    def analyze(self):
        if len(self.df) < 50: return None
        
        # --- 基础指标 ---
        c = self.df['c']
        h = self.df['h']
        l = self.df['l']
        
        ema20 = EMAIndicator(c, 20).ema_indicator()
        ema200 = EMAIndicator(c, 200).ema_indicator()
        rsi = RSIIndicator(c).rsi()
        macd = MACD(c).macd_diff()
        atr = AverageTrueRange(h, l, c).average_true_range()
        bb_h = BollingerBands(c).bollinger_hband()
        bb_l = BollingerBands(c).bollinger_lband()
        cci = CCIIndicator(h, l, c).cci()
        
        curr = self.df.iloc[-1]
        prev = self.df.iloc[-2]
        
        # --- 深度逻辑判断 ---
        
        # 1. 趋势状态 (Market Regime)
        trend_score = 0
        if curr['c'] > ema20.iloc[-1]: trend_score += 2
        if ema20.iloc[-1] > ema200.iloc[-1]: trend_score += 2
        if macd.iloc[-1] > 0: trend_score += 1
        if macd.iloc[-1] > macd.iloc[-2]: trend_score += 1 # 动能增强
        
        # 2. 反转/超买超卖 (Reversal Risk)
        osc_score = 0 # 正数利多，负数利空
        if rsi.iloc[-1] > 70: osc_score -= 3
        elif rsi.iloc[-1] < 30: osc_score += 3
        
        if cci.iloc[-1] > 100 and cci.iloc[-2] > cci.iloc[-1]: osc_score -= 2 # CCI拐头向下
        
        # 3. 关键点位 (Key Levels)
        support = bb_l.iloc[-1]
        resistance = bb_h.iloc[-1]
        
        # 综合评分 (-10 ~ +10)
        total_score = trend_score + osc_score
        
        # 4. 信号生成
        signal_type = "NEUTRAL"
        if total_score >= 4: signal_type = "STRONG BULL"
        elif total_score >= 1: signal_type = "WEAK BULL"
        elif total_score <= -4: signal_type = "STRONG BEAR"
        elif total_score <= -1: signal_type = "WEAK BEAR"
        
        # 5. 止盈止损建议
        volatility = atr.iloc[-1]
        sl_p = curr['c'] - 2*volatility if total_score > 0 else curr['c'] + 2*volatility
        tp_p = curr['c'] + 3*volatility if total_score > 0 else curr['c'] - 3*volatility
        
        return {
            "price": curr['c'],
            "score": total_score,
            "signal": signal_type,
            "trend_strength": abs(trend_score),
            "volatility": volatility,
            "support": support,
            "resistance": resistance,
            "sl": sl_p,
            "tp": tp_p,
            "rsi": rsi.iloc[-1],
            "is_squeeze": (bb_h.iloc[-1] - bb_l.iloc[-1]) < (2 * volatility) # 布林带挤压
        }

# ==========================================
# 4. 页面渲染 (The God View)
# ==========================================

def render_card(tf, data):
    """渲染单个周期的分析卡片"""
    if not data:
        st.error(f"{tf} No Data")
        return None
        
    # 样式逻辑
    color_class = "bull" if "BULL" in data['signal'] else ("bear" if "BEAR" in data['signal'] else "neutral")
    bg_color = "rgba(0, 255, 136, 0.05)" if "BULL" in data['signal'] else ("rgba(255, 51, 85, 0.05)" if "BEAR" in data['signal'] else "rgba(255,255,255,0.02)")
    border_color = "#00cc96" if "BULL" in data['signal'] else ("#ef553b" if "BEAR" in data['signal'] else "#444")
    
    arrow = "⬆" if "BULL" in data['signal'] else ("⬇" if "BEAR" in data['signal'] else "➡")
    
    st.markdown(f"""
    <div class="signal-card" style="border-left: 5px solid {border_color}; background: {bg_color}">
        <div style="display:flex; justify-content:space-between; align-items:center">
            <h3 style="margin:0">{tf} 周期</h3>
            <span class="{color_class}" style="font-size:1.2em">{arrow} {data['signal']}</span>
        </div>
        <div style="margin-top:10px; font-size:0.9em; color:#ccc">
            <div><span class="metric-label">RSI指标:</span> <span style="color:{'#f00' if data['rsi']>70 else '#0f0' if data['rsi']<30 else '#fff'}">{data['rsi']:.1f}</span></div>
            <div><span class="metric-label">建议入场:</span> {data['price']:.2f}</div>
            <div style="display:flex; justify-content:space-between; margin-top:5px">
                <span style="color:#ef553b">🛑 {data['sl']:.2f}</span>
                <span style="color:#00cc96">🎯 {data['tp']:.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    return data['score'] # 返回分数用于汇总

def main():
    # --- 侧边栏精简 ---
    st.sidebar.header("⚙️ TITAN SETTINGS")
    engine = DataEngine()
    symbols = engine.get_symbols()
    symbol = st.sidebar.selectbox("Symbol", symbols, index=0)
    
    if st.sidebar.button("⚡ SYSTEM SCAN", type="primary"):
        
        # 1. 头部行情区
        data_map = engine.fetch_all_timeframes(symbol)
        if not data_map.get('1d') is None and not data_map['1d'].empty:
            curr_price = data_map['1d']['c'].iloc[-1]
            chg_24h = (curr_price - data_map['1d']['c'].iloc[-2]) / data_map['1d']['c'].iloc[-2] * 100
            
            c1, c2, c3 = st.columns([2,1,1])
            c1.markdown(f"<h1 style='margin:0'>{symbol}</h1>", unsafe_allow_html=True)
            c2.metric("Price", f"{curr_price:.4f}")
            c3.metric("24H Change", f"{chg_24h:.2f}%", delta=f"{chg_24h:.2f}%")
        else:
            st.error("Data connection failed. Please retry.")
            return

        st.markdown("---")

        # 2. 并列分析矩阵 (The Matrix)
        st.markdown("### 🧬 MULTI-TIMEFRAME MATRIX")
        cols = st.columns(4)
        timeframes = ['15m', '1h', '4h', '1d']
        scores = []
        
        reports = {} # 存储每个周期的详细报告
        
        # 渲染四个并列卡片
        for idx, tf in enumerate(timeframes):
            with cols[idx]:
                df = data_map.get(tf)
                if df is not None and not df.empty:
                    analyst = Analyst(df)
                    res = analyst.analyze()
                    if res:
                        score = render_card(tf, res)
                        scores.append(score)
                        reports[tf] = res
                    else:
                        st.warning("Insufficient Data")
                else:
                    st.warning("Fetch Err")

        # 3. 首席策略合成 (Chief Strategist Synthesis)
        st.markdown("---")
        st.markdown("### 🧠 CHIEF ANALYST VERDICT")
        
        if len(scores) == 4:
            # 加权算法: 日线(30%) + 4H(30%) + 1H(25%) + 15m(15%)
            final_score = scores[3]*0.3 + scores[2]*0.3 + scores[1]*0.25 + scores[0]*0.15
            
            verdict_color = "#00ff88" if final_score > 2 else ("#ff3355" if final_score < -2 else "#888")
            verdict_text = "STRONG BUY" if final_score > 4 else ("BUY" if final_score > 1 else ("STRONG SELL" if final_score < -4 else ("SELL" if final_score < -1 else "WAIT & SEE")))
            
            # 构建深度分析文本
            analysis_text = ""
            if scores[3] > 0 and scores[0] < 0:
                analysis_text = "⚠️ **背离警告 (Divergence):** 宏观趋势(1D)看涨，但微观结构(15m)正在回调。建议等待15m周期RSI降至30附近尝试接多，不要盲目追高。"
            elif scores[3] > 0 and scores[2] > 0 and scores[1] > 0:
                analysis_text = "🚀 **共振突破 (Resonance):** 全周期多头共振！这通常意味着趋势加速阶段。激进者可现价介入，防守位设在1H周期的ATR下沿。"
            elif scores[3] < 0 and scores[2] < 0:
                analysis_text = "📉 **主跌浪 (Downtrend):** 日线与4小时同步看空，反弹即是空点。关注1H周期的压力位（布林上轨）作为做空入场点。"
            else:
                analysis_text = "⚖️ **震荡整理 (Consolidation):** 周期信号冲突，市场缺乏明确方向。建议采用网格策略或观望，等待关键点位突破。"

            # 最终大面板
            st.markdown(f"""
            <div style="background: #1a1a1a; padding: 25px; border-radius: 15px; border: 1px solid {verdict_color}; display:flex; align-items:center; gap:30px">
                <div style="text-align:center; min-width: 150px;">
                    <div style="font-size: 4em; line-height: 1em;">{ '🐂' if final_score > 0 else '🐻' }</div>
                    <h2 style="color: {verdict_color}; margin:10px 0">{verdict_text}</h2>
                    <div style="color:#aaa">Confidence: {abs(final_score)*10:.0f}%</div>
                </div>
                <div style="border-left: 2px solid #444; padding-left: 30px;">
                    <h4 style="color:#eee; margin-top:0">📈 策略逻辑合成 (Strategy Synthesis)</h4>
                    <p style="font-size: 1.1em; color: #ddd; line-height: 1.6;">{analysis_text}</p>
                    <div style="display:flex; gap: 20px; margin-top: 15px;">
                        <span style="background:#222; padding:5px 10px; border-radius:4px; font-size:0.9em; border:1px solid #444">主力周期: 4H</span>
                        <span style="background:#222; padding:5px 10px; border-radius:4px; font-size:0.9em; border:1px solid #444">波动率状态: {'High' if reports['4h']['volatility'] > reports['1d']['volatility']/4 else 'Normal'}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 4. 图表辅助 (只展示最重要的4H和1H)
            st.markdown("### 👁️ MARKET VISION")
            tab1, tab2 = st.tabs(["4H Structure (Trend)", "1H Structure (Entry)"])
            
            def plot_chart(tf):
                d = data_map[tf]
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=d.index, open=d['o'], high=d['h'], low=d['l'], close=d['c'], name='Price'), row=1, col=1)
                fig.add_trace(go.Scatter(x=d.index, y=EMAIndicator(d['c'], 20).ema_indicator(), line=dict(color='#ff9900', width=1), name='EMA 20'), row=1, col=1)
                # Add Entry/SL/TP Lines
                r = reports[tf]
                fig.add_hline(y=r['tp'], line_dash="dot", line_color="#00cc96", annotation_text="TP Target", row=1, col=1)
                fig.add_hline(y=r['sl'], line_dash="dot", line_color="#ef553b", annotation_text="SL Protect", row=1, col=1)
                
                # RSI
                rsi = RSIIndicator(d['c']).rsi()
                fig.add_trace(go.Scatter(x=d.index, y=rsi, line=dict(color='#aaddff', width=1.5), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with tab1: plot_chart('4h')
            with tab2: plot_chart('1h')
            
        else:
            st.error("Analysis incomplete due to missing timeframe data.")

    else:
        st.info("Waiting for command... Select symbol and click SYSTEM SCAN.")

if __name__ == "__main__":
    main()
