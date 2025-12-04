import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="Crypto Swing Trader (US Edition)",
    page_icon="🦅",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e6edf3; }
    .signal-card {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        padding: 20px; border-radius: 10px;
    }
    /* 强调 USD 符号 */
    .usd-tag { color: #85bb65; font-weight: bold; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据引擎 (Coinbase)
# ==========================================
class MarketData:
    def __init__(self):
        # 使用 Coinbase 交易所 (美国合规，无需 API Key 可获取公共行情)
        self.exchange = ccxt.coinbase({
            'enableRateLimit': True,
            'timeout': 30000,
            # 美国本地无需代理
        })

    def fetch_data(self, symbol, timeframe="4h", limit=150):
        try:
            # Coinbase 的 4h 数据可能需要映射，这里用 standard timeframe
            # 如果 fetch_ohlcv 报错，通常是因为交易对名称 (如 BTC/USD)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df, None
        except Exception as e:
            return pd.DataFrame(), f"Coinbase 连接错误: {str(e)}"

# ==========================================
# 3. 策略核心 (BB + KDJ + ATR)
# ==========================================
class SwingStrategy:
    def __init__(self, df):
        self.df = df

    def process_indicators(self):
        if self.df.empty: return self.df
        
        # 1. Bollinger Bands (20, 2)
        bb = ta.bbands(self.df['close'], length=20, std=2)
        self.df = pd.concat([self.df, bb], axis=1)
        
        # 2. KDJ (随机指标)
        kdj = ta.kdj(self.df['high'], self.df['low'], self.df['close'])
        self.df = pd.concat([self.df, kdj], axis=1)
        
        # 3. ATR (波动率)
        self.df['ATR'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=14)
        
        return self.df.dropna()

    def analyze_signal(self):
        curr = self.df.iloc[-1]
        prev = self.df.iloc[-2]
        
        price = curr['close']
        bbu = curr['BBU_20_2.0'] # 上轨
        bbl = curr['BBL_20_2.0'] # 下轨
        bbm = curr['BBM_20_2.0'] # 中轨
        
        # KDJ 值
        k_val = curr['K_9_3']
        d_val = curr['D_9_3']
        
        score = 0
        reasons = []
        
        # --- 逻辑 A: 布林带位置 ---
        bb_pos = (price - bbl) / (bbu - bbl) # 0=下轨, 1=上轨
        
        if bb_pos < 0.1:
            score += 2
            reasons.append("📉 价格触及布林带下轨 (超卖)，关注反弹")
        elif bb_pos > 0.9:
            score -= 2
            reasons.append("📈 价格触及布林带上轨 (超买)，关注回调")
            
        # --- 逻辑 B: KDJ 交叉 ---
        kdj_gold = (prev['K_9_3'] < prev['D_9_3']) and (curr['K_9_3'] > curr['D_9_3'])
        kdj_dead = (prev['K_9_3'] > prev['D_9_3']) and (curr['K_9_3'] < curr['D_9_3'])
        
        if kdj_gold and k_val < 40:
            score += 1.5
            reasons.append("⚡ KDJ 低位金叉确认")
        elif kdj_dead and k_val > 60:
            score -= 1.5
            reasons.append("⚡ KDJ 高位死叉确认")
            
        # --- 结论 ---
        direction = "观望 (Neutral)"
        signal_type = "neutral"
        
        if score >= 2.5:
            direction = "做多机会 (LONG ENTRY)"
            signal_type = "long"
        elif score <= -2.5:
            direction = "做空机会 (SHORT ENTRY)"
            signal_type = "short"
        elif score > 0: direction = "震荡偏多"
        elif score < 0: direction = "震荡偏空"
            
        return {
            "direction": direction,
            "type": signal_type,
            "score": score,
            "reasons": reasons,
            "price": price,
            "atr": curr['ATR'],
            "bb_upper": bbu, "bb_lower": bbl, "bb_mid": bbm
        }

# ==========================================
# 4. 主程序
# ==========================================
def main():
    with st.sidebar:
        st.header("🇺🇸 市场设置")
        # 主流币种选择 (Coinbase使用 USD 交易对)
        symbol_base = st.selectbox("选择币种", ["BTC", "ETH", "SOL", "DOGE", "LINK", "LTC"])
        symbol = f"{symbol_base}/USD"
        
        tf = st.selectbox("时间周期", ["1h", "4h", "1d"], index=1, 
                         help="波段交易推荐 4h")
        
        if st.button("开始分析", type="primary"):
            st.rerun()
        
        st.info(f"数据源: Coinbase Public API\n网络: 直连 (无需代理)")

    st.title(f"🦅 {symbol} 波段交易终端")

    # --- 获取数据 ---
    api = MarketData()
    with st.spinner(f"Connecting to Coinbase ({symbol})..."):
        raw_df, err = api.fetch_data(symbol, tf, limit=150)
        
    if err:
        st.error(f"无法获取数据: {err}")
        st.warning("Coinbase 可能暂时限制了请求，请稍后重试，或检查网络是否通畅。")
        return

    if raw_df.empty:
        st.error("获取到的数据为空，可能是 Coinbase 不支持该交易对的此周期。")
        return

    # --- 运行策略 ---
    strategy = SwingStrategy(raw_df)
    df = strategy.process_indicators()
    res = strategy.analyze_signal()
    
    atr = res['atr']
    price = res['price']
    
    # 计算止损止盈 (主流币波动小一点，ATR倍数稍微调低)
    sl_mult = 1.5
    
    if res['type'] == 'long':
        sl = price - (sl_mult * atr)
        tp1 = res['bb_mid']
        tp2 = res['bb_upper']
    elif res['type'] == 'short':
        sl = price + (sl_mult * atr)
        tp1 = res['bb_mid']
        tp2 = res['bb_lower']
    else:
        # 震荡参考 (做多视角)
        sl = price - (sl_mult * atr)
        tp1 = res['bb_mid']
        tp2 = res['bb_upper']

    # --- UI 展示 ---
    col1, col2 = st.columns([3, 2])
    with col1:
        color = "#8b949e"
        if res['type'] == 'long': color = "#3fb950"
        elif res['type'] == 'short': color = "#f85149"
        
        st.markdown(f"""
        <div class="signal-card" style="border-left: 5px solid {color};">
            <div style="color:#8b949e;">AI 策略建议</div>
            <div style="font-size:36px; font-weight:bold; color:{color}; margin: 10px 0;">{res['direction']}</div>
            <div style="font-size:20px;">现价: <span class="usd-tag">${price:,.2f}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""<div style="height:10px"></div>""", unsafe_allow_html=True)
        st.metric("市场波动值 (ATR)", f"{atr:.2f}", "用于计算安全止损")

    # 交易计划
    st.subheader("🎯 交易执行计划 (Trade Plan)")
    p1, p2, p3 = st.columns(3)
    p1.metric("🛑 止损位 (SL)", f"${sl:,.2f}", f"-{sl_mult} ATR")
    p2.metric("💰 目标一 (TP1)", f"${tp1:,.2f}", "中轨回归")
    p3.metric("🚀 目标二 (TP2)", f"${tp2:,.2f}", "极值利润")

    # 图表
    st.subheader(f"📈 {symbol} 趋势结构")
    fig = go.Figure()
    
    # 布林带
    fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', line=dict(width=0), showlegend=False, name='Bollinger'))
    
    # K线
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
    
    # 中轨
    fig.add_trace(go.Scatter(x=df.index, y=df['BBM_20_2.0'], line=dict(color='orange', width=1), name='MA20 Base'))
    
    # 标记点位
    if res['type'] != 'neutral':
        fig.add_hline(y=tp1, line_dash="dot", line_color="green", annotation_text="TP1")
        fig.add_hline(y=sl, line_dash="dot", line_color="red", annotation_text="SL")

    fig.update_layout(template='plotly_dark', height=500, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, plot_bgcolor='#0d1117', paper_bgcolor='#0d1117')
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("查看详细逻辑"):
        for r in res['reasons']: st.write(r)

if __name__ == "__main__":
    main()
