import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="ZEC Swing Trader (1-2 Days)",
    page_icon="🌊",
    layout="wide"
)

# 样式：强调波动交易的视觉
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e6edf3; }
    .signal-card {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .tag-long { background: #238636; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    .tag-short { background: #da3633; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    .metric-val { font-size: 24px; font-weight: bold; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据引擎 (含代理)
# ==========================================
class BinanceData:
    def __init__(self, proxy_url=None):
        config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'timeout': 30000
        }
        if proxy_url:
            config['proxies'] = {'http': proxy_url, 'https': proxy_url}
            
        self.exchange = ccxt.binance(config)
        self.proxy = proxy_url

    def fetch_data(self, symbol="ZEC/USDT", timeframe="4h", limit=150):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df, None
        except Exception as e:
            return pd.DataFrame(), str(e)

# ==========================================
# 3. 波段策略核心 (Swing Strategy)
# ==========================================
class SwingStrategy:
    def __init__(self, df):
        self.df = df

    def process_indicators(self):
        if self.df.empty: return self.df
        
        # 1. 布林带 (Bollinger Bands, 20, 2) - 核心波动区间
        bb = ta.bbands(self.df['close'], length=20, std=2)
        self.df = pd.concat([self.df, bb], axis=1)
        # 列名通常为: BBL_20_2.0 (下), BBM_20_2.0 (中), BBU_20_2.0 (上)
        
        # 2. KDJ (随机指标) - 敏感买卖点
        # pandas_ta 默认 KDJ: length=9, signal=3
        kdj = ta.kdj(self.df['high'], self.df['low'], self.df['close'])
        self.df = pd.concat([self.df, kdj], axis=1)
        # 列名: K_9_3, D_9_3, J_9_3
        
        # 3. ATR - 止损计算
        self.df['ATR'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=14)
        
        return self.df.dropna()

    def analyze_signal(self):
        curr = self.df.iloc[-1]
        prev = self.df.iloc[-2]
        
        price = curr['close']
        bbu = curr['BBU_20_2.0']
        bbl = curr['BBL_20_2.0']
        bbm = curr['BBM_20_2.0'] # 中轨 (MA20)
        
        k_val = curr['K_9_3']
        d_val = curr['D_9_3']
        
        score = 0
        reasons = []
        
        # --- 逻辑 A: 布林带位置 ---
        # 价格接近下轨 -> 偏多; 接近上轨 -> 偏空
        bb_pos = (price - bbl) / (bbu - bbl) # 0=下轨, 1=上轨
        
        if bb_pos < 0.1:
            score += 2
            reasons.append("📉 价格触及布林带下轨 (超卖区域)，有反弹需求")
        elif bb_pos > 0.9:
            score -= 2
            reasons.append("📈 价格触及布林带上轨 (超买区域)，有回调压力")
        elif bb_pos < 0.4:
            score += 0.5
            reasons.append("🔹 价格运行在布林带下半区，偏弱但有支撑")
        elif bb_pos > 0.6:
            score -= 0.5
            reasons.append("🔸 价格运行在布林带上半区，偏强但有阻力")
            
        # --- 逻辑 B: KDJ 交叉 (核心触发器) ---
        # 金叉：K线上穿D线
        kdj_gold = (prev['K_9_3'] < prev['D_9_3']) and (curr['K_9_3'] > curr['D_9_3'])
        kdj_dead = (prev['K_9_3'] > prev['D_9_3']) and (curr['K_9_3'] < curr['D_9_3'])
        
        if kdj_gold and k_val < 40:
            score += 1.5
            reasons.append("⚡ KDJ 低位金叉：短期动能转强信号")
        elif kdj_dead and k_val > 60:
            score -= 1.5
            reasons.append("⚡ KDJ 高位死叉：短期动能衰竭信号")
            
        # --- 综合判定 ---
        direction = "观望 (Neutral)"
        signal_type = "neutral"
        
        if score >= 2.5:
            direction = "波段做多 (LONG SWING)"
            signal_type = "long"
        elif score <= -2.5:
            direction = "波段做空 (SHORT SWING)"
            signal_type = "short"
        elif score > 0:
            direction = "震荡偏多 (Weak Bull)"
        elif score < 0:
            direction = "震荡偏空 (Weak Bear)"
            
        return {
            "direction": direction,
            "type": signal_type,
            "score": score,
            "reasons": reasons,
            "price": price,
            "atr": curr['ATR'],
            "bb_upper": bbu,
            "bb_lower": bbl,
            "bb_mid": bbm
        }

# ==========================================
# 4. 主程序
# ==========================================
def main():
    # --- 侧边栏 ---
    with st.sidebar:
        st.header("📡 设置")
        use_proxy = st.checkbox("启用代理", value=True)
        proxy_url = st.text_input("代理地址", "http://127.0.0.1:7890")
        
        st.divider()
        st.subheader("策略周期")
        # 1-2天波段通常看 4H K线最准
        tf = st.selectbox("分析周期", ["1h", "4h"], index=1, help="1h适合日内，4h适合1-2天波段")
        
        if st.button("执行分析", type="primary"):
            st.rerun()

    st.title("🌊 ZEC 波段猎手 (Swing Hunter)")
    st.caption(f"目标: 捕捉 {tf} 级别波动 | 策略: 布林带均值回归 + KDJ 动能")

    # --- 获取数据 ---
    api = BinanceData(proxy_url if use_proxy else None)
    with st.spinner("正在连接市场..."):
        raw_df, err = api.fetch_data("ZEC/USDT", tf, limit=100)
        
    if err:
        st.error(f"数据连接失败: {err}")
        return
    if raw_df.empty:
        st.warning("未获取到数据，请检查代理。")
        return

    # --- 运行策略 ---
    strategy = SwingStrategy(raw_df)
    df = strategy.process_indicators()
    res = strategy.analyze_signal()
    
    # --- 计算波段止盈止损 ---
    # 波段交易止损：通常放在布林带轨道外侧一点 + 1倍ATR
    # 止盈：第一目标是中轨(回归)，第二目标是另一侧轨道
    
    atr = res['atr']
    price = res['price']
    
    if res['type'] == 'long':
        sl = price - (1.5 * atr) # 稍微宽一点防震荡
        tp1 = res['bb_mid'] # 中轨回归
        tp2 = res['bb_upper'] # 趋势延续
    elif res['type'] == 'short':
        sl = price + (1.5 * atr)
        tp1 = res['bb_mid']
        tp2 = res['bb_lower']
    else:
        # 震荡中，假设做多给出参考
        sl = price - (1.5 * atr)
        tp1 = res['bb_mid']
        tp2 = res['bb_upper']

    # --- UI 展示 ---
    
    # 1. 信号卡片
    col1, col2 = st.columns([3, 2])
    
    with col1:
        color = "#8b949e"
        if res['type'] == 'long': color = "#3fb950"
        elif res['type'] == 'short': color = "#f85149"
        
        st.markdown(f"""
        <div class="signal-card" style="border-left: 5px solid {color};">
            <div style="color:#8b949e; font-size:14px;">当前策略建议</div>
            <div style="font-size:32px; font-weight:bold; color:{color}; margin: 10px 0;">{res['direction']}</div>
            <div style="font-size:16px;">现价: <b>${price:.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""<div style="height:10px"></div>""", unsafe_allow_html=True)
        st.info(f"📊 波动率 (ATR): {atr:.2f}")
        st.caption("ATR 越高，建议仓位越小")

    # 2. 交易计划 (表格化)
    st.subheader("🎯 波段交易计划 (1-2天)")
    
    plan_cols = st.columns(3)
    plan_cols[0].metric("🛑 止损 (SL)", f"${sl:.2f}", delta="-1.5 ATR风险", delta_color="inverse")
    plan_cols[1].metric("💰 目标一 (TP1)", f"${tp1:.2f}", "均值回归(中轨)")
    plan_cols[2].metric("🚀 目标二 (TP2)", f"${tp2:.2f}", "波段极值(对侧轨)")

    with st.expander("查看决策依据"):
        for r in res['reasons']:
            st.write(r)

    # 3. 布林带+KDJ 图表
    st.subheader("📈 市场波动结构")
    
    # 上图：K线 + 布林带
    fig = go.Figure()
    
    # 布林带区域 (填充)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BBU_20_2.0'],
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BBL_20_2.0'],
        fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)',
        line=dict(width=0), showlegend=False, hoverinfo='skip',
        name='Bollinger Band'
    ))
    
    # K线
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Price'
    ))
    
    # 中轨
    fig.add_trace(go.Scatter(x=df.index, y=df['BBM_20_2.0'], line=dict(color='orange', width=1), name='BB Mid'))
    
    # 止盈止损参考线
    if res['type'] != 'neutral':
        fig.add_hline(y=tp1, line_dash="dot", line_color="green", annotation_text="TP1")
        fig.add_hline(y=sl, line_dash="dot", line_color="red", annotation_text="SL")

    fig.update_layout(
        template='plotly_dark', height=500, margin=dict(l=0,r=0,t=0,b=0),
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 提示
    st.markdown("""
    > **波段交易心法**：
    > 1. **不做中间段**：尽量在价格触及布林带上下轨时才出手。
    > 2. **时间止损**：如果开仓后 24小时 价格还在原地不动，说明波动逻辑失效，建议平仓离场。
    > 3. **盈亏比**：ZEC 波动大，如果 TP1 距离太近（盈亏比<1:1），这笔交易不值得做。
    """)

if __name__ == "__main__":
    main()
