import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta

# --- 1. 配置 ---
st.set_page_config(page_title="Crypto Commander V7.1", layout="wide")
PROXY = "http://127.0.0.1:7890"

# --- 2. CSS 样式优化 (解决显示不全问题) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* 标题渐变 */
    .main-title {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        padding-bottom: 10px;
    }
    
    /* --- 自定义价格网格 (取代 st.metric) --- */
    .price-grid {
        display: grid;
        grid-template-columns: 1fr 1fr; /* 左右两列 */
        gap: 10px;
        margin-bottom: 15px;
        background: rgba(0,0,0,0.2);
        padding: 10px;
        border-radius: 8px;
    }
    
    .price-item {
        display: flex;
        flex-direction: column;
    }
    
    .price-label {
        font-size: 12px;
        color: #aaa;
        margin-bottom: 4px;
    }
    
    .price-val {
        font-family: 'JetBrains Mono', monospace; /* 等宽字体 */
        font-weight: 700;
        font-size: 1.1rem; /* 稍微调小字号，防止溢出 */
        word-break: break-all; /* 强制换行，杜绝 ... */
    }
    
    .tp-color { color: #00e676; }
    .sl-color { color: #ff5252; }
    
    /* 投资摘要框 */
    .memo-box {
        background-color: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #888;
        padding: 12px;
        border-radius: 0 5px 5px 0;
        font-size: 13px;
        color: #ddd;
        line-height: 1.5;
    }
    
    /* 信号灯 */
    .sig-long { color: #00e676; font-weight: 800; font-size: 18px; letter-spacing: 1px; }
    .sig-short { color: #ff1744; font-weight: 800; font-size: 18px; letter-spacing: 1px; }
    .sig-wait { color: #ff9100; font-weight: 800; font-size: 18px; letter-spacing: 1px; }
    
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">Crypto Commander Pro</div>', unsafe_allow_html=True)

# --- 3. 连接 ---
exchange = ccxt.binance({
    'proxies': {'http': PROXY, 'https': PROXY},
    'timeout': 30000, 'enableRateLimit': True,
})

# --- 4. 辅助函数 ---
def fmt_price(price):
    if price > 1000: return f"{price:,.2f}"
    elif price > 1: return f"{price:,.4f}"
    else: return f"{price:.6f}"

# --- 5. 数据获取 ---
def get_data(symbol, tf):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=500)
        if not bars or len(bars) < 200: return pd.DataFrame()
        df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        
        df['EMA20'] = ta.ema(df['close'], length=20)
        df['EMA50'] = ta.ema(df['close'], length=50)
        df['MA200'] = ta.sma(df['close'], length=200)
        
        macd = ta.macd(df['close'])
        if macd is not None:
            df['MACD'] = macd.iloc[:, 0]
            df['MACD_SIG'] = macd.iloc[:, 1]
            
        df['RSI'] = ta.rsi(df['close'], length=14)
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            df['BB_U'] = bb.iloc[:, 2]
            df['BB_L'] = bb.iloc[:, 0]
            df['BB_W'] = (df['BB_U'] - df['BB_L']) / df['EMA20']
        
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        return df
    except:
        return pd.DataFrame()

# --- 6. 智能分析 ---
def analyze(df, label, coin_name):
    if df.empty: return None
    c = df.iloc[-1]
    price = c['close']
    ma_val = c['MA200'] if not pd.isna(c['MA200']) else c['EMA50']
    
    # 评分
    score = 0
    if price > c['EMA20'] > c['EMA50']: score += 2
    elif price < c['EMA20'] < c['EMA50']: score -= 2
    elif price > ma_val: score += 1
    else: score -= 1
    if c['MACD'] > c['MACD_SIG']: score += 1
    else: score -= 1
    
    # 结果
    res = {}
    if score >= 2:
        res['sig_cls'] = "sig-long"
        res['sig_txt'] = "🟢 LONG (做多)"
        res['sl'] = price - 2.5 * c['ATR']
        res['tp'] = max(c['BB_U'], price + 3.0 * c['ATR'])
        act = "低吸做多"
    elif score <= -2:
        res['sig_cls'] = "sig-short"
        res['sig_txt'] = "🔴 SHORT (做空)"
        res['sl'] = price + 2.5 * c['ATR']
        res['tp'] = min(c['BB_L'], price - 3.0 * c['ATR'])
        act = "反弹做空"
    else:
        res['sig_cls'] = "sig-wait"
        res['sig_txt'] = "⚪ WAIT (观望)"
        res['sl'] = price * 0.98
        res['tp'] = price * 1.02
        act = "空仓等待"

    # 生成自然语言摘要
    res['memo'] = (
        f"【趋势】{label}周期下，价格位于{'多头' if score>0 else '空头'}区域。"
        f"MACD指标{'金叉增强' if c['MACD']>c['MACD_SIG'] else '死叉修正'}。"
        f"【策略】建议<b>{act}</b>。上方压力关注 {fmt_price(res['tp'])}，"
        f"下方防守位设在 {fmt_price(res['sl'])}。"
    )
    return res

# --- 7. UI 布局 ---
with st.sidebar:
    st.header("🎮 控制台")
    coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'DOGE/USDT', 'PEPE/USDT', 'WIF/USDT']
    coin = st.selectbox("选择资产", coins)
    if st.button("🔄 刷新数据"): st.rerun()

st.subheader("🌍 宏观看板")
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    try:
        g = exchange.fetch_ticker('PAXG/USDT')
        b = exchange.fetch_ticker('BTC/USDT')
        e = exchange.fetch_ticker('ETH/USDT')
        c1.metric("🥇 Gold", f"${g['last']:,.2f}", f"{g['percentage']:.2f}%")
        c2.metric("🚀 Bitcoin", f"${b['last']:,.2f}", f"{b['percentage']:.2f}%")
        c3.metric("💎 Ethereum", f"${e['last']:,.2f}", f"{e['percentage']:.2f}%")
    except:
        st.warning("连接中...")

st.divider()

# --- 8. 核心策略展示 ---
st.subheader(f"📊 {coin} 深度策略报告")

cols = st.columns(3)
periods = [("短线 (15m)", "15m"), ("中线 (4h)", "4h"), ("长线 (1d)", "1d")]

for i, (title, tf) in enumerate(periods):
    with cols[i]:
        df = get_data(coin, tf)
        res = analyze(df, title, coin)
        
        if res:
            with st.container(border=True):
                # 1. 信号标题
                st.markdown(f"<div style='font-size:14px; color:#888;'>{title}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='{res['sig_cls']}'>{res['sig_txt']}</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # 2. 自定义 HTML 价格网格 (完美解决数字过长显示不全的问题)
                st.markdown(f"""
                <div class="price-grid">
                    <div class="price-item">
                        <span class="price-label">🎯 目标止盈 (TP)</span>
                        <span class="price-val tp-color">{fmt_price(res['tp'])}</span>
                    </div>
                    <div class="price-item">
                        <span class="price-label">🛡️ 止损保护 (SL)</span>
                        <span class="price-val sl-color">{fmt_price(res['sl'])}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 3. 投资摘要
                st.markdown(f"""
                <div class="memo-box">
                    {res['memo']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("数据加载中")
