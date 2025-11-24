import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np

# --- 1. 全局配置 ---
st.set_page_config(page_title="Crypto Commander V8 (Full)", layout="wide", initial_sidebar_state="expanded")
PROXY = "http://127.0.0.1:7890"

# --- 2. CSS 样式增强 (高密度数据风格) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* 标题 */
    .main-title {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 2.8rem;
    }
    
    /* 策略卡片 */
    .strategy-card {
        border: 1px solid #444;
        border-radius: 8px;
        padding: 15px;
        background-color: #1e1e1e;
        height: 100%;
    }
    
    /* 价格网格 */
    .price-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        background: rgba(255,255,255,0.03);
        padding: 10px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .price-val {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .tp { color: #00e676; }
    .sl { color: #ff5252; }
    
    /* 摘要框 */
    .memo {
        font-size: 13px;
        color: #aaa;
        line-height: 1.5;
        border-top: 1px solid #333;
        padding-top: 10px;
        margin-top: 10px;
    }

    /* 深度表格样式 */
    .depth-table {
        width: 100%;
        font-size: 12px;
        border-collapse: collapse;
    }
    .depth-row { border-bottom: 1px solid #333; }
    .depth-ask { color: #ff5252; text-align: right; }
    .depth-bid { color: #00e676; text-align: left; }
    .depth-header { color: #888; font-weight: bold; padding-bottom: 5px;}
    
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">Crypto Commander V8</div>', unsafe_allow_html=True)

# --- 3. 交易所连接 ---
exchange = ccxt.binance({
    'proxies': {'http': PROXY, 'https': PROXY},
    'timeout': 30000, 'enableRateLimit': True,
})

# --- 4. 核心数据函数 ---
def fmt_price(price):
    if price > 1000: return f"{price:,.2f}"
    elif price > 1: return f"{price:,.4f}"
    else: return f"{price:.6f}"

def get_data(symbol, tf, limit=200):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        if not bars: return pd.DataFrame()
        df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        
        # 基础指标
        df['EMA20'] = ta.ema(df['close'], length=20)
        df['EMA50'] = ta.ema(df['close'], length=50)
        df['MA200'] = ta.sma(df['close'], length=200)
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'])
        if macd is not None:
            df['MACD'] = macd.iloc[:, 0]
            df['MACD_SIG'] = macd.iloc[:, 1]
            
        # Bollinger
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            df['BB_U'] = bb.iloc[:, 2]
            df['BB_L'] = bb.iloc[:, 0]
            df['BB_W'] = (df['BB_U'] - df['BB_L']) / df['EMA20']
            
        return df
    except:
        return pd.DataFrame()

# 获取实时盘口
def get_order_book(symbol):
    try:
        book = exchange.fetch_order_book(symbol, limit=10)
        return book
    except:
        return None

# 计算枢轴点 (Pivot Points)
def calc_pivots(df):
    last = df.iloc[-1]
    high = last['high']
    low = last['low']
    close = last['close']
    
    p = (high + low + close) / 3
    r1 = 2*p - low
    s1 = 2*p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)
    
    return {"R3": r3, "R2": r2, "R1": r1, "P": p, "S1": s1, "S2": s2, "S3": s3}

# 策略分析逻辑
def analyze_strategy(df, label):
    if df.empty: return None
    c = df.iloc[-1]
    price = c['close']
    ma_val = c['MA200'] if not pd.isna(c['MA200']) else c['EMA50']
    
    score = 0
    if price > c['EMA20'] > c['EMA50']: score += 2
    elif price < c['EMA20'] < c['EMA50']: score -= 2
    elif price > ma_val: score += 1
    else: score -= 1
    if c['MACD'] > c['MACD_SIG']: score += 1
    else: score -= 1
    
    res = {}
    if score >= 2:
        res['sig'] = "🟢 LONG"
        res['sl'] = price - 2.5 * c['ATR']
        res['tp'] = max(c['BB_U'], price + 3.0 * c['ATR'])
        res['txt'] = "趋势走强，建议低吸"
    elif score <= -2:
        res['sig'] = "🔴 SHORT"
        res['sl'] = price + 2.5 * c['ATR']
        res['tp'] = min(c['BB_L'], price - 3.0 * c['ATR'])
        res['txt'] = "空头排列，建议高空"
    else:
        res['sig'] = "⚪ WAIT"
        res['sl'] = price * 0.98
        res['tp'] = price * 1.02
        res['txt'] = "震荡行情，建议观望"
        
    res['memo'] = f"MACD{'金叉' if c['MACD']>c['MACD_SIG'] else '死叉'}，RSI为{c['RSI']:.1f}。{res['txt']}。"
    return res

# --- 5. 侧边栏 ---
with st.sidebar:
    st.header("🎮 驾驶舱")
    coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'DOGE/USDT', 'PEPE/USDT', 'WIF/USDT', 'ORDI/USDT']
    coin = st.selectbox("标的资产", coins)
    if st.button("🔄 刷新全盘"): st.rerun()

# --- 6. 宏观条 ---
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    try:
        g = exchange.fetch_ticker('PAXG/USDT')
        b = exchange.fetch_ticker('BTC/USDT')
        e = exchange.fetch_ticker('ETH/USDT')
        c1.metric("🥇 Gold", f"${g['last']:,.2f}", f"{g['percentage']:.2f}%")
        c2.metric("🚀 BTC", f"${b['last']:,.2f}", f"{b['percentage']:.2f}%")
        c3.metric("💎 ETH", f"${e['last']:,.2f}", f"{e['percentage']:.2f}%")
        # 计算波动率作为第四指标
        df_d = get_data(coin, '1d')
        volatility = (df_d.iloc[-1]['high'] - df_d.iloc[-1]['low']) / df_d.iloc[-1]['low'] * 100
        c4.metric("🌊 Volatility (Day)", f"{volatility:.2f}%", "日内波幅")
    except:
        st.warning("数据连接中...")

# --- 7. 核心策略三连 ---
st.subheader(f"📊 {coin} 核心策略")
cols = st.columns(3)
periods = [("短线 (15m)", "15m"), ("中线 (4h)", "4h"), ("长线 (1d)", "1d")]
cached_dfs = {} # 缓存数据给后面用

for i, (title, tf) in enumerate(periods):
    with cols[i]:
        df = get_data(coin, tf)
        cached_dfs[tf] = df
        res = analyze_strategy(df, title)
        
        if res:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.markdown(f"<h3 style='margin:0;'>{res['sig']}</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="price-grid">
                    <div><div style="font-size:12px; color:#aaa">🎯 TARGET</div><div class="price-val tp">{fmt_price(res['tp'])}</div></div>
                    <div><div style="font-size:12px; color:#aaa">🛡️ STOP</div><div class="price-val sl">{fmt_price(res['sl'])}</div></div>
                </div>
                <div class="memo">{res['memo']}</div>
                """, unsafe_allow_html=True)
        else:
            st.error("No Data")

# --- 8. 新增：深度数据面板 (填补空白) ---
st.markdown("---")
st.subheader("🧠 深度数据透视 (Deep Dive)")

# 使用 Tabs 分页，增加内容密度但不乱
tab1, tab2, tab3 = st.tabs(["🔑 关键支撑压力 (Pivot Points)", "📉 实时买卖盘口 (Order Book)", "📟 技术指标矩阵 (Indicators)"])

# Tab 1: 智能支撑压力位
with tab1:
    st.caption("基于日线(Daily) High/Low/Close 计算的斐波那契/经典阻力支撑位，适合挂单参考。")
    if not cached_dfs['1d'].empty:
        pivots = calc_pivots(cached_dfs['1d'])
        col_r, col_p, col_s = st.columns(3)
        
        with col_r:
            st.markdown("#### 🔴 阻力位 (Resistance)")
            st.metric("R3 (强阻力)", fmt_price(pivots['R3']))
            st.metric("R2 (中阻力)", fmt_price(pivots['R2']))
            st.metric("R1 (弱阻力)", fmt_price(pivots['R1']))
            
        with col_p:
            st.markdown("#### ⚪ 枢轴点 (Pivot)")
            st.metric("Pivot Point", fmt_price(pivots['P']))
            st.info("价格在 Pivot 之上偏多，之下偏空")
            
        with col_s:
            st.markdown("#### 🟢 支撑位 (Support)")
            st.metric("S1 (弱支撑)", fmt_price(pivots['S1']))
            st.metric("S2 (中支撑)", fmt_price(pivots['S2']))
            st.metric("S3 (强支撑)", fmt_price(pivots['S3']))
    else:
        st.warning("需要加载日线数据")

# Tab 2: 实时买卖盘口
with tab2:
    st.caption("实时抓取交易所前10档挂单，判断短期多空抛压。")
    book = get_order_book(coin)
    if book:
        col_bid, col_ask = st.columns(2)
        
        with col_bid:
            st.markdown("**🟢 买盘 (Bids) - 支撑**")
            # 简易表格渲染
            bids_df = pd.DataFrame(book['bids'], columns=['Price', 'Amount'])
            bids_df['Price'] = bids_df['Price'].apply(fmt_price)
            st.dataframe(bids_df, use_container_width=True, height=300, hide_index=True)
            
        with col_ask:
            st.markdown("**🔴 卖盘 (Asks) - 压力**")
            asks_df = pd.DataFrame(book['asks'], columns=['Price', 'Amount'])
            asks_df['Price'] = asks_df['Price'].apply(fmt_price)
            st.dataframe(asks_df, use_container_width=True, height=300, hide_index=True)
    else:
        st.warning("盘口数据获取失败")

# Tab 3: 指标矩阵
with tab3:
    st.caption("多周期核心指标读数，像飞行员一样监控仪表盘。")
    # 构建一个汇总表格
    metrics_data = []
    for tf in ['15m', '4h', '1d']:
        d = cached_dfs.get(tf)
        if d is not None and not d.empty:
            c = d.iloc[-1]
            metrics_data.append({
                "周期": tf,
                "RSI (14)": f"{c['RSI']:.1f}",
                "MACD 状态": "🟢 金叉" if c['MACD'] > c['MACD_SIG'] else "🔴 死叉",
                "布林带位置": "上轨" if c['close'] > c['BB_U'] else ("下轨" if c['close'] < c['BB_L'] else "中轨"),
                "EMA趋势": "看涨" if c['close'] > c['EMA20'] else "看跌"
            })
    
    m_df = pd.DataFrame(metrics_data)
    st.table(m_df)

st.markdown("---")
st.caption("Crypto Commander V8.0 | System Active | All calculations are based on real-time Binance public data.")
