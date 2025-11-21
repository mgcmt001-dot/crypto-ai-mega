import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime, timedelta
import time

# ==========================================
# 1. 配置与页面设置 (Configuration & UI Setup)
# ==========================================
st.set_page_config(page_title="Titan Alpha Pro | Quant Terminal", layout="wide", page_icon="🐺")

# 专业金融终端样式
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .metric-card { background-color: #1e222d; border: 1px solid #2e3346; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    h1, h2, h3, h4 { color: #e0e0e0; font-family: 'Roboto', sans-serif; }
    .stSelectbox > div > div { background-color: #262730; color: white; }
    .stButton>button { width: 100%; background-color: #2962ff; color: white; border-radius: 5px; font-weight: bold; border: none; padding: 0.5rem; }
    .stButton>button:hover { background-color: #0039cb; }
    .highlight-bull { color: #00cc96; font-weight: bold; }
    .highlight-bear { color: #ef553b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心类定义 (Core Classes)
# ==========================================

class DataFetcher:
    def __init__(self):
        self.exchange = ccxt.okx({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })

    @st.cache_data(ttl=3600) # 缓存1小时，避免频繁请求
    def get_available_symbols(_self):
        """获取所有USDT永续合约交易对"""
        try:
            markets = _self.exchange.load_markets()
            # 筛选 USDT 结算的永续合约 (SWAP)
            symbols = [symbol for symbol in markets.keys() if 'USDT' in symbol and ':' in symbol]
            symbols.sort()
            return symbols
        except Exception as e:
            return ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]

    def fetch_ohlcv(self, symbol, timeframe, limit=500):
        """获取K线数据"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 基础清洗：转为float
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].astype(float)
            return df
        except Exception as e:
            # 记录错误但不中断程序，返回空DF
            print(f"Error fetching {timeframe}: {e}")
            return pd.DataFrame()

    def get_fear_greed_index(self):
        # 模拟数据，实际可用API替换
        return 65, "Greed"

class QuantEngine:
    def __init__(self, df):
        self.df = df.copy()
    
    def check_data_quality(self):
        """检查数据长度是否足够计算指标"""
        return len(self.df) > 200

    def add_technical_indicators(self):
        if self.df.empty: return self.df
        
        try:
            close = self.df['close']
            high = self.df['high']
            low = self.df['low']

            # 1. Trend
            self.df['MACD_DIFF'] = MACD(close).macd_diff()
            self.df['EMA_20'] = EMAIndicator(close, window=20).ema_indicator()
            self.df['EMA_50'] = EMAIndicator(close, window=50).ema_indicator()
            self.df['ADX'] = ADXIndicator(high, low, close).adx()

            # 2. Momentum
            self.df['RSI'] = RSIIndicator(close).rsi()

            # 3. Volatility
            bb = BollingerBands(close)
            self.df['BB_UPPER'] = bb.bollinger_hband()
            self.df['BB_LOWER'] = bb.bollinger_lband()
            self.df['BB_WIDTH'] = bb.bollinger_wband()
            self.df['ATR'] = AverageTrueRange(high, low, close).average_true_range()

            # 移除计算产生的NaN值 (前几行)
            self.df.dropna(inplace=True)
            return self.df
        except Exception as e:
            st.error(f"指标计算错误: {str(e)}")
            return pd.DataFrame()

    def calculate_style_profile(self):
        if self.df.empty: return None
        
        current = self.df.iloc[-1]
        
        # 防御性编程：检查字段是否存在
        required_cols = ['close', 'EMA_20', 'EMA_50', 'MACD_DIFF', 'ADX', 'RSI', 'BB_UPPER', 'BB_LOWER', 'BB_WIDTH']
        for col in required_cols:
            if col not in current.index:
                return None

        # A. 趋势得分
        trend_score = 0
        if current['close'] > current['EMA_20'] > current['EMA_50']: trend_score += 4
        elif current['close'] < current['EMA_20'] < current['EMA_50']: trend_score -= 4
        
        if current['MACD_DIFF'] > 0: trend_score += 2
        else: trend_score -= 2
        
        # B. 反转得分
        rev_score = 0
        if current['RSI'] > 75: rev_score -= 4
        elif current['RSI'] < 25: rev_score += 4
        
        if current['close'] > current['BB_UPPER']: rev_score -= 3
        elif current['close'] < current['BB_LOWER']: rev_score += 3
        
        # C. 波动率
        vol_avg = self.df['BB_WIDTH'].rolling(50).mean().iloc[-1]
        vol_state = "High" if current['BB_WIDTH'] > vol_avg else "Low"
        
        total = max(min(trend_score + rev_score, 10), -10)
        
        return {
            "trend": trend_score,
            "reversal": rev_score,
            "volatility": vol_state,
            "total_score": total
        }

    def vectorized_backtest(self):
        if self.df.empty: return pd.DataFrame()
        
        df = self.df.copy()
        # 简单策略用于生成盈亏分布
        df['signal'] = np.where(df['close'] > df['EMA_20'], 1, -1) # 简化为均线策略演示
        df['return'] = np.log(df['close'] / df['close'].shift(1))
        df['strategy_ret'] = df['signal'].shift(1) * df['return']
        df['cum_ret'] = df['strategy_ret'].cumsum().apply(np.exp)
        
        # 标记交易
        df['trade_entry'] = df['signal'].diff().fillna(0) != 0
        return df

# ==========================================
# 3. 页面逻辑 (Main Logic)
# ==========================================

def main():
    # 初始化加载
    fetcher = DataFetcher()
    
    # --- Sidebar ---
    st.sidebar.title("🏦 Titan Alpha V2.0")
    st.sidebar.caption("Wall Street Grade Crypto Assistant")
    
    # 1. 获取交易对列表
    with st.spinner("正在连接 OKX 交易所获取最新合约列表..."):
        available_symbols = fetcher.get_available_symbols()
    
    # 2. 交易对选择器 (Selectbox)
    symbol = st.sidebar.selectbox("选择交易标的 (Symbol)", available_symbols, index=available_symbols.index("BTC/USDT:USDT") if "BTC/USDT:USDT" in available_symbols else 0)
    
    st.sidebar.markdown("---")
    capital = st.sidebar.number_input("总资金 (USDT)", value=10000)
    risk_per_trade = st.sidebar.slider("单笔风险 (Risk %)", 0.5, 5.0, 2.0) / 100
    
    timeframes = ['15m', '1h', '4h', '1d']
    selected_tfs = st.sidebar.multiselect("分析周期", timeframes, default=['1h', '4h'])
    
    run_btn = st.sidebar.button("🚀 执行深度量化分析")

    # --- Main Content ---
    if run_btn:
        if not selected_tfs:
            st.error("请至少选择一个时间周期！")
            return

        # 进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        analysis_results = {}
        
        # 循环抓取数据
        for i, tf in enumerate(selected_tfs):
            status_text.text(f"正在抓取 OKX {tf} 数据并进行因子计算...")
            df = fetcher.fetch_ohlcv(symbol, tf)
            
            if not df.empty:
                engine = QuantEngine(df)
                if engine.check_data_quality():
                    df_calc = engine.add_technical_indicators()
                    if not df_calc.empty:
                        profile = engine.calculate_style_profile()
                        backtest = engine.vectorized_backtest()
                        
                        if profile is not None:
                            analysis_results[tf] = {
                                'data': df_calc,
                                'profile': profile,
                                'backtest': backtest
                            }
            progress_bar.progress((i + 1) / len(selected_tfs))
        
        status_text.empty()
        progress_bar.empty()

        # 检查是否有有效结果
        if not analysis_results:
            st.error("❌ 所有选定周期的数据抓取或计算均失败。请检查网络连接或更换交易对。")
            return
            
        # 动态选择主周期 (取第一个成功的周期)
        main_tf = list(analysis_results.keys())[0]
        main_data = analysis_results[main_tf]['data']
        main_profile = analysis_results[main_tf]['profile']
        
        # 确保最新价格存在
        current_price = main_data['close'].iloc[-1]
        atr_value = main_data['ATR'].iloc[-1]

        # --- 仪表盘显示 ---
        st.markdown(f"## 📊 {symbol} 量化分析报告")
        
        # 顶栏指标
        col1, col2, col3, col4 = st.columns(4)
        f_val, f_state = fetcher.get_fear_greed_index()
        
        col1.metric("主分析周期", main_tf)
        col2.metric("当前价格", f"${current_price:,.4f}")
        col3.metric("ATR (波动率)", f"{atr_value:.4f}")
        col4.metric("市场情绪", f"{f_val} ({f_state})")

        # 核心信号卡片
        st.markdown("### 🧠 首席分析师决策模型")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            score = main_profile['total_score']
            score_color = "#00cc96" if score > 0 else "#ef553b"
            st.markdown(f"""
            <div class="metric-card" style="text-align:center">
                <h4 style="margin:0">Alpha 综合得分</h4>
                <h1 style="font-size: 4em; color: {score_color}; margin:0">{score}</h1>
                <p style="color: #888">区间: [-10, +10]</p>
                <hr style="border-color: #333">
                <div style="display:flex; justify-content:space-between">
                    <span>趋势: {main_profile['trend']}</span>
                    <span>反转: {main_profile['reversal']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            # 交易计划生成
            action = "做多 (LONG)" if score >= 3 else ("做空 (SHORT)" if score <= -3 else "观望 (WAIT)")
            action_color = "highlight-bull" if score >= 3 else ("highlight-bear" if score <= -3 else "")
            
            # 止损计算
            sl_dist = 2.0 * atr_value
            tp_dist = 4.0 * atr_value # 盈亏比 1:2
            
            stop_loss = current_price - sl_dist if score > 0 else current_price + sl_dist
            take_profit = current_price + tp_dist if score > 0 else current_price - tp_dist
            
            # 仓位计算
            risk_amount = capital * risk_per_trade
            # 避免除以零
            if sl_dist == 0: sl_dist = current_price * 0.01 
            
            position_size_coin = risk_amount / sl_dist
            position_value = position_size_coin * current_price
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>📑 交易执行计划 (Execution Plan)</h4>
                <p>建议方向: <span class="{action_color}" style="font-size:1.2em">{action}</span></p>
                <ul>
                    <li><strong>入场参考:</strong> {current_price:.4f}</li>
                    <li><strong>止损位 (SL):</strong> {stop_loss:.4f} <span style="color:#666">(2.0 ATR 动态止损)</span></li>
                    <li><strong>止盈位 (TP):</strong> {take_profit:.4f} <span style="color:#666">(盈亏比 1:2)</span></li>
                </ul>
                <hr style="border-color: #333">
                <h4>💰 资金管理 (Position Sizing)</h4>
                <ul>
                    <li>承受风险金额: ${risk_amount:.2f} ({risk_per_trade*100}%)</li>
                    <li><strong>建议开仓数量:</strong> {position_size_coin:.4f} 币</li>
                    <li>合约名义价值: ${position_value:.2f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # --- 图表区域 ---
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["🕯️ K线透视", "📈 净值回测", "🌊 因子雷达"])
        
        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            
            # K线
            fig.add_trace(go.Candlestick(x=main_data.index,
                            open=main_data['open'], high=main_data['high'],
                            low=main_data['low'], close=main_data['close'],
                            name='Price'), row=1, col=1)
            
            # 布林带
            fig.add_trace(go.Scatter(x=main_data.index, y=main_data['BB_UPPER'], line=dict(color='rgba(255, 255, 255, 0.3)', width=1), name='BB Upper'), row=1, col=1)
            fig.add_trace(go.Scatter(x=main_data.index, y=main_data['BB_LOWER'], line=dict(color='rgba(255, 255, 255, 0.3)', width=1), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', name='BB Lower'), row=1, col=1)
            
            # MACD
            fig.add_trace(go.Bar(x=main_data.index, y=main_data['MACD_DIFF'], marker_color=np.where(main_data['MACD_DIFF']<0, '#ef553b', '#00cc96'), name='MACD Hist'), row=2, col=1)
            
            fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            bt_df = analysis_results[main_tf]['backtest']
            if not bt_df.empty:
                # 绘制净值曲线
                fig_eq = px.line(bt_df, x=bt_df.index, y='cum_ret', title=f"{main_tf} 周期历史净值走势", color_discrete_sequence=['#2962ff'])
                fig_eq.update_layout(template="plotly_dark")
                st.plotly_chart(fig_eq, use_container_width=True)
                
                # 盈亏直方图
                rets = bt_df[bt_df['trade_entry']]['strategy_ret']
                if not rets.empty:
                    fig_hist = px.histogram(rets, nbins=30, title="盈亏分布 (PnL Distribution)", color_discrete_sequence=['#00cc96'])
                    fig_hist.update_layout(template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("回测数据不足。")

        with tab3:
            # 雷达图
            categories = ['Trend', 'Reversal', 'Volatility', 'Volume']
            # 归一化数据用于展示
            t_val = abs(main_profile['trend']) / 4 * 5
            r_val = abs(main_profile['reversal']) / 4 * 5
            v_val = 8 if main_profile['volatility'] == 'High' else 3
            
            fig_radar = px.line_polar(r=[t_val, r_val, v_val, 5], theta=categories, line_close=True, range_r=[0, 10])
            fig_radar.update_traces(fill='toself', line_color='#ff0055')
            fig_radar.update_layout(template="plotly_dark", title="市场风格因子剖面")
            st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("👈 首席分析师已就位。请在左侧选择交易对并点击【执行深度量化分析】。")

if __name__ == "__main__":
    main()
