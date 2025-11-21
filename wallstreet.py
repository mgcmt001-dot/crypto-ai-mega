import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime, timedelta
import time

# ==========================================
# 1. 配置与页面设置 (Configuration & UI Setup)
# ==========================================
st.set_page_config(page_title="Titan Alpha Quant Terminal", layout="wide", page_icon="📈")

# 自定义CSS，营造专业暗黑金融风
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .metric-card { background-color: #262730; border: 1px solid #414249; padding: 15px; border-radius: 5px; }
    h1, h2, h3 { color: #fafafa; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    .profit { color: #00cc96; }
    .loss { color: #ef553b; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心类定义 (Core Classes)
# ==========================================

class DataFetcher:
    """
    负责与OKX交易所通信，获取市场数据。
    """
    def __init__(self):
        # 初始化CCXT OKX实例
        # 注意：中国大陆地区可能需要配置 proxies 参数，例如 {'http': 'http://127.0.0.1:7890', ...}
        self.exchange = ccxt.okx({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'} # 默认为永续合约
        })

    def fetch_ohlcv(self, symbol, timeframe, limit=1000):
        try:
            # 获取K线数据
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            st.error(f"数据获取失败 ({timeframe}): {str(e)}")
            return pd.DataFrame()

    def get_fear_greed_index(self):
        """
        模拟恐慌贪婪指数获取 (因为CMC API需要Key，这里做模拟或抓取公开接口)
        在实际生产中，建议申请CMC API Key。
        """
        # 这里为了演示代码的完整性，我们模拟一个基于近期波动率的指数，
        # 或者你可以替换为 requests.get("https://api.alternative.me/fng/").json()
        try:
            import requests
            response = requests.get("https://api.alternative.me/fng/")
            data = response.json()
            value = int(data['data'][0]['value'])
            classification = data['data'][0]['value_classification']
            return value, classification
        except:
            return 50, "Neutral (Est)"

class QuantEngine:
    """
    量化分析引擎：计算指标、因子打分、回测。
    """
    def __init__(self, df):
        self.df = df.copy()
    
    def add_technical_indicators(self):
        if self.df.empty: return self.df
        
        # 1. 趋势因子 (Trend Factors)
        # MACD
        macd = MACD(close=self.df['close'])
        self.df['MACD'] = macd.macd()
        self.df['MACD_SIGNAL'] = macd.macd_signal()
        self.df['MACD_DIFF'] = macd.macd_diff()
        
        # EMA Ribbon (均线流)
        self.df['EMA_20'] = EMAIndicator(close=self.df['close'], window=20).ema_indicator()
        self.df['EMA_50'] = EMAIndicator(close=self.df['close'], window=50).ema_indicator()
        self.df['EMA_200'] = EMAIndicator(close=self.df['close'], window=200).ema_indicator()
        
        # ADX (趋势强度)
        adx = ADXIndicator(high=self.df['high'], low=self.df['low'], close=self.df['close'])
        self.df['ADX'] = adx.adx()

        # 2. 反转/动量因子 (Momentum/Reversal Factors)
        # RSI
        self.df['RSI'] = RSIIndicator(close=self.df['close']).rsi()
        
        # 3. 波动率因子 (Volatility Factors)
        # Bollinger Bands
        bb = BollingerBands(close=self.df['close'])
        self.df['BB_UPPER'] = bb.bollinger_hband()
        self.df['BB_LOWER'] = bb.bollinger_lband()
        self.df['BB_WIDTH'] = bb.bollinger_wband()
        
        # ATR (用于止损和仓位计算)
        self.df['ATR'] = AverageTrueRange(high=self.df['high'], low=self.df['low'], close=self.df['close']).average_true_range()
        
        self.df.dropna(inplace=True)
        return self.df

    def calculate_style_profile(self):
        """
        计算风格因子得分 (-10 到 10)
        """
        current = self.df.iloc[-1]
        
        # A. 趋势得分 (Trend Score)
        trend_score = 0
        if current['close'] > current['EMA_20'] > current['EMA_50']: trend_score += 4
        elif current['close'] < current['EMA_20'] < current['EMA_50']: trend_score -= 4
        if current['MACD_DIFF'] > 0: trend_score += 3
        else: trend_score -= 3
        if current['ADX'] > 25: trend_score *= 1.2 # 趋势增强
        
        # B. 反转得分 (Reversal Score)
        rev_score = 0
        if current['RSI'] > 70: rev_score -= 5 # 超买，看跌
        elif current['RSI'] < 30: rev_score += 5 # 超卖，看涨
        if current['close'] > current['BB_UPPER']: rev_score -= 3
        elif current['close'] < current['BB_LOWER']: rev_score += 3
        
        # C. 波动率状态 (Volatility State)
        vol_state = "High" if current['BB_WIDTH'] > self.df['BB_WIDTH'].rolling(100).mean().iloc[-1] else "Low"
        
        # 综合多空评分 (Total Signal Score)
        total_score = trend_score + rev_score
        
        # 归一化到 -10 到 10
        total_score = max(min(total_score, 10), -10)
        
        return {
            "trend": trend_score,
            "reversal": rev_score,
            "volatility": vol_state,
            "total_score": total_score
        }

    def vectorized_backtest(self, signal_threshold=3):
        """
        向量化回测：假设根据Total Score进行交易
        """
        df = self.df.copy()
        
        # 简化的信号生成逻辑
        # 趋势分 + 反转分 > 阈值做多，< -阈值做空
        # 注意：这里为了性能使用了简化的逻辑，而非完全复用 calculate_style_profile 的逐行逻辑
        
        # 向量化计算 Score
        df['trend_comp'] = np.where(df['close'] > df['EMA_50'], 1, -1)
        df['rsi_comp'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
        df['macd_comp'] = np.where(df['MACD_DIFF'] > 0, 1, -1)
        
        # 简单加权
        df['raw_signal'] = df['trend_comp'] * 2 + df['rsi_comp'] * 2 + df['macd_comp']
        
        # 生成持仓方向 (1: Long, -1: Short, 0: Flat)
        df['position'] = np.where(df['raw_signal'] >= signal_threshold, 1, 
                                  np.where(df['raw_signal'] <= -signal_threshold, -1, 0))
        
        # 将信号下移一格（避免未来函数，只能在下一根K线开盘执行）
        df['position'] = df['position'].shift(1)
        
        # 计算对数收益率
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # 计算策略收益
        df['strategy_ret'] = df['position'] * df['log_ret']
        
        # 扣除手续费 (假设双边万分之五)
        fee = 0.0005
        trades = df['position'].diff().abs()
        df['strategy_ret_net'] = df['strategy_ret'] - (trades * fee)
        
        # 累计净值
        df['cumulative_ret'] = df['strategy_ret_net'].cumsum().apply(np.exp)
        
        return df

# ==========================================
# 3. 页面逻辑 (Main Logic)
# ==========================================

def main():
    # --- Sidebar Controls ---
    st.sidebar.title("🏦 Titan Alpha 控制台")
    st.sidebar.markdown("---")
    
    symbol = st.sidebar.text_input("交易对 (Symbol)", value="BTC/USDT:USDT").upper()
    capital = st.sidebar.number_input("账户资金 (USDT)", value=10000, step=1000)
    risk_per_trade = st.sidebar.slider("单笔风险 (%)", 0.5, 5.0, 2.0) / 100
    leverage = st.sidebar.slider("目标杠杆 (Leverage)", 1, 20, 3)
    
    st.sidebar.markdown("### 分析周期设置")
    # 为了演示速度，默认抓取
    intervals = {'15m': '短线', '1h': '中线', '4h': '波段', '1d': '趋势'}
    selected_intervals = st.sidebar.multiselect("选择共振周期", list(intervals.keys()), default=['1h', '4h'])
    
    if st.sidebar.button("🚀 启动量化分析引擎"):
        with st.spinner('正在连接OKX节点... 计算因子暴露度... 运行蒙特卡洛模拟...'):
            
            fetcher = DataFetcher()
            
            # 1. 市场情绪面板
            fng_val, fng_class = fetcher.get_fear_greed_index()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("贪婪与恐惧指数", f"{fng_val}", fng_class)
            col2.metric("目标标的", symbol)
            col3.metric("账户总权益", f"${capital:,.2f}")
            
            # 2. 多周期数据抓取与分析
            analysis_results = {}
            latest_prices = {}
            
            for tf in selected_intervals:
                df = fetcher.fetch_ohlcv(symbol, tf, limit=1500) # 抓取足够数据用于回测
                if not df.empty:
                    engine = QuantEngine(df)
                    df_processed = engine.add_technical_indicators()
                    profile = engine.calculate_style_profile()
                    backtest_df = engine.vectorized_backtest()
                    
                    analysis_results[tf] = {
                        'data': df_processed,
                        'profile': profile,
                        'backtest': backtest_df
                    }
                    latest_prices[tf] = df['close'].iloc[-1]
                else:
                    st.error(f"无法获取 {tf} 数据，请检查网络或代码。")
                    return

            # 3. 核心仪表盘 (The Chief Analyst Dashboard)
            st.markdown("## 📊 深度市场剖面 (Market Profile)")
            
            # 选择主视角周期
            main_tf = selected_intervals[0]
            main_data = analysis_results[main_tf]['data']
            main_profile = analysis_results[main_tf]['profile']
            current_price = main_data['close'].iloc[-1]
            atr = main_data['ATR'].iloc[-1]
            
            # 显示得分
            score_col, advice_col = st.columns([1, 2])
            
            with score_col:
                score = main_profile['total_score']
                color = "green" if score > 0 else "red"
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style='text-align: center'>多空综合评分 ({main_tf})</h3>
                    <h1 style='text-align: center; color: {color}'>{score}/10</h1>
                    <p style='text-align: center'>趋势: {main_profile['trend']} | 反转: {main_profile['reversal']}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with advice_col:
                st.markdown("### 📢 首席分析师建议 (Analyst Verdict)")
                direction = "做多 (LONG)" if score >= 3 else ("做空 (SHORT)" if score <= -3 else "观望 (WAIT)")
                
                # 动态止损止盈计算
                stop_loss = current_price - (2 * atr) if score > 0 else current_price + (2 * atr)
                take_profit = current_price + (4 * atr) if score > 0 else current_price - (4 * atr)
                
                # 仓位计算 (基于ATR的波动率倒数模型)
                # 风险金额 = 总资金 * 单笔风险%
                # 仓位数量 = 风险金额 / |入场价 - 止损价|
                risk_amount = capital * risk_per_trade
                pos_size_coins = risk_amount / (2 * atr) # 2ATR为止损距离
                pos_value = pos_size_coins * current_price
                
                st.info(f"""
                **交易方向:** **{direction}**
                
                **关键点位:**
                - 🟢 当前价格: {current_price:.4f}
                - 🛑 建议止损 (SL): {stop_loss:.4f} (2.0 ATR)
                - 🎯 建议止盈 (TP): {take_profit:.4f} (4.0 ATR)
                - ⚖️ 盈亏比: 1:2
                
                **资金管理 (Kelly/Volatility Sizing):**
                - 建议仓位价值: ${pos_value:.2f} (约 {pos_size_coins:.4f} 币)
                - 实际杠杆率: {min(pos_value/capital, leverage):.2f}x
                """)

            # 4. 可视化图表
            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["🕯️ K线与技术分析", "📈 历史净值回测", "🎲 盈亏分布直方图"])
            
            with tab1:
                # 使用 Plotly 绘制专业K线
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, row_heights=[0.7, 0.3])

                # Candlestick
                fig.add_trace(go.Candlestick(x=main_data.index,
                                open=main_data['open'], high=main_data['high'],
                                low=main_data['low'], close=main_data['close'],
                                name='OHLC'), row=1, col=1)
                
                # EMA
                fig.add_trace(go.Scatter(x=main_data.index, y=main_data['EMA_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=main_data.index, y=main_data['EMA_50'], line=dict(color='blue', width=1), name='EMA 50'), row=1, col=1)
                
                # BB
                fig.add_trace(go.Scatter(x=main_data.index, y=main_data['BB_UPPER'], line=dict(color='gray', width=1, dash='dash'), name='BB Upper'), row=1, col=1)
                fig.add_trace(go.Scatter(x=main_data.index, y=main_data['BB_LOWER'], line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name='BB Lower'), row=1, col=1)

                # MACD
                fig.add_trace(go.Bar(x=main_data.index, y=main_data['MACD_DIFF'], name='MACD Hist', marker_color=np.where(main_data['MACD_DIFF']<0, 'red', 'green')), row=2, col=1)
                fig.add_trace(go.Scatter(x=main_data.index, y=main_data['MACD'], name='MACD Line'), row=2, col=1)
                fig.add_trace(go.Scatter(x=main_data.index, y=main_data['MACD_SIGNAL'], name='Signal Line'), row=2, col=1)

                fig.update_layout(title=f"{symbol} - {main_tf} Technical View", height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # 净值曲线
                backtest_data = analysis_results[main_tf]['backtest']
                # 截取最近3个月 (假设数据足够)
                recent_backtest = backtest_data[backtest_data.index > (datetime.now() - timedelta(days=90))]
                
                if not recent_backtest.empty:
                    fig_equity = px.line(recent_backtest, x=recent_backtest.index, y='cumulative_ret', 
                                         title='如果你过去3个月机械执行此模型的净值曲线 (Base=1)',
                                         labels={'cumulative_ret': '净值', 'timestamp': '日期'})
                    fig_equity.update_layout(template="plotly_dark")
                    
                    # 计算最大回撤
                    roll_max = recent_backtest['cumulative_ret'].cummax()
                    drawdown = recent_backtest['cumulative_ret'] / roll_max - 1.0
                    max_dd = drawdown.min()
                    total_ret = recent_backtest['cumulative_ret'].iloc[-1] - 1
                    
                    c1, c2 = st.columns(2)
                    c1.metric("区间总回报", f"{total_ret*100:.2f}%")
                    c2.metric("最大回撤 (Max Drawdown)", f"{max_dd*100:.2f}%")
                    
                    st.plotly_chart(fig_equity, use_container_width=True)
                else:
                    st.warning("数据不足，无法显示3个月回测。")

            with tab3:
                # 盈亏分布
                if not recent_backtest.empty:
                    trade_returns = recent_backtest[recent_backtest['position'].diff() != 0]['strategy_ret_net']
                    trade_returns = trade_returns[trade_returns != 0]
                    
                    fig_hist = px.histogram(trade_returns, nbins=50, 
                                            title="最近 N 次信号盈亏分布直方图",
                                            labels={'value': '单笔收益率'},
                                            color_discrete_sequence=['#636EFA'])
                    fig_hist.update_layout(template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    win_rate = len(trade_returns[trade_returns > 0]) / len(trade_returns) if len(trade_returns) > 0 else 0
                    st.markdown(f"#### 历史胜率: {win_rate*100:.2f}% (基于最近 {len(trade_returns)} 次信号)")

            # 5. 风格因子雷达图 (Style Radar)
            st.markdown("### 🕸️ 因子暴露分析 (Factor Exposure)")
            radar_data = pd.DataFrame(dict(
                r=[
                    abs(main_profile['trend']), 
                    abs(main_profile['reversal']), 
                    10 if main_profile['volatility'] == 'High' else 3,
                    abs(fng_val - 50) / 5 # 情绪偏离度
                ],
                theta=['Trend Strength', 'Reversal Potential', 'Volatility', 'Sentiment Divergence']
            ))
            fig_radar = px.line_polar(radar_data, r='r', theta='theta', line_close=True, range_r=[0,10])
            fig_radar.update_layout(template="plotly_dark")
            fig_radar.update_traces(fill='toself')
            st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.info("👈 请在左侧设置参数并点击启动按钮。")

if __name__ == "__main__":
    main()
