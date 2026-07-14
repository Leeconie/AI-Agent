"""
金融数据分析模块
支持数据获取、基础分析、技术指标计算和数据可视化
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import akshare as ak
from typing import Dict, Any, List, Optional
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
import logging
from tenacity import retry, wait_random, stop_after_attempt

# 忽略警告
warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class FinancialAnalyzer:
    def __init__(self):
        self.data = None
        self.symbol = None

    def fetch_data(self, symbol: str, period: str = "2y", interval: str = "1d", start_date: str = "", end_date: str = "") -> pd.DataFrame:
        try:
            self.symbol = symbol
            logging.info(f"开始获取数据，股票代码: {symbol}, 周期: {period}, 间隔: {interval}")
            # 解析市场后缀
            symbol = symbol.strip()
            if not symbol.endswith((".HK", ".SZ", ".SS", ".US")):
                raise ValueError(f"无效的股票代码: {symbol}")
            
            if symbol.endswith(".HK"):
                market, code = "hk", symbol.replace(".HK", "")
            elif symbol.endswith(".SZ"):
                market, code = "sz", symbol.replace(".SZ", "")
            elif symbol.endswith(".SS"):
                market, code = "sh", symbol.replace(".SS", "")
            else:
                market, code = "us", symbol  # 默认美股

            # 计算起止日期
            if start_date and end_date:
                end = datetime.strptime(end_date, "%Y%m%d").strftime("%Y%m%d")
                start = datetime.strptime(start_date, "%Y%m%d").strftime("%Y%m%d")
                start_date = start
                end_date = end
            else:
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days={"2y": 730, "1y": 365, "6m": 180, "3m": 90, "1m": 30}[period])).strftime("%Y%m%d")
                logging.info(f"起始日期: {start_date}, 结束日期: {end_date}")

            # 判断股票市场类型并调用相应的 akshare 接口
            @retry(wait=wait_random(min=1, max=5), stop=stop_after_attempt(5))
            def _fetch_with_retry():
                import loguru
                loguru.logger.debug("retry 调用 akshare") 
                if market == "hk":
                    return ak.stock_hk_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
                elif market in ("sz", "sh"):
                    return ak.stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
                else:  # us
                    return ak.stock_us_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")

            df = _fetch_with_retry()

            # 统一列名 → yfinance 格式
            df = df.rename(columns={
                "日期": "Date",
                "开盘": "Open",
                "收盘": "Close",
                "最高": "High",
                "最低": "Low",
                "成交量": "Volume"
            })[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")

            self.data = df
            self._calculate_technical_indicators()
            logging.info(f"数据获取成功，返回数据: {df.head()}")
            return df

        except Exception as e:
            logging.error(f"数据获取失败: {e}")
            return pd.DataFrame()  # 返回空的 DataFrame

    def _calculate_technical_indicators(self):
        """计算技术指标"""
        if self.data is None:
            return

        df = self.data
        
        # 移动平均线
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # 布林带
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
            df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # RSI
        df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
            df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # 成交量移动平均
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # 波动率
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # 价格变化率
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Price_Change_20d'] = df['Close'].pct_change(periods=20)

    def get_basic_info(self) -> Dict[str, Any]:
        """获取基础信息"""
        if self.data is None:
            return {}

        df = self.data
        latest = df.iloc[-1]
        
        return {
            "symbol": self.symbol,
            "current_price": latest['Close'],
            "high_52w": df['High'].max(),
            "low_52w": df['Low'].min(),
            "avg_volume": df['Volume'].mean(),
            "volatility": df['Volatility'].iloc[-1],
            "rsi": df['RSI'].iloc[-1],
            # "pe_ratio": getattr(yf.Ticker(self.symbol), 'info', {}).get('trailingPE', 'N/A'),
            # "market_cap": getattr(yf.Ticker(self.symbol), 'info', {}).get('marketCap', 'N/A')
            "pe_ratio": "N/A",
            "market_cap": "N/A"
        }
    
    def get_technical_analysis(self) -> Dict[str, Any]:
        """获取技术分析"""
        if self.data is None:
            return {}

        df = self.data
        latest = df.iloc[-1]
        
        # 趋势判断
        trend = self._get_trend(df)
        
        # 信号判断
        signals = self._get_signals(df)
        
        return {
            "trend": trend,
            "signals": signals,
            "current_price": latest['Close'],
            "ma_20": latest['MA_20'],
            "ma_50": latest['MA_50'],
            "rsi": latest['RSI'],
            "bb_position": self._get_bollinger_position(latest['Close'], latest['BB_upper'], latest['BB_lower']),
            "macd_signal": self._get_macd_signal(latest['MACD'], latest['MACD_signal'])
        }

    def _get_trend(self, df: pd.DataFrame) -> str:
        """判断趋势"""
        latest = df.iloc[-1]
        ma_20 = latest['MA_20']
        ma_50 = latest['MA_50']
        current_price = latest['Close']
        
        if current_price > ma_20 > ma_50:
            return "强势上涨"
        elif ma_20 > current_price > ma_50:
            return "温和上涨"
        elif ma_50 > current_price > ma_20:
            return "温和下跌"
        elif ma_50 > ma_20 > current_price:
            return "强势下跌"
        else:
            return "震荡"

    def _get_signals(self, df: pd.DataFrame) -> Dict[str, str]:
        """获取交易信号"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = {}
        
        # MA信号
        if latest['Close'] > latest['MA_20'] and prev['Close'] <= prev['MA_20']:
            signals['MA_20_cross'] = "金叉"
        elif latest['Close'] < latest['MA_20'] and prev['Close'] >= prev['MA_20']:
            signals['MA_20_cross'] = "死叉"
        
        # RSI信号
        rsi = latest['RSI']
        if rsi > 70:
            signals['RSI'] = "超买"
        elif rsi < 30:
            signals['RSI'] = "超卖"
        elif 30 <= rsi <= 70:
            signals['RSI'] = "中性"
        
        # MACD信号
        if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            signals['MACD'] = "金叉"
        elif latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            signals['MACD'] = "死叉"
        
        return signals

    def _get_bollinger_position(self, price: float, upper: float, lower: float) -> str:
        """获取布林带位置"""
        if price > upper:
            return "超买"
        elif price < lower:
            return "超卖"
        else:
            return "中性"

    def _get_macd_signal(self, macd: float, signal: float) -> str:
        """获取MACD信号"""
        if macd > signal:
            return "多头"
        else:
            return "空头"

    def integrate_news_analysis(self, news_analysis: dict) -> Dict[str, Any]:
        """
        将新闻分析结果整合到技术分析中
        """
        if self.data is None:
            return {}

        # 获取当前技术分析
        tech_analysis = self.get_technical_analysis()
        
        # 将新闻事件影响整合进来
        integrated_analysis = {
            **tech_analysis,
            "news_impact": {
                "score": news_analysis.get("impact_score", 0.0),
                "direction": news_analysis.get("event_direction", "neutral"),
                "reliability": news_analysis.get("reliability_score", 0.5),
                "events_count": len(news_analysis.get("news_events", []))
            },
            "combined_signal": self._combine_signals(tech_analysis, news_analysis)
        }
        
        return integrated_analysis

    def _combine_signals(self, tech_analysis: dict, news_analysis: dict) -> str:
        """
        结合技术分析和新闻分析生成综合信号
        """
        tech_signal = tech_analysis.get("trend", "震荡")
        news_impact = news_analysis.get("impact_score", 0.0)
        
        # 根据新闻影响调整技术信号
        if news_impact > 0.3:
            if tech_signal in ["强势上涨", "温和上涨"]:
                return "强烈看涨"
            elif tech_signal in ["强势下跌", "温和下跌"]:
                return "看涨（新闻面支撑）"
            else:
                return "看涨（新闻面驱动）"
        elif news_impact < -0.3:
            if tech_signal in ["强势下跌", "温和下跌"]:
                return "强烈看跌"
            elif tech_signal in ["强势上涨", "温和上涨"]:
                return "看跌（新闻面压制）"
            else:
                return "看跌（新闻面驱动）"
        else:
            return tech_signal  # 新闻影响较小，以技术面为主

    def create_analysis_chart(self) -> go.Figure:
        """创建分析图表"""
        if self.data is None:
            return go.Figure()

        df = self.data.tail(100)  # 只显示最近100天的数据

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.2, 0.15, 0.15],
            subplot_titles=['价格与移动平均线', '成交量', 'RSI', 'MACD']
        )

        # 价格和移动平均线
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='价格'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA_20'], name='MA20', line=dict(color='orange')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA_50'], name='MA50', line=dict(color='purple')),
            row=1, col=1
        )

        # 成交量
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color='lightblue'),
            row=2, col=1
        )

        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='red')),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', line=dict(color='orange')),
            row=4, col=1
        )
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_hist'], name='Histogram', marker_color='gray'),
            row=4, col=1
        )

        fig.update_layout(
            title=f"{self.symbol} 技术分析图表",
            height=800,
            showlegend=True
        )

        return fig

    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        if self.data is None:
            return {}

        df = self.data
        
        return {
            "data_period": f"{df.index[0].strftime('%Y-%m-%d')} 到 {df.index[-1].strftime('%Y-%m-%d')}",
            "total_days": len(df),
            "price_range": f"{df['Low'].min():.2f} - {df['High'].max():.2f}",
            "avg_daily_return": df['Price_Change'].mean() * 100,
            "volatility_annualized": df['Price_Change'].std() * np.sqrt(252) * 100,
            "sharpe_ratio": (df['Price_Change'].mean() / df['Price_Change'].std()) * np.sqrt(252) if df['Price_Change'].std() != 0 else 0,
            "max_drawdown": self._calculate_max_drawdown(df['Close'])
        }

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def get_technical_analysis_with_news(self, news_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取结合新闻特征的技术分析
        """
        if self.data is None:
            return {}

        # 获取基础技术分析
        base_analysis = self.get_technical_analysis()
        
        # 集成新闻特征
        integrated_analysis = {
            **base_analysis,
            "news_features": {
                "sentiment_score": news_features.get("sentiment_score", 0.0),
                "reliability_score": news_features.get("reliability_score", 0.5),
                "news_volume": news_features.get("news_volume", 0),
                "news_diversity": news_features.get("news_diversity", 0),
                "key_themes": news_features.get("key_themes", []),
                "event_types": news_features.get("event_types", [])
            },
            "combined_signal": self._combine_tech_and_news_signals(base_analysis, news_features)
        }
        
        return integrated_analysis

    def _combine_tech_and_news_signals(self, tech_analysis: dict, news_features: dict) -> str:
        """
        结合技术分析和新闻特征生成综合信号
        """
        tech_signal = tech_analysis.get("trend", "震荡")
        news_sentiment = news_features.get("sentiment_score", 0.0)
        
        # 根据新闻情绪调整技术信号
        if news_sentiment > 0.3:
            if tech_signal in ["强势上涨", "温和上涨"]:
                return "强烈看涨（技术+新闻双重确认）"
            elif tech_signal in ["强势下跌", "温和下跌"]:
                return "看涨（新闻面反转技术面）"
            else:
                return "看涨（新闻面驱动）"
        elif news_sentiment < -0.3:
            if tech_signal in ["强势下跌", "温和下跌"]:
                return "强烈看跌（技术+新闻双重确认）"
            elif tech_signal in ["强势上涨", "温和上涨"]:
                return "看跌（新闻面反转技术面）"
            else:
                return "看跌（新闻面驱动）"
        else:
            return tech_signal  # 新闻情绪中性，以技术面为主

    def create_analysis_chart_with_news(self, news_features: Dict[str, Any]) -> go.Figure:
        """
        创建结合新闻特征的分析图表
        """
        if self.data is None:
            return go.Figure()

        df = self.data.tail(100)  # 只显示最近100天的数据

        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
            subplot_titles=['价格与移动平均线', '成交量', 'RSI', '新闻情绪指数', '新闻数量']
        )

        # 价格和移动平均线
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='价格'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA_20'], name='MA20', line=dict(color='orange')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA_50'], name='MA50', line=dict(color='purple')),
            row=1, col=1
        )

        # 成交量
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color='lightblue'),
            row=2, col=1
        )

        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='red')),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # 新闻情绪指数 (模拟数据，实际需要根据新闻分析结果计算)
        news_sentiment = [0.0] * len(df)  # 这里需要根据实际新闻数据填充
        fig.add_trace(
            go.Scatter(x=df.index, y=news_sentiment, name='新闻情绪指数', line=dict(color='green')),
            row=4, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)

        # 新闻数量 (模拟数据)
        news_count = [0] * len(df)  # 这里需要根据实际新闻数据填充
        fig.add_trace(
            go.Bar(x=df.index, y=news_count, name='新闻数量', marker_color='yellow'),
            row=5, col=1
        )

        fig.update_layout(
            title=f"{self.symbol} 结合新闻分析的技术图表",
            height=1000,
            showlegend=True
        )

        return fig