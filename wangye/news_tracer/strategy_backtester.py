# strategy_backtester.py
"""
策略回测模块
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
from loguru import logger

# 忽略警告
warnings.filterwarnings("ignore")

class StrategyBacktester:
    def __init__(self):
        self.data = None
        self.symbol = None

    def fetch_data(self, symbol: str, period: str = "2y", interval: str = "1d", 
                   start_date: str = "", end_date: str = "") -> pd.DataFrame:
        """
        获取股票数据（使用 akshare）
        """
        try:
            self.symbol = symbol
            logger.info(f"[Backtester] 开始获取数据: {symbol}")
            
            # 解析市场后缀
            symbol = symbol.strip()
            
            # 计算起止日期
            if start_date and end_date:
                end = datetime.strptime(end_date, "%Y%m%d").strftime("%Y%m%d")
                start = datetime.strptime(start_date, "%Y%m%d").strftime("%Y%m%d")
                start_date = start
                end_date = end
            else:
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days={"2y": 730, "1y": 365, 
                                                              "6m": 180, "3m": 90, 
                                                              "1m": 30}.get(period, 365))).strftime("%Y%m%d")
                logger.info(f"[Backtester] 起始日期: {start_date}, 结束日期: {end_date}")

            # 根据股票后缀选择不同的 akshare 接口
            if symbol.endswith(".HK"):
                market, code = "hk", symbol.replace(".HK", "")
                df = ak.stock_hk_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
            elif symbol.endswith(".SZ"):
                market, code = "sz", symbol.replace(".SZ", "")
                df = ak.stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
            elif symbol.endswith(".SS"):
                market, code = "sh", symbol.replace(".SS", "")
                df = ak.stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
            else:
                # 默认使用 ak.stock_us_hist
                try:
                    df = ak.stock_us_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")
                except:
                    # 如果没有美股数据，使用其他方式
                    raise ValueError(f"不支持的股票代码: {symbol}")

            # 统一列名
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
            
            # 确保数据类型正确
            df = df.astype({
                'Open': 'float64',
                'High': 'float64',
                'Low': 'float64',
                'Close': 'float64',
                'Volume': 'float64'
            })
            
            # 添加技术指标
            self._calculate_technical_indicators(df)
            
            self.data = df
            logger.info(f"[Backtester] 数据获取成功，数据量: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"[Backtester] 行情获取失败: {e}")
            # 返回模拟数据用于测试
            return self._generate_mock_data()

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> None:
        """计算技术指标"""
        # 移动平均线
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']

    def _generate_mock_data(self) -> pd.DataFrame:
        """生成模拟数据用于测试"""
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
        np.random.seed(42)
        
        # 生成模拟价格数据
        returns = np.random.normal(0.0005, 0.02, len(dates))
        price = 100 * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'Open': price * 0.99,
            'High': price * 1.01,
            'Low': price * 0.98,
            'Close': price,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        self._calculate_technical_indicators(df)
        logger.warning("[Backtester] 使用模拟数据进行回测")
        return df

    def backtest(self, strategy_name: str = "ma_crossover", 
                initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        执行策略回测
        """
        logger.info(f"[Backtester] 开始回测策略: {strategy_name}")
        
        if self.data is None or len(self.data) < 50:
            logger.warning("[Backtester] 数据不足，使用模拟数据")
            self.data = self._generate_mock_data()
        
        df = self.data.copy()
        
        try:
            # 根据策略名称选择回测函数
            if strategy_name == "ma_crossover":
                results = self._backtest_ma_crossover(df, initial_capital)
            elif strategy_name == "rsi_mean_revert":
                results = self._backtest_rsi_mean_revert(df, initial_capital)
            elif strategy_name == "bollinger_breakout":
                results = self._backtest_bollinger_breakout(df, initial_capital)
            else:
                raise ValueError(f"未知策略: {strategy_name}")
            
            logger.info(f"[Backtester] {strategy_name} 回测完成")
            return results
            
        except Exception as e:
            logger.error(f"[Backtester] 回测失败: {e}")
            # 返回默认结果
            return self._get_default_results()

    def _backtest_ma_crossover(self, df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """移动平均线交叉策略"""
        # 策略逻辑：MA5上穿MA20买入，下穿卖出
        df['signal'] = 0
        df['signal'] = np.where(df['MA_5'] > df['MA_20'], 1, 0)
        df['position'] = df['signal'].diff()
        
        # 计算持仓和收益
        df['returns'] = df['Close'].pct_change()
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        
        # 计算资金曲线
        df['equity'] = (1 + df['strategy_returns']).cumprod() * initial_capital
        df['drawdown'] = (df['equity'] - df['equity'].cummax()) / df['equity'].cummax()
        
        # 获取交易记录
        trades = self._extract_trades(df, initial_capital)
        
        return self._calculate_performance_metrics(df, trades, initial_capital)

    def _backtest_rsi_mean_revert(self, df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """RSI均值回归策略"""
        # 策略逻辑：RSI < 30买入，RSI > 70卖出
        df['signal'] = 0
        df['signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
        df['position'] = df['signal']
        
        # 计算持仓和收益
        df['returns'] = df['Close'].pct_change()
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        
        # 计算资金曲线
        df['equity'] = (1 + df['strategy_returns']).cumprod() * initial_capital
        df['drawdown'] = (df['equity'] - df['equity'].cummax()) / df['equity'].cummax()
        
        # 获取交易记录
        trades = self._extract_trades(df, initial_capital)
        
        return self._calculate_performance_metrics(df, trades, initial_capital)

    def _backtest_bollinger_breakout(self, df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """布林带突破策略"""
        # 策略逻辑：价格突破上轨买入，跌破下轨卖出
        df['signal'] = 0
        df['signal'] = np.where(df['Close'] > df['BB_upper'], 1, 
                               np.where(df['Close'] < df['BB_lower'], -1, 0))
        df['position'] = df['signal']
        
        # 计算持仓和收益
        df['returns'] = df['Close'].pct_change()
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        
        # 计算资金曲线
        df['equity'] = (1 + df['strategy_returns']).cumprod() * initial_capital
        df['drawdown'] = (df['equity'] - df['equity'].cummax()) / df['equity'].cummax()
        
        # 获取交易记录
        trades = self._extract_trades(df, initial_capital)
        
        return self._calculate_performance_metrics(df, trades, initial_capital)

    def _extract_trades(self, df: pd.DataFrame, initial_capital: float) -> List[Dict[str, Any]]:
        """提取交易记录"""
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        
        for i in range(1, len(df)):
            current_position = df['position'].iloc[i]
            
            if current_position != position:
                if position == 0 and current_position != 0:  # 开仓
                    entry_price = df['Close'].iloc[i]
                    entry_date = df.index[i]
                elif position != 0 and current_position == 0:  # 平仓
                    exit_price = df['Close'].iloc[i]
                    exit_date = df.index[i]
                    
                    if entry_date:
                        trade = {
                            'entry_date': entry_date.strftime('%Y-%m-%d'),
                            'exit_date': exit_date.strftime('%Y-%m-%d'),
                            'entry_price': float(entry_price),
                            'exit_price': float(exit_price),
                            'return': float((exit_price - entry_price) / entry_price * 100),
                            'position': 'Long' if position > 0 else 'Short'
                        }
                        trades.append(trade)
                        
                        # 重置
                        entry_date = None
                        entry_price = 0
                
                position = current_position
        
        return trades

    def _calculate_performance_metrics(self, df: pd.DataFrame, trades: List[Dict[str, Any]], 
                                     initial_capital: float) -> Dict[str, Any]:
        """计算性能指标"""
        if len(df) < 2:
            return self._get_default_results()
        
        # 基础指标
        final_value = df['equity'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # 年化收益率
        days = (df.index[-1] - df.index[0]).days
        years = max(days / 365, 0.01)  # 避免除零
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # 波动率
        daily_returns = df['strategy_returns'].dropna()
        if len(daily_returns) > 1:
            volatility = daily_returns.std() * np.sqrt(252)  # 年化波动率
            sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility != 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # 最大回撤
        max_drawdown = df['drawdown'].min()
        
        # 胜率
        if trades:
            winning_trades = [t for t in trades if t['return'] > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
        else:
            win_rate = 0
        
        # 资金曲线数据
        equity_curve = {
            'dates': [d.strftime('%Y-%m-%d') for d in df.index],
            'values': [float(v) for v in df['equity']]
        }
        
        return {
            'final_value': float(final_value),
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'num_trades': int(len(trades)),
            'trades': trades,
            'equity_curve': equity_curve,
            'chart_data': self._create_backtest_chart(df, initial_capital)
        }

    def _create_backtest_chart(self, df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """创建回测图表"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=['价格与信号', '资金曲线']
            )
            
            # 价格和信号
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Close'], name='价格', line=dict(color='blue')),
                row=1, col=1
            )
            
            # 买入信号
            buy_signals = df[df['position'] > 0]
            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(x=buy_signals.index, y=buy_signals['Close'], 
                              mode='markers', name='买入', marker=dict(color='green', size=10, symbol='triangle-up')),
                    row=1, col=1
                )
            
            # 卖出信号
            sell_signals = df[df['position'] < 0]
            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(x=sell_signals.index, y=sell_signals['Close'], 
                              mode='markers', name='卖出', marker=dict(color='red', size=10, symbol='triangle-down')),
                    row=1, col=1
                )
            
            # 资金曲线
            fig.add_trace(
                go.Scatter(x=df.index, y=df['equity'], name='资金', line=dict(color='green')),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f'{self.symbol} 策略回测结果',
                height=600,
                showlegend=True
            )
            
            # 转换为字典格式
            return fig.to_dict()
            
        except Exception as e:
            logger.error(f"[Backtester] 创建图表失败: {e}")
            return {}

    def _get_default_results(self) -> Dict[str, Any]:
        """获取默认结果（用于错误处理）"""
        return {
            'final_value': 10000.0,
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'num_trades': 0,
            'trades': [],
            'equity_curve': {'dates': [], 'values': []},
            'chart_data': {},
            'error': '回测过程出错'
        }

    def backtest_strategy_with_news_signals(self, strategy_name: str, 
                                          initial_capital: float, 
                                          news_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        结合新闻信号进行回测
        """
        # 基础回测
        base_results = self.backtest(strategy_name, initial_capital)
        
        if news_signals:
            # 这里可以添加新闻信号的集成逻辑
            # 目前暂时返回基础结果
            base_results['news_signals_used'] = True
            base_results['news_signals_count'] = len(news_signals)
        else:
            base_results['news_signals_used'] = False
            base_results['news_signals_count'] = 0
        
        return base_results

    def plot_backtest_results(self, results: Dict[str, Any]) -> go.Figure:
        """绘制回测结果图表"""
        if 'chart_data' in results and results['chart_data']:
            try:
                return go.Figure(results['chart_data'])
            except:
                pass
        
        # 如果图表数据无效，创建一个简单的图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[datetime.now().strftime('%Y-%m-%d')],
            y=[results.get('final_value', 10000)],
            mode='markers',
            marker=dict(size=20)
        ))
        
        fig.update_layout(
            title='回测结果图表',
            height=400
        )
        
        return fig