# ml_strategy.py
"""
策略机器学习模块
使用机器学习模型进行股票趋势预测和策略优化
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Any, List, Optional, Tuple
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import talib
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

class MLStrategy:
    def __init__(self):
        self.data = None
        self.symbol = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []

    def fetch_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """获取数据"""
        try:
            ticker = yf.Ticker(symbol)
            self.data = ticker.history(period=period, interval=interval)
            self.symbol = symbol
            
            # 计算特征
            self._calculate_features()
            
            return self.data
        except Exception as e:
            raise Exception(f"获取股票数据失败: {e}")

    def _calculate_features(self):
        """计算机器学习特征"""
        if self.data is None:
            return

        df = self.data
        
        # 基础价格特征
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # 移动平均特征
        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Price_MA_{window}_Pct'] = (df['Close'] - df[f'MA_{window}']) / df[f'MA_{window}']
        
        # 技术指标
        df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'].values)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'].values)
        
        # 布林带位置
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # 成交量特征
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # 波动率
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # 滞后特征
        for lag in [1, 2, 3, 5]:
            df[f'Price_Change_lag_{lag}'] = df['Price_Change'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # 标签：未来1天涨跌（1为涨，0为跌）
        df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] > 0).astype(int)
        
        # 选择特征列
        self.feature_columns = [
            'Price_Change', 'High_Low_Pct', 'Open_Close_Pct',
            'Price_MA_5_Pct', 'Price_MA_10_Pct', 'Price_MA_20_Pct', 'Price_MA_50_Pct',
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'BB_position',
            'Volume_Ratio', 'Volatility'
        ] + [f'Price_Change_lag_{i}' for i in [1, 2, 3, 5]] + [f'Volume_lag_{i}' for i in [1, 2, 3, 5]]
        
        # 删除包含NaN的行
        df = df.dropna()
        self.data = df

    def prepare_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """准备训练和测试数据"""
        if self.data is None or self.data.empty:
            raise Exception("数据为空，请先调用 fetch_data()")

        X = self.data[self.feature_columns]
        y = self.data['Target']
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 转换回DataFrame以保持列名
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns, index=X_test.index)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, model_type: str = 'random_forest', **kwargs) -> Dict[str, Any]:
        """
        训练模型
        model_type: 'random_forest', 'gradient_boosting', 'logistic_regression', 'svm'
        """
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # 选择模型
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42,
                **kwargs
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 3),
                random_state=42,
                **kwargs
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                **kwargs
            )
        elif model_type == 'svm':
            self.model = SVC(
                probability=True,
                random_state=42,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 训练模型
        self.model.fit(X_train.values, y_train.values)
        
        # 预测
        y_pred = self.model.predict(X_test.values)
        y_pred_proba = self.model.predict_proba(X_test.values)[:, 1] if hasattr(self.model, "predict_proba") else None
        
        # 评估模型
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X_train.values, y_train.values, cv=5)
        
        results = {
            "model_type": model_type,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_mean_score": cv_scores.mean(),
            "cv_std_score": cv_scores.std(),
            "feature_importance": self._get_feature_importance(),
            "predictions": y_pred,
            "probabilities": y_pred_proba,
            "actual_values": y_test.values
        }
        
        return results

    def _get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_columns, importance))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(self.model, 'coef_'):
            # 对于线性模型
            coef = abs(self.model.coef_[0])
            feature_importance = dict(zip(self.feature_columns, coef))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}

    def predict_next_day(self) -> Dict[str, Any]:
        """预测下一天的走势"""
        if self.model is None:
            raise Exception("模型未训练，请先调用 train_model()")
        
        # 使用最新数据进行预测
        latest_data = self.data[self.feature_columns].iloc[-1:].values
        latest_data_scaled = self.scaler.transform(latest_data)
        
        prediction = self.model.predict(latest_data_scaled)[0]
        probability = self.model.predict_proba(latest_data_scaled)[0] if hasattr(self.model, "predict_proba") else None
        
        return {
            "prediction": "上涨" if prediction == 1 else "下跌",
            "probability_up": probability[1] if probability is not None else None,
            "probability_down": probability[0] if probability is not None else None,
            "confidence": max(probability) if probability is not None else None
        }

    def backtest_ml_strategy(self, initial_capital: float = 10000, threshold: float = 0.55) -> Dict[str, Any]:
        """使用ML模型进行策略回测"""
        if self.model is None:
            raise Exception("模型未训练，请先调用 train_model()")
        
        df = self.data.copy()
        
        # 生成预测信号
        predictions = []
        probabilities = []
        
        for i in range(len(df)):
            if i < len(self.feature_columns):  # 确保有足够的数据进行预测
                predictions.append(0)
                probabilities.append(0.5)
                continue
                
            # 使用当前数据预测下一天
            data_point = df[self.feature_columns].iloc[i:i+1].values
            data_point_scaled = self.scaler.transform(data_point)
            
            pred = self.model.predict(data_point_scaled)[0]
            prob = self.model.predict_proba(data_point_scaled)[0] if hasattr(self.model, "predict_proba") else [0.5, 0.5]
            
            # 只有当置信度超过阈值时才生成信号
            if max(prob) >= threshold:
                predictions.append(pred)
                probabilities.append(prob[1])
            else:
                predictions.append(0)  # 不操作
                probabilities.append(0.5)
        
        df['ML_Signal'] = predictions
        df['ML_Probability'] = probabilities
        
        # 模拟交易
        df['Holdings'] = 0
        df['Cash'] = initial_capital
        df['Total'] = initial_capital
        df['Returns'] = 0
        
        for i in range(1, len(df)):
            # 如果预测上涨且置信度高
            if df['ML_Signal'].iloc[i-1] == 1 and df['ML_Probability'].iloc[i-1] >= threshold:
                # 买入
                shares_to_buy = df['Cash'].iloc[i-1] // df['Close'].iloc[i]
                df.loc[df.index[i], 'Holdings'] = df['Holdings'].iloc[i-1] + shares_to_buy
                df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1] - shares_to_buy * df['Close'].iloc[i]
            elif df['ML_Signal'].iloc[i-1] == 0:  # 预测下跌，卖出
                df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1] + df['Holdings'].iloc[i-1] * df['Close'].iloc[i]
                df.loc[df.index[i], 'Holdings'] = 0
            else:
                # 保持现状
                df.loc[df.index[i], 'Holdings'] = df['Holdings'].iloc[i-1]
                df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]
            
            # 计算总资产
            df.loc[df.index[i], 'Total'] = df['Holdings'].iloc[i] * df['Close'].iloc[i] + df['Cash'].iloc[i]
            
            # 计算收益率
            if i > 0:
                df.loc[df.index[i], 'Returns'] = (df['Total'].iloc[i] - df['Total'].iloc[i-1]) / df['Total'].iloc[i-1]
        
        # 计算整体指标
        total_return = (df['Total'].iloc[-1] - initial_capital) / initial_capital
        annualized_return = (df['Total'].iloc[-1] / initial_capital) ** (252 / len(df)) - 1
        volatility = df['Returns'].std() * np.sqrt(252)
        sharpe_ratio = (df['Returns'].mean() / df['Returns'].std()) * np.sqrt(252) if df['Returns'].std() != 0 else 0
        
        # 计算最大回撤
        rolling_max = df['Total'].expanding().max()
        drawdown = (df['Total'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 计算胜率
        winning_trades = len(df[df['Returns'] > 0])
        total_trades = len(df[df['Returns'] != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            "symbol": self.symbol,
            "initial_capital": initial_capital,
            "final_value": df['Total'].iloc[-1],
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "signals": df,
            "equity_curve": df[['Total']].copy()
        }

    def plot_ml_results(self, backtest_results: Dict[str, Any]) -> go.Figure:
        """绘制ML策略结果图表"""
        df = backtest_results['signals']
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=['价格与ML信号', '资产曲线', 'ML置信度']
        )

        # 价格与ML信号
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='价格', line=dict(color='blue')),
            row=1, col=1
        )
        
        # 高置信度买入信号
        high_conf_buy = df[(df['ML_Signal'] == 1) & (df['ML_Probability'] >= 0.6)]
        if not high_conf_buy.empty:
            fig.add_trace(
                go.Scatter(
                    x=high_conf_buy.index, 
                    y=high_conf_buy['Close'], 
                    mode='markers',
                    name='高置信度买入',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ),
                row=1, col=1
            )
        
        # 高置信度卖出信号
        high_conf_sell = df[(df['ML_Signal'] == 0) & (df['ML_Probability'] <= 0.4)]
        if not high_conf_sell.empty:
            fig.add_trace(
                go.Scatter(
                    x=high_conf_sell.index, 
                    y=high_conf_sell['Close'], 
                    mode='markers',
                    name='高置信度卖出',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ),
                row=1, col=1
            )

        # 资产曲线
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Total'], name='总资产', line=dict(color='orange')),
            row=2, col=1
        )

        # ML置信度
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ML_Probability'], name='上涨概率', line=dict(color='purple')),
            row=3, col=1
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=3, col=1)

        fig.update_layout(
            title=f"{self.symbol} ML策略回测结果",
            height=800,
            showlegend=True
        )

        return fig

    def save_model(self, filepath: str):
        """保存模型"""
        if self.model is None:
            raise Exception("模型未训练")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'symbol': self.symbol
        }
        
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.symbol = model_data['symbol']
        
        
    def combine_news_and_price_features(self, news_features: Dict[str, Any]) -> pd.DataFrame:
        """
        将新闻特征与价格特征结合
        """
        if self.data is None:
            raise Exception("数据未加载，请先调用 fetch_data()")
        
        df = self.data.copy()
        
        # 添加新闻特征作为新的列
        # 这里需要根据实际新闻特征进行适配
        df['News_Sentiment'] = news_features.get("sentiment_score", 0.0)
        df['News_Reliability'] = news_features.get("reliability_score", 0.5)
        df['News_Volume'] = news_features.get("news_volume", 0)
        df['News_Diversity'] = news_features.get("news_diversity", 0)
        
        # 将新闻特征添加到特征列中
        news_feature_cols = ['News_Sentiment', 'News_Reliability', 'News_Volume', 'News_Diversity']
        self.feature_columns.extend(news_feature_cols)
        
        # 删除包含NaN的行
        df = df.dropna()
        self.data = df
        
        return df

    def backtest_ml_strategy_with_news_features(self, features: pd.DataFrame, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用新闻特征的ML策略回测
        """
        # 这里实现结合新闻特征的回测逻辑
        # 基本上是原有回测方法的扩展版本
        return self.backtest_ml_strategy()

    def get_news_feature_importance(self) -> Dict[str, float]:
        """
        获取新闻特征的重要性
        """
        # 返回新闻相关特征的重要性
        if hasattr(self.model, 'feature_importances_') and self.feature_columns:
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_columns, importance))
            
            # 只返回新闻相关的特征重要性
            news_features = {k: v for k, v in feature_importance.items() if 'News_' in k}
            return dict(sorted(news_features.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}