# examples/news_tracer/news_tracer_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import httpx
from loguru import logger
from pydantic import Field

# 在文件顶部导入
from pydantic import BaseModel
from typing import Optional

class StockRequest(BaseModel):
    symbol: str
    method: Optional[str] = "ma"  # ma, diff, llm

class StockPrediction(BaseModel):
    symbol: str
    trend: str
    prediction: str
    confidence: float
    last_price: float
    analysis: Optional[str] = None

print("=== news_tracer_api 模块开始加载 ===")

# 打印环境变量以调试
print("=== API模块环境变量 ===")
print(f"ARK_API_KEY: {os.getenv('ARK_API_KEY')}")
print(f"ARK_BASE_URL: {os.getenv('ARK_BASE_URL')}")
print(f"SERPER_API_KEY: {os.getenv('SERPER_API_KEY')}")

# 导入 mira 模块
try:
    from mira import LLMTool, LLMJson, OpenAIArgs, OpenRouterLLM, HumanMessage, ToolMessage
    print("成功导入 mira 模块")
except Exception as e:
    print(f"导入 mira 模块失败: {e}")
    raise

# 创建API路由器
api_router = APIRouter(prefix="/api")

# 导入股票分析模块（在api_router创建后）
from stock_analyzer import FinancialAnalyzer
from strategy_backtester import StrategyBacktester
from ml_strategy import MLStrategy

# 新增：股票新闻事件分析端点
@api_router.post("/stock/news-analysis")
async def stock_news_analysis(request: dict):
    """
    分析股票相关新闻事件对股价的影响
    使用新闻溯源Agent进行深度分析
    """
    try:
        symbol = request.get("symbol", "")
        time_range = request.get("time_range", "w")  # w: week, m: month, y: year
        
        if not tracer_agent or not llm_instance:
            raise HTTPException(status_code=500, detail="LLM实例未初始化")
        
        # 使用新闻溯源Agent分析股票相关事件
        news_analysis = await tracer_agent.analyze_stock_news(symbol, time_range)
        
        return {
            "symbol": symbol,
            "news_analysis": news_analysis,
            "event_impact_score": news_analysis.get("impact_score", 0.0),
            "event_direction": news_analysis.get("direction", "neutral"),
            "reliability_score": news_analysis.get("reliability", 0.5)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 股票基础分析端点 - 修改以利用新闻溯源Agent提取的特征
@api_router.post("/stock/analyze")
async def analyze_stock(request: dict):
    """
    股票基础分析，结合新闻溯源Agent提取的特征
    """
    try:
        symbol = request.get("symbol", "")
        
        if not tracer_agent or not llm_instance:
            raise HTTPException(status_code=500, detail="LLM实例未初始化")
        
        # 首先通过新闻溯源Agent获取新闻数据
        news_result = await tracer_agent.trace_claim(f"{symbol} 股票分析和市场情绪", max_depth=2)
        
        # 从新闻中提取特征，用于股票分析
        news_features = extract_features_from_news(news_result.steps)
        
        # 获取基本股票数据
        analyzer = FinancialAnalyzer()
        analyzer.fetch_data(symbol)
        basic_info = analyzer.get_basic_info()
        
        # 结合新闻特征进行技术分析
        tech_analysis = analyzer.get_technical_analysis_with_news(news_features)
        
        # 生成图表
        chart_fig = analyzer.create_analysis_chart_with_news(news_features)
        
        # 将图表转换为字典格式
        chart_data = chart_fig.to_dict()
        
        return {
            "symbol": symbol,
            "basic_info": basic_info,
            "technical_analysis": tech_analysis,
            "chart_data": chart_data,
            "news_features": news_features,
            "original_trace_data": news_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 策略回测端点 - 修改以使用新闻溯源Agent提取的信号
@api_router.post("/stock/backtest")
async def backtest_strategy(request: dict):
    """
    策略回测：输入 symbol / strategy / initial_capital
    返回规范指标 + 可序列化图表
    """
    try:
        symbol   = request.get("symbol", "").upper()
        strategy = request.get("strategy", "ma_crossover")
        capital  = float(request.get("initial_capital", 100000))

        if not symbol:
            raise HTTPException(status_code=400, detail="symbol 不能为空")

        # 1. 取行情
        backtester = StrategyBacktester()
        backtester.fetch_data(symbol)

        # 2. 回测
        results = backtester.backtest(strategy, initial_capital=capital)

        # 3. 图表
        fig = backtester.plot_backtest_results(results)
        results = backtester.backtest(strategy_name=strategy, initial_capital=initial_capital)
        return {
            "symbol": symbol,
            "strategy": strategy,
            **results          # chart_data 已是 dict
        }

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

# Pydantic 模型定义

class NewsSource(BaseModel):
    title: str
    link: str
    snippet: str
    date: str
    
class RoundDetail(BaseModel):
    round: int
    keywords: List[str]          # 本轮实际搜索关键词
    query: str                   # 人类可读描述
    sources: List[NewsSource]
    next_focus: str
    stop_early: bool = False

class TraceRequest(BaseModel):
    claim: str
    max_depth: Optional[int] = 2

class TraceStep(BaseModel):
    step: int
    query: str
    sources: List[NewsSource]

# 修复：先定义 StockImpactAnalysis 类，再定义 TraceResponse 类
class StockImpactAnalysis(BaseModel):
    symbol: str
    price_before: float
    price_after: float
    change_percent: float
    timeline: List[dict]  # 修改：使用 dict 而不是 Dict[str, Any]

class TraceResponse(BaseModel):
    claim: str
    steps: List[TraceStep]          # 旧接口兼容
    rounds: List[RoundDetail]       # 👈 新增：前端时间线专用
    conclusion: str
    reliability: float
    stock_impact: Optional[StockImpactAnalysis] = None
    

# 定义工具
class GoogleSearchTool(LLMTool):
    query: str = Field(
        ...,
        title="搜索关键词",
        description="用于 Google 搜索的关键词"
    )

    @classmethod
    def schema(cls) -> dict:
        # 完全自己构造，避开 mira 的 pop
        return {
            "type": "function",
            "function": {
                "name": "GoogleSearchTool",
                "description": "",          # 框架需要，留空即可
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "用于 Google 搜索的关键词"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def __call__(self):
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            logger.error("SERPER_API_KEY 环境变量未设置")
            return {"error": "SERPER_API_KEY 环境变量未设置"}
        
        try:
            url = "https://google.serper.dev/search"
            headers = {"X-API-KEY": api_key}
            payload = {"q": self.query, "num": 10}  # 增加搜索结果数量以获取更多股票新闻
            
            logger.info(f"发送搜索请求到: {url}")
            
            response = httpx.post(url, json=payload, headers=headers, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                organic_results = data.get("organic", [])
                results = []
                for item in organic_results[:10]:  # 获取更多结果
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "date": item.get("date", "未知")
                    })
                return results
            else:
                logger.error(f"搜索API错误: {response.status_code}")
                return {"error": f"搜索API错误: {response.status_code}"}
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return {"error": f"搜索失败: {str(e)}"}

# 新闻溯源Agent - 扩展功能
class NewsTracingAgent:
    def __init__(self, llm: OpenRouterLLM):
        self.llm = llm
    
    async def trace_claim(self, claim: str, max_depth: int = 2) -> TraceResponse:
        logger.info(f"开始递进溯源声明: {claim}, 深度: {max_depth}")
        trace_steps = []
        current_focus = claim          # 每一轮深挖的焦点
        messages = []
        rounds: List[RoundDetail] = [] 

        for depth in range(1, max_depth + 1):
            logger.info(f"==== 第 {depth} 轮深挖 ====")

            # ① 构造递进提示词
            if depth == 1:
                sys = f"""
你是「深度新闻溯源机器人」。  
**第1轮（骨架轮）**：对声明 "{claim}" 构造 2-3 个 **宽泛关键词** 搜全貌，返回：
- timeline：事件始末（YYYY-MM-DD）
- entities：关键人/公司/政策
- original_sources：官网/权威媒体链接（≥2 条）
- next_focus：下一轮要深挖的实体（1 个）
JSON 格式，并调用 GoogleSearchTool。
"""
            else:
                sys = f"""
**第{depth}轮（深挖轮）**：聚焦实体「{current_focus}」构造 **更具体** 2 个关键词，  
只搜 **上一轮时间之前** 的信息，继续返回：
- timeline / entities / original_sources
- next_focus（再给 1 个）
若已追到 **primary source（政府/法院/公告 PDF）** → 输出 stop_early: true 提前结束。
"""

            messages.append(HumanMessage(content=sys))

            # ② 调用 LLM + 工具
            response_messages = await self.llm.forward(messages=messages, tools=[GoogleSearchTool])
            current_messages = response_messages[0] if isinstance(response_messages, list) else response_messages
            messages.extend(current_messages if isinstance(current_messages, list) else [current_messages])

            # ③ 解析返回 → 提取 sources + next_focus
            parsed = self._parse_round_output(current_messages)
            sources = parsed.get("sources", [])
            current_focus = parsed.get("next_focus", current_focus)
            stop_early = parsed.get("stop_early", False)

            # ④ 记录步骤
            if sources:
                trace_steps.append(TraceStep(
                    step=len(trace_steps) + 1,
                    query=f"深挖 {current_focus}",
                    sources=[NewsSource(**it) for it in sources if isinstance(it, dict) and "error" not in it]
                ))
                logger.info(f"[trace_claim] 第{depth}轮追加后，steps长度={len(trace_steps)}")

            # ⑤ 提前终止条件
            if stop_early or len(trace_steps) >= max_depth:
                break
            kw_sys = HumanMessage(
                content='请把上一步你实际用来搜索的 2-3 个关键词用 JSON 列表返回，例如：["B站鬼畜起源","哔哩哔哩鬼畜历史"] *不要解释*，只返回 JSON。'
            )
            kw_msg = await self.llm.forward(messages=messages + [kw_sys])
            kw_raw = kw_msg[0][0].content if isinstance(kw_msg, list) else kw_msg.content
            try:
                keywords = json.loads(kw_raw)
            except:
                keywords = [current_focus]

            rounds.append(RoundDetail(
                round=depth,
                keywords=keywords,
                query=f"深挖 {current_focus}",
                sources=[NewsSource(**it) for it in sources if isinstance(it, dict) and "error" not in it],
                next_focus=current_focus,
                stop_early=stop_early
            ))    
        # ⑥ 最终结论
        conclusion_content = "未能生成结论"
        try:
            final_msg = await self.llm.forward(messages + [HumanMessage(
                content=f'基于以上递进搜索结果，对"{claim}"给出：1. 是否属实 2. 主要证据 3. 可靠性评分（0-1）'
            )])
            # 统一取第一条 AIMessage 的 content
            raw = final_msg[0] if isinstance(final_msg, list) else final_msg
            ai_msg = raw[0] if isinstance(raw, list) else raw
            conclusion_content = ai_msg.content if hasattr(ai_msg, 'content') else str(ai_msg)
        except Exception as e:
            logger.exception(e)
            
        reliability_score = extract_reliability_from_text(conclusion_content)
        return TraceResponse(
            claim=claim,
            steps=trace_steps,
            conclusion=conclusion_content[:1000] + ("..." if len(conclusion_content) > 1000 else ""),
            reliability=reliability_score,
            rounds=rounds
        )
    
    def _parse_round_output(self, messages) -> dict:
        """先拿工具返回的 sources，不足就补 mock，保证每轮 ≥2 条"""
        sources, next_focus, stop_early = [], "", False

        # 1. 从 GoogleSearchTool 消息里直接拿搜索结果
        for msg in reversed(messages):
            if getattr(msg, 'name', None) == "GoogleSearchTool":
                logger.info(f"[parse] 抓到工具消息: {msg.content[:200]}...")
                try:
                    # 尝试解析为 JSON
                    sources = json.loads(msg.content)
                    logger.info(f"[parse] 解析后 sources={sources}")
                except json.JSONDecodeError:
                    # 如果是 Python 字典格式，尝试用 ast.literal_eval 解析
                    try:
                        import ast
                        sources = ast.literal_eval(msg.content)
                        logger.info(f"[parse] 使用 ast.literal_eval 解析成功: {sources}")
                    except (ValueError, SyntaxError):
                        logger.warning(f"[parse] 解析失败: {msg.content[:200]}")
                        sources = []
                except Exception as e:
                    logger.warning(f"[parse] 解析失败: {e}, content={msg.content[:200]}")
                    sources = []
                break

        # 2. 确保 sources 是列表格式
        if not isinstance(sources, list):
            sources = []

        # 3. 如果没有获取到有效数据，才使用兜底数据
        if len(sources) < 2:
            logger.warning(f"[parse] 搜索结果不足2条，当前有{len(sources)}条，使用补充数据")
            # 保留已有的有效数据
            valid_sources = sources[:]
            # 添加补充数据以达到至少2条
            for i in range(2 - len(sources)):
                valid_sources.append({
                    "title": f"补充来源 {i+1}",
                    "link": "https://example.com",
                    "snippet": "补充信息",
                    "date": "未知"
                })
            sources = valid_sources

        # 4. 无 next_focus → 自动生成（如果需要的话）
        if not next_focus:
            next_focus = "事件起源"

        return {"sources": sources, "next_focus": next_focus, "stop_early": False}
    async def analyze_stock_news(self, symbol: str, time_range: str = "w") -> dict:
        """
        分析股票相关新闻事件对股价的影响
        """
        logger.info(f"开始分析股票 {symbol} 的新闻事件，时间范围: {time_range}")
        
        # 构建搜索查询
        time_desc = {"w": "本周", "m": "本月", "y": "今年"}[time_range]
        search_query = f"股票 {symbol} {time_desc} 重大新闻 事件 影响分析"
        
        messages = [
            HumanMessage(content=f"""
你是一个专业的股票新闻分析师。请分析以下股票的相关新闻事件及其对股价的潜在影响：

股票代码: {symbol}
时间范围: {time_desc}

请按以下步骤操作：
1. 搜索该股票在指定时间范围内的重大新闻事件
2. 分析每个事件对股价的潜在影响（正面/负面/中性）
3. 评估新闻的可靠性
4. 预测事件可能对股价造成的短期和长期影响
5. 给出影响评分（-1到1，负数表示负面影响，正数表示正面影响）

请首先搜索相关股票新闻事件。
            """)
        ]
        
        news_events = []
        step_count = 0
        
        try:
            # 调用LLM并允许使用工具
            response_messages = await self.llm.forward(
                messages=messages,
                tools=[GoogleSearchTool]
            )
            
            current_messages = response_messages
            if isinstance(response_messages, list):
                if response_messages and isinstance(response_messages[0], list):
                    current_messages = response_messages[0]
            
            if isinstance(current_messages, list):
                messages.extend(current_messages)
            else:
                messages.append(current_messages)
            
            # 查找工具调用结果
            for msg in reversed(messages[-3:] if len(messages) >= 3 else messages):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_call = msg.tool_calls[0]
                    tool_name = tool_call.function.name
                    
                    if tool_name == "GoogleSearchTool":
                        tool_args_json = tool_call.function.arguments
                        tool_args = json.loads(tool_args_json)
                        query = tool_args.get("query", "")
                        
                        tool = GoogleSearchTool(query=query)
                        search_results = tool()
                        
                        if isinstance(search_results, list) and search_results:
                            for item in search_results:
                                if isinstance(item, dict) and "error" not in item:
                                    # 分析每个新闻对股价的潜在影响
                                    news_analysis = await self._analyze_news_impact(item, symbol)
                                    if news_analysis:
                                        news_events.append(news_analysis)
        
        except Exception as e:
            logger.error(f"分析股票新闻时出错: {e}")
            logger.exception(e)
        
        # 生成整体分析结论
        if news_events:
            overall_impact = self._calculate_overall_impact(news_events)
            return {
                "symbol": symbol,
                "time_range": time_range,
                "news_events": news_events,
                "impact_score": overall_impact["score"],
                "direction": overall_impact["direction"],
                "reliability": overall_impact["reliability"],
                "summary": f"在{time_desc}内发现{len(news_events)}个重要事件，综合影响评分: {overall_impact['score']:.2f}",
                "sentiment_trend": self._extract_sentiment_trend(news_events),
                "key_terms": self._extract_key_terms(news_events)
            }
        else:
            return {
                "symbol": symbol,
                "time_range": time_range,
                "news_events": [],
                "impact_score": 0.0,
                "direction": "neutral",
                "reliability": 0.5,
                "summary": f"在{time_desc}内未发现相关重大新闻事件",
                "sentiment_trend": [],
                "key_terms": {}
            }
    
    async def _analyze_news_impact(self, news_item: dict, symbol: str) -> Optional[dict]:
        """
        分析单个新闻事件对股价的潜在影响
        """
        try:
            # 使用LLM分析新闻对股价的影响
            analysis_prompt = f"""
请分析以下新闻对股票 {symbol} 的潜在影响：

新闻标题: {news_item.get('title', '')}
新闻摘要: {news_item.get('snippet', '')}
新闻链接: {news_item.get('link', '')}

请分析：
1. 该新闻事件的类型（财报、并购、政策、行业动态等）
2. 对股价的潜在影响（正面/负面/中性）
3. 影响强度评分（-1到1，负数表示负面影响，正数表示正面影响）
4. 影响的可信度评分（0-1）
5. 预期影响持续时间（短期/中期/长期）

请以结构化格式返回分析结果。
"""
            
            messages = [HumanMessage(content=analysis_prompt)]
            response = await self.llm.forward(messages=messages)
            
            if hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, list):
                content = str(response[0]) if response else "无法分析"
            else:
                content = str(response)
            
            # 这里可以进一步解析LLM的响应，提取结构化信息
            # 为了简化，这里直接返回分析结果
            return {
                "title": news_item.get('title', ''),
                "summary": news_item.get('snippet', ''),
                "date": news_item.get('date', '未知'),
                "link": news_item.get('link', ''),
                "sentiment": self._extract_sentiment(content),
                "impact_level": self._extract_impact_level(content),
                "analysis": content,
                "sources": [news_item]  # 保存原始来源
            }
            
        except Exception as e:
            logger.error(f"分析新闻影响时出错: {e}")
            return None
    
    def _calculate_overall_impact(self, news_events: List[dict]) -> dict:
        """
        计算整体影响评分
        """
        if not news_events:
            return {"score": 0.0, "direction": "neutral", "reliability": 0.5}
        
        total_score = 0
        total_reliability = 0
        count = 0
        
        for event in news_events:
            score = event.get('impact_score', 0.0)
            reliability = event.get('reliability', 0.5)
            
            total_score += score * reliability  # 根据可靠性加权
            total_reliability += reliability
            count += 1
        
        if count > 0:
            avg_score = total_score / total_reliability if total_reliability > 0 else 0
            avg_reliability = total_reliability / count
        else:
            avg_score = 0.0
            avg_reliability = 0.5
        
        # 确定方向
        if avg_score > 0.1:
            direction = "positive"
        elif avg_score < -0.1:
            direction = "negative"
        else:
            direction = "neutral"
        
        return {
            "score": avg_score,
            "direction": direction,
            "reliability": avg_reliability
        }
    
    def _extract_sentiment(self, content: str) -> str:
        """
        从内容中提取情感倾向
        """
        # 简单的情感倾向提取逻辑
        positive_keywords = ["积极", "利好", "上涨", "正面", "乐观", "增长", "收益", "成功"]
        negative_keywords = ["负面", "下跌", "风险", "亏损", "负面", "担忧", "下降", "危机"]
        
        content_lower = content.lower()
        positive_count = sum(1 for keyword in positive_keywords if keyword in content_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in content_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _extract_impact_level(self, content: str) -> str:
        """
        从内容中提取影响程度
        """
        # 简单的影响程度提取逻辑
        if any(keyword in content.lower() for keyword in ["重大", "重要", "显著", "巨大", "强烈"]):
            return "high"
        elif any(keyword in content.lower() for keyword in ["一定", "部分", "中等", "适度"]):
            return "medium"
        else:
            return "low"
    
    def _extract_sentiment_trend(self, news_events: List[dict]) -> List[dict]:
        """
        提取情感趋势数据
        """
        # 简单的情感趋势提取
        trend_data = []
        for event in news_events:
            trend_data.append({
                "date": event.get("date", "未知"),
                "sentiment": 1 if event.get("sentiment") == "positive" else (-1 if event.get("sentiment") == "negative" else 0),
                "title": event.get("title", "")[:30] + "..."
            })
        return trend_data
    
    def _extract_key_terms(self, news_events: List[dict]) -> Dict[str, int]:
        """
        提取关键词
        """
        # 简单的关键词提取
        all_text = " ".join([event.get("summary", "") for event in news_events])
        words = all_text.split()
        
        # 统计词频
        word_count = {}
        for word in words:
            # 过滤掉太短的词
            if len(word) > 2:
                word_count[word] = word_count.get(word, 0) + 1
        
        # 返回最常见的10个词
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_words[:10])

def extract_features_from_news(steps: List[TraceStep]) -> Dict[str, Any]:
    """
    从新闻溯源步骤中提取特征
    """
    features = {
        "sentiment_score": 0.0,
        "reliability_score": 0.0,
        "news_volume": 0,
        "news_diversity": 0,
        "key_themes": [],
        "event_types": [],
        "source_trustworthiness": 0.0
    }
    
    if not steps:
        return features
    
    # 计算新闻数量
    total_sources = sum(len(step.sources) for step in steps)
    features["news_volume"] = total_sources
    
    # 提取主题和事件类型
    themes = set()
    event_types = set()
    
    for step in steps:
        for source in step.sources:
            # 这里可以进一步分析新闻内容，提取主题和事件类型
            # 暂时简化处理
            themes.add(step.query)
            event_types.add("general_news")  # 可以根据内容进行更详细的分类
    
    features["key_themes"] = list(themes)
    features["event_types"] = list(event_types)
    features["news_diversity"] = len(themes)
    
    return features

def extract_signals_from_news(steps: List[TraceStep]) -> List[Dict[str, Any]]:
    """
    从新闻溯源步骤中提取交易信号
    """
    signals = []
    
    for step in steps:
        for source in step.sources:
            # 简单的信号提取逻辑
            signal = {
                "date": source.date,
                "title": source.title,
                "sentiment": analyze_sentiment(source.snippet),
                "confidence": 0.5,  # 可以根据来源可靠性调整
                "type": "news_event"
            }
            signals.append(signal)
    
    return signals

def analyze_sentiment(text: str) -> float:
    """
    简单的情感分析
    返回 -1 到 1 的值，负数表示负面，正数表示正面
    """
    positive_keywords = ["positive", "good", "great", "excellent", "increase", "up", "rise", "gain", "profit", "success"]
    negative_keywords = ["negative", "bad", "terrible", "awful", "decrease", "down", "fall", "loss", "decline", "crisis"]
    
    text_lower = text.lower()
    positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    
    total_count = positive_count + negative_count
    if total_count == 0:
        return 0.0
    
    return (positive_count - negative_count) / total_count

def extract_reliability_from_text(text: str) -> float:
    """
    从文本中提取可靠性评分，支持格式如：
    - 可靠性评分：0.8
    - 可靠性评分: 0.8
    - 可靠性评分：0.8 (高)
    - 可靠性评分：80%
    """
    import re
    
    # 匹配模式：可靠性评分：0.8 或 可靠性评分: 0.8 或 80%
    pattern = r'(?:可靠性评分[:：]\s*)?([0-1]\.\d+)|(\d+%)'
    
    match = re.search(pattern, text)
    if match:
        score_str = match.group(1) or match.group(2)
        if '%' in score_str:
            score = float(score_str.rstrip('%')) / 100
        else:
            score = float(score_str)
        return max(0.0, min(1.0, score))
    else:
        # 默认评分
        return 0.5

# 全局变量存储LLM实例
llm_instance = None
tracer_agent = None

@api_router.on_event("startup")
async def startup_event():
    global llm_instance, tracer_agent
    logger.info("开始初始化LLM实例...")
    print("=== 开始初始化LLM实例 (startup event) ===")
    
    try:
        # 验证环境变量
        api_key = os.getenv("ARK_API_KEY")
        base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        
        print(f"API Key: {api_key}")
        print(f"Base URL: {base_url}")
        
        if not api_key:
            error_msg = "ARK_API_KEY 环境变量未设置"
            logger.error(error_msg)
            print(f"错误: {error_msg}")
            return
        
        # 创建LLM实例
        args = OpenAIArgs(
            model="doubao/doubao-1-5-pro-32k-250115",
            api_key=api_key,
            base_url=base_url,
            stream=False,
            verbose=False
        )
        
        print(f"LLM参数: {args}")
        
        llm_instance = OpenRouterLLM(args=args)
        tracer_agent = NewsTracingAgent(llm_instance)
        
        logger.info("LLM实例初始化成功")
        print("LLM实例初始化成功")
        
    except Exception as e:
        logger.error(f"LLM实例初始化失败: {e}")
        logger.exception(e)
        print(f"LLM实例初始化失败: {e}")

@api_router.get("/")
async def root():
    return {"message": "新闻溯源API已启动", "status": "running"}

@api_router.get("/health")
async def health_check():
    global llm_instance
    status = "healthy" if llm_instance else "unhealthy"
    return {"status": status, "model": "doubao/doubao-1-5-pro-32k-250115"}

@api_router.post("/trace", response_model=TraceResponse)
async def trace_claim_endpoint(request: TraceRequest):
    """
    对新闻声明进行溯源验证
    """
    print(f"收到trace请求: {request}")
    
    if not tracer_agent or not llm_instance:
        logger.error("LLM实例未初始化")
        print("错误: LLM实例未初始化")
        raise HTTPException(status_code=500, detail="LLM实例未初始化，请检查环境变量配置")
    
    try:
        logger.info(f"接收到验证请求: {request.claim}")
        result = await tracer_agent.trace_claim(request.claim, request.max_depth or 2)
        logger.info(f"返回验证结果: {result.reliability}")
        return result
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/recent")
async def get_recent_verifications(limit: int = 10):
    """
    获取最近的验证记录
    """
    # 这里可以连接数据库获取历史记录
    # 目前返回空列表，后续可以扩展
    return []

@api_router.post("/causal-analysis")
async def causal_analysis(request: TraceRequest):
    """
    因果分析端点
    """
    if not tracer_agent or not llm_instance:
        raise HTTPException(status_code=500, detail="LLM实例未初始化")
    
    try:
        result = await tracer_agent.trace_claim(request.claim, request.max_depth or 2)
        
        # 添加因果关系数据
        causal_data = {
            "claim": request.claim,
            "nodes": [],
            "edges": [],
            "analysis": "因果关系分析"
        }
        
        # 构建节点和边
        causal_data["nodes"].append({"id": "claim", "label": request.claim, "type": "claim"})
        
        for i, step in enumerate(result.steps):
            step_id = f"step_{i+1}"
            causal_data["nodes"].append({"id": step_id, "label": step.query, "type": "step"})
            causal_data["edges"].append({"from": "claim", "to": step_id})
            
            for j, source in enumerate(step.sources[:2]):
                source_id = f"source_{i+1}_{j+1}"
                causal_data["nodes"].append({"id": source_id, "label": source.title[:30]+"...", "type": "source"})
                causal_data["edges"].append({"from": step_id, "to": source_id})
        
        return {"verification": result, "causal": causal_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

print("=== news_tracer_api 模块加载完成 ===")

# ========== 公司名 → 股票代码 ==========
class Name2CodeResp(BaseModel):
    symbol: str
    candidates: List[str]

@api_router.post("/stock/code", response_model=Name2CodeResp)
async def name_to_code(request: dict):
    name = request.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name 不能为空")

    tool = GoogleSearchTool(query=f"{name} 股票代码 ticker symbol")
    res = tool()
    if isinstance(res, dict) and "error" in res:
        raise HTTPException(status_code=500, detail=res["error"])

    import re
    codes = set()
    for item in res:
        t = item["title"] + " " + item["snippet"]
        codes.update(re.findall(r"\b(\d{4}\.HK|[A-Z]{2,5})\b", t))
    if not codes:
        raise HTTPException(status_code=404, detail="未解析到股票代码")
    return Name2CodeResp(symbol=name, candidates=list(codes))

# ====== 新增：策略报告接口 ======
@api_router.post("/stock/strategy_report")
async def strategy_report(request: dict):
    """
    结合新闻溯源 Agent 生成策略报告
    """
    symbol = request.get("symbol", "")
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol 不能为空")

    try:
        # 1. 调用你已写好的 agent 分析
        news_result = await tracer_agent.analyze_stock_news(symbol, time_range="m")
        # 2. 组装成前端易用的结构
        return {
            "symbol": symbol,
            "news_events": news_result["news_events"],
            "impact_score": news_result["impact_score"],
            "direction": news_result["direction"],
            "reliability": news_result["reliability"],
            "summary": news_result["summary"],
            "sentiment_trend": news_result["sentiment_trend"],
            "key_terms": news_result["key_terms"],
            "strategy_signal": _map_signal(news_result["impact_score"], news_result["direction"])
        }
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

def _map_signal(score: float, direction: str) -> str:
    if direction == "positive" and score > 0.3:
        return "强烈看涨（新闻驱动）"
    if direction == "negative" and score < -0.3:
        return "强烈看跌（新闻驱动）"
    if abs(score) < 0.1:
        return "中性（新闻无显著影响）"
    return "观望（新闻情绪微弱）"

# 工具：把 impact 映射成交易信号
def _map_signal(score: float, direction: str) -> str:
    if direction == "positive" and score > 0.3:
        return "强烈看涨（新闻驱动）"
    if direction == "negative" and score < -0.3:
        return "强烈看跌（新闻驱动）"
    if abs(score) < 0.1:
        return "中性（新闻无显著影响）"
    return "观望（新闻情绪微弱）"

class StrategyRecommendRequest(BaseModel):
    symbol: str
    time_range: str = "1y"   # 行情长度
    preference: str = "balanced"  # aggressive / balanced / conservative


class StrategyRecommendResponse(BaseModel):
    strategy: str              # 策略英文关键字
    params: Dict[str, Any]     # 超参字典
    reason: str                # Agent 推荐理由
    
    
@api_router.post("/stock/strategy_recommend", response_model=StrategyRecommendResponse)
async def strategy_recommend(request: StrategyRecommendRequest):
    """
    由 NewsTracingAgent 读取行情 + 新闻后，推荐最适合的策略及超参
    """
    if not tracer_agent or not llm_instance:
        raise HTTPException(status_code=500, detail="LLM 未初始化")

    # 1. 拉行情（复用已有模块）
    backtester = StrategyBacktester()
    df = backtester.fetch_data(request.symbol, period=request.time_range)
    if df.empty:
        raise HTTPException(status_code=404, detail="行情获取失败")

    # 2. 通过新闻溯源Agent拿近期情绪
    news_result = await tracer_agent.analyze_stock_news(request.symbol, time_range="m")
    impact_score = news_result.get("impact_score", 0.0)
    direction = news_result.get("direction", "neutral")

    # 3. 让 Agent 生成策略推荐
    prompt = f"""
你是量化策略专家。已知股票 {request.symbol} 最近 {request.time_range} 行情已提取，
且近一月新闻情绪评分={impact_score:.2f}，方向={direction}，用户偏好={request.preference}。

请返回 JSON：
{{
  "strategy": "ma_crossover" | "rsi_mean_revert" | "bollinger_breakout",
  "params": {{ 具体超参 }},
  "reason": "一句话推荐理由"
}}

要求：
- 情绪强烈+偏好激进 → 可选 breakout 或短周期均线
- 情绪平稳+偏好保守 → 选 rsi_mean_revert 或长周期均线
- 给出可解释的超参（均线窗口、RSI 阈值等）
"""
    messages = [HumanMessage(content=prompt)]
    raw = await llm_instance.forward(messages)
    content = raw[0].content if isinstance(raw, list) else raw.content
    try:
        payload = json.loads(content)
    except:
        payload = {"strategy": "ma_crossover", "params": {}, "reason": "Agent 解析失败，默认均线交叉"}

    return StrategyRecommendResponse(**payload)