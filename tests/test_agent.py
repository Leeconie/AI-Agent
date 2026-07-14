# news_tracing_agent_final.py
import asyncio
import os
import json
import httpx
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import tyro
from loguru import logger

from mira import LLMTool, LLMJson, OpenAIArgs, OpenRouterLLM, HumanMessage, ToolMessage

# 定义工具
class GoogleSearchTool(LLMTool):
    """Google搜索工具"""
    query: str = Field(..., description="搜索查询词")
    
    def __call__(self):
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return {"error": "SERPER_API_KEY 环境变量未设置"}
        
        try:
            url = "https://google.serper.dev/search"
            headers = {"X-API-KEY": api_key}
            payload = {"q": self.query, "num": 3}
            
            response = httpx.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                organic_results = data.get("organic", [])
                results = []
                for item in organic_results[:3]:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "date": item.get("date", "未知")
                    })
                return results
            else:
                return {"error": f"搜索API错误: {response.status_code}"}
        except Exception as e:
            return {"error": f"搜索失败: {str(e)}"}

# 定义结构化输出模型
class NewsSource(BaseModel):
    """新闻来源"""
    title: str = Field(..., description="新闻标题")
    link: str = Field(..., description="新闻链接")
    snippet: str = Field(..., description="新闻摘要")
    date: str = Field(..., description="发布日期")

class TraceStep(BaseModel):
    """溯源步骤"""
    step: int = Field(..., description="步骤编号")
    query: str = Field(..., description="搜索查询")
    sources: List[NewsSource] = Field(..., description="找到的来源")

class NewsTraceResult(LLMJson):
    """新闻溯源结果"""
    claim: str = Field(..., description="被溯源的声明")
    steps: List[TraceStep] = Field(..., description="溯源步骤")
    conclusion: str = Field(..., description="溯源结论")
    reliability: float = Field(..., description="可靠性评分 (0-1)", ge=0, le=1)

# 新闻溯源Agent
class NewsTracingAgent:
    def __init__(self, llm: OpenRouterLLM):
        self.llm = llm
    
    async def trace_claim(self, claim: str, max_depth: int = 2) -> NewsTraceResult:
        """
        对新闻声明进行溯源
        
        Args:
            claim: 要溯源的声明
            max_depth: 最大溯源深度
            
        Returns:
            NewsTraceResult: 溯源结果
        """
        messages = [
            HumanMessage(content=f"""
你是一个专业的新闻事实核查员。请对以下声明进行验证：

声明: "{claim}"

请按以下步骤操作：
1. 构造适当的搜索关键词来验证此声明
2. 使用GoogleSearchTool工具执行搜索
3. 分析搜索结果并得出结论
4. 给出0-1之间的可靠性评分

请首先构造搜索关键词并调用GoogleSearchTool工具。
            """)
        ]
        
        trace_steps = []
        step_count = 0
        
        # 工具调用循环
        for depth in range(max_depth):
            logger.info(f"执行溯源步骤 {depth + 1}")
            
            try:
                # 调用LLM并允许使用工具
                response_messages = await self.llm.forward(
                    messages=messages,
                    tools=[GoogleSearchTool]
                )
                
                # 处理响应
                current_messages = response_messages
                if isinstance(response_messages, list):
                    if response_messages and isinstance(response_messages[0], list):
                        current_messages = response_messages[0]
                
                # 添加到消息历史
                if isinstance(current_messages, list):
                    messages.extend(current_messages)
                else:
                    messages.append(current_messages)
                
                # 检查工具调用
                tool_called = False
                # 检查最后几条消息中是否有工具调用
                for msg in reversed(messages[-3:] if len(messages) >= 3 else messages):
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_call = msg.tool_calls[0]
                        tool_name = tool_call.function.name
                        tool_args_json = tool_call.function.arguments
                        
                        logger.info(f"调用工具: {tool_name}")
                        
                        try:
                            tool_args = json.loads(tool_args_json)
                            
                            if tool_name == "GoogleSearchTool":
                                query = tool_args.get("query", "")
                                logger.info(f"执行搜索: {query}")
                                
                                tool = GoogleSearchTool(query=query)
                                search_results = tool()
                                
                                tool_message = ToolMessage(
                                    content=json.dumps(search_results, ensure_ascii=False),
                                    tool_call_id=tool_call.id,
                                    name="GoogleSearchTool"
                                )
                                messages.append(tool_message)
                                
                                # 记录步骤
                                if isinstance(search_results, list) and search_results:
                                    try:
                                        sources = [NewsSource(**item) for item in search_results if isinstance(item, dict) and "error" not in item]
                                        if sources:
                                            step = TraceStep(
                                                step=step_count + 1,
                                                query=query,
                                                sources=sources
                                            )
                                            trace_steps.append(step)
                                            step_count += 1
                                    except Exception as e:
                                        logger.error(f"处理搜索结果时出错: {e}")
                                
                                tool_called = True
                                break
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"解析工具参数时出错: {e}")
                        except Exception as e:
                            logger.error(f"执行工具时出错: {e}")
                
                if not tool_called:
                    logger.info("未检测到工具调用，结束工具调用循环")
                    break
                    
            except Exception as e:
                logger.error(f"溯源步骤 {depth + 1} 出错: {e}")
                break
        
        # 生成最终结论
        conclusion_content = "未能生成结论"
        try:
            conclusion_messages = messages + [HumanMessage(content=f"""
基于以上搜索结果，请对声明"{claim}"进行总结：
1. 声明是否属实？
2. 主要证据是什么？
3. 可靠性评分（0-1）？

请简洁明了地回答。
            """)]
            
            final_response = await self.llm.forward(messages=conclusion_messages)
            if isinstance(final_response, list) and final_response:
                final_msg = final_response[0] if not isinstance(final_response[0], list) else final_response[0][0]
                conclusion_content = getattr(final_msg, 'content', str(final_msg))
            elif hasattr(final_response, 'content'):
                conclusion_content = final_response.content
        except Exception as e:
            logger.error(f"生成结论时出错: {e}")
        
        result = NewsTraceResult(
            claim=claim,
            steps=trace_steps,
            conclusion=conclusion_content[:1000] + ("..." if len(conclusion_content) > 1000 else ""),
            reliability=0.75
        )
        
        return result

async def main():
    """主函数"""
    # 直接创建带有正确参数的OpenAIArgs实例
    args = OpenAIArgs(
        model="doubao/doubao-seed-1-6-lite-251015",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        stream=False,
        verbose=False
    )
    
    logger.info(f"使用模型: {args.model}")
    logger.info(f"Base URL: {args.base_url}")
    
    # 检查环境变量
    if not os.getenv("SERPER_API_KEY"):
        logger.warning("警告: SERPER_API_KEY 环境变量未设置")
    
    if not args.api_key:
        logger.error("错误: 未设置 ARK_API_KEY 环境变量")
        return
    
    llm = OpenRouterLLM(args=args)
    agent = NewsTracingAgent(llm)
    
    # 测试声明
    test_claim = "人工智能将在未来五年内取代大部分程序员的工作"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"开始溯源声明: {test_claim}")
    logger.info('='*60)
    
    try:
        result = await agent.trace_claim(test_claim, max_depth=2)
        
        logger.info("溯源结果:")
        logger.info(f"声明: {result.claim}")
        logger.info(f"可靠性评分: {result.reliability:.2f}")
        
        logger.info("溯源步骤:")
        if result.steps:
            for step in result.steps:
                logger.info(f"  步骤 {step.step}: 搜索 '{step.query}'")
                for i, source in enumerate(step.sources, 1):
                    logger.info(f"    来源 {i}: {source.title}")
                    logger.info(f"      链接: {source.link}")
                    logger.info(f"      摘要: {source.snippet[:100]}...")
        else:
            logger.info("  未执行搜索步骤")
        
        logger.info(f"结论: {result.conclusion}")
        
    except Exception as e:
        logger.error(f"处理声明时出错: {e}")
        logger.exception(e)

if __name__ == "__main__":
    asyncio.run(main())