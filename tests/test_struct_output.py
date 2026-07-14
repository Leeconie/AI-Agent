# improved_structured_output.py
import asyncio
import os
import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field
import tyro
from loguru import logger

from mira import LLMJson, OpenAIArgs, OpenRouterLLM, HumanMessage

class CalculationResponse(LLMJson):
    """计算任务响应格式"""
    expression: str = Field(..., description="数学表达式")
    steps: List[str] = Field(..., description="计算步骤")
    result: float = Field(..., description="计算结果")
    explanation: str = Field(..., description="计算过程解释")

class ResearchResponse(LLMJson):
    """研究任务响应格式"""
    topic: str = Field(..., description="研究主题")
    summary: str = Field(..., description="主题摘要")
    key_points: List[str] = Field(..., description="关键点")
    sources: List[str] = Field(..., description="信息来源")

class NewsSource(BaseModel):
    """新闻来源"""
    title: str = Field(..., description="新闻标题")
    url: str = Field(..., description="新闻链接")
    snippet: str = Field(..., description="新闻摘要")

class NewsAnalysisResponse(LLMJson):
    """新闻分析响应格式"""
    claim: str = Field(..., description="分析的声明")
    fact_check: str = Field(..., description="事实核查结果")
    sources: List[NewsSource] = Field(..., description="引用来源")
    conclusion: str = Field(..., description="结论")
    confidence: float = Field(..., description="置信度 (0-1)", ge=0, le=1)

async def parse_structured_response(response, response_model):
    """
    解析结构化响应
    """
    try:
        # 提取响应内容
        content = ""
        if isinstance(response, list):
            # 如果是嵌套列表，展开它
            if len(response) > 0 and isinstance(response[0], list):
                response = response[0]
            
            # 遍历列表找到内容
            for item in response:
                if hasattr(item, 'content'):
                    content = item.content
                    break
                else:
                    content = str(item)
        else:
            content = response.content if hasattr(response, 'content') else str(response)
        
        logger.debug(f"提取的内容: {content}")
        
        # 如果内容是repr形式的AIMessage，尝试从中提取JSON
        if content.startswith('[AIMessage(') or '"content":' in content:
            # 尝试提取content字段中的JSON
            match = re.search(r'"content":\s*"({.*?})"', content)
            if match:
                # 解码转义字符
                content = match.group(1).encode().decode('unicode_escape')
            else:
                # 尝试提取花括号内的内容
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    content = match.group(0)
        
        # 清理内容
        content = content.strip()
        
        # 移除可能的 markdown 代码块标记
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        # 进一步清理可能的转义字符
        content = content.replace('\\n', '\n').replace('\\"', '"')
        
        logger.debug(f"清理后的内容: {content}")
        
        # 解析JSON并验证是否符合模型
        data = json.loads(content)
        return response_model(**data)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {e}")
        logger.error(f"尝试解析的内容: {content}")
        raise e
    except Exception as e:
        logger.error(f"解析结构化响应时出错: {e}")
        logger.error(f"原始内容: {content}")
        raise e

async def test_calculation(args: OpenAIArgs):
    """测试计算任务的结构化输出"""
    llm = OpenRouterLLM(args=args)
    
    message = HumanMessage(content="""请计算 123 * 456 + 789 的结果，并按照以下JSON格式返回结果：
{
  "expression": "数学表达式",
  "steps": ["步骤1", "步骤2", "..."],
  "result": 计算结果,
  "explanation": "计算过程解释"
}
请只返回JSON，不要包含其他文字，不要使用代码块标记。""")
    
    try:
        response = await llm.forward(messages=[message])
        structured_result = await parse_structured_response(response, CalculationResponse)
        
        logger.info("=== 计算任务结构化输出 ===")
        logger.info(f"表达式: {structured_result.expression}")
        logger.info(f"结果: {structured_result.result}")
        logger.info(f"解释: {structured_result.explanation}")
        logger.info("步骤:")
        for i, step in enumerate(structured_result.steps, 1):
            logger.info(f"  {i}. {step}")
            
        return structured_result
            
    except Exception as e:
        logger.error(f"处理计算任务时出错: {e}")
        return None

async def test_research(args: OpenAIArgs):
    """测试研究任务的结构化输出"""
    llm = OpenRouterLLM(args=args)
    
    message = HumanMessage(content="""请研究人工智能在医疗领域的应用，并按照以下JSON格式返回结果：
{
  "topic": "研究主题",
  "summary": "主题摘要",
  "key_points": ["关键点1", "关键点2", "..."],
  "sources": ["来源1", "来源2", "..."]
}
请只返回JSON，不要包含其他文字，不要使用代码块标记。""")
    
    try:
        response = await llm.forward(messages=[message])
        structured_result = await parse_structured_response(response, ResearchResponse)
        
        logger.info("=== 研究任务结构化输出 ===")
        logger.info(f"主题: {structured_result.topic}")
        logger.info(f"摘要: {structured_result.summary}")
        logger.info("关键点:")
        for i, point in enumerate(structured_result.key_points, 1):
            logger.info(f"  {i}. {point}")
        logger.info("来源:")
        for i, source in enumerate(structured_result.sources, 1):
            logger.info(f"  {i}. {source}")
            
        return structured_result
            
    except Exception as e:
        logger.error(f"处理研究任务时出错: {e}")
        return None

async def main(args: OpenAIArgs):
    """主函数"""
    args.model = "doubao/doubao-1-5-pro-32k-250115"
    args.stream = False
    args.verbose = False
    
    logger.info(f"使用模型: {args.model}")
    
    # 测试计算任务
    logger.info("=== 测试计算任务 ===")
    result1 = await test_calculation(args)
    
    logger.info("\n" + "="*50 + "\n")
    
    # 测试研究任务
    logger.info("=== 测试研究任务 ===")
    result2 = await test_research(args)
    
    return [result1, result2]

if __name__ == "__main__":
    args = tyro.cli(OpenAIArgs)
    args.api_key = os.getenv("ARK_API_KEY") or os.getenv("ONEAPI_API_KEY")
    args.base_url = os.getenv("ARK_BASE_URL") or os.getenv("ONEAPI_BASE_URL")
    
    logger.info(f"API配置完成")
    
    asyncio.run(main(args))