import asyncio
import json
import os
from typing import List
from loguru import logger

from mira import LLMTool, LLMJson, OpenAIArgs, OpenRouterLLM, HumanMessage, ToolMessage

class ColdJokeGeneratorAgent:
    def __init__(self, llm: OpenRouterLLM):
        self.llm = llm

    async def generate_cold_joke_analysis(self, joke_text: str) -> dict:
        """
        使用agent分析冷笑话的幽默原因
        
        Args:
            joke_text: 冷笑话文本
            
        Returns:
            包含instruction, input, output的字典
        """
        prompt = f"""
请阅读以下文字，分析其幽默的原因。幽默性指该文本是否可能引起读者发笑，或通过语言技巧（如双关语、讽刺、夸张、荒诞或逻辑上的意外等等方式）营造幽默效果。请你写出以下文字幽默的原因：

笑话内容：{joke_text}
        """
        
        messages = [
            HumanMessage(content=prompt)
        ]
        
        try:
            response_messages = await self.llm.forward(messages=messages)
            
            # 处理响应
            if isinstance(response_messages, list):
                if response_messages and isinstance(response_messages[0], list):
                    response_content = response_messages[0][0].content if hasattr(response_messages[0][0], 'content') else str(response_messages[0][0])
                else:
                    response_content = response_messages[0].content if hasattr(response_messages[0], 'content') else str(response_messages[0])
            elif hasattr(response_messages, 'content'):
                response_content = response_messages.content
            else:
                response_content = str(response_messages)
            
            # 构造结果
            result = {
                "instruction": "请阅读以下文字，分析其幽默的原因。幽默性指该文本是否可能引起读者发笑，或通过语言技巧（如双关语、讽刺、夸张、荒诞或逻辑上的意外等等方式）营造幽默效果。请你写出以下文字幽默的原因：",
                "input": joke_text,
                "output": response_content
            }
            
            return result
            
        except Exception as e:
            logger.error(f"生成冷笑话分析时出错: {e}")
            # 返回一个默认结果
            return {
                "instruction": "请阅读以下文字，分析其幽默的原因。幽默性指该文本是否可能引起读者发笑，或通过语言技巧（如双关语、讽刺、夸张、荒诞或逻辑上的意外等等方式）营造幽默效果。请你写出以下文字幽默的原因：",
                "input": joke_text,
                "output": f"分析过程中出现错误: {str(e)}"
            }

    async def generate_dialogue_expansion(self, joke_text: str) -> dict:
        """
        以input为最后一句话，往前扩展出3-10轮对话
        
        Args:
            joke_text: 冷笑话文本（作为对话的最后一句话）
            
        Returns:
            包含chat字段的字典（符合jsonl格式）
        """
        # 如果joke_text包含"标题："，则只取标题后的部分作为最后一句话
        if "标题：" in joke_text:
            final_sentence = joke_text.split("标题：", 1)[1]
        else:
            final_sentence = joke_text

        prompt = f"""
请以以下句子为对话的最后一句话，向前扩展出3-10轮自然流畅的对话，要求逻辑自然流畅，句子之间内容相关自然，不要跳话题：

最后一句话：{final_sentence}

请生成一个完整的对话，包含多轮交流，最后一句话必须是"{final_sentence}"，前面的对话要与这个笑话自然衔接。
请严格按照JSON格式输出，只输出对话内容，不要添加称呼（如A、B等）或其他说明。
输出格式应该如下：
{{"chat": ["第一句话","第二句话","第三句话","...","{final_sentence}"]}}

注意：最后一句话必须是"{final_sentence}"，不要包含"标题："这些字。
        """
        
        messages = [
            HumanMessage(content=prompt)
        ]
        
        try:
            response_messages = await self.llm.forward(messages=messages)
            
            # 处理响应
            if isinstance(response_messages, list):
                if response_messages and isinstance(response_messages[0], list):
                    response_content = response_messages[0][0].content if hasattr(response_messages[0][0], 'content') else str(response_messages[0][0])
                else:
                    response_content = response_messages[0].content if hasattr(response_messages[0], 'content') else str(response_messages[0])
            elif hasattr(response_messages, 'content'):
                response_content = response_messages.content
            else:
                response_content = str(response_messages)
            
            # 尝试解析JSON格式的响应
            try:
                # 如果模型直接输出了JSON格式，尝试解析
                result = json.loads(response_content)
                if 'chat' in result:
                    chat_list = result['chat']
                    # 确保最后一句话是目标句子
                    if chat_list and chat_list[-1] != final_sentence:
                        if final_sentence not in chat_list:
                            chat_list.append(final_sentence)
                        else:
                            chat_list.remove(final_sentence)
                            chat_list.append(final_sentence)
                    return {"chat": chat_list}
            except json.JSONDecodeError:
                # 如果不是JSON格式，则按提示的格式处理
                # 查找chat数组的开始和结束
                import re
                # 尝试匹配 "chat": [...] 格式
                chat_match = re.search(r'"chat"\s*:\s*\[(.*?)\]', response_content, re.DOTALL)
                if chat_match:
                    chat_content = chat_match.group(1)
                    # 分割并清理每个对话项
                    items = [item.strip().strip('"').strip("'").strip(',') for item in chat_content.split(',')]
                    cleaned_items = []
                    for item in items:
                        # 去除可能的多余空格和引号
                        cleaned_item = item.strip().strip('"').strip("'").strip(',')
                        if cleaned_item:
                            cleaned_items.append(cleaned_item)
                    
                    # 确保最后一句话是目标句子
                    if cleaned_items and cleaned_items[-1] != final_sentence:
                        if final_sentence not in cleaned_items:
                            cleaned_items.append(final_sentence)
                        else:
                            cleaned_items.remove(final_sentence)
                            cleaned_items.append(final_sentence)
                    
                    return {"chat": cleaned_items}
                else:
                    # 如果没有找到chat数组格式，尝试其他方式解析
                    lines = [line.strip().strip('"').strip("'") for line in response_content.strip().split('\n') if line.strip() and not line.strip().startswith('{') and not line.strip().startswith('}') and not line.strip().startswith('"chat"')]
                    
                    # 过滤掉可能的编号或格式
                    cleaned_lines = []
                    for line in lines:
                        import re
                        cleaned_line = re.sub(r'^[A-Z][A-Z]*：?', '', line).strip()
                        cleaned_line = re.sub(r'^\d+\.?\s*', '', cleaned_line).strip()
                        cleaned_line = cleaned_line.strip().strip('"').strip("'").strip(',')
                        if cleaned_line and cleaned_line != '"chat":' and cleaned_line != '[' and cleaned_line != ']' and '}' not in cleaned_line and '{' not in cleaned_line:
                            cleaned_lines.append(cleaned_line)
                    
                    # 确保最后一句话是目标句子
                    if cleaned_lines and cleaned_lines[-1] != final_sentence:
                        if final_sentence not in cleaned_lines:
                            cleaned_lines.append(final_sentence)
                        else:
                            cleaned_lines.remove(final_sentence)
                            cleaned_lines.append(final_sentence)
                    
                    return {"chat": cleaned_lines}
            
        except Exception as e:
            logger.error(f"生成对话扩展时出错: {e}")
            # 返回一个默认结果
            return {
                "chat": [f"生成对话时出现错误: {str(e)}", final_sentence]
            }

    async def generate_multiple_cold_jokes(self, jokes_list: List[str]) -> List[dict]:
        """
        批量生成冷笑话分析
        
        Args:
            jokes_list: 冷笑话列表
            
        Returns:
            包含多个冷笑话分析的列表
        """
        results = []
        for i, joke in enumerate(jokes_list):
            logger.info(f"处理第 {i+1}/{len(jokes_list)} 个笑话")
            result = await self.generate_cold_joke_analysis(joke)
            results.append(result)
        return results

    async def generate_multiple_dialogue_expansions(self, jokes_list: List[str]) -> List[dict]:
        """
        批量生成对话扩展
        
        Args:
            jokes_list: 冷笑话列表（作为对话的最后一句话）
            
        Returns:
            包含多个对话扩展的列表
        """
        results = []
        for i, joke in enumerate(jokes_list):
            logger.info(f"生成第 {i+1}/{len(jokes_list)} 个对话扩展")
            result = await self.generate_dialogue_expansion(joke)
            results.append(result)
        return results


async def main():
    """主函数"""
    # 创建LLM实例
    args = OpenAIArgs(
        model="doubao/doubao-seed-1-6-lite-251015",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        stream=False,
        verbose=False
    )
    
    logger.info(f"使用模型: {args.model}")
    
    if not args.api_key:
        logger.error("错误: 未设置 ARK_API_KEY 环境变量")
        return
    
    # 创建agent实例
    llm = OpenRouterLLM(args=args)
    agent = ColdJokeGeneratorAgent(llm)
    
    # 从现有的JSON文件读取冷笑话数据
    input_json_file = "你的冷笑话数据集.json"
    
    try:
        with open(input_json_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        # 提取所有的input字段作为笑话文本
        jokes_list = []
        for item in existing_data:
            if 'input' in item:
                jokes_list.append(item['input'])
        
        logger.info(f"从 {input_json_file} 读取了 {len(jokes_list)} 个冷笑话")
        
        # 使用agent生成对话扩展
        logger.info("开始使用agent生成对话扩展...")
        results = await agent.generate_multiple_dialogue_expansions(jokes_list[:2048])  # 限制为前3个以节省时间
        
        # 保存为JSONL格式（符合要求的格式）
        output_jsonl_file = ".\mira\output\CFunSet_dialogue_expansions_agent.jsonl"
        with open(output_jsonl_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"对话扩展已保存到: {output_jsonl_file}")
        
        # 也可以保存为JSON格式
        output_json_file = "CFunSet_dialogue_expansions_agent.json"
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"对话扩展已保存到: {output_json_file}")
        
    except FileNotFoundError:
        logger.error(f"找不到文件: {input_json_file}")
        
        # 如果找不到现有文件，可以使用示例笑话
        sample_jokes = [
            "标题：体重秤是免费的查重机器。",
            "标题：比起段子，应该还是我的论文更加搞笑一点。",
            "标题：少壮不努力，老大肯德基。"
        ]
        
        logger.info("使用示例笑话进行测试...")
        results = await agent.generate_multiple_dialogue_expansions(sample_jokes)
        
        # 保存为JSONL格式
        output_jsonl_file = "sample_dialogue_expansions_agent.jsonl"
        with open(output_jsonl_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"示例对话扩展已保存到: {output_jsonl_file}")


if __name__ == "__main__":
    asyncio.run(main())