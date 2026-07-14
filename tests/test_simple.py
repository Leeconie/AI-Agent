import asyncio
import os

import tyro
from loguru import logger
from pydantic import Field
import httpx

from mira import HumanMessage, LLMTool, OpenAIArgs, OpenRouterLLM

prompt = """
介绍一下“爬虫”这个概念，并使用谷歌搜索工具查找最新的相关信息，然后总结搜索结果。
"""


class GoogleSearchTool(LLMTool):
    """Search Google for information"""

    query: str = Field(..., description="the search query")

    def __call__(self):
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return "SERPER_API_KEY not set"

        url = f"https://google.serper.dev/search?q={self.query}&apiKey={api_key}"
        response = httpx.get(url)
        if response.status_code == 200:
            data = response.json()
            results = data.get("organic", [])
            # Return top 5 results
            return [{"title": r.get("title"), "link": r.get("link"), "snippet": r.get("snippet")} for r in results[:5]]
        else:
            return f"Error: {response.status_code}"


async def main(args: OpenAIArgs):
    llm = OpenRouterLLM(args=args)
    m = [HumanMessage(content=prompt)]

    while True:
        dm = await llm.forward(messages=m, tools=[GoogleSearchTool])
        m = m + dm[0]
        if not dm[0] or not hasattr(dm[0][-1], 'tool_calls') or not dm[0][-1].tool_calls:
            break

    logger.info(m)


if __name__ == "__main__":
    args = tyro.cli(OpenAIArgs)
    args.model = "doubao/doubao-1-5-pro-32k-250115"
    args.api_key = os.getenv("ARK_API_KEY")
    args.base_url = os.getenv("ARK_BASE_URL")
    args.stream = False
    args.verbose = False
    logger.info(args)

    asyncio.run(main(args))