# Mira 项目完整文档

## 项目简介

**Mira** 是一个整合了部署、请求、上下文管理的大模型调用框架，支持 vLLM、Transformers、第三方 API 接口，设计目的是为了更加自由地设计 Agent，能够获取概率、注意力等复杂信息。

本项目包含两个核心部分：

1. **Mira 核心框架** - 灵活的大模型调用和 Agent 开发库
2. **新闻溯源验证器** - 基于 AI 的新闻声明真实性验证 Web 应用

---

## 目录结构

```
da_zuo_ye/
└── mira/
    ├── mira/                          # Mira 核心包
    │   ├── __init__.py
    │   ├── args.py                    # 参数配置
    │   ├── inference.py               # 推理引擎
    │   ├── oai_protocol.py            # OpenAI 协议兼容
    │   ├── openrouter.py              # OpenRouter 集成
    │   ├── types.py                   # 数据类型定义
    │   └── utils.py                   # 工具函数
    ├── tests/                         # 测试用例
    │   ├── test_simple.py             # 简单使用示例
    │   ├── test_tool.py               # 工具调用示例
    │   ├── test_struct_output.py      # 结构化输出示例
    │   ├── test_agent.py              # 新闻溯源 Agent 示例
    │   ├── cold_jokes.py              # 冷笑话数据生成
    │   └── eval/                      # 评估脚本
    ├── wangye/                        # Web 应用
    │   └── news_tracer/               # 新闻溯源验证器
    │       ├── streamlit_app.py       # Streamlit 前端应用
    │       ├── server.py              # FastAPI 后端服务
    │       ├── news_tracer_api.py     # API 路由
    │       ├── stock_analyzer.py      # 金融数据分析
    │       ├── ml_strategy.py         # 机器学习策略
    │       ├── strategy_backtester.py # 策略回测
    │       └── index.html             # 备用 HTML 页面
    ├── .env.example                   # 环境变量模板（复制为 .env）
    ├── pyproject.toml                 # 项目配置文件
    ├── README.md                      # 原项目文档
    └── PROJECT_README.md              # 本文档
```

---

## 核心功能

### 1. Mira 框架特性

- ✅ **兼容 OpenAI 协议** - 支持所有 OpenAI 兼容的 API
- ✅ **获取概率信息** - 方便进行 Rollout 和 pass@K 样本生成
- ✅ **多模型支持** - 本地 vLLM、HF Transformers、第三方 API
- ✅ **丰富的模型生态** - 支持 OpenAI、Claude、Gemini、OpenRouter、豆包等
- ✅ **自定义 Tool** - 基于 BaseModel 的函数工具，支持线程管理
- ✅ **灵活的上下文管理** - 支持自定义上下文工程处理

### 2. 新闻溯源验证器功能

- 🔍 **新闻溯源** - 自动搜索相关新闻，分析声明可靠性
- 📈 **金融数据分析** - 股票技术分析、策略回测
- 🤖 **AI 投研** - 基于 AI 的策略推荐和投研仪表盘
- 🌐 **可视化展示** - 层级图谱、时间线、词频统计

---

## 环境要求

- Python 3.11+
- 操作系统：Linux / macOS / Windows
- CUDA 兼容 GPU（可选，用于本地模型推理）

---

## 🚀 快速开始

### 步骤 1：安装依赖

```bash
cd mira
pip install --upgrade pip setuptools
pip install -e .
```

如需完整功能（包含本地模型部署）：

```bash
pip install -e ".[all]"
```

### 步骤 2：配置 API 密钥

#### 2.1 必需的 API 密钥

要运行项目，你需要配置以下 API 密钥：

| API 服务 | 用途 | 获取地址 | 必需 |
|---------|------|---------|------|
| **火山引擎 Ark (豆包)** | 大模型调用 | https://www.volcengine.com/product/ark | 可更换 |
| **Serper** | 谷歌搜索 | https://serper.dev/ | ✅ 必需（新闻溯源） |

#### 2.2 配置步骤

```bash
# 1. 复制环境变量模板
cd mira
copy .env.example .env  # Windows
# 或
cp .env.example .env    # Linux/macOS
```

然后打开 `.env` 文件，填入你的 API 密钥：

```env
# ==============================================
# 必需配置（必须填写）
# ==============================================

# 豆包 API（火山引擎 Ark）
# 获取地址：https://www.volcengine.com/product/ark
ARK_API_KEY = "在这里填入你的 Ark API Key"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# Serper API（用于谷歌搜索）
# 获取地址：https://serper.dev/
SERPER_API_KEY = "在这里填入你的 Serper API Key"

# ==============================================
# 可选配置（根据需要填写）
# ==============================================

# OpenAI API
OPENAI_API_KEY = ""
OPENAI_BASE_URL = "https://api.openai-proxy.com/v1"

# OpenRouter API（聚合多个模型）
OPENROUTER_API_KEY = ""
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Hugging Face Token（用于下载模型）
HF_TOKEN = ""
```

#### 2.3 验证配置

配置完成后，运行简单测试验证：

```bash
cd mira
python -m tests.test_simple
```

### 步骤 3：启动应用

#### 方式一：启动新闻溯源验证器（推荐）

需要同时启动后端和前端两个服务。

#### 终端 1 - 启动后端服务

```bash
cd mira\wangye\news_tracer
python server.py
```

✅ **成功标志**：看到 `Uvicorn running on http://0.0.0.0:8000`

- 后端地址：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`

#### 终端 2 - 启动前端应用

```bash
cd mira\wangye\news_tracer
streamlit run streamlit_app.py
```

✅ **成功标志**：浏览器自动打开 `http://localhost:8501`

---

### 方式二：运行测试脚本

项目提供了多个测试脚本，按难度递进：

#### 🟢 入门级测试

```bash
# 测试简单对话（最简单）
cd mira
python -m tests.test_simple
```

#### 🟡 进阶级测试

```bash
# 测试工具调用
python -m tests.test_tool

# 测试结构化输出
python -m tests.test_struct_output
```

#### 🔴 高级测试

```bash
# 测试新闻溯源 Agent（完整应用）
python -m tests.test_agent
```

---

### 方式三：启动 Mira API 服务

如果你想把 Mira 作为 OpenAI 兼容的 API 服务使用：

```bash
mira-server
```

服务将在 `http://localhost:8090` 启动，兼容 OpenAI SDK。

---

## 📖 使用说明

### 1. 新闻溯源验证器使用

1. 确保已配置 `ARK_API_KEY` 和 `SERPER_API_KEY`
2. 启动后端和前端服务（见"步骤 3"）
3. 在浏览器中打开 Streamlit 应用
4. **新闻溯源**标签页：
   - 输入要验证的新闻声明（例如："人工智能将取代程序员"）
   - 选择溯源深度（1-3层）
   - 点击"开始验证"
5. **金融数据分析**标签页：
   - 输入公司名称（例如："特斯拉"、"阿里巴巴"）
   - 查询股票代码
   - 进行技术分析、策略回测
   - 查看 AI 投研报告

### 2. Mira 框架开发示例

```python
from mira import HumanMessage, LLMTool, OpenAIArgs, OpenRouterLLM
from pydantic import Field
import os

# 定义自定义工具
class GoogleSearchTool(LLMTool):
    """搜索谷歌获取信息"""
    query: str = Field(..., description="搜索查询")
    
    def __call__(self):
        # 实现搜索逻辑
        api_key = os.getenv("SERPER_API_KEY")
        # ...

# 配置模型参数
args = OpenAIArgs(
    model="doubao/doubao-1-5-pro-32k-250115",
    api_key=os.getenv("ARK_API_KEY"),
    base_url=os.getenv("ARK_BASE_URL"),
    temperature=0.7
)

# 初始化 LLM
llm = OpenRouterLLM(args=args)

# 发送消息
messages = [HumanMessage(content="介绍一下人工智能")]

# 获取回复（自动处理工具调用）
response = await llm.forward(messages=messages, tools=[GoogleSearchTool])
print(response)
```

---

## 🧪 测试指南

### 快速验证配置

```bash
cd mira

# 测试 1：简单对话（验证 API Key 是否有效）
python -m tests.test_simple

# 测试 2：工具调用（验证搜索功能）
python -m tests.test_tool
```

### 测试文件说明

| 测试文件 | 功能 | 难度 |
|---------|------|------|
| `test_simple.py` | 基础对话 + 搜索工具 | ⭐ |
| `test_tool.py` | 多工具协同（加法、乘法） | ⭐⭐ |
| `test_struct_output.py` | 结构化 JSON 输出 | ⭐⭐ |
| `test_agent.py` | 完整新闻溯源 Agent | ⭐⭐⭐ |
| `cold_jokes.py` | 批量数据生成 | ⭐⭐⭐ |

---

## 📡 API 接口说明

### 新闻溯源 API

| 端点                              | 方法 | 说明         |
| --------------------------------- | ---- | ------------ |
| `/api/trace`                    | POST | 新闻溯源验证 |
| `/api/stock/code`               | POST | 查询股票代码 |
| `/api/stock/backtest`           | POST | 策略回测     |
| `/api/stock/strategy_recommend` | POST | AI 策略推荐  |
| `/api/stock/strategy_report`    | POST | 生成投研报告 |

**完整 API 文档**：启动后端后访问 `http://localhost:8000/docs`

---

## 🛠️ 技术栈

### 后端

- **FastAPI** - Web 框架
- **Streamlit** - 数据应用框架
- **Mira** - 大模型调用框架

### 前端

- **Streamlit** - 交互式 UI
- **Plotly** - 数据可视化
- **NetworkX** - 图算法

### 数据分析

- **Pandas** - 数据处理
- **NumPy** - 数值计算
- **yfinance** - 金融数据

---

## 💡 开发建议

1. **环境隔离**：建议使用 Conda 或 venv 创建虚拟环境
2. **API 密钥保护**：`.env` 文件已在 `.gitignore` 中，不会被提交到 GitHub
3. **日志调试**：设置 `CHONKIE_LOG_LEVEL=1` 开启详细日志
4. **本地模型**：如需使用本地模型，安装 `vllm` 和相关依赖：
   ```bash
   pip install -e ".[all]"
   ```

---

## ❓ 常见问题

### Q: 后端启动失败？

A: 检查端口 8000 是否被占用，或修改 `server.py` 中的端口号

### Q: Streamlit 无法连接后端？

A: 确保后端服务已启动，并检查 `streamlit_app.py` 中的 `API_BASE_URL` 配置（通常是 `http://localhost:8000`）

### Q: 搜索功能不工作？

A: 检查 `SERPER_API_KEY` 是否正确配置，并确认 Serper 账户有余额

### Q: API 返回 401 未授权？

A: 检查 `.env` 文件中的 `ARK_API_KEY` 是否正确

### Q: 在哪里获取 API Key？

- **Ark (豆包)**: https://www.volcengine.com/product/ark
- **Serper**: https://serper.dev/

---

## 📄 许可证

MIT License

---

## 👥 联系方式

- 项目作者：leeconie
- 邮箱：1730112483@qq.com
- Mira 原作者：luyukun
- 邮箱：lcyqky@icloud.com
