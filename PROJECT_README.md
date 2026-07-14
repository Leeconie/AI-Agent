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
    ├── .env                           # 环境变量配置
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

## 快速开始

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

### 步骤 2：配置环境变量

项目已包含 `.env` 文件，主要配置项：

```env
# 豆包 API（火山引擎 Ark）
ARK_API_KEY = "your-ark-api-key-here"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# Serper API（用于搜索）
SERPER_API_KEY = "your-serper-api-key-here"

# 其他可选配置
OPENAI_API_KEY = ""
OPENROUTER_API_KEY = ""
HF_TOKEN = ""
```

---

## 启动方式

### 方式一：启动新闻溯源验证器（推荐）

需要同时启动后端和前端两个服务。

**终端 1 - 启动后端服务：**

```bash
cd .\mira\wangye\news_tracer
python server.py
```

后端服务将在 `http://localhost:8000` 启动，API 文档访问：`http://localhost:8000/docs`

**终端 2 - 启动前端应用：**

```bash
cd .\mira\wangye\news_tracer
streamlit run streamlit_app.py
```

前端应用将在浏览器中自动打开，通常是 `http://localhost:8501`

### 方式二：测试 Mira 框架

```bash
cd .\mira
python -m tests.test_simple
```

其他测试示例：

- `python -m tests.test_tool` - 测试工具调用
- `python -m tests.test_struct_output` - 测试结构化输出

### 方式三：启动 Mira API 服务

```bash
mira-server
```

---

## 使用说明

### 1. 新闻溯源验证器使用

1. 启动后端和前端服务
2. 在浏览器中打开 Streamlit 应用
3. **新闻溯源**标签页：
   - 输入要验证的新闻声明
   - 选择溯源深度
   - 点击"开始验证"
4. **金融数据分析**标签页：
   - 输入公司名称查询股票代码
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
        pass

# 使用模型
args = OpenAIArgs(
    model="doubao/doubao-1-5-pro-32k-250115",
    api_key=os.getenv("ARK_API_KEY"),
    base_url=os.getenv("ARK_BASE_URL")
)
llm = OpenRouterLLM(args=args)

# 发送消息
messages = [HumanMessage(content="你好，请介绍一下自己")]
response = await llm.forward(messages=messages)
```

---

## API 接口说明

### 新闻溯源 API

| 端点                              | 方法 | 说明         |
| --------------------------------- | ---- | ------------ |
| `/api/trace`                    | POST | 新闻溯源验证 |
| `/api/stock/code`               | POST | 查询股票代码 |
| `/api/stock/backtest`           | POST | 策略回测     |
| `/api/stock/strategy_recommend` | POST | AI 策略推荐  |
| `/api/stock/strategy_report`    | POST | 生成投研报告 |

完整 API 文档请访问：`http://localhost:8000/docs`

---

## 技术栈

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
- **yfinance** - 金融数据（如适用）

---

## 开发建议

1. **环境隔离**：建议使用 Conda 或 venv 创建虚拟环境
2. **API 密钥**：注意保护 `.env` 文件中的密钥信息
3. **日志调试**：设置 `CHONKIE_LOG_LEVEL=1` 开启详细日志
4. **本地模型**：如需使用本地模型，安装 `vllm` 和相关依赖

---

## 常见问题

### Q: 后端启动失败？

A: 检查端口 8000 是否被占用，或修改 `server.py` 中的端口号

### Q: Streamlit 无法连接后端？

A: 确保后端服务已启动，并检查 `streamlit_app.py` 中的 `API_BASE_URL` 配置

### Q: 搜索功能不工作？

A: 检查 `SERPER_API_KEY` 是否正确配置

---

## 许可证

MIT License

---

## 联系方式

- 项目作者：leeconie
- 邮箱：1730112483@qq.com
- mira作者：luyukun
- 邮箱：lcyqky@icloud.com
