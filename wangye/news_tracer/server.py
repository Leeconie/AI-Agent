# examples/news_tracer/server.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse


print("=== 服务器启动 ===")
print(f"当前工作目录: {os.getcwd()}")

# 加载环境变量
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
print(f"尝试加载 .env 文件: {dotenv_path}")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(".env 文件加载成功")
else:
    print(".env 文件未找到")

# 打印环境变量
print(f"ARK_API_KEY: {os.getenv('ARK_API_KEY')}")
print(f"ARK_BASE_URL: {os.getenv('ARK_BASE_URL')}")
print(f"SERPER_API_KEY: {os.getenv('SERPER_API_KEY')}")

# 创建主应用
app = FastAPI(title="新闻溯源验证器")

# 导入API路由
try:
    from news_tracer_api import api_router
    app.include_router(api_router)
    print("成功导入并包含 news_tracer_api 路由")
except Exception as e:
    print(f"导入 news_tracer_api 路由失败: {e}")
    raise

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "index.html")
    
    # 打印调试信息
    print(f"查找文件路径: {file_path}")
    print(f"文件是否存在: {os.path.exists(file_path)}")
    
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return HTMLResponse(content="<h1>欢迎使用新闻溯源验证器</h1><p>API文档请访问 <a href='/docs'>/docs</a></p>")

# 添加一个测试端点
@app.get("/test")
async def test():
    return {"message": "服务器运行正常"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)