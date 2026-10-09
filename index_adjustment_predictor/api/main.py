"""
FastAPI 主程序 - API接口层

提供REST API接口用于预测和回测
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes.predict import router as predict_router
from api.routes.backtest import router as backtest_router
from api.routes.chat import router as chat_router
import uvicorn
import yaml
import os


def load_env():
    """加载 .env 文件中的环境变量（token、LLM 配置等）"""
    env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(env_file):
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            os.environ[key] = value


def _expand_env_vars(config):
    """递归替换配置中的 ${VAR_NAME} 引用"""
    import re
    if isinstance(config, dict):
        return {k: _expand_env_vars(v) for k, v in config.items()}
    if isinstance(config, list):
        return [_expand_env_vars(i) for i in config]
    if isinstance(config, str):
        return re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), ""), config)
    return config


def load_config():
    """加载配置文件（含 .env 环境变量展开）"""
    load_env()
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    config = {}
    for f in ["config.yaml", "data_source.yaml"]:
        path = os.path.join(config_dir, f)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fp:
                config.update(yaml.safe_load(fp) or {})
    return _expand_env_vars(config)


app = FastAPI(
    title="A股指数成分股调整预测API",
    description="预测沪深300指数成分股调整",
    version="1.0.0"
)

# 加载配置并存到 app.state，供路由读取
app.state.config = load_config()

# 注册路由
app.include_router(predict_router, prefix="/api/predict", tags=["预测"])
app.include_router(backtest_router, prefix="/api/backtest", tags=["回测"])
app.include_router(chat_router, prefix="/api/chat", tags=["聊天"])


@app.get("/")
async def root():
    """返回前端首页"""
    return FileResponse(os.path.join(_PROJECT_DIR, "static", "index.html"))


@app.get("/health")
async def health():
    return {"status": "healthy"}


# 项目根目录（api/ 的上一级）
_PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

# 静态文件服务：前端资源
app.mount("/static", StaticFiles(directory=os.path.join(_PROJECT_DIR, "static")), name="static")
# 静态文件服务：可视化 HTML 报告（prediction_*.html / backtest_*.html）
app.mount("/reports", StaticFiles(directory=os.path.join(_PROJECT_DIR, "reports")), name="reports")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
