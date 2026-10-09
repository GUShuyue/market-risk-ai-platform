"""
港股通监控系统 - FastAPI 后端

提供自然语言对话API和静态文件服务

用法:
    python main_chat.py --api           # 启动服务 (端口8000)
    python main_chat.py --api --port 9000
"""

import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from interface.chat_handler import ChatHandler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="港股通监控系统 API",
    description="自然语言驱动的港股通监控与预测系统",
    version="1.0.0",
)


# ========== 模型定义 ==========

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    report_paths: list = []
    report_html_urls: list = []
    intent: str = "unknown"


# ========== 全局 ChatHandler 单例 ==========

_chat_handler: ChatHandler = None
_chat_lock = None


def _get_handler() -> ChatHandler:
    """获取（首次创建并缓存）全局 ChatHandler 单例"""
    global _chat_handler, _chat_lock
    if _chat_handler is None:
        _chat_handler = ChatHandler()
        _chat_lock = asyncio.Lock()
    return _chat_handler


# ========== 路由 ==========

@app.post("/api/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """自然语言对话入口

    输入一句话（如"预测2026年3月港股通调整"），
    返回回复文本及可视化报告链接。

    说明: 预测/回测涉及数据采集与 LLM 调用，单次可能耗时数十秒到数分钟。
    """
    handler = _get_handler()

    # ChatHandler.process 是同步阻塞且内部有状态（多轮上下文），
    # 放到线程池执行避免阻塞事件循环，加锁保证串行执行。
    async with _chat_lock:
        result = await asyncio.to_thread(handler.process_with_meta, request.message)

    return {
        "reply": result.get("reply", ""),
        "report_paths": result.get("report_paths", []),
        "report_html_urls": result.get("report_html_urls", []),
        "intent": result.get("intent", "unknown"),
    }


@app.post("/api/chat/stream/")
async def chat_stream(request: ChatRequest):
    """自然语言对话入口 (SSE流式版)

    返回 text/event-stream，逐条推送进度事件，最后推送结果。

    事件格式:
        data: {"type": "progress", "message": "正在扫描..."}\n\n
        data: {"type": "done", "reply": "...", "report_html_urls": [...]}\n\n
        data: {"type": "error", "message": "..."}\n\n
    """
    handler = _get_handler()
    loop = asyncio.get_event_loop()
    progress_queue = asyncio.Queue()

    def progress_cb(message):
        """工作线程中调用的进度回调，线程安全地推入队列"""
        loop.call_soon_threadsafe(
            progress_queue.put_nowait,
            {"type": "progress", "message": message},
        )

    async def generate():
        async with _chat_lock:
            handler.set_progress_callback(progress_cb)

            result_container = {}

            def run_work():
                try:
                    result_container["data"] = handler.process_with_meta(request.message)
                except Exception as e:
                    result_container["error"] = str(e)

            # 在线程池中执行（不阻塞事件循环）
            task = loop.run_in_executor(None, run_work)

            # 持续推送进度事件，直到工作完成
            while not task.done():
                try:
                    event = await asyncio.wait_for(progress_queue.get(), timeout=0.3)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    pass

            # 排空残余进度事件
            while not progress_queue.empty():
                event = progress_queue.get_nowait()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            await task
            handler.set_progress_callback(None)

            # 推送最终结果
            if "error" in result_container:
                final = {"type": "error", "message": result_container["error"]}
            else:
                data = result_container.get("data", {})
                final = {
                    "type": "done",
                    "reply": data.get("reply", ""),
                    "report_html_urls": data.get("report_html_urls", []),
                    "intent": data.get("intent", "unknown"),
                }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ========== 静态文件服务 ==========

@app.get("/")
async def root():
    """返回前端首页"""
    return FileResponse(os.path.join(PROJECT_ROOT, "static", "index.html"))


# 静态资源
_static_dir = os.path.join(PROJECT_ROOT, "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# 报告文件服务 (output目录下的HTML报告)
_output_dir = os.path.join(PROJECT_ROOT, "output")
if os.path.isdir(_output_dir):
    app.mount("/output", StaticFiles(directory=_output_dir), name="output")
