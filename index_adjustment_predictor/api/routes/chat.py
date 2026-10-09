"""自然语言聊天接口路由"""

import os
import asyncio
import threading
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from interface.chat_handler import ChatHandler

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


def _get_handler(http_request: Request) -> ChatHandler:
    """获取（首次创建并缓存）全局 ChatHandler 单例，保持多轮对话上下文。"""
    handler = getattr(http_request.app.state, "chat_handler", None)
    if handler is None:
        handler = ChatHandler(http_request.app.state.config)
        http_request.app.state.chat_handler = handler
        http_request.app.state.chat_lock = asyncio.Lock()
    return handler


def _report_url(report_path: str) -> str:
    if not report_path:
        return ""
    filename = os.path.basename(report_path)
    if filename.endswith(".md"):
        filename = filename[:-3] + ".html"
    return f"/reports/{filename}"


@router.post("/")
async def chat(request: ChatRequest, http_request: Request):
    """处理用户自然语言输入，同步等待执行完成后返回回复"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    handler = _get_handler(http_request)
    lock: asyncio.Lock = http_request.app.state.chat_lock

    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()

    def worker():
        try:
            result = handler.process_with_meta(request.message)
        except BaseException as e:  # noqa: BLE001 - 异常透传给请求方（与 to_thread 行为一致）
            payload, err = None, e
        else:
            payload, err = result, None
        try:
            loop.call_soon_threadsafe(_finish, payload, err)
        except RuntimeError:
            pass  # 事件循环已随服务关闭（Ctrl+C），无需回传

    def _finish(payload, err):
        if not fut.done():
            if err is not None:
                fut.set_exception(err)
            else:
                fut.set_result(payload)

    async with lock:
        # daemon 线程：asyncio.to_thread 的线程池在退出时会被等待，
        # 导致 Ctrl+C 后进程要等采集任务跑完才退出；daemon 线程随进程直接结束
        threading.Thread(target=worker, daemon=True).start()
        result = await fut

    return {
        "reply": result.get("reply", ""),
        "report_html_url": _report_url(result.get("report_path", "")),
        "analysis_report_html_url": _report_url(
            result.get("analysis_report_html_path", "")
        ),
        "intent": result.get("intent", "unknown"),
    }
