#!/usr/bin/env python3
"""
港股通监控系统 - 自然语言交互 CLI

用法:
    python main_chat.py              # 启动交互式对话
    python main_chat.py --api        # 启动Web API服务 (端口8000)
    python main_chat.py --api --port 9000  # 指定端口

交互模式下直接输入自然语言，系统自动识别意图并执行:
    - "扫描港股通"              → 模式A监控扫描
    - "预测2026年3月港股通调整"  → 模式B预测
    - "回测2026年3月"           → 模式B回测
    - "查看HSCI编制规则"        → 规则查询
    - "帮助"                    → 帮助信息
"""

import os
import sys
import argparse

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def run_cli():
    """启动交互式命令行对话"""
    from interface.chat_handler import ChatHandler

    handler = ChatHandler()

    print("=" * 60)
    print("  港股通监控系统 · 自然语言交互")
    print("=" * 60)
    print()
    print("支持的操作:")
    print("  模式A - 扫描: '扫描港股通'")
    print("  模式B - 预测: '预测2026年3月港股通调整'")
    print("  模式B - 回测: '回测2026年3月'")
    print("  规则查询:     '查看HSCI编制规则'")
    print("  退出:         'exit' 或 'quit'")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "退出"):
            print("再见!")
            break

        print()
        try:
            reply = handler.process(user_input)
            print(reply)
        except Exception as e:
            print(f"处理出错: {e}")
        print()


def run_api(port=8000):
    """启动 Web API 服务"""
    import uvicorn
    from api_main import app

    print(f"启动 Web API 服务: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    parser = argparse.ArgumentParser(
        description="港股通监控系统 - 自然语言交互"
    )
    parser.add_argument(
        "--api", action="store_true",
        help="启动 Web API 服务 (而非交互式CLI)"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Web API 服务端口 (默认8000)"
    )
    args = parser.parse_args()

    if args.api:
        run_api(args.port)
    else:
        run_cli()


if __name__ == "__main__":
    main()
