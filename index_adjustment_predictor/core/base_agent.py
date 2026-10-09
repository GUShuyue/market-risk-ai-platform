"""
Base Agent - 统一 Agent 接口

所有业务 Agent 遵循同一约定:
    run(inputs: dict) -> dict

inputs 是上一步骤/参数传入的字典，outputs 是产出的字典。
skill.md 中的工作流步骤通过 agent 名 + 输入引用把各 Agent 串起来，
新增能力只需实现一个 run(inputs) 并在 skill.md 里加一步即可。
"""

from typing import Dict


class BaseAgent:
    """
    统一 Agent 基类

    子类需实现 run(inputs) 方法。config 由注册表在构造时传入。
    """

    # 逻辑名，供注册表与 skill.md 引用
    name = "base"

    def __init__(self, config: dict = None):
        self.config = config or {}

    def run(self, inputs: Dict) -> Dict:
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现 run(inputs) 方法"
        )
