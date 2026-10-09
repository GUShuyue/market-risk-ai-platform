"""
Orchestrator - Agent编排器（配置驱动）

不再硬编码工作流分支，而是：
1. 通过 SkillLoader 读取 skills/skill.md 的 workflows 定义
2. 通过 AgentRegistry 按名获取 Agent
3. 通用执行引擎按步骤顺序调用 Agent 的 run(inputs)，用 context 传递数据

新增能力 = 实现一个 run(inputs) 的 Agent 并注册 + 在 skill.md 里加一步。
对外接口保持不变：run_workflow(workflow_name, params) 及
data_agent / predict_agent / backtest_agent / report_agent 兼容属性。
"""

import re
from typing import Any

from core.agent_registry import AgentRegistry
from core.skill_loader import SkillLoader


class Orchestrator:
    """
    Agent编排器 - 协调各 Agent 执行 workflow

    工作流定义见 skills/skill.md 的 frontmatter。
    """

    def __init__(self, config: dict):
        self.config = config
        self.registry = AgentRegistry(config)

        # 加载工作流定义（来自 skill.md）
        self.skill = SkillLoader().load()
        self.workflows = self.skill.get("workflows", {})

        # 兼容属性：供 main.py / chat_handler 直接访问
        self.data_agent = self.registry.get("data_agent")
        self.predict_agent = self.registry.get("predict_agent")
        self.backtest_agent = self.registry.get("backtest_agent")
        self.report_agent = self.registry.get("report_agent")

    def run_workflow(
        self, workflow_name: str, params: dict
    ) -> dict[str, Any]:
        """
        执行指定的工作流

        参数:
            workflow_name: 工作流名称（对应 skill.md 中 workflows 的键）
            params: 工作流参数（通过 {params.xxx} 引用）

        返回:
            按 workflow 的 returns 定义组装的结果字典
        """
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise ValueError(
                f"未知的工作流: {workflow_name}，可用: {list(self.workflows.keys())}"
            )

        # 补齐默认参数
        params = dict(params)
        params.setdefault("use_llm", True)

        context: dict[str, Any] = {"params": params}

        for step in workflow.get("steps", []):
            step_id = step.get("id")
            agent_name = step.get("agent")
            if not step_id or not agent_name:
                raise ValueError(f"工作流 {workflow_name} 存在缺少 id/agent 的步骤: {step}")

            agent = self.registry.get(agent_name)
            inputs = self._resolve_inputs(step.get("inputs", {}), context)
            outputs = agent.run(inputs)
            context[step_id] = outputs

        # 按 returns 定义组装返回结果
        result = {}
        for key, ref in workflow.get("returns", {}).items():
            result[key] = self._resolve_value(ref, context)
        return result

    def _resolve_inputs(self, inputs: dict, context: dict) -> dict:
        """解析一个步骤的 inputs，替换 {path} 引用"""
        return {
            key: self._resolve_value(value, context)
            for key, value in inputs.items()
        }

    def _resolve_value(self, value: Any, context: dict) -> Any:
        """解析单个值；字符串若为完整 {path} 引用则替换为 context 中的值"""
        if isinstance(value, str):
            match = re.fullmatch(r"\{([^}]+)\}", value.strip())
            if match:
                return self._get_path(context, match.group(1).strip())
        return value

    @staticmethod
    def _get_path(context: dict, path: str) -> Any:
        """按点号路径从 context 取值，缺失时返回 None"""
        current = context
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
