"""
Agent Registry - Agent 注册表

维护「逻辑名 → 工厂函数」的映射，负责按名构造并缓存 Agent 实例。
skill.md 的工作流步骤通过 agent 名引用这里的 Agent，从而解耦编排与实现：
新增能力只需在 registry 里注册一个工厂 + 在 skill.md 里加一步。
"""

from typing import Callable, Dict


class AgentRegistry:
    """Agent 注册表：按名构造并缓存 Agent 实例"""

    def __init__(self, config: dict):
        self.config = config
        self._factories: Dict[str, Callable] = {}
        self._instances: Dict[str, object] = {}
        self._register_defaults()

    def register(self, name: str, factory: Callable):
        """注册一个 Agent 工厂。factory(config) -> agent 实例"""
        self._factories[name] = factory

    def get(self, name: str):
        """获取 Agent 实例（惰性构造 + 单例缓存）"""
        if name not in self._instances:
            factory = self._factories.get(name)
            if factory is None:
                raise ValueError(f"未注册的 Agent: {name}")
            self._instances[name] = factory(self.config)
        return self._instances[name]

    def names(self) -> list:
        """返回已注册的 Agent 逻辑名列表"""
        return list(self._factories.keys())

    def _register_defaults(self):
        """注册系统内置的业务 Agent"""
        from agents.data_agent import DataAgent
        from agents.rule_agent import RuleAgent
        from agents.predict_agent import PredictAgent
        from agents.backtest_agent import BacktestAgent
        from agents.report_agent import ReportAgent
        from agents.llm_report_agent import LLMReportAgent

        self.register("data_agent", lambda c: DataAgent(c))
        # RuleAgent 在 run 里根据 inputs 的 index_name 加载规则，故无参构造
        self.register("rule_agent", lambda c: RuleAgent())
        self.register("predict_agent", lambda c: PredictAgent(c))
        self.register("backtest_agent", lambda c: BacktestAgent(c))
        self.register("report_agent", lambda c: ReportAgent(c))
        self.register("llm_report_agent", lambda c: LLMReportAgent(c))
