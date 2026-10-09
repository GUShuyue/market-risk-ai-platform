"""
Prompt Manager - Prompt模板管理器

管理所有Agent使用的prompt模板
"""

import os
from typing import Dict


class PromptManager:
    """
    Prompt管理器 - 加载和填充prompt模板
    
    模板目录结构:
        prompts/
        ├── system/     # 系统提示词
        └── user/       # 用户提示词模板
    """

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self._cache: Dict[str, str] = {}

    def load(self, template_path: str) -> str:
        """
        加载prompt模板

        参数:
            template_path: 模板路径 (如 "user/predict_template.md")

        返回:
            模板内容
        """
        if template_path in self._cache:
            return self._cache[template_path]

        full_path = os.path.join(self.prompts_dir, template_path)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        self._cache[template_path] = content
        return content

    def fill(self, template: str, **kwargs) -> str:
        """
        填充模板变量

        参数:
            template: 模板内容
            **kwargs: 变量值

        返回:
            填充后的prompt
        """
        result = template
        for key, value in kwargs.items():
            placeholder = "{{" + key + "}}"
            result = result.replace(placeholder, str(value))
        return result

    def get_system_prompt(self, agent_name: str) -> str:
        """获取Agent的系统提示词"""
        return self.load(f"system/{agent_name}.md")
