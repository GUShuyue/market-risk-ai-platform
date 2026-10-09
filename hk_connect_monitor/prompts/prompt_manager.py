"""
Prompt Manager - Prompt模板管理器 (共享)

管理所有prompt模板的加载和填充。

模板目录结构:
    prompts/
    ├── system/     # 系统提示词
    └── user/       # 用户提示词模板
"""

import os
from typing import Dict


class PromptManager:
    """
    Prompt管理器 - 加载和填充prompt模板

    用法:
        pm = PromptManager()
        system_prompt = pm.get_system_prompt("hk_connect_report")
        template = pm.load("user/prediction_report_template.md")
        prompt = pm.fill(template, cutoff_date="2025-12-31", ...)
    """

    def __init__(self, prompts_dir: str = None):
        """
        参数:
            prompts_dir: prompts目录路径，默认为项目根目录下的 prompts/
        """
        if prompts_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            prompts_dir = os.path.join(project_root, "prompts")
        self.prompts_dir = prompts_dir
        self._cache: Dict[str, str] = {}

    def load(self, template_path: str) -> str:
        """
        加载prompt模板

        参数:
            template_path: 模板路径 (如 "user/prediction_report_template.md")

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

    def get_system_prompt(self, name: str) -> str:
        """获取系统提示词 (如 'hk_connect_report')"""
        return self.load(f"system/{name}.md")

    def get_user_prompt(self, name: str, **kwargs) -> str:
        """
        获取填充后的用户提示词

        参数:
            name: 模板名称 (不含路径和扩展名，如 'prediction_report_template')
            **kwargs: 模板变量

        返回:
            填充后的prompt
        """
        template = self.load(f"user/{name}.md")
        return self.fill(template, **kwargs)
