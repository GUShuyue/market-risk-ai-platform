"""
Skill Loader - 技能定义加载器

解析 skills/skill.md 的 YAML frontmatter，提取 workflows 定义。
skill.md 采用与规则文件一致的「YAML frontmatter + Markdown 正文」格式：
    ---
    name: ...
    workflows:
      ...
    ---
    (人可读的正文说明)

orchestrator 读取 workflows 后按步骤编排 Agent，实现「增减功能只改 skill.md」。
"""

import os
import re
from typing import Dict

import yaml


class SkillLoader:
    """从 skill.md 加载工作流定义"""

    def __init__(self, skill_file: str = "skills/skill.md"):
        self.skill_file = skill_file

    def load(self) -> dict:
        """加载并解析 skill.md，返回 frontmatter 内容（含 workflows）"""
        if not os.path.exists(self.skill_file):
            raise FileNotFoundError(f"技能文件不存在: {self.skill_file}")

        with open(self.skill_file, "r", encoding="utf-8") as f:
            content = f.read()

        frontmatter, _ = self._split_content(content)
        if not frontmatter:
            raise ValueError(f"技能文件缺少 YAML frontmatter: {self.skill_file}")

        data = yaml.safe_load(frontmatter) or {}
        if "workflows" not in data:
            raise ValueError(f"技能文件 frontmatter 缺少 workflows 定义: {self.skill_file}")
        return data

    @staticmethod
    def _split_content(content: str) -> tuple:
        """分离 YAML frontmatter 与 Markdown 正文"""
        pattern = r'^---\s*\n(.*?)\n---\s*\n?(.*)'
        match = re.match(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "", content.strip()
