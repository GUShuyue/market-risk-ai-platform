"""
Rule Parser - 指数编制规则解析器

解析指数编制规则文档（YAML frontmatter + Markdown正文格式），
生成结构化规则对象供RuleAgent使用。
"""

import os
import re
import yaml
from typing import Dict, List, Optional


class RuleParser:
    """
    规则解析器 - 解析指数编制规则文档
    
    文件格式：
        ---
        YAML frontmatter (机器可解析的量化参数)
        ---
        
        Markdown正文 (人可读的完整规则描述，供LLM作为context)
    
    解析后的规则结构:
    {
        "index_name": "沪深300",
        "index_code": "000300.SH",
        "sample_count": 300,
        "sample_space": {...},
        "selection_method": {...},
        "buffer_zone": {...},
        "special_rules": {...},
        "weighting": {...},
        "rule_text": "Markdown正文内容",  # 供LLM阅读
        "document_version": "2023年9月",
        "source_url": "..."
    }
    """

    def parse(self, rule_file: str) -> dict:
        """
        解析规则文件

        参数:
            rule_file: 规则文件路径

        返回:
            结构化规则字典
        """
        if not os.path.exists(rule_file):
            raise FileNotFoundError(f"规则文件不存在: {rule_file}")

        with open(rule_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 分离 YAML frontmatter 和 Markdown 正文
        frontmatter, rule_text = self._split_content(content)

        # 解析 YAML frontmatter
        rules = yaml.safe_load(frontmatter) if frontmatter else {}
        
        # 附加 Markdown 正文
        rules["rule_text"] = rule_text

        return rules

    def _split_content(self, content: str) -> tuple:
        """
        分离 YAML frontmatter 和 Markdown 正文
        
        格式:
            ---
            yaml content
            ---
            markdown content
        
        返回:
            (yaml_str, markdown_str)
        """
        # 匹配以 --- 开头和结尾的 YAML frontmatter
        pattern = r'^---\s*\n(.*?)\n---\s*\n?(.*)'
        match = re.match(pattern, content, re.DOTALL)
        
        if match:
            yaml_content = match.group(1).strip()
            markdown_content = match.group(2).strip()
            return yaml_content, markdown_content
        else:
            # 没有 frontmatter，全部当作正文
            return "", content.strip()

    def get_quantitative_params(self, rules: dict) -> dict:
        """
        从规则中提取量化预测所需的核心参数
        
        返回:
            {
                "sample_count": 300,
                "buffer_in_rank": 240,
                "buffer_out_rank": 360,
                "volume_cutoff_ratio": 0.5,   # 成交金额剔除比例
                "max_adjustment_ratio": 0.10,
                "listing_time_rules": [...],
                "exclusions": [...],
            }
        """
        buffer = rules.get("buffer_zone", {})
        selection = rules.get("selection_method", {})
        sample_space = rules.get("sample_space", {})

        # 根据 selection_method 推断成交金额剔除比例
        step1_action = selection.get("step1_action", "")
        if "后50%" in step1_action:
            volume_cutoff_ratio = 0.5
        elif "后20%" in step1_action:
            volume_cutoff_ratio = 0.8  # 保留80%
        else:
            volume_cutoff_ratio = 1.0  # 不剔除

        return {
            "sample_count": rules.get("sample_count", 0),
            "buffer_in_rank": buffer.get("priority_in_rank", 0),
            "buffer_out_rank": buffer.get("priority_out_rank", 0),
            "volume_cutoff_ratio": volume_cutoff_ratio,
            "max_adjustment_ratio": rules.get("max_adjustment_ratio", 0.10),
            "listing_time_rules": sample_space.get("listing_time_rules", []),
            "exclusions": sample_space.get("exclusions", []),
        }
