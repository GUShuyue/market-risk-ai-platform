"""测试RuleAgent"""

import pytest
from agents.rule_agent import RuleAgent


class TestRuleAgent:
    """RuleAgent测试"""

    def test_load_csi300_rules(self):
        """测试加载沪深300规则"""
        agent = RuleAgent("csi300")
        assert agent.rules is not None
        assert agent.rules["index_name"] == "沪深300"
        assert agent.rules["sample_count"] == 300
        assert agent.rules["buffer_zone"]["priority_in_rank"] == 240
        assert agent.rules["buffer_zone"]["priority_out_rank"] == 360

    def test_load_unsupported_index(self):
        """测试加载不支持的指数"""
        with pytest.raises(ValueError):
            RuleAgent("unsupported")

    def test_get_rule_text(self):
        """测试获取规则Markdown正文"""
        agent = RuleAgent("csi300")
        rule_text = agent.get_rule_text()
        assert "沪深300指数编制规则" in rule_text
        assert "样本空间" in rule_text
        assert "选样方法" in rule_text
