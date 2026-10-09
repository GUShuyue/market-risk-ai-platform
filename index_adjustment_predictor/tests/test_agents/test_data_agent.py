"""测试DataAgent"""

import pytest
from agents.data_agent import DataAgent


class TestDataAgent:
    """DataAgent测试"""

    def setup_method(self):
        self.config = {"data_source": {"provider": "tushare", "api_key": "test"}}
        self.agent = DataAgent(self.config)

    def test_date_validation_valid(self):
        """测试日期验证 - 合法日期"""
        self.agent._validate_dates("2024-06", "2024-04-30")

    def test_date_validation_invalid(self):
        """测试日期验证 - 非法日期"""
        with pytest.raises(ValueError):
            self.agent._validate_dates("2024-06", "2024-06-30")
