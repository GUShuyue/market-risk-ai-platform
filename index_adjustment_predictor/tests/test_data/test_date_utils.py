"""测试日期工具"""

import pytest
from utils.date_utils import (
    get_cutoff_date,
    get_announcement_date,
    get_review_period
)


class TestDateUtils:
    """日期工具测试"""

    def test_get_cutoff_date_june(self):
        """测试6月调整的截止日"""
        assert get_cutoff_date("2024-06") == "2024-04-30"

    def test_get_cutoff_date_december(self):
        """测试12月调整的截止日"""
        assert get_cutoff_date("2024-12") == "2024-10-31"

    def test_get_cutoff_date_invalid(self):
        """测试无效月份"""
        with pytest.raises(ValueError):
            get_cutoff_date("2024-03")

    def test_get_announcement_date(self):
        """测试公告日期"""
        assert get_announcement_date("2024-06") == "2024-06-10"

    def test_get_review_period_june(self):
        """测试6月调整的考察期"""
        start, end = get_review_period("2024-04-30")
        assert start == "2023-05-01"
        assert end == "2024-04-30"
