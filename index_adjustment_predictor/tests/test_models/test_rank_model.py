"""测试RankModel"""

import pytest
from models.quantitative.rank_model import RankModel


class TestRankModel:
    """RankModel测试"""

    def setup_method(self):
        self.config = {
            "sample_count": 300,
            "buffer_in": 240,
            "buffer_out": 360
        }
        self.model = RankModel(self.config)

    def test_predict_add_candidates(self):
        """测试调入候选识别"""
        rankings = [
            {"code": "000001", "rank": 1},
            {"code": "000002", "rank": 2},
        ]
        current = ["000002"]
        result = self.model.predict(rankings, current)
        assert "000001" in [s["code"] for s in result["add_candidates"]]

    def test_predict_remove_candidates(self):
        """测试调出候选识别"""
        rankings = [
            {"code": "000001", "rank": 400},
        ]
        current = ["000001"]
        result = self.model.predict(rankings, current)
        assert "000001" in [s["code"] for s in result["remove_candidates"]]
