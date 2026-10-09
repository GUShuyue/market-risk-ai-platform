"""
Mode B 编排器

协调HSCI覆盖率预测模型和港股通映射层, 完成完整的调入调出预测工作流。

工作流:
    1. predict: 数据采集 → HSCI预测 → 港股通映射 → 生成报告
    2. backtest: 加载历史预测 → 获取实际调整 → 对比分析 → 生成报告
"""

import os
import sys
import json
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from mode_b.data.hsci_history_loader import HSCIHistoryLoader
from mode_b.models.hsci_coverage_model import HSCICoverageModel, CoveragePrediction
from mode_b.models.stock_connect_mapper import StockConnectMapper, StockConnectPrediction

logger = logging.getLogger(__name__)


class OrchestratorB:
    """
    Mode B 编排器

    用法:
        orch = OrchestratorB()
        # 预测
        result = orch.run_predict(
            market_cap_data=[...],
            cutoff_date=date(2024, 6, 30),
        )
        # 回测
        bt_result = orch.run_backtest(
            prediction=result,
            target_adjustment_date=date(2024, 6, 10),
        )
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.history_loader = HSCIHistoryLoader()
        self.coverage_model = HSCICoverageModel()
        self.mapper = StockConnectMapper()
        self._loaded = False

    def _ensure_loaded(self):
        """确保历史数据已加载"""
        if not self._loaded:
            self.history_loader.load()
            self._loaded = True

    # ================================================================
    # 预测工作流
    # ================================================================

    def run_predict(
        self,
        market_cap_data: List[Dict],
        cutoff_date: date,
        stock_info_map: Dict[str, Dict] = None,
        connect_pool_codes: set = None,
        review_date: date = None,
        use_current_constituents: bool = True,
        min_market_cap: float = 0,
        month_end_market_cap_map: Dict[str, float] = None,
    ) -> Dict:
        """
        执行完整预测工作流

        参数:
            market_cap_data: 市值数据 [{code, name, avg_market_cap}, ...]
            cutoff_date: 数据截止日 (6月30日 或 12月31日)
            stock_info_map: 股票信息映射 (用于港股通资格检查)
            connect_pool_codes: 当前港股通标的池代码集合
            review_date: 预计检讨日期
            use_current_constituents: 是否使用当前HSCI成份股作为基准
            min_market_cap: 最低市值过滤阈值 (港币元), 0=不过滤
            month_end_market_cap_map: 12个月平均月末市值映射
                (实施办法第70/73条: 港股通小型股市值门槛使用)
                若为None则回退到avg_market_cap_map

        返回:
            {
                "hsci_prediction": CoveragePrediction序列化,
                "connect_prediction": StockConnectPrediction序列化,
                "report_data": {...},
            }
        """
        self._ensure_loaded()
        stock_info_map = stock_info_map or {}
        connect_pool_codes = connect_pool_codes or set()

        logger.info(f"开始Mode B预测, cutoff_date={cutoff_date}, min_market_cap={min_market_cap/1e8:.0f}亿")

        # 1. 获取当前HSCI成份股
        official_count = None
        if use_current_constituents:
            current_constituents = self.history_loader.current_constituents
        else:
            # 使用历史快照
            snapshot = self.history_loader.get_constituents_at(cutoff_date)
            current_constituents = snapshot.constituents
            official_count = snapshot.official_count
            if snapshot.count_discrepancy != 0:
                logger.warning(
                    f"使用官方成分股数量: {official_count} "
                    f"(重建数量={snapshot.total_count}, 差异={snapshot.count_discrepancy:+d})"
                )

        # 显示成分股数量: 优先使用官方count_after, 其次使用重建数量
        display_count = official_count if official_count else len(current_constituents)
        logger.info(f"当前HSCI成份股: {display_count} 只 (官方count_after)")

        # 2. 运行HSCI覆盖率模型
        hsci_prediction = self.coverage_model.predict(
            market_cap_data=market_cap_data,
            current_constituents=current_constituents,
            cutoff_date=cutoff_date,
            review_date=review_date,
            min_market_cap=min_market_cap,
            official_count=official_count,
        )

        logger.info(
            f"HSCI预测: 调入{len(hsci_prediction.additions)}只, "
            f"调出{len(hsci_prediction.removals)}只"
        )

        # 3. 构建avg_market_cap_map
        avg_mc_map = {
            s["code"].strip().zfill(5): float(s.get("avg_market_cap", 0))
            for s in market_cap_data
        }

        # 4. 映射到港股通
        connect_prediction = self.mapper.map(
            hsci_prediction=hsci_prediction,
            stock_info_map=stock_info_map,
            connect_pool_codes=connect_pool_codes,
            avg_market_cap_map=avg_mc_map,
            month_end_market_cap_map=month_end_market_cap_map,
        )

        logger.info(
            f"港股通预测: 调入{len(connect_prediction.connect_additions)}只, "
            f"调出{len(connect_prediction.connect_removals)}只"
        )

        # 5. 序列化结果
        result = {
            "cutoff_date": cutoff_date.isoformat(),
            "review_date": (review_date or cutoff_date).isoformat(),
            "hsci_prediction": self._serialize_hsci_prediction(hsci_prediction),
            "connect_prediction": self._serialize_connect_prediction(connect_prediction),
            "report_data": {
                "current_hsci_count": hsci_prediction.current_constituent_count,
                "predicted_hsci_count": hsci_prediction.predicted_constituent_count,
                "total_universe": hsci_prediction.total_universe,
                "hsci_additions": len(hsci_prediction.additions),
                "hsci_removals": len(hsci_prediction.removals),
                "connect_additions": len(connect_prediction.connect_additions),
                "connect_removals": len(connect_prediction.connect_removals),
                "ineligible": len(connect_prediction.ineligible_additions),
            },
        }

        return result

    # ================================================================
    # 回测工作流
    # ================================================================

    def run_backtest(
        self,
        prediction: Dict,
        target_adjustment_date: date,
        comparison_window_days: int = 30,
    ) -> Dict:
        """
        执行回测工作流

        参数:
            prediction: 预测结果 (run_predict的返回值)
            target_adjustment_date: 实际调整生效日期
            comparison_window_days: 对比窗口天数

        返回:
            回测结果字典
        """
        self._ensure_loaded()

        from mode_b.agents.backtest_agent_b import BacktestAgentB

        backtest_agent = BacktestAgentB()

        # 获取实际HSCI调整
        window_start = target_adjustment_date - timedelta(days=comparison_window_days)
        actual_hsci_changes = self.history_loader.get_changes_between(
            window_start, target_adjustment_date + timedelta(days=7)
        )

        # 获取实际港股通调整
        actual_connect_changes = self.history_loader.get_hk_connect_changes_between(
            window_start, target_adjustment_date + timedelta(days=7)
        )

        # 执行回测
        result = backtest_agent.backtest(
            prediction=prediction,
            actual_hsci_changes=actual_hsci_changes,
            actual_connect_changes=actual_connect_changes,
            target_date=target_adjustment_date,
        )

        return result

    # ================================================================
    # 历史回测 (使用历史数据进行回测验证)
    # ================================================================

    def run_historical_backtest(
        self,
        market_cap_data: List[Dict],
        cutoff_date: date,
        target_adjustment_date: date,
        stock_info_map: Dict[str, Dict] = None,
        connect_pool_codes: set = None,
    ) -> Dict:
        """
        完整的历史回测: 预测 + 对比实际结果

        参数:
            market_cap_data: cutoff_date时点的市值数据
            cutoff_date: 数据截止日
            target_adjustment_date: 实际调整生效日期
            stock_info_map: 股票信息
            connect_pool_codes: 港股通池

        返回:
            包含预测和回测结果的字典
        """
        # 1. 先执行预测 (使用历史快照作为当前成份股)
        prediction = self.run_predict(
            market_cap_data=market_cap_data,
            cutoff_date=cutoff_date,
            stock_info_map=stock_info_map,
            connect_pool_codes=connect_pool_codes,
            use_current_constituents=False,  # 使用cutoff_date时的历史快照
        )

        # 2. 执行回测
        backtest_result = self.run_backtest(
            prediction=prediction,
            target_adjustment_date=target_adjustment_date,
        )

        return {
            "prediction": prediction,
            "backtest": backtest_result,
            "cutoff_date": cutoff_date.isoformat(),
            "target_adjustment_date": target_adjustment_date.isoformat(),
        }

    # ================================================================
    # 序列化辅助
    # ================================================================

    @staticmethod
    def _serialize_hsci_prediction(pred: CoveragePrediction) -> Dict:
        """序列化HSCI预测结果"""
        return {
            "review_date": pred.review_date.isoformat(),
            "cutoff_date": pred.cutoff_date.isoformat(),
            "total_universe": pred.total_universe,
            "current_constituent_count": pred.current_constituent_count,
            "predicted_constituent_count": pred.predicted_constituent_count,
            "additions": [
                {
                    "code": s.code,
                    "name": s.name,
                    "rank": s.rank,
                    "coverage": round(s.cumulative_coverage, 2),
                    "avg_market_cap": s.avg_market_cap,
                    "avg_market_cap_hkd_yi": round(s.avg_market_cap / 1e8, 2),
                    "predicted_sub_index": s.predicted_sub_index.value,
                    "buffer_status": s.buffer_status,
                }
                for s in pred.additions
            ],
            "removals": [
                {
                    "code": s.code,
                    "name": s.name,
                    "rank": s.rank,
                    "coverage": round(s.cumulative_coverage, 2),
                    "avg_market_cap": s.avg_market_cap,
                    "avg_market_cap_hkd_yi": round(s.avg_market_cap / 1e8, 2),
                    "current_sub_index": s.current_sub_index.value,
                    "buffer_status": s.buffer_status,
                }
                for s in pred.removals
            ],
            "coverage_at_95": pred.coverage_at_95,
        }

    @staticmethod
    def _serialize_connect_prediction(pred: StockConnectPrediction) -> Dict:
        """序列化港股通预测结果"""
        return {
            "hsci_additions": pred.hsci_additions,
            "hsci_removals": pred.hsci_removals,
            "connect_additions": pred.connect_additions,
            "connect_removals": pred.connect_removals,
            "ineligible_additions": pred.ineligible_additions,
            "summary": pred.summary,
        }
