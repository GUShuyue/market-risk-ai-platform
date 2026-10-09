"""
港股通映射层

将HSCI成份股变动预测映射到港股通调入调出预测。

映射逻辑:
    1. HSCI调入 → 检查港股通资格 → 预测港股通调入
    2. HSCI调出 → 若在港股通池中 → 预测港股通调出

港股通资格规则 (来自实施办法):
    - 必须是HSCI成份股 (大型/中型/小型)
    - 非第二上市 (-S/-SW后缀不可纳入, 除非已转为主要上市)
    - WVR公司须满足上市时长+市值+成交额条件
    - 小型股须满足12个月平均市值≥50亿港币
    - A+H股H股不受指数成份股限制 (但须在HSCI中)
"""

import os
import sys
import re
import logging
from datetime import date
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum

# 添加项目根目录到path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# 导入Mode A的配置和模型
try:
    import config as mode_a_config
    from models import IndexType, format_hkd
except ImportError:
    mode_a_config = None
    logger.warning("无法导入Mode A配置, 小型股市值门槛将使用默认值")


class ConnectEligibility(str, Enum):
    """港股通资格状态"""
    ELIGIBLE = "合资格"
    SECONDARY_LISTING = "第二上市排除"
    WVR_PENDING = "WVR待满足"
    SMALL_CAP_BELOW = "小型股市值不足"
    NOT_IN_HSCI = "非HSCI成份股"
    EXCLUDED = "其他排除"


@dataclass
class StockConnectPrediction:
    """港股通调入调出预测结果"""
    hsci_additions: List[Dict] = field(default_factory=list)       # HSCI调入列表
    hsci_removals: List[Dict] = field(default_factory=list)        # HSCI调出列表
    connect_additions: List[Dict] = field(default_factory=list)    # 预测港股通调入
    connect_removals: List[Dict] = field(default_factory=list)     # 预测港股通调出
    ineligible_additions: List[Dict] = field(default_factory=list) # HSCI调入但不合资格
    summary: Dict = field(default_factory=dict)                    # 汇总信息


class StockConnectMapper:
    """
    港股通映射层

    将HSCI预测结果映射到港股通调入调出预测

    用法:
        mapper = StockConnectMapper()
        result = mapper.map(
            hsci_prediction=coverage_prediction,
            stock_info_map={code: StockInfo, ...},
            connect_pool_codes=set("00700", ...),
        )
    """

    # 小型股市值门槛 (来自实施办法)
    SMALL_CAP_INCLUSION = 5_000_000_000   # 50亿港币
    SMALL_CAP_REMOVAL = 4_000_000_000     # 40亿港币

    def __init__(self):
        # 尝试从Mode A配置加载门槛
        if mode_a_config:
            self.SMALL_CAP_INCLUSION = mode_a_config.SMALL_CAP_INCLUSION_MARKET_CAP
            self.SMALL_CAP_REMOVAL = mode_a_config.SMALL_CAP_REMOVAL_MARKET_CAP

    def map(
        self,
        hsci_prediction,
        stock_info_map: Dict[str, Dict],
        connect_pool_codes: Set[str],
        avg_market_cap_map: Dict[str, float] = None,
        month_end_market_cap_map: Dict[str, float] = None,
    ) -> StockConnectPrediction:
        """
        将HSCI预测映射到港股通预测

        参数:
            hsci_prediction: HSCICoverageModel的预测结果 (CoveragePrediction)
            stock_info_map: 股票信息映射
                {code: {name, is_secondary, is_wvr, is_a_plus_h, currency, ...}}
            connect_pool_codes: 当前港股通标的池中的股票代码集合
            avg_market_cap_map: 各股票的12个月日均市值映射 (用于HSCI覆盖率排序, 非港股通门槛)
                {code: avg_market_cap}
            month_end_market_cap_map: 各股票的12个月平均月末市值映射
                (实施办法第70/73条: 小型股市值门槛使用月末市值)
                优先使用; 若为空则回退到avg_market_cap_map
                {code: month_end_avg_market_cap}

        返回:
            StockConnectPrediction
        """
        avg_market_cap_map = avg_market_cap_map or {}
        month_end_market_cap_map = month_end_market_cap_map or {}
        # 港股通小型股市值门槛优先使用月末市值, 无则回退到日均市值
        connect_mc_map = month_end_market_cap_map or avg_market_cap_map
        result = StockConnectPrediction()

        # 处理HSCI调入
        for add in hsci_prediction.additions:
            code = add.code
            info = stock_info_map.get(code, {})
            entry = {
                "code": code,
                "name": add.name,
                "rank": add.rank,
                "coverage": round(add.cumulative_coverage, 2),
                "avg_market_cap": add.avg_market_cap,
                "predicted_sub_index": add.predicted_sub_index.value,
                "is_secondary": info.get("is_secondary_listing", False),
                "is_wvr": info.get("is_wvr", False),
                "is_a_plus_h": info.get("is_a_plus_h", False),
            }

            result.hsci_additions.append(entry)

            # 检查港股通资格 (使用月末市值, 实施办法第70条)
            eligibility = self._check_eligibility(
                code, info, add.predicted_sub_index.value,
                connect_mc_map.get(code, add.avg_market_cap),
                in_pool=code in connect_pool_codes,
            )

            if eligibility == ConnectEligibility.ELIGIBLE:
                entry["eligibility"] = "合资格"
                entry["reason"] = self._build_eligible_reason(info, add)
                result.connect_additions.append(entry)
            else:
                entry["eligibility"] = eligibility.value
                entry["reason"] = self._build_ineligible_reason(eligibility, info, add)
                result.ineligible_additions.append(entry)

        # 处理HSCI调出
        for rem in hsci_prediction.removals:
            code = rem.code
            info = stock_info_map.get(code, {})
            entry = {
                "code": code,
                "name": rem.name,
                "rank": rem.rank,
                "coverage": round(rem.cumulative_coverage, 2),
                "avg_market_cap": rem.avg_market_cap,
                "current_sub_index": rem.current_sub_index.value,
                "in_pool": code in connect_pool_codes,
            }

            result.hsci_removals.append(entry)

            # 港股通调出判断 (实施办法第73条):
            # 不再属于HSCI成份股"及"第70条第四项(A+H股)范围 → 调出
            # A+H股即使HSCI调出, 仍属于第70条第四项范围, 不应调出
            if code in connect_pool_codes:
                if info.get("is_a_plus_h", False):
                    entry["reason"] = "HSCI调出, 但属于A+H股(第70条第四项), 保留港股通资格"
                else:
                    entry["reason"] = "HSCI成份股调出, 对应港股通调出"
                    result.connect_removals.append(entry)

        # 生成汇总
        result.summary = {
            "hsci_additions": len(result.hsci_additions),
            "hsci_removals": len(result.hsci_removals),
            "connect_additions": len(result.connect_additions),
            "connect_removals": len(result.connect_removals),
            "ineligible_additions": len(result.ineligible_additions),
        }

        return result

    def _check_eligibility(
        self,
        code: str,
        info: Dict,
        sub_index: str,
        avg_market_cap: float,
        in_pool: bool,
    ) -> ConnectEligibility:
        """
        检查股票的港股通资格

        参数:
            code: 股票代码
            info: 股票信息字典
            sub_index: 子指数类型 (大型/中型/小型)
            avg_market_cap: 12个月平均市值
            in_pool: 是否已在港股通池中
        """
        # 1. 第二上市排除
        if info.get("is_secondary_listing", False) and not in_pool:
            return ConnectEligibility.SECONDARY_LISTING

        # 2. WVR待满足 (仅检查是否WVR且不在池中, 具体条件由Mode A规则引擎判断)
        if info.get("is_wvr", False) and not in_pool:
            # WVR股票需要满足额外条件, 这里标记为待检查
            # 实际条件判断在orchestrator中调用Mode A的RuleEngine
            return ConnectEligibility.WVR_PENDING

        # 3. 小型股市值门槛
        if "小型" in str(sub_index):
            if not in_pool:
                # 不在池中: 检查是否达到调入线
                if avg_market_cap < self.SMALL_CAP_INCLUSION:
                    return ConnectEligibility.SMALL_CAP_BELOW
            else:
                # 在池中: 检查是否低于调出线
                if avg_market_cap < self.SMALL_CAP_REMOVAL:
                    return ConnectEligibility.SMALL_CAP_BELOW

        # 4. 货币检查 (非港币报价)
        currency = info.get("currency", "HKD")
        if currency and currency != "HKD":
            return ConnectEligibility.EXCLUDED

        return ConnectEligibility.ELIGIBLE

    def _build_eligible_reason(self, info: Dict, add) -> str:
        """构建合资格原因"""
        parts = []
        sub = add.predicted_sub_index.value
        parts.append(f"HSCI调入(预测{sub}股)")

        if info.get("is_a_plus_h"):
            parts.append("A+H股")
        if info.get("is_wvr"):
            parts.append("WVR公司(已满足首次纳入条件)")
        if "小型" in sub:
            mc = add.avg_market_cap
            parts.append(f"12个月平均市值{mc/1e8:.1f}亿≥{self.SMALL_CAP_INCLUSION/1e8:.0f}亿")

        return ", ".join(parts)

    def _build_ineligible_reason(self, eligibility: ConnectEligibility, info: Dict, add) -> str:
        """构建不合资格原因"""
        reasons = {
            ConnectEligibility.SECONDARY_LISTING: (
                f"第二上市股票({'-SW' if info.get('is_wvr') else '-S'}后缀), "
                f"不可纳入港股通 (须转换为在港主要上市后方可纳入)"
            ),
            ConnectEligibility.WVR_PENDING: (
                f"WVR公司, 需满足上市满6个月+20交易日、"
                f"183日日均市值≥200亿、183日成交额≥60亿等条件"
            ),
            ConnectEligibility.SMALL_CAP_BELOW: (
                f"小型股12个月平均市值{add.avg_market_cap/1e8:.1f}亿, "
                f"未达到调入标准({self.SMALL_CAP_INCLUSION/1e8:.0f}亿)"
            ),
            ConnectEligibility.NOT_IN_HSCI: "非HSCI成份股",
            ConnectEligibility.EXCLUDED: f"报价货币{info.get('currency', '未知')}非港币",
        }
        return reasons.get(eligibility, "未知原因")

    @staticmethod
    def detect_secondary_listing(name: str) -> bool:
        """
        通过股票名称检测第二上市

        第二上市股票名称包含 -S 或 -SW 后缀
        """
        if not name:
            return False
        # 匹配 -S, -SW 后缀 (不匹配包含S的普通名称)
        return bool(re.search(r'\s-S(W)?$', name.strip()))

    @staticmethod
    def detect_wvr(name: str) -> bool:
        """
        通过股票名称检测WVR (不同投票权架构)

        WVR股票名称包含 -W 或 -SW 后缀
        """
        if not name:
            return False
        return bool(re.search(r'\s-(S)?W$', name.strip()))
