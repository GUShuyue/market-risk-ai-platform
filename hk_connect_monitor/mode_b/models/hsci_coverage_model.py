"""
HSCI覆盖率预测模型

实现恒生综合指数的95%累计市值覆盖率选股规则和缓冲区机制,
预测定期检讨时的成份股调入调出, 并对调整后的全部成分股按覆盖率统一切分大/中/小型。

核心规则 (来自 rules/hsci.md + 编算细则PDF + 编算总则PDF):
    1. 按12个月日均市值降序排列所有合资格证券
       - 市值不包括其他上市股本, H股公司只计H股部分 (编算总则6.1)
    2. 计算每只证券的累计市值覆盖率
    3. 累计覆盖率 ≤ 95% 的证券被选为成份股
    4. 缓冲区:
       - 非成份股覆盖率 ≤ 94% → 加入指数
       - 现有成份股覆盖率 > 96% → 从指数剔除
       - 94%-96% 之间维持现状
    5. 子指数分级 (对调整后的全部成分股按覆盖率统一切分):
       - 大型: 覆盖率 ≤ 77%
       - 中型: 覆盖率 77%-93%
       - 小型: 覆盖率 > 93%
"""

import os
import sys
import logging
import yaml
from datetime import date
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SubIndexType(str, Enum):
    """子指数类型"""
    LARGE = "大型"
    MID = "中型"
    SMALL = "小型"
    NONE = "非成份股"


@dataclass
class StockRanking:
    """单只股票的排名和覆盖率信息"""
    code: str                       # 股票代码 (5位)
    name: str                       # 股票名称
    avg_market_cap: float           # 12个月日均市值 (港币元)
    rank: int                       # 市值排名 (1=最大)
    cumulative_coverage: float      # 累计市值覆盖率 (0-100)
    is_constituent: bool            # 是否为当前HSCI成份股
    current_sub_index: SubIndexType # 当前子指数类型 (仅用于显示, 不影响预测)
    predicted_sub_index: SubIndexType = SubIndexType.NONE  # 预测子指数类型
    buffer_status: str = ""         # 缓冲区状态描述
    is_addition_candidate: bool = True  # 是否可作为调入候选 (低市值非成份股标记为False, 但仍参与覆盖率分母)


@dataclass
class CoveragePrediction:
    """HSCI覆盖率模型预测结果"""
    review_date: date                           # 检讨日期
    cutoff_date: date                           # 数据截止日
    total_universe: int                         # 选股范畴总数
    current_constituent_count: int              # 当前成份股数
    predicted_constituent_count: int            # 预测成份股数
    additions: List[StockRanking] = field(default_factory=list)    # 预测调入
    removals: List[StockRanking] = field(default_factory=list)     # 预测调出
    rankings: List[StockRanking] = field(default_factory=list)     # 完整排名列表
    coverage_at_95: float = 0.0                 # 95%覆盖率处的市值


class HSCICoverageModel:
    """
    HSCI覆盖率预测模型

    用法:
        model = HSCICoverageModel()

        prediction = model.predict(
            market_cap_data=[{code, name, avg_market_cap}, ...],
            current_constituents=[{code, name, index_type}, ...],
            cutoff_date=date(2024, 6, 30),
        )
    """

    # 默认规则参数
    DEFAULT_PARAMS = {
        "coverage_threshold": 95,        # 成份股选择门槛: 累计覆盖率95%
        "buffer_inclusion": 94,          # 非成份股加入门槛
        "buffer_removal": 96,            # 现有成份股剔除门槛
        # 子指数分级缓冲区矩阵 (编算细则)
        # 大型股门槛: 根据当前分类, 覆盖率≤此值则归为/保持大型
        "sub_large_from_large": 83,      # 现有大型股: ≤83% 保持大型
        "sub_large_from_mid": 77,        # 现有中型股: ≤77% 升为大型
        "sub_large_from_small": 77,      # 现有小型股: ≤77% 升为大型
        "sub_large_from_new": 77,        # 非成份股(新调入): ≤77% 归为大型
        # 中型股门槛: 不满足大型门槛时, 覆盖率≤此值则归为/保持中型
        "sub_mid_from_large": 97,        # 现有大型股: ≤97% 降为中型
        "sub_mid_from_mid": 97,          # 现有中型股: ≤97% 保持中型
        "sub_mid_from_small": 93,        # 现有小型股: ≤93% 升为中型
        "sub_mid_from_new": 93,          # 非成份股(新调入): ≤93% 归为中型
    }

    def __init__(self, rule_file: str = None):
        self.rule_file = rule_file or os.path.join(
            PROJECT_ROOT, "rules", "hsci.md"
        )
        self.params = dict(self.DEFAULT_PARAMS)
        self.rule_text = ""
        self._load_rules()

    def _load_rules(self):
        """从hsci.md加载规则参数"""
        try:
            with open(self.rule_file, "r", encoding="utf-8") as f:
                content = f.read()

            # 分离YAML frontmatter
            import re
            pattern = r'^---\s*\n(.*?)\n---\s*\n?(.*)'
            match = re.match(pattern, content, re.DOTALL)

            if match:
                yaml_content = match.group(1).strip()
                rules = yaml.safe_load(yaml_content)
                self.rule_text = match.group(2).strip()

                # 提取参数
                buffer = rules.get("buffer_zone", {})

                self.params["buffer_inclusion"] = buffer.get("inclusion_cutoff", 94)
                self.params["buffer_removal"] = buffer.get("removal_cutoff", 96)

                # 子指数分级缓冲区矩阵
                sub = rules.get("sub_indices", {})
                large_sub = sub.get("large", {})
                mid_sub = sub.get("mid", {})
                self.params["sub_large_from_large"] = large_sub.get("from_large", 83)
                self.params["sub_large_from_mid"] = large_sub.get("from_mid", 77)
                self.params["sub_large_from_small"] = large_sub.get("from_small", 77)
                self.params["sub_large_from_new"] = large_sub.get("from_new", 77)
                self.params["sub_mid_from_large"] = mid_sub.get("from_large", 97)
                self.params["sub_mid_from_mid"] = mid_sub.get("from_mid", 97)
                self.params["sub_mid_from_small"] = mid_sub.get("from_small", 93)
                self.params["sub_mid_from_new"] = mid_sub.get("from_new", 93)

                logger.info(f"HSCI规则加载完成: {self.rule_file}")
            else:
                logger.warning("无法解析YAML frontmatter, 使用默认参数")

        except FileNotFoundError:
            logger.warning(f"规则文件不存在: {self.rule_file}, 使用默认参数")
        except Exception as e:
            logger.warning(f"规则加载失败: {e}, 使用默认参数")

    def predict(
        self,
        market_cap_data: List[Dict],
        current_constituents: List[Dict],
        cutoff_date: date,
        review_date: date = None,
        min_market_cap: float = 0,
        official_count: int = None,
    ) -> CoveragePrediction:
        """
        执行HSCI覆盖率预测

        参数:
            market_cap_data: 市值数据列表
                [{code: "00700", name: "腾讯控股", avg_market_cap: 3.5e12}, ...]
            current_constituents: 当前HSCI成份股列表
                [{code: "00700", name: "腾讯控股", index_type: "恒生综合大型股指数"}, ...]
            cutoff_date: 数据截止日 (如 6月30日 或 12月31日)
            review_date: 检讨日期 (如不提供则默认为cutoff_date)
            min_market_cap: 最低市值过滤阈值 (港币元), 0=不过滤
            official_count: 官方记录的成份股数量 (来自count_after)

        返回:
            CoveragePrediction
        """
        review_date = review_date or cutoff_date

        # 1. 构建成份股查找映射
        constituent_map = {}
        for c in current_constituents:
            code = c["code"].strip().zfill(5)
            index_type = c.get("index_type", "")
            if "大型" in index_type:
                sub_index = SubIndexType.LARGE
            elif "中型" in index_type:
                sub_index = SubIndexType.MID
            elif "小型" in index_type:
                sub_index = SubIndexType.SMALL
            else:
                sub_index = SubIndexType.NONE
            constituent_map[code] = {
                "name": c.get("name", ""),
                "sub_index": sub_index,
            }

        # 2. 构建股票列表
        # 低市值非成份股不作为调入候选, 但仍参与覆盖率分母计算 (避免分母过小导致覆盖率偏高)
        stocks = []
        filtered_low_mc = 0
        for s in market_cap_data:
            code = s["code"].strip().zfill(5)
            name = s.get("name", "")
            avg_mc = float(s.get("avg_market_cap", 0))

            if avg_mc <= 0:
                continue

            is_constituent = code in constituent_map

            # 最低市值过滤 (近似流动性测试)
            # 现有HSCI成份股不受此过滤影响
            # 被过滤的股票仍保留在stocks中参与分母, 仅标记为不可调入
            is_addition_candidate = True
            if not is_constituent and min_market_cap > 0 and avg_mc < min_market_cap:
                filtered_low_mc += 1
                is_addition_candidate = False

            current_sub = constituent_map.get(code, {}).get("sub_index", SubIndexType.NONE)

            stocks.append(StockRanking(
                code=code,
                name=name,
                avg_market_cap=avg_mc,
                rank=0,
                cumulative_coverage=0.0,
                is_constituent=is_constituent,
                current_sub_index=current_sub,
                is_addition_candidate=is_addition_candidate,
            ))

        if filtered_low_mc > 0:
            logger.info(f"标记低市值非成份股(不可调入但仍参与分母): {filtered_low_mc} 只 (阈值: {min_market_cap/1e8:.0f}亿)")

        # 按市值降序排序
        stocks.sort(key=lambda x: x.avg_market_cap, reverse=True)

        # 3. 计算排名和累计覆盖率
        total_market_cap = sum(s.avg_market_cap for s in stocks)
        cumulative = 0.0
        for i, s in enumerate(stocks):
            s.rank = i + 1
            cumulative += s.avg_market_cap
            s.cumulative_coverage = (cumulative / total_market_cap) * 100

        # 4. 应用缓冲区规则预测调入调出
        additions = []
        removals = []

        p = self.params

        for s in stocks:
            if not s.is_constituent:
                # 非成份股: 判断是否应加入
                if not s.is_addition_candidate:
                    s.buffer_status = "不候选 (低市值/未通过换手率测试)"
                elif s.cumulative_coverage <= p["buffer_inclusion"]:
                    s.buffer_status = "加入 (覆盖率≤94%)"
                    additions.append(s)
                elif s.cumulative_coverage <= p["coverage_threshold"]:
                    s.buffer_status = "灰色区间 (94%-95%)"
                else:
                    s.buffer_status = "不加入 (覆盖率>95%)"
            else:
                # 现有成份股: 判断是否应剔除
                if s.cumulative_coverage > p["buffer_removal"]:
                    s.buffer_status = "剔除 (覆盖率>96%)"
                    removals.append(s)
                elif s.cumulative_coverage > p["coverage_threshold"]:
                    s.buffer_status = "灰色区间 (95%-96%)"
                else:
                    s.buffer_status = "保留"

        # 5. 对全部股票统一分配子指数 (基于覆盖率, 不考虑原有分级)
        self._assign_sub_index(stocks)

        # 6. 计算预测成份股总数
        current_count = official_count if official_count is not None else len(constituent_map)
        predicted_count = current_count + len(additions) - len(removals)

        # 95%覆盖率处的市值
        coverage_95_mc = 0.0
        for s in stocks:
            if s.cumulative_coverage >= 95:
                coverage_95_mc = s.avg_market_cap
                break

        logger.info(
            f"预测完成: 选股范畴{len(stocks)}只, "
            f"当前成份股{current_count}只, 预测{predicted_count}只, "
            f"调入{len(additions)}只, 调出{len(removals)}只"
        )

        return CoveragePrediction(
            review_date=review_date,
            cutoff_date=cutoff_date,
            total_universe=len(stocks),
            current_constituent_count=current_count,
            predicted_constituent_count=predicted_count,
            additions=additions,
            removals=removals,
            rankings=stocks,
            coverage_at_95=coverage_95_mc,
        )

    def _assign_sub_index(self, stocks: List[StockRanking]) -> None:
        """
        按编算细则缓冲区矩阵分配子指数

        缓冲区矩阵 (编算细则):
        | 目标分类 | 现有大型股 | 现有中型股 | 现有小型股 | 非成份股 |
        | 大型    | 83%       | 77%       | 77%       | 77%     |
        | 中型    | 97%       | 97%       | 93%       | 93%     |
        | 小型    | 100%      | 100%      | 100%      | 100%    |

        规则:
            1. 先判断是否满足大型门槛 (门槛取决于当前分类)
            2. 不满足大型则判断中型门槛
            3. 都不满足则为小型

        门槛含义:
            - 现有大型股: ≤83%保持大型, ≤97%降为中型, 否则小型
            - 现有中型股: ≤77%升为大型, ≤97%保持中型, 否则小型
            - 现有小型股: ≤77%升为大型, ≤93%升为中型, 否则保持小型
            - 非成份股:   ≤77%归为大型, ≤93%归为中型, 否则小型
        """
        p = self.params

        # 大型股门槛: 根据当前分类查表
        large_thresholds = {
            SubIndexType.LARGE: p["sub_large_from_large"],
            SubIndexType.MID: p["sub_large_from_mid"],
            SubIndexType.SMALL: p["sub_large_from_small"],
            SubIndexType.NONE: p["sub_large_from_new"],
        }
        # 中型股门槛
        mid_thresholds = {
            SubIndexType.LARGE: p["sub_mid_from_large"],
            SubIndexType.MID: p["sub_mid_from_mid"],
            SubIndexType.SMALL: p["sub_mid_from_small"],
            SubIndexType.NONE: p["sub_mid_from_new"],
        }

        for s in stocks:
            current = s.current_sub_index
            cov = s.cumulative_coverage

            # 1. 判断大型
            large_threshold = large_thresholds.get(current, 77)
            if cov <= large_threshold:
                s.predicted_sub_index = SubIndexType.LARGE
                continue

            # 2. 判断中型
            mid_threshold = mid_thresholds.get(current, 93)
            if cov <= mid_threshold:
                s.predicted_sub_index = SubIndexType.MID
                continue

            # 3. 小型
            s.predicted_sub_index = SubIndexType.SMALL
