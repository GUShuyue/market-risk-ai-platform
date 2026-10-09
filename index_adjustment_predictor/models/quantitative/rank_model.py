"""排名预测模型 - 量化模型核心"""

from typing import List


class RankModel:
    """
    排名预测模型

    基于日均总市值排名和缓冲区规则预测调入调出

    缓冲区规则（以沪深300为例，sample_count=300, buffer_in=240, buffer_out=360）:
        1. 对所有候选股票按日均总市值降序排名
        2. 新样本排名 ≤ buffer_in（240）: 优先进入
        3. 老样本排名 ≤ buffer_out（360）: 优先保留
        4. 总数维持 sample_count（300）不变

    选样逻辑（符合编制规则真实流程）:
        步骤1: 确定基础名单 —— 排名1-300的股票天然入选（不管新老）
        步骤2: 缓冲区调整 —— 老样本排名301-360优先保留，挤占新样本241-300的名额
               但新样本排名1-240不会被挤占（优先进入权高于老样本缓冲区）
        步骤3: 补位 —— 若仍不足sample_count，从剩余候选按排名补位

    实现方式:
        给每只股票计算一个"选样优先级"，按优先级降序排序后取前sample_count只。
        优先级设计确保上述规则自然满足，无需复杂的分层截断逻辑。
    """

    def __init__(self, index_config: dict):
        self.sample_count = index_config["sample_count"]
        self.buffer_in = index_config["buffer_in"]
        self.buffer_out = index_config["buffer_out"]

    def _compute_priority(self, rank: int, is_constituent: bool) -> int:
        """
        计算股票的选样优先级（数值越大越优先入选）

        优先级分层（以沪深300为例，sample_count=300, buffer_in=240, buffer_out=360）:
            第1层 (优先级 5000+): 排名 ≤ buffer_in（240）的股票，不管新老
                - 新样本排名1-240优先进入，不会被老样本缓冲区挤占
                - 老样本排名1-240也在其中（稳定保留）
            第2层 (优先级 4000+): 老样本排名 buffer_in+1 ~ sample_count（241-300）
                - 老样本排名在前300内，稳定保留
            第3层 (优先级 3000+): 老样本排名 sample_count+1 ~ buffer_out（301-360）
                - 缓冲区规则: 优先保留，挤占新样本241-300的名额
            第4层 (优先级 2000+): 新样本排名 buffer_in+1 ~ sample_count（241-300）
                - 被老样本301-360挤占后的补位候选
            第5层 (优先级 600-700): 非成分股排名 > sample_count
                - 按排名补位（老样本>360不在此层，将被调出）
            第6层 (优先级 -1000): 老样本排名 > buffer_out
                - 确定调出

        参数:
            rank: 日均总市值排名（1=最大）
            is_constituent: 是否为当前成分股

        返回:
            优先级数值（越大越优先）
        """
        if rank <= self.buffer_in:
            # 第1层: 排名1-240，不管新老
            # rank=1 → 5239, rank=240 → 5000
            return 5000 + (self.buffer_in - rank)
        elif is_constituent and rank <= self.sample_count:
            # 第2层: 老样本排名241-300（稳定保留）
            # rank=241 → 4159, rank=300 → 4100
            return 4000 + (self.sample_count - rank)
        elif is_constituent and rank <= self.buffer_out:
            # 第3层: 老样本排名301-360（缓冲区保留，挤占新样本241-300）
            # rank=301 → 3359, rank=360 → 3300
            return 3000 + (self.buffer_out - rank)
        elif (not is_constituent) and rank <= self.sample_count:
            # 第4层: 新样本排名241-300（被挤占的补位候选）
            # rank=241 → 2159, rank=300 → 2100
            return 2000 + (self.sample_count - rank)
        elif not is_constituent:
            # 第5层: 非成分股排名>300，按排名补位
            # rank=301 → 699, rank=500 → 500
            return 1000 - rank
        else:
            # 第6层: 老样本排名>buffer_out，确定调出
            return -1000

    def predict(self, rankings: List[dict], current_constituents: List[str]) -> dict:
        """
        执行排名预测

        参数:
            rankings: 所有候选股票的排名列表（已按日均总市值降序排列）
            current_constituents: 当前成分股代码列表

        返回:
            {
                "add_candidates": [...],        # 调入候选
                "add_alternatives": [...],      # 备选候选
                "remove_candidates": [...],     # 调出候选
                "adjustment_count": int         # 调整数
            }
        """
        current_set = set(current_constituents)

        # === 计算每只股票的选样优先级 ===
        for stock in rankings:
            stock["_priority"] = self._compute_priority(
                stock["rank"], stock["code"] in current_set
            )

        # === 按优先级降序排序（优先级相同时排名靠前的优先）===
        sorted_stocks = sorted(
            rankings,
            key=lambda s: (-s["_priority"], s["rank"])
        )

        # === 取前 sample_count 只作为最终入选名单 ===
        final_constituents = sorted_stocks[: self.sample_count]
        final_codes = {s["code"] for s in final_constituents}

        # === 确定调入调出 ===
        # 调入: 最终入选名单中原本不是成分股的
        add_primary = [
            s for s in final_constituents
            if s["code"] not in current_set
        ]

        # 调出: 原300只成分股中不在新选的300只里的
        # 遍历全部当前成分股（而非 rankings），这样被样本空间
        # 剔除的成分股（如ST/退市）也能被正确计入调出
        rank_map = {s["code"]: s for s in rankings}
        remove_candidates = []
        for code in current_constituents:
            if code in final_codes:
                continue
            if code in rank_map:
                # 在 rankings 里但排名靠后没选上
                remove_candidates.append(rank_map[code])
            else:
                # 不在 rankings 里（被样本空间剔除，如ST/退市/上市时间不足）
                remove_candidates.append({
                    "code": code,
                    "name": "",
                    "rank": 999999,
                    "total_mv": 0,
                    "avg_total_mv": 0,
                    "buffer_status": "excluded",
                })

        # 备选: 未入选的非成分股中排名靠前的5只
        add_alternatives = [
            s for s in sorted_stocks[self.sample_count:]
            if s["code"] not in current_set
        ][:5]

        # === 标记缓冲区状态 ===
        for stock in add_primary:
            if stock["rank"] <= self.buffer_in:
                stock["buffer_status"] = "in_buffer"
            elif stock["rank"] <= self.sample_count:
                stock["buffer_status"] = "normal"
            else:
                stock["buffer_status"] = "supplementary"

        for stock in remove_candidates:
            if stock.get("rank", 0) >= 999999:
                stock["buffer_status"] = "excluded"  # 样本空间剔除
            elif stock["rank"] > self.buffer_out:
                stock["buffer_status"] = "out_buffer"
            else:
                stock["buffer_status"] = "buffered_out"

        # 清理临时字段
        for stock in rankings:
            stock.pop("_priority", None)

        return {
            "add_candidates": add_primary,
            "add_alternatives": add_alternatives,
            "remove_candidates": remove_candidates,
            "adjustment_count": len(remove_candidates),
        }
