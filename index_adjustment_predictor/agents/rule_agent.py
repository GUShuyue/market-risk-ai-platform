"""
Rule Agent - 规则计算Agent

职责：加载指数编制规则，计算候选股票排名
每个指数有独立的规则文件 (rules/xxx.md)，规则文件包含YAML frontmatter + Markdown正文
"""

from typing import Dict, List
from rules.rule_parser import RuleParser


class RuleAgent:
    """
    规则计算Agent - 基于指数编制规则进行量化评分
    
    支持的指数:
        - csi300: 沪深300
    """

    # 指数规则文件映射
    RULE_FILES = {
        "csi300": "rules/csi300.md",
    }

    def __init__(self, index_name: str = None):
        self.index_name = index_name
        self.parser = RuleParser()
        self.rules = self._load_rules() if index_name else None

    def run(self, inputs: dict) -> dict:
        """
        统一接口：加载规则并计算排名

        inputs 必含: index_name, stock_data (DataAgent.collect 的完整返回)
        返回: {"rule_result": {...}}
        """
        self.index_name = inputs["index_name"]
        self.rules = self._load_rules()
        return {"rule_result": self.calculate_scores(inputs["stock_data"])}

    def _load_rules(self) -> dict:
        """加载指数编制规则（从YAML frontmatter解析）"""
        rule_file = self.RULE_FILES.get(self.index_name)
        if not rule_file:
            raise ValueError(f"不支持的指数: {self.index_name}，目前仅支持: csi300")
        return self.parser.parse(rule_file)

    def get_rule_text(self) -> str:
        """获取规则文件的Markdown正文（供LLM作为context阅读）"""
        rule_file = self.RULE_FILES.get(self.index_name)
        if not rule_file:
            return ""
        with open(rule_file, "r", encoding="utf-8") as f:
            content = f.read()
        # 去掉YAML frontmatter，只返回Markdown正文
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
        return content

    def calculate_scores(self, stock_data: dict) -> dict:
        """
        根据编制规则计算候选股票排名

        参数:
            stock_data: DataAgent采集的市场数据

        返回:
            {
                "rankings": [...],        # 按规则排序的候选股票
                "eligible": [...],        # 满足准入条件的股票
                "ineligible": [...],      # 不满足条件的股票及原因
                "current_rank": {...}     # 当前成分股排名
            }
        """
        # 1. 筛选满足准入条件的股票
        cutoff_date = stock_data.get("cutoff_date", "")
        eligible = self._filter_eligible_stocks(stock_data, cutoff_date)

        # 2. 计算综合排名
        rankings = self._calculate_rankings(eligible, stock_data)

        # 3. 标记当前成分股的排名位置
        current_rank = self._mark_current_constituents(
            rankings, stock_data.get("current_constituents", [])
        )

        return {
            "index_name": self.index_name,
            "rules": self.rules,
            "rankings": rankings,
            "eligible": eligible,
            "ineligible": self._get_ineligible_reasons(stock_data),
            "current_rank": current_rank,
            "rules_version": self.rules.get("document_version", "unknown")
        }

    def _filter_eligible_stocks(self, stock_data: dict, cutoff_date: str = "") -> List[dict]:
        """
        筛选满足准入条件的股票
        
        根据规则的YAML frontmatter中的sample_space条件筛选：
        - 上市时间检查（区分科创板/创业板和其他）
        - ST状态检查
        - 退市风险检查
        - 财务亏损检查（仅排除非成分股，科创板除外）
        """
        market_data = stock_data.get("market_data", [])
        listing_info = stock_data.get("listing_info", [])
        financial_flags = stock_data.get("financial_flags", {})

        # 当前成分股代码集合（成分股不受财务亏损排除限制）
        current_constituent_codes = {
            c.get("con_code") or c.get("ts_code", "")
            for c in stock_data.get("current_constituents", [])
            if isinstance(c, dict)
        }

        sample_space = self.rules.get("sample_space", {})
        listing_rules = sample_space.get("listing_time_rules", [])
        exclusions = sample_space.get("exclusions", [])

        eligible = []
        ineligible_list = []

        for stock in market_data:
            stock_code = stock.get("ts_code", "")
            stock_name = stock.get("name", "")
            inelig_reason = None

            # ST状态检查
            is_st = "ST" in stock_name or "*ST" in stock_name
            if is_st and "ST、*ST股票" in exclusions:
                inelig_reason = "ST/*ST股票"

            # 北交所股票排除 (沪深300样本空间不含北交所)
            if not inelig_reason and stock_code.endswith(".BJ"):
                inelig_reason = "北交所股票"

            # 上市时间检查
            if not inelig_reason:
                info = next(
                    (s for s in listing_info if s.get("ts_code") == stock_code),
                    None
                )
                if info:
                    inelig_reason = self._check_listing_time(
                        stock_code, stock_name, info, listing_rules, cutoff_date
                    )

            # 财务亏损检查
            # 编制规则: "财务亏损的证券原则上不列为候选新样本"
            # 例外: "科创板上市证券除外"
            # 仅对非成分股生效（成分股按排名决定去留，不受此限制）
            if not inelig_reason:
                inelig_reason = self._check_financial_loss(
                    stock_code, stock_name,
                    financial_flags,
                    current_constituent_codes
                )

            if inelig_reason:
                ineligible_list.append({
                    "code": stock_code,
                    "name": stock_name,
                    "reason": inelig_reason
                })
            else:
                eligible.append(stock)

        self._ineligible_cache = ineligible_list
        return eligible

    def _check_financial_loss(
        self,
        stock_code: str,
        stock_name: str,
        financial_flags: dict,
        current_constituent_codes: set
    ) -> str:
        """
        检查财务亏损

        规则: "财务亏损的证券原则上不列为候选新样本"
        例外: "科创板上市证券除外"
        限制: 仅对非成分股生效

        返回: 不合格原因字符串，合格返回 None
        """
        # 成分股不受此限制
        if stock_code in current_constituent_codes:
            return None

        # 科创板除外
        board = self._get_board(stock_code)
        if board == "科创板":
            return None

        # 检查财务数据
        flag = financial_flags.get(stock_code)
        if not flag:
            return None  # 无财务数据，放行（避免误杀数据缺失的股票）

        if flag.get("is_loss"):
            net_profit = flag.get("net_profit", 0)
            return f"财务亏损(净利润={net_profit/1e8:.2f}亿)"

        return None

    def _check_listing_time(
        self, code: str, name: str, info: dict, listing_rules: list,
        cutoff_date: str = ""
    ) -> str:
        """
        检查上市时间是否满足要求

        根据规则的listing_time_rules:
            - 科创板/创业板: 至少满12个月
            - 其他板: 至少满12个月 (部分指数为3个月)
        返回不合格原因字符串，合格则返回None
        """
        listing_months = info.get("listing_months", 0)
        if listing_months <= 0:
            list_date = info.get("list_date", "")
            if list_date:
                try:
                    from datetime import datetime
                    list_dt = datetime.strptime(str(list_date), "%Y%m%d")
                    # 使用cutoff_date而非datetime.now()，避免看到未来数据
                    ref_date = (
                        datetime.strptime(cutoff_date, "%Y-%m-%d")
                        if cutoff_date else datetime.now()
                    )
                    listing_months = (ref_date.year - list_dt.year) * 12 + (
                        ref_date.month - list_dt.month
                    )
                except ValueError:
                    pass

        if listing_months <= 0:
            return None  # 无法判断，放行

        # 判断板块
        board = self._get_board(code)

        # 匹配规则
        for rule in listing_rules:
            rule_board = rule.get("board", "")
            if rule_board == board or rule_board == "其他":
                min_months = rule.get("min_months", 0)
                if listing_months < min_months:
                    return f"上市时间不足{min_months}个月 (实际{listing_months}个月, {board})"
                return None

        return None

    @staticmethod
    def _get_board(code: str) -> str:
        """根据股票代码判断板块"""
        if code.startswith(("688", "689")):
            return "科创板"
        elif code.startswith(("300", "301")):
            return "创业板"
        elif code.endswith(".BJ"):
            return "北交所"
        else:
            return "其他"

    def _calculate_rankings(self, eligible_stocks: List[dict], stock_data: dict = None) -> List[dict]:
        """
        计算综合排名

        沪深300选样方法: 先按日均成交金额排名剔除后50%，再按日均总市值排名

        参数:
            eligible_stocks: 满足准入条件的股票列表
            stock_data: 完整的市场数据字典，包含 current_constituents
        """
        if not eligible_stocks:
            return []

        stock_data = stock_data or {}
        selection = self.rules.get("selection_method", {})
        buffer = self.rules.get("buffer_zone", {})
        sample_count = self.rules.get("sample_count", 300)

        # Step 1: 按日均成交金额排名
        sorted_by_volume = sorted(
            eligible_stocks,
            key=lambda x: x.get("avg_amount", 0),
            reverse=True
        )

        # 剔除后50%
        volume_cutoff = int(len(sorted_by_volume) * 0.5)

        # 老样本成交金额缓冲区
        # 编制规则§6.4: 老样本日均成交金额在样本空间中排名前60%才可参与市值排名
        # 即：被成交额后50%剔除的老样本中，只有成交额排名在前60%的才能保留
        old_sample_volume_rule = buffer.get("old_sample_volume_rule", "")
        volume_filtered = sorted_by_volume[:volume_cutoff]

        if old_sample_volume_rule:
            current_constituent_codes = {
                c.get("con_code") or c.get("ts_code", "")
                for c in stock_data.get("current_constituents", [])
                if isinstance(c, dict)
            }
            if current_constituent_codes:
                # 前60%的阈值：只有成交额排名在前60%的老样本才保留
                volume_60pct_cutoff = int(len(sorted_by_volume) * 0.6)
                top_60pct_codes = {
                    s.get("ts_code") for s in sorted_by_volume[:volume_60pct_cutoff]
                }

                removed = sorted_by_volume[volume_cutoff:]
                old_samples_kept = [
                    s for s in removed
                    if s.get("ts_code") in current_constituent_codes
                    and s.get("ts_code") in top_60pct_codes
                ]
                volume_filtered = volume_filtered + old_samples_kept

        # Step 2: 按日均总市值排名
        # avg_total_mv=0 表示全年日均市值数据缺失（新股或数据获取失败）
        # 对于非成分股：排除出排名候选（不能用单日快照冒充全年均值）
        # 对于成分股：回退到 total_mv（避免因数据缺失误判调出）
        current_constituent_codes = set()
        if old_sample_volume_rule:
            current_constituent_codes = {
                c.get("con_code") or c.get("ts_code", "")
                for c in stock_data.get("current_constituents", [])
                if isinstance(c, dict)
            }

        # 分离有/无 avg_total_mv 的股票
        stocks_with_avg = [
            s for s in volume_filtered
            if (s.get("avg_total_mv", 0) or 0) > 0
        ]
        stocks_without_avg = [
            s for s in volume_filtered
            if (s.get("avg_total_mv", 0) or 0) <= 0
        ]

        # 有日均市值的正常排名
        sorted_with_avg = sorted(
            stocks_with_avg,
            key=lambda x: x.get("avg_total_mv", 0),
            reverse=True
        )

        # 无日均市值的：成分股用 total_mv 回退，非成分股排到最后
        constituent_fallback = [
            s for s in stocks_without_avg
            if s.get("ts_code") in current_constituent_codes
        ]
        non_constituent_no_data = [
            s for s in stocks_without_avg
            if s.get("ts_code") not in current_constituent_codes
        ]

        sorted_fallback = sorted(
            constituent_fallback,
            key=lambda x: x.get("total_mv", 0),
            reverse=True
        )

        # 最终排序：有日均市值 > 成分股回退 > 非成分股无数据
        sorted_by_mv = sorted_with_avg + sorted_fallback + non_constituent_no_data

        # 赋予排名
        rankings = []
        for i, stock in enumerate(sorted_by_mv):
            rank = i + 1
            stock_rank = {
                "code": stock.get("ts_code"),
                "name": stock.get("name"),
                "rank": rank,
                "total_mv": stock.get("total_mv", 0),
                "avg_total_mv": stock.get("avg_total_mv", 0),
                "avg_amount": stock.get("avg_amount", 0),
                "in_buffer_zone": (
                    rank <= buffer.get("priority_in_rank", 0)
                ),
                "out_buffer_zone": (
                    rank > buffer.get("priority_out_rank", 999999)
                ),
            }
            rankings.append(stock_rank)

        return rankings

    def _mark_current_constituents(self, rankings, constituents):
        """标记当前成分股在排名中的位置"""
        if isinstance(constituents, list) and constituents:
            if isinstance(constituents[0], dict):
                constituent_codes = {c.get("con_code") or c.get("ts_code") for c in constituents}
            else:
                constituent_codes = set(constituents)
        else:
            constituent_codes = set()

        result = {}
        for stock in rankings:
            if stock["code"] in constituent_codes:
                result[stock["code"]] = {
                    "rank": stock["rank"],
                    "in_buffer_zone": stock["in_buffer_zone"],
                    "out_buffer_zone": stock["out_buffer_zone"]
                }
        return result

    def _get_ineligible_reasons(self, stock_data: dict) -> List[dict]:
        """获取不满足条件的股票及原因"""
        # 使用 _filter_eligible_stocks 中缓存的不合格列表
        if hasattr(self, "_ineligible_cache"):
            return self._ineligible_cache
        return []
