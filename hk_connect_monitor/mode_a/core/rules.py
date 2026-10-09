"""
港股通标的池监控系统 - 规则引擎

实现《上海证券交易所沪港通业务实施办法》中的调入/调出/排除条件。

核心规则映射：
  - 第七十条：调入条件（小型股市值门槛、指数成份股、A+H股H股）
  - 第七十条第二款：排除条件（A股ST/退市、非港币报价）
  - 第七十一条：WVR首次纳入条件（上市时长、183日市值、183日成交额）
  - 第七十三条：调出条件（不再属于指数成份股、小型股市值低于门槛）
  - 第七十四条：市值标准仅在恒生综合指数定期调整时执行

模式A定位：
  持续扫描可量化边界条件，识别"接近门槛"的股票。
  不预测何时调整，而是持续回答"谁有风险"。
  真正的调入调出执行发生在恒生综合指数定期调整生效日（模式B）。
"""

from datetime import date, timedelta
from typing import List, Optional

from mode_a import config
from mode_a.models import (
    StockInfo, MonthlyMarketCap, DailyMarketData,
    AssessmentResult, IndexType, RiskType, RiskLevel,
    format_hkd,
)


class RuleEngine:
    """规则引擎 - 对单只股票执行全部规则检查"""

    def __init__(self, reference_date: Optional[date] = None):
        self.reference_date = reference_date or date.today()

    def assess_stock(
        self,
        stock: StockInfo,
        monthly_data: List[MonthlyMarketCap],
        daily_data: List[DailyMarketData],
    ) -> AssessmentResult:
        """
        对单只股票进行完整评估。

        评估优先级：
        1. 排除条件检查（第七十条第二款）→ 事件型，最高优先
        2. 第二上市排除（WVR须主要上市方可纳入）→ 不在池时不可纳入
        3. WVR首次纳入条件检查（第七十一条）→ 仅第一上市WVR且不在池时
        4. 小型股市值门槛检查（第七十条第三项 / 第七十三条第二款）
        5. 其他股票（大型/中型/A+H）→ 默认安全
        """
        # 1. 排除条件（最高优先级）
        result = self._check_exclusion(stock)
        if result:
            return result

        # 2. 第二上市且不在池 → 不可纳入（WVR须主要上市后方可纳入）
        if stock.is_secondary_listing and not stock.in_pool:
            return AssessmentResult(
                code=stock.code,
                name=stock.name,
                index_type=stock.index_type,
                in_pool=False,
                risk_type=RiskType.EXCLUSION_EVENT,
                risk_level=RiskLevel.LOW,
                details=(
                    f"第二上市股票（{'-SW' if stock.is_wvr else '-S'}后缀），"
                    f"不可纳入港股通。"
                    f"{'WVR公司须转换为在港主要上市后方可纳入' if stock.is_wvr else '需转换为双重主要上市后方可纳入'}"
                    f"（实施办法第六十五条、第七十一条）"
                ),
                metrics={
                    "is_secondary_listing": True,
                    "is_wvr": stock.is_wvr,
                },
            )

        # 3. WVR首次纳入条件（仅第一上市WVR）
        if stock.is_wvr and not stock.in_pool:
            return self._assess_wvr_inclusion(stock, daily_data)

        # 4. 小型股市值门槛
        if stock.index_type == IndexType.HSCI_SMALL:
            return self._assess_small_cap(stock, monthly_data)

        # 5. 其他：大型股/中型股/非指数成份股/A+H股 → 无量化门槛，默认安全
        return AssessmentResult(
            code=stock.code,
            name=stock.name,
            index_type=stock.index_type,
            in_pool=stock.in_pool,
            risk_type=RiskType.SAFE,
            risk_level=RiskLevel.NONE,
            details=f"{stock.index_type.value}，无额外量化门槛",
        )

    # ============================================================
    # 规则一：排除条件检查（第七十条第二款）
    # ============================================================

    def _check_exclusion(self, stock: StockInfo) -> Optional[AssessmentResult]:
        """
        检查是否触発排除条件：
        (1) A+H公司A股为ST或退市整理 → 对应H股不纳入/应调出
        (2) 非港币报价 → 不纳入
        (3) 其他本所认定情形（本系统不覆盖）
        """
        # (1) A+H公司A股被ST或处于退市整理期
        if stock.is_a_plus_h and (stock.a_share_st or stock.a_share_delisting):
            st_label = "ST/*ST" if stock.a_share_st else "退市整理"
            return AssessmentResult(
                code=stock.code,
                name=stock.name,
                index_type=stock.index_type,
                in_pool=stock.in_pool,
                risk_type=RiskType.EXCLUSION_EVENT,
                risk_level=RiskLevel.HIGH,
                details=(
                    f"A+H股公司A股({stock.a_share_code})处于{st_label}状态，"
                    f"对应H股{'应调出港股通' if stock.in_pool else '不纳入港股通'}"
                    f"（实施办法第七十条第二款第一项）"
                ),
                metrics={
                    "a_share_code": stock.a_share_code,
                    "a_share_st": stock.a_share_st,
                    "a_share_delisting": stock.a_share_delisting,
                },
            )

        # (2) 非港币报价
        if stock.currency != config.ELIGIBLE_CURRENCY:
            return AssessmentResult(
                code=stock.code,
                name=stock.name,
                index_type=stock.index_type,
                in_pool=stock.in_pool,
                risk_type=RiskType.EXCLUSION_EVENT,
                risk_level=RiskLevel.HIGH,
                details=(
                    f"以{stock.currency}报价交易，"
                    f"{'不纳入港股通' if not stock.in_pool else '应调出港股通'}"
                    f"（实施办法第七十条第二款第二项）"
                ),
                metrics={
                    "currency": stock.currency,
                    "eligible_currency": config.ELIGIBLE_CURRENCY,
                },
            )

        return None

    # ============================================================
    # 规则二：小型股市值门槛（第七十条第三项 / 第七十三条第二款）
    # ============================================================

    def _assess_small_cap(
        self,
        stock: StockInfo,
        monthly_data: List[MonthlyMarketCap],
    ) -> AssessmentResult:
        """
        评估恒生综合小型股指数成份股的市值条件。

        调入线：前12个月平均月末市值 >= 50亿港币（第七十条第三项）
        调出线：前12个月平均月末市值 < 40亿港币（第七十三条第二款）
        灰色区间：40亿 ~ 50亿（维持现状，但需关注）

        注意：市值标准仅在恒生综合指数定期调整时执行（第七十四条）。
        本评估输出的是"风险距离"，不是"即刻调入调出"。
        """
        if not monthly_data:
            return AssessmentResult(
                code=stock.code,
                name=stock.name,
                index_type=stock.index_type,
                in_pool=stock.in_pool,
                risk_type=RiskType.SAFE,
                risk_level=RiskLevel.NONE,
                details="数据不足，无法评估",
            )

        # 计算前12个月（或实际可用月份）平均月末市值
        # 实施办法规定：上市不足12个月的按实际上市时间计算
        avg_market_cap = sum(d.market_cap for d in monthly_data) / len(monthly_data)
        months_used = len(monthly_data)

        inclusion_line = config.SMALL_CAP_INCLUSION_MARKET_CAP
        removal_line = config.SMALL_CAP_REMOVAL_MARKET_CAP

        if stock.in_pool:
            # === 在池股票：评估调出风险 ===
            if avg_market_cap < removal_line:
                # 低于调出线
                distance = (removal_line - avg_market_cap) / removal_line
                risk_level = self._calc_risk_level_by_distance(distance)
                return AssessmentResult(
                    code=stock.code,
                    name=stock.name,
                    index_type=stock.index_type,
                    in_pool=True,
                    risk_type=RiskType.REMOVAL_RISK,
                    risk_level=risk_level,
                    details=(
                        f"前{months_used}个月平均月末市值 {format_hkd(avg_market_cap)}，"
                        f"低于调出线 {format_hkd(removal_line)}，"
                        f"距边界 {distance:+.1%}。"
                        f"下次恒生综合指数定期调整生效日有调出风险"
                        f"（第七十三条第二款、第七十四条）"
                    ),
                    metrics={
                        "avg_monthly_market_cap": avg_market_cap,
                        "removal_line": removal_line,
                        "inclusion_line": inclusion_line,
                        "distance_to_removal": distance,
                        "months_used": months_used,
                    },
                )
            elif avg_market_cap < inclusion_line:
                # 灰色区间（40-50亿）
                distance_to_removal = (avg_market_cap - removal_line) / removal_line
                risk_level = RiskLevel.MEDIUM
                return AssessmentResult(
                    code=stock.code,
                    name=stock.name,
                    index_type=stock.index_type,
                    in_pool=True,
                    risk_type=RiskType.GREY_ZONE,
                    risk_level=risk_level,
                    details=(
                        f"前{months_used}个月平均月末市值 {format_hkd(avg_market_cap)}，"
                        f"处于灰色区间({format_hkd(removal_line)}~{format_hkd(inclusion_line)})，"
                        f"距调出线 {distance_to_removal:+.1%}。"
                        f"暂不调出，但需持续关注"
                    ),
                    metrics={
                        "avg_monthly_market_cap": avg_market_cap,
                        "removal_line": removal_line,
                        "inclusion_line": inclusion_line,
                        "distance_to_removal": distance_to_removal,
                        "months_used": months_used,
                    },
                )
            else:
                # 安全（高于调入线）
                return AssessmentResult(
                    code=stock.code,
                    name=stock.name,
                    index_type=stock.index_type,
                    in_pool=True,
                    risk_type=RiskType.SAFE,
                    risk_level=RiskLevel.NONE,
                    details=(
                        f"前{months_used}个月平均月末市值 {format_hkd(avg_market_cap)}，"
                        f"高于调入线 {format_hkd(inclusion_line)}，市值充足"
                    ),
                    metrics={
                        "avg_monthly_market_cap": avg_market_cap,
                        "inclusion_line": inclusion_line,
                        "months_used": months_used,
                    },
                )
        else:
            # === 不在池股票：评估调入可能性 ===
            if avg_market_cap >= inclusion_line:
                # 达到调入标准
                distance = (avg_market_cap - inclusion_line) / inclusion_line
                risk_level = self._calc_risk_level_by_distance(distance, ascending=True)
                return AssessmentResult(
                    code=stock.code,
                    name=stock.name,
                    index_type=stock.index_type,
                    in_pool=False,
                    risk_type=RiskType.INCLUSION_CANDIDATE,
                    risk_level=risk_level,
                    details=(
                        f"前{months_used}个月平均月末市值 {format_hkd(avg_market_cap)}，"
                        f"达到调入标准({format_hkd(inclusion_line)})，"
                        f"超出边界 {distance:+.1%}。"
                        f"下次恒生综合指数定期调整生效日有调入可能"
                        f"（第七十条第三项、第七十四条）"
                    ),
                    metrics={
                        "avg_monthly_market_cap": avg_market_cap,
                        "inclusion_line": inclusion_line,
                        "distance_to_inclusion": distance,
                        "months_used": months_used,
                    },
                )
            elif avg_market_cap >= removal_line:
                # 灰色区间（40-50亿）
                distance_to_inclusion = (inclusion_line - avg_market_cap) / inclusion_line
                risk_level = RiskLevel.MEDIUM
                return AssessmentResult(
                    code=stock.code,
                    name=stock.name,
                    index_type=stock.index_type,
                    in_pool=False,
                    risk_type=RiskType.GREY_ZONE,
                    risk_level=risk_level,
                    details=(
                        f"前{months_used}个月平均月末市值 {format_hkd(avg_market_cap)}，"
                        f"处于灰色区间({format_hkd(removal_line)}~{format_hkd(inclusion_line)})，"
                        f"距调入线 {distance_to_inclusion:+.1%}。"
                        f"暂不调入，但需持续关注"
                    ),
                    metrics={
                        "avg_monthly_market_cap": avg_market_cap,
                        "removal_line": removal_line,
                        "inclusion_line": inclusion_line,
                        "distance_to_inclusion": distance_to_inclusion,
                        "months_used": months_used,
                    },
                )
            else:
                # 远低于调入线
                return AssessmentResult(
                    code=stock.code,
                    name=stock.name,
                    index_type=stock.index_type,
                    in_pool=False,
                    risk_type=RiskType.SAFE,
                    risk_level=RiskLevel.NONE,
                    details=(
                        f"前{months_used}个月平均月末市值 {format_hkd(avg_market_cap)}，"
                        f"低于调出线 {format_hkd(removal_line)}，暂无调入可能"
                    ),
                    metrics={
                        "avg_monthly_market_cap": avg_market_cap,
                        "removal_line": removal_line,
                        "months_used": months_used,
                    },
                )

    # ============================================================
    # 规则三：WVR首次纳入条件（第七十一条）
    # ============================================================

    def _assess_wvr_inclusion(
        self,
        stock: StockInfo,
        daily_data: List[DailyMarketData],
    ) -> AssessmentResult:
        """
        评估WVR（不同投票权架构）股票的首次纳入条件。

        条件（全部满足才可首次纳入）：
        (1) 在联交所上市满6个月及其后20个港股交易日
        (2) 首次纳入考察日前183日（含当日）日均市值 >= 200亿港币
        (3) 首次纳入考察日前183日（含当日）港股总成交额 >= 60亿港币
        (4) 未因违反联交所规定受到公开指责等（本系统不覆盖，需人工判断）
        (5) 本所认定的其他条件（本系统不覆盖）

        首次纳入考察日 = 上市满6个月后的第19个港股交易日
        （本系统近似为上市日后约6个月零27个日历日）
        """
        # 条件(1)：上市时长检查
        listing_days_needed = (
            config.WVR_LISTING_MIN_MONTHS * 30  # 近似6个月
            + int(config.WVR_LISTING_EXTRA_TRADING_DAYS * config.TRADING_DAYS_TO_CALENDAR_DAYS)
        )
        days_since_listing = (self.reference_date - stock.listing_date).days
        listing_months_actual = days_since_listing / 30.0

        listing_met = days_since_listing >= listing_days_needed

        # 计算首次纳入考察日（近似）
        assessment_date = stock.listing_date + timedelta(days=listing_days_needed)

        # 条件(2)(3)：183日市值和成交额
        if not daily_data:
            return AssessmentResult(
                code=stock.code,
                name=stock.name,
                index_type=stock.index_type,
                in_pool=False,
                risk_type=RiskType.WVR_PENDING,
                risk_level=RiskLevel.NONE,
                details="数据不足，无法评估WVR条件",
                metrics={"listing_met": listing_met},
            )

        # 取最近183个日历日内的交易日数据
        cutoff_date = self.reference_date - timedelta(days=config.WVR_ASSESSMENT_DAYS)
        recent_data = [d for d in daily_data if d.date >= cutoff_date]

        if not recent_data:
            return AssessmentResult(
                code=stock.code,
                name=stock.name,
                index_type=stock.index_type,
                in_pool=False,
                risk_type=RiskType.WVR_PENDING,
                risk_level=RiskLevel.NONE,
                details="近183日无交易日数据，无法评估",
                metrics={"listing_met": listing_met},
            )

        avg_daily_market_cap = sum(d.market_cap for d in recent_data) / len(recent_data)
        total_trading_value = sum(d.trading_value for d in recent_data)

        cap_met = avg_daily_market_cap >= config.WVR_MARKET_CAP_THRESHOLD
        value_met = total_trading_value >= config.WVR_TRADING_VALUE_THRESHOLD

        all_met = listing_met and cap_met and value_met

        # 构建详情
        checks = []
        checks.append(f"上市时长 {'✓' if listing_met else '✗'} "
                      f"({listing_months_actual:.0f}个月 / 需{config.WVR_LISTING_MIN_MONTHS}个月+{config.WVR_LISTING_EXTRA_TRADING_DAYS}交易日)")
        checks.append(f"183日日均市值 {'✓' if cap_met else '✗'} "
                      f"({format_hkd(avg_daily_market_cap)} / 需{format_hkd(config.WVR_MARKET_CAP_THRESHOLD)})")
        checks.append(f"183日成交总额 {'✓' if value_met else '✗'} "
                      f"({format_hkd(total_trading_value)} / 需{format_hkd(config.WVR_TRADING_VALUE_THRESHOLD)})")

        if all_met:
            risk_level = RiskLevel.HIGH
            details = "满足全部首次纳入量化条件。" + " ".join(checks)
        else:
            # 判断距离最近门槛的距离
            distances = []
            if not cap_met:
                distances.append((config.WVR_MARKET_CAP_THRESHOLD - avg_daily_market_cap) / config.WVR_MARKET_CAP_THRESHOLD)
            if not value_met:
                distances.append((config.WVR_TRADING_VALUE_THRESHOLD - total_trading_value) / config.WVR_TRADING_VALUE_THRESHOLD)
            min_distance = min(distances) if distances else 0
            risk_level = self._calc_risk_level_by_distance(min_distance)
            details = "未满足全部首次纳入条件。" + " ".join(checks)

        return AssessmentResult(
            code=stock.code,
            name=stock.name,
            index_type=stock.index_type,
            in_pool=False,
            risk_type=RiskType.WVR_PENDING,
            risk_level=risk_level,
            details=details,
            metrics={
                "listing_met": listing_met,
                "listing_months_actual": round(listing_months_actual, 1),
                "listing_months_required": config.WVR_LISTING_MIN_MONTHS,
                "assessment_date_estimated": assessment_date.isoformat(),
                "avg_daily_market_cap_183d": avg_daily_market_cap,
                "market_cap_threshold": config.WVR_MARKET_CAP_THRESHOLD,
                "cap_met": cap_met,
                "total_trading_value_183d": total_trading_value,
                "trading_value_threshold": config.WVR_TRADING_VALUE_THRESHOLD,
                "value_met": value_met,
                "all_conditions_met": all_met,
                "trading_days_in_window": len(recent_data),
            },
        )

    # ============================================================
    # 辅助方法
    # ============================================================

    def _calc_risk_level_by_distance(
        self, distance: float, ascending: bool = False
    ) -> RiskLevel:
        """
        根据距边界的百分比计算风险等级。

        参数：
            distance: 距边界的比例（正数=超出边界，负数=未达边界）
            ascending: True表示"超出越多风险越高"（如调入候选超出调入线），
                       False表示"低于越多风险越高"（如调出风险低于调出线）

        对于调出风险（ascending=False）：
            distance = (removal_line - avg) / removal_line
            distance >= 0 表示已低于调出线
            distance 越大风险越高

        对于调入候选（ascending=True）：
            distance = (avg - inclusion_line) / inclusion_line
            distance >= 0 表示已超出调入线
            distance 越大可能性越高
        """
        abs_dist = abs(distance)
        if abs_dist <= config.RISK_HIGH_THRESHOLD:
            return RiskLevel.HIGH
        elif abs_dist <= config.RISK_MEDIUM_THRESHOLD:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
