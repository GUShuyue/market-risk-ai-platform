"""
生成 A+H 股映射清单

数据源:
  - AKShare stock_zh_ah_name: 返回所有 A+H 上市公司的港股代码和名称
  - Tushare stock_basic: 返回所有 A 股的代码和名称

匹配逻辑:
  港股名称和 A 股名称常有差异(如"比亚迪股份" vs "比亚迪"),
  需要多轮匹配策略:
    1. 精确匹配
    2. 去掉"股份"后缀匹配
    3. 去掉"集团"后缀匹配
    4. 去掉常见后缀组合匹配
    5. 模糊匹配(包含关系)

输出:
  config/ah_stock_mapping.json
  格式: {"601318.SH": {"hk_code": "02318", "name": "中国平安"}, ...}

使用方法:
  python generate_ah_mapping.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import akshare as ak
import tushare as ts
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_PATH = PROJECT_ROOT / "config" / "ah_stock_mapping.json"


def init_tushare() -> ts.pro_api:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("TUSHARE_API_TOKEN", "").strip()
    if not token:
        print("[ERROR] 未在 .env 中找到 TUSHARE_API_TOKEN")
        sys.exit(1)
    ts.set_token(token)
    return ts.pro_api()


def fetch_ah_list() -> list:
    """用 AKShare 获取 A+H 股清单(港股代码 + 名称)"""
    print("[1] 获取 A+H 股清单 (AKShare stock_zh_ah_name)...")
    df = ak.stock_zh_ah_name()
    print(f"    获取到 {len(df)} 条 A+H 股记录")
    records = []
    for _, row in df.iterrows():
        records.append({
            "hk_code": str(row["代码"]).strip(),
            "hk_name": str(row["名称"]).strip(),
        })
    return records


def fetch_a_stock_names(api: ts.pro_api) -> dict:
    """用 Tushare 获取所有 A 股代码到名称的映射"""
    print("[2] 获取 A 股基础信息 (Tushare stock_basic)...")
    df = api.stock_basic(exchange="", list_status="L", fields="ts_code,name")
    print(f"    获取到 {len(df)} 只 A 股")
    return dict(zip(df["ts_code"], df["name"]))


def clean_name(name: str) -> str:
    """清理公司名称用于匹配,去掉常见后缀"""
    name = name.strip()
    # 去掉港股常见的后缀
    suffixes = ["股份", "集团", "控股", "国际", "置业", "实业"]
    for suffix in suffixes:
        if name.endswith(suffix) and len(name) > len(suffix) + 2:
            name = name[: -len(suffix)]
            break
    return name


def match_ah_stocks(ah_list: list, a_stock_map: dict) -> tuple:
    """
    将 A+H 股清单与 A 股代码匹配

    返回:
        (matched, unmatched)
        matched: [{"a_code": "601318.SH", "hk_code": "02318", "name": "中国平安"}, ...]
        unmatched: [{"hk_code": "02318", "hk_name": "中国平安"}, ...]
    """
    print("[3] 匹配 A 股代码与 H 股代码...")

    # 人工补充映射:港股名称与 A 股名称差异大,无法自动匹配的公司
    # key: 港股代码, value: A 股代码
    MANUAL_MAPPING = {
        "01375": "601375.SH",  # 中州证券 -> 中原证券
        "00323": "600808.SH",  # 马鞍山钢铁股份 -> 马钢股份
        "00338": "600688.SH",  # 上海石油化工股份 -> 上海石化
        "03996": "601868.SH",  # 中国能源建设 -> 中国能建
        "01812": "000488.SZ",  # 晨鸣纸业 -> 晨鸣纸业(名称相同但可能数据问题)
        "02768": "002768.SZ",  # 国恩科技 -> 国恩股份
        "01385": "688385.SH",  # 上海复旦 -> 复旦微电
        "01513": "000513.SZ",  # 丽珠医药 -> 丽珠集团
        "00883": "600938.SH",  # 中国海洋石油 -> 中国海油
        "06826": "688366.SH",  # 昊海生物科技 -> 昊海生科
        "00038": "601038.SH",  # 第一拖拉机股份 -> 一拖股份
        "02883": "601808.SH",  # 中海油田服务 -> 中海油服
        "00548": "600548.SH",  # 深圳高速公路股份 -> 深高速
        "03618": "601077.SH",  # 重庆农村商业银行 -> 渝农商行
        "01339": "601319.SH",  # 中国人民保险集团 -> 中国人保
        "01800": "601800.SH",  # 中国交通建设 -> 中国交建
        "00956": "600956.SH",  # 新天绿色能源 -> 新天绿能
        "00670": "600115.SH",  # 中国东方航空股份 -> 中国东航
        "00187": "600860.SH",  # 京城机电股份 -> 京城股份
        "01816": "003816.SZ",  # 中广核电力 -> 中国广核
        "02202": "000002.SZ",  # 万科企业 -> 万科A
    }

    # 构建 A 股名称到代码的反向映射(精确名称)
    a_name_to_code = {}
    # 构建清理后名称到代码的映射
    a_clean_to_code = {}
    for a_code, a_name in a_stock_map.items():
        a_name_to_code[a_name] = a_code
        clean = clean_name(a_name)
        if clean not in a_clean_to_code:
            a_clean_to_code[clean] = a_code

    matched = []
    unmatched = []

    for ah in ah_list:
        hk_code = ah["hk_code"]
        hk_name = ah["hk_name"]
        a_code = None
        match_method = ""

        # 策略1: 精确匹配
        if hk_name in a_name_to_code:
            a_code = a_name_to_code[hk_name]
            match_method = "精确"

        # 策略2: 港股名称去掉后缀后精确匹配
        if not a_code:
            hk_clean = clean_name(hk_name)
            if hk_clean in a_name_to_code:
                a_code = a_name_to_code[hk_clean]
                match_method = "港股去后缀"

        # 策略3: 清理后名称匹配
        if not a_code:
            hk_clean = clean_name(hk_name)
            if hk_clean in a_clean_to_code:
                a_code = a_clean_to_code[hk_clean]
                match_method = "双向去后缀"

        # 策略4: 包含关系匹配 (一个名称包含另一个)
        if not a_code:
            for a_name, code in a_name_to_code.items():
                if len(hk_name) >= 4 and hk_name in a_name:
                    a_code = code
                    match_method = "港股名⊂A股名"
                    break
                if len(a_name) >= 4 and a_name in hk_name:
                    a_code = code
                    match_method = "A股名⊂港股名"
                    break

        # 策略5: 去掉后缀后包含匹配
        if not a_code:
            hk_clean = clean_name(hk_name)
            for a_name, code in a_name_to_code.items():
                a_clean = clean_name(a_name)
                if len(hk_clean) >= 3 and hk_clean in a_clean:
                    a_code = code
                    match_method = "去后缀包含"
                    break
                if len(a_clean) >= 3 and a_clean in hk_clean:
                    a_code = code
                    match_method = "去后缀包含"
                    break

        # 策略6: 人工补充映射(名称差异大,无法自动匹配)
        if not a_code and hk_code in MANUAL_MAPPING:
            manual_a_code = MANUAL_MAPPING[hk_code]
            if manual_a_code in a_stock_map:
                a_code = manual_a_code
                match_method = "人工映射"

        if a_code:
            matched.append({
                "a_code": a_code,
                "hk_code": hk_code,
                "name": a_stock_map[a_code],
                "hk_name": hk_name,
                "match_method": match_method,
            })
        else:
            unmatched.append(ah)

    print(f"    匹配成功: {len(matched)} / {len(ah_list)}")
    print(f"    未匹配  : {len(unmatched)}")

    if matched:
        methods = {}
        for m in matched:
            methods[m["match_method"]] = methods.get(m["match_method"], 0) + 1
        print(f"    匹配方法统计: {methods}")

    return matched, unmatched


def save_mapping(matched: list) -> None:
    """保存映射清单为 JSON"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 构建最终 JSON: a_code 为 key
    mapping = {}
    for item in matched:
        mapping[item["a_code"]] = {
            "hk_code": item["hk_code"],
            "name": item["name"],
        }

    # 按 A 股代码排序
    mapping = dict(sorted(mapping.items()))

    output = {
        "description": "A+H 股映射清单,用于市值修正(A股总股本 = total_share - H股股本)",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "ah_list": "AKShare stock_zh_ah_name",
            "a_stock": "Tushare stock_basic",
        },
        "count": len(mapping),
        "usage": "当 daily_basic 的 total_share 包含 H 股时,用此清单查找 H 股代码,再调 stock_hk_financial_indicator_em 获取 H 股股本",
        "mapping": mapping,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[4] 清单已保存: {OUTPUT_PATH}")
    print(f"    共 {len(mapping)} 条 A+H 股映射")


def main():
    print("=" * 60)
    print("生成 A+H 股映射清单")
    print("=" * 60)

    api = init_tushare()
    ah_list = fetch_ah_list()
    a_stock_map = fetch_a_stock_names(api)

    matched, unmatched = match_ah_stocks(ah_list, a_stock_map)

    # 打印未匹配的记录供人工检查
    if unmatched:
        print(f"\n[!] 未匹配的 {len(unmatched)} 条记录(需人工核对):")
        for item in unmatched:
            print(f"    {item['hk_code']:6s}  {item['hk_name']}")

    save_mapping(matched)

    # 打印部分样本
    print("\n样本数据:")
    for item in matched[:10]:
        print(f"  {item['a_code']:12s} -> {item['hk_code']:6s}  {item['name']:8s}  ({item['match_method']})")


if __name__ == "__main__":
    main()
