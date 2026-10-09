#!/usr/bin/env python3
"""
恒生综合指数成分股PDF → CSV 转换脚本

将恒生官网下载的三个PDF（大型/中型/小型股指数成分股）转换为
统一的 hsci_classification.csv，供监控系统运行时查询。

用法：
    python -m mode_a.scripts.convert_hsci_pdf

输入：项目根目录下的三个PDF（或通过参数指定）
输出：hsci_classification.csv（code, name, index_type）位于项目根目录
"""

import os
import sys
import csv
import pdfplumber

# 项目根目录（mode_a/scripts/ 向上两级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PDF文件名映射 → IndexType枚举值
PDF_MAP = {
    "恒生综合大型股指数成分股.pdf": "恒生综合大型股指数",
    "恒生综合中型股指数成分股.pdf": "恒生综合中型股指数",
    "恒生综合小型股指数成分股.pdf": "恒生综合小型股指数",
}

OUTPUT_CSV = "hsci_classification.csv"


def extract_codes_from_pdf(pdf_path: str, index_type: str):
    """从单个PDF提取成分股代码和名称"""
    entries = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                # 表头行：日期 | 指数 | 股份代号 | 股份名称 | ...
                for row in table[1:]:
                    if not row or len(row) < 4:
                        continue
                    code_raw = str(row[2]).strip() if row[2] else ""
                    name = str(row[3]).strip() if row[3] else ""
                    if not code_raw or not code_raw.isdigit():
                        continue
                    code = code_raw.zfill(5)
                    entries.append((code, name, index_type))
    return entries


def main():
    script_dir = PROJECT_ROOT

    all_entries = []
    for pdf_name, index_type in PDF_MAP.items():
        # 先在项目根目录找，再在当前工作目录找
        pdf_path = os.path.join(script_dir, pdf_name)
        if not os.path.exists(pdf_path):
            pdf_path = os.path.join(os.getcwd(), pdf_name)
        if not os.path.exists(pdf_path):
            print(f"  [跳过] 未找到PDF: {pdf_name}")
            continue

        entries = extract_codes_from_pdf(pdf_path, index_type)
        print(f"  {index_type}: {len(entries)} 只  ← {pdf_name}")
        all_entries.extend(entries)

    # 去重（同一代码可能在多个指数中出现，保留第一个）
    seen = set()
    unique = []
    for code, name, idx_type in all_entries:
        if code not in seen:
            seen.add(code)
            unique.append((code, name, idx_type))

    # 按代码排序
    unique.sort(key=lambda x: x[0])

    # 写CSV（输出到项目根目录）
    output_path = os.path.join(script_dir, OUTPUT_CSV)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "name", "index_type"])
        writer.writerows(unique)

    print(f"\n  合计: {len(unique)} 只成分股")
    print(f"  输出: {output_path}")

    # 统计各类型数量
    from collections import Counter
    counts = Counter(x[2] for x in unique)
    for idx_type, count in sorted(counts.items()):
        print(f"    {idx_type}: {count} 只")


if __name__ == "__main__":
    print("恒生综合指数成分股PDF → CSV 转换")
    print("=" * 50)
    main()
    print("\n转换完成。")
