"""
数据下载脚本 — 在 GitHub Actions 环境里运行
下载标普500 + 标普400 + ADR 的价格和基本面数据
输出到 data/ 目录
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import date
from data_loader import get_index_constituents, download_price_data, download_fundamentals

os.makedirs("data", exist_ok=True)

START_DATE = "2019-01-01"
END_DATE = date.today().isoformat()

print(f"=== 数据下载开始 {END_DATE} ===")

print("步骤1: 获取成分股列表...")
universe = get_index_constituents()
tickers = universe["ticker"].tolist()
print(f"  共 {len(tickers)} 只")

print("步骤2: 下载价格数据...")
prices = download_price_data(tickers, START_DATE, END_DATE)
prices.to_csv("data/prices_wide.csv")
print(f"  价格数据: {prices.shape}")

print("步骤3: 下载基本面数据 (支持断点续传)...")
fund = download_fundamentals(tickers, output_path="data/fundamentals_raw.csv")
fund = fund.merge(universe[["ticker","sector"]], on="ticker", how="left")
fund.to_csv("data/fundamentals.csv", index=False)
print(f"  基本面数据: {len(fund)} 只")

print("=== 下载完成 ===")
