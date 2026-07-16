"""
数据获取模块
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os


def _fetch_wikipedia_tables(url):
    import requests, io
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def get_index_constituents():
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp400_url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    sp500 = _fetch_wikipedia_tables(sp500_url)[0][["Symbol", "GICS Sector"]]
    sp500.columns = ["ticker", "sector"]
    sp500["index_name"] = "SP500"
    sp400_tables = _fetch_wikipedia_tables(sp400_url)
    sp400 = sp400_tables[0]
    ticker_col = "Ticker symbol" if "Ticker symbol" in sp400.columns else "Symbol"
    sector_col = "GICS Sector" if "GICS Sector" in sp400.columns else "GICS  Sector"
    sp400 = sp400[[ticker_col, sector_col]]
    sp400.columns = ["ticker", "sector"]
    sp400["index_name"] = "SP400"
    adr = get_major_adr_list()
    universe = pd.concat([sp500, sp400, adr], ignore_index=True)
    universe["ticker"] = universe["ticker"].str.replace(".", "-", regex=False)
    universe = universe.drop_duplicates(subset="ticker").reset_index(drop=True)
    return universe


def get_major_adr_list():
    adr_data = [
        ("TSM", "Information Technology"), ("ASML", "Information Technology"),
        ("BABA", "Consumer Discretionary"), ("PDD", "Consumer Discretionary"),
        ("NVO", "Health Care"), ("HSBC", "Financials"), ("UL", "Consumer Staples"),
        ("SHEL", "Energy"), ("SAP", "Information Technology"), ("TM", "Consumer Discretionary"),
        ("SONY", "Consumer Discretionary"), ("NVS", "Health Care"), ("AZN", "Health Care"),
        ("TTE", "Energy"), ("RY", "Financials"), ("BHP", "Materials"),
        ("DEO", "Consumer Staples"), ("SNY", "Health Care"), ("MUFG", "Financials"),
        ("BUD", "Consumer Staples"),
    ]
    df = pd.DataFrame(adr_data, columns=["ticker", "sector"])
    df["index_name"] = "ADR"
    return df


def download_price_data(tickers, start_date, end_date, batch_size=50):
    import yfinance as yf
    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"下载第 {i}-{i+len(batch)} 只股票价格...")
        data = yf.download(batch, start=start_date, end=end_date,
                           auto_adjust=True, progress=False, group_by="ticker")
        for t in batch:
            try:
                all_data[t] = data[t]["Close"]
            except Exception:
                continue
        time.sleep(1)
    return pd.DataFrame(all_data)


def download_fundamentals(tickers, output_path=None, max_retries=3, retry_delay=3):
    import yfinance as yf
    existing_df = pd.DataFrame()
    already_done = set()
    if output_path and os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        already_done = set(existing_df["ticker"].tolist())
        print(f"已有 {len(already_done)} 只,跳过")
    remaining = [t for t in tickers if t not in already_done]
    print(f"待下载: {len(remaining)} 只")
    records = []
    for idx, t in enumerate(remaining):
        for attempt in range(1, max_retries + 1):
            try:
                info = yf.Ticker(t).info
                records.append({
                    "ticker": t,
                    "roe": info.get("returnOnEquity", np.nan),
                    "gross_margin": info.get("grossMargins", np.nan),
                    "leverage": info.get("debtToEquity", np.nan),
                    "revenue_growth": info.get("revenueGrowth", np.nan),
                    "estimate_revision": info.get("earningsQuarterlyGrowth", np.nan),
                    "market_cap": info.get("marketCap", np.nan),
                    "avg_volume": info.get("averageVolume", np.nan),
                })
                break
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    print(f"{t} 失败: {e}")
        if output_path and (idx + 1) % 20 == 0:
            pd.concat([existing_df, pd.DataFrame(records)], ignore_index=True).to_csv(output_path, index=False)
            print(f"  已处理 {idx+1}/{len(remaining)}")
    final_df = pd.concat([existing_df, pd.DataFrame(records)], ignore_index=True)
    if output_path:
        final_df.to_csv(output_path, index=False)
    return final_df
