"""
数据获取模块
------------
职责:
1. 获取标普500 + 标普400成分股列表
2. 按照流动性/市值/上市时长筛选出可交易股票池
3. 下载价格数据(用于动量计算)和基本面数据(用于成长/质量因子)

注意: 本模块依赖联网环境运行 (yfinance + 维基百科成分股列表)。
在沙盒环境中无法验证网络请求,请在你本地/服务器环境安装依赖后运行:
    pip install yfinance pandas numpy requests lxml
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os


def _fetch_wikipedia_tables(url):
    """
    带浏览器 User-Agent 请求维基百科页面再交给 pandas 解析,
    避免直接用 pd.read_html(url) 时被维基百科的反爬虫拦截返回 403。
    """
    import requests
    import io

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def get_index_constituents():
    """
    从维基百科抓取标普500和标普400的最新成分股列表,
    并额外并入一份流动性好、市值大的知名美股ADR清单。
    返回: DataFrame[ticker, sector, index_name]
    """
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp400_url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

    sp500 = _fetch_wikipedia_tables(sp500_url)[0][["Symbol", "GICS Sector"]]
    sp500.columns = ["ticker", "sector"]
    sp500["index_name"] = "SP500"

    sp400_tables = _fetch_wikipedia_tables(sp400_url)
    sp400 = sp400_tables[0]
    # 标普400表格列名可能是 "Ticker symbol" 或 "Symbol",做一下兼容
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
    """
    知名美股ADR清单(手工维护,而非从指数自动抓取,因为ADR不属于标普500/400)。
    只挑选流动性好、市值大、大家熟悉的龙头公司,避免引入流动性/数据质量差的长尾ADR。

    注意: ADR公司的财报口径(IFRS vs GAAP)、财报发布节奏(部分为半年报)、
    汇率暴露等和美股本土公司不完全可比,使用这份清单时需留意这层数据可比性瑕疵。
    """
    adr_data = [
        ("TSM", "Information Technology"),   # 台积电
        ("ASML", "Information Technology"),  # 阿斯麦
        ("BABA", "Consumer Discretionary"),  # 阿里巴巴
        ("PDD", "Consumer Discretionary"),   # 拼多多
        ("NVO", "Health Care"),              # 诺和诺德
        ("HSBC", "Financials"),              # 汇丰
        ("UL", "Consumer Staples"),          # 联合利华
        ("SHEL", "Energy"),                  # 壳牌
        ("SAP", "Information Technology"),   # SAP
        ("TM", "Consumer Discretionary"),    # 丰田
        ("SONY", "Consumer Discretionary"),  # 索尼
        ("NVS", "Health Care"),              # 诺华
        ("AZN", "Health Care"),              # 阿斯利康
        ("TTE", "Energy"),                   # 道达尔能源
        ("RY", "Financials"),                # 加拿大皇家银行
        ("BHP", "Materials"),                # 必和必拓
        ("DEO", "Consumer Staples"),         # 帝亚吉欧
        ("SNY", "Health Care"),              # 赛诺菲
        ("MUFG", "Financials"),              # 三菱日联
        ("BUD", "Consumer Staples"),         # 百威英博
    ]
    df = pd.DataFrame(adr_data, columns=["ticker", "sector"])
    df["index_name"] = "ADR"
    return df


def download_price_data(tickers, start_date, end_date, batch_size=50):
    """
    分批下载历史价格数据(避免单次请求过多股票导致失败)
    返回: 宽表 DataFrame, index=日期, columns=ticker, values=调整后收盘价
    """
    import yfinance as yf

    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"下载第 {i}-{i+len(batch)} 只股票的价格数据...")
        data = yf.download(
            batch, start=start_date, end=end_date,
            auto_adjust=True, progress=False, group_by="ticker"
        )
        for t in batch:
            try:
                all_data[t] = data[t]["Close"]
            except Exception:
                continue
        time.sleep(1)  # 避免请求过快被限流

    prices = pd.DataFrame(all_data)
    return prices


def download_fundamentals(tickers, output_path=None, max_retries=3, retry_delay=3):
    """
    下载基本面数据: ROE、毛利率、负债率、营收增长率、分析师预期修正等。
    yfinance 的 .info / .financials 接口字段不完全稳定,实际生产环境建议
    替换为更稳定的数据源(如 FactSet、Refinitiv、或付费的 financialmodelingprep API)。

    新增两个健壮性机制:
    1. 失败重试(max_retries次,每次间隔retry_delay秒) —— 应对网络抖动、DNS解析失败这类临时性错误
    2. 断点续传(如果传入output_path且该文件已存在)—— 跳过已经成功下载过的股票,
       只补齐缺失的部分,避免网络中断后需要从头重新下载几百只股票

    返回: DataFrame[ticker, roe, gross_margin, leverage, revenue_growth, estimate_revision, market_cap, avg_volume]
    """
    import yfinance as yf
    import time

    # ---- 断点续传: 如果已有部分结果,先读进来,跳过已成功的ticker ----
    existing_df = pd.DataFrame()
    already_done = set()
    if output_path and os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        already_done = set(existing_df["ticker"].tolist())
        print(f"检测到已有结果文件,已成功下载 {len(already_done)} 只,将跳过这些股票只补齐剩余部分")

    remaining_tickers = [t for t in tickers if t not in already_done]
    print(f"待下载: {len(remaining_tickers)} 只")

    records = []
    for idx, t in enumerate(remaining_tickers):
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                tk = yf.Ticker(t)
                info = tk.info

                roe = info.get("returnOnEquity", np.nan)
                gross_margin = info.get("grossMargins", np.nan)
                debt_to_equity = info.get("debtToEquity", np.nan)
                revenue_growth = info.get("revenueGrowth", np.nan)
                estimate_revision = info.get("earningsQuarterlyGrowth", np.nan)

                records.append({
                    "ticker": t,
                    "roe": roe,
                    "gross_margin": gross_margin,
                    "leverage": debt_to_equity,
                    "revenue_growth": revenue_growth,
                    "estimate_revision": estimate_revision,
                    "market_cap": info.get("marketCap", np.nan),
                    "avg_volume": info.get("averageVolume", np.nan),
                })
                success = True
                break
            except Exception as e:
                if attempt < max_retries:
                    print(f"{t} 第{attempt}次尝试失败({e}),{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    print(f"{t} 已重试{max_retries}次仍失败,跳过: {e}")

        # ---- 每处理20只,增量保存一次,避免再次中断时前功尽弃 ----
        if output_path and (idx + 1) % 20 == 0:
            partial_df = pd.concat([existing_df, pd.DataFrame(records)], ignore_index=True)
            partial_df.to_csv(output_path, index=False)
            print(f"  已处理 {idx+1}/{len(remaining_tickers)},增量保存至 {output_path}")

    final_df = pd.concat([existing_df, pd.DataFrame(records)], ignore_index=True)
    if output_path:
        final_df.to_csv(output_path, index=False)
    return final_df


def apply_liquidity_filters(universe_df, fundamentals_df,
                             min_market_cap=2e9, min_adv_dollar=1e7,
                             price_data=None, min_list_months=12):
    """
    应用流动性/市值/上市时长筛选规则:
    - 市值 > 20亿美元
    - 近3个月日均成交额 > 1000万美元
    - 上市满12个月以上 (需要 price_data 判断首个有效交易日)
    """
    df = fundamentals_df.copy()
    df["adv_dollar"] = df["avg_volume"] * df.get("price", np.nan)

    mask = (df["market_cap"] >= min_market_cap)
    if "adv_dollar" in df.columns:
        mask &= (df["adv_dollar"] >= min_adv_dollar) | df["adv_dollar"].isna()

    filtered = df[mask].copy()

    if price_data is not None:
        min_date_cutoff = price_data.index.max() - pd.DateOffset(months=min_list_months)
        valid_tickers = []
        for t in filtered["ticker"]:
            if t in price_data.columns:
                first_valid = price_data[t].first_valid_index()
                if first_valid is not None and first_valid <= min_date_cutoff:
                    valid_tickers.append(t)
        filtered = filtered[filtered["ticker"].isin(valid_tickers)]

    return filtered.merge(universe_df[["ticker", "sector"]], on="ticker", how="left")


if __name__ == "__main__":
    print("本模块需要联网环境运行,请在本地/服务器安装依赖后执行:")
    print("  pip install yfinance pandas numpy requests lxml")
