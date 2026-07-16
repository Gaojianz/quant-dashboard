"""
因子计算 + 看板生成脚本
读取 data/ 里的价格和基本面数据
输出完整看板到 docs/index.html
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from factors import compute_momentum, compute_growth, compute_quality, combine_factor_scores
from portfolio import score_weighted_target_weights, DEFAULT_CONFIG

os.makedirs("docs", exist_ok=True)
print("=== 开始构建看板 ===")

prices = pd.read_csv("data/prices_wide.csv", index_col=0, parse_dates=True)
fund   = pd.read_csv("data/fundamentals.csv")
as_of  = prices.index.max()
print(f"数据截至: {as_of.date()}")

fund_idx = fund.set_index("ticker")
fund_idx["latest_price"] = prices.loc[as_of].reindex(fund_idx.index)
fund_idx["adv_dollar"]   = fund_idx["avg_volume"] * fund_idx["latest_price"]
fund_idx["first_valid"]  = prices.apply(lambda c: c.first_valid_index()).reindex(fund_idx.index)
cutoff = as_of - pd.DateOffset(months=12)
mask = (
    (fund_idx["market_cap"]  >= 2e9) &
    (fund_idx["adv_dollar"]  >= 1e7) &
    (fund_idx["first_valid"] <= cutoff) &
    (fund_idx["latest_price"].notna())
)
filtered   = fund_idx[mask].copy()
sector_map = filtered["sector"].to_dict()
eligible   = filtered.index.tolist()
print(f"筛选后股票池: {len(eligible)} 只")

momentum = compute_momentum(prices[eligible], as_of, 12).dropna()
fund_ff  = filtered.reset_index()
growth   = compute_growth(fund_ff)
quality  = compute_quality(fund_ff)
common   = momentum.index.intersection(growth.index).intersection(quality.index)
ranking  = combine_factor_scores(momentum[common], growth[common], quality[common], sector_map)
ranking["sector"] = ranking.index.map(sector_map)
print(f"排名完成: {len(ranking)} 只")

start = as_of - pd.DateOffset(months=12)
sec_cn = {
    "Information Technology":"信息技术","Financials":"金融","Health Care":"医疗保健",
    "Consumer Discretionary":"可选消费","Industrials":"工业","Materials":"材料",
    "Consumer Staples":"必需消费","Energy":"能源","Communication Services":"通讯服务",
    "Real Estate":"房地产","Utilities":"公用事业",
}

def ret12(t):
    try:
        s = prices.loc[:start, t].dropna().iloc[-1]
        e = prices.loc[as_of, t]
        return round(float((e/s-1)), 4) if pd.notna(s) and s > 0 else None
    except: return None

sectors_data = []
for sec in ranking["sector"].dropna().unique():
    sub  = ranking[ranking["sector"] == sec]
    rets = [r for r in [ret12(t) for t in sub.index] if r is not None]
    median_mom = float(pd.Series(rets).median()) if rets else 0
    stocks = []
    for i, (t, row) in enumerate(sub.iterrows(), 1):
        stocks.append({
            "ticker": t, "rank": i,
            "score": round(float(row["composite_score"]), 2),
            "mom":   round(float(row["momentum_z"]), 2),
            "gro":   round(float(row["growth_z"]), 2),
            "qua":   round(float(row["quality_z"]), 2),
            "ret12m": ret12(t),
            "mine": False,
        })
    sectors_data.append({
        "sector": sec_cn.get(sec, sec),
        "count": len(sub),
        "medianMom": round(median_mom, 3),
        "stocks": stocks,
    })
sectors_data.sort(key=lambda x: -x["medianMom"])

display_start = as_of - pd.DateOffset(years=5)
top30_per_sec = {s["sector"]: [x["ticker"] for x in s["stocks"][:30]] for s in sectors_data}
all_top30 = set(t for v in top30_per_sec.values() for t in v)

def calc_macd(series):
    full = series.dropna()
    if len(full) < 50: return []
    w = full.resample("W-FRI").last().dropna()
    e12 = w.ewm(span=12, adjust=False).mean()
    e26 = w.ewm(span=26, adjust=False).mean()
    dif = e12 - e26
    dea = dif.ewm(span=9, adjust=False).mean()
    df  = pd.DataFrame({"c":w,"dif":dif,"dea":dea,"h":(dif-dea)*2}).loc[display_start:]
    return [[r.strftime("%y-%m-%d"), round(float(v["c"]),2),
             round(float(v["dif"]),2), round(float(v["dea"]),2), round(float(v["h"]),2)]
            for r, v in df.iterrows()]

chart_data = {}
for t in all_top30:
    if t in prices.columns:
        d = calc_macd(prices[t])
        if d: chart_data[t] = d
print(f"MACD数据: {len(chart_data)} 只")

data_js = (
    "window.T30="     + json.dumps(top30_per_sec, ensure_ascii=False, separators=(',',':')) + ";\n"
    "window.CD="      + json.dumps(chart_data,    ensure_ascii=False, separators=(',',':')) + ";\n"
    "window.SECTORS=" + json.dumps({"sectors": sectors_data, "asOf": as_of.strftime("%Y-%m-%d")},
                                    ensure_ascii=False, separators=(',',':')) + ";\n"
)
with open("docs/data.js", "w") as f:
    f.write(data_js)
print(f"data.js: {len(data_js)//1024} KB")

template_path = "templates/sector_tab_board.html"
if os.path.exists(template_path):
    with open(template_path) as f:
        html = f.read()
    html = html.replace("</head>", '<script src="data.js"></script>\n</head>')
    with open("docs/index.html", "w") as f:
        f.write(html)
    print("index.html 生成完成")
else:
    print(f"警告: 找不到模板 {template_path}")

print("=== 构建完成 ===")
print("访问地址: https://gaojianz.github.io/quant-dashboard/")
