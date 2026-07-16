"""
因子计算模块
------------
实现策略确定的三因子模型:
  - 动量 40%: 过去12个月完整收益率(不剔除最近1个月)
  - 成长 30%: 营收增长率 + 分析师预期修正
  - 质量 30%: ROE + 毛利率稳定性 + 负债率(反向,负债越高得分越低)

每个因子先在行业内做 z-score 标准化(行业中性化),
避免某个因子实际上只是在赌某个行业的暴露。
"""

import pandas as pd
import numpy as np


FACTOR_WEIGHTS = {
    "momentum": 0.40,
    "growth": 0.30,
    "quality": 0.30,
}


def compute_momentum(price_panel, as_of_date, lookback_months=12):
    end_date = pd.Timestamp(as_of_date)
    start_date = end_date - pd.DateOffset(months=lookback_months)
    prices = price_panel.loc[:end_date]
    if prices.empty:
        return pd.Series(dtype=float)
    start_prices = prices.loc[:start_date].iloc[-1] if not prices.loc[:start_date].empty else np.nan
    end_prices = prices.iloc[-1]
    return (end_prices / start_prices) - 1.0


def compute_growth(fundamentals_df, winsorize_limits=(-1.0, 2.0)):
    df = fundamentals_df.copy()
    low, high = winsorize_limits
    df["estimate_revision_capped"] = df["estimate_revision"].clip(lower=low, upper=high)
    rev_z = zscore(df["revenue_growth"])
    est_z = zscore(df["estimate_revision_capped"])
    growth_score = 0.5 * rev_z + 0.5 * est_z
    return pd.Series(growth_score.values, index=df["ticker"].values)


def compute_quality(fundamentals_df, roe_cap=2.0, leverage_cap=300.0):
    df = fundamentals_df.copy()
    roe_capped = df["roe"].clip(lower=-roe_cap, upper=roe_cap)
    leverage_capped = df["leverage"].clip(upper=leverage_cap)
    roe_z = zscore(roe_capped)
    margin_z = zscore(df["gross_margin"])
    leverage_z = zscore(leverage_capped)
    quality_score = (roe_z + margin_z - leverage_z) / 3.0
    return pd.Series(quality_score.values, index=df["ticker"].values)


def zscore(series):
    s = series.astype(float)
    filled = s.fillna(s.mean())
    std = filled.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (filled - filled.mean()) / std


def sector_neutral_zscore(raw_scores, sector_map):
    df = pd.DataFrame({"value": raw_scores})
    df["sector"] = df.index.map(sector_map)
    def _z(group):
        std = group.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / std
    df["z"] = df.groupby("sector")["value"].transform(_z)
    return df["z"]


def combine_factor_scores(momentum, growth, quality, sector_map, weights=None):
    weights = weights or FACTOR_WEIGHTS
    mom_z = sector_neutral_zscore(momentum, sector_map)
    growth_z = sector_neutral_zscore(growth, sector_map)
    quality_z = sector_neutral_zscore(quality, sector_map)
    combined = pd.DataFrame({
        "momentum_z": mom_z,
        "growth_z": growth_z,
        "quality_z": quality_z,
    }).fillna(0.0)
    combined["composite_score"] = (
        weights["momentum"] * combined["momentum_z"] +
        weights["growth"]   * combined["growth_z"] +
        weights["quality"]  * combined["quality_z"]
    )
    return combined.sort_values("composite_score", ascending=False)
