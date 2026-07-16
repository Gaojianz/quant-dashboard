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
    """
    计算过去 lookback_months 个月的完整收益率(不剔除最近1个月)。
    price_panel: 宽表 DataFrame, index=日期, columns=ticker
    返回: Series[ticker -> momentum_return]
    """
    end_date = pd.Timestamp(as_of_date)
    start_date = end_date - pd.DateOffset(months=lookback_months)

    prices = price_panel.loc[:end_date]
    if prices.empty:
        return pd.Series(dtype=float)

    # 找到起始日和结束日最近的有效交易日
    start_prices = prices.loc[:start_date].iloc[-1] if not prices.loc[:start_date].empty else np.nan
    end_prices = prices.iloc[-1]

    momentum = (end_prices / start_prices) - 1.0
    return momentum


def compute_growth(fundamentals_df, winsorize_limits=(-1.0, 2.0)):
    """
    成长因子 = 营收增长率 与 分析师预期修正 的等权合成(合成前先各自标准化)。

    winsorize_limits: 对 estimate_revision 做极值截断,默认压缩到 [-100%, +200%] 区间。
    这个字段实际取的是 earningsQuarterlyGrowth(季度盈利同比增长率),容易受"低基数效应"
    影响 —— 如果去年同期盈利基数极低(接近0或负数),哪怕今年只是恢复正常水平,
    同比增长率也可能被算成几百甚至几千个百分点,这不代表真实的成长性,而是数据本身的
    统计缺陷。不做截断的话,这类异常值会在z-score标准化后主导整个成长因子得分,
    把排名结果搞偏(实测发现约40%的前30名单因此虚高排名)。
    """
    df = fundamentals_df.copy()

    # 极值截断: 把超出[low, high]的值压缩到边界值,而不是直接丢弃这些股票
    low, high = winsorize_limits
    df["estimate_revision_capped"] = df["estimate_revision"].clip(lower=low, upper=high)

    rev_z = zscore(df["revenue_growth"])
    est_z = zscore(df["estimate_revision_capped"])
    growth_score = 0.5 * rev_z + 0.5 * est_z
    return pd.Series(growth_score.values, index=df["ticker"].values)


def compute_quality(fundamentals_df, roe_cap=2.0, leverage_cap=300.0):
    """
    质量因子 = ROE(正向) + 毛利率稳定性(正向,用毛利率本身作代理,
    生产环境建议用历史毛利率的标准差的倒数衡量"稳定性") + 负债率(反向)。
    三者等权合成。

    roe_cap / leverage_cap: 对ROE和负债率做极值截断。
    这两个字段都发现过异常值 —— ROE异常通常是股东权益因大量股票回购被侵蚀到接近0,
    导致比率被人为放大(实测发现最高达8457%);负债率(debtToEquity)最大值曾出现
    12260这种明显不合理的数字。不截断的话,这些异常值会在z-score标准化后
    主导质量因子得分,把排名结果搞偏。
    """
    df = fundamentals_df.copy()
    roe_capped = df["roe"].clip(lower=-roe_cap, upper=roe_cap)
    leverage_capped = df["leverage"].clip(upper=leverage_cap)

    roe_z = zscore(roe_capped)
    margin_z = zscore(df["gross_margin"])
    leverage_z = zscore(leverage_capped)  # 负债率越高越差,合成时取负

    quality_score = (roe_z + margin_z - leverage_z) / 3.0
    return pd.Series(quality_score.values, index=df["ticker"].values)


def zscore(series):
    """标准 z-score 标准化,自动处理缺失值(缺失值填充为该列均值,即中性分)。"""
    s = series.astype(float)
    filled = s.fillna(s.mean())
    std = filled.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (filled - filled.mean()) / std


def sector_neutral_zscore(raw_scores, sector_map):
    """
    行业中性化: 在每个行业内部分别做 z-score,而不是全市场统一标准化。
    raw_scores: Series[ticker -> raw_factor_value]
    sector_map: dict 或 Series[ticker -> sector]
    """
    df = pd.DataFrame({"value": raw_scores})
    df["sector"] = df.index.map(sector_map)

    def _z(group):
        std = group.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / std

    df["z"] = df.groupby("sector")["value"].transform(_z)
    return df["z"]


def combine_factor_scores(momentum, growth, quality, sector_map,
                           weights=None):
    """
    将三个原始因子分别做行业中性化 z-score,再按权重合成综合得分。
    返回: DataFrame[ticker, momentum_z, growth_z, quality_z, composite_score]
    """
    weights = weights or FACTOR_WEIGHTS

    mom_z = sector_neutral_zscore(momentum, sector_map)
    growth_z = sector_neutral_zscore(growth, sector_map)
    quality_z = sector_neutral_zscore(quality, sector_map)

    combined = pd.DataFrame({
        "momentum_z": mom_z,
        "growth_z": growth_z,
        "quality_z": quality_z,
    })
    combined = combined.fillna(0.0)

    combined["composite_score"] = (
        weights["momentum"] * combined["momentum_z"] +
        weights["growth"] * combined["growth_z"] +
        weights["quality"] * combined["quality_z"]
    )

    return combined.sort_values("composite_score", ascending=False)
