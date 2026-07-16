"""
组合构建模块(账户平衡师)
------------------------
实现再平衡规则:
  - 目标持仓数量: 25-35只(默认30只)
  - 缓冲带: 换出门槛设在第50名(而不是第30/35名),降低不必要的换手
  - 单票权重上限: 5%-6%
  - 权重漂移容忍度: ±1.5%-2%,偏离小于此不调整
  - 不设行业上限(接受集中)
"""

import pandas as pd
import numpy as np


DEFAULT_CONFIG = {
    "target_holdings": 30,
    "buy_in_rank": 30,       # 新股票要进入前30名才买入
    "sell_out_rank": 50,     # 现有持仓跌出前50名才卖出(缓冲带核心)
    "max_weight": 0.06,      # 单票权重上限 6%
    "weight_drift_tolerance": 0.02,  # 权重偏离超过2%才调整
}


def score_weighted_target_weights(ranked_scores, holdings, max_weight):
    """
    按因子得分加权计算目标权重(得分越高权重越高),并施加单票上限。
    ranked_scores: Series[ticker -> composite_score], 已按分数排序
    holdings: 要持有的股票列表
    """
    scores = ranked_scores.loc[holdings]
    # 因子得分可能为负,先平移到正数区间再加权,避免负权重
    shifted = scores - scores.min() + 0.01
    raw_weights = shifted / shifted.sum()

    # 施加单票上限,多次迭代重新分配超额部分,直到收敛
    weights = raw_weights.copy()
    for _ in range(20):
        excess = (weights - max_weight).clip(lower=0)
        if excess.sum() < 1e-6:
            break
        weights = weights.clip(upper=max_weight)
        under_cap = weights < max_weight
        if under_cap.sum() == 0:
            break
        redistribute = excess.sum()
        weights[under_cap] += redistribute * (weights[under_cap] / weights[under_cap].sum())

    weights = weights / weights.sum()  # 归一化确保加总为1
    return weights


def rebalance_portfolio(current_holdings, factor_ranking, config=None):
    """
    核心再平衡逻辑:
    1. 现有持仓: 只要仍在 sell_out_rank 名以内就保留(缓冲带)
    2. 新股票: 必须排进 buy_in_rank 名以内才能新买入
    3. 补齐到 target_holdings 数量
    4. 重新计算目标权重(应用权重上限)
    5. 只对偏离超过 weight_drift_tolerance 的仓位生成交易指令

    factor_ranking: DataFrame,index=ticker,包含 composite_score,已按排名排序(第0行=第1名)
    current_holdings: dict {ticker: current_weight}
    返回: (new_holdings_weights: dict, trades: DataFrame, turnover: float)
    """
    config = config or DEFAULT_CONFIG
    ranked_tickers = factor_ranking.index.tolist()
    rank_map = {t: i + 1 for i, t in enumerate(ranked_tickers)}

    # 第一步: 现有持仓里,排名仍在缓冲带内的予以保留
    retained = [
        t for t in current_holdings
        if rank_map.get(t, 9999) <= config["sell_out_rank"]
    ]

    # 第二步: 需要补充的名额,从 buy_in_rank 以内、尚未持有的股票里按排名补齐
    needed = config["target_holdings"] - len(retained)
    candidates = [
        t for t in ranked_tickers[:config["buy_in_rank"]]
        if t not in retained
    ]
    new_buys = candidates[:max(needed, 0)]

    new_holdings_list = retained + new_buys

    # 如果不足 target_holdings(比如缓冲带内候选不够),放宽到 buy_in_rank 之外补齐
    if len(new_holdings_list) < config["target_holdings"]:
        extra_needed = config["target_holdings"] - len(new_holdings_list)
        more_candidates = [
            t for t in ranked_tickers if t not in new_holdings_list
        ][:extra_needed]
        new_holdings_list += more_candidates

    # 第三步: 计算目标权重
    target_weights = score_weighted_target_weights(
        factor_ranking["composite_score"], new_holdings_list, config["max_weight"]
    )

    # 第四步: 只对偏离超过容忍度的仓位生成交易
    trades = []
    all_tickers = set(current_holdings.keys()) | set(target_weights.index)
    final_weights = dict(current_holdings)  # 默认保持不变

    for t in all_tickers:
        old_w = current_holdings.get(t, 0.0)
        new_w = target_weights.get(t, 0.0)
        drift = abs(new_w - old_w)

        if new_w == 0.0 and old_w > 0.0:
            # 完全换出
            trades.append({"ticker": t, "action": "SELL_ALL", "old_weight": old_w, "new_weight": 0.0})
            final_weights[t] = 0.0
        elif drift > config["weight_drift_tolerance"]:
            action = "BUY" if new_w > old_w else "TRIM"
            trades.append({"ticker": t, "action": action, "old_weight": old_w, "new_weight": new_w})
            final_weights[t] = new_w
        else:
            final_weights[t] = old_w  # 偏离不大,保持不动

    final_weights = {t: w for t, w in final_weights.items() if w > 0}
    trades_df = pd.DataFrame(trades)

    # 换手率 = 本次调整涉及的权重变动绝对值之和 / 2 (买卖各算一次会重复计)
    turnover = trades_df.apply(
        lambda r: abs(r["new_weight"] - r["old_weight"]), axis=1
    ).sum() / 2 if not trades_df.empty else 0.0

    return final_weights, trades_df, turnover


def prioritize_tax_lots(trades_df, holding_periods_days, long_term_days=365):
    """
    税务优先级规则: 再平衡需要卖出时,优先卖出已持有满12个月(长期资本利得)的仓位。
    holding_periods_days: dict {ticker: 持有天数}
    在 trades_df 里标记每笔卖出交易是否享受长期税率,供执行前参考。
    """
    if trades_df.empty:
        return trades_df

    df = trades_df.copy()
    df["holding_days"] = df["ticker"].map(holding_periods_days).fillna(0)
    df["tax_treatment"] = np.where(
        df["holding_days"] >= long_term_days, "长期(优惠税率)", "短期(普通税率)"
    )
    return df
