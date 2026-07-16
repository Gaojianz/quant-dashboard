"""
组合构建模块
"""

import pandas as pd
import numpy as np


DEFAULT_CONFIG = {
    "target_holdings": 30,
    "buy_in_rank": 30,
    "sell_out_rank": 50,
    "max_weight": 0.06,
    "weight_drift_tolerance": 0.02,
}


def score_weighted_target_weights(ranked_scores, holdings, max_weight):
    scores = ranked_scores.loc[holdings]
    shifted = scores - scores.min() + 0.01
    raw_weights = shifted / shifted.sum()
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
    return weights / weights.sum()


def rebalance_portfolio(current_holdings, factor_ranking, config=None):
    config = config or DEFAULT_CONFIG
    ranked_tickers = factor_ranking.index.tolist()
    rank_map = {t: i + 1 for i, t in enumerate(ranked_tickers)}
    retained = [t for t in current_holdings if rank_map.get(t, 9999) <= config["sell_out_rank"]]
    needed = config["target_holdings"] - len(retained)
    candidates = [t for t in ranked_tickers[:config["buy_in_rank"]] if t not in retained]
    new_buys = candidates[:max(needed, 0)]
    new_holdings_list = retained + new_buys
    if len(new_holdings_list) < config["target_holdings"]:
        extra_needed = config["target_holdings"] - len(new_holdings_list)
        more = [t for t in ranked_tickers if t not in new_holdings_list][:extra_needed]
        new_holdings_list += more
    target_weights = score_weighted_target_weights(
        factor_ranking["composite_score"], new_holdings_list, config["max_weight"]
    )
    trades = []
    all_tickers = set(current_holdings.keys()) | set(target_weights.index)
    final_weights = dict(current_holdings)
    for t in all_tickers:
        old_w = current_holdings.get(t, 0.0)
        new_w = target_weights.get(t, 0.0)
        drift = abs(new_w - old_w)
        if new_w == 0.0 and old_w > 0.0:
            trades.append({"ticker": t, "action": "SELL_ALL", "old_weight": old_w, "new_weight": 0.0})
            final_weights[t] = 0.0
        elif drift > config["weight_drift_tolerance"]:
            action = "BUY" if new_w > old_w else "TRIM"
            trades.append({"ticker": t, "action": action, "old_weight": old_w, "new_weight": new_w})
            final_weights[t] = new_w
        else:
            final_weights[t] = old_w
    final_weights = {t: w for t, w in final_weights.items() if w > 0}
    trades_df = pd.DataFrame(trades)
    turnover = trades_df.apply(lambda r: abs(r["new_weight"] - r["old_weight"]), axis=1).sum() / 2 if not trades_df.empty else 0.0
    return final_weights, trades_df, turnover
