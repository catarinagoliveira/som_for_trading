"""
voting_strategies.py

This module contains multiple ensemble voting strategies for combining signals from different SOM models
(technical, sentiment, hybrid). Each function takes a row of signals and outputs a final ensemble signal.
"""

from collections import Counter

def majority_vote(row):
    """
    Majority voting strategy (hard voting).
    Chooses the signal (buy/hold/sell) that appears most often.
    """
    signals = [row['signal_tech'], row['signal_sent'], row['signal_hybrid']]
    signals = [s for s in signals if s in ['buy', 'hold', 'sell']]
    return Counter(signals).most_common(1)[0][0] if signals else 'hold'

def weighted_vote(row, weights=None):
    """
    Weighted voting strategy (soft voting).
    Allows assigning different importance to each SOM model.
    """
    if weights is None:
        weights = {'signal_tech': 0.5, 'signal_sent': 0.5, 'signal_hybrid': 0}
    
    scores = {'buy': 0, 'hold': 0, 'sell': 0}
    for signal_name, weight in weights.items():
        signal = row.get(signal_name)
        if signal in scores:
            scores[signal] += weight
    return max(scores, key=scores.get)

def unanimous_vote(row):
    """
    Conservative ensemble strategy.
    Only returns a signal if all three agree. Otherwise returns 'hold'.
    """
    s = {row['signal_tech'], row['signal_sent'], row['signal_hybrid']}
    return list(s)[0] if len(s) == 1 else 'hold'

def aggressive_vote(row):
    """
    Aggressive ensemble strategy.
    If any model says 'sell', return 'sell'.
    If any model says 'buy', return 'buy'.
    Else return 'hold'.
    """
    if 'sell' in [row['signal_tech'], row['signal_sent'], row['signal_hybrid']]:
        return 'sell'
    elif 'buy' in [row['signal_tech'], row['signal_sent'], row['signal_hybrid']]:
        return 'buy'
    return 'hold'

def probabilistic_vote(row):
    """
    Returns probabilities for each signal based on vote frequency.
    Useful for future probabilistic or threshold-based ensemble strategies.
    """
    signals = [row['signal_tech'], row['signal_sent'], row['signal_hybrid']]
    counter = Counter(signals)
    return {
        'buy_prob': counter['buy'] / 3,
        'hold_prob': counter['hold'] / 3,
        'sell_prob': counter['sell'] / 3,
    }
