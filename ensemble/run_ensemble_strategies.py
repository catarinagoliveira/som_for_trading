
import pandas as pd
import matplotlib.pyplot as plt
from evaluation.strategy_eval import simulate_strategy, evaluate_strategy, compute_fear_greed_index
from voting_strategies import majority_vote, weighted_vote, unanimous_vote, aggressive_vote, probabilistic_vote

def run_majority_ensemble(df_signals):
    df = df_signals.copy()
    df['ensemble_signal'] = df.apply(majority_vote, axis=1)
    result = simulate_strategy(df, signal_col='ensemble_signal', price_col='Close')
    metrics = evaluate_strategy(result)
    plot_result(result, 'Ensemble Majority Strategy')
    return metrics, df


def run_weighted_ensemble(df_signals, weights, to_plot=True):
    df = df_signals.copy()
    df['ensemble_weighted'] = df.apply(lambda row: weighted_vote(row, weights), axis=1)
    result = simulate_strategy(df, signal_col='ensemble_weighted', price_col='Close')
    metrics = evaluate_strategy(result)
    if to_plot:
        plot_result(result, 'Ensemble Weighted Strategy')
    return metrics, df


def run_unanimous_ensemble(df_signals):
    df = df_signals.copy()
    df['ensemble_unanimous'] = df.apply(unanimous_vote, axis=1)
    result = simulate_strategy(df, signal_col='ensemble_unanimous', price_col='Close')
    metrics = evaluate_strategy(result)
    plot_result(result, 'Ensemble Unanimous Strategy')
    return metrics, df


def run_aggressive_ensemble(df_signals):
    df = df_signals.copy()
    df['ensemble_aggressive'] = df.apply(aggressive_vote, axis=1)
    result = simulate_strategy(df, signal_col='ensemble_aggressive', price_col='Close')
    metrics = evaluate_strategy(result)
    plot_result(result, 'Ensemble Aggressive Strategy')
    return metrics, df


def run_probabilistic_ensemble(df_signals):
    df = df_signals.copy()
    probs = df.apply(probabilistic_vote, axis=1, result_type='expand')
    df = pd.concat([df, probs], axis=1)
    return df  # This returns probabilities, not simulated metrics


def plot_result(result_df, title):
    plt.figure(figsize=(12, 6))
    plt.plot(result_df.index, result_df['cumulative_return'], label='Strategy')
    plt.plot(result_df.index, result_df['buy_hold_return'], label='Buy & Hold', linestyle='--', color='black')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

