import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def assign_signals(cluster_series, node_action_map):
    """
    Map clusters to trading signals.
    """
    return cluster_series.map(node_action_map)


def simulate_strategy(
    df,
    signal_col='signal',
    price_col='Close',
    cost=0.001,
    trade_size=0.2,  # 20% of capital per trade
    max_position=0.5,  # Max position size (50% of capital)
    min_position=0.0,
    verbose=False
):
    """
    Simulate a trading strategy that allows partial buys/sells (not all-in).
    
    Parameters:
        df (pd.DataFrame): DataFrame with signals and prices
        signal_col (str): Name of column with signals ('buy', 'sell', 'hold')
        price_col (str): Column with asset prices
        cost (float): Transaction cost per position change
        trade_size (float): Fraction of capital to trade per signal
        max_position (float): Max position size allowed (e.g., 1.0 = 100%)
        min_position (float): Min position size (e.g., 0.0 = fully out)
        verbose (bool): If True, print trade actions
    
    Returns:
        df (pd.DataFrame): DataFrame with returns and strategy performance
    """
    df = df.copy()
    df['position'] = 0.0
    position = 0.0

    for i in range(len(df)):
        signal = df.iloc[i][signal_col]

        if signal == 'buy' and position < max_position:
            position = min(position + trade_size, max_position)
            if verbose:
                print(f"{df.index[i]}: BUY -> position: {position:.2f}")

        elif signal == 'sell' and position > min_position:
            position = max(position - trade_size, min_position)
            if verbose:
                print(f"{df.index[i]}: SELL -> position: {position:.2f}")

        df.at[df.index[i], 'position'] = position

    # Calculate returns
    df['returns'] = df[price_col].pct_change().fillna(0)
    df['position_shifted'] = df['position'].shift(1).fillna(0)

    # Transaction cost on change in position
    trades = df['position_shifted'].diff().abs()
    df['strategy_returns'] = df['position_shifted'] * df['returns'] - trades * cost

    df['cumulative_return'] = (1 + df['strategy_returns']).cumprod()
    df['buy_hold_return'] = (1 + df['returns']).cumprod()

    return df


def evaluate_strategy(df):
    """
    Compute advanced performance metrics for a trading strategy.
    """
    returns = df['strategy_returns']
    cumulative = df['cumulative_return'].iloc[-1]
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    max_drawdown = ((df['cumulative_return'].cummax() - df['cumulative_return']) / df['cumulative_return'].cummax()).max()

    # Win rate
    win_rate = (returns > 0).sum() / len(returns)

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Number of trades (position change > 0)
    num_trades = (df['position_shifted'].diff().abs() > 1e-4).sum()

    return {
        'Cumulative Return': cumulative,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Number of Trades': num_trades
    }


def compute_fear_greed_index(df):
    """
    Estima um índice de Fear & Greed com base em indicadores simples:
    - Volatilidade (ATR)
    - Retorno diário
    - Volume relativo

    Output: Série normalizada de 0 (medo extremo) a 100 (ganância extrema)
    """
    
    fg = pd.DataFrame(index=df.index)

    # Normalized return (momentum)
    fg['return'] = df['Close'].pct_change().rolling(3).mean()

    # Inversed volatility (the highest the stability, the bigger the greed)
    fg['volatility'] = df['Close'].pct_change().rolling(7).std()
    fg['inv_volatility'] = 1 / (fg['volatility'] + 1e-8)

    # Relative volume
    fg['volume'] = df['Volume'].rolling(3).mean()
    fg['volume_norm'] = fg['volume'] / fg['volume'].rolling(30).mean()

    # Score
    fg['score'] = (
        fg['return'].rank(pct=True) +
        fg['inv_volatility'].rank(pct=True) +
        fg['volume_norm'].rank(pct=True)
    ) / 3

    # Escalar para 0–100
    fg['fear_greed_index'] = (fg['score'] * 100).clip(0, 100)

    fg['fg_signal'] = 'hold'

    if 'fear_greed_index' in fg.columns:
        fg.loc[fg['fear_greed_index'] <= 30, 'fg_signal'] = 'buy'
        fg.loc[fg['fear_greed_index'] >= 70, 'fg_signal'] = 'sell'

    return fg[['fear_greed_index', 'fg_signal']]

def generate_momentum_signal(df, window=10, buy_th=0.02, sell_th=-0.02):
    df = df.copy()
    df['momentum'] = df['Close'].pct_change(window)
    df['momentum_signal'] = 'hold'
    df.loc[df['momentum'] > buy_th, 'momentum_signal'] = 'buy'
    df.loc[df['momentum'] < sell_th, 'momentum_signal'] = 'sell'
    return df

def generate_sentiment_signal(df, score_col='avg_sentiment_score', buy_th=0.9, sell_th=0.7):
    df = df.copy()
    df['sent_score_signal'] = 'hold'
    df.loc[df[score_col] > buy_th, 'sent_score_signal'] = 'buy'
    df.loc[df[score_col] < sell_th, 'sent_score_signal'] = 'sell'
    return df

def compare_strategy(df_full, model_name, test_df_cut, price_col='Close', to_plot_signals=True):
    """
    Compare strategy performance for a single SOM model vs:
    - Fear & Greed
    - Momentum (if price history is available)
    - Sentiment Score (if sentiment columns are present)
    - Buy & Hold
    """

    results = {}
    metrics = {}

    # SOM Strategy 
    result_som = simulate_strategy(test_df_cut, signal_col='signal', price_col=price_col, cost=0.001)
    metrics_som = evaluate_strategy(result_som)
    results[f"{model_name} SOM"] = result_som
    metrics[f"{model_name} SOM"] = metrics_som

    # Fear & Greed
    df_fg = test_df_cut.copy()
    fg_signals = compute_fear_greed_index(df_full)
    df_fg['fear_greed_index'] = fg_signals.loc[df_fg.index, 'fear_greed_index']
    df_fg['fg_signal'] = 'hold'
    df_fg.loc[df_fg['fear_greed_index'] <= 30, 'fg_signal'] = 'buy'
    df_fg.loc[df_fg['fear_greed_index'] >= 70, 'fg_signal'] = 'sell'

    result_fg = simulate_strategy(df_fg, signal_col='fg_signal', price_col=price_col, cost=0.001)
    metrics_fg = evaluate_strategy(result_fg)
    results['Fear & Greed'] = result_fg
    metrics['Fear & Greed'] = metrics_fg

    # Momentum Strategy 
    if 'Close' in test_df_cut.columns:
        df_mom = generate_momentum_signal(test_df_cut)
        result_mom = simulate_strategy(df_mom, signal_col='momentum_signal', price_col=price_col, cost=0.001)
        metrics_mom = evaluate_strategy(result_mom)
        results['Momentum'] = result_mom
        metrics['Momentum'] = metrics_mom

    # Sentiment Score Strategy
    sentiment_cols = {'avg_sentiment_score'}
    if sentiment_cols.issubset(set(test_df_cut.columns)):
        df_sent = generate_sentiment_signal(test_df_cut)
        result_sent = simulate_strategy(df_sent, signal_col='sent_score_signal', price_col=price_col, cost=0.001)
        metrics_sent = evaluate_strategy(result_sent)
        results['Sentiment Score'] = result_sent
        metrics['Sentiment Score'] = metrics_sent

    # Plot
    if to_plot_signals:
        plt.figure(figsize=(12, 6))
        for name, res in results.items():
            plt.plot(res['cumulative_return'], label=name)

        plt.plot(result_fg['buy_hold_return'], label='Buy & Hold', color='black', linestyle='--')
        plt.title(f"{model_name} SOM Benchmark Comparison")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Metrics
    print("\nMETRICS")
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'Drawdown':>10} {'WinRate':>8} {'PF':>6} {'Trades':>8}")
    print("-" * 80)
    for name, m in metrics.items():
        print(f"{name:<25} {m['Cumulative Return']:>10.2f} {m['Sharpe Ratio']:>8.2f} {m['Max Drawdown']:>10.2%} "
              f"{m['Win Rate']:>8.2%} {m['Profit Factor']:>6.2f} {m['Number of Trades']:>8}")
    print(f"{'Buy & Hold':<25} {result_fg['buy_hold_return'].iloc[-1]:>10.2f} {'—':>8} {'—':>10} {'—':>8} {'—':>6} {'—':>8}")

    return results, metrics

def compare_trading_strategies(
    df_full,
    som_dfs: dict,
    model_prefixes: dict = None,
    df_signals=None,
    price_col='Close',
    to_plot_signals=True
):
    """
    Unified comparison of SOM-based strategies, benchmarks (Fear & Greed, Momentum),
    sentiment-based, and ensemble strategies.

    Parameters:
        df_full: Full price + sentiment dataframe (used for Fear & Greed)
        som_dfs: Dict with keys like 'technical', 'sentiment', 'hybrid' and their test_dfs
        model_prefixes: Optional custom labels like {'technical': 'Tech SOM'}
        df_signals: DataFrame with ensemble signals (optional)
        price_col: Column name of the price
        to_plot_signals: Whether to plot cumulative returns

    Returns:
        strategies: dict with strategy name as key, tuple(result_df, metrics) as value
    """

    strategies = {}

    # SOM Models 
    for key, test_df_cut in som_dfs.items():
        label = model_prefixes.get(key, f"{key.capitalize()} SOM") if model_prefixes else f"{key.capitalize()} SOM"
        result = simulate_strategy(test_df_cut, signal_col='signal', price_col=price_col, cost=0.001)
        metrics = evaluate_strategy(result)
        strategies[label] = (result, metrics)

    # Fear & Greed 
    df_fg = list(som_dfs.values())[0].copy()  # Use any test_df_cut to align dates
    fg_data = compute_fear_greed_index(df_full)
    df_fg['fear_greed_index'] = fg_data.loc[df_fg.index, 'fear_greed_index']
    df_fg['fg_signal'] = 'hold'
    df_fg.loc[df_fg['fear_greed_index'] <= 30, 'fg_signal'] = 'buy'
    df_fg.loc[df_fg['fear_greed_index'] >= 70, 'fg_signal'] = 'sell'
    result_fg = simulate_strategy(df_fg, signal_col='fg_signal', price_col=price_col, cost=0.001)
    metrics_fg = evaluate_strategy(result_fg)
    strategies['Fear & Greed'] = (result_fg, metrics_fg)

    # Momentum
    if price_col in df_fg.columns:
        df_mom = generate_momentum_signal(df_fg)
        result_mom = simulate_strategy(df_mom, signal_col='momentum_signal', price_col=price_col, cost=0.001)
        metrics_mom = evaluate_strategy(result_mom)
        strategies['Momentum'] = (result_mom, metrics_mom)

    # Sentiment Score (if available) 
    if {'avg_sentiment_score'}.issubset(df_fg.columns):
        df_sent = generate_sentiment_signal(df_fg)
        result_sent = simulate_strategy(df_sent, signal_col='sent_score_signal', price_col=price_col, cost=0.001)
        metrics_sent = evaluate_strategy(result_sent)
        strategies['Sentiment Score'] = (result_sent, metrics_sent)

    # Ensemble Strategies 
    ensemble_signals = {
        'Ensemble Majority': 'ensemble_signal',
        'Ensemble Weighted': 'ensemble_weighted',
        'Ensemble Unanimous': 'ensemble_unanimous',
        'Ensemble Aggressive': 'ensemble_aggressive',
    }

    for name, signal_col in ensemble_signals.items():
        if df_signals is not None and signal_col in df_signals.columns:
            result = simulate_strategy(df_signals, signal_col=signal_col, price_col=price_col, cost=0.001)
            metrics = evaluate_strategy(result)
            strategies[name] = (result, metrics)

    # Plot 
    if to_plot_signals:
        plt.figure(figsize=(14, 7))
        for name, (res_df, _) in strategies.items():
            plt.plot(res_df.index, res_df['cumulative_return'], label=name)
        plt.plot(result_fg.index, result_fg['buy_hold_return'], label='Buy & Hold', color='black', linestyle='--')
        plt.title("Comparison of Trading Strategies")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Metrics 
    print("\nMETRICS")
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'Drawdown':>10} {'WinRate':>8} {'PF':>6} {'Trades':>8}")
    print("-" * 80)
    for name, (_, m) in strategies.items():
        print(f"{name:<25} {m['Cumulative Return']:>10.2f} {m['Sharpe Ratio']:>8.2f} "
              f"{m['Max Drawdown']:>10.2%} {m['Win Rate']:>8.2%} {m['Profit Factor']:>6.2f} {m['Number of Trades']:>8}")
    print(f"{'Buy & Hold':<25} {result_fg['buy_hold_return'].iloc[-1]:>10.2f} {'—':>8} {'—':>10} {'—':>8} {'—':>6} {'—':>8}")

    return strategies
