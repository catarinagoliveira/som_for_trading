import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def assign_signals(cluster_series, node_action_map):
    """
    Map clusters to trading signals.
    """
    return cluster_series.map(node_action_map)

def simulate_strategy(df, signal_col='signal', price_col='Close', cost=0.001):
    """
    Strategy simulation with overtrading (attemp of) control and cost per transaction.
    """
    df = df.copy()
    df['position'] = 0

    position = 0
    for i in range(len(df)):
        signal = df.iloc[i][signal_col]
        if signal == 'buy' and position == 0:
            position = 1
        elif signal == 'sell' and position == 1:
            position = 0
        
        df.at[df.index[i], 'position'] = position

    # return calculation
    df['returns'] = df[price_col].pct_change().fillna(0)
    df['position_shifted'] = df['position'].shift(1).fillna(0)

    # subtract transaction costs
    trades = df['position_shifted'].diff().abs()
    df['strategy_returns'] = df['position_shifted'] * df['returns'] - trades * cost

    df['cumulative_return'] = (1 + df['strategy_returns']).cumprod()
    df['buy_hold_return'] = (1 + df['returns']).cumprod()

    return df


def evaluate_strategy(df):
    """
    Compute performance metrics for the strategy.
    """
    returns = df['strategy_returns']
    cumulative = df['cumulative_return'].iloc[-1]
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    max_drawdown = ((df['cumulative_return'].cummax() - df['cumulative_return']) / df['cumulative_return'].cummax()).max()

    return {
        'Cumulative Return': cumulative,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown
    }

def plot_strategy(df, title='Strategy vs Buy & Hold'):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(df['cumulative_return'], label='Strategy')
    plt.plot(df['buy_hold_return'], label='Buy & Hold')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

def compute_fear_greed_index(df):
    """
    Estima um índice de Fear & Greed com base em indicadores simples:
    - Volatilidade (ATR)
    - Retorno diário
    - Volume relativo

    Output: Série normalizada de 0 (medo extremo) a 100 (ganância extrema)
    """
    
    fg = pd.DataFrame(index=df.index)

    # 1. Normalized return (momentum)
    fg['return'] = df['Close'].pct_change().rolling(3).mean()

    # 2. Inversed volatility (the highest the stability, the bigger the greed)
    fg['volatility'] = df['Close'].pct_change().rolling(7).std()
    fg['inv_volatility'] = 1 / (fg['volatility'] + 1e-8)

    # 3. Relative volume
    fg['volume'] = df['Volume'].rolling(3).mean()
    fg['volume_norm'] = fg['volume'] / fg['volume'].rolling(30).mean()

    # 4. Score
    fg['score'] = (
        fg['return'].rank(pct=True) +
        fg['inv_volatility'].rank(pct=True) +
        fg['volume_norm'].rank(pct=True)
    ) / 3

    # 5. Escalar para 0–100
    fg['fear_greed_index'] = (fg['score'] * 100).clip(0, 100)

    fg['fg_signal'] = 'hold'

    if 'fear_greed_index' in fg.columns:
        fg.loc[fg['fear_greed_index'] <= 30, 'fg_signal'] = 'buy'
        fg.loc[fg['fear_greed_index'] >= 70, 'fg_signal'] = 'sell'

    return fg[['fear_greed_index', 'fg_signal']]




def compare_strategies(df, model_name, test_df_cut, price_col='Close'):
    """
    Compares 3 strategies: SOM, Fear & Greed, Buy & Hold
    """
    df_fg = test_df_cut.copy()

    fg_signals = compute_fear_greed_index(df_fg)
    df_fg = df_fg.join(fg_signals)

    result_som = simulate_strategy(test_df_cut, signal_col='signal', price_col=price_col, cost=0.001)
    metrics_som = evaluate_strategy(result_som)

    result_fg = simulate_strategy(df_fg, signal_col='fg_signal', price_col=price_col, cost=0.001)
    metrics_fg = evaluate_strategy(result_fg)

    plt.figure(figsize=(12, 6))
    plt.plot(result_som['cumulative_return'], label=f'{model_name} SOM Strategy')
    plt.plot(result_fg['cumulative_return'], label='Fear & Greed Strategy')
    plt.plot(result_fg['buy_hold_return'], label='Buy & Hold')
    plt.title(f"{model_name} benchmark comparison")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nMETRICS")
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>10} {'Drawdown':>12}")
    print("-" * 60)
    print(f"{f'{model_name} SOM Strategy':<25} {metrics_som['Cumulative Return']:>10.2f} {metrics_som['Sharpe Ratio']:>10.2f} {metrics_som['Max Drawdown']:>12.2%}")
    print(f"{'Fear & Greed Strategy':<25} {metrics_fg['Cumulative Return']:>10.2f} {metrics_fg['Sharpe Ratio']:>10.2f} {metrics_fg['Max Drawdown']:>12.2%}")
    print(f"{'Buy & Hold':<25} {result_fg['buy_hold_return'].iloc[-1]:>10.2f} {'—':>10} {'—':>12}")


def compare_all_strategies(
    df_tech, df_sent, df_hybrid, 
    test_df_cut_tech, test_df_cut_sent, test_df_cut_hybrid, 
    price_col='Close'
):
    strategies = {}

    # === 1. SOM Technical ===
    result_tech = simulate_strategy(test_df_cut_tech, signal_col='signal', price_col=price_col, cost=0.001)
    metrics_tech = evaluate_strategy(result_tech)
    strategies['Tech SOM'] = (result_tech, metrics_tech)

    # === 2. SOM Sentiment ===
    result_sent = simulate_strategy(test_df_cut_sent, signal_col='signal', price_col=price_col, cost=0.001)
    metrics_sent = evaluate_strategy(result_sent)
    strategies['Sentiment SOM'] = (result_sent, metrics_sent)

    # === 3. SOM Hybrid ===
    result_hybrid = simulate_strategy(test_df_cut_hybrid, signal_col='signal', price_col=price_col, cost=0.001)
    metrics_hybrid = evaluate_strategy(result_hybrid)
    strategies['Hybrid SOM'] = (result_hybrid, metrics_hybrid)

    # === 4. Fear & Greed Strategy ===
    df_fg = test_df_cut_tech.copy()  # Or hybrid/sent if preferred
    fg_data = compute_fear_greed_index(df_tech)
    df_fg['fear_greed_index'] = fg_data.loc[df_fg.index]['fear_greed_index']
    df_fg['fg_signal'] = 'hold'
    df_fg.loc[df_fg['fear_greed_index'] <= 30, 'fg_signal'] = 'buy'
    df_fg.loc[df_fg['fear_greed_index'] >= 70, 'fg_signal'] = 'sell'

    result_fg = simulate_strategy(df_fg, signal_col='fg_signal', price_col=price_col, cost=0.001)
    metrics_fg = evaluate_strategy(result_fg)
    strategies['Fear & Greed'] = (result_fg, metrics_fg)

    # === Plot all ===
    plt.figure(figsize=(12, 6))
    for name, (result_df, _) in strategies.items():
        plt.plot(result_df['cumulative_return'], label=name)

    # Add Buy & Hold once
    plt.plot(result_fg['buy_hold_return'], label='Buy & Hold', color='black', linestyle='--')

    plt.title("All Strategies Comparison")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()

    # === Print Metrics ===
    print("\n METRICS")
    print(f"{'Strategy':<20} {'Return':>10} {'Sharpe':>10} {'Drawdown':>12}")
    print("-" * 55)
    for name, (_, metrics) in strategies.items():
        print(f"{name:<20} {metrics['Cumulative Return']:>10.2f} {metrics['Sharpe Ratio']:>10.2f} {metrics['Max Drawdown']:>12.2%}")
    print(f"{'Buy & Hold':<20} {result_fg['buy_hold_return'].iloc[-1]:>10.2f} {'—':>10} {'—':>12}")
