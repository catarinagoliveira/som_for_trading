import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from collections import Counter
from dateutil.relativedelta import relativedelta
from strategy_eval import simulate_strategy, evaluate_strategy

def get_datasets_from_df(df, train_start='2022-01-01', train_end='2022-01-01'):
    train_df = df[df.index < train_start].copy()
    test_df = df[df.index >= train_end].copy()
    return train_df, test_df

def generate_cluster_signals(df, cluster_ids, price_col='Close', days_ahead=5, buy_th=0.01, sell_th=-0.01):
    df = df.copy()

    # Ajustar o comprimento de cluster_ids e df
    valid_len = min(len(df) - days_ahead, len(cluster_ids))
    df = df.iloc[:valid_len]
    cluster_ids = cluster_ids[:valid_len]

    df['cluster'] = cluster_ids

    future_returns = (df[price_col].shift(-days_ahead) - df[price_col]) / df[price_col]
    df['future_return'] = future_returns

    cluster_mean_return = df.groupby('cluster')['future_return'].mean()
    
    counts = df['cluster'].value_counts()
    valid_clusters = counts[counts >= 5].index  # usa clusters com pelo menos 5 amostras
    cluster_mean_return = cluster_mean_return.loc[valid_clusters]

    buy_th = cluster_mean_return.quantile(0.66)
    sell_th = cluster_mean_return.quantile(0.33)

    node_action = {}
    for cluster, mean_ret in cluster_mean_return.items():
        if mean_ret > buy_th:
            node_action[cluster] = 'buy'
        elif mean_ret < sell_th:
            node_action[cluster] = 'sell'
        else:
            node_action[cluster] = 'hold'

    return node_action, cluster_mean_return



def plot_umatrix(som, title="U-Matrix"):
    plt.figure(figsize=(10, 8))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar(label='Distance')
    plt.title(title)
    plt.show()


def plot_signals(df, price_col='Close', title="Sinais SOM vs Preço"):
    plt.figure(figsize=(12, 4))
    plt.plot(df[price_col], label='Price')
    plt.scatter(df[df['signal'] == 'buy'].index, df[df['signal'] == 'buy'][price_col], marker='^', color='green', label='Buy')
    plt.scatter(df[df['signal'] == 'sell'].index, df[df['signal'] == 'sell'][price_col], marker='v', color='red', label='Sell')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def process_and_signal(df, som, feature_cols, price_col='Close'):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df[feature_cols].dropna())
    cluster_ids = [f"{x[0]}_{x[1]}" for x in [som.winner(x) for x in X_train]]

    df_cut = df[feature_cols].dropna().copy()
    df_cut[price_col] = df.loc[df_cut.index][price_col]
    df_cut['cluster'] = cluster_ids

    node_action, _ = generate_cluster_signals(df_cut, cluster_ids, price_col=price_col)

    X_test = scaler.transform(df[feature_cols].dropna())
    cluster_ids_test = [f"{x[0]}_{x[1]}" for x in [som.winner(x) for x in X_test]]

    test_df_cut = df[feature_cols].dropna().copy()
    test_df_cut[price_col] = df.loc[test_df_cut.index][price_col]
    test_df_cut['cluster'] = cluster_ids_test
    test_df_cut['signal'] = test_df_cut['cluster'].map(node_action)

    plot_signals(test_df_cut, price_col=price_col, title="Sinais SOM de Sentimento vs Preço")
    return test_df_cut

def process_and_signal_split(train_df, test_df, som, feature_cols, price_col='Close'):
    # Fit scaler only on training data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].dropna())
    cluster_ids_train = [f"{x[0]}_{x[1]}" for x in [som.winner(x) for x in X_train]]

    # Create training cut
    train_cut = train_df[feature_cols].dropna().copy()
    train_cut[price_col] = train_df.loc[train_cut.index][price_col]
    train_cut['cluster'] = cluster_ids_train

    # Learn actions
    node_action, _ = generate_cluster_signals(train_cut, cluster_ids_train, price_col=price_col)

    # Apply to test set
    X_test = scaler.transform(test_df[feature_cols].dropna())
    cluster_ids_test = [f"{x[0]}_{x[1]}" for x in [som.winner(x) for x in X_test]]

    test_cut = test_df[feature_cols].dropna().copy()
    test_cut[price_col] = test_df.loc[test_cut.index][price_col]
    test_cut['cluster'] = cluster_ids_test
    test_cut['signal'] = test_cut['cluster'].map(node_action)

    plot_signals(test_cut, price_col=price_col, title="Sinais SOM (aplicados ao Teste)")
    return test_cut

def run_rolling_som_strategy(
    df, feature_cols, price_col='Close',
    window_years=2, step_months=1, days_ahead=5,
    som_size=(10, 10), num_iteration=500,
    cost=0.001, strategy_name="SOM"
):
    """
    Applies a SOM-based strategy using a rolling window.

    Args:
        df (pd.DataFrame): DataFrame containing feature columns and price.
        feature_cols (list): List of columns used to train the SOM.
        price_col (str): Column containing the price (default: 'Close').
        window_years (int): Size of the training window in years.
        step_months (int): Step size of the rolling window in months.
        days_ahead (int): Forecast horizon in days for future return.
        som_size (tuple): Size (x, y) of the SOM grid.
        num_iteration (int): Number of training iterations for the SOM.
        cost (float): Transaction cost per trade.
        strategy_name (str): Name of the strategy for logging purposes.

    Returns:
        pd.DataFrame: Table with performance metrics per window.
    """

    start_date = pd.to_datetime("2017-01-01")
    end_date = pd.to_datetime("2023-01-01")
    results = []

    while start_date + relativedelta(years=window_years) < end_date:
        train_start = start_date
        train_end = start_date + relativedelta(years=window_years)
        test_start = train_end
        test_end = test_start + relativedelta(months=step_months)

        train_df = df[(df.index >= train_start) & (df.index < train_end)].copy()
        test_df = df[(df.index >= test_start) & (df.index < test_end)].copy()

        if len(train_df) < 200 or len(test_df) < 20:
            start_date += relativedelta(months=step_months)
            continue

        # 1. Normalize data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].dropna())
        X_test = scaler.transform(test_df[feature_cols].dropna())

        # 2. Train SOM
        som = MiniSom(x=som_size[0], y=som_size[1], input_len=len(feature_cols),
                      sigma=1.0, learning_rate=0.5)
        som.random_weights_init(X_train)
        som.train_random(X_train, num_iteration=num_iteration)

        # 3. Cluster IDs
        cluster_ids_train = [f"{x[0]}_{x[1]}" for x in [som.winner(x) for x in X_train]]
        cluster_ids_test = [f"{x[0]}_{x[1]}" for x in [som.winner(x) for x in X_test]]

        # 4. Signal mapping
        train_df_cut = train_df[feature_cols].dropna().copy()
        train_df_cut[price_col] = train_df.loc[train_df_cut.index][price_col]
        node_action, _ = generate_cluster_signals(train_df_cut, cluster_ids_train,
                                                  price_col=price_col, days_ahead=days_ahead)

        # 5. Apply to test set
        test_df_cut = test_df[feature_cols].dropna().copy()
        test_df_cut[price_col] = test_df.loc[test_df_cut.index][price_col]
        test_df_cut['cluster'] = cluster_ids_test
        test_df_cut['signal'] = test_df_cut['cluster'].map(node_action)

        # 6. Simulation
        result = simulate_strategy(test_df_cut, signal_col='signal',
                                   price_col=price_col, cost=cost)
        metrics = evaluate_strategy(result)
        metrics['window_start'] = train_start
        metrics['window_end'] = train_end
        metrics['strategy'] = strategy_name
        results.append(metrics)

        start_date += relativedelta(months=step_months)

    return pd.DataFrame(results)


def summarize_rolling_results(rolling_results):
    pd.set_option("display.max_columns", None)
    print(rolling_results)
    print("\nAverage of metrics:")
    print(rolling_results.mean(numeric_only=True))

    print("\nBest window:")
    print(rolling_results.loc[rolling_results['Cumulative Return'].idxmax()])

    print("\nWorse window:")
    print(rolling_results.loc[rolling_results['Cumulative Return'].idxmin()])

    rolling_results.set_index('window_end').plot(
        y=['Sharpe Ratio', 'Max Drawdown'],
        title='Sharpe Ratio e Drawdown ao longo do tempo',
        figsize=(10, 4)
    )
    plt.show()