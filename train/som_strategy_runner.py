import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from collections import Counter
from dateutil.relativedelta import relativedelta
from strategy_eval import simulate_strategy, evaluate_strategy
import joblib
import os

def get_datasets_from_df(df, train_end='2022-01-01'):
    train_val_df = df[df.index < train_end].copy()
    train_df = train_val_df[:'2020-01-01']
    val_df = train_val_df['2020-01-02':]
    test_df = df[df.index >= train_end].copy()
    return train_val_df, train_df, val_df, test_df

def generate_cluster_signals(df, cluster_ids, price_col='Close', days_ahead=5, buy_th=0.01, sell_th=-0.01, min_cluster_size=5):
    df = df.copy()

    valid_len = min(len(df) - days_ahead, len(cluster_ids))
    df = df.iloc[:valid_len]
    cluster_ids = cluster_ids[:valid_len]

    # Compute future returns
    df['future_return'] = (df[price_col].shift(-days_ahead) - df[price_col]) / df[price_col]

    # Filter clusters with enough samples
    counts = df['cluster'].value_counts()
    valid_clusters = counts[counts >= min_cluster_size].index
    df = df[df['cluster'].isin(valid_clusters)]

    # Compute Sharpe stats
    cluster_stats = df.groupby('cluster')['future_return'].agg(['mean', 'std'])
    cluster_stats['sharpe'] = cluster_stats['mean'] / (cluster_stats['std'] + 1e-6)
    
    # Blend return and sharpe
    cluster_stats['score'] = (
        0.6 * cluster_stats['sharpe'] +
        0.4 * cluster_stats['mean']
        )

    # Use Score-based thresholds
    buy_th = cluster_stats['score'].quantile(0.75)
    sell_th = cluster_stats['score'].quantile(0.25)

    # Assign actions based on Score
    node_action = {}
    for cluster, row in cluster_stats.iterrows():
        if row['score'] > buy_th:
            node_action[cluster] = 'buy'
        elif row['score'] < sell_th:
            node_action[cluster] = 'sell'
        else:
            node_action[cluster] = 'hold'

    return node_action, cluster_stats

def plot_umatrix(som, title="U-Matrix"):
    plt.figure(figsize=(10, 8))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar(label='Distance')
    plt.title(title)
    plt.show()


def plot_signals(df, price_col='Close', title="Signals SOM over time"):
    plt.figure(figsize=(12, 4))
    plt.plot(df[price_col], label='Price')
    plt.scatter(df[df['signal'] == 'buy'].index, df[df['signal'] == 'buy'][price_col], marker='^', color='green', label='Buy')
    plt.scatter(df[df['signal'] == 'sell'].index, df[df['signal'] == 'sell'][price_col], marker='v', color='red', label='Sell')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def process_and_signal(train_df, test_df, som, feature_cols, model_name,
                       to_plot_signals=True, to_save=False,
                       price_col='Close', save_dir="models"):
    
    # Cluster the train set
    scaler = MinMaxScaler()
    train_X = train_df[feature_cols].dropna()
    X_train_scaled = scaler.fit_transform(train_X)
    cluster_ids_train = [f"{x[0]}_{x[1]}" for x in [som.winner(x) for x in X_train_scaled]]

    # Build labeled training DataFrame
    train_cut = train_df.loc[train_X.index].copy()
    train_cut['cluster'] = cluster_ids_train

    # Generate signals based on training clusters
    node_action, _ = generate_cluster_signals(train_cut, cluster_ids_train, price_col=price_col)

    # Apply to test set (only valid rows for features)
    # Fill forward missing values (for test only, training should stay untouched to avoid lookahead bias)
    test_df_filled = test_df.copy()
    test_df_filled[feature_cols] = test_df_filled[feature_cols].ffill().bfill()  # ffill first, bfill just in case
    
    # Now continue with filled values
    test_X = test_df_filled[feature_cols].dropna()

    X_test_scaled = scaler.transform(test_X)
    cluster_ids_test = [f"{x[0]}_{x[1]}" for x in [som.winner(x) for x in X_test_scaled]]

    test_cut_partial = test_df.loc[test_X.index].copy()
    test_cut_partial['cluster'] = cluster_ids_test
    test_cut_partial['signal'] = test_cut_partial['cluster'].map(node_action)

    # Align to full test_df index (to preserve all dates)
    full_test = test_df[[price_col]].copy()
    full_test['signal'] = test_cut_partial['signal']  # only fills known indices, rest stay NaN
    full_test['cluster'] = test_cut_partial['cluster']

    # Optional: fill missing with 'hold' (safer for ensemble voting)
    # full_test['signal'] = full_test['signal'].fillna('hold')

    if to_plot_signals:
        plot_signals(full_test, price_col=price_col, title=f"{model_name.capitalize()} SOM signals (full test set)")

    if to_save:
        joblib.dump(node_action, os.path.join(save_dir, f"node_action_{model_name}.pkl"))

    return full_test, node_action

def plot_density_map(som, train_X, title="SOM Density Map"):
    """
    Plots the density heatmap showing the number of data points assigned to each SOM node.
    
    Parameters:
    - som: trained MiniSom instance
    - train_X: original training DataFrame with selected features
    - title: title of the plot
    """

    scaler = MinMaxScaler()
    scaler.fit(train_X)
    data = scaler.transform(train_X)

    hit_map = np.zeros((som._weights.shape[0], som._weights.shape[1]))
    
    for x in data:
        winner = som.winner(x)
        hit_map[winner] += 1

    plt.figure(figsize=(10, 8))
    plt.pcolor(hit_map.T, cmap='viridis')
    plt.colorbar(label='Hits per Neuron')
    plt.title(title)
    plt.tight_layout()
    plt.show()