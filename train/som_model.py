import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score
import joblib
import os
import random
from som_strategy_runner import process_and_signal
from evaluation.strategy_eval import simulate_strategy, evaluate_strategy
from itertools import product
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


np.random.seed(42)
random.seed(42)

def train_som_model(
    data: pd.DataFrame,
    feature_cols: list,
    model_name: str,
    som_size=(10, 10),
    sigma=1.0,
    learning_rate=0.5,
    num_iter=1000,
    save_dir: str = ".",
    to_save=False,
    to_print=False
):
    """
    General SOM training function for any feature set.

    Args:
        data (pd.DataFrame): Input DataFrame
        feature_cols (list): Columns to use for SOM training
        model_name (str): Prefix for saving model/scaler/CSV
        som_size (tuple): Dimensions of SOM grid
        sigma (float): Sigma parameter for SOM
        learning_rate (float): Learning rate
        num_iter (int): Number of iterations
        save_dir (str): Directory to save models and clusters

    Returns:
        X_result (pd.DataFrame): Data with cluster labels
        som (MiniSom): Trained SOM object
    """




    X = data[feature_cols].copy()

    # Drop rows with any NaNs or infs
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    X = X.loc[:, X.nunique() > 1]  # Keep only non-constant columns

    if X.empty or X.shape[0] == 0:
        raise ValueError("No valid data left after dropping NaNs/Infs/constant columns.")


    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled += np.random.normal(loc=0.0, scale=1e-8, size=X_scaled.shape)


    som = MiniSom(som_size[0], som_size[1], X.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed=42)
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, num_iter)

    winners = np.array([som.winner(x) for x in X_scaled])
    node_ids = [f"{x[0]}_{x[1]}" for x in winners]
    X_result = data.loc[X.index].copy()
    X_result['cluster'] = node_ids

    if to_save:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(som, os.path.join(save_dir, f"som_{model_name}__{som_size}_{sigma}_{learning_rate}_{num_iter}.pkl"))
        joblib.dump(scaler, os.path.join(save_dir, f"scaler_{model_name}__{som_size}_{sigma}_{learning_rate}_{num_iter}.pkl"))
        X_result.to_csv(os.path.join(save_dir, f"{model_name}_clusters__{som_size}_{sigma}_{learning_rate}_{num_iter}.csv"))

    quantization_error = som.quantization_error(X_scaled)
    topo_error = som.topographic_error(X_scaled)

    if to_print:
        print(f"Quantization Error: {quantization_error:.4f}")
        print(f"Topographic Error: {topo_error:.4f}")

    return X_result, som

def grid_search_som(model_name, train_df, val_df, cols, save_dir="models", disable=False):
    grid = list(product(
        [(5, 5), (10, 10), (15, 15), (20, 20)],  # SOM sizes
        [0.5, 1.0, 1.5],                         # Sigma
        [0.1, 0.3, 0.5],                         # Learning rate
        [500, 1000, 1500]                        # Iterations
    ))

    results_list = []

    for som_size, sigma, lr, n_iter in tqdm(grid, disable=disable):
        train_clusters, som_model = train_som_model(
            data=train_df,
            feature_cols=cols,
            model_name=model_name,
            som_size=som_size,
            sigma=sigma,
            learning_rate=lr,
            num_iter=n_iter,
            save_dir=save_dir,
            to_save=False,
            to_print=False
        )

        val_df_cut, _ = process_and_signal(train_df, val_df, som_model, cols, to_plot_signals=False, model_name=model_name)
        result = simulate_strategy(val_df_cut, signal_col='signal', price_col='Close', cost=0.001)
        metrics = evaluate_strategy(result)

        results_list.append({
            'Model Name': model_name,
            'SOM Size': som_size,
            'Sigma': sigma,
            'Learning Rate': lr,
            'Iterations': n_iter,
            'Cumulative Return': metrics['Cumulative Return'],
            'Sharpe Ratio': metrics['Sharpe Ratio'],
            'Max Drawdown': metrics['Max Drawdown'],
            'Features': cols  # Optional: store used feature subset
        })

    return pd.DataFrame(results_list)



def evaluate_cluster_stability(data, feature_cols, model_name, som_size, sigma, learning_rate, num_iter, save_dir=None, to_save=False, runs=5):
    all_labels = []
    
    for _ in range(runs):
        clusters, _ = train_som_model(
            data=data,
            feature_cols=feature_cols,
            model_name=model_name,
            som_size=som_size,
            sigma=sigma,
            learning_rate=learning_rate,
            num_iter=num_iter,
            save_dir=save_dir,
            to_save=to_save,
            to_print=False
        )
        # Convert string labels like "9_4" to unique integers
        str_labels = clusters["cluster"].astype(str).values
        unique_str_labels = {label: idx for idx, label in enumerate(sorted(set(str_labels)))}
        labels = [unique_str_labels[label] for label in str_labels]
        all_labels.append(np.array(labels))
    
    scores = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            score = adjusted_rand_score(all_labels[i], all_labels[j])
            scores.append(score)
    
    return np.mean(scores)


