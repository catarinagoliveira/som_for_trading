import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def train_som_model(
    data: pd.DataFrame,
    feature_cols: list,
    model_name: str,
    som_size=(10, 10),
    sigma=1.0,
    learning_rate=0.5,
    num_iter=1000,
    save_dir: str = "."
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

    X = data[feature_cols].dropna()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    som = MiniSom(som_size[0], som_size[1], len(feature_cols), sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, num_iter)

    winners = np.array([som.winner(x) for x in X_scaled])
    node_ids = [f"{x[0]}_{x[1]}" for x in winners]
    X_result = data.loc[X.index].copy()
    X_result['cluster'] = node_ids

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(som, os.path.join(save_dir, f"som_{model_name}.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, f"scaler_{model_name}.pkl"))
    X_result.to_csv(os.path.join(save_dir, f"{model_name}_clusters.csv"))

    return X_result, som
