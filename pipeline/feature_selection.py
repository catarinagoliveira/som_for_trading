import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from itertools import combinations
import random
from tqdm import tqdm
from strategy_eval import simulate_strategy, evaluate_strategy
from sklearn.metrics import accuracy_score
import importlib
import som_model
importlib.reload(som_model)
from som_model import grid_search_som

from itertools import combinations
import random

def grid_search_features_som(model_name, train_df, train_val, full_feature_list, 
                             save_dir="models", feature_subset_size=8, num_trials=20, 
                             eval_metric="Sharpe Ratio"):

    best_result = None
    best_features = None
    all_results = []

    for i in tqdm(range(num_trials), desc=f"Feature search for {model_name}"):

        # sample feature subset
        feature_subset = random.sample(full_feature_list, feature_subset_size)

        try:
            # get all grid results for the current feature subset
            result_df = grid_search_som(
                model_name=model_name,
                train_df=train_df,
                val_df=train_val,
                cols=feature_subset,
                save_dir=save_dir,
                disable=True
            )
        except Exception as e:
            print(f"[{model_name}] Skipped trial due to error: {e}")
            continue

        # add the feature subset used to every row of this result_df
        result_df["Features"] = [feature_subset] * len(result_df)

        # append all SOM runs for this feature subset
        all_results.append(result_df)

        # track best SOM config for current feature subset
        top_result = result_df.sort_values(by=eval_metric, ascending=False).iloc[0]

        if (best_result is None) or (top_result[eval_metric] > best_result[eval_metric]):
            best_result = top_result
            best_features = feature_subset

    # combine all trial results
    final_df = pd.concat(all_results, ignore_index=True)
    final_df = final_df.sort_values(by=eval_metric, ascending=False).reset_index(drop=True)

    return final_df, best_features
