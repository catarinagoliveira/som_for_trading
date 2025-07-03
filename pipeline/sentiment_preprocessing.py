import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_sentiment(df: pd.DataFrame, use_total: bool = False) -> pd.DataFrame:
    """
    Preprocessing of the sentiment dataset.

    Parameters:
        df (pd.DataFrame): Original DataFrame.
        use_total (bool): If True, includes the variable 'total'.

    Returns:
        df (pd.DataFrame): DataFrame with processed columns.
    """
    df = df.copy()
    sentiment_cols = [
        "avg_sentiment_score",
        "avg_sentiment_label_score",
        "pct_positive",
        "pct_negative",
        "pct_neutral",
        "total"
    ]
    df[sentiment_cols] = df[sentiment_cols].fillna(0)

    df["sentiment_score_combined"] = (
        0.7 * df["avg_sentiment_score"] + 0.3 * df["avg_sentiment_label_score"]
    )

    df["sentiment_polarization"] = df["pct_positive"] - df["pct_negative"]

    final_sentiment_cols = [
        "sentiment_score_combined",
        "sentiment_polarization"
    ]

    if use_total:
        df["log_total_sentiment"] = np.log1p(df["total"])
        final_sentiment_cols.append("log_total_sentiment")

    # Z-score
    scaler = StandardScaler()
    df[["z_" + col for col in final_sentiment_cols]] = scaler.fit_transform(df[final_sentiment_cols])

    return df
