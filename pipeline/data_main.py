import pandas as pd
import os
from pipeline.data_collector import download_numerical_data
from pipeline.technical_indicators import add_technical_indicators
from pipeline.sentiment_pipeline import clean_text, run_finbert, aggregate_sentiment
from pipeline.sentiment_preprocessing import preprocess_sentiment

from config import TICKERS, START_DATE, END_DATE, RAW_DIR, DATA_DIR

def merge_data(df_tech, df_sent):
    df_tech["Date"] = pd.to_datetime(df_tech["Date"])
    df_sent["Date"] = pd.to_datetime(df_sent["Date"])
    df_merged = pd.merge(df_tech, df_sent, on="Date", how="left")
    return df_merged


def print_final_stats(df: pd.DataFrame, name="Dataset"):
    print(f"\n{name} Info")
    print("-" * 40)
    print(f"Columns: {list(df.columns)}")
    print(df.describe().T)
    if 'date' in df.columns:
        print(f"\nDate range: {df['date'].min()} → {df['date'].max()}")

def run_full_pipeline():
    os.makedirs(os.path.join("..", RAW_DIR), exist_ok=True)
    os.makedirs(os.path.join("..", DATA_DIR), exist_ok=True)

    # Download 
    download_numerical_data(TICKERS, START_DATE, END_DATE, save_dir=RAW_DIR)
    df_price = pd.read_csv(f"{RAW_DIR}/BTC-USD_data.csv", skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"])
    df_price["Date"] = pd.to_datetime(df_price["Date"])


    # Technical Indicators 
    df_tech = add_technical_indicators(df_price)
    df_tech.to_csv(f"{DATA_DIR}/numerical.csv", index=False)

    # Sentiment Analysis
    df_news = pd.read_csv("hf://datasets/edaschau/bitcoin_news/BTC_yahoo.csv")
    df_news["cleaned_text"] = df_news["article_text"].apply(clean_text)
    df_news = run_finbert(df_news)
    df_sentiment = aggregate_sentiment(df_news)
    df_sentiment = preprocess_sentiment(df_sentiment)
    df_sentiment.to_csv(f"{DATA_DIR}/sentiment.csv", index=False)
    

    # Merge
    df_merged = merge_data(df_tech, df_sentiment)

    # Save final dataset
    os.makedirs(DATA_DIR, exist_ok=True)
    df_merged.to_csv(f"{DATA_DIR}/merged.csv", index=False)
    print("Dataset saved to data/merged.csv")

    print_final_stats(df_merged, name="Merged Dataset")

    return df_merged
