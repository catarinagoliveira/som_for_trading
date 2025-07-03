import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def run_finbert(df_news):
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    sentiment_results = []
    for text in tqdm(df_news["cleaned_text"], desc="Running FinBERT"):
        if not text:
            sentiment_results.append({"label": "NEUTRAL", "score": 0.0})
            continue
        try:
            result = finbert(text[:512])[0]
            sentiment_results.append(result)
        except:
            sentiment_results.append({"label": "NEUTRAL", "score": 0.0})

    df_news["sentiment_label"] = [r["label"] for r in sentiment_results]
    df_news["sentiment_score"] = [r["score"] for r in sentiment_results]
    return df_news

def aggregate_sentiment(df_news):
    df_news["Date"] = pd.to_datetime(df_news["date_time"]).dt.date.astype(str)
    sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    df_news['sentiment_numeric'] = df_news['sentiment_label'].map(sentiment_map)
    
    agg_sentiment = df_news.groupby('date').agg(
        avg_sentiment_score=('sentiment_score', 'mean'),
        avg_sentiment_label_score=('sentiment_numeric', 'mean'),
        pos_count=('sentiment_label', lambda x: (x == 'Positive').sum()),
        neut_count=('sentiment_label', lambda x: (x == 'Neutral').sum()),
        neg_count=('sentiment_label', lambda x: (x == 'Negative').sum()),
        total=('sentiment_label', 'count')
    ).reset_index()

    agg_sentiment['pct_positive'] = agg_sentiment['pos_count'] / agg_sentiment['total']
    agg_sentiment['pct_neutral'] = agg_sentiment['neut_count'] / agg_sentiment['total']
    agg_sentiment['pct_negative'] = agg_sentiment['neg_count'] / agg_sentiment['total']
    
    return agg_sentiment[
        ['Date', 'avg_sentiment_score', 'avg_sentiment_label_score',
         'pos_count', 'neut_count', 'neg_count', 'total',
         'pct_positive', 'pct_neutral', 'pct_negative']
    ]