import pandas as pd
import pandas_ta as ta

def add_technical_indicators(df):
    df = df.copy()

    # To transform the date column to numeric if not already
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["RSI"] = ta.rsi(df["Close"])
    df["MACD"] = ta.macd(df["Close"])["MACD_12_26_9"]
    df["CCI"] = ta.cci(df["High"], df["Low"], df["Close"])
    df["WILLR"] = ta.willr(df["High"], df["Low"], df["Close"])
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)
    df["EMA_50"] = ta.ema(df["Close"], length=50)
    df["EMA_200"] = ta.ema(df["Close"], length=200)
    
    bb = ta.bbands(df["Close"], length=5, std=2.0)
    df["Bollinger_Upper"] = bb["BBU_5_2.0"]
    df["Bollinger_Lower"] = bb["BBL_5_2.0"]

    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"])
    df["OBV"] = ta.obv(df["Close"], df["Volume"])
    df["CMF"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"])
    df["Momentum"] = ta.mom(df["Close"])
    df["ROC"] = ta.roc(df["Close"])
    df["UO"] = ta.uo(df["High"], df["Low"], df["Close"])
    
    return df
