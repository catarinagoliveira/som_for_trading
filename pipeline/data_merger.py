import pandas as pd

def merge_data(df_tech, df_sent):
    df_tech['date'] = pd.to_datetime(df_tech['Date']).dt.date.astype(str)
    df_merged = pd.merge(df_tech, df_sent, on='date', how='left')
    return df_merged