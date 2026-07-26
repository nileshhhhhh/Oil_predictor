import pandas as pd

def data_cleaning(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['percentChange', 'change']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df
