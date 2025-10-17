import pandas as pd

def load_data(filepath):
    """Load customer dataset"""
    df = pd.read_csv(filepath)
    return df

def describe_features(df):
    """Print basic statistics of numeric features"""
    print(df[['Age', 'Annual Income (k$)', 'Spending Score (1–100)']].describe())
