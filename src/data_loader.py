import pandas as pd

def load_csv(path: str, target: str):
    df = pd.read_csv(path)
    y = df[target]
    X = df.drop(columns=[target])
    return X, y
