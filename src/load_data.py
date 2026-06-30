import pandas as pd

def load_raw_data(path):
    return pd.read_csv(path)


def load_processed_data(path):
    return pd.read_csv(
        path,
        keep_default_na=False
    )