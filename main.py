import pandas as pd
df = pd.read_csv("data/raw/train.csv")

print("Shape: ")
print(df.shape)

print("/nFirst 5 rows: ")
print(df.head())

print("/nDataset info: ")
print(df.info())