import pandas as pd


df = pd.read_csv("data_csv/object_distence.csv")

df.sort_values(by = "distence", inplace=True)
print(df)