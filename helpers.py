import pandas as pd
import numpy as np

def convert_quarter_to_date(quarter):
    year, qtr = quarter.split(" ")
    year = int(year)
    if qtr == "Q1":
        return pd.Timestamp(f"{year}-01-01")
    elif qtr == "Q2":
        return pd.Timestamp(f"{year}-04-01")
    elif qtr == "Q3":
        return pd.Timestamp(f"{year}-07-01")
    elif qtr == "Q4":
        return pd.Timestamp(f"{year}-10-01")

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)].values)
        y.append(data.iloc[i + time_step])
    return np.array(X), np.array(y)