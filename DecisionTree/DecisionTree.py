import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pprint import pprint
import configparser

cong = configparser.ConfigParser()
cong.read('../config.ini')
data_path = cong["Default"]["data_path"]


def load_data():
    df = pd.read_csv(data_path)
    df = df.drop("Id", axis=1)
    df = df.rename(columns={"species": "label"})
    return df


def train_test_split(df, test_size):
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df,test_df


