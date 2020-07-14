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

    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


def classify_data(data):

    label_column = data[:, -1]
    unique_classes, count_unique_classes = np.unique(label_column, return_counts=True)

    index = count_unique_classes.argmax()
    classification = unique_classes[index]
    return classification


df = load_data()
random.seed(0)
df_train, df_test = train_test_split(df, 0.2)
data = df_train.values
print(classify_data(data))
