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


def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape

    for column_index in range(n_columns - 1):
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)
        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2

                potential_splits[column_index].append(potential_split)

    return potential_splits


def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]
    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values > split_value]
    return data_below, data_above


def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy


def calculate_overall_entropy(data_below, data_above):
    n_data_points = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n_data_points
    p_data_above = len(data_above) / n_data_points
    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))
    return overall_entropy


def determine_best_split(data, potential_splits):
    overall_entropy = 999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    return best_split_column, best_split_value


df = load_data()
random.seed(0)
df_train, df_test = train_test_split(df, 20)
data = df_train.values
potential_splits = get_potential_splits(data)
# sns.lmplot(data=df_train, x="petal_width", y="petal_length", hue="label", fit_reg=False, height=6, aspect=1.5)
# plt.vlines(x=potential_splits[3], ymin=1, ymax=7)
# plt.hlines(y=potential_splits[2], xmin=0, xmax=2.5)
# plt.show()
data_below, data_above = split_data(data, 3, 1.05)
# plotting_df = pd.DataFrame(data,columns=df.columns)
# sns.lmplot(data=plotting_df, x="petal_width", y="petal_length", hue="label", fit_reg=False, height=6, aspect=1.5)
# plt.vlines(x=0.8, ymin=1, ymax=7)
# plt.show()
print(determine_best_split(data,potential_splits))
