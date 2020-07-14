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


def decision_tree_algorithm(df, counter=0, min_sample=2, max_depth=5, ):
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values

    else:
        data = df

    if check_purity(data) or (len(data) < min_sample) or (counter == max_depth):
        classification = classify_data(data)
        return classification
    else:

        counter += 1
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)

        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}

        yes_answer = decision_tree_algorithm(data_below, counter, min_sample, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_sample, max_depth)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()

    if example[feature_name] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)


def calculate_accuracy(df,tree):
    df["classification"] = df.apply(classify_example,axis=1, args=(tree,))
    df["classification_correct"] = df.classification == df.label
    accuracy = df.classification_correct.mean()
    return accuracy

df = load_data()
random.seed(0)
df_train, df_test = train_test_split(df, 20)
# data = df_train.values
# potential_splits = get_potential_splits(data)
# sns.lmplot(data=df_train, x="petal_width", y="petal_length", hue="label", fit_reg=False, height=6, aspect=1.5)
# plt.vlines(x=potential_splits[3], ymin=1, ymax=7)
# plt.hlines(y=potential_splits[2], xmin=0, xmax=2.5)
# plt.show()
# data_below, data_above = split_data(data, 3, 1.05)
# plotting_df = pd.DataFrame(data,columns=df.columns)
# sns.lmplot(data=plotting_df, x="petal_width", y="petal_length", hue="label", fit_reg=False, height=6, aspect=1.5)
# plt.vlines(x=0.8, ymin=1, ymax=7)
# plt.show()
# print(determine_best_split(data, potential_splits))
tree = decision_tree_algorithm(df_train, max_depth=3)

acc = calculate_accuracy(df_test,tree)
print(acc)
print(df_test.loc[77])


