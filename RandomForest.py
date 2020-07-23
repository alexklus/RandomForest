from builtins import print

import numpy as np
import pandas as pd
import configparser

from numpy import long

from DecisionTree.DecisionTree import decision_tree_algorithm, train_test_split, calculate_accuracy, \
    decision_tree_predictions, print_tree


def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    return df_bootstrapped


def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, df_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=df_max_depth, random_subspace=n_features)
        forest.append(tree)
    return forest


def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions
    df_predictions = pd.DataFrame(df_predictions)

    return df_predictions


def load_and_prepare_data(data_set_name):
    config = configparser.ConfigParser()
    config.read("config.ini")

    df = pd.read_csv(config[data_set_name]['data'])
    unnecessary_features = config[data_set_name]["unnecessary_features"].split(", ")

    if unnecessary_features[0] == "None":
        unnecessary_features = []

    label = config[data_set_name]['label']
    df["label"] = df[label]

    unnecessary_features.append(label)
    df = df.drop(unnecessary_features, axis=1)
    df = handle_missing_data(df)
    return df


def handle_missing_data(df):
    feature_types = determine_type_of_feature(df)
    i = 0
    for column in df.columns:
        if feature_types[i] == "categorical":
            fill_value = df[column].mode()[0]
            df = df.fillna({column: fill_value})
        else:
            fill_value = df[column].median()
            df = df.fillna({column: fill_value})
            i += 1
    return df


def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_threshold = 15
    for column in df.columns:
        unique_values = df[column].unique()
        example_value = unique_values[0]

        if isinstance(example_value, str) or (len(unique_values) <= n_unique_values_threshold):
            feature_types.append("categorical")
        else:
            feature_types.append("continuous")
    return feature_types


df = load_and_prepare_data("Iris")

train_df, test_df = train_test_split(df, test_size=0.2)
forest = random_forest_algorithm(train_df, n_trees=4, n_bootstrap=500, n_features=4, df_max_depth=6)
prediction = random_forest_predictions(test_df, forest)

for i in range(len(forest)):
    print("Tree " + str(i + 1))
    print_tree(forest[i])
