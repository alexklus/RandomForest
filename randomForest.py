from builtins import print

import numpy as np
import pandas as pd
import configparser

from DecisionTree.DecisionTree import decision_tree_algorithm, train_test_split, calculate_accuracy, \
    decision_tree_predictions, make_predictions,print_tree

config = configparser.ConfigParser()
config.read("config.ini")


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
    random_forest_predictions = df_predictions.mode(axis=1)[0]

    return random_forest_predictions


def load_and_prepare_data(data_set_name):
    df = pd.read_csv(config[data_set_name]['path'])
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


class RandomForest:
    def __init__(self, data_set_name, n_boostrap, n_features, test_size=0.2, n_trees=10, tree_max_depth=3):
        self.data_set_name = data_set_name

        self.df = load_and_prepare_data(data_set_name)

        self.train_df, self.test_df = train_test_split(self.df, test_size)

        self.forest = random_forest_algorithm(self.train_df,
                                              n_trees=n_trees,
                                              n_bootstrap=n_boostrap,
                                              n_features=n_features,
                                              df_max_depth=tree_max_depth)
        self.test_predictions = None
        self.accuracy = None

    def test_model(self):
        self.test_predictions = random_forest_predictions(self.test_df, self.forest)
        self.accuracy = calculate_accuracy(self.test_predictions, self.test_df.label)

    def prepare_new_data(self, data):
        label = config[self.data_set_name]['label']
        data['label'] = 0
        unnecessary_features = config[self.data_set_name]["unnecessary_features"].split(", ")
        if unnecessary_features[0] != "None":
            for unnecessary_feature in unnecessary_features:
                if unnecessary_feature in data.columns:

                    data = data.drop(unnecessary_feature,axis=1)
        return data

    def predict_new_data(self, data):
        data = handle_missing_data(data)
        new_predictions = random_forest_predictions(data, self.forest)
        data['label'] = new_predictions.values
        data.to_csv(r"result.csv")

    def print_forest(self):
        for tree in self.forest:
            print_tree(tree, "")


