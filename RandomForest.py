import numpy as np
import pandas as pd
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





# df = pd.read_csv("./Data/iris.csv")
# train_df, test_df = train_test_split(df, test_size=0.2)
# forest = random_forest_algorithm(train_df, n_trees=4, n_bootstrap=80, n_features=4, df_max_depth=6)
# pred = random_forest_predictions(test_df, forest)
#
# for i in range(len(forest)):
#     print("Tree " + str(i+1))
#     print_tree(forest[i])
