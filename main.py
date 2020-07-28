from randomForest import RandomForest
import pandas as pd


def make_new_prediction(file_path, forest):
    new_data = pd.read_csv(file_path)
    prepared_data = forest.prepare_new_data(new_data)
    forest.predict_new_data(prepared_data)


def main():
    forest = RandomForest("mushrooms", n_boostrap=100, n_features=24, test_size=0.2, n_trees=20, tree_max_depth=10)

    forest.test_model()
    forest.print_forest()
    #print("Forest accuracy " + str(forest.accuracy * 100) + "%")


if __name__ == "__main__":
    main()
