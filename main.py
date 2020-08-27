import random
import seaborn as sns
from randomForest import RandomForest
import pandas as pd
import matplotlib.pyplot as plt


def make_new_prediction(file_path, forest):
    new_data = pd.read_csv(file_path)
    prepared_data = forest.prepare_new_data(new_data)
    forest.predict_new_data(prepared_data)


def fail_example():
    forest = RandomForest("diabetes",
                          n_boostrap=50,
                          n_features=8,
                          test_size=0.2,
                          n_trees=20,
                          tree_max_depth=10
                          )

    forest.test_model()
    forest.print_forest()
    print("Forest accuracy " + str(forest.accuracy * 100) + "%")


def succsesful_example():
    forest = RandomForest("mushrooms",
                          n_boostrap=100,
                          n_features=10,
                          test_size=0.2,
                          n_trees=10,
                          tree_max_depth=10
                          )

    forest.test_model()
    forest.print_forest()
    print("Forest accuracy " + str(forest.accuracy * 100) + "%")


def plot_hard_dataset():
    df = pd.read_csv("./Data/diabetes.csv")
    data = df.values
    plo = pd.DataFrame(data, columns=df.columns)
    columns = [column for column in df.columns if column != "Outcome"]
    for column in columns:
        sns.lmplot(data=plo, x=column, y="Outcome", fit_reg=False, aspect=1.5)
        plt.show()


def main():
    fail_example()
    succsesful_example()



if __name__ == "__main__":
    main()
    #plot_hard_dataset()

