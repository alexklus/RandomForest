import configparser
import pandas as pd

config = configparser.ConfigParser()


# config["Default"] = {'Data':'../Data/iris.csv', 'label': 'species', 'unnecessary_features': []}
# config["Iris"] = {'Data':'../Data/iris.csv', 'label': 'species', 'unnecessary_features': []}
# config["Titanic"] = {'Data':'../Data/Titanic.csv', 'label': 'Survived',
#                      'unnecessary_features': ["PassengerId", "Name", "Ticket", "Cabin"]}
#
# with open('config.ini','w') as configfile:
#     config.write(configfile)


def load_and_prepare_data(data_set_name):
    config.read("config.ini")

    df = pd.read_csv(config[data_set_name]['data'])
    unnecessary_features = config[data_set_name]["unnecessary_features"].split(", ")

    label = config[data_set_name]['label']
    df["label"] = df[label]

    unnecessary_features.append(label)
    df = df.drop(unnecessary_features, axis=1)
    df = handle_missing_data(df)
    return df


def handle_missing_data(df: pd):
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
    n_unique_values_trashold = 15
    for column in df.columns:
        unique_values = df[column].unique()
        example_value = unique_values[0]

        if isinstance(example_value, str) or (len(unique_values) <= n_unique_values_trashold):
            feature_types.append("categorical")
        else:
            feature_types.append("continuous")
    return feature_types


df = load_and_prepare_data("Titanic")
