import sys
import statistics

import pandas as pd

from backpropgationNeuralNetwork import BackpropagationNeuralNetwork
from machineLearningUtilities import split_into_random_stratified_groups, one_hot_encoder


def run_backpropagation(df, num_features, num_hidden):
    """
    This function runs a backpropagation neural network on the data frame and outputs statistics from five experiments
    :param df: The data set to run the algorithm on=
    :param num_features: The number of features in this dataset
    """
    # Split dataset 5-fold stratified
    print(f"Size of total dataset = {len(df)}")
    train1, train2, train3, train4, train5 = split_into_random_stratified_groups(df)
    datasets = [train1, train2, train3, train4, train5]
    lg_scores = []
    for i, d in enumerate(datasets):
        print("-------------")
        print(f"Experiment #{i + 1}")
        print("-------------")

        # Use one subset as a test set
        df_test = datasets[i]
        print(f"Test set size = {len(df_test)}")
        training_sets = datasets.copy()

        # Create a training set from remaining subsets
        del training_sets[i]
        df_train = pd.concat(training_sets)
        print(f"Training set size = {len(df_train)}")

        # Create Logistic Regression
        print(df_train.iloc[:, 0:num_features + 1].head())
        lg = BackpropagationNeuralNetwork(df_train.columns[0:num_features],
                                          df_train.iloc[:, 0:num_features + 1],
                                          df_train.iloc[:, num_features],
                                          df_test.iloc[:, 0:num_features + 1],
                                          df_test.iloc[:, num_features],
                                          int(num_hidden))

        # Train with logistic regression
        lg.learn()

        # Test the logistic regression accuracy
        lg_accuracy = lg.make_predictions()
        print('Percent accurate: ' + repr(lg_accuracy) + '%')
        lg_scores.append(lg_accuracy)

    return statistics.mean(lg_scores)


def run_on_breast(file, num_hidden):
    """
    This function runs logistic regression and naive bayes classifier on the breast data set, it encodes
    the classes to Benign = 0, Malignant = 1, and removes missing values
    :param file: input file
    """
    print("_______________________________")
    print("Reading in Breast data set...")
    df_breast = pd.read_csv(file, header=None)
    df_breast.columns = ["Sample Id", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
                         "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin",
                         "Normal Nucleoli", "Mitoses", "Class"]

    # Find missing values and remove them, since there are so few
    # The documentation notes that there are16 missing values in group 1 and 6 denoted by '?'
    # I found 16 values in Group 6
    # Since there are so few missing values I dropped those rows
    df_breast = df_breast[df_breast["Bare Nuclei"] != '?']

    # Drop Sample Id
    df_breast = df_breast.drop('Sample Id', axis=1)

    # Generate boolean classifiers (0 = Benign, 1 = Malignant)
    df_breast.loc[df_breast["Class"] == 2, "Class"] = 0
    df_breast.loc[df_breast["Class"] == 4, "Class"] = 1

    averages = run_backpropagation(df_breast, 9, int(num_hidden))

    print("----------------------------")
    print(f"Averages over 5 experiments")
    print("----------------------------")
    print(f"Averages over all 5 experiments = {averages}%")


def run_on_glass(file, num_hidden):
    """
    This function runs logistic regression and naive bayes classifier on the glass data set
    :param file: input file
    """
    print("_______________________________")
    print("Reading in Glass data set...")
    # Read in glass data
    df_glass = pd.read_csv(file, header=None)
    df_glass.columns = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Class"]

    # This data set has no missing values, so we will skip that step

    # Drop Id
    df_glass = df_glass.drop('Id', axis=1)

    # Encode the class
    df_glass = one_hot_encoder(df_glass, ['Class'])
    df_glass = df_glass.rename(columns={"Class_1": "Class"})
    df_glass = df_glass.drop(columns=['Class_2', 'Class_3', 'Class_5', 'Class_6', 'Class_7'])

    averages = run_backpropagation(df_glass, 9, int(num_hidden))

    print("----------------------------")
    print(f"Averages over 5 experiments")
    print("----------------------------")
    print(f"Averages over all 5 experiments = {averages}%")


def run_on_iris(file, num_hidden):
    """
       This function runs logistic regression and naive bayes classifier on the iris data set
       :param file: input file
       """
    print("_______________________________")
    print("Reading in Iris data set...")
    # Read in iris data
    df_iris = pd.read_csv(file, header=None)
    df_iris.columns = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "Class"]

    # Generate boolean classifiers (0 = not-Iris-virginica, 1 = Iris-virginica)
    df_iris.loc[df_iris["Class"] == "Iris-virginica", "Class_Bool"] = 1
    df_iris.loc[df_iris["Class"] != "Iris-virginica", "Class_Bool"] = 0
    df_iris = df_iris.drop(columns=['Class'])
    df_iris = df_iris.rename(columns={"Class_Bool": "Class"})
    print(df_iris.head())

    # I encoded these by hand based on the box plots that I created
    sepal_length_cond_1 = df_iris["sepal length (cm)"] > 6
    sepal_length_cond_0 = df_iris["sepal length (cm)"] <= 6
    sepal_width_cond_1 = (df_iris["sepal width (cm)"] > 2.7) & (df_iris["sepal width (cm)"] < 3.25)
    sepal_width_cond_0 = (df_iris["sepal width (cm)"] <= 2.7) | (df_iris["sepal width (cm)"] >= 3.25)
    petal_length_cond_1 = df_iris["petal length (cm)"] > 5
    petal_length_cond_0 = df_iris["petal length (cm)"] <= 5
    petal_width_cond_1 = df_iris["petal width (cm)"] > 1.5
    petal_width_cond_0 = df_iris["petal width (cm)"] <= 1.5

    df_iris_encoded = df_iris.copy()
    df_iris_encoded.loc[sepal_length_cond_1, "sepal_length_cond_bool"] = 1
    df_iris_encoded.loc[sepal_length_cond_0, "sepal_length_cond_bool"] = 0
    df_iris_encoded.loc[sepal_width_cond_1, "sepal_width_cond_bool"] = 1
    df_iris_encoded.loc[sepal_width_cond_0, "sepal_width_cond_bool"] = 0
    df_iris_encoded.loc[petal_length_cond_1, "petal_length_cond_bool"] = 1
    df_iris_encoded.loc[petal_length_cond_0, "petal_length_cond_bool"] = 0
    df_iris_encoded.loc[petal_width_cond_1, "petal_width_cond_bool"] = 1
    df_iris_encoded.loc[petal_width_cond_0, "petal_width_cond_bool"] = 0

    df_iris_encoded = df_iris_encoded.drop(columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    df_iris_encoded = df_iris_encoded[['sepal_length_cond_bool',  'sepal_width_cond_bool',  'petal_length_cond_bool',  'petal_width_cond_bool', 'Class']]

    averages = run_backpropagation(df_iris_encoded, 4, int(num_hidden))

    print("----------------------------")
    print(f"Averages over 5 experiments")
    print("----------------------------")
    print(f"Averages over all 5 experiments = {averages}%")


def run_on_soybean(file, num_hidden):
    print("_______________________________")
    print("Reading in Soybean data set...")
    # Read in soybean data
    df_soybean = pd.read_csv(file, header=None)

    # Generate boolean classifiers
    df_soybean = df_soybean.rename(columns={35: 'Class'})
    df_soybean.loc[df_soybean["Class"] == "D1", "Class"] = 0
    df_soybean.loc[df_soybean["Class"] == "D2", "Class"] = 0
    df_soybean.loc[df_soybean["Class"] == "D3", "Class"] = 1
    df_soybean.loc[df_soybean["Class"] == "D4", "Class"] = 1

    # One hot encode breast data set for naive bayes
    columns_to_encode = df_soybean.columns.values.tolist()
    del columns_to_encode[35]
    df_soybean_encoded = one_hot_encoder(df_soybean, columns_to_encode)

    averages = run_backpropagation(df_soybean, 35, int(num_hidden))

    print("----------------------------")
    print(f"Averages over 5 experiments")
    print("----------------------------")
    print(f"Averages over all 5 experiments = {averages}%")


def run_on_votes(file, num_hidden):
    print("_______________________________")
    print("Reading in Votes data set...")
    # Read in votes data
    df_vote = pd.read_csv(file, header=None)

    # Generate boolean classifiers (0 = republican, 1 = democrat)
    df_vote.loc[df_vote[0] == "republican", 0] = 0
    df_vote.loc[df_vote[0] == "democrat", 0] = 1

    # Generate boolean classifiers (0 = n, 1 = y)
    # Set ? to n, since there is no good way to impute votes
    df_vote.replace('n', 0, inplace=True)
    df_vote.replace('y', 1, inplace=True)
    df_vote.replace('?', 0, inplace=True)
    print(df_vote.head())

    # Rename and reorder columns to work with algorithms
    df_vote = df_vote.rename(columns={0: 'Class', 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
                                      8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 14,
                                      16: 15})
    df_vote = df_vote[[c for c in df_vote if c not in ['Class']] + ['Class']]

    averages = run_backpropagation(df_vote, 16, int(num_hidden))

    print("----------------------------")
    print(f"Averages over 5 experiments")
    print("----------------------------")
    print(f"Averages over all 5 experiments = {averages}%")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Please add options for data set (--breast --glass --iris --soybean --vote) "
              "and the full file path when running this script followed by the number of "
              "hidden layers to train")
        print("Example: python runAlgorithms.py --breast  ./data/breast.data 2")
        exit()

    if sys.argv[1] == "--breast":
        run_on_breast(sys.argv[2], sys.argv[3])
        exit()

    if sys.argv[1] == "--glass":
        run_on_glass(sys.argv[2], sys.argv[3])
        exit()

    if sys.argv[1] == "--iris":
        run_on_iris(sys.argv[2], sys.argv[3])
        exit()

    if sys.argv[1] == "--soybean":
        run_on_soybean(sys.argv[2], sys.argv[3])
        exit()

    if sys.argv[1] == "--votes":
        run_on_votes(sys.argv[2], sys.argv[3])
        exit()

    exit()

