#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages
import pandas as pd
import numpy as np


# # Utilities

def one_hot_encoder(df, columns):
    """
    This function one hot encodes a set of data
    :param df: data to encode
    :param columns: columns to encode
    :return: one hot encoded data
    """
    one_hot = pd.get_dummies(df, columns=columns, prefix=columns)
    df = df.drop(columns, axis=1)
    return one_hot


def split_test_train(df):
    """
    This function splits test and train sets by 2/3 and 1/3 randomly
    :param df: data to split
    :return: train and test set
    """
    train = df.sample(frac=2 / 3)
    test = df.loc[~df.index.isin(train.index), :]
    return train, test


def get_num_similarities(array1, array2):
    """
    This function compares two arrays for similiaries and returns number of equal elements
    :param array1: first array to compare
    :param array2: second array to compare
    :return: the number of similar elements
    """
    a = np.array(array1)
    b = np.array(array2)
    return np.sum(a == b)


# # Winnow-2
# TODO: move this to a class
def promote(X, w, alpha):
    """
    This function performs promotion for the winnow-2 algorithm
    :param X: the set of features values for the sample
    :param w: the current weight array
    :param alpha: alpha hyperparamter
    :return: new weight array
    """
    for index, x in enumerate(X):
        if x == 1:
            w[index] *= alpha

    return w


def demote(X, w, alpha):
    """
    This function performs demotion for the winnow-2 algorithm
    :param X: the set of features values for the sample
    :param w: the current weight array
    :param alpha: alpha hyperparamter
    :return: new weight array
    """
    for index, x in enumerate(X):
        if x == 1:
            w[index] /= alpha

    return w


def make_prediction(X, w, theta):
    """
    This function makes predictions for the winnow-2 algorithms
    :param X: the set of feature values for the sample
    :param w: the current weight array
    :param theta: theta hyperparameter
    :return: prediction 1 or 0
    """
    sum = 0
    for index, x in enumerate(X):
        sum += w[index] * int(x)

    if sum > theta:
        return 1

    return 0


def winnow_2(X, w, cls, theta, alpha):
    """
    This function implements the winnow-2 algorithm for a single example
    :param X: the feature set
    :param w: the weights
    :param cls: the correct class
    :param theta: theta hyperparameter
    :param alpha: alpha hyperparameter
    :return: new weights and prediction
    """
    # Make the prediction on X
    prediction = make_prediction(X, w, theta)

    # Test prediction against class
    if prediction == cls:
        return w, prediction

    if prediction == 0 and cls == 1:
        w = promote(X, w, alpha)
    else:
        w = demote(X, w, alpha)

    return w, prediction


def train_winnow2(X_train, classes, alpha, theta):
    """
    Run winnow-2 on the training set
    :param X_train: the training set
    :param classes: the set of actual classes to be compared to predictions
    :param alpha: the alpha hyperparameter
    :param theta: the theta hyperparameter
    :return: new weight array
    """

    print("\n")
    print("WINNOW 2")
    print("\n")

    # Initialize w to all 1s
    w = [1] * (X_train.shape[1])
    print('w = ' + str(w))
    print("Number of attributes = " + str(len(w)))

    # TRAIN
    predictions = []
    for index, row in X_train.iterrows():
        (w, p) = winnow_2(row, w, classes[index], theta, alpha)
        predictions.append(p)

    # Compare predictions to actual
    print("Training Predictions (% Correct)")
    print(get_num_similarities(classes, predictions) / len(X_train) * 100)

    return w


def test_winnow2(X_test, classes, alpha, theta, w):
    """
    Run winnow-2 on the tes set, with the weights
    created by the training function
    :param X_test: the test set
    :param classes: the set of actual classes to be compared to predictions
    :param alpha: the alpha hyperparameter
    :param theta: the theta hyperparameter
    :param w: new weight array
    """
    predictions = []
    for index, row in X_test.iterrows():
        (w, p) = winnow_2(row, w, classes[index], theta, alpha)
        predictions.append(p)

    # Compare predictions to actual
    print("Testing Predictions (% Correct)")
    print(get_num_similarities(classes, predictions) / len(X_test) * 100)


# # Naive Bayes
# TODO: move this to a class
def get_naive_bayes_model(classes, features):
    """
    This function creates the naive bayes model.
    :param classes: the set of actual class values
    :param features: the set of feature values for each sample
    :return: the model, which consists of P(C=0), P(C=1), P(F=1|C=0), P(F=0|C=0), P(F=1|C=1) and P(F=0|C=1)
    """
    # Create the model
    value_counts = classes.value_counts().to_dict()
    num_samples = len(classes)
    prob_0 = value_counts[0] / num_samples if 0 in value_counts else 0
    prob_1 = value_counts[1] / num_samples if 1 in value_counts else 1
    feature_probs_f0_given0 = []
    feature_probs_f1_given0 = []
    feature_probs_f0_given1 = []
    feature_probs_f1_given1 = []
    zero_class_indexes = [i for i, x in enumerate(classes) if x == 0]
    one_class_indexes = [i for i, x in enumerate(classes) if x == 1]
    num_zero_classes = len(zero_class_indexes)
    num_one_classes = len(one_class_indexes)
    for index, colName in enumerate(features.columns):
        f_value_counts_given_0 = features.iloc[zero_class_indexes, index].value_counts().to_dict()
        f_value_counts_given_1 = features.iloc[one_class_indexes, index].value_counts().to_dict()
        prob_f0_given_c0 = f_value_counts_given_0[0] / num_zero_classes if 0 in f_value_counts_given_0 else 0
        prob_f1_given_c0 = f_value_counts_given_0[1] / num_zero_classes if 1 in f_value_counts_given_0 else 0
        prob_f0_given_c1 = f_value_counts_given_1[0] / num_one_classes if 0 in f_value_counts_given_1 else 0
        prob_f1_given_c1 = f_value_counts_given_1[1] / num_one_classes if 1 in f_value_counts_given_1 else 0
        feature_probs_f0_given0.append(prob_f0_given_c0)
        feature_probs_f1_given0.append(prob_f1_given_c0)
        feature_probs_f0_given1.append(prob_f0_given_c1)
        feature_probs_f1_given1.append(prob_f1_given_c1)

    print("THE MODEL")
    print("Probability of C = 0:", prob_0)
    print("Probability of C = 1:", prob_1)
    print("Probabilities of f = 0, given c = 0:", feature_probs_f0_given0)
    print("Probabilities of f = 1, given c = 0:", feature_probs_f1_given0)
    print("Probabilities of f = 0, given c = 1:", feature_probs_f0_given1)
    print("Probabilities of f = 1, given c = 1:", feature_probs_f1_given1)
    return (prob_0, prob_1, feature_probs_f0_given0,
            feature_probs_f1_given0, feature_probs_f0_given1,
            feature_probs_f1_given1)


def naive_bayes_make_predictions(data, prob_0, prob_1, feature_probs_f0_given0,
                                 feature_probs_f1_given0, feature_probs_f0_given1,
                                 feature_probs_f1_given1):
    """
    This function makes predictions for the naive bayes algorithm
    :param data: The data to create the model from for naive bayes
    :param prob_0: P(C=0)
    :param prob_1: P(C=1)
    :param feature_probs_f0_given0: P(F=0|C=0)
    :param feature_probs_f1_given0: P(F=1|C=0)
    :param feature_probs_f0_given1: P(F=0|C=1)
    :param feature_probs_f1_given1: P(F=1|C=1)
    :return: the predictions made on the set of values, based on the model given
    """

    # Calculate predictions by calculating the probability of each 
    # C=1 and C=0 in each sample(row)
    predictions = []
    for index, row in data.iterrows():
        c0_product = prob_0
        c1_product = prob_1
        for feature_index, feature in enumerate(row):
            if feature == 0:
                c0_product *= feature_probs_f0_given0[feature_index]
                c1_product *= feature_probs_f0_given1[feature_index]
            else:
                c0_product *= feature_probs_f1_given0[feature_index]
                c1_product *= feature_probs_f1_given1[feature_index]

        if c0_product > c1_product:
            predictions.append(0)
        else:
            predictions.append(1)

    return predictions


def naive_bayes(train_classes, test_classes, X_train, X_test):
    """
    This function runs the naive bayes algorithm on a training set, then a test set
    and compares predictions with actual classes and outputs the predictions
    :param train_classes: the training set of classes
    :param test_classes: the test set of classes
    :param X_train: the training datset
    :param X_test: the test dataset
    """

    print("\n")
    print("NAIVE BAYES")
    print("\n")

    (prob_0, prob_1,
     feature_probs_f0_given0,
     feature_probs_f1_given0,
     feature_probs_f0_given1,
     feature_probs_f1_given1) = get_naive_bayes_model(train_classes, X_train)

    # TRAIN
    predictions_train = naive_bayes_make_predictions(X_train, prob_0, prob_1,
                                                     feature_probs_f0_given0,
                                                     feature_probs_f1_given0,
                                                     feature_probs_f0_given1,
                                                     feature_probs_f1_given1)

    # Compare training predictions to actual
    print("\n")
    print("TRAINING PREDICTIONS")
    print(predictions_train)
    print("Training Predictions (% Correct)")
    print(get_num_similarities(train_classes, predictions_train) / len(X_train) * 100)

    # TEST
    predictions = naive_bayes_make_predictions(X_test, prob_0, prob_1,
                                               feature_probs_f0_given0,
                                               feature_probs_f1_given0,
                                               feature_probs_f0_given1,
                                               feature_probs_f1_given1)

    # Compare training predictions to actual
    print("\n")
    print("TEST PREDICTIONS")
    print(predictions)
    print("Testing Predictions (% Correct)")
    print(get_num_similarities(test_classes, predictions) / len(X_test) * 100)


def run_both_breast():
    """
    This function runs the Winnow-2 algorithm and the naive bayes algorithm
     on the breast cancer dataset and allows you to tune theta and alpha
    :param theta: the theta hyperparameter
    :param alpha: the alpha hyperparameter
    """

    print("\n")
    print("\n")
    print("BREAST DATASET")
    print("\n")
    print("\n")

    # Read in breast cancer data
    df_breast = pd.read_csv('./data/breast-cancer-wisconsin.data', header=None)

    # This data has nine attributes including the class attribute (which is Benign = 2, Malignant = 4)
    df_breast.columns = ["Sample Id", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
                         "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin",
                         "Normal Nucleoli", "Mitoses", "Class: Benign or Malignant"]

    # Find missing values and remove them, since there are so few
    # The documentation notes that there are16 missing values in group 1 and 6 denoted by '?'
    # I found 16 values in Group 6
    # Since there are so few missing values I dropped those rows
    df_breast_all = df_breast[df_breast["Bare Nuclei"] != '?']

    # Drop Sample Id
    df_breast_all = df_breast_all.drop('Sample Id', axis=1)

    # Generate boolean classifiers (0 = Benign, 1 = Malignant)
    df_breast_all.loc[df_breast_all["Class: Benign or Malignant"] == 2, "Class: Benign or Malignant"] = 0
    df_breast_all.loc[df_breast_all["Class: Benign or Malignant"] == 4, "Class: Benign or Malignant"] = 1

    # Split breast dataset
    X_breast_train, X_breast_test = split_test_train(df_breast_all)
    print("Sample size = ", len(df_breast_all))
    print("Training set size = ", len(X_breast_train))
    print("Test set size = ", len(X_breast_test))

    # Run Winnow-2 on breast
    theta = 38  # Chosen based on box plot of sums
    alpha = 1.6  # Started with 2 and tuned until I got the best result
    print("theta = " + str(theta))
    print("alpha = " + str(alpha))
    w = train_winnow2(X_breast_train.iloc[:, 0:9], X_breast_train.iloc[:, 9], alpha, theta)
    test_winnow2(X_breast_test.iloc[:, 0:9], X_breast_test.iloc[:, 9], alpha, theta, w)

    # One hot encode
    columns_to_encode = df_breast_all.columns.values.tolist()
    del columns_to_encode[9]
    df_breast_encoded = one_hot_encoder(df_breast_all, columns_to_encode)

    # Split breast dataset
    X_breast_train_encoded, X_breast_test_encoded = split_test_train(df_breast_encoded)

    naive_bayes(X_breast_train_encoded["Class: Benign or Malignant"],
                X_breast_test_encoded["Class: Benign or Malignant"],
                X_breast_train_encoded.iloc[:, 1:90],
                X_breast_test_encoded.iloc[:, 1:90])


def run_both_iris():
    """
    This function runs the Winnow-2 algorithm and the naive bayes algorithm
    on the iris dataset
    :return:
    """
    print("\n")
    print("\n")
    print("IRIS DATASET")
    print("\n")
    print("\n")
    # Read in Iris dataset and set column names
    df_iris = pd.read_csv('./data/iris.data', header=None)
    df_iris.columns = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "Class"]
    df_iris[df_iris["Class"] == "Iris-virginica"].head()

    # Generate boolean classifiers (0 = not-Iris-virginica, 1 = Iris-virginica)
    df_iris.loc[df_iris["Class"] == "Iris-virginica", "Class_Bool"] = 1
    df_iris.loc[df_iris["Class"] != "Iris-virginica", "Class_Bool"] = 0

    # Split iris dataset
    X_iris_train, X_iris_test = split_test_train(df_iris)
    print("Sample size = ", len(df_iris))
    print("Training set size = ", len(X_iris_train))
    print("Test set size = ", len(X_iris_test))
    X_iris_train.head()

    # Run Winnow-2 on Iris, I made this a two class problem by running
    # it for Iris-virginica (1) or not-Iris-virginica (0)
    # Given more time, I would write the algorithm to handle all three classes
    theta = 16  # Chosen based on box plot of sums
    alpha = 2  # Started with 2 and tuned until I got the best result
    print("theta = " + str(theta))
    print("alpha = " + str(alpha))
    w = train_winnow2(X_iris_train.iloc[:, 0:4], X_iris_train.iloc[:, 5], alpha, theta)
    test_winnow2(X_iris_test.iloc[:, 0:4], X_iris_test.iloc[:, 5], alpha, theta, w)

    # Run Naive Bayes on Iris
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
    df_iris_encoded.head()

    # Split iris dataset
    X_iris_train, X_iris_test = split_test_train(df_iris_encoded)
    print("Sample size = ", len(df_iris))
    print("Training set size = ", len(X_iris_train))
    print("Test set size = ", len(X_iris_test))
    X_iris_train.head()

    naive_bayes(X_iris_train.iloc[:, 5], X_iris_test.iloc[:, 5], X_iris_train.iloc[:, 6:], X_iris_test.iloc[:, 6:])


def run_both_votes():
    """This function runs the Winnow-2 algorithm and the naive bayes algorithm
    on the votes dataset
    """
    print("\n")
    print("\n")
    print("VOTES DATASET")
    print("\n")
    df_vote = pd.read_csv('./data/house-votes-84.data', header=None)
    df_vote.head()

    # Generate boolean classifiers (0 = republican, 1 = democrat)
    df_vote.loc[df_vote[0] == "republican", 0] = 0
    df_vote.loc[df_vote[0] == "democrat", 0] = 1

    # Generate boolean classifiers (0 = n, 1 = y)
    # Set ? to n, since there is no good way to impute votes
    df_vote.replace('n', 0, inplace=True)
    df_vote.replace('y', 1, inplace=True)
    df_vote.replace('?', 0, inplace=True)
    df_vote.head()

    # Split vote dataset
    X_vote_train, X_vote_test = split_test_train(df_vote)
    print("Sample size = ", len(df_vote))
    print("Training set size = ", len(X_vote_train))
    print("Test set size = ", len(X_vote_test))
    X_vote_train.head()

    # Run Winnow-2
    theta = 8.25  # Chosen based on box plot of sums
    alpha = 2  # Started with 2 and tuned until I got the best result
    print("theta = " + str(theta))
    print("alpha = " + str(alpha))
    w = train_winnow2(X_vote_train.iloc[:, 1:], X_vote_train.iloc[:, 0], alpha, theta)
    test_winnow2(X_vote_test.iloc[:, 1:], X_vote_test.iloc[:, 0], alpha, theta, w)

    # Run Naive Bayes
    naive_bayes(X_vote_train[0], X_vote_test[0], X_vote_train.iloc[:, 1:16], X_vote_test.iloc[:, 1:16])


if __name__== "__main__":
    run_both_breast()
    run_both_iris()
    run_both_votes()

