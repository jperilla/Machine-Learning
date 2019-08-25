import math
import random
import pandas as pd
import numpy as np


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
    train = df.sample(frac=2/3)
    test = df.loc[~df.index.isin(train.index), :]
    return train, test


def split_into_random_groups(df, k):
    """
    This function splits a data set into k random groups
    :param df: data to split
    :param k: the number of groups
    :return: array of groups
    """
    groups = []
    remaining = df.copy()
    if k > 0:
        for i in range(1, k):
            group = remaining.sample(frac=1/(k-(i-1)))
            remaining = remaining.loc[~remaining.index.isin(group.index), :]
            groups.append(group)

        groups.append(remaining)

    return groups


def split_into_random_stratified_groups(df):
    """
    This function splits the data frame into 5 stratified groups for 5-fold cross validation
    :param df: The dataframe
    :param test: The test set
    :param train1: Training set 1
    :param train2: Training set 2
    :param train3: Training set 3
    :param train4: Training set 4
    :return:
    """
    train1 = pd.DataFrame(columns=df.columns.values.tolist())
    train2 = pd.DataFrame(columns=df.columns.values.tolist())
    train3 = pd.DataFrame(columns=df.columns.values.tolist())
    train4 = pd.DataFrame(columns=df.columns.values.tolist())
    test = pd.DataFrame(columns=df.columns.values.tolist())

    unique_classes = df.Class.unique()
    for className in unique_classes:
        groups = split_into_random_groups(df.loc[df.Class == className], 5)
        train1 = train1.append(groups[0])
        train2 = train2.append(groups[1])
        train3 = train3.append(groups[2])
        train4 = train4.append(groups[3])
        test = test.append(groups[4])

    return train1, train2, train3, train4, test


def generate_random_numbers(r, n):
    """
    This function generates random integers in range(1:r)
    :param r: the range to generate random integers in
    :param n: the number of integers to generate
    :return: array of random numbers
    """
    randoms = []
    for x in range(1, n+1):
        randoms.append(random.randint(1, r))

    return randoms


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


def get_mean_squared_error(array1, array2):
    """
    This function calculates the mean squared error between the two arrays
    :param array1: first array to compare
    :param array2: second array to compare
    :return: the mean squared error
    """
    return np.mean((array1 - array2)**2 / 100)


def get_percent_similarities(array1, array2):
    """
    This function compares two arrays for similiaries and returns number of equal elements
    :param array1: first array to compare
    :param array2: second array to compare
    :return: the number of similar elements
    """
    a = np.array(array1)
    b = np.array(array2)
    return np.sum(a == b) / len(a) * 100


def euclidean_distance(array1, array2):
    """
    This function computes the euclidean distance between two arrays
    :param array1: first array
    :param array2: second array
    :return: distance
    """
    length = len(array1)
    distance = 0
    for x in range(length):
        distance += pow((array1[x] - array2[x]), 2)
    return math.sqrt(distance)


def calculate_silhouette_coefficient(X, labels):
    """
    This function calculates the silhouette coefficient of a data set, and it's labels for clustering
    :param X: The dataset
    :param labels: cluster labels
    :return: silhouette coefficient
    """
    return np.mean(get_silhouettes(X, labels))


def get_mean_intra_cluster_distances(x, index, X, labels):
    """ 
    This function gets the mean of all of the distances to each sample in the 
    cluster from x
    :param x: The selected sample
    :param index: the index of the selected sample
    :param X: the sample dataset
    :param labels: the set of labels for data points (which cluster it's in)
    """
    distances = []
    x_cluster = labels[index]
    rows_in_cluster = [i for i, value in enumerate(labels) if value == x_cluster]
    for i, sample in X.ix[rows_in_cluster].iterrows():
        # Find the distance to x
        d = get_euclidean_distance(x.values, sample.values)
        distances.append(d)
        
    return np.mean(distances)


def get_nearest_cluster_distance(x, index, X, labels):
    """
    This function gets the distance to the cluster nearest to
    x, that is not the cluster that x is in.
    :param x: The data point to test distances from
    :param index: The index of the datapoint
    :param X: The data set
    :param labels: The cluster labels of each data point
    :return:
    """
    minimum_distance = float('inf')
    clusters = set(labels)
    x_cluster = labels[index]
    for c in clusters:
        if c != x_cluster:
            distances = []
            rows_in_cluster = [i for i, value in enumerate(labels) if value == c]
            for i, sample in X.ix[rows_in_cluster].iterrows():
                distance = get_euclidean_distance(sample.values, x.values)
                distances.append(distance)

            mean = np.mean(distances)
            if mean < minimum_distance:
                minimum_distance = mean

    return minimum_distance


def get_silhouettes(X, labels):
    """ This function calculates the silhouette coefficient for each sample
    :param X: The dataset
    :param labels: cluster labels
    :return: silhouette coefficient
    """
    silhouettes = []
    for index, x in X.iterrows():
        A = get_mean_intra_cluster_distances(x, index, X, labels)
        B = get_nearest_cluster_distance(x, index, X, labels)
        silhouette = (B - A) / np.maximum(A, B)
        silhouettes.append(np.nan_to_num(silhouette))

    return silhouettes



