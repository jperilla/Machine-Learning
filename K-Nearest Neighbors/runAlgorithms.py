import sys
import statistics

import pandas as pd

from kNearestNeighbors import KNearestNeighbors
from machineLearningUtilities import split_into_random_stratified_groups, one_hot_encoder


def run_on_ecoli(file, k):
    """
    This function runs k-nearest neighbors on the Ecoli data set
    :param file: input file
    """
    print("_______________________________")
    print("Reading in Ecoli data set...")
    df_ecoli = pd.read_csv(file, header=None, sep='\s+')

    df_ecoli.columns = ["Id", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "Class"]
    # This data set has no missing values, so we will skip that step

    # Drop Id
    df_ecoli = df_ecoli.drop("Id", axis=1)

    run_k_nearest_neighbor_experiments(df_ecoli, k, True, classification=True)


def run_on_image(file, k):
    """
    This function runs k-nearest neighbors on the Image Segmentation data set
    :param file: input file
    """
    print("_______________________________")
    print("Reading in Image Segmentation data set...")
    df_image = pd.read_csv(file, header=None)
    print(df_image.head())

    df_image.columns = ["Class", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
                        "16", "17", "18", "19"]

    run_k_nearest_neighbor_experiments(df_image, k, True, classification=True)


def run_on_computer(file, k):
    """
    This function runs k-nearest neighbors on the Computer Hardware data set
    :param file: input file
    """
    print("_______________________________")
    print("Reading in Computer Hardware data set...")
    df_computer = pd.read_csv(file, header=None)
    print(df_computer.head())

    df_computer.columns = ["0", "1", "2", "3", "4", "5", "6", "7", "Class", "9"]

    # One hot encode categorical values
    df_computer = one_hot_encoder(df_computer, ["0", "1"])
    print(df_computer.head())

    run_k_nearest_neighbor_experiments(df_computer, k, False, classification=False)


def run_on_forest(file, k):
    """
    This function runs k-nearest neighbors on the Forest Fires data set
    :param file: input file
    """
    print("_______________________________")
    print("Reading in Forest Fires data set...")
    df_forest = pd.read_csv(file, header=None)
    print(df_forest.head())

    df_forest.columns = []

    run_k_nearest_neighbor_experiments(df_forest, k, False, classification=False)


def run_k_nearest_neighbor_experiments(df, k, run_condensed, classification=True):
    # Split dataset 5-fold stratified
    print(f"Size of total dataset = {len(df)}")
    train1, train2, train3, train4, train5 = split_into_random_stratified_groups(df)
    # Run five experiments, using one of the sets as a test set each time
    k_scores = []
    k_condensed_scores = []
    datasets = [train1, train2, train3, train4, train5]
    for i, d in enumerate(datasets):
        print("-------------")
        print(f"Experiment #{i + 1}")
        print("-------------")
        df_test = datasets[i]
        print(len(df_test))
        training_sets = datasets.copy()
        del training_sets[i]
        df_train = pd.concat(training_sets)
        print(len(df_train))

        # Run K-Nearest Neighbors
        print(f"k = {k}")
        print("Running k nearest neighbors...")
        knn = KNearestNeighbors(df_test, k, df.columns, classification)
        accuracy = knn.run(df_train)
        print('Percent accurate: ' + repr(accuracy) + '%')
        k_scores.append(accuracy)

        if run_condensed:
            # Run Condensed K-Nearest Neighbors
            knn = KNearestNeighbors(df_test, k, df.columns, classification)
            accuracy = knn.run_condensed(df_train)
            print('Percent accurate: ' + repr(accuracy) + '%')
            k_condensed_scores.append(accuracy)

    print("----------------------------------------")
    print(f"Averages over 5 experiments where k={k}")
    print("----------------------------------------")
    print(f"k-Nearest Neighbors = {statistics.mean(k_scores)}")
    if run_condensed:
        print(f"Condensed k-Nearest Neighbors = {statistics.mean(k_condensed_scores)}")


if __name__== "__main__":
    if len(sys.argv) < 4:
        print("Please add options for data set (--ecoli, --image, --computer or --forest) "
              "and value of k then file path when running this script")
        print("Example: python runAlgorithms --ecoli 3 ./data/iris.data")
        exit()

    if sys.argv[1] == "--ecoli":
        run_on_ecoli(sys.argv[3], int(sys.argv[2]))
        exit()

    if sys.argv[1] == "--image":
        run_on_image(sys.argv[3], int(sys.argv[2]))
        exit()

    if sys.argv[1] == "--computer":
        run_on_computer(sys.argv[3], int(sys.argv[2]))
        exit()

    if sys.argv[1] == "--forest":
        run_on_forest(sys.argv[3], int(sys.argv[2]))
        exit()

    exit()

