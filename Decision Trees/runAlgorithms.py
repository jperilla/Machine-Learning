import sys
import statistics

import pandas as pd

from DecisionTree import DecisionTree
from machineLearningUtilities import split_into_random_stratified_groups


def run_id3_on_abalone(file, prune=False):
    """
    This function runs ID3 algorithm on the abalone data set, which is a classification set
    :param file: input file
    """
    print("_______________________________")
    print("Reading in Abalone data set...")
    df_abalone = pd.read_csv(file, header=None)
    df_abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole", "Shucked",
                          "Viscera", "Shell", "Class"]

    # This data set has no missing values, so we will skip that step

    # Encode Sex column
    df_abalone.loc[df_abalone["Sex"] == "M", "Sex"] = 0
    df_abalone.loc[df_abalone["Sex"] == "F", "Sex"] = 1
    df_abalone.loc[df_abalone["Sex"] == "I", "Sex"] = 2
    print(df_abalone.head())

    run_id3_decision_tree(df_abalone, prune)


def run_id3_on_car(file, prune=False):
    """
    This function runs ID3 algorithm on the car evaluation data set, which is a classification set
    :param file: input file
    """
    print("_______________________________")
    print("Reading in Car Evaluation data set...")
    df_car = pd.read_csv(file, header=None)
    df_car.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "Class"]

    # This data set has no missing values, so we will skip that step

    print(df_car.head())

    run_id3_decision_tree(df_car, prune)


def run_id3_on_image(file, prune=False):
    """
        This function runs the ID3 algorithm on the Image Segmentation data set, which is a classification set
        :param file: input file
        """
    print("_______________________________")
    print("Reading in Image Segmentation data set...")
    df_image = pd.read_csv(file, header=None)
    df_image.columns = ["Class", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
                        "16", "17", "18", "19"]
    print(df_image.head())

    run_id3_decision_tree(df_image, prune)


def run_id3_decision_tree(df, prune=False):
    # Split dataset 5-fold stratified
    print(f"Size of total dataset = {len(df)}")
    train1, train2, train3, train4, train5 = split_into_random_stratified_groups(df)
    datasets = [train1, train2, train3, train4, train5]
    scores = []
    pruned_scores = []
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

        # Build the decision tree from the training set
        id3 = DecisionTree(df_train)
        id3.build_id3_tree()
        #id3.print_tree()

        # Test the decision tree
        accuracy = id3.validate(id3.root, df_test)
        print('Percent accurate: ' + repr(accuracy) + '%')
        scores.append(accuracy)

        # If pruning is turned on, test pruned tree accuracy
        if prune:
            p_accuracy = id3.validate_pruned_tree(df_test)
            print('Pruned Tree Percent Accurate: ' + repr(p_accuracy) + '%')
            pruned_scores.append(p_accuracy)

    print("----------------------------")
    print(f"Averages over 5 experiments")
    print("----------------------------")
    print(f"ID3 Decision Tree Averages = {statistics.mean(scores)}%")
    if prune:
        print(f"Pruned ID3 Decision Tree Averages = {statistics.mean(pruned_scores)}%")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please add options for data set (--abalone --car --image --computer --forest --wine) "
              "and the full file path when running this script")
        print("Example: python runAlgorithms.py --classification  ./data/abalone.data")
        exit()

    prune = False
    if len(sys.argv) == 4:
        prune = True if sys.argv[3] == "-p" else False

    if sys.argv[1] == "--abalone":
        run_id3_on_abalone(sys.argv[2], prune=prune)
        exit()

    if sys.argv[1] == "--car":
        run_id3_on_car(sys.argv[2], prune=prune)
        exit()

    if sys.argv[1] == "--image":
        run_id3_on_image(sys.argv[2], prune=prune)
        exit()

    exit()

