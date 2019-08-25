import operator
import statistics

import pandas as pd
from machineLearningUtilities import euclidean_distance, get_percent_similarities, get_mean_squared_error


class KNearestNeighbors:
    """
    This class implements the k nearest neighbors algorihtm
    """
    def __init__(self, test, k, columns, classification=True):
        """
        This function initialize the k nearest neighbors class with data and value of k
        :param test: The training dataset
        :param test: The test dataset
        :param k: k = number of clusters to create
        """
        self.test = test
        self.k = k
        self.columns = columns
        self.classification = classification
        self.predictions = []

    def run(self, train):
        """
        This function runs the k nearest neighbors algorithm
        """
        for i, x in self.test.iterrows():
            neighbors = self.get_k_nearest_neighbors(train, x)
            result = self.get_prediction(neighbors)
            self.predictions.append(result)
            # print('> predicted=' + repr(result) + ', actual=' + repr(x["Class"]))

        if self.classification:
            accuracy = get_percent_similarities(self.test["Class"], self.predictions)
        else:
            accuracy = get_mean_squared_error(self.test["Class"], self.predictions)

        return accuracy

    def run_condensed(self, train):
        """
        This function runs the condensed k nearest neighbors algorithm
        """
        print("Running Condensed k-Nearest Neighbors...")
        Z = pd.DataFrame(columns=self.columns)
        # Loop through data adding only misclassified data
        # Sample data to shuffle it, selecting x at random
        for i, x in train.sample(frac=1).iterrows():
            additions = False
            if len(Z) <= self.k:
                Z = Z.append(x)
                additions = True
            else:
                z_closest = self.get_closest_point(x, Z)
                if z_closest["Class"] != x["Class"]:
                    Z = Z.append(z_closest)
                    additions = True

            if not additions:
                break

        return self.run(Z)

    def get_k_nearest_neighbors(self, train, test_instance):
        """ This function finds the k-nearest neighbors of the test instance in the training set
        :param: test_instance
        :returns: The k-nearest neighbors
        """
        # Find distances from each
        distances = []
        for i, x in train.iterrows():
            distance = euclidean_distance(test_instance.drop(labels=['Class']), x.drop(labels=['Class']))
            distances.append((x, distance))

        # Sort distances ascending
        distances.sort(key=operator.itemgetter(1))

        # Find neighbors by taking k-nearest from the top
        neighbors = []
        for x in range(self.k):
            neighbors.append(distances[x][0])

        return neighbors

    def get_prediction(self, neighbors):
        """ This function gets the class based on votes for class in neighbors """
        if self.classification:
            class_votes = {}
            for x in range(len(neighbors)):
                vote = neighbors[x]["Class"]
                if vote in class_votes:
                    class_votes[vote] += 1
                else:
                    class_votes[vote] = 1

            sorted_votes = sorted(class_votes, key=operator.itemgetter(1), reverse=True)
            return sorted_votes[0]
        else:
            return statistics.mean(x["Class"] for x in neighbors)

    @staticmethod
    def get_closest_point(x, Z):
        min_distance = float('inf')
        closest = None
        for i, z in Z.iterrows():
            distance = euclidean_distance(x.drop(labels=['Class']), z.drop(labels=['Class']))
            if distance < min_distance:
                min_distance = distance
                closest = z

        return closest












