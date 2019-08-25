import pandas as pd

from machineLearningUtilities import generate_random_numbers, get_euclidean_distance


class KMeansClustering:
    """
    This class implements the k-means clustering algorithm
    """
    def __init__(self, data, k):
        """
        This function initialize the k-means clustering class with data and value of k
        :param data: The dataset to cluster
        :param k: k = number of clusters to create
        """
        self.data = data
        self.k = k
        self.centers = []

    def run(self):
        """
        This function runs the k-means clustering algorithm
        :return: labels of clusters
        """
        print("k = " + str(self.k))
        self.data.head()
        # Initialize random centers
        center_indices = generate_random_numbers(len(self.data), self.k)
        self.centers = self.data.iloc[center_indices, :]
        self.centers.index = range(1, len(self.centers.index) + 1)

        while True:
            # Find cluster labels for each data point
            labels = self.get_labels()

            # Calculate new centers based on new clusters
            new_centers = pd.DataFrame([self.data.loc[[j for j, x in enumerate(labels) if x == i], :].mean(axis=0)
                                        for i in range(1, self.k + 1)])
            new_centers.index = range(1, len(new_centers.index) + 1)

            # If centers converged, break
            if self.centers.equals(new_centers):
                break

            # Otherwise, continue
            self.centers = new_centers

        return labels

    def get_labels(self):
        """
        This function labels the data for suggested cluster
        :return:
        """
        labels = []
        for x_index, x in self.data.iterrows():
            min_distance = float('inf')
            min_center = None
            for c_index, center in self.centers.iterrows():
                current_distance = get_euclidean_distance(x, center)
                if current_distance < min_distance:
                    min_center = c_index
                    min_distance = current_distance

            # Label is the closest center index
            labels.append(min_center)

        return labels









