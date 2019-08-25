import math
import random

from machineLearningUtilities import get_num_similarities


class LogisticRegression:
    def __init__(self, features, x_train, x_train_classes, x_test, x_test_classes):
        self.features = features
        self.training_classes = x_train_classes
        self.train = x_train
        self.test = x_test
        self.test_classes = x_test_classes
        self.max_iterations = 50
        self.eta = 0.01  # learning rate
        self.weights = [0] * len(self.features)

    def learn(self):
        """
        This function runs the logistic regression algorithm on the training set provided
        """
        print("Running Logistic Regression...")
        for j, value in enumerate(self.features):
            self.weights[j] = random.uniform(-1.0, 1.0)

        print(f"Starting weights = {self.weights}")
        converged = False
        i = 0
        while not converged:
            weight_deltas = [0] * len(self.features)

            for x_index, x_values in self.train.iterrows():
                o = 0
                for j, value in enumerate(self.features):
                    o += self.weights[j] * float(x_values[j])

                y = self.sigmoid(o)

                for j, value in enumerate(self.features):
                    weight_deltas[j] += (self.training_classes[x_index] - y) * float(x_values[j])

            for j, value in enumerate(self.features):
                self.weights[j] += self.eta * weight_deltas[j]

            if i > self.max_iterations:
                converged = True

            i += 1

        print(f"Ending eights = {self.weights}")

    def validate(self):
        """
        This function determines accuracy of model using the test data set
        and applying the linear function using the weights
        """
        print("Testing...")
        predictions = []
        for x_index, x_values in self.test.iterrows():
            # Calculate linear value by adding up x values and their weights
            o = 0
            for j, value in enumerate(self.features):
                o += float(x_values[j]) * self.weights[j]

            y = self.sigmoid(o)

            if y > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return get_num_similarities(predictions, self.test_classes) / len(self.test_classes) * 100

    @staticmethod
    def sigmoid(x):
        """
        This function calculates the logistic sigmoid of x
        :param x: The value
        :return: The logistic sigmoid
        """
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return float('inf')
