from machineLearningUtilities import get_num_similarities


class NaiveBayesModel:
    def __init__(self, p0, p1, f0p0, f1p0, f0p1, f1p1):
        """
        This class holds the model for the Naive Bayes algorithm
        :param p0: The probability of class = 0 in the entire data set
        :param p1: The probability of class = 1 in the entire data set
        :param fp00: An array that holds probabilities of f = 0, given c = 0
        :param fp10: An array that holds probabilities of f = 1, given c = 0
        :param fp01: An array that holds probabilities of f = 0, given c = 1
        :param fp11: An array that holds probabilities of f = 1, given c = 1
        """
        self.prob_0 = p0
        self.prob_1 = p1
        self.feature_probs_f0_given0 = f0p0
        self.feature_probs_f1_given0 = f1p0
        self.feature_probs_f0_given1 = f0p1
        self.feature_probs_f1_given1 = f1p1


class NaiveBayes:
    def __init__(self, train, training_classes, test, test_classes):
        self.training_classes = training_classes
        self.test_classes = test_classes
        self.train = train
        self.test = test
        self.model = None
        self.predictions = []

    def learn(self):
        """
        This function creates the naive bayes model.
        """
        print("Training Naive Bayes...")
        # Create the model
        value_counts = self.training_classes.value_counts().to_dict()
        num_samples = len(self.training_classes)
        prob_0 = value_counts[0] / num_samples if 0 in value_counts else 0
        prob_1 = value_counts[1] / num_samples if 1 in value_counts else 1
        feature_probs_f0_given0 = []
        feature_probs_f1_given0 = []
        feature_probs_f0_given1 = []
        feature_probs_f1_given1 = []
        zero_class_indexes = [i for i, x in enumerate(self.training_classes) if x == 0]
        one_class_indexes = [i for i, x in enumerate(self.training_classes) if x == 1]
        num_zero_classes = len(zero_class_indexes)
        num_one_classes = len(one_class_indexes)
        for index, colName in enumerate(self.train.columns):
            f_value_counts_given_0 = self.train.iloc[zero_class_indexes, index].value_counts().to_dict()
            f_value_counts_given_1 = self.train.iloc[one_class_indexes, index].value_counts().to_dict()
            prob_f0_given_c0 = f_value_counts_given_0[0] / num_zero_classes if 0 in f_value_counts_given_0 else 0
            prob_f1_given_c0 = f_value_counts_given_0[1] / num_zero_classes if 1 in f_value_counts_given_0 else 0
            prob_f0_given_c1 = f_value_counts_given_1[0] / num_one_classes if 0 in f_value_counts_given_1 else 0
            prob_f1_given_c1 = f_value_counts_given_1[1] / num_one_classes if 1 in f_value_counts_given_1 else 0
            feature_probs_f0_given0.append(prob_f0_given_c0)
            feature_probs_f1_given0.append(prob_f1_given_c0)
            feature_probs_f0_given1.append(prob_f0_given_c1)
            feature_probs_f1_given1.append(prob_f1_given_c1)

        self.model = NaiveBayesModel(prob_0, prob_1, feature_probs_f0_given0,
                                     feature_probs_f1_given0, feature_probs_f0_given1,
                                     feature_probs_f1_given1)

    def validate(self):
        """
        This function makes predictions for the naive bayes algorithm
        """
        print("Testing Naive Bayes Accuracy...")
        if not self.model:
            print("Please call the function train first!")
            return

        # Calculate predictions by calculating the probability of each
        # C=1 and C=0 in each sample(row)
        for index, row in self.test.iterrows():
            c0_product = self.model.prob_0
            c1_product = self.model.prob_1
            for feature_index, feature in enumerate(row):
                if feature == 0:
                    c0_product *= self.model.feature_probs_f0_given0[feature_index]
                    c1_product *= self.model.feature_probs_f0_given1[feature_index]
                else:
                    c0_product *= self.model.feature_probs_f1_given0[feature_index]
                    c1_product *= self.model.feature_probs_f1_given1[feature_index]

            if c0_product > c1_product:
                self.predictions.append(0)
            else:
                self.predictions.append(1)

        return get_num_similarities(self.predictions, self.test_classes) / len(self.test_classes) * 100
