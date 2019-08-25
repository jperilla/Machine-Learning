from machineLearningUtilities import get_num_similarities


class StepwiseForwardSelection:
    """
    This clas performs Stepwise Forward Selection for feature selection
    """
    def __init__(self, all_features, d_train, d_valid, c_train, c_test, f_learn, f_perf):
        """
        This function initialize the parameters for SFS
        :param all_features: full set of features
        :param d_train: training data
        :param d_valid: validation data
        :param c_train: training classes
        :param c_test: validation classes
        :param f_learn: The learning function
        :param f_perf: The performance monitoring function
        """
        self.features = all_features
        self.train = d_train
        self.valid = d_valid
        self.train_classes = c_train
        self.valid_classes = c_test
        self.learn = f_learn
        self.validate = f_perf
        self.base_performance = -float("inf")

    def run(self):
        """
        This function runs the SFS algorithm for feature selection
        :return: The set reduced set of features
        """
        print("Running StepWise Forward Selection...")
        features_0 = []
        while self.features:
            best_performance = -float("inf")
            for feature in self.features:
                features_0.append(feature)
                hypothesis = self.learn(self.train_classes, self.train[features_0])
                current_performance = self.test_performance(self.valid[features_0], *hypothesis)
                if current_performance > best_performance:
                    best_performance = current_performance
                    best_feature = feature
                features_0.remove(feature)

            if best_performance > self.base_performance:
                self.base_performance = best_performance
                self.features.remove(best_feature)
                features_0.append(best_feature)
            else:
                break

        print("Final best performance...")
        print(f"{self.base_performance * 100}%")
        print("Final feature set")
        print(features_0)
        return features_0

    def test_performance(self, data, *hypothesis):
        """
        This function measures performance using the validation function
        :param data: The data to test on
        :param hypothesis: The model to use
        :return: The performance as a percentage correct
        """
        predictions = self.validate(data, *hypothesis)
        return get_num_similarities(self.valid_classes, predictions) / len(self.valid_classes)



