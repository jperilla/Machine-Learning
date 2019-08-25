
def learn(classes, data):
    """
    This function creates the naive bayes model.
    :param classes: the set of actual class values
    :param data: the set of data
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
    for index, colName in enumerate(data.columns):
        f_value_counts_given_0 = data.iloc[zero_class_indexes, index].value_counts().to_dict()
        f_value_counts_given_1 = data.iloc[one_class_indexes, index].value_counts().to_dict()
        prob_f0_given_c0 = f_value_counts_given_0[0] / num_zero_classes if 0 in f_value_counts_given_0 else 0
        prob_f1_given_c0 = f_value_counts_given_0[1] / num_zero_classes if 1 in f_value_counts_given_0 else 0
        prob_f0_given_c1 = f_value_counts_given_1[0] / num_one_classes if 0 in f_value_counts_given_1 else 0
        prob_f1_given_c1 = f_value_counts_given_1[1] / num_one_classes if 1 in f_value_counts_given_1 else 0
        feature_probs_f0_given0.append(prob_f0_given_c0)
        feature_probs_f1_given0.append(prob_f1_given_c0)
        feature_probs_f0_given1.append(prob_f0_given_c1)
        feature_probs_f1_given1.append(prob_f1_given_c1)

    return (prob_0, prob_1, feature_probs_f0_given0,
            feature_probs_f1_given0, feature_probs_f0_given1,
            feature_probs_f1_given1)


def test(data, prob_0, prob_1, feature_probs_f0_given0,
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
