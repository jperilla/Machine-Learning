import math
from collections import deque

from machineLearningUtilities import get_num_similarities


class Node:
    def __init__(self, feature=None, value=None):
        self.feature = feature
        self.value = value
        self.label = None
        self.children = []
        self.next = None


class DecisionTree:
    """
    This class implements a decision tree algorithm
    """
    def __init__(self, train):
        self.train = train
        self.unique_labels = set(train["Class"])
        self.root = None

    def build_id3_tree(self):
        """
        This function builds the id3 decision tree from the root
        """
        features = [col for col in self.train.columns if col != "Class"]
        self.root = self.build_id3_branch(self.train, features)

    def build_id3_branch(self, subset, features, value=None):
        """
        This function builds a subset of the id3 decision tree.
        :param subset: The current subset
        :param features: The features left
        :param value: The value that led to this subset
        :return: the root of the current subset
        """

        # Return if only one class left
        class_labels_left = set(subset["Class"])
        if len(class_labels_left) == 1:
            leaf = Node(value=value)
            leaf.label = list(class_labels_left)[0]
            return leaf

        # Return if last feature
        if len(features) == 1:
            # print("Creating Leaf")
            leaf = Node(value=value)
            leaf.label = self.mode(subset, "Class")
            return leaf

        # Otherwise, create a new decision node
        decision = Node(value=value)

        # Calculate the entropy I(c1, c2, ... ck) of the subset
        entropy = self.get_entropy(subset, "Class")
        # print(f"entropy = {entropy}")
        if entropy == 0:
            return None

        # Find the feature with the most information gain
        largest_gain_ratio = 0
        feature_largest_gain_ratio = None
        # print(features)
        for feature in features:

            # Find the expected entropy for this feature
            expected_entropy = self.get_expected_entropy(subset, feature)
            # print(f"Expected Entropy for feature [{feature}] = {expected_entropy}")

            # Calculate the Gain Ratio for this feature
            information_value = self.get_information_value(subset, feature)
            gain_ratio = (entropy - expected_entropy) / information_value
            # print(f"Gain for feature {feature} = {gain_ratio}")
            if gain_ratio > largest_gain_ratio:
                largest_gain_ratio = gain_ratio
                feature_largest_gain_ratio = feature

        # print(f"Feature with largest gain = {feature_largest_gain_ratio} at {largest_gain_ratio}")

        # For this feature, create a child node for each feature value
        decision.feature = feature_largest_gain_ratio
        feature_value_counts = subset[feature_largest_gain_ratio].value_counts()
        features.remove(feature_largest_gain_ratio)

        # Sort feature values
        sorted_fvs = []
        for key in sorted(feature_value_counts.keys()):
            sorted_fvs.append(key)

        for fv in sorted_fvs:
            child_subset = subset.loc[subset[feature_largest_gain_ratio] == fv]
            child_tree = self.build_id3_branch(child_subset, features, value=fv)
            if child_tree:
                decision.children.append(child_tree)

        return decision

    def get_entropy(self, data, column):
        entropy = 0
        num_samples = len(data)
        label_counts = data[column].value_counts()
        for label in self.unique_labels:
            if label in label_counts:
                num_this_label = label_counts[label]
                class_prob = num_this_label / num_samples
                entropy += (-class_prob * math.log2(class_prob))

        return entropy

    def get_expected_entropy(self, data, column):
        expected_entropy = 0
        num_samples = len(data)
        feature_value_counts = data[column].value_counts()
        unique_feature_values = set(data[column])
        for fv in unique_feature_values:
            num_this_feature_value = feature_value_counts[fv]
            feature_in_sample_prob = num_this_feature_value / num_samples
            expected_entropy += (feature_in_sample_prob
                                 * self.get_entropy_partition(data, column, fv, num_this_feature_value))

        return expected_entropy

    def get_entropy_partition(self, data, column, fv, num_total):
        entropy_partition = 0
        class_values_in_fv = data.loc[data[column] == fv, ["Class"]]["Class"]
        label_counts_in_fv = class_values_in_fv.value_counts()
        for label in self.unique_labels:
            if label in label_counts_in_fv:
                num_this_label = label_counts_in_fv[label]
                class_in_fv_prob = num_this_label / num_total
                entropy_partition += (-class_in_fv_prob * math.log2(class_in_fv_prob))

        return entropy_partition

    @staticmethod
    def get_information_value(data, column):
        iv = 0
        num_samples = len(data)
        feature_value_counts = data[column].value_counts()
        unique_feature_values = set(data[column])
        for fv in unique_feature_values:
            num_this_feature_value = feature_value_counts[fv]
            feature_in_sample_prob = num_this_feature_value / num_samples
            iv += (-feature_in_sample_prob * math.log2(feature_in_sample_prob))

        return iv

    def print_tree(self):
        """
        This function prints the decision tree
        """
        print("Printing Decision Tree...")
        if self.root:
            nodes = deque()
            nodes.append(self.root)
            while len(nodes) > 0:
                node = nodes.popleft()
                if node:
                    print(f"Decision = {node.feature}")
                    if node.children:
                        print(node.children)
                        for child in node.children:
                            if child:
                                if child.value:
                                    print(f"({child.value})")

                                if child.label:
                                    print("---Leaf Node---")
                                    print(f"Class Label = {child.label}")
                                else:
                                    nodes.append(child)
        else:
            print("Decision Tree has not been built.")

    def validate_pruned_tree(self, data):
        pruned_tree = self.prune(self.root, data)
        return self.validate(pruned_tree, data)

    def prune(self, node, data):
        """
        This function prunes the tree and validates accuracy against the validation set
        :return: pruned tree accuracy
        """
        if len(node.children) == 0:
            return

        if len(data) == 0:
            return

        # Loop through children and flag for pruning
        for child in node.children:
            subset = data[data[node.feature] == child.value]
            self.prune(child, subset)

        # Test if pruning improves accuracy
        if self.can_prune(node, data):
            node.children = None
            node.label = self.mode(data, 'Class')
            print(f"Pruning node, replacing with leaf node, label = {node.label}")

        return node

    def can_prune(self, node, data):

        if len(data) == 0:
            return True

        correct = 0
        majority_class = self.mode(data, "Class")
        for index, row in data.iterrows():
            if row['Class'] == majority_class:
                correct += 1

        pruned_accuracy = (float(correct) / len(data) * 100)
        node_accuracy = self.validate(node, data)
        if pruned_accuracy >= node_accuracy:
            return True

        return False

    @staticmethod
    def validate(root, df_test):
        """
        This function makes predictions on a test set, based on the decision tree built
        :param root: The root of the current tree
        :param df_test: The test set
        :return: accuracy
        """
        if root:
            predictions = []
            for index, row in df_test.iterrows():
                # print(row)
                leaf_found = False
                node = root
                class_label = None
                while not leaf_found:
                    # Check if node is a leaf
                    if node.label:
                        leaf_found = True
                        class_label = node.label
                    elif node.children:
                        # Loop through current node's children
                        for child in node.children:
                            # print(f"Decision = {node.feature}")
                            # print(f"Child value = {child.value}")
                            if isinstance(child.value, str) and row[node.feature] == child.value:
                                node = child
                                break
                            elif row[node.feature] <= child.value:
                                node = child
                                break
                            elif child == node.children[-1]:  # This is the last child
                                node = child
                    else:
                        print("Something went wrong")
                        leaf_found = True

                # print(f"Prediction = {class_label}")
                predictions.append(class_label)

        # Compare predictions and actual class labels
        # print(f"Comparing {len(predictions)} predictions against {len(df_test['Class'])} class labels...")
        accuracy = get_num_similarities(df_test["Class"], predictions) / len(predictions) * 100
        return accuracy

    @staticmethod
    def mode(subset, column):
        label_value_counts = subset[column].value_counts()
        return label_value_counts.argmax()
