import numpy as np
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class Tree:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.Tree(dataset)

    def Tree(self, dataset, current_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        if num_samples >= self.min_samples_split and current_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["info_gain"] > 0:
                left_subtree = self.Tree(best_split["left_dataset"], current_depth + 1)
                right_subtree = self.Tree(best_split["right_dataset"], current_depth + 1)
                return Node(
                    best_split["feature_index"],
                    best_split["threshold"],
                    left_subtree,
                    right_subtree,
                    best_split["info_gain"]
                )
        leaf_value = self.calculate_node_value(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                left_dataset, right_dataset = self.split_dataset(dataset, feature_index, threshold)
                if len(left_dataset) > 0 and len(right_dataset) > 0:
                    y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    current_info_gain = self.information_gained(y, left_y, right_y, "gini")
                    if current_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
                        best_split["info_gain"] = current_info_gain
                        max_info_gain = current_info_gain
        if "info_gain" not in best_split:
            best_split["info_gain"] = 0
        return best_split

    def split_dataset(self, dataset, feature_index, threshold):
        left = np.array([row for row in dataset if row[feature_index] <= threshold])
        right = np.array([row for row in dataset if row[feature_index] > threshold])
        return left, right

    def information_gained(self, parent, left_child, right_child, mode="gini"):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_left * self.gini_index(left_child) + weight_right * self.gini_index(right_child))
        else:
            gain = self.entropy(parent) - (weight_left * self.entropy(left_child) + weight_right * self.entropy(right_child))
        return gain

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p = len(y[y == cls]) / len(y)
            entropy -= p * np.log2(p) if p > 0 else 0
        return entropy

    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 1
        for cls in class_labels:
            p = len(y[y == cls]) / len(y)
            gini -= p ** 2
        return gini

    def calculate_node_value(self, Y):
        Y = list(Y)
        return max(set(Y), key=Y.count)

    def print_tree(self, tree=None, indent=" "):
        if tree is None:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft -> " % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright -> " % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def predict(self, X):
        return [self.predict_sample(x, self.root) for x in X]

    def predict_sample(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature_index]
        if feature_value <= tree.threshold:
            return self.predict_sample(x, tree.left)
        else:
            return self.predict_sample(x, tree.right)

