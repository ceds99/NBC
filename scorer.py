import collections
from typing import Any, Dict, Sequence, Set, Tuple

import numpy as np
from numpy.typing import ArrayLike


class Scorer:
    """
    This class represents a scorer for a decision tree.

    Attributes:
        class_labels (ArrayLike): A list of the class labels.
        alpha (int): The alpha value for Laplace smoothing.

    """

    def __init__(self, type: str, class_labels: Sequence, alpha: int = 1) -> None:
        """
        The constructor for the Scorer class. Saves the class labels to
        `self.class_labels` and the alpha value to `self.alpha`.

        Parameters:
            type (str): The type of scorer to use. Either "information" or "gini".
            class_labels (Sequence): A list or set of unique class labels.
            alpha (int): The alpha value for Laplace smoothing.

        Returns:
            None

        Examples:
            >>> scorer = Scorer("information", ["A", "B"])
            >>> scorer.type
            'information'
            >>> sorted(scorer.class_labels)
            ['A', 'B']
            >>> scorer.alpha
            1
        """

        if type not in ["information", "gini"]:
            raise ValueError("type must be either 'information' or 'gini'")
        
        # >>> YOUR CODE HERE >>>
        self.type = type
        self.class_labels = sorted(list(class_labels))
        self.alpha = alpha
        # <<< END OF YOUR CODE <<<


    def compute_class_probabilities(self, labels: ArrayLike) -> Dict[Any, float]:
        """
        This function computes the class probabilities for a set of labels.

        Parameters:
            labels (ArrayLike): A list of labels.

        Returns:
            Dict[Any, float]: A dictionary mapping the class label to the
                probability of that class label.

        Examples:
            >>> scorer = Scorer("information", ["A", "B"])
            >>> scorer.compute_class_probabilities(["A", "A"])
            {'A': 0.75, 'B': 0.25}
            >>> scorer.compute_class_probabilities([])
            {'A': 0.5, 'B': 0.5}
            >>> scorer = Scorer("information", [1, 2])
            >>> scorer.compute_class_probabilities([1, 1, 2])
            {1: 0.6, 2: 0.4}
            
        """

        class_probabilities = {}
        # >>> YOUR CODE HERE >>>
        #使用dict存储每个label出现的次数
        class_counts = {label: 0 for label in self.class_labels}

        for label in labels:
            if label in class_counts:
                class_counts[label] += 1

        #对数量为0的label采用拉普拉斯平滑
        #拉普拉斯平滑分母:
        total_count = len(labels) + self.alpha * len(self.class_labels)

        class_probabilities = {
            label: (class_counts[label] + self.alpha) / total_count
            for label in self.class_labels
        }
        # <<< END OF YOUR CODE <<<


        return class_probabilities

    def score(self, labels: ArrayLike) -> float:
        """
        This function calculates the score for a set of labels.

        Parameters:
            labels (ArrayLike): A list of labels.

        Returns:
            float: The score for the set of labels.
        """

        if self.type == "information":
            return self.information_score(labels)
        elif self.type == "gini":
            return self.gini_score(labels)
        
        raise ValueError("type must be either 'information' or 'gini'")
    
    def gain(self, data: ArrayLike, labels: ArrayLike, split_attribute: int) -> float:
        """
        This function calculates the gain for a set of labels and a split
        attribute.

        Parameters:
            data (ArrayLike): A 2D array of data.
            labels (ArrayLike): A list of labels.
            split_attribute (int): The index of the attribute to split on.

        Returns:
            float: The gain for the set of labels and the split attribute.
        """

        if self.type == "information":
            return self.information_gain(data, labels, split_attribute)
        elif self.type == "gini":
            return self.gini_gain(data, labels, split_attribute)
        
        raise ValueError("type must be either 'information' or 'gini'")
    
    
    def subset_data(self, data: ArrayLike, labels: ArrayLike, split_attribute: int, split_value: Any) -> Tuple[ArrayLike, ArrayLike]:
        """
        This function subsets the data and labels based on the split attribute

        Parameters:
            data (ArrayLike): A 2D array of data.
            labels (ArrayLike): A 1D array of labels.
            split_attribute (int): The index of the attribute to split on.
            split_value (Any): The value of the attribute to split on.

        Returns:
            Tuple[ArrayLike, ArrayLike]: A tuple containing the subset of data
                and labels that have `split_value` for the attribute at index
                `split_attribute`.

        Examples:
            >>> X = np.array([                                 \
                    ['NA', 'no', 'sophomore',],                \
                    ['below average', 'yes', 'sophomore',],    \
                    ['above average', 'yes', 'junior',],       \
                    ['NA', 'no', 'senior',],                   \
                    ['above average', 'yes', 'senior',],       \
                    ['below average', 'yes', 'junior',],       \
                    ['above average', 'no', 'junior',],        \
                    ['below average', 'no', 'junior',],        \
                    ['above average', 'yes', 'sophomore',],    \
                    ['above average', 'no', 'senior',],        \
                    ['below average', 'yes', 'senior',],       \
                    ['above average', 'NA', 'junior',],        \
                    ['below average', 'no', 'senior',],        \
                    ['above average', 'no', 'sophomore',],     \
                ])
            >>> y = np.array(["A", "A", "B", "A", "B", "A", "B", \
                              "A", "A", "A", "B", "B", "A", "A"])
            >>> scorer = Scorer("information", set(y))
            >>> data_subset, labels_subset = scorer.subset_data(X, y, 2, "sophomore")
            >>> data_subset
            array([['NA', 'no', 'sophomore'],
                   ['below average', 'yes', 'sophomore'],
                   ['above average', 'yes', 'sophomore'],
                   ['above average', 'no', 'sophomore']]...)
            >>> labels_subset
            array(['A', 'A', 'A', 'A']...)
        """

        # >>> YOUR CODE HERE >>>
        #采用布尔掩码来筛选符合条件的数据
        mask = data[:, split_attribute] == split_value
        data_subset = np.array(data[mask], dtype = str)
        labels_subset = np.array(labels[mask], dtype = str)
        # <<< END OF YOUR CODE <<<


        return data_subset, labels_subset

    def split_on_best(self, data: ArrayLike, labels: ArrayLike, exclude: Set=set()) -> Tuple[int, Dict[Any, Tuple[ArrayLike, ArrayLike]]]:
        """
        This function finds the best attribute to split on and splits the data
        and labels based on that attribute.

        Parameters:
            data (ArrayLike): A 2D array of data.
            labels (ArrayLike): A 1D array of labels.
            exclude (Set): A set of attributes to exclude from consideration.

        Returns:
            Tuple[int, Dict[Any, Tuple[ArrayLike, ArrayLike]]]: A tuple
                containing the index of the best attribute to split on and a
                dictionary mapping each possible value of that attribute to a
                tuple containing the subset of data and labels that have that
                value for the attribute.

        """

        feature_count = data.shape[1]

        best_gain, best_feature = -float("inf"), None

        for feature in range(feature_count):
            if feature in exclude:
                continue
            gain = self.gain(data, labels, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        subsets = collections.defaultdict(lambda: (np.empty(0), np.empty(0)))

        unique_values = np.unique(data[:, best_feature])
        for value in unique_values:
            subsets[value] = self.subset_data(data, labels, best_feature, value)

        return best_feature, subsets

    def information_score(self, labels: ArrayLike) -> float:
        """
        This function calculates the information score for a set of labels.

        Parameters:
            labels (ArrayLike): A list of labels.

        Returns:
            float: The information score (entropy) for the set of labels.

        Examples:
            >>> scorer = Scorer("information", ["A", "B"])
            >>> y = np.array(["A", "A", "B", "A", "B", "A", "B", \
                              "A", "A", "A", "B", "B", "A", "A"])
            >>> scorer.information_score(y)
            0.9544340...
        """

        class_probabilities = self.compute_class_probabilities(labels)
        
        entropy = 0
        # >>> YOUR CODE HERE >>>
        #计算信息熵
        for prob in  class_probabilities.values():
            if prob > 0:
                entropy -= prob*np.log2(prob)
        # <<< END OF YOUR CODE <<<

        return float(entropy)
    
    def gini_score(self, labels: ArrayLike) -> float:
        """
        This function calculates the gini score for a set of labels.

        Parameters:
            labels (ArrayLike): A list of labels.

        Returns:
            float: The gini score for the set of labels.

        Examples:
            >>> scorer = Scorer("gini", ["A", "B"])
            >>> y = np.array(["A", "A", "B", "A", "B", "A", "B", \
                              "A", "A", "A", "B", "B", "A", "A"])
            >>> scorer.gini_score(y)
            0.46875
        """

        class_probabilities = self.compute_class_probabilities(labels)
        
        gini = 0
        # >>> YOUR CODE HERE >>>
        gini = 1
        for prob in class_probabilities.values():
            gini -= prob**2
        # <<< END OF YOUR CODE <<<

        return gini
    
    def information_gain(self, data: ArrayLike, labels: ArrayLike, split_attribute: int) -> float:
        """
        This function calculates the information gain for a split on a given attribute.

        Parameters:
            data (ArrayLike): A 2D array of examples (row) and attributes (column).
            labels (ArrayLike): A 1D array of labels.
            split_attribute (int): The attribute to split on.

        Returns:
            float: The information gain for the split on the given attribute.

        Examples:
            >>> X = np.array([                                 \
                    ['NA', 'no', 'sophomore',],                \
                    ['below average', 'yes', 'sophomore',],    \
                    ['above average', 'yes', 'junior',],       \
                    ['NA', 'no', 'senior',],                   \
                    ['above average', 'yes', 'senior',],       \
                    ['below average', 'yes', 'junior',],       \
                    ['above average', 'no', 'junior',],        \
                    ['below average', 'no', 'junior',],        \
                    ['above average', 'yes', 'sophomore',],    \
                    ['above average', 'no', 'senior',],        \
                    ['below average', 'yes', 'senior',],       \
                    ['above average', 'NA', 'junior',],        \
                    ['below average', 'no', 'senior',],        \
                    ['above average', 'no', 'sophomore',],     \
                ])
            >>> y = np.array(["A", "A", "B", "A", "B", "A", "B", \
                              "A", "A", "A", "B", "B", "A", "A"])
            >>> scorer = Scorer("information", set(y))
            >>> [scorer.information_gain(X, y, i) for i in range(X.shape[1])]
            [0.03474..., 0.07816..., 0.06497...]
        """

        extropy_before = self.information_score(labels)
        extropy_after = 0
        # >>> YOUR CODE HERE >>>
        #计算节点分裂以后的熵
        values, counts = np.unique(data[:, split_attribute], return_counts=True)
        total_samples = len(labels)

        #当选取value作为分类特征的时候data和label的subset
        for value, count in zip(values, counts):
            data_subset, labels_subset = self.subset_data(data, labels, split_attribute, value)

            #计算每一个subset的entropy
            subset_entropy = self.information_score(labels_subset)
            weight = count / total_samples
            extropy_after += weight * subset_entropy


        information_gain = extropy_before - extropy_after
        #information_gain = [float(gain) for gain in information_gain]
        # <<< END OF YOUR CODE <<<

        return float(information_gain)

    def gini_gain(self, data: ArrayLike, labels: ArrayLike, split_attribute: int) -> float:
        """
        This function calculates the gini gain for a split on a given attribute.

        Parameters:
            data (ArrayLike): A 2D array of examples (row) and attributes (column).
            labels (ArrayLike): A 1D array of labels.
            split_attribute (int): The attribute to split on.

        Returns:
            float: The gini gain for the split on the given attribute.

        Examples:
            >>> X = np.array([                                 \
                    ['NA', 'no', 'sophomore',],                \
                    ['below average', 'yes', 'sophomore',],    \
                    ['above average', 'yes', 'junior',],       \
                    ['NA', 'no', 'senior',],                   \
                    ['above average', 'yes', 'senior',],       \
                    ['below average', 'yes', 'junior',],       \
                    ['above average', 'no', 'junior',],        \
                    ['below average', 'no', 'junior',],        \
                    ['above average', 'yes', 'sophomore',],    \
                    ['above average', 'no', 'senior',],        \
                    ['below average', 'yes', 'senior',],       \
                    ['above average', 'NA', 'junior',],        \
                    ['below average', 'no', 'senior',],        \
                    ['above average', 'no', 'sophomore',],     \
                ])
            >>> y = np.array(["A", "A", "B", "A", "B", "A", "B", \
                              "A", "A", "A", "B", "B", "A", "A"])
            >>> scorer = Scorer("gini", set(y))
            >>> [scorer.gini_gain(X, y, i) for i in range(X.shape[1])]
            [0.02249..., 0.04987..., 0.03953...]
        """

        gini_before = self.score(labels)
        gini_after = 0
        # >>> YOUR CODE HERE >>>
        values, counts = np.unique(data[:, split_attribute], return_counts=True)
        total_samples = len(labels)

        for value, count in zip(values, counts):
            data_subset, labels_subset = self.subset_data(data, labels, split_attribute, value)

            subset_gini = self.gini_score(labels_subset)
            weight = count / total_samples
            gini_after += weight * subset_gini

        gini_gain = gini_before - gini_after
        #gini_gain = [float(gain) for gain in gini_gain]
        # <<< END OF YOUR CODE <<<
        return float(gini_gain)

    def __repr__(self) -> str:
        return self.type

if __name__ == "__main__":
    import doctest
    import os

    from utils import (decision_tree_zero_one_loss, print_green, print_red,
                       read_hw1_data)

    # # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    # # Run the doctests. If all tests pass, print "All tests passed!"
    # # You may ignore PYDEV DEBUGGER WARNINGS that appear in the console.
    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green("\nAll tests passed!\n")
    else:
        print_red("\nSome tests failed!\n")

    # X, y = read_hw1_data(os.path.join(os.path.dirname(__file__), "yelp.csv"))

    # scorer = Scorer("information", set(y), 0)

    # model, loss, _ = decision_tree_zero_one_loss(X, y, X, y, scorer, max_depth=1)
    # print(f"Model: {model}, 0-1 Loss: {loss}")

