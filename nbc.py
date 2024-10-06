from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


class NaiveBayes:
    """
    This class implements the Naive Bayes classifier.

    Attributes:
        alpha (float): The smoothing parameter for Laplace smoothing. Defaults
            to 1.0.
        n_features (int): The number of features in the training data. Initialized
            to `None`.
        class_labels (np.ndarray): The unique class labels in the training data.
            Initialized to `None`.
        class_probabilities (dict): The prior probability of each class. Initialized
            to `None`.
        feature_probabilities (list): The conditional probability of each feature
            given each class. Initialized to `None`.
                
    """
    def __init__(self, alpha: float = 1.0) -> None:
        """
        The constructor for NaiveBayes class. Initializes the smoothing parameter
        to `alpha`.

        Parameters:
            alpha (float): The smoothing parameter for Laplace smoothing. Defaults
                to 1.0.

        Returns:
            None

        Examples:
            >>> nb = NaiveBayes()
            >>> nb.alpha
            1.0
            >>> nb.n_features
            >>> nb.class_labels
            >>> nb.class_probabilities
            >>> nb.feature_probabilities
        """
        # >>> YOUR CODE HERE >>>
        self.alpha = alpha
        self.n_features = None
        self.class_labels = None
        self.class_probabilities = None
        self.feature_probabilities = None
        # <<< END OF YOUR CODE <<<

    def compute_class_probabilities(self, y_train: np.ndarray) -> Dict[Any, float]: #计算先验概率P(y)
        """
        Computes the prior probability of each class.

        Parameters:
            y_train (np.ndarray): The class labels for the training data.

        Returns:
            class_probabilities (dict): The prior probability of each class.
                Each key is a class label and each value is the prior probability
                of that class in y_train.

        Examples:
            >>> nb = NaiveBayes(1)
            >>> y_train = np.array([1, 1, 2, 3, 3])
            >>> nb.compute_class_probabilities(y_train)
            {1: 0.4, 2: 0.2, 3: 0.4}
        """
        classes, counts = np.unique(y_train, return_counts=True)
        
        class_probabilities = {}
        # >>> YOUR CODE HERE >>>
        #计算每一个类的先验概率P(y)
        total_samples = len(y_train)
        class_probabilities = {int(cls): round(float(count) / total_samples, 10) for cls, count in zip(classes, counts)}
        # <<< END OF YOUR CODE <<<
        return class_probabilities
    
    def compute_feature_probabilities(self, X_j_train: np.ndarray, y_train: np.ndarray) -> Dict[Tuple[Any, Any], float]:
        """
        Computes the conditional probability of each feature given each class.

        Parameters:
            X_j_train (np.ndarray): A 1D array of the values of a given feature
                for all training examples.
            y_train (np.ndarray): The class labels for all training examples.

        Returns:
            feature_probabilities (dict): The conditional probability of each feature
                given each class. Each key is a tuple of the form (feature_value, class_label)
                and each value is the conditional probability of that feature value given
                that class. i.e:
                    feature_probabilities[(feature_value, class_label)] = P(X_j = feature_value | Y = class_label)

        Examples:
            >>> nb = NaiveBayes(1)
            >>> y = np.array([0, 0, 0, 0, 1, 1, 1])
            >>> x = np.array(["a", "a", "b", "b", "c", "b", "NA"])
            >>> actual = nb.compute_feature_probabilities(x, y)
            >>> {k: v for k, v in sorted(actual.items()) if k[1] == 0} # sorted for doctest, can be safely ignored
            {('NA', 0): 0.125, ('a', 0): 0.375, ('b', 0): 0.375, ('c', 0): 0.125}
            >>> {k: v for k, v in sorted(actual.items()) if k[1] == 1} # sorted for doctest, can be safely ignored
            {('NA', 1): 0.285714..., ('a', 1): 0.142857..., ('b', 1): 0.285714..., ('c', 1): 0.285714...}
        """
        feature_probabilities = {}

        class_labels = np.unique(list(y_train))
        unique_features = set(X_j_train)

        # If 'NA' is not in the unique features, add it
        if 'NA' not in unique_features:
            # >>> YOUR CODE HERE >>>
            unique_features.add('NA')
            # <<< END OF YOUR CODE <<<

        # For each class, compute the conditional probability of each feature value
        # given that class
        for c in class_labels:
            # >>> YOUR CODE HERE >>>
            #计算当前类的条件概率:P(X = feature_value|y = c) = count(X = feature_value|y = c)/count(y = c)
            #是在当前label为c的时候,特征值为feature_value的概率
            for c in class_labels: #遍历所有可能的label取值c
                class_indices = np.where(y_train == c)[0] #找到label = c的
                X_j_class = X_j_train[class_indices] #是label = c的样本的第j个feature的value
                #构建一个字典输出对应的特征值在label = c中出现的次数: 也就是count(X = feature_value|y = c)
                feature_counts = {feature: np.sum(X_j_class == feature) for feature in unique_features}
                #计算count(y = c)
                total_count_class = len(X_j_class)
                #对于每一个独特的特征都计算条件概率:P(X = feature_value|y = c), 并且采用拉普拉斯平滑
                for feature in unique_features:
                    feature_probabilities[(feature, c)] = (feature_counts[feature] + self.alpha) / (total_count_class + self.alpha * len(unique_features))
            # <<< END OF YOUR CODE <<<

        return feature_probabilities
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fits a Naive Bayes model to the training data.

        Parameters:
            X_train (np.ndarray): The training data.
            y_train (np.ndarray): The class labels for the training data.

        Returns:
            None

        Examples:
            >>> nb = NaiveBayes(1)
            >>> X = np.array([ \
                    ["A", "1", "XX"], \
                    ["B", "2", "YY"], \
                    ["B", "2", "NA"], \
                    ["A", "1", "XX"], \
                    ["B", "2", "XX"], \
                ])
            >>> y = np.array([1, 1, 1, 0, 0])
            >>> nb.fit(X, y)
            >>> nb.n_features
            3
            >>> nb.class_labels
            array([0, 1])
            >>> nb.class_probabilities
            {0: 0.428571..., 1: 0.571428...}
            >>> {k: v for k, v in sorted(nb.feature_probabilities[0].items()) if k[1] == 1} # sorted for doctest
            {('A', 1): 0.333333..., ('B', 1): 0.5, ('NA', 1): 0.166666...}
            >>> {k: v for k, v in sorted(nb.feature_probabilities[1].items()) if k[1] == 0} # sorted for doctest
            {('1', 0): 0.4, ('2', 0): 0.4, ('NA', 0): 0.2}
        """

        _, n_features = X_train.shape
        # Store the number of features
        self.n_features = n_features
        # Store the class labels
        self.class_labels = np.unique(list(y_train))
        # Compute the prior probability of each class

        # >>> YOUR CODE HERE >>>
        # You solution must satisfy the following requirements:
        # This dictionary will store the prior probability of each class
        # self.class_probabilities[class_label] = P(Y = class_label)
        self.class_probabilities = self.compute_class_probabilities(y_train) #计算每个类的先验概率
        # This list will store the conditional probability of each feature
        # given each class. It should have `n_features` dictionaries, one for
        # each feature. Each dictionary will have the following form:
        # self.feature_probabilities[i][(feature_value, class_label)] = P(X_i = feature_value | Y = class_label)
        self.feature_probabilities = [{} for _ in range(n_features)]
        #对于每个特征都计算出当前特征的条件概率
        for i in range(n_features):
            self.feature_probabilities[i] = self.compute_feature_probabilities(X_train[:, i], y_train)
        # <<< END OF YOUR CODE <<<


    def predict_probabilities(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts the probability of each class for each test example.

        Parameters:
            X_test (np.ndarray): The test data.

        Returns:
            np.ndarray: The predicted probabilities for each class for each test example.

        Examples:
            >>> nb = NaiveBayes(1)
            >>> X = np.array([ \
                    ["A", "1"], \
                    ["B", "NA"], \
                    ["A", "1"], \
                    ["B", "2"], \
                    ["A", "2"], \
                ])
            >>> X_test = np.array([ \
                    ["A", "1"], \
                    ["B", "NA"], \
                    ["C", "2"], \
                ])
            >>> y = np.array([1, 1, 0, 0, 0])
            >>> nb.fit(X, y)
            >>> nb.class_labels = np.array([1, 0])
            >>> nb.predict_probabilities(X_test)
            array([[0.418..., 0.581...],
                   [0.683..., 0.316...],
                   [0.264..., 0.735...]])
        """

        assert X_test.shape[1] == self.n_features, "Number of features in X_test must match number of features in X_train"
        assert self.class_probabilities is not None, "Model has not been fit yet"
        assert self.feature_probabilities is not None, "Model has not been fit yet"
        assert self.class_labels is not None, "Model has not been fit yet"
        assert self.n_features is not None, "Model has not been fit yet"

        # You need to compute a n x c matrix of probabilities, where n is the
        # number of test examples and c is the number of classes. You should
        # make sure that the order of columns in this matrix matches the order
        # of class labels in `self.class_labels`.

        # Create a matrix of zeros with the correct shape: 概率矩阵存储: [col:特征以及row:label]
        probabilities = np.zeros((X_test.shape[0], len(self.class_labels)))
        # For each test example, compute the probability of each class
        for i in range(X_test.shape[0]):
            # For each class, compute the probability of the test example
            for c, label in enumerate(self.class_labels):
                # Store the probability in the correct position in the matrix
                # >>> YOUR CODE HERE >>>
                #对于联合概率P(Y=c|X) = P(Y=c)*P(X|Y=c),我们可以采用log形式的加和来替代联成,这样子还能防止underflow
                #先从P(y)开始:
                prob = np.log(self.class_probabilities[label])
                # <<< END OF YOUR CODE <<<
                # For each feature, compute the probability of the test example: 然后再计算P(X|Y=c)
                for j in range(self.n_features):
                    # Store the probability in the correct position in the matrix
                    # If the feature value is not in the training data, use the
                    # probability of NA
                    if (X_test[i, j], label) not in self.feature_probabilities[j]:
                        # >>> YOUR CODE HERE >>>
                        #如果这个特征值没有在测试集中出现,就是用'NA'的条件概率:
                        prob += np.log(self.feature_probabilities[j][('NA', label)])
                        # <<< END OF YOUR CODE <<<
                    else:
                        # If the feature value is in the training data, use the
                        # probability of the feature value
                        # >>> YOUR CODE HERE >>>
                        #如果出现了的话,就用对应的条件概率,对于测试集的i sample j feature
                        prob += np.log(self.feature_probabilities[j][(X_test[i,j], label)])
                        # <<< END OF YOUR CODE <<<

        # Depending on your implementation, you may need to normalize the
        # probabilities. You may ignore this if you don't need to.
        # >>> YOUR CODE HERE >>>
        #我们将pro.exp复原:
        probabilities = np.exp(probabilities)
        #同时归一化,确保所有的概率都是小于等于1的
        rows_num = probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities / rows_num
        # <<< END OF YOUR CODE <<<

        return probabilities

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Predicts the class for each test example.

        Parameters:
            probabilities (np.ndarray): The predicted probabilities for each
                class for each test example.

        Returns:
            np.ndarray: The predicted class for each test example.

        Examples:
            >>> nb = NaiveBayes(1)
            >>> X = np.array([ \
                    ["A", "1"], \
                    ["B", "NA"], \
                    ["A", "1"], \
                    ["B", "2"], \
                    ["A", "2"], \
                ])
            >>> y = np.array([1, 1, 0, 0, 0])
            >>> nb.fit(X, y)
            >>> nb.class_labels = np.array([1, 0])
            >>> probs = np.array([ \
                    [0.41860465, 0.58139535], \
                    [0.6835443 , 0.3164557 ], \
                    [0.26470588, 0.73529412], \
                ])
            >>> nb.predict(probs)
            array([0, 1, 0])
        """

        assert self.class_labels is not None, "Model has not been fit yet"

        # >>> YOUR CODE HERE >>>
        #选取预测出来的最大概率label的index
        max_prob_indices = np.argmax(probabilities, axis=1)
        y_pred = self.class_labels[max_prob_indices]
        # <<< END OF YOUR CODE <<<
        return y_pred
    
    def evaluate(self, y_test: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
        """
        Evaluates the model on the test data. Computes the zero-one loss and
        squared loss.

        Parameters:
            y_test (np.ndarray): The true class labels for the test data.
            probabilities (np.ndarray): The predicted probabilities for each
                class for each test example.

        Returns:
            Tuple[float, float]: The zero-one loss and squared loss.

        Examples:
            >>> nb = NaiveBayes(1)
            >>> probs = np.array([ \
                    [0.41860465, 0.58139535], \
                    [0.6835443 , 0.3164557 ], \
                    [0.26470588, 0.73529412], \
                ])
            >>> y_test = np.array([1, 1, 0])
            >>> nb.class_labels = np.array([1, 0])
            >>> zero_one_loss, squared_loss = nb.evaluate(y_test, probs)
            >>> zero_one_loss
            0.333333...
            >>> squared_loss
            0.169411...
        """

        zero_one_loss = 0
        squared_loss = 0

        y_pred = self.predict(probabilities)

        # >>> YOUR CODE HERE >>>
        #0/1损失:
        zero_one_loss = np.mean(y_pred != y_test)
        #平方差损失:
        for i in range(len(y_test)):
            for j, label in enumerate(self.class_labels):
                true_label_indicator = 1 if y_test[i] == label else 0
                squared_loss += (probabilities[i, j] - true_label_indicator) ** 2
        # <<< END OF YOUR CODE <<<
        return zero_one_loss, squared_loss

if __name__ == "__main__":
    import doctest
    import os
    import warnings

    from utils import (assert_less_equal, load_hw2_pickle, print_green,
                       print_red)

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    # Suppress warnings
    warnings.filterwarnings('ignore')

    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green(f"\nDoctests passed!\n")

        X, y = load_hw2_pickle(os.path.join(os.path.dirname(__file__), "train.pkl"))
        X, y = np.array(X), np.array(y).reshape(-1)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42)
        
        nb = NaiveBayes()
        nb.fit(X_train, y_train)

        zero_one_loss, squared_loss = nb.evaluate(y_valid, nb.predict_probabilities(X_valid))
        
        assert_less_equal(zero_one_loss, 0.3, f"Actual zero-one loss: {zero_one_loss}\n")
        assert_less_equal(squared_loss, 0.25, f"Actual squared loss: {squared_loss}\n")

        
    else:
        print_red("\nDoctests failed!\n")