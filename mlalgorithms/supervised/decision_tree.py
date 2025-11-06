from mlalgorithms.model import Model
from mlalgorithms.utils import gini

import torch
from abc import abstractmethod
from typing import Optional

class TreeNode(Model):
    """ Node of a decision tree """


    def __init__(
        self, 
        feature_index: int, 
        criterion_value: float,
        dispersion: float,
        categorical: bool = False,
        decision: Optional[float] = None
    ):
        """
        feature_index   : Index of the feature used to split the data
        criterion_value : Value used to split the data with the feature at `feature_index`
        dispersion      : Measure of dispersion created by that node (gini, entropy, etc)
        categorical     : Whether the feature is categorical or not. In the case of a categorical
                          feature, the split will be based on if the value is == or != to the criterion
                          value. If the feature is not categorical, the split will be based on whether
                          the value is < or >= to the criterion value
        decision        : Decision on classification for that node. Only applicable if node is a leaf
        """
        self.feature_index = feature_index
        self.criterion_value = criterion_value
        self.dispersion = dispersion
        self.categorical = categorical
        self.decision = decision

        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_depth(self):
        """Returns the depth of the tree underneath the node"""
        
        if self.is_leaf():
            return 0

        return max(self.left.get_depth(), self.right.get_depth()) + 1

    def get_size(self):
        """Returns the number of nodes in the tree underneath the node"""
        if self.is_leaf():
            return 1

        return self.left.get_size() + self.right.get_size() + 1

    def split(self, X: torch.Tensor):
        """
        Splits a the data into its two children
        
        Returns at tuple containing all indices that must be sent to the left children
        and all indices that must be sent to the right children
        """
        if self.is_leaf():
            return ([], [])

        if self.categorical:
            left = torch.where(X[:, self.feature_index] == self.criterion_value)
            right = torch.where(X[:, self.feature_index] != self.criterion_value)
        else:
            left = torch.where(X[:, self.feature_index] < self.criterion_value)
            right = torch.where(X[:, self.feature_index] >= self.criterion_value)
        return left, right

    def predict(self, X: torch.Tensor) -> torch.Tensor:

        if self.is_leaf():
            return torch.ones(X.shape[0]) * self.decision

        left, right = self.split(X)
        predictions = torch.zeros(X.shape[0])
        predictions[left] = self.left.predict(X[left])
        predictions[right] = self.right.predict(X[right])
        return predictions

class DecisionTree(Model):
    # TODO : Improve efficiency
    # TODO : Add pruning
    # TODO : Add entropy and log-loss

    CRITERIA = ['gini']

    def __init__(
        self, 
        criterion: str = 'gini', 
        categorical_features = [],
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        eps: float = 1e-5
    ):
        super().__init__()

        self.criterion = criterion
        self._validate_criterion()

        self.categorical_features = categorical_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.eps = eps

        self.tree: Optional[TreeNode] = None

    @abstractmethod
    def _get_dispersion_score(self, left_group: torch.Tensor, right_group: torch.Tensor, Y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def _get_leaf_decision(Y: torch.Tensor):
        pass

    def _validate_criterion(self):
        assert self.criterion in DecisionTree.CRITERIA, f'The criterion must be one of {DecisionTree.CRITERIA}'

    def get_depth(self):
        return self.tree.get_depth()

    def _get_unique_values_of_tensor(self, X: torch.Tensor):
        """ Returns the indices of the unique values of a tensor"""
        if len(X) == 0:
            return []
        sorted_tensor, indices = torch.sort(X)
        return indices[torch.cat((torch.tensor([True]), sorted_tensor[1:] != sorted_tensor[:-1]))]

    def _get_best_split(self, X: torch.Tensor, Y: torch.Tensor):
        nb_samples, nb_features = X.shape

        best_score = float('inf')
        labels = X.unique()
        best_idx = (0, 0)  # Default to first feature and first value
        best_split = (torch.tensor([]), torch.tensor([]))  # Empty splits as default

        for feature in range(nb_features):
            unique_features = self._get_unique_values_of_tensor(X[:, feature])
            if len(unique_features) == 0:
                continue

            for unique_feature in unique_features:
                if feature in self.categorical_features:
                    left_group = torch.where(X[:, feature] == X[unique_feature, feature])[0]
                    right_group = torch.where(X[:, feature] != X[unique_feature, feature])[0]
                else:
                    left_group = torch.where(X[:, feature] < X[unique_feature, feature])[0]
                    right_group = torch.where(X[:, feature] >= X[unique_feature, feature])[0]

                if len(left_group) == 0 or len(right_group) == 0:
                    continue

                score = self._get_dispersion_score(left_group, right_group, Y, labels)
                if score < best_score:
                    best_score = score
                    best_idx = unique_feature, feature
                    best_split = (left_group, right_group)

        return best_idx, best_score, best_split

    def _build_node(self, X: torch.Tensor, Y: torch.Tensor, depth: int):
        nb_samples, nb_features = X.shape

        (unique_feature, feature), score, (left_split, right_split) = self._get_best_split(X, Y)

        current_node = TreeNode(
            feature_index=feature, 
            criterion_value=X[unique_feature, feature],
            dispersion=score,
            categorical=feature in self.categorical_features
        )

        max_depth_reached = self.max_depth is not None and self.max_depth <= depth + 1
        empty_splits = len(left_split) == 0 and len(right_split) == 0

        if max_depth_reached or empty_splits or self.min_samples_split >= nb_samples or score < self.eps:
            # Create a leaf node since we either reached the max depth, the minimum number of samples
            # that can be used to make a decision or the minimum dispersion score alled
            current_node.decision = self._get_leaf_decision(Y)
        else:
            current_node.left = self._build_node(X[left_split], Y[left_split], depth + 1)
            current_node.right = self._build_node(X[right_split], Y[right_split], depth + 1)

        return current_node

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        self.tree = self._build_node(X, Y, 0)

    def predict(self, X: torch.Tensor):
        return self.tree.predict(X).to(dtype=torch.int64)


class DecisionTreeClassifier(DecisionTree):

    def __init__(self, criterion = 'gini', categorical_features=[], max_depth = None, min_samples_split = 2, eps = 0.00001):
        super().__init__(criterion, categorical_features, max_depth, min_samples_split, eps)

    def _get_dispersion_score(self, left_group: torch.Tensor, right_group: torch.Tensor, Y: torch.Tensor, labels: torch.Tensor):
        return gini([Y[left_group], Y[right_group]], labels)

    def _get_leaf_decision(self, Y: torch.Tensor) -> float:
        return torch.bincount(Y).argmax().item() if len(Y) > 0 else 0 # Use the most frequent label as the decision

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        assert Y.dtype in [torch.int32, torch.int64], "Labels must be integers"
        return super().fit(X, Y)

class DecisionTreeRegressor(DecisionTree):

    def __init__(self, criterion = 'gini', categorical_features=[], max_depth = None, min_samples_split = 2, eps = 0.00001):
        super().__init__(criterion, categorical_features, max_depth, min_samples_split, eps)

    def _get_dispersion_score(self, left_group: torch.Tensor, right_group: torch.Tensor, Y: torch.Tensor, labels: torch.Tensor):
        nb_samples = len(left_group) + len(right_group)
        Y_left_group, Y_right_group = Y[left_group], Y[right_group]
        left_group_score = torch.square(Y_left_group - Y_left_group.mean()).sum() * len(Y_left_group) / nb_samples
        right_group_score = torch.square(Y_right_group - Y_right_group.mean()).sum() * len(Y_right_group) / nb_samples
        return left_group_score + right_group_score

    def _get_leaf_decision(self, Y: torch.Tensor) -> float:
        return Y.mean()

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        assert Y.dtype in [torch.float32, torch.float64], "Labels must be floats"
        return super().fit(X, Y)
