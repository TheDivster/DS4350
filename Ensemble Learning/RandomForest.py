import sys
import math
import typing
import numpy as np
import pandas as pd
from typing import Callable
from DecisionTreePackage.DecisionTree import DecisionTree


class RandomForest:
    def __init__(self, data: pd.DataFrame):
        self.__data: pd.DataFrame = data
        self.__trees: list[DecisionTree] = []
        self.ENTROPY: typing.Callable = DecisionTree.entropy
        self.GINNI_INDEX: typing.Callable = DecisionTree.ginni_index
        self.MAJORITY_ERROR: typing.Callable = DecisionTree.majority_error

    def build(self, num_trees: int, num_samples: int, attributes: set, label: typing.Any, splitting_criteria: Callable,
              set_depth: int = sys.maxsize) -> None:
        """
        Learn many trees with bootstrapped sample
        I am defining a smaller attribute subset as one that is approximately half the size
        Note: There will be problems if we try to predict outcomes for labels we don't have if we don't set num_samples to hign enough
        :param num_trees: number of trees to use to vote
        :param num_samples: number of samples to bootstrap
        :param attributes: set of attributes
        :param label: label to predict
        :param splitting_criteria: Function to use to decide best feature to split on
        :param set_depth: max depth of the tree
        :return: Void
        """
        for i in range(num_trees):
            sample: pd.DataFrame = self.__data.sample(num_samples, replace=True)
            attributes_sub_set: set = set(np.random.choice(list(attributes), math.ceil(len(attributes)/4)))
            tree = DecisionTree(sample)
            tree.build(attributes_sub_set, label, splitting_criteria, set_depth)
            self.__trees.append(tree)

    def predict(self, data: pd.DataFrame, row_num: int) -> typing.Any:
        """
        Predict using the statistical mode of the predictions
        :param data: data to make prediction from
        :param row_num: which observation to make a prediction of
        :return: prediction
         """
        predictions: list[typing.Any] = []
        for tree in self.__trees:
            predictions.append(tree.predict(data, row_num))

        # Find the most common value
        return pd.Series(predictions).mode()[0]