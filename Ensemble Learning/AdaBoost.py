import math
import typing
import pandas as pd
import numpy as np
from typing import Callable
from DecisionStump import DecisionStump


class AdaBoost:
  """
    Used to build and predict AdaBoost models.
    """

  def __init__(self, data: pd.DataFrame) -> None:
    self.__data = data
    self.__stumps: list[DecisionStump] = []
    self.__alpha: list[float] = []
    self.ENTROPY = DecisionStump.entropy
    self.GINNI_INDEX = DecisionStump.ginni_index
    self.MAJORITY_ERROR = DecisionStump.majority_error

  def build(self, attributes: set, label: typing.Any, splitting_criteria: Callable, num_trees: int,
            depth: int = 2) -> None:
    """
        Implements AdaBoost algorithm on the bases of the DecisionStump class.
        Note, A tree that will perfectly fit will give a zero division error
        :param attributes: Features
        :param label: What we are predicting
        :param splitting_criteria: What function should we use to split
        :param num_trees: The number of stumps we should use to make a prediction
        :param depth: The max depth of the stumps
        :return: void
        """
    m: int = self.__data.shape[0]  # number of examples
    weights: np.array = np.ones(shape=m) / m
    for i in range(num_trees):
      tree: DecisionStump = DecisionStump(self.__data)
      tree.build(attributes, label, splitting_criteria, weights, set_depth=depth)
      epsilon_t: float = 0
      for index in range(m):
        if tree.predict(self.__data, index) != (self.__data.iloc[index])[label]:
          epsilon_t += weights[index]

      alpha_t: float = 1 / 2 * math.log((1 - epsilon_t) / epsilon_t)
      self.__alpha.append(alpha_t)
      for index in range(m):
        if tree.predict(self.__data, index) == (self.__data.iloc[index])[label]:
          weights[index] = weights[index] * np.exp(-alpha_t * 1)
        else:
          weights[index] = weights[index] * np.exp(-alpha_t * -1)
      weights = weights / np.sum(weights)  # normalize to add weights to 1
      self.__stumps.append(tree)

  def predict(self, data: pd.DataFrame, row_num: int) -> typing.Any:
    """
        Predict using the largest alpha values of sum for each unique prediction
        :param data: data to make prediction from
        :param row_num: which observation to make a prediction of
        :return: prediction
        """
    predictions: dict[typing.Any, float] = {}
    for stump, alpha in zip(self.__stumps, self.__alpha):
      prediction: typing.Any = stump.predict(data, row_num)
      if prediction not in predictions:
        predictions[prediction] = alpha
      else:
        predictions[prediction] += alpha

    # Find the key with the maximum value in the dictionary
    return max(predictions, key=lambda k: predictions[k])
