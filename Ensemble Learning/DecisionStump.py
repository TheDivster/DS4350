import math
import typing
import pandas as pd
import numpy as np
from typing import Callable
from DecisionTreePackage.Node import Node


class DecisionStump:
  def __init__(self, data: pd.DataFrame) -> None:
    self.__head = None
    self.__data = data

  def _total_information_gain(self, data: pd.DataFrame, attributes: set, label: typing.Any, criteria: Callable) -> str:
    '''
    The criteria is the function that we use to decide the "best" split. Each of its inputs should be a percent.
    '''
    gain_dict: dict[float, str] = {}  # for quick retrival of attribute based on max information gain
    gain_list: list[float] = []  # for quick max calculation
    if criteria != self.majority_error:
      proportions: list[float] = self.__fractional_proportions(data, label)  # TODO: Check this for changes
      s_criteria = criteria(*proportions)
    else:
      s_criteria = criteria(data, label)
    for attribute in attributes:
      gain = s_criteria  # shortcut that stands for the entropy (or other splitting criteria) of s so, we don't
      # have to recalculate s_criteria each iteration
      for values in data[attribute].unique():
        s_v: pd.DataFrame = data[data[attribute] == values]
        if criteria != self.majority_error:
          proportions: list[float] = self.__fractional_proportions(s_v, label)  # TODO: Check this for changes
          criteria_s_v = criteria(*proportions)
        else:
          criteria_s_v = criteria(s_v, label)
        gain -= (s_v.shape[0] / data.shape[0]) * criteria_s_v
      gain_dict[gain] = attribute
      gain_list.append(gain)
    return gain_dict[max(gain_list)]

  def __fractional_proportions(self, data: pd.DataFrame, label: str) -> list[float]:
    '''
    Calculate entropy based on every example being fractional with the weights being in a column named weights
    :param data: The section of data to calculate fractional proportions of
    :param label: The label for what we are predicting
    :return: A list of the proportions with all the proportions adding to one
    '''
    output: list = []
    attribute_values: list = data[label].unique()
    for attribute in attribute_values:
      subset: pd.DataFrame = data[data[label] == attribute]
      proportion: float = 0
      for row in range(subset.shape[0]):  # loop over rows
        proportion += (subset.iloc[row])['weights']  # add the weights
        # TODO: Check if we need to normalize proportions
      output.append(proportion)
    return output

  def _id3(self, data: pd.DataFrame, attributes: set, label: typing.Any, splitting_criteria: Callable, set_depth: int, depth: int = 0) -> Node:
    """
        Implements id3 algorithm
        Note: visualize is an experimental feature purely used for debugging
        Assumes set_depth >= 0, but does not enforce this.
        splitting criteria is the criteria used to define the "best" attribute to split on
        Currently, criteria only works with self.entropy, self.ginni_index, self.majority_error
    """
    # Base case
    if len(data[label].unique()) == 1:
      # Note: mode returns an array
      return Node(data[label].mode()[0])  # all columns of the label should have the same attribute
    if len(attributes) == 0 or depth >= set_depth:
      most_common_value_node: Node = Node(data[label].mode()[0])
      return most_common_value_node

    a: typing.Any = self._total_information_gain(data, attributes, label, splitting_criteria)
    root: Node = Node(a)
    for values in self.__data[a].unique():
      s_v: pd.DataFrame = data[data[a] == values]
      if s_v.shape[0] == 0:  # shape[0] gives the number of rows/observations
        leaf_node: Node = Node(data[label].mode()[0])  # can't do this if s_v shape is empty
        root.add_child(values, leaf_node)
      else:
        this_node: Node = self._id3(s_v, attributes - {a}, label, splitting_criteria, set_depth, depth + 1)
        root.add_child(values, this_node)
    return root

  def build(self, attributes: set, label: typing.Any, splitting_criteria: Callable, weights: np.array, set_depth: int = 2,
            depth: int = 0) -> None:
    '''
    Builds and stores the decision tree
    Make sure to set weights to 1 for normal case
    Weights are implicitly included in the data being used
    '''
    data = self.__data.copy()
    data["weights"] = pd.Series(weights)
    self.__head = self._id3(data, attributes, label, splitting_criteria, set_depth, depth)

  def ginni_index(self, *args: float) -> float:
    calculated_ginni: float = 1
    for arg in args:
      calculated_ginni -= arg ** 2
    return calculated_ginni

  def majority_error(self, data: pd.DataFrame, label: str) -> float:
    target_series: pd.Series = data[label]
    majority_label: typing.Any = target_series.mode()[0]
    value_count: pd.Series = target_series.value_counts()
    error: float = (target_series.shape[0] - target_series[target_series == majority_label].shape[0]) / sum(value_count)
    return error

  def predict(self, data: pd.DataFrame, row_num: int) -> typing.Any:
    '''
    Only works with multiple observations
    For one tree only
    '''
    current_node: Node = self.__head
    while current_node.has_children():
      # the value of the node determines what attribute to look down
      value_node: typing.Any = current_node.get_value()
      value_data_at_node: typing.Any = data.iloc[row_num][value_node]
      current_node = current_node.get_child_edge(value_data_at_node)
    return current_node.get_value()

  def entropy(self, *args: float) -> float:
    """
    Calculate and return entropy based on the proportion data observed
    """
    calculated_entropy: float = 0
    for arg in args:
      calculated_entropy -= arg * math.log2(arg)
    return calculated_entropy
