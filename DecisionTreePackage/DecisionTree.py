import math
import typing
import graphviz
import pandas as pd
from Node import Node
from typing import Callable

"""
Builds and represents a decision tree
"""
class DecisionTree:
    def __init__(self, data: pd.DataFrame) -> None:
        self.__head = None
        self.__data = data
        self._dot = graphviz.Digraph('tree')

    """
    implements id3 algorithm
    Note: visualize is an experimental feature purely used for debugging
    Assumes set_depth >= 0, but does not enforce this.
    splitting criteria is the criteria used to define the "best" attribute to split on
    Currently, criteria only works with self.entropy, self.ginni_index, self.majority_error
    """
    def _id3(self, data: pd.DataFrame, attributes: set, label: typing.Any, splitting_criteria: Callable, set_depth: int, depth: int = 0, visualize: bool = False) -> Node:
        # Base case
        if len(data[label].unique()) == 1:
            # Note: mode returns an array
            return Node(data[label].mode()[0])  # all columns of the label should have the same attribute
        if len(attributes) == 0 or depth >= set_depth:
            most_common_value_node: Node = Node(data[label].mode()[0])
            if set_depth == 0:
                self._dot.node(str(most_common_value_node.get_value()))
            return most_common_value_node

        a: typing.Any = self._total_information_gain(data, attributes, label, splitting_criteria)
        root: Node = Node(a)
        if visualize:
            self._dot.node(str(root.get_value()))
        for values in self.__data[a].unique():  # TODO: check this line
            s_v: pd.DataFrame = data[data[a] == values]
            if s_v.shape[0] == 0:  # shape[0] gives the number of rows/observations
                leaf_node: Node = Node(data[label].mode()[0]) # can't do this if s_v shape is empty
                root.add_child(values, leaf_node)
                if visualize:
                    self._dot.node(str(leaf_node.get_value()))
                    self._dot.edge(str(root.get_value()), str(leaf_node.get_value()))
            else:
                this_node: Node = self._id3(s_v, attributes - {a}, label, splitting_criteria, set_depth, depth + 1, visualize)
                root.add_child(values, this_node)
                if visualize:
                    self._dot.node(str(this_node.get_value()))
                    self._dot.edge(str(root.get_value()), str(this_node.get_value()), label=str(values))
        return root

    '''
    Builds and stores the decision tree
    '''
    def build(self, attributes: set, label: typing.Any, splitting_criteria: Callable, set_depth: int, depth: int = 0, visualize: bool = False) -> None:
        self.__head = self._id3(self.__data, attributes, label, splitting_criteria, set_depth, depth, visualize)

    '''
    The criteria is the function that we use to decide the "best" split. Each of its inputs should be a percent.
    '''
    def _total_information_gain(self, data: pd.DataFrame, attributes: set, label: typing.Any, criteria: Callable) -> str:
        gain_dict: dict[float, str] = {}  # for quick retrival of attribute based on max information gain
        gain_list: list[float] = []  # for quick max calculation
        if criteria != self.majority_error:
            s_criteria = criteria(*data[label].value_counts(normalize=True))
        else:
            s_criteria = criteria(data, label)
        for attribute in attributes:
            gain = s_criteria  # shortcut that stands for the entropy (or other splitting criteria) of s so, we don't
            # have to recalculate s_criteria each iteration
            for values in data[attribute].unique():
                s_v: pd.DataFrame = data[data[attribute] == values]
                if criteria != self.majority_error:
                    criteria_s_v = criteria(*self._match_positive_negative(s_v, label)) # TODO: change this to value counts
                else:
                    criteria_s_v = criteria(s_v, label)
                gain -= (s_v.shape[0] / data.shape[0]) * criteria_s_v
            gain_dict[gain] = attribute
            gain_list.append(gain)
        return gain_dict[max(gain_list)]

    """
    Calculate and return entropy based on the proportion data observed
    """
    def entropy(self, *args: float) -> float:
        calculated_entropy: float = 0
        for arg in args:
            calculated_entropy -= arg * math.log2(arg)
        return calculated_entropy

    """
    Finds the proportion of examples that match the values the splitting attributes can take
    """
    def _match_positive_negative(self, s_v: pd.DataFrame, label: str) -> list[float]:
        return_list: list[float] = []
        for proportion in s_v[label].value_counts(normalize=True):
            return_list.append(proportion)
        return return_list

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

    '''
    Only works with multiple observations
    '''
    def predict(self, data: pd.DataFrame, row_num: int) -> typing.Any:
        current_node: Node = self.__head
        while current_node.has_children():
            # the value of the node determines what attribute to look down
            value_node: typing.Any = current_node.get_value()
            value_data_at_node: typing.Any = data.iloc[row_num][value_node]
            current_node = current_node.get_child_edge(value_data_at_node)
        return current_node.get_value()
