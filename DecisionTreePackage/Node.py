from __future__ import annotations
import typing


class Node:
    def __init__(self, value: str) -> None:
        self.__value: typing.Any = value
        self.__children: dict[typing.Any, (int, Node)] = {}
        self.__edges: dict[typing.Any, Node] = {}

    def add_child(self, edge: typing.Any, ref_next_child: Node) -> None:
        """
        Setter method for adding new child
        :param edge: Value feature can take
        :param ref_next_child: reference to next child
        :return: void
        """
        self.__children[ref_next_child.__value] = (edge, ref_next_child)
        self.__edges[edge] = ref_next_child

    """
    Removes all children
    """
    def remove_all_children(self) -> None:
        self.__children.clear()

    """
    Removes a child
    """
    def remove_child(self, key: str) -> None:
        del self.__children[key]

    def get_value(self):
        return self.__value

    def has_child(self, value: typing.Any) -> bool:
        return self.__children.has_key(value)

    def has_children(self):
        return len(self.__children) > 0

    '''
    returns the child node of a value
    '''
    def get_child(self, value: typing.Any):
        return self.__children[value][1]

    def get_edge(self, value: typing.Any):
        return self.__children[value][0]

    def get_child_edge(self, edge: typing.Any) -> Node:
        """
        Gets the child associated with the edge
        :param edge: value feature can take
        :return: child node
        """
        return self.__edges[edge]
