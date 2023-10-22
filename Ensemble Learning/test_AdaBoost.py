from unittest import TestCase
from AdaBoost import AdaBoost
import pandas as pd
import numpy as np
from DecisionStump import DecisionStump


class TestAdaBoost(TestCase):
  def test_build_boost(self):
    df: pd.DataFrame = pd.DataFrame()
    df["o"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
    df["t"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
    df["h"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
    df["w"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
    df["play?"] = pd.Series([-1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1])
    dtree: AdaBoost = AdaBoost(df)
    dtree.build_boost( {"o", "t", "h", "w"}, "play?", dtree.entropy, 20)

  def test_build_boost_2(self):
    df: pd.DataFrame = pd.DataFrame()
    df["o"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
    df["t"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
    df["h"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
    df["w"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
    df["play?"] = pd.Series([-1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1])
    dtree: AdaBoost = AdaBoost(df)
    dtree.build_boost( {"o", "t", "h", "w"}, "play?", dtree.entropy, 20)

  def test_data_depth(self):
    bank_train: pd.DataFrame = pd.read_csv("/Users/divytripathy/PycharmProjects/Machine Learning/DecisionTreePackage/bank/train.csv",
                                           names=["age", "job", "marital", "education", "default", "balance", "housing",
                                                  "loan", "contact", "day", "month", "duration", "campaign", "pdays",
                                                  "previous", "poutcome", "y"])
    bank_test: pd.DataFrame = pd.read_csv("/Users/divytripathy/PycharmProjects/Machine Learning/DecisionTreePackage/bank/test.csv",
                                          names=["age", "job", "marital", "education", "default", "balance", "housing",
                                                 "loan", "contact", "day", "month", "duration", "campaign", "pdays",
                                                 "previous", "poutcome", "y"])
    tree: DecisionStump = DecisionStump(bank_train)

    def make_numerical_attributes_binary(data):
      column_types: dict[str, str] = data.dtypes.to_dict()
      for column in column_types:
        if column_types[column] == "int64" or column_types[column] == "float64":
          data[column] = data[column].apply(lambda x: x >= data[column].median())

    make_numerical_attributes_binary(bank_test)
    make_numerical_attributes_binary(bank_train)

    tree.build(attributes=set(bank_train.columns) - {"y"}, label="y", splitting_criteria=tree.entropy, weights=np.ones(bank_train.shape[0]), set_depth=2)

    def predict_dataframe(model: DecisionStump, test_data: pd.DataFrame, label: str) -> float:
      accuracy_count: int = 0  # represents the count of correct predictions
      for j in range(test_data.shape[0]):
        if model.predict(test_data, j) == test_data[label].iloc[j]:
          accuracy_count += 1
      return accuracy_count / len(test_data)

    print(predict_dataframe(tree, bank_test, "y"))

  # def test_data_depth(self):
  #   bank_train: pd.DataFrame = pd.read_csv("/Users/divytripathy/PycharmProjects/Machine Learning/DecisionTreePackage/bank/train.csv",
  #                                          names=["age", "job", "marital", "education", "default", "balance", "housing",
  #                                                 "loan", "contact", "day", "month", "duration", "campaign", "pdays",
  #                                                 "previous", "poutcome", "y"])
  #   bank_test: pd.DataFrame = pd.read_csv("/Users/divytripathy/PycharmProjects/Machine Learning/DecisionTreePackage/bank/test.csv",
  #                                         names=["age", "job", "marital", "education", "default", "balance", "housing",
  #                                                "loan", "contact", "day", "month", "duration", "campaign", "pdays",
  #                                                "previous", "poutcome", "y"])
  #   tree: AdaBoost = AdaBoost(bank_train)
  #   tree.build_boost(attributes=set(bank_train.columns) - {"y"}, label="y", splitting_criteria=tree.entropy, num_trees=2)
  #
  #   def predict_dataframe(model: AdaBoost, test_data: pd.DataFrame, label: str) -> float:
  #     accuracy_count: int = 0  # represents the count of correct predictions
  #     for j in range(test_data.shape[0]):
  #       if model.predict(test_data, j) == test_data[label].iloc[j]:
  #         accuracy_count += 1
  #     return accuracy_count / len(test_data)
  #
  #   print(predict_dataframe(tree, bank_test, "label"))
