from unittest import TestCase
from AdaBoost import AdaBoost
import pandas as pd


class TestAdaBoost(TestCase):
  def test_build_boost(self):
    df: pd.DataFrame = pd.DataFrame()
    df["o"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
    df["t"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
    df["h"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
    df["w"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
    df["play?"] = pd.Series([-1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1])
    dtree: AdaBoost = AdaBoost(df)
    dtree.build( {"o", "t", "h", "w"}, "play?", dtree.ENTROPY, 20, 1)

  def test_build_boost_ginni_index(self):
    df: pd.DataFrame = pd.DataFrame()
    df["o"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
    df["t"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
    df["h"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
    df["w"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
    df["play?"] = pd.Series([-1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1])
    dtree: AdaBoost = AdaBoost(df)
    dtree.build( {"o", "t", "h", "w"}, "play?", dtree.GINNI_INDEX, 20, 1)

  def test_predict(self):
    bank_train: pd.DataFrame = pd.read_csv(
      "/Users/divytripathy/PycharmProjects/Machine Learning/DecisionTreePackage/bank/train.csv",
      names=["age", "job", "marital", "education", "default", "balance", "housing",
             "loan", "contact", "day", "month", "duration", "campaign", "pdays",
             "previous", "poutcome", "y"])
    bank_test: pd.DataFrame = pd.read_csv(
      "/Users/divytripathy/PycharmProjects/Machine Learning/DecisionTreePackage/bank/test.csv",
      names=["age", "job", "marital", "education", "default", "balance", "housing",
             "loan", "contact", "day", "month", "duration", "campaign", "pdays",
             "previous", "poutcome", "y"])
    tree: AdaBoost = AdaBoost(bank_train)

    def make_numerical_attributes_binary(data):
      column_types: dict[str, str] = data.dtypes.to_dict()
      for column in column_types:
        if column_types[column] == "int64" or column_types[column] == "float64":
          data[column] = data[column].apply(lambda x: x >= data[column].median())

    make_numerical_attributes_binary(bank_test)
    make_numerical_attributes_binary(bank_train)

    tree.build(attributes=set(bank_train.columns) - {"y"}, label="y", splitting_criteria=tree.ENTROPY,
               num_trees=10, depth=5)

    def predict_dataframe(model: AdaBoost, test_data: pd.DataFrame, label: str) -> float:
      accuracy_count: int = 0  # represents the count of correct predictions
      for j in range(test_data.shape[0]):
        if model.predict(test_data, j) == test_data[label].iloc[j]:
          accuracy_count += 1
      return accuracy_count / len(test_data)

    print(predict_dataframe(tree, bank_test, "y"))
    print(predict_dataframe(tree, bank_train, "y"))
