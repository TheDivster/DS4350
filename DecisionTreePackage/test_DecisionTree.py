import math
import pandas as pd
from unittest import TestCase
from DecisionTreePackage.DecisionTree import DecisionTree


class TestDecisionTree(TestCase, DecisionTree):

    def test__entropy(self):
        df: pd.DataFrame = pd.DataFrame()
        df["col1"] = pd.Series([1, 2, 3, 2])
        dtree: DecisionTree = DecisionTree(df)
        self.assertEqual(dtree.entropy(*df["col1"].value_counts(normalize=True)),
                         - (1 / 4) * math.log2(1 / 4) * 2 - (1 / 2) * math.log2(
                             1 / 2))  # the * in front of an iterable expands it

    def test_id3(self):
        df: pd.DataFrame = pd.DataFrame()
        df["x1"] = pd.Series([0, 0, 0, 1, 0, 1, 0])
        df["x2"] = pd.Series([0, 1, 0, 0, 1, 1, 1])
        df["x3"] = pd.Series([1, 0, 1, 0, 1, 0, 0])
        df["x4"] = pd.Series([0, 0, 1, 1, 0, 0, 1])
        df["y"] = pd.Series([0, 0, 1, 1, 0, 0, 0])
        dtree: DecisionTree = DecisionTree(df)
        dtree._id3(df, {"x1", "x2", "x3", "x4"}, "y", dtree.entropy, 100, visualize=True)
        try:
            dtree._dot.render(filename='tree.dot')
        except Exception as e:
            print("something went wrong with the visualization", e)
        self.assertEqual(dtree._id3(df, {"x1", "x2", "x3", "x4"}, "y", dtree.entropy, 100).get_value(), "x2")

    def test_id3_categorical(self):
        df: pd.DataFrame = pd.DataFrame()
        df["o"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
        df["t"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
        df["h"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
        df["w"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
        df["play?"] = pd.Series(["-", "-", "+", "+", "+", "-", "+", "-", "+", "+", "+", "+", "+", "-"])
        dtree: DecisionTree = DecisionTree(df)
        dtree._id3(df, {"o", "t", "h", "w"}, "play?", dtree.entropy, 100, visualize=True)
        try:
            dtree._dot.render(filename='tree.dot')
        except Exception as e:
            print("something went wrong with the visualization", e)
        self.assertEqual(dtree._id3(df, {"o", "t", "h", "w"}, "play?", dtree.entropy, 100).get_value(), "o")

    def test_id3_categorical_ginni(self):
        df: pd.DataFrame = pd.DataFrame()
        df["o"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
        df["t"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
        df["h"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
        df["w"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
        df["play?"] = pd.Series(["-", "-", "+", "+", "+", "-", "+", "-", "+", "+", "+", "+", "+", "-"])
        dtree: DecisionTree = DecisionTree(df)
        dtree._id3(df, {"o", "t", "h", "w"}, "play?", dtree.ginni_index, 100, visualize=True)
        try:
            dtree._dot.render(filename='tree.dot')
        except Exception as e:
            print("something went wrong with the visualization", e)
        self.assertEqual(dtree._id3(df, {"o", "t", "h", "w"}, "play?", dtree.ginni_index, 100).get_value(), "o")

    def test_majority_label(self):
        df: pd.DataFrame = pd.DataFrame()
        df["col1"] = pd.Series([1, 2, 2, 2])
        df["target"] = pd.Series([1, 0, 1, 1])
        dtree: DecisionTree = DecisionTree(df)
        self.assertEqual(0.25, dtree.majority_error(df, "target"))

    def test_majority_label_2(self):
        df: pd.DataFrame = pd.DataFrame()
        df["col1"] = pd.Series([1, 2, 2, 2])
        df["target"] = pd.Series([1, 0, 0, 1])
        dtree: DecisionTree = DecisionTree(df)
        self.assertEqual(0.5, dtree.majority_error(df, "target"))

    def test_id3_categorical_majority(self):
        df: pd.DataFrame = pd.DataFrame()
        df["outlook"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
        df["temp"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
        df["humidity"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
        df["windy"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
        df["play?"] = pd.Series(["-", "-", "+", "+", "+", "-", "+", "-", "+", "+", "+", "+", "+", "-"])
        dtree: DecisionTree = DecisionTree(df)
        dtree._id3(df, {"outlook", "temp", "humidity", "windy"}, "play?", dtree.majority_error, 100, visualize=True)
        try:
            dtree._dot.render(filename='tree.dot')
        except Exception as e:
            print("something went wrong with the visualization", e)
        self.assertEqual(dtree._id3(df, {"outlook", "temp", "humidity", "windy"}, "play?", dtree.majority_error, 100).get_value(), "humidity")  # What should we expect

    def test_majority_2(self):
        df: pd.DataFrame = pd.DataFrame()
        df["x1"] = pd.Series([0, 0, 0, 1, 0, 1, 0])
        df["x2"] = pd.Series([0, 1, 0, 0, 1, 1, 1])
        df["x3"] = pd.Series([1, 0, 1, 0, 1, 0, 0])
        df["x4"] = pd.Series([0, 0, 1, 1, 0, 0, 1])
        df["y"] = pd.Series([0, 0, 1, 1, 0, 0, 0])
        dtree: DecisionTree = DecisionTree(df)
        dtree._id3(df, {"x1", "x2", "x3", "x4"}, "y", dtree.majority_error, 100, visualize=True)
        try:
            dtree._dot.render(filename='tree.dot')
        except Exception as e:
            print("something went wrong with the visualization", e)
        self.assertEqual(dtree._id3(df, {"x1", "x2", "x3", "x4"}, "y", dtree.majority_error, 100).get_value(), "x2")

    def test_predict_numerical(self):
        df: pd.DataFrame = pd.DataFrame()
        df["x1"] = pd.Series([0, 0, 0, 1, 0, 1, 0])
        df["x2"] = pd.Series([0, 1, 0, 0, 1, 1, 1])
        df["x3"] = pd.Series([1, 0, 1, 0, 1, 0, 0])
        df["x4"] = pd.Series([0, 0, 1, 1, 0, 0, 1])
        df["y"] = pd.Series([0, 0, 1, 1, 0, 0, 0])
        dtree: DecisionTree = DecisionTree(df)
        dtree.build({"x1", "x2", "x3", "x4"}, "y", dtree.entropy, 100)
        self.assertEqual(0, dtree.predict(df, 0))
        self.assertEqual(0, dtree.predict(df, 1))
        self.assertEqual(1, dtree.predict(df, 2))
        self.assertEqual(1, dtree.predict(df, 3))
        self.assertEqual(0, dtree.predict(df, 4))
        self.assertEqual(0, dtree.predict(df, 5))
        self.assertEqual(0, dtree.predict(df, 6))

    def test_predict_numerical_majority(self):
        df: pd.DataFrame = pd.DataFrame()
        df["x1"] = pd.Series([0, 0, 0, 1, 0, 1, 0])
        df["x2"] = pd.Series([0, 1, 0, 0, 1, 1, 1])
        df["x3"] = pd.Series([1, 0, 1, 0, 1, 0, 0])
        df["x4"] = pd.Series([0, 0, 1, 1, 0, 0, 1])
        df["y"] = pd.Series([0, 0, 1, 1, 0, 0, 0])
        dtree: DecisionTree = DecisionTree(df)
        dtree.build({"x1", "x2", "x3", "x4"}, "y", dtree.majority_error, 100)
        self.assertEqual(0, dtree.predict(df, 0))
        self.assertEqual(0, dtree.predict(df, 1))
        self.assertEqual(1, dtree.predict(df, 2))
        self.assertEqual(1, dtree.predict(df, 3))
        self.assertEqual(0, dtree.predict(df, 4))
        self.assertEqual(0, dtree.predict(df, 5))
        self.assertEqual(0, dtree.predict(df, 6))

    def test_predict_categorical_ginni(self):
        df: pd.DataFrame = pd.DataFrame()
        df["o"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
        df["t"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
        df["h"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
        df["w"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
        df["play?"] = pd.Series(["-", "-", "+", "+", "+", "-", "+", "-", "+", "+", "+", "+", "+", "-"])
        dtree: DecisionTree = DecisionTree(df)
        dtree.build({"o", "t", "h", "w"}, "play?", dtree.ginni_index, 100)
        self.assertEqual("-", dtree.predict(df, 0))
        self.assertEqual("+", dtree.predict(df, 2))
        self.assertEqual("+", dtree.predict(df, -2))
        self.assertEqual("+", dtree.predict(df, -3))
        self.assertEqual("+", dtree.predict(df, -6))
        self.assertEqual("-", dtree.predict(df, 5))

    def test_predict_categorical_majority(self):
        df: pd.DataFrame = pd.DataFrame()
        df["o"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
        df["t"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
        df["h"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
        df["w"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
        df["play?"] = pd.Series(["-", "-", "+", "+", "+", "-", "+", "-", "+", "+", "+", "+", "+", "-"])
        dtree: DecisionTree = DecisionTree(df)
        dtree.build({"o", "t", "h", "w"}, "play?", dtree.majority_error, 100)
        self.assertEqual("-", dtree.predict(df, 0))
        self.assertEqual("+", dtree.predict(df, 2))
        self.assertEqual("+", dtree.predict(df, -2))
        self.assertEqual("+", dtree.predict(df, -3))
        self.assertEqual("+", dtree.predict(df, -6))
        self.assertEqual("-", dtree.predict(df, 5))

    def test_predict_categorical_entropy(self):
        df: pd.DataFrame = pd.DataFrame()
        df["o"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
        df["t"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
        df["h"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
        df["w"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
        df["play?"] = pd.Series(["-", "-", "+", "+", "+", "-", "+", "-", "+", "+", "+", "+", "+", "-"])
        dtree: DecisionTree = DecisionTree(df)
        dtree.build({"o", "t", "h", "w"}, "play?", dtree.majority_error, 100)
        self.assertEqual("-", dtree.predict(df, 0))
        self.assertEqual("+", dtree.predict(df, 2))
        self.assertEqual("+", dtree.predict(df, -2))
        self.assertEqual("+", dtree.predict(df, -3))
        self.assertEqual("+", dtree.predict(df, -6))
        self.assertEqual("-", dtree.predict(df, 5))

    def test_depth_ginni_index(self):
        df: pd.DataFrame = pd.DataFrame()
        df["o"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
        df["t"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
        df["h"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
        df["w"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
        df["play?"] = pd.Series(["-", "-", "+", "+", "+", "-", "+", "-", "+", "+", "+", "+", "+", "-"])
        dtree: DecisionTree = DecisionTree(df)
        dtree.build({"o", "t", "h", "w"}, "play?", dtree.ginni_index, 1, visualize=True)
        try:
            dtree._dot.render(filename='tree.dot')
        except Exception as e:
            print("something went wrong with the visualization", e)

    def test_depth_2_ginni_index(self):
        df: pd.DataFrame = pd.DataFrame()
        df["o"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
        df["t"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
        df["h"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
        df["w"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
        df["play?"] = pd.Series(["-", "-", "+", "+", "+", "-", "+", "-", "+", "+", "+", "+", "+", "-"])
        dtree: DecisionTree = DecisionTree(df)
        dtree.build({"o", "t", "h", "w"}, "play?", dtree.ginni_index, 2, visualize=True)
        try:
            dtree._dot.render(filename='tree.dot')
        except Exception as e:
            print("something went wrong with the visualization", e)

    def test_depth_0_ginni_index(self):
        df: pd.DataFrame = pd.DataFrame()
        df["o"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
        df["t"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
        df["h"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
        df["w"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
        df["play?"] = pd.Series(["-", "-", "+", "+", "+", "-", "+", "-", "+", "+", "+", "+", "+", "-"])
        dtree: DecisionTree = DecisionTree(df)
        dtree.build({"o", "t", "h", "w"}, "play?", dtree.ginni_index, 0, visualize=True)
        try:
            dtree._dot.render(filename='tree.dot')
        except Exception as e:
            print("something went wrong with the visualization", e)
        self.assertEqual("+", dtree.predict(df, 0))
        self.assertEqual("+", dtree.predict(df, 2))
        self.assertEqual("+", dtree.predict(df, -2))
        self.assertEqual("+", dtree.predict(df, -3))
        self.assertEqual("+", dtree.predict(df, -6))
        self.assertEqual("+", dtree.predict(df, 5))

    def test_data_depth(self):
        car_train: pd.DataFrame = pd.read_csv("Car/train.csv",
                                              names=["buying", "maint", "doors", "persons", "lug_boot", "safety",
                                                     "label"])
        car_test_data: pd.DataFrame = pd.read_csv("Car/test.csv",
                                                  names=["buying", "maint", "doors", "persons", "lug_boot", "safety",
                                                         "label"])
        tree: DecisionTree = DecisionTree(car_train)
        tree.build(attributes=set(car_train.columns)-{"label"}, label="label", splitting_criteria=tree.entropy, set_depth=5, visualize=True)
        try:
            tree._dot.render(filename='tree.dot')
        except Exception as e:
            print("something went wrong with the visualization", e)

        def predict_dataframe(model: DecisionTree, test_data: pd.DataFrame, label: str) -> float:
            accuracy_count: int = 0  # represents the count of correct predictions
            for j in range(test_data.shape[0]):
                if model.predict(test_data, j) == test_data[label].iloc[j]:
                    accuracy_count += 1
            return accuracy_count / len(test_data)

        print(predict_dataframe(tree, car_test_data, "label"))
