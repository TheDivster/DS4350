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
    dtree.build_boost( {"o", "t", "h", "w"}, "play?", dtree.entropy, 20)

  def test_build_boost(self):
    df: pd.DataFrame = pd.DataFrame()
    df["o"] = pd.Series(["s", "s", "o", "r", "r", "r", "o", "s", "s", "r", "s", "o", "o", "r"])
    df["t"] = pd.Series(["h", "h", "h", "m", "c", "c", "c", "m", "c", "m", "m", "m", "h", "m"])
    df["h"] = pd.Series(["h", "h", "h", "h", "n", "n", "n", "h", "n", "n", "n", "h", "n", "h"])
    df["w"] = pd.Series(["w", "s", "w", "w", "w", "s", "s", "w", "w", "w", "s", "s", "w", "s"])
    df["play?"] = pd.Series([-1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1])
    dtree: AdaBoost = AdaBoost(df)
    dtree.build_boost( {"o", "t", "h", "w"}, "play?", dtree.entropy, 20)
