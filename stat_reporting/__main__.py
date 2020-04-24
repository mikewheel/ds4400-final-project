"""
Main module for statistic analyses

Written by Michael Wheeler and Jay Sherman.
"""

from scipy import stats
from pandas import read_csv
from statistics import mean, stdev
import sys
print(sys.path)
from config import INPUT_DATA_DIR


wine_data = {
    "red": read_csv(INPUT_DATA_DIR / "wine_quality_red.csv"),
    "white": read_csv(INPUT_DATA_DIR / "wine_quality_white.csv"),
}

wine_features = {
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality"
}

t_test_dict = {}
stats_dict = {}

for feature in wine_features:
    red_wine_feature = wine_data["red"][feature]
    white_wine_feature = wine_data["white"][feature]
    t, p_value = stats.ttest_ind(red_wine_feature, white_wine_feature, equal_var=True)
    t_test_dict[feature] = t
    stats_dict[feature] = [mean(red_wine_feature), stdev(red_wine_feature),
                           mean(white_wine_feature), stdev(white_wine_feature)]

t_tests = list(t_test_dict)
t_tests.sort(key = lambda f: t_test_dict[f])

for feature in t_tests:
    print(feature, ":", t_test_dict[feature])
    print(stats_dict[feature], "\n")

