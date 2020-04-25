"""
Functions common to the execution of all models in the project.

Written by Michael Wheeler and Jay Sherman.
"""

from typing import Tuple

from pandas import DataFrame

from config import make_logger

logger = make_logger(__name__)


def split_data(data: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Separates data into test, training, and validation data sets randomly (deterministic).

    Training and validation data sets are each 45% of the original data set, and the
    test data is 10%. The random selection of data is not stratified based on quality.

    :param data: the data to split into test, train, and validation sets
    :returns: in order, the train, validation, and test data sets
    """
    test_data: DataFrame = data.sample(frac=0.1, random_state=0)
    nontest_data: DataFrame = data[~(data.isin(test_data))].dropna(how="all")
    train_data: DataFrame = nontest_data.sample(frac=0.5, random_state=0)
    validation_data: DataFrame = nontest_data[(nontest_data.isin(train_data))].dropna(how="all")
    return train_data, validation_data, test_data


def run_models(train_x: DataFrame,
               valid_x: DataFrame,
               test_x: DataFrame,
               train_y: DataFrame,
               valid_y: DataFrame,
               test_y: DataFrame,
               **kwargs):
    """
    
    :param train_x:
    :param valid_x:
    :param test_x:
    :param train_y:
    :param valid_y:
    :param test_y:
    :param kwargs:
    :return:
    """
    pass
