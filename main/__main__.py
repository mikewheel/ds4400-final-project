"""
The main driver for running linear regression and classification on wine data

Written by Michael Wheeler and Jay Sherman
"""

from argparse import ArgumentParser
from typing import Tuple

import numpy as np
import pandas as pd

from basis_function_expansions.calculate_bfe import basis_expansion
from config import INPUT_DATA_DIR, make_logger
from linear.model_creation import run_linear_models
from logistic.model_creation import run_logistic_models
from svm.model_creation import run_svm_models

logger = make_logger(__name__)


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separates data into test, training, and validation data sets randomly (deterministic).

    Training and validation data sets are each 45% of the original data set, and the
    test data is 10%. The random selection of data is not stratified based on quality.

    :param data: the data to split into test, train, and validation sets
    :returns: in order, the train, validation, and test data sets
    """
    test_data: pd.DataFrame = data.sample(frac=0.1, random_state=0)
    nontest_data: pd.DataFrame = data[~(data.isin(test_data))].dropna(how="all")
    train_data: pd.DataFrame = nontest_data.sample(frac=0.5, random_state=0)
    validation_data: pd.DataFrame = nontest_data[(nontest_data.isin(train_data))].dropna(how="all")
    return train_data, validation_data, test_data


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--model", nargs=1, type=str, metavar="model")
    args = parser.parse_args()
    model = args.model[0]
    
    red_wines = pd.read_csv(INPUT_DATA_DIR / "wine_quality_red.csv")
    white_wines = pd.read_csv(INPUT_DATA_DIR / "wine_quality_white.csv")
    
    logger.info(f'BEGIN: split data into training, validation, and test sets')
    rw_train, rw_valid, rw_test = split_data(red_wines)
    ww_train, ww_valid, ww_test = split_data(white_wines)
    
    logger.info(f'BEGIN: separate input data from output data')
    rw_train_x, rw_valid_x, rw_test_x = [df.iloc[:, range(11)]
                                         for df in [rw_train, rw_valid, rw_test]]
    ww_train_x, ww_valid_x, ww_test_x = [df.iloc[:, range(11)]
                                         for df in [ww_train, ww_valid, ww_test]]
    rw_train_y, rw_valid_y, rw_test_y = [df.iloc[:, range(11, 12)]
                                         for df in [rw_train, rw_valid, rw_test]]
    ww_train_y, ww_valid_y, ww_test_y = [df.iloc[:, range(11, 12)]
                                         for df in [ww_train, ww_valid, ww_test]]
    
    logger.info(f'BEGIN: calculate basis function expansions')
    # the values for exponents for all basis function expansions
    powers_list = [[1 for i in range(11)] for j in range(23)]
    # setting 11 of the inner lists to have 0 for a single feature (will remove the feature)
    for i in range(1, 12):
        powers_list[i][i - 1] = 0
    # setting 11 of the inner lists to have 2 for a single feature (x_i and x_i^2 will be present in the bfe)
    for i in range(12, 23):
        powers_list[i][i - 12] = 2
    # generating a list of phi(x) for different basis functions
    rw_train_x_list, rw_valid_x_list, rw_test_x_list = [
        [basis_expansion(df, powers)
         for powers in powers_list]
        for df in [rw_train_x, rw_valid_x, rw_test_x]
    ]
    ww_train_x_list, ww_valid_x_list, ww_test_x_list = [
        [basis_expansion(df, powers)
         for powers in powers_list]
        for df in [ww_train_x, ww_valid_x, ww_test_x]
    ]
    
    logger.info(f'BEGIN: merge red wine and white wine data for classification purposes.')
    all_train_x_list = []
    all_valid_x_list = []
    all_test_x_list = []
    for i in range(len(ww_train_x_list)):
        combined_train_x = pd.concat([ww_train_x_list[i], rw_train_x_list[i]], ignore_index=True)
        all_train_x_list.append(combined_train_x)
        
        combined_valid_x = pd.concat([ww_valid_x_list[i], rw_valid_x_list[i]], ignore_index=True)
        all_valid_x_list.append(combined_valid_x)
        
        combined_test_x = pd.concat([ww_test_x_list[i], rw_test_x_list[i]], ignore_index=True)
        all_test_x_list.append(combined_test_x)
    
    all_train_y = np.repeat([1, 0], [ww_train_x_list[0].shape[0], rw_train_x_list[0].shape[0]], axis=0)
    all_valid_y = np.repeat([1, 0], [ww_valid_x_list[0].shape[0], rw_valid_x_list[0].shape[0]], axis=0)
    all_test_y = np.repeat([1, 0], [ww_test_x_list[0].shape[0], rw_test_x_list[0].shape[0]], axis=0)
    
    if model == "logistic":
        logger.info(f'BEGIN: start running logistic classifier')
        run_logistic_models(all_train_x_list, all_valid_x_list, all_test_x_list,
                            all_train_y, all_valid_y, all_test_y)
    elif model == "linear":
        logger.info(f'BEGIN: start running linear classifier.')
        run_linear_models(rw_train_x_list, rw_valid_x_list, rw_test_x_list,
                          rw_train_y, rw_valid_y, rw_test_y,
                          ww_train_x_list, ww_valid_x_list, ww_test_x_list,
                          ww_train_y, ww_valid_y, ww_test_y)
    elif model == "svm":
        logger.info(f'BEGIN: start running SVM classifier.')
        run_svm_models(all_train_x_list, all_valid_x_list, all_test_x_list,
                       all_train_y, all_valid_y, all_test_y)
    else:
        raise ValueError(f'Model type not recognized: "{model}"')
