"""
The controller for running linear regression and classification on wine data.

Written by Michael Wheeler and Jay Sherman
"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd

from basis_functions.expansions import expand_basis
from config import INPUT_DATA_DIR, make_logger
from models.classifiers.logistic import run_logistic_models
from models.classifiers.svm import run_svm_models
from models.regressions.linear import run_linear_models
from models.utils import split_data

logger = make_logger(__name__)

if __name__ == "__main__":
    
    parser = ArgumentParser(description="DS4400 Final Project Spring 2020, by Michael Wheeler and Jay Sherman. "
                                        "Classifiers and regressions on wine data.")
    parser.add_argument("--model", nargs=1, type=str, metavar="MODEL_CHOICE",
                        help='"linear", "logistic", "svm", or "all"')
    args = parser.parse_args()
    model_choice = args.model[0].lower()
    
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
        [expand_basis(df, powers)
         for powers in powers_list]
        for df in [rw_train_x, rw_valid_x, rw_test_x]
    ]
    ww_train_x_list, ww_valid_x_list, ww_test_x_list = [
        [expand_basis(df, powers)
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
    
    # FIXME -- convert these to dataframes?
    all_train_y = np.repeat([1, 0], [ww_train_x_list[0].shape[0], rw_train_x_list[0].shape[0]], axis=0)
    all_valid_y = np.repeat([1, 0], [ww_valid_x_list[0].shape[0], rw_valid_x_list[0].shape[0]], axis=0)
    all_test_y = np.repeat([1, 0], [ww_test_x_list[0].shape[0], rw_test_x_list[0].shape[0]], axis=0)
    
    if model_choice == "logistic":
        logger.info(f'BEGIN: logistic classifier')
        run_logistic_models(all_train_x_list, all_valid_x_list, all_test_x_list,
                            all_train_y, all_valid_y, all_test_y)
    elif model_choice == "linear":
        logger.info(f'BEGIN: linear regression.')
        run_linear_models(rw_train_x_list, rw_valid_x_list, rw_test_x_list,
                          rw_train_y, rw_valid_y, rw_test_y,
                          ww_train_x_list, ww_valid_x_list, ww_test_x_list,
                          ww_train_y, ww_valid_y, ww_test_y)
    elif model_choice == "svm":
        logger.info(f'BEGIN: SVM classifier.')
        run_svm_models(all_train_x_list, all_valid_x_list, all_test_x_list,
                       all_train_y, all_valid_y, all_test_y)
    elif model_choice == "all":
        logger.info(f'BEGIN: all models.')
        run_logistic_models(all_train_x_list, all_valid_x_list, all_test_x_list,
                            all_train_y, all_valid_y, all_test_y)
        run_linear_models(rw_train_x_list, rw_valid_x_list, rw_test_x_list,
                          rw_train_y, rw_valid_y, rw_test_y,
                          ww_train_x_list, ww_valid_x_list, ww_test_x_list,
                          ww_train_y, ww_valid_y, ww_test_y)
        run_svm_models(all_train_x_list, all_valid_x_list, all_test_x_list,
                       all_train_y, all_valid_y, all_test_y)
    else:
        raise ValueError(f'Model type not recognized: "{model_choice}"')
