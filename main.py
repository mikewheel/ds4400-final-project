"""
The controller for running linear regression and classification on wine data.

Written by Michael Wheeler and Jay Sherman
"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd

from basis_functions.expansions import expand_basis, generate_exponents
from config import INPUT_DATA_DIR, make_logger
from models.classifiers.logistic import LogisticModelFactory
from models.classifiers.svm import SupportVectorModelFactory
from models.regressions.linear import LinearRegressionModelFactory
from models.utils import split_data, run_models_all_bfes

logger = make_logger(__name__)

if __name__ == "__main__":
    parser = ArgumentParser(description="DS4400 Final Project Spring 2020, by Michael Wheeler and Jay Sherman. "
                                        "Classifiers and regressions on wine data.")
    parser.add_argument("--model", choices=["linear", "logistic", "svm", "all"], type=str.lower,
                        metavar="MODEL_CHOICE", help='"linear", "logistic", "svm", or "all"')
    args = parser.parse_args()
    model_choice = args.model
    
    logger.info('Reading in red and white wine datasets from disk...')
    red_wines = pd.read_csv(INPUT_DATA_DIR / "wine_quality_red.csv")
    white_wines = pd.read_csv(INPUT_DATA_DIR / "wine_quality_white.csv")
    
    logger.info('Splitting data into training, validation, and test sets.')
    rw_train, rw_valid, rw_test = split_data(red_wines)
    ww_train, ww_valid, ww_test = split_data(white_wines)
    
    logger.info('Separating explanatory and response variables.')
    rw_train_x, rw_valid_x, rw_test_x = [df.iloc[:, range(11)]
                                         for df in [rw_train, rw_valid, rw_test]]
    ww_train_x, ww_valid_x, ww_test_x = [df.iloc[:, range(11)]
                                         for df in [ww_train, ww_valid, ww_test]]
    rw_train_y, rw_valid_y, rw_test_y = [df.iloc[:, range(11, 12)]
                                         for df in [rw_train, rw_valid, rw_test]]
    ww_train_y, ww_valid_y, ww_test_y = [df.iloc[:, range(11, 12)]
                                         for df in [ww_train, ww_valid, ww_test]]
    
    logger.info(f'Generating a list of phi(x) for different basis functions.')
    powers_list = generate_exponents()
    rw_train_x_list, rw_valid_x_list, rw_test_x_list = [
        [expand_basis(df, powers) for powers in powers_list]
        for df in [rw_train_x, rw_valid_x, rw_test_x]]
    
    ww_train_x_list, ww_valid_x_list, ww_test_x_list = [
        [expand_basis(df, powers) for powers in powers_list]
        for df in [ww_train_x, ww_valid_x, ww_test_x]]
    
    logger.info(f'Merging red wine and white wine datasets for classifier training.')
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
    
    if model_choice in ("logistic", "all"):
        logger.info(f'BEGIN: logistic classifier.')
        run_models_all_bfes(LogisticModelFactory, all_train_x_list, all_valid_x_list, all_test_x_list,
                            all_train_y, all_valid_y, all_test_y)
    
    if model_choice in ("svm", "all"):
        logger.info(f'BEGIN: SVM classifier.')
        kernels = ["rbf", "linear", "poly"]
        for kernel in kernels:
            run_models_all_bfes(SupportVectorModelFactory, all_train_x_list, all_valid_x_list, all_test_x_list,
                                all_train_y, all_valid_y, all_test_y, kenel=kernel)
    
    if model_choice in ("linear", "all"):
        logger.info(f'BEGIN: linear regression.')
        run_models_all_bfes(LinearRegressionModelFactory, rw_train_x_list, rw_valid_x_list, rw_test_x_list,
                            rw_train_y, rw_valid_y, rw_test_y, color="red")
        run_models_all_bfes(LinearRegressionModelFactory, ww_train_x_list, ww_valid_x_list, ww_test_x_list,
                            ww_train_y, ww_valid_y, ww_test_y, color="white")
