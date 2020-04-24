"""
A file for creating linear regression models and saving them to file.

Written by Michael Wheeler and Jay Sherman
"""

import os
import pickle
from typing import List
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

from config import OUTPUT_DATA_DIR, BFE_DESCS, make_logger
from logging_models.logging_utils import log_linear_regression
logger = make_logger(__name__)

def run_linear_models_help(train_x: pd.DataFrame, valid_x: pd.DataFrame, test_x: pd.DataFrame,
                           train_y: pd.DataFrame, valid_y: pd.DataFrame, test_y: pd.DataFrame,
                           color: str, bfe_desc: str):
    """Run linear regression to predict the quality of wine based on physicochemical properties.

    The optimal value of theta is found using the training data, then the optimal value of the regularization
    parameter is found by seeing which of {0,0.01,0.1,1,10} yields the lowest validation error. Saves the model to a
    pickle file in the appropriate location inside the output/linear
    directory based on the values of color and bfe_desc, and saves a text file to the same directory
    describing the validation error for each regularization parameter value, which regularization parameter was used,
    the value of theta, the training error, and the test error.

    :param train_x: the input data for the model for the training set
    :param valid_x: the input data for the model for the validation set
    :param test_x: the input data for the model for the test set
    :param train_y: the quality of the wines in the training set
    :param valid_y: the quality of the wines in the validation set
    :param test_y: the quality of the wines in the test set
    :param color: the color of the wine ("red" or "white")
    :param bfe_desc: a description of the basis function expansion used
    """

    logger.info(f"Running linear model with BFE = {bfe_desc}")
    models = [[lam, Ridge(random_state=0, alpha=lam, fit_intercept=False, normalize=False)]
              for lam in [0.01, 0.1, 1, 10]]
    models.append([0, LinearRegression()])

    for model in models:
        model[1].fit(train_x, train_y)
        theta = model[1].coef_
        model.append(theta)
        error = mean_squared_error(valid_y, model[1].predict(valid_x))
        model.append(error)

    models.sort(key=lambda a: a[3])
    best_model = models[0]
    logger.debug(f"Found optimal lambda for linear model with BFE = {bfe_desc}: {best_model[0]}")

    dir = OUTPUT_DATA_DIR / "linear" / color / bfe_desc
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass

    pickle.dump(best_model[1], open(dir / "model.p", "wb"))

    train_error = mean_squared_error(train_y, best_model[1].predict(train_x))
    valid_error = mean_squared_error(valid_y, best_model[1].predict(valid_x))
    test_error = mean_squared_error(test_y, best_model[1].predict(test_x))

    log_linear_regression(best_model[2], best_model[0], train_error, valid_error, test_error, dir)




def run_linear_models(rw_train_x_bfes: List[pd.DataFrame], rw_valid_x_list: List[pd.DataFrame],  # FIXME Shift F6
                      rw_test_x_list: List[pd.DataFrame], rw_train_y: pd.DataFrame,
                      rw_valid_y: pd.DataFrame, rw_test_y: pd.DataFrame,
                      ww_train_x_list: List[pd.DataFrame], ww_valid_x_list: List[pd.DataFrame],
                      ww_test_x_list: List[pd.DataFrame], ww_train_y: pd.DataFrame,
                      ww_valid_y: pd.DataFrame, ww_test_y: pd.DataFrame):
    """Run linear regression to predict the quality of red and white wine.

    Runs separate models for red and white wine. The optimal value of theta is found using the training data,
    then the optimal value of the regularization parameter is found by seeing which of {0,0.01,0.1,1,10} yields
    the lowest validation error (using mean squared error). Saves the model to a pickle file in the appropriate
    location inside the output/linear directory based on the wine color and the basis function expansion, and saves a
    text file to the same directory describing the validation error for each regularization parameter value, which
    regularization parameter was used, the value of theta, the training error, and the test error.

    :param rw_train_x_bfes: a list of the input features for the training set with basis function expansions applied
                            for red wines
    :param rw_valid_x_list: a list of the input features for the validation set with basis function expansions applied
                            for red wines
    :param rw_test_x_list: a list of the input features for the test set with basis function expansions applied
                           for red wines
    :param rw_train_y: the quality of the wines in the training set for red wines
    :param rw_valid_y: the quality of the wines in the validation set for red wines
    :param rw_test_y: the quality of the wines in the test set for red wines
    :param ww_train_x_list: a list of the input features for the training set with basis function expansions applied
                            for white wines
    :param ww_valid_x_list: a list of the input features for the validation set with basis function expansions applied
                            for white wines
    :param ww_test_x_list: a list of the input features for the test set with basis function expansions applied
                           for white wines
    :param ww_train_y: the quality of the wines in the training set for white wines
    :param ww_valid_y: the quality of the wines in the validation set for white wines
    :param ww_test_y: the quality of the wines in the test set for white wines
    """

    #making appropriate directories for saving the models to file
    dir = OUTPUT_DATA_DIR / "linear"
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    try:
        os.mkdir(dir / "red")
    except FileExistsError:
        pass
    try:
        os.mkdir(dir / "white")
    except FileExistsError:
        pass

    for index, bfe_desc in enumerate(BFE_DESCS):
        run_linear_models_help(rw_train_x_bfes[index], rw_valid_x_list[index], rw_test_x_list[index],
                               rw_train_y, rw_valid_y, rw_test_y, "red", bfe_desc)
        run_linear_models_help(ww_train_x_list[index], ww_valid_x_list[index], ww_test_x_list[index],
                               ww_train_y, ww_valid_y, ww_test_y, "white", bfe_desc)

