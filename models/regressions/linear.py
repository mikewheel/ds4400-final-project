"""
Classes for creating and saving linear regression models.

Written by Michael Wheeler and Jay Sherman
"""
import os
from contextlib import suppress
from typing import List

import pandas as pd

from config import OUTPUT_DATA_DIR, BFE_DESCS, make_logger

logger = make_logger(__name__)


class LinearRegressionModelFactory:
    """
    Factory for linear regressions to predict the quality of wine based on physical and chemical properties.
    Finds optimal theta using the training set, then finds optimal value of the regularization parameter out of
    `{0, 0.01, 0.1, 1, 10}` using the validation set. Saves coefficients to pickle files under the class's
    `output_root` directory. Produces text reports to record regularization parameter, theta, and loss.
    """
    output_root = OUTPUT_DATA_DIR / "linear"
    
    def __init__(self):
        self.ensure_output_dirs_exist()
    
    def ensure_output_dirs_exist(self):
        logger.debug("Checking for linear regression model output directories...")
        with suppress(FileExistsError):
            os.mkdir(self.__class__.output_root)
        with suppress(FileExistsError):
            os.mkdir(self.__class__.output_root / "red")
        with suppress(FileExistsError):
            os.mkdir(self.__class__.output_root / "white")


def run_linear_models(rw_train_x_bfes: List[pd.DataFrame],
                      rw_valid_x_list: List[pd.DataFrame],
                      rw_test_x_list: List[pd.DataFrame],
                      rw_train_y: pd.DataFrame,
                      rw_valid_y: pd.DataFrame,
                      rw_test_y: pd.DataFrame,
                      ww_train_x_list: List[pd.DataFrame],
                      ww_valid_x_list: List[pd.DataFrame],
                      ww_test_x_list: List[pd.DataFrame],
                      ww_train_y: pd.DataFrame,
                      ww_valid_y: pd.DataFrame,
                      ww_test_y: pd.DataFrame):
    """
    Run linear regression to predict the quality of red and white wine.

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
    
    for index, bfe_desc in enumerate(BFE_DESCS):
        logger.info(f'Training linear regression models with BFE {bfe_desc} on red wine data...')
        run_linear_models_help(rw_train_x_bfes[index], rw_valid_x_list[index], rw_test_x_list[index],
                               rw_train_y, rw_valid_y, rw_test_y, "red", bfe_desc)
        logger.info(f'Training linear regression models with BFE {bfe_desc} on white wine data...')
        run_linear_models_help(ww_train_x_list[index], ww_valid_x_list[index], ww_test_x_list[index],
                               ww_train_y, ww_valid_y, ww_test_y, "white", bfe_desc)
