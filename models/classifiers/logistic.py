"""
Classes for creating and saving logistic regression models.

Written by Michael Wheeler and Jay Sherman
"""

import os
from contextlib import suppress
from typing import List

import pandas as pd

from config import OUTPUT_DATA_DIR, BFE_DESCS, make_logger

logger = make_logger(__name__)


class LogisticModelFactory:
    """
    Factory for logistic regressions to classify the wine as white or red. White wine is the positive class,
    red wine is negative.
    
    Finds optimal omega using the training set, then finds optimal value of the regularization parameter out of
    `{0.001, 0.01, 0.1, 1, 10}` using the validation set. Saves coefficients to pickle files under the class's
    `output_root` directory. Produces text reports to record regularization parameter, omega, confusion matrices, and
    derived metrics (e.g. precision, recall).
    """
    output_root = OUTPUT_DATA_DIR / "logistic"
    
    def __init__(self):
        self.ensure_output_dirs_exist()
    
    def ensure_output_dirs_exist(self):
        logger.debug("Checking for logistic regression model output directories...")
        with suppress(FileExistsError):
            os.mkdir(self.__class__.output_root)


def run_logistic_models(train_x_list: List[pd.DataFrame],
                        valid_x_list: List[pd.DataFrame],
                        test_x_list: List[pd.DataFrame],
                        train_y: pd.DataFrame,
                        valid_y: pd.DataFrame,
                        test_y: pd.DataFrame):
    """
    Run logistic regression to classify the wine as white or red.
    
    :param train_x_list: a list of the input features for the training set with basis function expansions applied
    :param valid_x_list: a list of the input features for the validation set with basis function expansions applied
    :param test_x_list: a list of the input features for the test set with basis function expansions applied
    :param train_y: the quality of the wines in the training set
    :param valid_y: the quality of the wines in the validation set
    :param test_y: the quality of the wines in the test set
    """
    
    for index, bfe_desc in enumerate(BFE_DESCS):
        logger.info(f'Training logistic regression models with BFE {bfe_desc}...')
        run_logistic_models_help(train_x_list[index], valid_x_list[index], test_x_list[index],
                                 train_y, valid_y, test_y, bfe_desc)
