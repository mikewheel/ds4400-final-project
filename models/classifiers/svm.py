"""
Classes for creating and saving support vector machine models.

Written by Michael Wheeler and Jay Sherman.
"""

import os
from contextlib import suppress
from typing import List

import pandas as pd

from config import OUTPUT_DATA_DIR, BFE_DESCS, make_logger

logger = make_logger(__name__)


class SupportVectorModelFactory:
    """
    Factory for support vector machines to classify the wine as white or red. White wine is the positive class,
    red wine is negative.
    
    Finds optimal omega using the training set, then finds optimal value of the regularization parameter out of
    `{0.001, 0.01, 0.1, 1, 10}` using the validation set. Saves coefficients to pickle files under the class's
    `output_root` directory. Produces text reports to record regularization parameter, omega, confusion matrices, and
    derived metrics (e.g. precision, recall).
    """
    output_root = OUTPUT_DATA_DIR / "svm"
    
    def __init__(self):
        self.ensure_output_dirs_exist()
    
    def ensure_output_dirs_exist(self):
        logger.debug("Checking for SVM model output directories...")
        with suppress(FileExistsError):
            os.mkdir(self.__class__.output_root)
        with suppress(FileExistsError):
            os.mkdir(self.__class__.output_root / "rbf")
        with suppress(FileExistsError):
            os.mkdir(self.__class__.output_root / "linear")
        with suppress(FileExistsError):
            os.mkdir(self.__class__.output_root / "poly")
