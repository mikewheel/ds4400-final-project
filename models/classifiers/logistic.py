"""
Classes for creating and saving logistic regression models.

Written by Michael Wheeler and Jay Sherman
"""

import os
from contextlib import suppress

from sklearn.linear_model import LogisticRegression

from config import OUTPUT_DATA_DIR, make_logger
from models.abc import ModelFactory

logger = make_logger(__name__)


class LogisticModelFactory(ModelFactory):
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
    
    def generate_model_instances(self):
        return [[lam, LogisticRegression(random_state=0, C=(1 / lam if lam != 0 else 0), penalty="l2",
                                         fit_intercept=False, solver="liblinear")]
                for lam in [0.001, 0.01, 0.1, 1, 10]]
