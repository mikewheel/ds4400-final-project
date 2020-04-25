"""
Classes for creating and saving linear regression models.

Written by Michael Wheeler and Jay Sherman
"""
import os
from contextlib import suppress

from sklearn.linear_model import LinearRegression, Ridge

from config import OUTPUT_DATA_DIR, make_logger
from models.abc import ModelFactory

logger = make_logger(__name__)


class LinearRegressionModelFactory(ModelFactory):
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
    
    def generate_model_instances(self):
        output = [[lam, Ridge(random_state=0, alpha=lam, fit_intercept=False, normalize=False)]
                  for lam in [0.01, 0.1, 1, 10]]
        output.append([0, LinearRegression()])
        return output
