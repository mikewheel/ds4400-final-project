"""
Classes for creating and saving linear regression models.

Written by Michael Wheeler and Jay Sherman
"""
from contextlib import suppress
from os import mkdir
from pickle import dump

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

from config import OUTPUT_DATA_DIR, make_logger
from models.types import ModelFactory
from text_reports.utils import log_linear_regression

logger = make_logger(__name__)


class LinearRegressionModelFactory(ModelFactory):
    """
    Factory for linear regressions to predict the quality of wine based on physical and chemical properties.
    Finds optimal theta using the training set, then finds optimal value of the regularization parameter out of
    `{0, 0.01, 0.1, 1, 10}` using the validation set. Saves coefficients to pickle files under the class's
    `output_root` directory. Produces text reports to record regularization parameter, theta, and loss.
    """
    output_root = OUTPUT_DATA_DIR / "linear"
    lambdas = (0, 0.01, 0.1, 1, 10)
    
    def __init__(self, bfe_desc: str = None, color: str = None):
        self.bfe_desc: str = bfe_desc
        self.color: str = color
        self.ensure_output_dirs_exist()
        self.models = self.generate_model_instances()
    
    def ensure_output_dirs_exist(self) -> None:
        logger.debug("Checking for linear regression model output directories...")
        with suppress(FileExistsError):
            mkdir(self.__class__.output_root)
        with suppress(FileExistsError):
            mkdir(self.__class__.output_root / "red")
        with suppress(FileExistsError):
            mkdir(self.__class__.output_root / "white")
    
    def generate_model_instances(self):
        output = {lambda_: {"model": Ridge(random_state=0, alpha=lambda_, fit_intercept=False, normalize=False)}
                  for lambda_ in self.__class__.lambdas}
        output[0] = {"model": LinearRegression()}
        return output
    
    @staticmethod
    def get_coeffs(model):
        return {"θ": model.coef_}
    
    @staticmethod
    def get_error(model, X, y):
        return mean_squared_error(y, model.predict(X))
    
    def best_model(self):
        best_model_key = min(self.models, key=lambda entry: entry["error"])
        return best_model_key, self.models[best_model_key]
    
    def target_output_dir(self):
        if not self.color or not self.bfe_desc:
            raise ValueError(f'Cannot create path when color = {self.color} and bfe_desc = {self.bfe_desc}')
        return self.__class__.output_root / self.color / self.bfe_desc
    
    def report_test_results(self, train_x, valid_x, test_x, train_y, valid_y, test_y):
        lambda_, best_model = self.best_model()
        theta = self.get_coeffs(best_model)["θ"]
        
        target_output_dir = self.target_output_dir()
        with suppress(FileExistsError):
            mkdir(target_output_dir)

        logger.info("Writing coefficients to disk...")
        with open(target_output_dir / "model.p", "wb") as f:
            dump(best_model, f)

        train_error = self.get_error(best_model, train_x, train_y)
        valid_error = self.get_error(best_model, valid_x, valid_y)
        test_error = self.get_error(best_model, test_x, test_y)

        logger.info("Writing performance report to disk...")
        log_linear_regression(theta, lambda_, train_error, valid_error, test_error, target_output_dir)
