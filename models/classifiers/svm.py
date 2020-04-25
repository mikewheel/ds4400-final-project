"""
Classes for creating and saving support vector machine models.

Written by Michael Wheeler and Jay Sherman.
"""

from contextlib import suppress
from os import mkdir
from pickle import dump

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from config import OUTPUT_DATA_DIR, make_logger
from models.types import ModelFactory
from text_reports.utils import log_classification

logger = make_logger(__name__)


class SupportVectorModelFactory(ModelFactory):
    """
    Factory for support vector machines to classify the wine as white or red. White wine is the positive class,
    red wine is negative.
    
    Finds optimal omega using the training set, then finds optimal value of the regularization parameter out of
    `{0.001, 0.01, 0.1, 1, 10}` using the validation set. Saves coefficients to pickle files under the class's
    `output_root` directory. Produces text reports to record regularization parameter, omega, confusion matrices, and
    derived metrics (e.g. precision, recall).
    """
    output_root = OUTPUT_DATA_DIR / "svm"
    lambdas = (0.001, 0.01, 0.1, 1, 10)
    
    def __init__(self, bfe_desc: str = None, kernel: str = None):
        self.bfe_desc: str = bfe_desc
        self.kernel: str = kernel
        if not self.kernel:
            raise ValueError("Kernel selection is required for SVM!")
        self.ensure_output_dirs_exist()
        self.models = self.generate_model_instances()
    
    def ensure_output_dirs_exist(self) -> None:
        logger.debug("Checking for SVM model output directories...")
        with suppress(FileExistsError):
            mkdir(self.__class__.output_root)
        with suppress(FileExistsError):
            mkdir(self.__class__.output_root / self.kernel)
    
    def generate_model_instances(self):
        return {lambda_: {
            "model": SVC(random_state=0, C=(1 / lambda_ if lambda_ != 0 else 0), kernel=self.kernel, degree=3)}
            for lambda_ in self.__class__.lambdas}
    
    @staticmethod
    def get_coeffs(model):
        if model.kernel == "linear":
            return {"ω": model[1].coef_}
        else:
            return dict()
    
    @staticmethod
    def get_error(model, X, y):
        predictions = model[1].predict(X)
        return sum([1 for _ in range(len(y)) if y[_] != predictions[_]]) / len(predictions)
    
    def best_model(self):
        best_model_key = min(self.models, key=lambda entry: entry["error"])
        return best_model_key, self.models[best_model_key]["model"]
    
    def target_output_dir(self):
        if not self.kernel or not self.bfe_desc:
            raise ValueError(f'Cannot create path when kernel = {self.kernel} and bfe_desc = {self.bfe_desc}')
        return self.__class__.output_root / self.kernel / self.bfe_desc
    
    def report_test_results(self, train_x, valid_x, test_x, train_y, valid_y, test_y):
        lambda_, best_model = self.best_model()
        omega = self.get_coeffs(best_model).get("ω")
        
        target_output_dir = self.target_output_dir()
        with suppress(FileExistsError):
            mkdir(target_output_dir)
        
        logger.info("Writing coefficients to disk...")
        with open(target_output_dir / "model.p", "wb") as f:
            dump(best_model, f)
        
        pred_train = best_model.predict(train_x)
        pred_valid = best_model.predict(valid_x)
        pred_test = best_model.predict(test_x)
        
        train_cm = confusion_matrix(train_y, pred_train)
        valid_cm = confusion_matrix(valid_y, pred_valid)
        test_cm = confusion_matrix(test_y, pred_test)
        
        logger.info("Writing performance report to disk...")
        log_classification(omega, lambda_, train_cm, valid_cm, test_cm, target_output_dir)
