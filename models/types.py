"""
Home of the model factory abstract base class.
"""
from abc import ABC, abstractmethod


class ModelFactory(ABC):
    @abstractmethod
    def ensure_output_dirs_exist(self):
        """Checks that the directory structure for saving model results exists on disk."""
        pass
    
    @abstractmethod
    def generate_model_instances(self):
        """Enumerates instances of model classes based on hyperparameter.
        :return: A dictionary mapping lambda to sklearn model (and later coeffs and error).
        """
        pass

    @staticmethod
    @abstractmethod
    def get_coeffs(model):
        """Provides the coefficients of the fully-trained model.
        :param model: A trained model of the same type as the factory itself.
        :return: An array of floats.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_error(model, X, y):
        """Calculates error on a trained model according to a metric appropriate for this model type.
        :param model: A trained model of the same type as the factory itself.
        :param X: A matrix of explanatory variables.
        :param y: A vector of response variables.
        :return: A float.
        """
        pass
    
    @abstractmethod
    def best_model(self):
        """Returns the trained model with the lowest computed error out of all of `self.models`
        :return: A trained model of the same type as the factory itself."""
        pass
    
    @abstractmethod
    def target_output_dir(self):
        """Generates an appropriate subdirectory under the output dirs based on model hyperparameters and factory
        attributes.
        :return: A pathlib Path under `self.output_root`.
        """
        pass
    
    @abstractmethod
    def report_test_results(self, train_x, valid_x, test_x, train_y, valid_y, test_y):
        """Serializes `self.best_model()` and writes it to disk. Computes model performance metrics across train,
        validation, and test sets. Produces report text and plots if applicable."""
        pass
