"""
Home of the model factory abstract base class.
"""
from abc import ABC, abstractmethod


class ModelFactory(ABC):
    @abstractmethod
    def ensure_output_dirs_exist(self):
        pass
    
    @abstractmethod
    def generate_model_instances(self):
        pass

    @staticmethod
    @abstractmethod
    def get_coeffs(model):
        pass

    @staticmethod
    @abstractmethod
    def get_error(model, X, y):
        pass
    
    @abstractmethod
    def best_model(self):
        pass
    
    @abstractmethod
    def target_output_dir(self):
        pass
    
    @abstractmethod
    def report_test_results(self, train_x, valid_x, test_x, train_y, valid_y, test_y):
        pass
