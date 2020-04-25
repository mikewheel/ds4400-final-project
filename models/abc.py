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

