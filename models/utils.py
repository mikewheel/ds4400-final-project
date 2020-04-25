"""
Functions common to the execution of all models in the project.

Written by Michael Wheeler and Jay Sherman.
"""
from typing import Tuple

from pandas import DataFrame

from config import BFE_DESCS, make_logger

logger = make_logger(__name__)


def split_data(data: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Separates data into test, training, and validation data sets randomly (deterministic).

    Training and validation data sets are each 45% of the original data set, and the
    test data is 10%. The random selection of data is not stratified based on quality.

    :param data: the data to split into test, train, and validation sets
    :returns: in order, the train, validation, and test data sets
    """
    test_data: DataFrame = data.sample(frac=0.1, random_state=0)
    nontest_data: DataFrame = data[~(data.isin(test_data))].dropna(how="all")
    train_data: DataFrame = nontest_data.sample(frac=0.5, random_state=0)
    validation_data: DataFrame = nontest_data[(nontest_data.isin(train_data))].dropna(how="all")
    return train_data, validation_data, test_data


def run_models_all_bfes(model_factory_class, train_x, valid_x, test_x, train_y, valid_y, test_y, **kwargs):
    """
    Trains and evaluates models across all basis function expansion permutations.
    :param model_factory_class: Utility class for generating the blank models to train.
    :param train_x: Expanded input features for the training set.
    :param valid_x: Expanded input features for the validation set.
    :param test_x: Expanded input features for the test set.
    :param train_y: Response vector for the training set.
    :param valid_y: Response vector for the validation set.
    :param test_y: Response vector for the test set.
    :param kwargs: Anything else that's model-type-specific.
    :return: The best fitting model on that basis function expansion.
    """
    best_models = {}
    for index, bfe_desc in enumerate(BFE_DESCS):
        logger.info(f'Begin training: {model_factory_class}, BFE {bfe_desc}, {kwargs if kwargs else ""}...')
        model_factory = model_factory_class(bfe_desc=bfe_desc, **kwargs)
        best_models[bfe_desc] = run_models_one_bfe(model_factory, train_x[index], valid_x[index], test_x[index],
                                                   train_y, valid_y, test_y)
    return best_models


def run_models_one_bfe(model_factory, train_x, valid_x, test_x, train_y, valid_y, test_y):
    """Trains and evaluates models across a single basis function expansion."""
    for lambda_, model_dict in model_factory.models.items():
        model = model_dict["model"]
        model.fit(train_x, train_y)
        model_factory.models[lambda_]["coeffs"] = model_factory.get_coeffs(model)
        model_factory.models[lambda_]["error"] = model_factory.get_error(model, valid_x, valid_y)
    
    model_factory.report_test_results(train_x, valid_x, test_x, train_y, valid_y, test_y)
    return model_factory.best_model()
