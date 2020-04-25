"""
Functions common to the execution of all models in the project.

Written by Michael Wheeler and Jay Sherman.
"""
from contextlib import suppress
from os import mkdir
from pickle import dump
from typing import Tuple

from pandas import DataFrame

from config import BFE_DESCS, make_logger
from models.abc import ModelFactory

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


def run_models_all_bfes(model_factory: ModelFactory,
                        train_x: DataFrame,
                        valid_x: DataFrame,
                        test_x: DataFrame,
                        train_y: DataFrame,
                        valid_y: DataFrame,
                        test_y: DataFrame,
                        **kwargs):
    for index, bfe_desc in enumerate(BFE_DESCS):
        logger.info(f'Begin training: Model {None}, BFE {bfe_desc}, {kwargs if kwargs else ""}...')
        run_models(model_factory, train_x, valid_x, test_x, train_y, valid_y, test_y, bfe_desc, **kwargs)


def run_models(model_factory: ModelFactory,
               train_x: DataFrame,  # FIXME include model factory class
               valid_x: DataFrame,
               test_x: DataFrame,
               train_y: DataFrame,
               valid_y: DataFrame,
               test_y: DataFrame,
               bfe_desc: str,
               **kwargs):
    """
    TODO write me
    :param model_factory: Utility class for generating the blank models to train.
    :param train_x: Expanded input features for the training set.
    :param valid_x: Expanded input features for the validation set.
    :param test_x: Expanded input features for the test set.
    :param train_y: Response vector for the training set
    :param valid_y: Response vector for the validation set
    :param test_y: Response vector for the test set
    :param bfe_desc: A description of the basis function expansion
    :param kwargs: anything else that's model-type-specific
    :return: The best fitting model on that basis function expansion.
    """
    color = None  # FIXME linear kwarg
    kernel = None  # FIXME SVM kwarg
    
    models = model_factory.generate_model_instances()
    
    logger.debug(f'Training and evaluating {len(models)} model-hyperparameter combos...')
    for index, model in enumerate(models):
        model[1].fit(train_x, train_y)
        theta = None
        omega = None
        # Linear theta = model[1].coef_
        # Logistic omega = model[1].coef_
        # SVM omega = model[1].coef_ if kernel == "linear" else "N/A"
        error = None
        # Linear
        # error = mean_squared_error(valid_y, model[1].predict(valid_x))
        # Logistic
        # predictions = model[1].predict(valid_x)
        # error = len([i for i in range(len(valid_y)) if valid_y[i] != predictions[i]]) / len(predictions)  # FIXME
        # SVM
        # predictions = model[1].predict(valid_x)
        # error = len([i for i in range(len(valid_y)) if valid_y[i] != predictions[i]]) / len(predictions)
        
        models[index].append(theta)
        models[index].append(error)

    models.sort(key=lambda a: a[3])  # TODO -- check reverse between logistic and SVM
    best_model = models[0]
    logger.debug(f"Found optimal lambda for model with BFE = {bfe_desc}: {best_model[0]}")
    
    """ Linear
    train_error = mean_squared_error(train_y, best_model[1].predict(train_x))
    valid_error = mean_squared_error(valid_y, best_model[1].predict(valid_x))
    test_error = mean_squared_error(test_y, best_model[1].predict(test_x))
    """
    
    """ Logistic
    pred_train = best_model[1].predict(train_x)
    pred_valid = best_model[1].predict(valid_x)
    pred_test = best_model[1].predict(test_x)
    
    train_cm = confusion_matrix(train_y, pred_train)
    valid_cm = confusion_matrix(valid_y, pred_valid)
    test_cm = confusion_matrix(test_y, pred_test)
    """
    
    """ SVM
    pred_train = best_model[1].predict(train_x)
    pred_valid = best_model[1].predict(valid_x)
    pred_test = best_model[1].predict(test_x)
    
    train_cm = confusion_matrix(train_y, pred_train)
    valid_cm = confusion_matrix(valid_y, pred_valid)
    test_cm = confusion_matrix(test_y, pred_test)
    """
    
    target_output_dir = None
    # Linear: target_output_dir = output_root / color / bfe_desc
    # Logistic: target_output_dir = output_root / bfe_desc
    # SVM: target_output_dir = output_root / kernel / bfe_desc
    with suppress(FileExistsError):
        mkdir(target_output_dir)

    logger.info("Writing coefficients to disk...")
    with open(target_output_dir / "model.p", "wb") as f:
        # Linear: dump(best_model[1], f)
        # Logistic: dump(best_model, f)
        # SVM: pickle.dump(best_model, f)
        dump(None, f)

    logger.info("Writing performance report to disk...")
    # Linear
    # log_linear_regression(best_model[2], best_model[0], train_error, valid_error, test_error, target_output_dir)
    # Logistic
    # log_classification(best_model[2], best_model[0], train_cm, valid_cm, test_cm, target_output_dir)
    # SVM
    # log_classification(best_model[2], best_model[0], train_cm, valid_cm, test_cm, target_output_dir)
    
    return best_model
