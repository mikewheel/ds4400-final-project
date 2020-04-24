"""
A file for creating support vector machine models and saving them to file

Written by Michael Wheeler and Jay Sherman
"""

"""
A file for creating logistic regression models and saving them to file

Written by Michael Wheeler and Jay Sherman
"""

import os
import pickle
from typing import List

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from config import OUTPUT_DATA_DIR, BFE_DESCS, make_logger
from logging_models.logging_utils import log_classification
logger = make_logger(__name__)

def run_svm_models_help(train_x: pd.DataFrame, valid_x: pd.DataFrame, test_x: pd.DataFrame,
                        train_y: pd.DataFrame, valid_y: pd.DataFrame, test_y: pd.DataFrame,
                        kernel: str, bfe_desc: str):
    """Run support vector machines to classify the wine as white or red.

    The optimal value of omega is found using the training data,
    then the optimal value of the regularization parameter is found by seeing which of {0.001,0.01,0.1,1,10} yields
    the lowest validation error (as determine by the percent of true positives and true negatives).
    Saves the model to a pickle file in the appropriate location inside the output/logistic
    directory based on the basis function expansion, and saves a text file to the same directory
    describing the confusion matrices and relevant computations (like power and recall) for the training, validation,
    and test data sets, the regularization parameter value, which regularization parameter was used, and
    the value of omega, the training error.

    White wine is considered "positive", and red wine is considered "negative".

    :param train_x: the input features for the training set with basis function expansions applied
    :param valid_x: the input features for the validation set with basis function expansions applied
    :param test_x: the input features for the test set with basis function expansions applied
    :param train_y: the quality of the wines in the training set
    :param valid_y: the quality of the wines in the validation set
    :param test_y: the quality of the wines in the test set
    :param bfe_desc: a description of the basis function expansion
    """
    logger.info(f"Running SVM model with BFE = {bfe_desc}, kernel = {kernel}")
    models = [[lam, SVC(random_state=0, C=(1 / lam if lam != 0 else 0), kernel = kernel, degree = 3)]
              for lam in {0.001,0.01,0.1,1,10}]

    for model in models:
        model[1].fit(train_x, train_y)
        omega = model[1].coef_ if kernel == "linear" else "N/A"
        model.append(omega)
        predictions = model[1].predict(valid_x)
        error = len([i for i in range(len(valid_y)) if valid_y[i] != predictions[i]]) / len(predictions)
        model.append(error)

    models.sort(key=lambda a: a[3], reverse = True)
    best_model = models[0]
    logger.debug(f"Found optimal lambda for SVM model with BFE = {bfe_desc}: {best_model[0]}")

    pred_train = best_model[1].predict(train_x)
    pred_valid = best_model[1].predict(valid_x)
    pred_test = best_model[1].predict(test_x)

    train_cm = confusion_matrix(train_y, pred_train)
    valid_cm = confusion_matrix(valid_y, pred_valid)
    test_cm = confusion_matrix(test_y, pred_test)

    dir = OUTPUT_DATA_DIR / "svm" / kernel / bfe_desc
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass

    pickle.dump(best_model, open(dir / "model.p", "wb"))

    log_classification(best_model[2], best_model[0], train_cm, valid_cm, test_cm, dir)

def run_svm_models(train_x_list: List[pd.DataFrame], valid_x_list: List[pd.DataFrame],
                   test_x_list: List[pd.DataFrame], train_y: pd.DataFrame,
                   valid_y: pd.DataFrame, test_y: pd.DataFrame):
    """Run support vector machines to classify the wine as white or red.

    The optimal value of omega is found using the training data,
    then the optimal value of the regularization parameter is found by seeing which of {0,0.01,0.1,1,10} yields
    the lowest validation error (as determine by the percent of true positives and true negatives).
     Saves the model to a pickle file in the appropriate location inside the output/logistic
    directory based on the basis function expansion, and saves a text file to the same directory
    describing the confusion matrices and relevant computations (like power and recall) for the training, validation,
    and test data sets, the regularization parameter value, which regularization parameter was used, and
    the value of omega, the training error..

    :param train_x_list: a list of the input features for the training set with basis function expansions applied
    :param valid_x_list: a list of the input features for the validation set with basis function expansions applied
    :param test_x_list: a list of the input features for the test set with basis function expansions applied
    :param train_y: the quality of the wines in the training set
    :param valid_y: the quality of the wines in the validation set
    :param test_y: the quality of the wines in the test set
    """

    #making appropriate directories for saving the models to file
    dir = OUTPUT_DATA_DIR / "svm"
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    try:
        os.mkdir(dir / "rbf")
    except FileExistsError:
        pass
    try:
        os.mkdir(dir / "linear")
    except FileExistsError:
        pass
    try:
        os.mkdir(dir / "poly")
    except FileExistsError:
        pass

    for index, bfe_desc in enumerate(BFE_DESCS):
        run_svm_models_help(train_x_list[index], valid_x_list[index], test_x_list[index],
                            train_y, valid_y, test_y, "rbf", bfe_desc)
        run_svm_models_help(train_x_list[index], valid_x_list[index], test_x_list[index],
                            train_y, valid_y, test_y,  "linear", bfe_desc)
        run_svm_models_help(train_x_list[index], valid_x_list[index], test_x_list[index],
                            train_y, valid_y, test_y, "poly", bfe_desc)