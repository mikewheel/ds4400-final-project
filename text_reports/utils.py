"""
A utilities file for logging the results of models.

Written by Michael Wheeler and Jay Sherman
"""
from pathlib import Path

from numpy import ndarray


def log_classification(omega: float, lam: float, train_cm: ndarray, valid_cm: ndarray, test_cm: ndarray,
                       dir_: Path):
    """Logs the information regarding a logistic regression model.

    Logs the value of omega and the regularization parameter, and the accuracy, sensitivity, specificity, precision,
    and power of the training data, validation data, and test data.

    :param omega: the value of omega for the logistic regression
    :param lam: the value of the regularization parameter for the logistic regression
    :param train_cm: the confusion matrix associated with the training data for the model
    :param valid_cm: the confusion matrix associated with the validation data for the model
    :param test_cm: the confusion matrix associated with the testing data for the model
    :param dir_: the directory to save the log to
    """
    
    with open(dir_ / "log.txt", "w") as f:
        f.write(f"omega: {omega}\n")
        f.write(f"lambda: {lam}\n\n")
        
        f.write("For Training Set\n")
        f.write(f"Accuracy: {(train_cm[0][0] + train_cm[1][1]) / sum([sum(train_cm[i]) for i in range(2)])}\n")
        f.write(f"Sensitivity: {(train_cm[1][1]) / (train_cm[1][0] + train_cm[1][1])}\n")
        f.write(f"Specificity: {(train_cm[0][0]) / (train_cm[0][1] + train_cm[0][0])}\n")
        f.write(f"Precision: {(train_cm[1][1]) / (train_cm[0][1] + train_cm[1][1])}\n")
        f.write(f"Power: {(train_cm[0][0]) / (train_cm[0][0] + train_cm[1][0])}\n\n")
        
        f.write("For Validation Set\n")
        f.write(f"Accuracy: {(valid_cm[0][0] + valid_cm[1][1]) / sum([sum(valid_cm[i]) for i in range(2)])}\n")
        f.write(f"Sensitivity: {(valid_cm[1][1]) / (valid_cm[1][0] + valid_cm[1][1])}\n")
        f.write(f"Specificity: {(valid_cm[0][0]) / (valid_cm[0][1] + valid_cm[0][0])}\n")
        f.write(f"Precision: {(valid_cm[1][1]) / (valid_cm[0][1] + valid_cm[1][1])}\n")
        f.write(f"Power: {(valid_cm[0][0]) / (valid_cm[0][0] + valid_cm[1][0])}\n\n")
        
        f.write("For Test Set\n")
        f.write(f"Validation Accuracy: {(test_cm[0][0] + test_cm[1][1]) / sum([sum(test_cm[i]) for i in range(2)])}")
        f.write(f"\ntest accuracy: {(test_cm[0][0] + test_cm[1][1]) / sum([sum(test_cm[i]) for i in range(2)])}\n")
        f.write(f"Sensitivity: {(test_cm[1][1]) / (test_cm[1][0] + test_cm[1][1])}\n")
        f.write(f"Specificity: {(test_cm[0][0]) / (test_cm[0][1] + test_cm[0][0])}\n")
        f.write(f"Precision: {(test_cm[1][1]) / (test_cm[0][1] + test_cm[1][1])}\n")
        f.write(f"Power: {(test_cm[0][0]) / (test_cm[0][0] + test_cm[1][0])}")


def log_linear_regression(theta: float, lam: float, train_error: float, valid_error: float, test_error: float,
                          dir_: Path):
    """Logs the information for linear regression

    Logs the optimal value for theta, the optimal value of the regression parameter, and the mean squared error for
    the training, validation, and test data sets.

    :param theta: the optimal value of theta for the model
    :param lam: the optimal value of lambda for the model
    :param train_error: the mean-squared error on the training set
    :param valid_error: the mean-squared error on the validation set
    :param test_error: the mean-squared error on the test set
    :param dir_: the directory to save the information to
    """
    
    with open(dir_ / "log.txt", "w") as f:
        f.write(f"theta: {theta}")
        f.write(f"lambda: {lam}")
        f.write(f"Training MSE: {train_error}")
        f.write(f"Validation MSE: {valid_error}")
        f.write(f"Test MSE: {test_error}")
