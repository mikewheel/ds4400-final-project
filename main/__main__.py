"""
The main driver for running linear regression and classification on wine data

Written by Michael Wheeler and Jay Sherman
"""

import os
import pickle
from argparse import ArgumentParser
from typing import List, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC

from config import INPUT_DATA_DIR, OUTPUT_DATA_DIR


def basis_expansion(data: pd.DataFrame, powers=None) -> pd.DataFrame:
    """
    Performs a basis expansion of arbitrary data features.

    Each feature (column) in data should have a corresponding element
    in powers. If the corresponding value for a feature <= 0
    then the feature will not be included in the basis expansion. If the
    corresponding value for a feature is positive, then every power between
    1 and the that value for the feature will be included in the basis
    expansion. If powers is None (as default), then each feature will not be
    transformed or represent differently in the basis expansion. The last
    element of the basis expansion will always be 1, to account for the
    possibility of a nonzero intercept.

   :param data: the features for all observations. Columns represent features, and rows
        represent observations- comparable to X in the course notes
    :param powers: a list of ints that should be the same length as the number of columns
        in data. Each value specifies how a specific features should be
        represented in the basis expansion, as explained above. If None,
        it is replaced with a list of 1s as long as the number of columns in
        data
    :returns phi : the dataframe that relates the basis expansion for all observations-
        comparable to capital phi in the course notes

    Raises
    ------
    ValueError
        If powers is not the same length as the number of columns in data
    """
    if not powers:
        powers = [1 for i in range(data.shape[1])]
    
    if len(powers) != data.shape[1]:
        print(powers)
        print(data.shape)
        raise ValueError("Length of maximal powers is not equal to number of features.")
    
    phi = pd.DataFrame()
    
    for index, row in data.iterrows():
        new_row = []
        for i in range(len(row)):
            new_row += [pow(row[i], power) for power in range(1, powers[i] + 1)]
        new_row.append(1)
        phi = pd.concat([phi, pd.DataFrame([pd.Series(new_row)], index=[str(i)])],
                        ignore_index=True)
    
    return phi


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separates data into test, training, and validation data sets randomly (deterministic).

    Training and validation data sets are each 45% of the original data set, and the
    test data is 10%. The random selection of data is not stratified based on quality.

    :param data: the data to split into test, train, and validation sets
    :returns: in order, the train, validation, and test data sets
    """
    test_data: pd.DataFrame = data.sample(frac=0.1, random_state=0)
    nontest_data: pd.DataFrame = data[~(data.isin(test_data))].dropna(how="all")
    train_data: pd.DataFrame = nontest_data.sample(frac=0.5, random_state=0)
    validation_data: pd.DataFrame = nontest_data[(nontest_data.isin(train_data))].dropna(how="all")
    return train_data, validation_data, test_data


def run_linear_models_help(train_x: pd.DataFrame, valid_x: pd.DataFrame, test_x: pd.DataFrame,
                           train_y: pd.DataFrame, valid_y: pd.DataFrame, test_y: pd.DataFrame,
                           color: str, bfe_desc: str):
    """Run linear regression to predict the quality of wine based on physicochemical properties.

    The optimal value of theta is found using the training data, then the optimal value of the regularization
    parameter is found by seeing which of {0,0.01,0.1,1,10} yields the lowest validation error. Saves the model to a
    pickle file in the appropriate location inside the output/linear
    directory based on the values of color and bfe_desc, and saves a text file to the same directory
    describing the validation error for each regularization parameter value, which regularization parameter was used,
    the value of theta, the training error, and the test error.

    :param train_x: the input data for the model for the training set
    :param valid_x: the input data for the model for the validation set
    :param test_x: the input data for the model for the test set
    :param train_y: the quality of the wines in the training set
    :param valid_y: the quality of the wines in the validation set
    :param test_y: the quality of the wines in the test set
    :param color: the color of the wine ("red" or "white")
    :param bfe_desc: a description of the basis function expansion used
    """
    models = [[lam, Ridge(random_state=0, alpha=lam, fit_intercept=False, normalize=False)]
              for lam in [0.01, 0.1, 1, 10]]
    models.append([0, LinearRegression()])
    
    for model in models:
        model[1].fit(train_x, train_y)
        theta = model[1].coef_
        model.append(theta)
        error = mean_squared_error(valid_y, model[1].predict(valid_x))
        model.append(error)
    
    models.sort(key=lambda a: a[3], reverse=True)
    best_model = models[0]
    
    train_error = mean_squared_error(train_y, best_model[1].predict(train_x))
    valid_error = mean_squared_error(valid_y, best_model[1].predict(valid_x))
    test_error = mean_squared_error(test_y, best_model[1].predict(test_x))
    
    dir = OUTPUT_DATA_DIR / "linear" / color / bfe_desc
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    
    pickle.dump(best_model, open(dir / "model.p", "wb"))
    
    with open(dir / "log.txt", "w") as f:
        f.write(f"theta: {best_model[2]}")
        f.write(f"training error: {train_error}")
        f.write(f"validation error: {valid_error}")
        f.write(f"test error: {test_error}")
        models.sort(key=lambda a: a[0], reverse=True)
        
        for model in models:
            f.write("")
            f.write(f"lambda: {model[0]}")
            f.write(f"validation error: {model[3]}")


def run_linear_models(rw_train_x_list: List[pd.DataFrame], rw_valid_x_list: List[pd.DataFrame],
                      rw_test_x_list: List[pd.DataFrame], rw_train_y: pd.DataFrame,
                      rw_valid_y: pd.DataFrame, rw_test_y: pd.DataFrame,
                      ww_train_x_list: List[pd.DataFrame], ww_valid_x_list: List[pd.DataFrame],
                      ww_test_x_list: List[pd.DataFrame], ww_train_y: pd.DataFrame,
                      ww_valid_y: pd.DataFrame, ww_test_y: pd.DataFrame):
    """Run linear regression to predict the quality of red and white wine.

    Runs separate models for red and white wine. The optimal value of theta is found using the training data,
    then the optimal value of the regularization parameter is found by seeing which of {0,0.01,0.1,1,10} yields
    the lowest validation error. Saves the model to a pickle file in the appropriate location inside the output/linear
    directory based on the wine color and the basis function expansion, and saves a text file to the same directory
    describing the validation error for each regularization parameter value, which regularization parameter was used,
    the value of theta, the training error, and the test error.

    :param rw_train_x_list: a list of the input features for the training set with basis function expansions applied
                            for red wines
    :param rw_valid_x_list: a list of the input features for the validation set with basis function expansions applied
                            for red wines
    :param rw_test_x_list: a list of the input features for the test set with basis function expansions applied
                           for red wines
    :param rw_train_y: the quality of the wines in the training set for red wines
    :param rw_valid_y: the quality of the wines in the validation set for red wines
    :param rw_test_y: the quality of the wines in the test set for red wines
    :param ww_train_x_list: a list of the input features for the training set with basis function expansions applied
                            for white wines
    :param ww_valid_x_list: a list of the input features for the validation set with basis function expansions applied
                            for white wines
    :param ww_test_x_list: a list of the input features for the test set with basis function expansions applied
                           for white wines
    :param ww_train_y: the quality of the wines in the training set for white wines
    :param ww_valid_y: the quality of the wines in the validation set for white wines
    :param ww_test_y: the quality of the wines in the test set for white wines
    """
    bfe_dict = {1: "base", 2: "fixed_acidity_removed", 3: "volatine_acidity_removed", 4: "citric_acid_removed",
                5: "residual_sugar_removed", 6: "chlorides_removed", 7: "free_sulfur_dioxide_removed",
                8: "total_sulfur_dioxide_removed", 9: "density_removed", 10: "pH_removed", 11: "suplhates_removed",
                12: "alcohol_removed", 13: "fixed_acidity_squared", 14: "volatine_acidity_squared",
                15: "citric_acid_squared",
                16: "residual_sugar_squared", 17: "chlorides_squared", 18: "free_sulfur_dioxide_squared",
                19: "total_sulfur_dioxide_squared", 20: "density_squared", 21: "pH_squared", 22: "suplhates_squared",
                23: "alcohol_squared"}
    
    for i in range(23):
        run_linear_models_help(rw_train_x_list[i], rw_valid_x_list[i], rw_test_x_list[i],
                               rw_train_y, rw_valid_y, rw_test_y, "red", bfe_dict[i + 1])
        run_linear_models_help(ww_train_x_list[i], ww_valid_x_list[i], ww_test_x_list[i],
                               ww_train_y, ww_valid_y, ww_test_y, "white", bfe_dict[i + 1])


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--model", nargs=1, type=str, metavar="model")
    args = parser.parse_args()
    model = args.model[0]
    
    red_wines = pd.read_csv(INPUT_DATA_DIR / "wine_quality_red.csv")
    white_wines = pd.read_csv(INPUT_DATA_DIR / "wine_quality_white.csv")
    rw_train, rw_valid, rw_test = split_data(red_wines)
    ww_train, ww_valid, ww_test = split_data(white_wines)
    rw_train_x, rw_valid_x, rw_test_x = [df.iloc[:, range(11)]
                                         for df in [rw_train, rw_valid, rw_test]]
    ww_train_x, ww_valid_x, ww_test_x = [df.iloc[:, range(11)]
                                         for df in [ww_train, ww_valid, ww_test]]
    rw_train_y, rw_valid_y, rw_test_y = [df.iloc[:, range(11, 12)]
                                         for df in [rw_train, rw_valid, rw_test]]
    ww_train_y, ww_valid_y, ww_test_y = [df.iloc[:, range(11, 12)]
                                         for df in [ww_train, ww_valid, ww_test]]
    
    # the values for exponents for all basis function expansions
    powers_list = [[1 for i in range(11)] for j in range(23)]
    
    # setting 11 of the inner lists to have 0 for a single feature (will remove the feature)
    for i in range(1, 12):
        powers_list[i][i - 1] = 0
    # setting 11 of the inner lists to have 2 for a single feature (x_i and x_i^2 will be present in the bfe)
    for i in range(12, 23):
        powers_list[i][i - 12] = 2
    
    # generating a list of phi(x) for different basis functions
    rw_train_x_list, rw_valid_x_list, rw_test_x_list = [
        [basis_expansion(df, powers)
         for powers in powers_list]
        for df in [rw_train_x, rw_valid_x, rw_test_x]
    ]
    ww_train_x_list, ww_valid_x_list, ww_test_x_list = [
        [basis_expansion(df, powers)
         for powers in powers_list]
        for df in [ww_train_x, ww_valid_x, ww_test_x]
    ]
    
    if model == "logistic":
        model = LogisticRegression("l2", random_state=0)
    elif model == "linear":
        run_linear_models(rw_train_x_list, rw_valid_x_list, rw_test_x_list,
                          rw_train_y, rw_valid_y, rw_test_y,
                          ww_train_x_list, ww_valid_x_list, ww_test_x_list,
                          ww_train_y, ww_valid_y, ww_test_y)
    elif model == "svm":
        model = SVC(kernel="linear", random_state=0)
    else:
        raise ValueError(f'Model type not recognized: "{model}"')
    
    red_X = red_wines.ix[:, range(11)]
    print(red_wines)
    print(red_X)
