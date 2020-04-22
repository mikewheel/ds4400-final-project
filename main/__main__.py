"""
The main driver for running linear regression and classification on wine data

Written by Michael Wheeler and Jay Sherman
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import List
import pandas as pd
from config import INPUT_DATA_DIR
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression


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
        raise ValueError("Length of maximal powers is not equal to number of features.")

    phi = pd.DataFrame()
    
    for index, row in data.iterrows():
        new_row = []
        for i in range(len(row)):
            new_row += [pow(row[i], power) for power in range(1, powers[i]+1)]
        new_row.append(1)
        phi = pd.concat([phi, pd.DataFrame([pd.Series(new_row)], index=[str(i)])],
                        ignore_index=True)
    
    return phi

def split_data(data: pd.DataFrame) -> List[pd.DataFrame]:
    """Separates data into test, training, and validation data sets randomly (deterministic).

    Training and validation data sets are each 45% of the original data set, and the
    test data is 10%. The random selection of data is not stratified based on quality.

    :param data: the data to split into test, train, and validation sets
    :returns: in order, the train, validation, and test data sets
    """
    test_data = data.sample(frac = 0.1, random_state = 0)
    nontest_data = data[~(data.isin(test_data))].dropna(how = "all")
    train_data = nontest_data.sample(frac = 0.5, random_state=0)
    validation_data = nontest_data[(nontest_data.isin(train_data))].dropna(how = "all")
    return train_data, validation_data, test_data


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", nargs=1, type=str, metavar="model")
    args = parser.parse_args()
    model = args.model[0]

    red_wines = pd.read_csv(INPUT_DATA_DIR / "wine_quality_red.csv")
    white_wines = pd.read_csv(INPUT_DATA_DIR / "wine_quality_white.csv")
    rw_train, rw_valid, rw_test = split_data(red_wines)
    ww_train, ww_valid, ww_test = split_data(white_wines)
    rw_train_x, rw_valid_x, rw_test_x = [df.ix[:, range(11)]
                                         for df in [rw_train, rw_valid, rw_test]]
    ww_train_x, ww_valid_x, ww_test_x = [df.ix[:, range(11)]
                                         for df in [ww_train, ww_valid, ww_test]]
    rw_train_y, rw_valid_y, rw_test_y = [df.ix[:, range(11,12)]
                                         for df in [rw_train, rw_valid, rw_test]]
    ww_train_y, ww_valid_y, ww_test_y = [df.ix[:, range(11,12)]
                                         for df in [ww_train, ww_valid, ww_test]]

    powers_list = [[1 for i in range(12)] for j in range(23)] #the values for exponents for all basis function expansions
    #setting 11 of the inner lists to have 0 for a single feature (will remove the feature)
    for i in range(1,12):
        powers_list[i][i - 1] = 0
    #setting 11 of the inner lists to have 2 for a single feature (x_i and x_i^2 will be present in the bfe)
    for i in range(12, 23):
        powers_list[i][i - 12] = 2

    #generating a list of phi(x) for different basis functions
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
        model = LinearRegression(random_state=0)
    elif model == "svm":
        model = SVC(kernel="linear", random_state=0)
    else:
        raise ValueError(f'Model type not recognized: "{model}"')

    red_X = red_wines.ix[:, range(11)]
    print(red_wines)
    print(red_X)
