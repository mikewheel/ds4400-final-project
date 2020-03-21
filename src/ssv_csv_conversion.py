from argparse import ArgumentParser
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression

def main():
    red_wines = pd.read_csv("wine_quality_red.csv")
    white_wines = pd.read_csv("wine_quality_white.csv")
    phi = basis_expansion(red_wines.head(5))

    parser = ArgumentParser()
    parser.add_argument("--model", nargs = 1, type=str, metavar="model")
    args = parser.parse_args()
    model = args.model[0]
    if model == "logistic":
        model = LogisticRegression("l2", random_state=0)
    elif model == "linear":
        model = LinearRegression(random_state = 0)
    elif model == "svm":
        model = SVC(kernel="linear", random_state=0)
    else:
        raise ValueError("Model type \"" + model + "\"not recognized")

    red_X = red_wines.ix[:, range(11)]
    print(red_wiens)
    print(red_X)
    print(phi)

def basis_expansion(data, powers = None):
    """Returns a basis expansion of input features.

    Each feature (column) in data should have a corresponding element
    in powers. If the corresponding value for a feature <= 0
    then the feature will not be included in the basis expansion. If the
    corresponding value for a feature is positive, then every power between
    1 and the that value for the feature will be included in the basis
    expansion. If powers is None (as default), then each feature will not be
     transformed or represent differently in the basis expansion. The last
     element of the basis expansion will always be 1, to account for the
     possibility of a nonzero intercept.

    Parameters
    ----------
    data : pandas.DataFrame
        The features for all observations. Columns represent features, and rows
        represent observations- comparable to X in the course notes
    powers : list
        A list of ints that should be the same length as the number of columns
        in data. Each value specifies how a specific features should be
        represented in the basis expansion, as explained above. If None,
        it is replaced with a list of 1s as long as the number of columns in
        data

    Returns
    -------
    phi : pandas.DataFrame
        The dataframe that relates the basis expansion for all observations-
        comparable to capital phi in the course notes

    Raises
    ------
    ValueError
        If powers is not the same length as the number of columns in data
    """
    if (powers == None):
        powers = [1 for i in range(data.shape[1])]
    elif (len(powers) != data.shape[1]):
        raise ValueError("Length of maximal powers is not equal to number of features.")

    phi = pd.DataFrame()
    for index, row in data.iterrows():
        new_row = []
        for i in range(len(row)):
            new_row += [pow(row[i], power) for power in range(1, powers[i]+1)]
        new_row.append(1)
        phi = pd.concat([phi, pd.DataFrame([pd.Series(new_row)], index = [str(i)])],
                        ignore_index=True)
    return phi

if __name__ == "__main__":
    main()

