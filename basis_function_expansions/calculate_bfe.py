"""
A utility for calculating basis function expansions

Written by Mike Wheeler and Jay Sherman
"""
import pandas as pd

from config import make_logger

logger = make_logger(__name__)


def expand_basis(data: pd.DataFrame, powers=None) -> pd.DataFrame:
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
    :returns phi: the dataframe that relates the basis expansion for all observations-
        comparable to capital phi in the course notes

    :raises ValueError: If powers is not the same length as the number of columns in data
    """
    logger.debug(f'BEGIN: calculate basis function expansions with powers {powers}')
    if not powers:
        powers = [1 for i in range(data.shape[1])]
    
    if len(powers) != data.shape[1]:
        logger.debug(powers)
        logger.debyg(data.shape)
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
