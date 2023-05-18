# -*- encoding: utf-8 -*-
"""
@File    :   analysis.py
@Time    :   2023/05/16 18:08:00
@Author  :   UynajGI
@Version :   beta0.0.1
@Contact :   suquan12148@outlook.com
@License :   (MIT)Copyright 2023
"""

# here put the import lib
import numpy as np
from typing import List, Dict


def _rename(column):
    if column == "E" or column == "e":
        column = "energy"
    elif column == "M" or column == "m":
        column = "magnetization"
    return column


def mean(algorithm, uid, column):
    """
    Calculate the mean value of a column of data.
    """
    data = algorithm.iter_data
    column = _rename(column)
    array: np.array = data.loc[uid][column]
    return np.mean(array)


def std(algorithm, uid, column):
    """
    Calculate the standard deviation of a column of data.
    """
    data = algorithm.iter_data
    column = _rename(column)
    array: np.array = data.loc[uid][column]
    return np.std(array)


def cv(algorithm, uid, column):
    """
    Calculate the specific heat of a column of data.
    """
    data = algorithm.iter_data
    column = _rename(column)
    array: np.array = data.loc[uid][column]

    try:
        return np.std(array) / np.mean(array)
    except ZeroDivisionError:
        raise ValueError("The mean value of the data is zero.")


def diff(algorithm, uid, column, n=1):
    """
    Calculate the difference of a column of data.
    """
    data = algorithm.iter_data
    column = _rename(column)
    array: np.array = data.loc[uid][column]
    return np.diff(array, n)


def getcolumn(algorithm, uid, column):
    """
    Get a column of data.
    """
    data = algorithm.iter_data
    column = _rename(column)
    array: np.array = data.loc[uid][column]
    return array


def spin2svd(algorithm, uid) -> float:
    data = algorithm.iter_data
    column: str = 'spin'
    spin_lst = data.loc[uid][column]
    spin_matrix = []
    for spin in spin_lst:
        spin = spin.reshape(-1)
        spin_matrix.append(spin)
    spin_matrix = np.matrix(spin_matrix)
    _, s, _ = np.linalg.svd(spin_matrix, full_matrices=True)
    s_norm = s / np.linalg.norm(s)
    return max(s_norm)


def uid2svd(algorithm: object, uid_lst: List[int] or Dict) -> List[float]:
    if not isinstance(uid_lst, list):
        if isinstance(uid_lst, dict):
            if 'uid' in uid_lst.keys():
                uid_lst = uid_lst['uid']
    svd_lst = []
    for uid in uid_lst:
        svd_lst.append(spin2svd(algorithm, uid))
    return svd_lst
