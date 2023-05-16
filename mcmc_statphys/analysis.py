# -*- encoding: utf-8 -*-
'''
@File    :   analysis.py
@Time    :   2023/05/16 18:08:00
@Author  :   UynajGI
@Version :   beta0.0.1
@Contact :   suquan12148@outlook.com
@License :   (MIT)Copyright 2023
'''

# here put the import lib
import numpy as np


def _rename(column):
    if column == 'E' or column == 'e':
        column = 'energy'
    elif column == 'M' or column == 'm':
        column = 'magnetization'
    return column


def mean(algorithm, uid, column):
    '''
    Calculate the mean value of a column of data.
    '''
    data = algorithm.iter_data
    column = _rename(column)
    array: np.array = data.loc[uid][column]
    return np.mean(array)


def std(algorithm, uid, column):
    '''
    Calculate the standard deviation of a column of data.
    '''
    data = algorithm.iter_data
    column = _rename(column)
    array: np.array = data.loc[uid][column]
    return np.std(array)


def diff(algorithm, uid, column, n=1):
    '''
    Calculate the difference of a column of data.
    '''
    data = algorithm.iter_data
    column = _rename(column)
    array: np.array = data.loc[uid][column]
    return np.diff(array, n)


def getcolumn(algorithm, uid, column):
    '''
    Get a column of data.
    '''
    data = algorithm.iter_data
    column = _rename(column)
    array: np.array = data.loc[uid][column]
    return array
