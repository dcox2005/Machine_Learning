from unittest import loader
import numpy as np
import pandas as pd
import os

from numpy.random import rand as rand
from numpy.random import seed as seed

from code_linear_regression import linear_regression as LR
from misc.utils import MyUtils


def main():
    (X_train, y_train, X_test, y_test) = loadData()
    testCF(X_train, y_train, X_test, y_test)
    testGD(X_train, y_train, X_test, y_test)


def testGD(X_train, y_train, X_test, y_test):
    errors = loadErrors('GD_Error.npz')
    threshold = 1
    row = 0

    for lam in [0, 10]:
        for z_r in [1, 2]:
            for eta in [0.01, 0.001]:
                # Create a new lr object each time. No assumption made that the weights will reset.
                lr = LR.LinearRegression()
                lr.fit(X_train, y_train, CF=False, lam=lam,
                       eta=eta, epochs=10000, degree=z_r)
                train_error = lr.error(X_train, y_train)
                test_error = lr.error(X_test, y_test)

                (mikes_train_error, mikes_test_error) = errors[row]
                row += 1
                if abs(mikes_test_error - test_error) > threshold or abs(mikes_train_error - train_error) > threshold:
                    print(
                        f'For GD, and the following params:\nlam: {lam}, z_r: {z_r}, eta: {eta}')
                    print(
                        f'Expected train/test error:\n{mikes_train_error}, {mikes_test_error}')
                    print(
                        f'Found train/test error:\n{train_error}, {test_error}\n')
    print("Please note that for Gradient descent, 0 initialization of the weights is assumed.")


def testCF(X_train, y_train, X_test, y_test):
    errors = loadErrors('CF_Error.npz')
    threshold = 1
    row = 0

    for lam in [0, 0.1]:
        for z_r in [1, 2, 4]:
            # Create a new lr object each time. No assumption made that the weights will reset.
            lr = LR.LinearRegression()
            lr.fit(X_train, y_train, CF=True, lam=lam,
                   eta=0.01, epochs=1000, degree=z_r)
            train_error = lr.error(X_train, y_train)
            test_error = lr.error(X_test, y_test)

            (mikes_train_error, mikes_test_error) = errors[row]
            # mikes_test_error += 2
            row += 1
            if abs(mikes_test_error - test_error) > threshold or abs(mikes_train_error - train_error) > threshold:
                print(
                    f'For CF, and the following params:\nlam: {lam}, z_r: {z_r}')
                print(
                    f'Expected train/test error:\n{mikes_train_error}, {mikes_test_error}')
                print(
                    f'Found train/test error:\n{train_error}, {test_error}\n')


def loadErrors(file):
    container = np.load(file)
    data = [container[key] for key in container]
    errors = np.array(data)
    return errors


def loadData():
    # Reads the files into pandas dataframes from the respective .csv files.
    df_X_train = pd.read_csv(os.path.join(
        'code_linear_regression', 'houseprice', 'x_train.csv'), header=None)
    df_y_train = pd.read_csv(os.path.join(
        'code_linear_regression', 'houseprice', 'y_train.csv'), header=None)
    df_X_test = pd.read_csv(os.path.join(
        'code_linear_regression', 'houseprice', 'x_test.csv'), header=None)
    df_y_test = pd.read_csv(os.path.join(
        'code_linear_regression', 'houseprice', 'y_test.csv'), header=None)

    # Convert the input data into numpy arrays and normalize.
    X_train = df_X_train.to_numpy()
    X_test = df_X_test.to_numpy()
    n_train = X_train.shape[0]

    X_all = MyUtils.normalize_0_1(np.concatenate((X_train, X_test), axis=0))
    X_train = X_all[:n_train]
    X_test = X_all[n_train:]

    y_train = df_y_train.to_numpy()
    y_test = df_y_test.to_numpy()

    # Insure that the data correctly loaded in.
    assert X_train.shape == (
        404, 13), "Incorrect input, expected (404, 13), found " + X_train.shape
    assert y_train.shape == (
        404, 1), "Incorrect input, expected (404, 1), found " + y_train.shape
    assert X_test.shape == (
        102, 13), "Incorrect input, expected (102, 13), found " + X_test.shape
    assert y_test.shape == (
        102, 1), "Incorrect input, expected (102, 1), found " + y_test.shape

    return (X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
