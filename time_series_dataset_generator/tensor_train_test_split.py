
from sklearn.model_selection import train_test_split
import numpy as np

def tensor_train_test_split(_x, _y, test_size=0, shuffle=False):
    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []
    for i in np.arange(_x.shape[0]):
        X_train, X_test, y_train, y_test = train_test_split(_x[i, :, :], _y[i, :, :], test_size=test_size, shuffle=shuffle)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    X_train = np.stack(X_trains)
    y_train = np.stack(y_trains)
    X_test = np.stack(X_tests)
    y_test = np.stack(y_tests)
    return X_train, y_train, X_test, y_test