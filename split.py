import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def split_train_test(x, method, test_ratio):
    # split into train and test sets
    train, test = None, None
    test_ratio *= 0.01

    if method == 'start':
        test_size = int(len(x) * test_ratio)
        train_size = len(x) - test_size

        train, test = x.iloc[:train_size, :], x.iloc[train_size:, :]

    elif method == 'middle':
        test_size = int(len(x) * test_ratio)
        train_size = len(x) - test_size

        train_1, train_2 = x.iloc[:train_size//2, :], x.iloc[-train_size//2:, :]
        train = pd.concat([train_1, train_2])
        test = x.iloc[train_size//2: -train_size//2, :]

    elif method == 'end':
        test_size = int(len(x) * test_ratio)

        train, test = x.iloc[test_size:, :], x.iloc[:test_size, :]

    elif method == 'loso':
        subjects = x['subject'].unique()
        n_subjects = len(subjects)

        n_test_subjects = int(n_subjects * test_ratio)

        shuffled_subjects = subjects.copy()
        np.random.shuffle(shuffled_subjects)

        test_subjects = shuffled_subjects[:n_test_subjects]
        train_subjects = shuffled_subjects[n_test_subjects:]

        test = x[x['subject'].isin(test_subjects)]
        train = x[x['subject'].isin(train_subjects)]

    return train, test


def split_train_val(z, method, val_ratio):
    # split into train and test sets
    X, y = z

    train, val = None, None
    val_ratio *= 0.01

    if method == 'start':
        val_size = int(X.shape[0] * val_ratio)
        train_size = X.shape[0] - val_size

        train = (X[:train_size], y[:train_size])
        val = (X[train_size:], y[train_size:])

    elif method == 'middle':
        val_size = int(X.shape[0] * val_ratio)
        train_size = X.shape[0] - val_size

        train = (X.r_[:train_size//2, -train_size//2:], y.r_[:train_size//2, -train_size//2:])
        val = (X[train_size//2: -train_size//2], y[train_size//2: -train_size//2])

    elif method == 'end':
        val_size = int(X.shape[0] * val_ratio)

        train = (X[val_size:], y[val_size:])
        val = (X[:val_size], y[:val_size])

    elif method == 'random':
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio)
        train = (X_train, y_train)
        val = (X_val, y_val)

    elif method == 'none':
        train, val = (X, y), None

    return train, val


