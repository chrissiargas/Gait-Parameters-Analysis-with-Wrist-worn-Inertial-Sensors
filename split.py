import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(x, method, test_fraq):
    # split into train and test sets
    train, test = None, None
    test_fraq *= 0.01
    test_size = int(len(x) * test_fraq)
    train_size = len(x) - test_size

    if method == 'start':
        train, test = x.iloc[:train_size, :], x.iloc[train_size:, :]

    elif method == 'middle':
        train_1, train_2 = x.iloc[:train_size//2, :], x.iloc[-train_size//2:, :]
        train = pd.concat([train_1, train_2])
        test = x.iloc[train_size//2: -train_size//2, :]

    elif method == 'end':
        train, test = x.iloc[test_size:, :], x.iloc[:test_size, :]

    elif method == 'loso':
        pass

    return train, test


def split_train_val(z, method, val_fraq):
    # split into train and test sets
    X, y = z
    X = X[:-1]

    train, val = None, None
    val_fraq *= 0.01

    val_size = int(X.shape[0] * val_fraq)
    train_size = X.shape[0] - val_size

    if method == 'start':
        train = (X[:train_size], y[:train_size])
        val = (X[train_size:], y[train_size:])

    elif method == 'middle':
        train = (X.r_[:train_size//2, -train_size//2:], y.r_[:train_size//2, -train_size//2:])
        val = (X[train_size//2: -train_size//2], y[train_size//2: -train_size//2])

    elif method == 'end':
        train = (X[val_size:], y[val_size:])
        val = (X[:val_size], y[:val_size])

    elif method == 'random':
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_fraq)
        train = (X_train, y_train)
        val = (X_val, y_val)

    elif method == 'none':
        train, val = (X, y), None

    return train, val


