import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math


def row_split(row, activity):
    if row[activity + '_walknrun'] == 1 and row[activity + '_walk'] == 0:
        return 1
    return 0


def split_walknrun(x, activities, drop=False):
    x = x.copy()

    for activity in activities:
        x[activity + '_run'] = x.apply(lambda row: row_split(row, activity),
                                       axis=1)

    if drop:
        walknrun_cols = [col for col in x if col.endswith('walknrun')]
        x = x.drop(walknrun_cols, axis=1)

    return x


def to_categorical(x):
    x = x.copy()

    walk_cols = [col for col in x if col.endswith('walk')]
    run_cols = [col for col in x if col.endswith('run')]
    walknrun_cols = [col for col in x if col.endswith('walknrun')]
    activity_names = [*walk_cols, *run_cols, *walknrun_cols]

    activities = x[activity_names]
    x['activity'] = activities.idxmax(axis=1)

    act_map = {name: num for num, name in enumerate(x.activity.unique())}
    x['activity'] = x['activity'].map(act_map)

    x = x.drop(activity_names, axis=1)

    return x, act_map


def drop_activities(x):
    x = x.copy()

    walk_cols = [col for col in x if col.endswith('walk')]
    run_cols = [col for col in x if col.endswith('run')]
    walknrun_cols = [col for col in x if col.endswith('walknrun')]
    activity_names = [*walk_cols, *run_cols, *walknrun_cols]

    x = x.drop(activity_names, axis=1)

    activity_indices = [col for col in x if col.endswith('_index')]
    x = x.drop(activity_indices, axis=1)

    return x


def oversample(x, events, w):
    x = x.copy()

    if w > 1:
        for event in events:
            n = w // 2
            original = x[event].to_numpy()
            orig_indices = np.where(original == 1)[0]
            oversampled = np.zeros_like(original)
            over_indices = np.array([np.arange(index - n, index + n + 1) for index in orig_indices])
            over_indices = over_indices.ravel()
            over_indices = np.clip(over_indices, 0, len(original) - 1)
            oversampled[over_indices] = 1
            x[event] = oversampled

    return x


def calc_virtuals(x):
    x = x.copy()

    positions = ['LF', 'RF', 'Waist', 'Wrist']

    for position in positions:
        X = 'accX_' + position
        Y = 'accY_' + position
        Z = 'accZ_' + position

        x['acc_' + position] = np.sqrt(x[X] ** 2 + x[Y] ** 2 + x[Z] ** 2)
        x['accXY_' + position] = np.sqrt(x[X] ** 2 + x[Y] ** 2)
        x['accYZ_' + position] = np.sqrt(x[Y] ** 2 + x[Z] ** 2)
        x['accXZ_' + position] = np.sqrt(x[X] ** 2 + x[Z] ** 2)

    return x


def filter(x, filter_type, w):
    x = x.copy()
    acc_cols = [col for col in x if col.startswith('acc')]

    if filter_type == 'median':
        for col in acc_cols:
            x[col] = np.convolve(x[col], np.ones(w), 'same') / w

    return x


def scale(x, scaler):
    x = x.copy()
    acc_cols = [col for col in x if col.startswith('acc')]
    acc = x[acc_cols].values

    if scaler == 'MinMax':
        scaler = MinMaxScaler(feature_range=(0, 5))

    elif scaler == 'Standard':
        scaler = StandardScaler()

    acc = scaler.fit_transform(acc)

    x[acc_cols] = acc

    return x


def Xy_split(x, events, time_info=True, has_virtuals=False):
    if x is None:
        return None

    acc = [col for col in x if col.startswith('acc')]
    X, y = x[acc].copy(), x[events].copy()

    # re-ordering
    if has_virtuals:
        X = X[['accX_LF', 'accY_LF', 'accZ_LF', 'acc_LF', 'accXY_LF', 'accYZ_LF', 'accXZ_LF',
               'accX_RF', 'accY_RF', 'accZ_RF', 'acc_RF', 'accXY_RF', 'accYZ_RF', 'accXZ_RF',
               'accX_Waist', 'accY_Waist', 'accZ_Waist', 'acc_Waist', 'accXY_Waist', 'accYZ_Waist', 'accXZ_Waist',
               'accX_Wrist', 'accY_Wrist', 'accZ_Wrist', 'acc_Wrist', 'accXY_Wrist', 'accYZ_Wrist', 'accXZ_Wrist']]

    else:
        X = X[['accX_LF', 'accY_LF', 'accZ_LF',
               'accX_RF', 'accY_RF', 'accZ_RF',
               'accX_Waist', 'accY_Waist', 'accZ_Waist',
               'accX_Wrist', 'accY_Wrist', 'accZ_Wrist']]

    if time_info:
        X['subject'] = x['subject'].values
        y['subject'] = x['subject'].values
        X['activity'] = x['activity'].values
        y['activity'] = x['activity'].values
        X['time'] = x['time'].values
        y['time'] = x['time'].values

    return X, y


def segment(Set, length, stride, target):
    if Set is None:
        return None

    X, y = Set
    X = X.to_numpy()
    y = y.to_numpy().astype(int)

    n_windows = math.ceil((X.shape[0] - length + 1) / stride)
    n_windows = max(0, n_windows)

    X = np.lib.stride_tricks.as_strided(
        X,
        shape=(n_windows, length, X.shape[1]),
        strides=(stride * X.strides[0], X.strides[0], X.strides[1]))

    n_windows = math.ceil((y.shape[0] - length) / stride)
    n_windows = max(0, n_windows)

    y = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_windows, length + 1, y.shape[1]),
        strides=(stride * y.strides[0], y.strides[0], y.strides[1]))

    if target == 'past':
        y = y[:, 0, :]
    elif target == 'present':
        y = y[:, length // 2, :]
    elif target == 'future':
        y = y[:, -1, :]

    return X, y
