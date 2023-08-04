import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
from scipy.signal import resample
import pandas as pd
from scipy.interpolate import interp1d
from scipy import ndimage
from math import sqrt, log

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


def filter_subjects(x, subjects):
    x = x.copy()

    if subjects == 'indoors':
        indoor_subs = [*range(1, 12)]
        x = x[x['subject'].isin(indoor_subs)]
    elif subjects == 'outdoors':
        outdoor_subs = [*range(12, 21)]
        x = x[x['subject'].isin(outdoor_subs)]
    else:
        x = x[x['subject'].isin(subjects)]

    return x


def filter_activities(x, activities, split_wnr):
    x = x.copy()

    if split_wnr:
        activities_ = []
        for a, activity in enumerate(activities):
            if activity.endswith('walknrun'):
                walk = activity.replace('walknrun', 'walk')
                run = activity.replace('walknrun', 'run')
                activities_.extend([walk, run])

            else:
                activities_.append(activity)

    else:
        activities_ = activities

    x = x[x[activities_].eq(1).any(axis=1)]
    return x


def to_categorical(x):
    x = x.copy()

    walk_cols = [col for col in x if col.endswith('walk')]
    run_cols = [col for col in x if col.endswith('run')]
    walknrun_cols = [col for col in x if col.endswith('walknrun')]
    activity_names = [*walk_cols, *run_cols, *walknrun_cols]

    activities = x[activity_names]
    x['activity'] = activities.idxmax(axis=1)
    x.loc[~activities.any(axis='columns'), 'activity'] = 'undefined'

    act_map = {name: num for num, name in enumerate(x.activity.unique())}
    x['activity'] = x['activity'].map(act_map)

    x = x.drop(activity_names, axis=1)
    if 'undefined' in act_map.keys():
        x = x[x['activity'] != 'undefined']

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


def oversample(x, events, method, w):
    x = x.copy()

    if w > 1:
        for event in events:
            if method == 'simple':
                n = w // 2
                original = x[event].to_numpy()
                orig_indices = np.where(original == 1)[0]
                oversampled = np.zeros_like(original)
                over_indices = np.array([np.arange(index - n, index + n + 1) for index in orig_indices])
                over_indices = over_indices.ravel()
                over_indices = np.clip(over_indices, 0, len(original) - 1)
                oversampled[over_indices] = 1
                x[event] = oversampled

            elif method == 'gaussian':
                fwhm = w - 1
                sigma = fwhm / (2 * math.sqrt(2 * log(2)))
                original = x[event].to_numpy()
                oversampled = ndimage.gaussian_filter1d(np.float_(original), sigma)
                oversampled /= oversampled.max()
                x[event] = oversampled

    return x


def resample_acc(segment_df, old_fs, new_fs, how='simple'):
    N = segment_df.shape[0]
    cols = [col for col in segment_df if col.startswith('acc')]

    fs_scale = new_fs / old_fs
    new_samples = int(N * fs_scale)

    if how == 'simple':
        resampled_acc = pd.DataFrame(columns=cols)
        for col in cols:
            resampled_acc[col] = resample(segment_df[col], new_samples)

    elif how == 'interp_linear':
        t = np.arange(N) / old_fs
        f = interp1d(t, segment_df[cols], kind='linear', axis=0, fill_value='extrapolate')
        tq = np.arange(new_samples) / new_fs
        resampled_acc = f(tq)
        resampled_acc = pd.DataFrame(resampled_acc, columns=cols)

    elif how == 'interp_nearest':
        t = np.arange(N) / old_fs
        f = interp1d(t, segment_df[cols], kind='nearest', axis=0, fill_value='extrapolate')
        tq = np.arange(new_samples) / new_fs
        resampled_acc = f(tq)
        resampled_acc = pd.DataFrame(resampled_acc, columns=cols)

    return resampled_acc


def resample_events(segment_df, old_fs, new_fs, how='simple'):
    N = segment_df.shape[0]
    cols = [col for col in segment_df if col.startswith('LF') or col.startswith('RF')]

    fs_scale = new_fs / old_fs
    new_samples = int(segment_df.shape[0] * fs_scale)

    if how == 'simple':
        resampled_events = pd.DataFrame(columns=cols)
        for col in cols:
            resampled = resample(segment_df[col], new_samples)
            resampled_events[col] = np.where(resampled > 0.5, 1, 0)

    elif how == 'interp_linear':
        t = np.arange(N) / old_fs
        f = interp1d(t, segment_df[cols], kind='linear', axis=0, fill_value='extrapolate')
        tq = np.arange(new_samples) / new_fs
        resampled_events = f(tq)
        resampled_events = np.where(resampled_events > 0.5, 1, 0)
        resampled_events = pd.DataFrame(resampled_events, columns=cols)

    elif how == 'interp_nearest':
        t = np.arange(N) / old_fs
        f = interp1d(t, segment_df[cols].values, kind='nearest', axis=0, fill_value='extrapolate')
        tq = np.arange(new_samples) / new_fs
        resampled_events = f(tq)
        resampled_events = pd.DataFrame(resampled_events, columns=cols)

    elif how == 'preserve':
        resampled_events = pd.DataFrame(columns=cols)
        for col in cols:
            ixs = np.where(segment_df[col])[0]
            new_ixs = (ixs * fs_scale).astype(int)
            new_evs = np.zeros(new_samples)
            new_evs[new_ixs] = 1.
            resampled_events[col] = new_evs

    return resampled_events


def resample_time(segment_df, sub_act, old_fs, new_fs, start):
    subject, activity = sub_act

    fs_scale = new_fs / old_fs

    resampled_time = pd.DataFrame(columns=['subject', 'activity', 'time'])
    new_t = start + np.arange(int(segment_df.shape[0] * fs_scale)) / new_fs
    resampled_time['time'] = new_t
    resampled_time['subject'] = subject
    resampled_time['activity'] = activity

    return resampled_time


def resampl1d(x, resamplers, old_fs, new_fs):
    x = x.copy()

    acc_resampler, event_resampler = resamplers
    sessions = x.groupby(['subject', 'activity'])

    resampled_acc = [resample_acc(session, old_fs, new_fs, acc_resampler) for _, session in sessions]
    acc_df = pd.concat(resampled_acc, ignore_index=True)

    resampled_events = [resample_events(session, old_fs, new_fs, event_resampler) for _, session in sessions]
    events_df = pd.concat(resampled_events, ignore_index=True)

    resampled_time = [resample_time(session, sna, old_fs, new_fs, session.iloc[0]['time']) for sna, session in sessions]
    time_df = pd.concat(resampled_time, ignore_index=True)

    x = pd.concat([acc_df, events_df, time_df], axis=1)

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


def segment(Set, length, stride, target, distance=0):
    if Set is None:
        return None

    X, y = Set
    X = X.to_numpy()
    y = y.to_numpy().astype(float)

    n_windows = math.ceil((X.shape[0] - length + 1) / stride)
    n_windows = max(0, n_windows)

    X = np.lib.stride_tricks.as_strided(
        X,
        shape=(n_windows, length, X.shape[1]),
        strides=(stride * X.strides[0], X.strides[0], X.strides[1]))

    offset = 0
    if target == 'future':
        offset = distance
        X = X[:-offset]
    elif target == 'past':
        offset = distance
        X = X[offset:]

    n_windows = math.ceil((y.shape[0] - length - offset + 1) / stride)
    n_windows = max(0, n_windows)

    y = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_windows, length + offset, y.shape[1]),
        strides=(stride * y.strides[0], y.strides[0], y.strides[1]))

    start = y[:, 0, -3:-1]
    end = y[:, -1, -3:-1]
    transitions = np.logical_or.reduce(start != end, axis=1)

    X = X[~transitions]
    y = y[~transitions]

    if target == 'past':
        y = y[:, [0], :]
    elif target == 'present':
        y = y[:, [length // 2], :]
    elif target == 'future':
        y = y[:, [-1], :]
    elif target == 'full':
        pass

    return X, y
