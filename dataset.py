import numpy as np
from configParser import Parser
import os
from builder import builder
import pandas as pd
from transformers import temporal, binary
import tensorflow as tf
from preprocessing import split_walknrun, to_categorical, drop_activities, \
    oversample, filter, scale, Xy_split, segment, calc_virtuals, filter_activities, filter_subjects, resampl1d
from split import split_train_test, split_train_val
import sys
import pickle
import matplotlib.pyplot as plt

# user = self.args.subjects[0]
# for position in ['RF', 'LF', 'Waist', 'Wrist']:
#     cols = ['accX_' + position, 'accY_' + position, 'accZ_' + position, 'time']
#     event_cols = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO', 'time']
#     RF_5 = df[df['subject'] == user]
#     RF_5_acc = RF_5[cols]
#     RF_5_events = RF_5[event_cols]
#     start = 5000
#     end = 5400
#
#     plt.style.use("seaborn-dark")
#     fig, ((ax1), (ax3)) = plt.subplots(2, 1)
#     fig.set_size_inches(14.5, 10.5, forward=True)
#     RF_5_acc.iloc[start:end].plot(x='time', lw=1, fontsize=5, ax=ax1, grid=True, linestyle='-')
#     RF_5_events.iloc[start:end].plot(x='time', lw=1, fontsize=5, ax=ax3, grid=True, linestyle='-')
#
#     plt.savefig(str(user) + '_' + position + '_' + self.args.activities[0] + '.png')
#     plt.close(fig)


# user = self.args.subjects[0]
# position = 'Wrist'
#
# cols = ['accX_' + position, 'accY_' + position, 'accZ_' + position, 'time']
# event_cols = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO', 'time']
#
# start = 5000
# end = 5400
#
# plt.style.use("seaborn-dark")
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# fig.set_size_inches(14.5, 10.5, forward=True)
#
# RF_5 = df[df['subject'] == user]
# RF_5_acc = RF_5[cols]
# RF_5_events = RF_5[event_cols]
# RF_5_acc.iloc[start:end].plot(x='time', lw=1, fontsize=5, ax=ax1, grid=True, linestyle='-')
# RF_5_events.iloc[start:end].plot(x='time', lw=1, fontsize=5, ax=ax3, grid=True, linestyle='-')


# RF_5 = df[df['subject'] == user]
# RF_5_acc = RF_5[cols]
# RF_5_events = RF_5[event_cols]
# RF_5_acc.iloc[start:end].plot(x='time', lw=1, fontsize=5, ax=ax2, grid=True, linestyle='-')
# RF_5_events.iloc[start:end].plot(x='time', lw=1, fontsize=5, ax=ax4, grid=True, linestyle='-')
#
# plt.savefig('Rescaling_' + str(user) + '_' + position + '_' + self.args.activities[0] + '.png')


class Dataset:

    def __init__(self, regenerate=False):
        self.act_map = None
        self.test = None
        self.val = None
        self.train = None
        self.test_size = None
        self.val_size = None
        self.train_size = None
        self.time_shape = None
        self.input_time_shape = None
        self.time_type = None
        self.input_type = None
        self.input_shape = None
        self.xTransformer = None
        self.output_type = None
        self.output_shape = None
        self.yTranformer = None

        self.args = Parser()
        self.args.get_args()

        if self.args.dataset == 'MAREA':
            self.path = os.path.join(
                os.path.expanduser('~'),
                self.args.path,
                'MAREA',
            )

            self.event_list = {
                'LF_HS': 0,
                'RF_HS': 1,
                'LF_TO': 2,
                'RF_TO': 3
            }

            self.position_list = {
                'LF': 0,
                'RF': 1,
                'Waist': 2,
                'Wrist': 3
            }

            self.rec_fs = 128

        if regenerate:
            gen = builder()
            self.data = gen()

        else:
            # Read in the data
            data_path = os.path.join(self.path, 'Data_csv format')
            self.data = pd.read_csv(os.path.join(data_path, 'All Subjects' + '.csv'), header=0)

    def init_data(self, oversampling=True, inplace=True):

        df = self.data.copy()
        df = df.drop(df.columns[0], axis=1)

        # subject filtering
        if self.args.subjects != 'all':
            df = filter_subjects(df, self.args.subjects)

        # activity filtering
        if self.args.split_walk_run:
            df = split_walknrun(df, ['treadmill', 'indoor', 'outdoor'], drop=True)

        if self.args.activities != 'all':
            df = filter_activities(df, self.args.activities, self.args.split_walk_run)

        df, self.act_map = to_categorical(df)
        df = drop_activities(df)

        # time filtering
        df['time'] = df.pop("time") / self.rec_fs
        df = df.reset_index(drop=True)

        if self.args.sampling_rate != 128:
            resamplers = self.args.acc_resampler, self.args.event_resampler
            df = resampl1d(df, resamplers, self.rec_fs, self.args.sampling_rate)

        # calculate virtual signals
        if self.args.bf_preprocessing:
            df = calc_virtuals(df)

        # filtering
        if self.args.filter is not None:
            df = filter(df, self.args.filter, self.args.filter_window)

        # scaling
        if self.args.scaler is not None:
            df = scale(df, self.args.scaler)

        # train, val, test splitting
        train, test = split_train_test(df, method=self.args.train_test_split,
                                       test_ratio=self.args.test_size)

        # oversampling
        if oversampling:
            train = oversample(train, self.args.events, self.args.oversampling, self.args.sampling_window)

        if self.args.train_val_split == 'loso':
            train, val = split_train_test(train, method=self.args.train_val_split,
                                          test_ratio=self.args.val_size)
        else:
            val = None

        # X, y splitting
        train = Xy_split(train, self.event_list.keys(), has_virtuals=self.args.bf_preprocessing)
        test = Xy_split(test, self.event_list.keys(), has_virtuals=self.args.bf_preprocessing)
        if val is not None:
            val = Xy_split(val, self.event_list.keys(), has_virtuals=self.args.bf_preprocessing)

        # segmenting
        train = segment(train, self.args.length, self.args.stride, self.args.target, self.args.offset)
        test = segment(test, self.args.length, self.args.stride, self.args.target, self.args.offset)
        if val is not None:
            val = segment(val, self.args.length, self.args.stride, self.args.target, self.args.offset)

        if val is None:
            train, val = split_train_val(train, method=self.args.train_val_split, val_ratio=self.args.val_size)

        self.train_size = train[0].shape[0]
        self.val_size = val[0].shape[0]
        self.test_size = test[0].shape[0]

        if inplace:
            self.train = train
            self.val = val
            self.test = test

        else:
            return train, val, test

    def save_data(self, path):
        save_path = os.path.join(path, 'data.pkl')

        my_data = {'train': self.train,
                   'val': self.val,
                   'test': self.test}
        output = open(save_path, 'wb')
        pickle.dump(my_data, output)

        output.close()

    def load_data(self, path):
        load_path = os.path.join(path, 'data.pkl')

        pkl_file = open(load_path, 'rb')
        my_data = pickle.load(pkl_file)
        self.train = my_data['train']
        self.test = my_data['test']
        self.val = my_data['val']

        self.train_size = self.train[0].shape[0]
        self.val_size = self.val[0].shape[0]
        self.test_size = self.test[0].shape[0]

        pkl_file.close()

    def init_transformers(self):
        self.xTransformer = temporal()
        self.input_shape = self.xTransformer.get_shape()
        self.input_time_shape = self.xTransformer.get_time_shape()
        self.input_type = tf.float32

        self.yTranformer = binary()
        self.output_shape = self.yTranformer.get_shape()
        self.output_type = tf.float32

        self.time_type = (tf.float32, tf.float32)
        self.time_shape = (self.input_time_shape, 3)

    def to_generator(self, is_train=False, is_val=False, is_test=False, time_info=False):
        Set = None

        if is_train:
            Set = self.train
        elif is_val:
            Set = self.val
        elif is_test:
            Set = self.test

        if Set is None:
            return None

        def gen():
            for window, events in zip(Set[0], Set[1]):
                X, XTime = self.xTransformer(window,
                                             training=is_train,
                                             time_info=time_info)

                y, yTime = self.yTranformer(events, time_info=time_info)
                if time_info:
                    yield X, y, (XTime, yTime)

                else:
                    yield X, y

        if time_info:
            return tf.data.Dataset.from_generator(
                gen,
                output_types=(self.input_type,
                              self.output_type,
                              self.time_type),
                output_shapes=(self.input_shape,
                               self.output_shape,
                               self.time_shape)
            )

        else:
            return tf.data.Dataset.from_generator(
                gen,
                output_types=(self.input_type,
                              self.output_type),
                output_shapes=(self.input_shape,
                               self.output_shape)
            )

    def batch_and_prefetch(self, train, val, test):

        bnp_train = (train.
                     cache().
                     shuffle(1000).
                     repeat().
                     batch(batch_size=self.args.batch_size).
                     prefetch(tf.data.AUTOTUNE))

        bnp_val = (val.
                   cache().
                   repeat().
                   batch(batch_size=self.args.batch_size).
                   prefetch(tf.data.AUTOTUNE))

        bnp_test = (test.
                    cache().
                    repeat().
                    batch(batch_size=self.args.batch_size).
                    prefetch(tf.data.AUTOTUNE))

        return bnp_train, bnp_val, bnp_test

    def __call__(self, path, batch_prefetch=True, time_info=False):

        if self.args.load_data:
            self.load_data(path)

        else:
            self.init_data()
            self.save_data(path)

        self.init_transformers()

        train = self.to_generator(is_train=True, time_info=time_info)
        val = self.to_generator(is_val=True, time_info=time_info)
        test = self.to_generator(is_test=True, time_info=time_info)

        if batch_prefetch:
            return self.batch_and_prefetch(train, val, test)

        else:
            return train, val, test
