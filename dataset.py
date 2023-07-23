import numpy as np
from configParser import Parser
import os
from builder import builder
import pandas as pd
from transformers import temporal, binary
import tensorflow as tf
from preprocessing import split_walknrun, to_categorical, drop_activities, \
    oversample, filter, scale, Xy_split, segment, calc_virtuals
from split import split_train_test, split_train_val
import sys


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
            df = df[df['subject'].isin(self.args.subjects)]

        # activity filtering
        if self.args.split_walk_run:
            df = split_walknrun(df, ['treadmill', 'indoor', 'outdoor'], drop=False)

        if self.args.activities != 'all':
            df = df[df[self.args.activities].eq(1).any(axis=1)]
            df, self.act_map = to_categorical(df)
            df = drop_activities(df)

        else:
            df, self.act_map = to_categorical(df)
            df = drop_activities(df)

        # time filtering
        df['time'] = df.pop("time")
        df = df.reset_index(drop=True)

        # oversampling
        if oversampling:
            df = oversample(df, self.args.events, self.args.oversampling)

        # calculate virtual signals
        if self.args.bf_preprocessing:
            df = calc_virtuals(df)

        # filtering
        df = filter(df, self.args.filter, self.args.filter_window)

        # scaling
        df = scale(df, self.args.scaler)

        # train, val, test splitting
        train, test = split_train_test(df, method=self.args.train_test_split, test_fraq=self.args.test_size)

        # X, y splitting
        train = Xy_split(train, self.event_list.keys(), has_virtuals=self.args.bf_preprocessing)
        test = Xy_split(test, self.event_list.keys(), has_virtuals=self.args.bf_preprocessing)

        # segmenting
        train = segment(train, self.args.length, self.args.stride, self.args.target)
        test = segment(test, self.args.length, self.args.stride, self.args.target)

        train, val = split_train_val(train, method=self.args.train_val_split, val_fraq=self.args.val_size)

        self.train_size = train[0].shape[0]
        self.val_size = val[0].shape[0]
        self.test_size = test[0].shape[0]

        if inplace:
            self.train = train
            self.val = val
            self.test = test

        else:
            return train, val, test

    def init_transformers(self):
        self.xTransformer = temporal()
        self.input_shape = self.xTransformer.get_shape()
        self.input_time_shape = self.xTransformer.get_time_shape()
        self.input_type = tf.float32

        self.yTranformer = binary()
        self.output_shape = self.yTranformer.get_shape()
        self.output_type = tf.float32

        self.time_type = (tf.uint8, tf.uint8)
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

        bnp_train = train.cache().shuffle(1000).repeat().batch(batch_size=self.args.batch_size).prefetch(
            tf.data.AUTOTUNE)

        if val is None:
            bnp_val = None
        else:
            bnp_val = val.cache().repeat().batch(batch_size=self.args.batch_size).prefetch(tf.data.AUTOTUNE)

        bnp_test = test.cache().repeat().batch(batch_size=self.args.batch_size).prefetch(tf.data.AUTOTUNE)

        return bnp_train, bnp_val, bnp_test

    def __call__(self, batch_prefetch=True, time_info=False):

        self.init_data()
        _, _, self.test_ = self.init_data(oversampling=False, inplace=False)

        self.init_transformers()

        train = self.to_generator(is_train=True, time_info=time_info)
        val = self.to_generator(is_val=True, time_info=time_info)
        test = self.to_generator(is_test=True, time_info=time_info)

        if batch_prefetch:
            return self.batch_and_prefetch(train, val, test)

        else:
            return train, val, test
