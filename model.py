from configParser import Parser
from keras.layers import Input, LSTM, Conv2D, Conv1D, Dense, Dropout
from keras.models import Model
from dataset import Dataset
import os
import shutil
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.metrics import binary_accuracy
from metrics import Metrics
from postprocessing import get_eval, get_stats
from tcn import TCN, tcn_full_summary
import pandas as pd

def get_Tan(input_shape, args: Parser):
    shape = list(input_shape)
    input_tensor = Input(shape=shape)

    X = input_tensor

    LSTM_layer = LSTM(44, return_sequences=True, dropout=0.1)
    dense_layer = Dense(44, activation='relu')

    X = LSTM_layer(X)
    X = dense_layer(X)

    LSTM_layer = LSTM(44, return_sequences=False, dropout=0.1)
    dense_layer = Dense(44, activation='relu')
    dropout_layer = Dropout(rate=0.5)

    X = LSTM_layer(X)
    X = dense_layer(X)
    X = dropout_layer(X)

    final_layers = []
    outputs = []
    for e, event in enumerate(args.events):
        final_layers.append(Dense(units=1, activation='sigmoid', name=event))
        outputs.append(final_layers[e](X))

    return Model(
        inputs=input_tensor,
        outputs=outputs,
        name='Tan_Model'
    )


def get_Romi(input_shape, args: Parser):
    input_tensor = Input(shape=(None, input_shape))

    X = input_tensor

    tcn_layer = TCN(
        nb_filters=16,
        kernel_size=5,
        dilations=[1, 2, 4],
        padding='same',
        use_batch_norm=True,
        use_skip_connections=True,
        return_sequences=True,
        name='tcn'
    )
    X = tcn_layer(X)

    final_layers = []
    outputs = []
    for e, event in enumerate(args.events):
        final_layers.append(Dense(units=1, activation='sigmoid', name=event))
        outputs.append(final_layers[e](X))

    return Model(
        inputs=input_tensor,
        outputs=outputs,
        name='Romi_Model'
    )


def train_evaluate(data: Dataset,
                   summary=False,
                   verbose=0,
                   mVerbose=True):
    args = Parser()
    args.get_args()

    data_dir = os.path.join('saved', 'saved_data', args.model + '_data')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    if args.load_model:
        _, _, test = data(data_dir)
    else:
        train, val, test = data(data_dir)

    if args.model == 'Tan':
        optimizer = Adam(learning_rate=float(args.learning_rate))
        loss = BinaryCrossentropy()
        metrics = [binary_accuracy]
    elif args.model == 'Romi':
        optimizer = Adam(learning_rate=float(args.learning_rate))
        loss = MeanSquaredError()
        metrics = None

    model = Model()
    if args.model == 'Tan':
        model = get_Tan(data.input_shape, args)
    elif args.model == 'Romi':
        model = get_Romi(data.input_shape[-1], args)

    if summary and verbose:
        print(model.summary())

    model.compile(optimizer, loss, metrics)

    log_dir = os.path.join('logs', args.model + '_model_tb')
    save_dir = os.path.join('saved', 'saved_models', args.model + '_model')
    model_type = args.model
    model_name = '%s_model.h5' % model_type
    model_dir = os.path.join(save_dir, model_name)

    if args.load_model:
        if not os.path.isdir(save_dir):
            return

        test_steps = data.test_size // args.batch_size
        test_metrics = Metrics('test', test, test_steps,
                               log_dir, on='test_end', scores=True,
                               tables=mVerbose, average=args.average)

    else:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        try:
            shutil.rmtree(log_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        try:
            os.remove(model_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        train_steps = data.train_size // args.batch_size
        val_steps = data.val_size // args.batch_size
        test_steps = data.test_size // args.batch_size

        val_epoch_metrics = Metrics('val', val, val_steps,
                                    log_dir, on='epoch_end', scores=True,
                                    tables=False, average=args.average)

        val_end_metrics = Metrics('val', val, val_steps,
                                  log_dir, on='train_end', scores=True,
                                  tables=True, average=args.average)

        test_metrics = Metrics('test', test, test_steps,
                               log_dir, on='test_end', scores=True,
                               tables=mVerbose, average=args.average)

        tensorboard_callback = TensorBoard(log_dir, histogram_freq=1)

        save_model = ModelCheckpoint(
            filepath=model_dir,
            monitor='val_loss',
            verbose=verbose,
            save_best_only=True,
            mode='min',
            save_weights_only=True)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=5,
            mode='auto',
            verbose=verbose)

        if mVerbose:
            callbacks = [
                tensorboard_callback,
                save_model,
                early_stopping,
                val_epoch_metrics,
                val_end_metrics
            ]
        else:
            callbacks = [tensorboard_callback,
                         save_model,
                         early_stopping]

        model.fit(train,
                  epochs=args.epochs,
                  steps_per_epoch=train_steps,
                  validation_data=val,
                  validation_steps=val_steps,
                  callbacks=callbacks,
                  use_multiprocessing=True,
                  verbose=verbose)

    if mVerbose:
        callbacks = [test_metrics]
    else:
        callbacks = []

    model.load_weights(model_dir)
    model.evaluate(test, steps=test_steps, callbacks=callbacks)

    pd.set_option('display.max_columns', None)
    eval = get_eval(data, model, args)
    print(eval)
    stats = get_stats(eval, args)

    if verbose:
        print()
        print(stats)

    del train
    del val
    del test
    del model

    return stats, eval
