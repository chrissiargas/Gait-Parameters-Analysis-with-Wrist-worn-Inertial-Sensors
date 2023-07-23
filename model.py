from configParser import Parser
from keras.layers import Input, LSTM, Conv2D, Conv1D, Dense, Dropout
from keras.models import Model
from dataset import Dataset
import os
import shutil
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import binary_accuracy
from metrics import Metrics
from postprocessing import get_eval, get_stats


def get_Tan(input_shape, args: Parser):
    shape = list(input_shape)
    input_tensor = Input(shape=shape)

    X = input_tensor

    LSTM_layer = LSTM(44, return_sequences=True, dropout=0.1, recurrent_dropout=0)
    dense_layer = Dense(44, activation='relu')

    X = LSTM_layer(X)
    X = dense_layer(X)

    LSTM_layer = LSTM(44, return_sequences=False, dropout=0.1, recurrent_dropout=0)
    dense_layer = Dense(44, activation='relu')
    dropout_layer = Dropout(rate=0.5)

    X = LSTM_layer(X)
    X = dense_layer(X)
    X = dropout_layer(X)

    dense_layer = Dense(len(args.events), activation='sigmoid')
    X = dense_layer(X)

    output = X

    return Model(
        inputs=input_tensor,
        outputs=output,
        name='Tan_Model'
    )


def train_evaluate(data: Dataset,
                   summary=False,
                   verbose=0,
                   mVerbose=True):
    args = Parser()
    args.get_args()

    if args.load:
        _, _, test = data()
    else:
        train, val, test = data()

    optimizer = Adam(learning_rate=float(args.learning_rate))
    loss = BinaryCrossentropy()
    metrics = [binary_accuracy]

    model = Model()
    if args.model == 'Tan':
        model = get_Tan(data.input_shape, args)

    if summary and verbose:
        print(model.summary())

    model.compile(optimizer, loss, metrics)

    log_dir = os.path.join('logs', args.model + '_model_tb')
    save_dir = os.path.join('saved', 'saved_models', args.model + '_model')
    model_type = args.model
    model_name = '%s_model.h5' % model_type
    model_dir = os.path.join(save_dir, model_name)

    if args.load:
        if not os.path.isdir(save_dir):
            return

        test_steps = data.test_size // args.batch_size
        test_metrics = Metrics('test', test, test_steps,
                               log_dir, on='test_end', scores=True,
                               tables=mVerbose, average='binary')

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
                                    tables=False, average='binary')

        val_end_metrics = Metrics('val', val, val_steps,
                                  log_dir, on='train_end', scores=True,
                                  tables=True, average='binary')

        test_metrics = Metrics('test', test, test_steps,
                               log_dir, on='test_end', scores=True,
                               tables=mVerbose, average='binary')

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

    eval = get_eval(data, model, args)
    stats = get_stats(eval, args)

    if verbose:
        print(stats)

    del train
    del val
    del test
    del model

    return stats
