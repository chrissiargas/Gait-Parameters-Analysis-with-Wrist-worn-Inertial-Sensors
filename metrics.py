import io

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from configParser import Parser
from sklearn.metrics import confusion_matrix, f1_score, recall_score, \
    precision_score, accuracy_score, balanced_accuracy_score, \
    multilabel_confusion_matrix, classification_report
import seaborn as sns
import pandas as pd


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


class Metrics(Callback):
    def __init__(self, name, Set, steps, file_writer,
                 on='epoch_end', scores=False,
                 tables=False, average=None, verbose=0):
        super(Metrics, self).__init__()

        self.args = Parser()
        self.args.get_args()

        self.set = Set
        self.batch_size = self.args.batch_size
        self.steps = steps
        self.verbose = verbose
        self.name = name
        self.on = on

        self.scores = {
            'f1': None,
            'precision': None,
            'recall': None,
            'accuracy': None,
            'balanced': None
        }
        if name == 'val':
            self.in_scores = self.args.val_scores
        elif name == 'test':
            self.in_scores = self.args.test_scores

        self.tables = {
            'confusion': None
        }
        if name == 'val':
            self.in_tables = self.args.val_tables
        elif name == 'test':
            self.in_tables = self.args.test_tables

        self.show_scores = scores
        self.show_plots = tables

        self.file_writer = file_writer
        self.average = average

        if self.args.class_mode == 'multi_label':
            self.label_names = self.args.events
            self.multi_label = True
            self.multi_class = False

        elif self.args.class_mode == 'multi_class':
            self.label_names = ['No Event', *self.args.events]
            self.multi_class = True
            self.multi_label = False

        self.n_labels = len(self.args.events)
        self.length = self.args.length if self.args.target == 'full' else 1

        if self.average is None:
            self.average = None if self.multi_label else 'macro'

    def get_scores(self, true, pred):

        if 'f1' in self.in_scores:
            self.scores['f1'] = f1_score(true, pred,
                                         average=self.average, zero_division=0)

        if 'recall' in self.in_scores:
            self.scores['recall'] = recall_score(true, pred,
                                                 average=self.average, zero_division=0)

        if 'precision' in self.in_scores:
            self.scores['precision'] = precision_score(true, pred,
                                                       average=self.average, zero_division=0)

        if 'accuracy' in self.in_scores:
            self.scores['accuracy'] = accuracy_score(true, pred)

        if 'balanced_accuracy' in self.in_scores:
            self.scores['balanced'] = balanced_accuracy_score(true, pred)

        out = ''
        for metric in self.in_scores:
            if metric == 'classification_report':
                continue

            out += '- ' + self.name + ' ' + metric + ': ' + str(self.scores[metric]) + ' '

        print(out)

        if 'classification_report' in self.in_scores and self.average != 'binary':
            print(classification_report(true, pred, target_names=self.label_names))

    def get_tables(self, true, pred):
        figure = None

        if 'confusion' in self.in_tables:
            cm_writer = tf.summary.create_file_writer(self.file_writer + '/' + self.name + '_cm')
            if self.multi_label:
                self.tables['confusion'] = multilabel_confusion_matrix(true, pred)
                figure, axes = plt.subplots(1, len(self.label_names), figsize=(10, 10))

                if len(self.label_names):
                    axes = [axes]

                for i, (cm, label_name, ax) in enumerate(zip(self.tables['confusion'], self.label_names, axes)):
                    cm_df = pd.DataFrame(cm,
                                         index=['N', 'Y'],
                                         columns=['N', 'Y'])
                    cm_df = cm_df.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    heatmap = sns.heatmap(cm_df, annot=True, cbar=False, ax=ax)
                    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
                    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)

                    ax.set_title('event ' + label_name)
                    if i == 0:
                        ax.set_ylabel('Actual Label')
                    ax.set_xlabel('Predicted Label')

            elif self.multi_class:
                cm = confusion_matrix(true, pred)
                self.tables['confusion'] = cm
                cm_df = pd.DataFrame(cm,
                                     index=self.label_names,
                                     columns=self.label_names)
                cm_df = cm_df.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                figure = plt.figure(figsize=(10, 10))
                sns.heatmap(cm_df, annot=True)
                plt.title('Confusion Matrix')
                plt.ylabel('Actual Values')
                plt.xlabel('Predicted Values')

            figure.tight_layout()
            cm_image = plot_to_image(figure)
            with cm_writer.as_default():
                tf.summary.image('Confusion Matrix', cm_image, step=1)

    def get_metrics(self):
        total_size = self.batch_size * self.steps
        step = 0

        if self.multi_label:
            pred = np.zeros((total_size, self.length, self.n_labels), dtype=np.int8)
            true = np.zeros((total_size, self.length, self.n_labels), dtype=np.int8)

        elif self.multi_class:
            pred = np.zeros((total_size, self.length))
            true = np.zeros((total_size, self.length))

        for batch in self.set.take(self.steps):
            X = batch[0]
            y = batch[1]

            if self.multi_label:
                batch_pred = np.where(
                    np.asarray(self.model.predict(X, verbose=self.verbose)) > 0.5, 1, 0
                )

                if self.length != 1:
                    batch_pred = np.transpose(batch_pred.squeeze(), [1, 2, 0])
                else:
                    batch_pred = batch_pred[..., np.newaxis]

                pred[step * self.batch_size: (step + 1) * self.batch_size] = batch_pred
                true[step * self.batch_size: (step + 1) * self.batch_size] = y

            elif self.multi_class:
                pred[step * self.batch_size: (step + 1) * self.batch_size] = np.argmax(
                    np.asarray(self.model.predict(X, verbose=self.verbose)), axis=1
                )

                true[step * self.batch_size: (step + 1) * self.batch_size] = np.argmax(y, axis=1)

            step += 1

        if self.multi_label and self.length != 1:
            true = true.reshape((total_size * self.length, self.n_labels))
            pred = pred.reshape((total_size * self.length, self.n_labels))

        true = true.squeeze()
        pred = pred.squeeze()

        if self.show_scores:
            self.get_scores(true, pred)

        if self.show_plots:
            self.get_tables(true, pred)

    def on_epoch_end(self, epoch, logs={}):
        if self.on == 'epoch_end':
            self.get_metrics()

        return

    def on_train_end(self, logs={}):
        if self.on == 'train_end':
            self.get_metrics()

        return

    def on_test_end(self, logs={}):
        if self.on == 'test_end':
            self.get_metrics()

        return
