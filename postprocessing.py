from dataset import Dataset
from model import Model
from configParser import Parser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_yy_(data: Dataset, model: Model):
    windows, events = data.test

    N = len(events)
    length = 0
    true_sequence = []
    true = []
    predicted = []
    lengths = []
    inputs = []

    threshold = 2
    for index, event in enumerate(events):

        X, _ = data.xTransformer(windows[index], training=False)
        inputs.append(X)
        y, info = data.yTranformer(event, time_info=True)
        y, info = y[0], info[0]
        true_sequence.append(y)
        length += 1

        subject, activity, time = info
        if index == N - 1 or events[index + 1, 0, -1] - time > threshold or \
                events[index + 1, 0, -2] != activity or \
                events[index + 1, 0, -3] != subject:

            if length != 0:
                outputs = model.predict(np.array(inputs), verbose=0)
                true.extend(true_sequence)
                predicted.extend(outputs)
                lengths.append(length)

            true_sequence = []
            inputs = []
            length = 0

    return np.array(true), np.array(predicted), lengths


def get_eval(data: Dataset, model: Model, args: Parser):
    Y, Y_, lengths = get_yy_(data, model)

    eval_df = pd.DataFrame()
    begin = 0
    if args.class_mode == 'multi_label':
        event_names = args.events
    elif args.class_mode == 'multi_class':
        event_names = ['N_E', *args.events]

    for session_id, length in enumerate(lengths):

        true_seq = Y[begin: begin + length]
        if args.class_mode == 'multi_label':
            true_dict = [{'true ' + k: v for k, v in zip(event_names, true)} for true in true_seq]
        elif args.class_mode == 'multi_class':
            true_seq = np.argmax(true_seq, axis=1)
            true_dict = [{'true': true} for true in true_seq]
        true_df = pd.DataFrame.from_dict(true_dict, orient='columns')

        probs_seq = Y_[begin: begin + length]
        probs_dict = [{'prob ' + k: v for k, v in zip(event_names, probs)} for probs in probs_seq]
        probs_df = pd.DataFrame.from_dict(probs_dict, orient='columns')

        if args.class_mode == 'multi_label':
            pred_seq = np.where(probs_seq > 0.5, 1, 0)
            pred_dict = [{'pred ' + k: v for k, v in zip(event_names, pred)} for pred in pred_seq]
        elif args.class_mode == 'multi_class':
            pred_seq = np.argmax(probs_seq, axis=1)
            pred_dict = [{'pred': pred} for pred in pred_seq]
        pred_df = pd.DataFrame.from_dict(pred_dict, orient='columns')

        seq_df = pd.concat([true_df, pred_df, probs_df], axis=1)
        seq_df['session_id'] = session_id

        eval_df = pd.concat([eval_df, seq_df], ignore_index=True)
        begin += length
        session_id += 1

    return eval_df


def get_conf(x: pd.DataFrame, event_name, n=1):
    pred_mask = x['pred ' + event_name] == 1
    true_mask = x['true ' + event_name] == 1
    nb_pred_mask = pred_mask.rolling(2 * n + 1, center=True, min_periods=1).max()
    nb_true_mask = true_mask.rolling(2 * n + 1, center=True, min_periods=1).max()

    true_positives = pred_mask & nb_true_mask
    n_tp = true_positives.sum()

    false_positives = pred_mask & nb_true_mask.replace({0: 1, 1: 0})
    n_fp = false_positives.sum()

    false_negatives = true_mask & nb_pred_mask.replace({0: 1, 1: 0})
    n_fn = false_negatives.sum()

    return n_tp, n_fp, n_fn


def get_scores(tp, fp, fn):
    if tp != 0 or fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp != 0 or fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return precision, recall, f1_score


def get_min_error(w, n):
    w = w.to_numpy()
    e = np.abs(n - np.argwhere(w).reshape((-1))).min() if len(np.argwhere(w)) else -1
    return e


def get_sum_error(w, n):
    w = w.to_numpy()
    e = np.abs(n - np.argwhere(w).reshape((-1))).sum() if len(np.argwhere(w)) else -1
    return e


def get_count_error(w, n):
    w = w.to_numpy()
    e = np.abs(n - np.argwhere(w).reshape((-1))).shape[0] if len(np.argwhere(w)) else -1
    return e


def get_time_error(x: pd.DataFrame, event_name, n=1, how='min'):
    pred_mask = x['pred ' + event_name] == 1
    true_mask = x['true ' + event_name] == 1
    nb_true_mask = true_mask.rolling(2 * n + 1, center=True, min_periods=1).max()
    true_positives = (pred_mask & nb_true_mask).astype(bool)
    time_error = 0

    if how == 'all':
        dists = true_mask.rolling(2 * n + 1, center=True, min_periods=1).apply(lambda w: get_sum_error(w, n))
        counts = true_mask.rolling(2 * n + 1, center=True, min_periods=1).apply(lambda w: get_count_error(w, n))

        tp_dists = dists[true_positives]
        tp_counts = counts[true_positives]

        time_error = tp_dists.sum() / tp_counts.sum()

    if how == 'min':
        dists = true_mask.rolling(2 * n + 1, center=True, min_periods=1).apply(lambda w: get_min_error(w, n))
        tp_dists = dists[true_positives]

        time_error = tp_dists.mean()

    return tp_dists, time_error


def get_stats(eval_data: pd.DataFrame, args: Parser, plot=False):
    event_stats = {event: None for event in args.events}

    fig, ax = plt.subplots(figsize=(12, 5))

    for event in args.events:
        tp, fp, fn = get_conf(eval_data, event, 5)
        precision, recall, f1_score = get_scores(tp, fp, fn)
        time_dists, time_error = get_time_error(eval_data, event, 5)

        event_stats[event] = {
            'true positives': tp,
            'false positives': fp,
            'false negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1 score': f1_score,
            'time error': time_error
        }

        if plot:
            sns.set(style="ticks", palette="pastel")
            sns.set_context("notebook", font_scale=1.3)
            sns.boxplot(data=time_dists, ax=ax)

    if plot:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    return event_stats
