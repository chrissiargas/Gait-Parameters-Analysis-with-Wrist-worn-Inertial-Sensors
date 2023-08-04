import pandas as pd

from dataset import Dataset
from model import train_evaluate
import ruamel.yaml
import os
import time
import gc

from parameters import TAN_params, ROMI_params


def reset_tensorflow_keras_backend():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    _ = gc.collect()


def config_edit(args, parameter, value):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        data = yaml.load(fp)

    for param in data[args]:

        if param == parameter:
            data[args][param] = value
            break

    with open('config.yaml', 'w') as fb:
        yaml.dump(data, fb)


def config_save(paramsFile):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        parameters = yaml.load(fp)

    with open(paramsFile, 'w') as fb:
        yaml.dump(parameters, fb)


def save(path, scores, ds, hparams=None):
    if not hparams:
        try:
            os.makedirs(path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    else:
        try:
            path = os.path.join(path, hparams)
            os.makedirs(path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    scoresFile = os.path.join(path, "scores.csv")
    annotationFile = os.path.join(path, "annotations.csv")
    paramsFile = os.path.join(path, "parameters.yaml")

    scores.to_csv(scoresFile, index=False)
    ds.to_csv(annotationFile, index=False)
    config_save(paramsFile)


def Tan_experiment(subject, activity, event, tracked_point=None):
    if tracked_point is not None:
        config_edit('train_args', 'positions', [tracked_point])

    config_edit('train_args', 'subjects', [subject])
    config_edit('train_args', 'activities', [activity])
    config_edit('train_args', 'events', [event])

    data = Dataset(regenerate=False)
    stats, ds = train_evaluate(data, summary=False, verbose=1, mVerbose=False)

    results = stats[event]
    results['subject'] = subject
    results['activity'] = activity
    results['event'] = event
    results['tracked_point'] = tracked_point

    del data

    return results, ds


def Romi_experiment(recording, target, round):
    events = [target + '_HS', target + '_TO']
    config_edit('train_args', 'position', [recording])
    config_edit('train_args', 'events', events)

    data = Dataset(regenerate=False)
    results = train_evaluate(data, summary=True, verbose=1, mVerbose=True)

    results['recording'] = recording
    results['target'] = target
    results['round'] = round

    del data

    return results


def Tan(archive_path):
    parameters = TAN_params

    for param_name, param_value in parameters.items():
        config_edit('train_args', param_name, param_value)

    archive = os.path.join(archive_path, "save-" + time.strftime("%Y%m%d-%H%M%S"))

    cross_vals = ['start']
    enviroments = ['indoors', 'outdoors']

    indoor_subs = [*range(1, 12)]
    outdoor_subs = [*range(12, 21)]
    sub_sets = [indoor_subs, outdoor_subs]

    indoor_activities = ['treadmill_walk', 'treadmill_walknrun', 'treadmill_run',
                         'treadmill_slope_walk', 'indoor_walk', 'indoor_run', 'indoor_walknrun']
    outdoor_activities = ['outdoor_walk', 'outdoor_run', 'outdoor_walknrun']
    act_sets = [indoor_activities, outdoor_activities]

    tracked_points = ['RF', 'LF', 'Waist', 'Wrist']

    events = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']
    n_events = 4

    for cross_val in cross_vals:
        config_edit('train_args', 'train_test_split', cross_val)
        for n, (env, subs, acts) in enumerate(zip(enviroments, sub_sets, act_sets)):
            if n == 0:
                continue
            results = pd.DataFrame()
            ds_test = pd.DataFrame()
            for tracked_point in tracked_points:
                for sub in subs:
                    if sub == 4:
                        continue
                    for act in acts:
                        ds_test_per_sa = pd.DataFrame()
                        for e, event in enumerate(events):
                            reset_tensorflow_keras_backend()
                            print(cross_val, env, tracked_point, sub, act, event)
                            results_per_x, ds_test_per_x = Tan_experiment(sub, act, event, tracked_point)

                            results = pd.concat([results, pd.DataFrame([results_per_x])], ignore_index=True)
                            if e != n_events - 1:
                                ds_test_per_x = ds_test_per_x.drop('session_id', axis=1)
                            ds_test_per_sa = pd.concat([ds_test_per_sa, ds_test_per_x], axis=1)

                            save(archive, results, ds_test,
                                 hparams='split-' + cross_val + '-environment-' + env)

                        ds_test_per_sa['subject'] = sub
                        ds_test_per_sa['activity'] = act
                        ds_test_per_sa['tracked_point'] = tracked_point

                        ds_test = pd.concat([ds_test, ds_test_per_sa], ignore_index=True)
                        save(archive, results, ds_test,
                             hparams='split-' + cross_val + '-environment-' + env)


def Romi(archive_path):
    parameters = ROMI_params

    for param_name, param_value in parameters.items():
        config_edit('train_args', param_name, param_value)

    archive = os.path.join(archive_path, "save-" + time.strftime("%Y%m%d-%H%M%S"))

    rec_positions = ['LF', 'RF', 'Waist', 'Wrist']
    target_positions = ['LF', 'RF']
    n_reps = 1

    results = pd.DataFrame()
    for rec_position in rec_positions:
        for target_position in target_positions:
            for rep in range(n_reps):
                reset_tensorflow_keras_backend()
                results_per_xp = Romi_experiment(rec_position, target_position, round)
                results = pd.concat([results, pd.DataFrame([results_per_xp])], ignore_index=True)
                save(archive, results, hparams=None)
