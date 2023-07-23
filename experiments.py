import pandas as pd

from dataset import Dataset
from model import train_evaluate
import ruamel.yaml
import os
import time
from keras import backend as K


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


def save(path, scores, hparams=None):
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
    paramsFile = os.path.join(path, "parameters.yaml")

    scores.to_csv(scoresFile, index=False)
    config_save(paramsFile)


def Tan_experiment(subject, activity, event):
    config_edit('train_args', 'subjects', [subject])
    config_edit('train_args', 'activities', [activity])
    config_edit('train_args', 'events', [event])

    data = Dataset(regenerate=False)
    stats = train_evaluate(data, summary=False, verbose=1, mVerbose=False)

    results = stats[event]
    results['subject'] = subject
    results['activity'] = activity
    results['event'] = event

    del data

    return results


def Tan(archive_path):
    archive = os.path.join(archive_path, "save-" + time.strftime("%Y%m%d-%H%M%S"))

    cross_vals = ['start', 'middle', 'end']
    enviroments = ['indoors', 'outdoors']

    indoor_subs = [*range(1, 12)]
    outdoor_subs = [*range(12, 21)]
    sub_sets = [indoor_subs, outdoor_subs]

    indoor_activities = ['treadmill_walk', 'treadmill_walknrun', 'treadmill_run',
                         'treadmill_slope_walk', 'indoor_walk', 'indoor_run', 'indoor_walknrun']
    outdoor_activities = ['outdoor_walk', 'outdoor_run', 'outdoor_walknrun']
    act_sets = [indoor_activities, outdoor_activities]

    events = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']

    for cross_val in cross_vals:
        config_edit('train_args', 'train_test_split', cross_val)
        for n, (env, subs, acts) in enumerate(zip(enviroments, sub_sets, act_sets)):
            if n == 0:
                continue
            results = pd.DataFrame()
            for sub in subs:
                for act in acts:
                    for event in events:
                        results_per_x = Tan_experiment(sub, act, event)
                        results = pd.concat([results, pd.DataFrame([results_per_x])], ignore_index=True)
                        save(archive, results,
                             hparams='split-' + cross_val + '-enviroment-' + env)
                        K.clear_session()

