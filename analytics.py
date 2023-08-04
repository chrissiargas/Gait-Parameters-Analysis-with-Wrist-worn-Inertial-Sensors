
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from configParser import Parser
from analytics_utils import find_matches, find_gait_params
from analytics_plots import box_plots, scatter_plots


def find(model, date, cross_val, enviroment):
    archive = os.path.join("archive", model)
    path = os.path.join(archive, "save-" + date)
    hparams = 'split-' + cross_val + '-environment-' + enviroment
    path = os.path.join(path, hparams)
    scores_file = os.path.join(path, "scores.csv")
    annots_file = os.path.join(path, "annotations.csv")

    return scores_file, annots_file


analytics = ['Romi']
model = 'Tan'
if model == 'Tan':
    fs = 128.
elif model == 'Romi':
    fs = 200.

args = Parser()
args.get_args()

scores_file, annots_file = find(model, '20230804-141515', 'start', 'outdoors')

if 'Tan' in analytics:
    results = pd.read_csv(scores_file)
    results['event_'] = results['event'].str[-2:]

    plt.figure()
    sns.set_context("notebook", font_scale=1.3)
    g = sns.boxplot(x="activity", y="f1 score",
                    hue="event_",
                    data=results)
    g.set_xticklabels(g.get_xticklabels(), rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()

    plt.figure()
    results = results[results['true positives'] != 0]
    sns.set_context("notebook", font_scale=1.3)
    g = sns.boxplot(x="activity", y="time error",
                    hue="event_",
                    data=results)

    g.set_xticklabels(g.get_xticklabels(), rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()

if 'Romi' in analytics:
    ds = pd.read_csv(annots_file)
    df = find_matches(ds)

    subs = ds.subject.unique()
    acts = ds.activity.unique()
    events = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']

    hs_cols = ['LF_HS', 'RF_HS']
    to_cols = ['LF_TO', 'RF_TO']

    df_hs = df.loc[df['event'].isin(hs_cols)]
    df_to = df.loc[df['event'].isin(to_cols)]

    df_hs_clean = df_hs[(df_hs["ix_true"].notna()) & (df_hs["ix_pred"].notna())].copy()
    df_to_clean = df_to[(df_to["ix_true"].notna()) & (df_to["ix_pred"].notna())].copy()

    df_hs_clean["diff_msec"] = (df_hs_clean["ix_true"] - df_hs_clean["ix_pred"]) * 1000 / fs
    df_to_clean["diff_msec"] = (df_to_clean["ix_true"] - df_to_clean["ix_pred"]) * 1000 / fs

    box_plots(df_hs_clean, df_to_clean)

    df_gait_metrics = find_gait_params(df, fs)

    scatter_plots(df_gait_metrics)

