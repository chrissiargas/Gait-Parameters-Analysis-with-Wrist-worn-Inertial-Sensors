import pandas as pd
from scipy.signal import find_peaks
import numpy as np
from math import isnan


def compare_events(true, pred, thres=50):
    if len(true) == 0 and len(pred) == 0:
        print("No gait events annotated, no gait events detected!")
        return np.array([]), np.array([]), np.array([])

    if len(true) != 0 and len(pred) == 0:
        print(f"{len(true)} gait events annotated, but none were detected.")

        return np.array([-999 for _ in range(len(true))]), np.array([]), np.array(
            [-999 for _ in range(len(true))])

    if len(true) == 0 and len(pred) != 0:
        print(f"No gait events annotated, but {len(pred)} events were detected.")
        return np.array([]), np.array([-999 for _ in range(len(pred))]), np.array(
            [-999 for _ in range(len(pred))])

    abs_diff = np.abs(pred[:, np.newaxis] - true)
    true2pred = np.argmin(abs_diff, axis=0)

    abs_diff = np.abs(true[:, np.newaxis] - pred)
    pred2true = np.argmin(abs_diff, axis=0)

    duplicate_values = np.where(np.bincount(true2pred) > 1)[0]
    for duplicate_value in duplicate_values:
        duplicate_indices = np.where(true2pred == duplicate_value)[0]
        true2pred[np.setdiff1d(duplicate_indices, pred2true[duplicate_value])] = -999

    duplicate_values = np.where(np.bincount(pred2true) > 1)[0]
    for duplicate_value in duplicate_values:
        duplicate_indices = np.where(pred2true == duplicate_value)[0]
        pred2true[np.setdiff1d(duplicate_indices, true2pred[duplicate_value])] = -999

    time_diffs = pred[pred2true > -999] - true[true2pred > -999]

    indices_t2p = true2pred[true2pred > -999]
    indices_p2t = pred2true[pred2true > -999]

    for ti, (time_diff, index_t2p, index_p2t) in enumerate(zip(time_diffs, indices_t2p, indices_p2t)):
        if time_diff > thres:
            true2pred[index_p2t] = -999
            pred2true[index_t2p] = -999

    time_diffs = pred[pred2true > -999] - true[true2pred > -999]
    return true2pred, pred2true, time_diffs


def find_matches(x):
    x = x.copy()

    subs = x.subject.unique()
    acts = x.activity.unique()
    events = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']

    sub_ids, act_ids, event_types, ix_refs, ix_matches = [], [], [], [], []
    for sub in subs:
        for act in acts:
            ann = x.loc[(x['subject'] == sub) & (x['activity'] == act)]

            ix_pred_events = {'LF_HS': None, 'RF_HS': None, 'LF_TO': None, 'RF_TO': None}
            ix_true_events = {'LF_HS': None, 'RF_HS': None, 'LF_TO': None, 'RF_TO': None}
            event_type, ix_ref, ix_match = [], [], []

            # Get true and predicted events
            for event in events:
                ix_pred_events[event], _ = find_peaks(ann['prob ' + event].values, height=0.5, distance=50)
                ix_true_events[event], _ = find_peaks(ann['true ' + event].values, height=0.5, distance=50)

                true2pred, pred2true, dt = compare_events(ix_true_events[event], ix_pred_events[event])

                for i, ix_true in enumerate(ix_true_events[event]):
                    event_type.append(event)
                    ix_ref.append(ix_true)
                    if true2pred[i] > -999:
                        ix_match.append(ix_pred_events[event][true2pred[i]])
                    else:
                        ix_match.append(None)

                for i, ix_pred in enumerate(ix_pred_events[event]):
                    if pred2true[i] == -999:
                        event_type.append(event)
                        ix_ref.append(None)
                        ix_match.append(ix_pred)

            sub_id = [sub for _ in range(len(event_type))]
            act_id = [act for _ in range(len(event_type))]

            sub_ids += sub_id
            act_ids += act_id
            event_types += event_type
            ix_refs += ix_ref
            ix_matches += ix_match

    df = pd.DataFrame({
        "subject": sub_ids,
        "activity": act_ids,
        "event": event_types,
        "ix_true": ix_refs,
        "ix_pred": ix_matches
    })

    return df


def find_gait_params(x, fs):
    x = x.copy()

    subs = x.subject.unique()
    acts = x.activity.unique()

    df = {"sub_id": [],
                       "act_id": [],
                       "ix_true": [],
                       "ix_pred": [],
                       "stride_time_true": [],
                       "stride_time_pred": [],
                       "stance_time_true": [],
                       "stance_time_pred": [],
                       "swing_time_true": [],
                       "swing_time_pred": []}

    for sub in subs:
        for act in acts:
            df_sel = x.loc[(x['subject'] == sub) & (x['activity'] == act)]
            for position in ['RF', 'LF']:
                df_sel = df_sel.loc[df_sel["event"].isin([position+'_HS', position+'_TO'])]

                ix_HS_true = df_sel.loc[df_sel["event"] == position + '_HS']["ix_true"].values
                ix_HS_pred = df_sel.loc[df_sel["event"] == position + '_HS']["ix_pred"].values
                ix_TO_true = df_sel.loc[df_sel["event"] == position + '_TO']["ix_true"].values
                ix_TO_pred = df_sel.loc[df_sel["event"] == position + '_TO']["ix_pred"].values

                if len(ix_HS_true) == 0 or len(ix_HS_pred) == 0 or len(ix_TO_true) == 0 or len(ix_TO_pred) == 0:
                    continue

                for i in range(len(ix_HS_true) - 1):
                    f = np.argwhere(ix_TO_true > ix_HS_true[i])[:, 0][0]
                    if isnan(ix_HS_pred[i]) or isnan(ix_TO_pred[f]) or isnan(ix_HS_pred[i + 1]) is None:
                        continue

                    stride_time_true = (ix_HS_true[i + 1] - ix_HS_true[i]) / fs
                    stride_time_pred = (ix_HS_pred[i + 1] - ix_HS_pred[i]) / fs
                    stance_time_true = (ix_TO_true[f] - ix_HS_true[i]) / fs
                    stance_time_pred = (ix_TO_pred[f] - ix_HS_pred[i]) / fs
                    swing_time_true = (ix_HS_true[i + 1] - ix_TO_true[f]) / fs
                    swing_time_pred = (ix_HS_pred[i + 1] - ix_TO_pred[f]) / fs

                    # Add to dict
                    df["sub_id"].append(sub)
                    df["act_id"].append(act)
                    df["ix_true"].append(ix_HS_true[i])
                    df["ix_pred"].append(ix_HS_pred[i])
                    df["stride_time_true"].append(stride_time_true)
                    df["stride_time_pred"].append(stride_time_pred)
                    df["stance_time_true"].append(stance_time_true)
                    df["stance_time_pred"].append(stance_time_pred)
                    df["swing_time_true"].append(swing_time_true)
                    df["swing_time_pred"].append(swing_time_pred)

    df = pd.DataFrame(df)
    return df
