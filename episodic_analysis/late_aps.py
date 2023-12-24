"""Model APs that admit late (after start of episode)."""
import logging

import numpy as np
import pandas as pd

from . import train_ap
from . import ep_utils


def eval_with_middle_hits(dfc_, Y_cols, *, at_=None, y_label='Y_pred', threshold=0.5, flip=False):
    filter_ = at_.rl.filter
    def parse_ep(keyx, dff):
        dff = dff.sort_values('feat_ep_acc_i')
        lost_st = 0
        admitted_at = None
        for _, row in dff.iterrows():
            accept = row[y_label] > threshold
            if flip:
                accept = not accept
            if accept:
                admitted_at = row['feat_ep_acc_i']
                break
            else:
                lost_st += row[f'marginal_sts__{filter_}']
        erow = {'block_id': keyx[0], 'feat_ep_i': keyx[1], 'key': keyx,
                'admitted_at': admitted_at,
                'admitted_binary': admitted_at is not None,
                'lost_st_from_late': lost_st if admitted_at is not None else 0,
                'num_chunks': row['feat_eps|opt-num_chunks'],
                'net_st_after_late': row[f'service_time_saved__{filter_}'] - lost_st if admitted_at is not None else 0,
                }
        for k in list(Y_cols) + ['feat_ep_id']:
            if k in row and k not in (f'marginal_sts__{filter_}', 'key'):
                erow[k] = row[k]
        return erow
    erows = [parse_ep(*args) for args in dfc_.groupby(['key', 'feat_ep_i'])]
    d_ = pd.DataFrame(erows)
    d_['epsFN'] = d_['threshold_binary'] & ~d_['admitted_binary']
    d_['epsFP'] = ~d_['threshold_binary'] & d_['admitted_binary']
    d_['epsTP'] = d_['threshold_binary'] & d_['admitted_binary']
    d_['epsTN'] = ~d_['threshold_binary'] & ~d_['admitted_binary']
    return d_


def _get_row_stats(d_, at_, factor=1.):
    st_orig_ex_ = d_['service_time_orig'].sum()
    dx_ = d_[d_['admitted_binary']]
    filter_ = at_.rl.filter
    return {
        'ServiceTimeOrig': st_orig_ex_,
        'NoLateServiceTimeSavedRatio': dx_[f'service_time_saved__{filter_}'].sum() / st_orig_ex_,
        'ServiceTimeSavedRatio': dx_['net_st_after_late'].sum() / st_orig_ex_,
        'ServiceTimeLostFromLateRatio': dx_['lost_st_from_late'].sum() / st_orig_ex_,
        'NoLateServiceTimeLostFromCompleteFNsRatio': d_[d_['epsFN']][f'service_time_saved__{filter_}'].sum() / st_orig_ex_,
        'ServiceTimeSavedFromFPRatio': d_[d_['epsFP']][f'service_time_saved__{filter_}'].sum() / st_orig_ex_,
        'WriteRate': at_.rl.th.chunks_to_wr_mbps(d_[d_['admitted_binary']]['num_chunks'].sum() / factor),
        'WriteRateOnFPs': at_.rl.th.chunks_to_wr_mbps(d_[d_['epsFP']]['num_chunks'].sum() / factor),
        'epsFP': d_['epsFP'].sum(),
        'epsFN': d_['epsFN'].sum(),
        'epsTP': d_['epsTP'].sum(),
        'epsTN': d_['epsTN'].sum(),
    }


def get_dfc(at_, label):
    sel_feat = ep_utils.make_unique(at_.feat_cols + train_ap.subsets(at_.df_X)['opt'])
    dfc = pd.concat([at_.df_X[sel_feat], at_.df_Y], axis=1)
    dfc['Y_pred'] = at_._predicter(label)(at_.df_X[at_.feat_cols])
    return dfc


def eval_at_thresholds(at_, label, thresholds, use='test'):
    dfs = []
    dfc = get_dfc(at_, label)
    factor = 1.
    if use == 'test':
        dfc = dfc.iloc[at_.split_idxes[use]]
        factor = at_.test_size

    dfes = {}
    for t in thresholds:
        df_ = eval_with_middle_hits(
            dfc, at_.df_Y.columns, at_=at_, threshold=t, flip=label == 'threshold')
        dfe = _get_row_stats(df_, at_, factor=factor)
        dfes[t] = df_
        dfe['Threshold'] = t
        dfe['Label'] = label
        dfe['Split'] = use
        dfs.append(dfe)
    return pd.DataFrame(dfs), dfes, dfc


def find_opt_threshold(at_, label_name, init, target_wr=34, verbose=False, atol=2., **kwargs):
    if verbose:
        print("Running ", init)
    df_c, dfes, dfc = eval_at_thresholds(at_, label_name, init, **kwargs)
    for i in range(30):
        df_c = df_c.sort_values(['WriteRate', 'Threshold']).drop_duplicates(
            keep='first', subset=[c for c in df_c.columns if c != 'Threshold'])
        idx = np.searchsorted(df_c['WriteRate'], target_wr)
        check = df_c['Threshold'].is_monotonic or df_c['Threshold'].is_monotonic_decreasing
        assert check, df_c
        if idx >= len(df_c):
            new_t = df_c.iloc[-1]['Threshold'] * 2
        elif idx == 0:
            # If zero is included, this shouldn't happen
            raise NotImplementedError
        else:
            if np.isclose(df_c.iloc[idx]['WriteRate'], target_wr, atol=atol):
                return df_c.iloc[idx:idx+1], dfes[df_c['Threshold'].iloc[idx]]
            new_t = df_c.iloc[[idx-1, idx]]['Threshold'].mean()
        new_dfc, dfe_, _ = eval_at_thresholds(at_, label_name, [new_t], **kwargs)
        dfes.update(dfe_)
        if verbose:
            print(i, "Running ", new_t, df_c.iloc[idx]['WriteRate'], idx)
        df_c = pd.concat([df_c, new_dfc])
    logging.error("Error: Exceeded max iterations")
    return df_c, None
