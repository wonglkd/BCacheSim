"""These should be updated together."""

# Used in base.py
EXP_CONFIG = {
    'prefetch_range': 'Prefetch-Range',
    'prefetch_when': 'Prefetch-When',
    'prefetch_when_threshold': 'PrefetchWhenThreshold',
    'label_rejectx': 'Train-LabelRejectX',
    'ap_feat_subset': 'Feat-Subset',
    'ap_acc_cutoff': 'Ap-Acc-Cutoff',
    'granularity': 'Granularity',
    'sim_ap': 'SimAp',

    'exp_ap': 'ExpAp',

    'train_ts_start': 'TrainTsStart',
    'train_ts_end': 'TrainTsEnd',

    'admit_chunk_threshold': 'admit_chunk_threshold',
    'chunk_sample_ratio': 'Train-ChunkSampleRatio',
    'hybrid_ap_threshold': 'Hybrid-Threshold',
    'opt_ap_threshold': 'OPT-Threshold',
    'rejectx_ap_threshold': 'RejectX-Threshold',
    'rejectx_ap_factor': 'RejectX-Factor',

    'eviction_policy': 'EvictionPolicy',

    'log_interval': 'LogInterval',
    'trace_group': 'TraceGroup',
}

# Used in factory.py
FILTERS_DEFAULT = {
    # Manually ordered by dropping order
    'SamplingRatio': 'sample_ratio',  # TODO: Deprecate
    'SampleRatio': 'sample_ratio',
    'TraceGroup': 'trace_group',
    'PrefetchWhenThreshold': 'prefetch_when_threshold',
    'Prefetch-Range': 'prefetch_range',
    'Prefetch-When': 'prefetch_when',
    'Train-LabelRejectX': 'label_rejectx',
    'Feat-Subset': 'ap_feat_subset',
    'Ap-Acc-Cutoff': 'ap_acc_cutoff',
    'Granularity': 'granularity',
    'Policy': '_policy.name_',
    'EvictionPolicy': 'eviction_policy',
    'LogInterval': 'log_interval',
    'TrainTsStart': 'train_ts_start',
    'TrainTsEnd': 'train_ts_end',
    # ExpAp corresponds to what we think of Ap
    # SimAp corresponds to different classes
    'ExpAp': 'ap',  # TODO: Inconsistency (vs exp_ap). Replace ap with exp_ap in factory.py.
    # 'ExpAp': 'exp_ap',
    # 'ApOpt': 'sim_ap',  # TODO: Eventually ApOpt should be changed to SimAp
    'SimAp': 'sim_ap',
    # Do not drop
    'Region': 'region',
}

# Used in factory.py
TRY_DROP = ['SampleStart', 'SamplingRatio',  # TODO: Deprecate.
            'TraceGroup',
            'SampleRatio', 'LogInterval',
            'PrefetchWhenThreshold',
            'Prefetch-Range', 'Prefetch-When',
            'Train-LabelRejectX', 'Feat-Subset', 'Ap-Acc-Cutoff', 'Granularity',
            'TrainTsStart', 'TrainTsEnd',
            'EvictionPolicy',
            'Policy',
            'ExpAp', 'SimAp', 'ApOpt',
            'AdmissionPolicy']  # Not in filters


# lbm_new = {
#     ('episode', 'never'): ('No prefetching', dict(color='green')),
# }
# lbm_range = {
#     'acctime-episode': 'OPT-Range',
#     'acctime-episode-predict': 'ML-Range',
#     'acctime-all': '8MB',
#     'chunk2': 'OPT(Frac)-Chunk',
# }
# lbm_when = {
#     'at_start': 'OPT-Ep-Start',
#     'predict': 'ML-When',
#     'partial': 'Partial-Hit',
#     'always': 'Every Miss',
#     'benefit': 'OPT-Benefit',
#     'rejectfirst': '2nd-Miss',
# }
# lbs = dict(zip(lbm_when.keys(), COLORLISTS[6]))
# lb1 = dict(zip(lbm_range.keys(), LINESTYLES))
# lb2 = dict(zip(lbm_range.keys(), MARKERS))
# for pp, ppv in lbm_range.items():
#     for qq, qqv in lbm_when.items():
#         lbm_new[(pp, qq)] = (ppv + ' on ' + qqv, dict(color=lbs[qq], ls=lb1[pp], marker=lb2[pp]))
