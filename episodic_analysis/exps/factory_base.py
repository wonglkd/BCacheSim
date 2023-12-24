import itertools
import os

from tqdm.auto import tqdm

from .. import episodes
from .. import experiments
from .. import local_cluster
from .. import monitor_exps
from .. import policies
from ..trace_utils import dwpd_to_wr
from .constants import FILTERS_DEFAULT
from .constants import TRY_DROP
try:
    from ..constants_meta import REGIONS_DEFAULT, REGIONS_ALL
except ImportError:
    from ..constants_public import REGIONS_DEFAULT, REGIONS_ALL

# reigon, sampling rate
# csize, wr generator

# options, prefix setter

# APs:
# mlnew
# opt
# [no EA convergence]
# rejectx
# coinflip
# analysis, maxwr, maxcsize
# fractional

# Scenarios
# - New code: rerun
# - More parameters
# - New traces/sample_ratio/etc

OUTPUT_BASE_DIR = local_cluster.OUTPUT_LOCATION+"/"

csize_gb = 357.44751
ramsize_gb = 9.0271
DEFAULT_CSIZE = csize_gb + ramsize_gb
DEFAULT_WR = 34
WRs = [DEFAULT_WR, 50, 100, 75, 20, 10, 60, 90, 30]
WRs_limited = [34, 50, 75, 100, 150]
WRs_ALL = WRs + list(range(10, 100, 10))
max_csize_wrs = list(range(1, 301))
CSIZES = [DEFAULT_CSIZE, csize_gb, ramsize_gb, 400, 100, 50, 200, 600, 800]
# CSIZES_limited = [DEFAULT_CSIZE]

RATIOS_FAST = [.01]
RATIOS_DEFAULT = [.05, .01]
RATIOS_ALL = [.005, .05, .01, .1]
N_SAMPLES_DEFAULT = 1

# REGION_MAX_WRS = {
# }


def traces(regions=REGIONS_DEFAULT, ratios=RATIOS_FAST,
           sample_start=0, max_samples=N_SAMPLES_DEFAULT):
    regions_s = []
    for region in regions:
        if '/' in region:
            parts = region.split('/')
            trace_group, region = '/'.join(parts[:-1]), parts[-1]
        else:
            trace_group = local_cluster.infer_trace_group(region)
        for sample_ratio in ratios:
            for i in range(sample_start, max_samples):
                start = i * sample_ratio
                if start >= 100:
                    break
                regions_s += [(trace_group, region, sample_ratio, start)]
    return {'region_srs': regions_s, '_regions': regions, '_sample_ratios': ratios, '_sample_start': sample_start, '_max_samples': max_samples}


def _postprocess_wr_combos(combos):
    combos = [(round(wr, 3), round(csize, 5)) for wr, csize in combos]
    combos = list(dict.fromkeys(map(tuple, combos)))
    return combos


def csize_wrs(*, wrs=WRs,
              # For analysis
              csizes=CSIZES,
              extra_csizes=False,
              grid=False,
              csize_ratios=[.5, 2, 4, 8, 16, 32, 64, 128, 256, 512],
              add_max=False,
              wrs_limited=WRs_limited,
              limit_cutoff=8,
              ):
    combos = []
    if grid:
        for wr in wrs:
            for csize in csizes:
                combos.append([wr, csize])
    else:
        if len(csizes) > 0:
            for wr in wrs:
                combos.append([wr, csizes[0]])
        if len(wrs) > 0:
            for csize in csizes:
                combos.append([wrs[0], csize])

    # ram_cache_sizes = [ramsize_gb]

    if extra_csizes:
        for csize_ratio in csize_ratios:
            wrs_chosen = wrs if csize_ratio <= limit_cutoff else wrs_limited
            for wr in wrs_chosen:
                combos.append([wr, (csize_gb+ramsize_gb)*csize_ratio])
                if wr * csize_ratio < 100:
                    combos.append([wr*csize_ratio, (csize_gb+ramsize_gb)*csize_ratio])
        csizes = list(set(csize for wr, csize in combos))

    combos = _postprocess_wr_combos(combos)
    _wrs = WRs if len(wrs) < len(WRs) else wrs
    if add_max:
        for csize in csizes:
            combos.append(['MAX', csize])
            combos.append(['MAX-OPT', csize])
        combos.append(['ALL', 'MAX'])

    return {'wr_csize': combos, '_wrs': _wrs, '_csizes': csizes}


def csize_wrs_from_dwpds(*, csizes=None, dwpds=None, grid=False, rescale_for_cachelib_overhead=True):
    combos = []
    if grid:
        for dwpd in dwpds:
            for csize in csizes:
                combos.append([dwpd_to_wr(dwpd, csize, rescale_for_cachelib_overhead=rescale_for_cachelib_overhead), csize])
    else:
        for dwpd in dwpds:
            wr_ = dwpd_to_wr(dwpd, csizes[0], rescale_for_cachelib_overhead=rescale_for_cachelib_overhead)
            combos.append([wr_, csizes[0]])
        for csize in csizes:
            wr_ = dwpd_to_wr(dwpds[0], csize, rescale_for_cachelib_overhead=rescale_for_cachelib_overhead)
            combos.append([wr_, csize])
    combos = _postprocess_wr_combos(combos)
    return {'wr_csize': combos, '_dwpds': dwpds, '_csizes': csizes}


POLICY_FNS = {
    'servicetimesize': policies.PolicyUtilityServiceTimeSize2,
    'peakservicetimesize': policies.PolicyUtilityPeakServiceTimeSize,
    'peakservicetimesizeweighted': policies.PolicyUtilityPeakServiceTimeWeightedSize,
    'servicetimedensity': policies.PolicyUtilityServiceTimeDensity,
    'size': policies.PolicyUtilitySize2,
    'hits': policies.PolicyUtilityHits,
}


def policy_fns(select=['servicetimesize']):
    fns = dict(POLICY_FNS)
    if select:
        for k in POLICY_FNS:
            if k not in select:
                del fns[k]
    return {'policy_fn': fns}

# buffer_sizes = [1]


pf_no_only = [('never', 'episode')]
pf_basic_others = [
    ('at_start', 'acctime-episode'),
    ('predict', 'acctime-episode-predict'),
    ('partial', 'acctime-all'),
    ('partial', 'acctime-episode-predict'),
]
pf_rest = [
    ('partial', 'acctime-episode-predict'),
    ('always', 'acctime-episode'),
    ('benefit', 'acctime-episode'),
    ('at_start', 'acctime-episode-predict'),
    ('predict', 'acctime-episode'),
    ('always', 'acctime-episode-predict'),
]
PF_NO = pf_no_only
PF_OPT = [('at_start', 'acctime-episode')]
PF_ML = [('predict', 'acctime-episode-predict')]
PF_PARTIAL = [('partial', 'acctime-all')]
PF_NO_AND_ML = pf_no_only + PF_ML
PF_NO_AND_OPT = pf_no_only + PF_OPT
PF_NO_ML = pf_no_only + PF_OPT + PF_PARTIAL
PF_BASIC = PF_NO + pf_basic_others
PF_ALL = PF_BASIC + pf_rest
PF_CHUNK = PF_NO + [('always', 'chunk2')]


def prefetchs(pf=PF_NO):
    return {'prefetch_combo': pf}


def exp_args(exp_kwargs):
    ek = exp_kwargs
    dct = dict(
        name=ek['_exp_name'],
        policy=ek['policy'],
        suffix=ek['suffix'],
        csize_gb=ek['wr_csize'][1],
        wr=ek['wr_csize'][0],
        prefetch_when=ek['prefetch_combo'][0],
        prefetch_range=ek['prefetch_combo'][1],
        exp_ap=ek['ap'],
        sim_ap=ek['sim_ap'],
        eviction_policy=ek['eviction_policy'],
        do_not_load=ek.get('do_not_load', False),
        **ek['_common_args'],
    )
    if 'guess' in ek:
        dct['init_guess'] = ek['guess']
    if 'prefetch_when_threshold' in ek:
        dct['prefetch_when_threshold'] = ek['prefetch_when_threshold']
        assert ek['prefetch_combo'][0] == 'predict'
    if 'rejectx_ap_threshold' in ek:
        dct['rejectx_ap_threshold'] = ek['rejectx_ap_threshold']
        assert 'rejectx' in ek['ap']
    for k in ['ram_csize_gb', 'log_interval']:
        if k in ek:
            dct[k] = ek[k]
    return dct


def exp_iter(exp_params, df_guess=None, filters_main=FILTERS_DEFAULT, region_max_wrs=None):
    # Keys that start with an _ are considered constants and set in every dict,
    # as-is (e.g., _wrs, group). They will be used in filters to guess at
    # values. Keys that start with two underscores (__) will have their
    # underscores stripped for plugging in as a constant value, but will be
    # ignored for generating suffixes.

    vals = [v for k, v in exp_params.items() if not k.startswith('_')]
    keys = [k for k in exp_params if not k.startswith('_')]
    constant_vars = []
    for k in exp_params:
        if k.startswith('__'):
            if k[2:] in keys:
                print(f"Error: {k[2:]} already in use, ignoring {k}")
            else:
                constant_vars.append(k[2:])
    for vv in itertools.product(*vals):
        ek = dict(zip(keys, vv))
        for k, v in exp_params.items():
            if k.startswith('__'):
                if k[2:] in constant_vars:
                    ek[k[2:]] = v
            elif k.startswith('_'):
                ek[k] = v

        cft = dict(zip(['trace_group', 'region', 'sample_ratio', 'start'], ek['region_srs']))
        ek['_common_args'] = dict(trace_kwargs=cft, output_base_dir=OUTPUT_BASE_DIR + ek['_group'])

        # Shortcuts
        ek['prefetch_when'] = ek['prefetch_combo'][0]
        ek['prefetch_range'] = ek['prefetch_combo'][1]
        ek['wr'], ek['csize'] = ek['wr_csize']
        ek['trace_group'], ek['region'], ek['sample_ratio'], ek['sample_start'] = ek['region_srs']

        if region_max_wrs and ek['region'] in region_max_wrs:
            if ek['wr'] > region_max_wrs[ek['region']]:
                continue

        pol_class = POLICY_FNS[ek['policy_fn']]
        pol_args = {}
        if 'train_ts' in ek:
            pol_args['train_ts_start'] = ek['train_ts'][0]
            pol_args['train_ts_end'] = ek['train_ts'][1]
            ek['train_ts_start'] = ek['train_ts'][0]
            ek['train_ts_end'] = ek['train_ts'][1]
        pol_args['rl_init_kwargs'] = dict(filter_='noprefetch' if ek['prefetch_when'] == 'never' else 'prefetch')
        if 'EarlyEviction' in ek['eviction_policy']:
            pol_args['rl_init_kwargs']['early_eviction'] = True
        if ek['ap'] == 'fractional':
            pol_args['res_fn_kwargs'] = dict(residency_fn=episodes.process_obj_fractional)
        if ek['ap'] == 'analysis':
            pol_csizes = [ek['csize']]
            pol_wrs = [ek['wr']]
            ek['sim_ap'] = 'analysis'
            if ek['wr'] in ('MAX', 'MAX-OPT'):
                pol_wrs = [34]
                ek['sim_ap'] = 'analysis_maxwr'
                if ek['wr'] == 'MAX-OPT':
                    ek['sim_ap'] += '_opt'
            elif ek['csize'] == 'MAX':
                pol_wrs = max_csize_wrs
                # TODO: Review if this is needed.
                pol_csizes = [csize_gb+ramsize_gb]
                ek['sim_ap'] = 'analysis_maxcsize'
            ek['policy'] = pol_class(target_cache_sizes=pol_csizes, target_wrs=pol_wrs, **pol_args)
        else:
            ek['policy'] = pol_class(target_cache_sizes=ek['_csizes'], target_wrs=ek['_wrs'], **pol_args)

        if 'sim_ap' not in ek:
            ek['sim_ap'] = ek['ap']

        ek['suffix'] = make_suffix(ek, constant_vars)

        if df_guess is not None:
            guess_params = dict(wr=ek['wr'], csize=ek['csize'])
            if type(guess_params['wr']) == str:
                del guess_params['wr']
            if type(guess_params['csize']) == str:
                del guess_params['csize']
            filters = {}
            for k, kk in filters_main.items():
                if kk == '_policy.name_':
                    filters[k] = ek['policy'].name
                elif k == 'Region' and ek[kk] == 'atn1':
                    filters[k] = 'atn1*'
                elif kk in ek:
                    filters[k] = ek[kk]
                elif '_' + kk in ek:
                    filters[k] = ek['_' + kk]
            if ek['ap'] == 'analysis':
                del filters['SimAp']
                filters['AdmissionPolicy'] = 'OfflineAnalysis'
            ek['guess'] = lambda: monitor_exps.guess_v2(
                df_guess, filters=filters,
                allow_mean=True, verbose=False,
                try_drop=TRY_DROP, **guess_params)
            # del ek['guess']['threshold_guess']
        yield ek


def make_suffix(ek, constant_vars=[]):
    params = {
        'ap': 'ap',
        'policy_fn': 'policy',
        'prefetch': 'prefetch',
        'granularity': 'g',
        'ap_feat_subset': 'fs',
        'ap_acc_cutoff': 'accs',
        'label_rejectx': 'label_rejectx',
        'log_interval': 'log',
        'prefetch_when_threshold': 'pfw',
        'eviction_policy': 'evict',
        'train_ts': 'traints',
        'retrain_interval_hrs': 'retrain',
    }
    suffix = ''
    for k, shortk in params.items():
        if k in constant_vars:
            continue
        elif k == 'prefetch':
            suffix += '/prefetch_{}_{}'.format(*ek["prefetch_combo"])
        elif k == 'train_ts' and k in ek:
            suffix += f'/{shortk}_{ek[k][0]}_{ek[k][1]}'
        elif k in ek:
            suffix += f'/{shortk}_{ek[k]}'
    return suffix


def make_exp(exp_kwargs):
    ek = exp_kwargs
    bs = ek.get('batch_size', 16)
    if ek['ap'] == 'rejectx':
        return experiments.ExpRejectX(**exp_args(ek))
    elif ek['ap'] == 'flashield':
        return experiments.ExpFlashield(**exp_args(ek))
    elif ek['ap'] == 'flashieldprob':
        return experiments.ExpFlashieldProb(**exp_args(ek))
    elif ek['ap'] == 'coinflip':
        return experiments.ExpCoinFlip(**exp_args(ek))
    elif ek['ap'] in ('opt', 'fractional'):
        return experiments.ExpOPT(batch_size=bs, **exp_args(ek))
    elif ek['ap'] == 'opt-fixedea':
        return experiments.ExpOPTFixedEA(
            batch_size=bs,
            fixed_ea=ek.get('fixed_ea', 3600*24*7),
            **exp_args(ek))
    elif ek['ap'] == 'mlnew':
        # not tested - fix
        return experiments.ExpNewML(
            batch_size=bs,
            ap_feat_subset=ek.get('ap_feat_subset', 'meta+block+chunk'),
            ap_acc_cutoff=ek.get('ap_acc_cutoff', 15),
            granularity='both',
            train_ts_start=ek.get('train_ts_start', 0),
            train_ts_end=ek.get('train_ts_end', 24*3600),
            **exp_args(ek))
    elif ek['ap'] == 'ml-fixedea':
        return experiments.ExpMLFixedEA(
            batch_size=bs,
            fixed_ea=ek.get('fixed_ea', 3600*24*7),
            ap_feat_subset=ek.get('ap_feat_subset', 'meta+block+chunk'),
            ap_acc_cutoff=ek.get('ap_acc_cutoff', 15),
            granularity='both',
            train_ts_start=ek.get('train_ts_start', 0),
            train_ts_end=ek.get('train_ts_end', 24*3600),
            **exp_args(ek))
    elif ek['ap'] == 'ml':
        size_on = 'size' in ek['policy_fn']
        return experiments.ExpMLCC(
            learned_size=size_on,
            batch_size=bs,
            granularity=ek.get('granularity', 'chunk'),
            chunk_sample_ratio=ek.get('chunk_sample_ratio', 1),
            label_rejectx=ek.get('label_rejectx', 1),
            size_opt=ek.get('size_opt', 'access'),
            train_ts_start=ek.get('train_ts_start', 0),
            train_ts_end=ek.get('train_ts_end', 24*3600),
            **exp_args(ek))
    elif ek['ap'] == 'analysis':
        if ek['sim_ap'] == 'analysis':
            return experiments.ExpAnalysis(**exp_args(ek))
        elif ek['sim_ap'] == 'analysis_maxwr':
            assert ek['wr'] == 'MAX'
            return experiments.ExpAnalysisMaxWR(**exp_args(ek))
        elif ek['sim_ap'] == 'analysis_maxwr_opt':
            assert ek['wr'] == 'MAX-OPT'
            return experiments.ExpAnalysisMaxWROptimal(**exp_args(ek))
        elif ek['sim_ap'] == 'analysis_maxcsize':
            assert ek['csize'] == 'MAX'
            return experiments.ExpAnalysisFixedEA(fixed_ea=3600*24*7, **exp_args(ek))
        else:
            raise NotImplementedError(ek['sim_ap'])
    else:
        raise NotImplementedError


EXP_BASIC = {
    **traces(regions=REGIONS_DEFAULT, ratios=RATIOS_DEFAULT),
    **csize_wrs(wrs=WRs[:3]),
    **policy_fns(),
    **prefetchs(),
    **{'__eviction_policy': 'LRU',
       '__log_interval': 600},
}

EXP_OPT = {
    **EXP_BASIC,
    **{'ap': ['opt']},
    **prefetchs(PF_BASIC),
}

EXP_MLNEW = {
    **EXP_BASIC,
    **{
        'ap': ['mlnew'],
        # 'ap_acc_cutoff': [15],
        # 'ap_feat_subset': ['meta', 'meta+block', 'meta+block+chunk'],
        '__ap_acc_cutoff': 15,
        '__ap_feat_subset': 'meta+block+chunk',
    },
    **prefetchs(PF_BASIC),
}

EXP_MLNEW_NOSIZEFEAT = {
    **EXP_MLNEW,
    **{'ap_feat_subset': ['meta_nosize+block+chunk']},
    **prefetchs(PF_NO),
}

EXP_MLNEW_NOSIZEGOAL = {
    **EXP_MLNEW,
    **policy_fns(['hits']),
    **prefetchs(PF_NO),
}

EXP_MLNEW_NOSIZEGOALFEAT = {
    **EXP_MLNEW_NOSIZEGOAL,
    **{'ap_feat_subset': ['meta_nosize+block+chunk']},
}

EXP_STATIC = {
    **EXP_BASIC,
    **policy_fns(['hits']),
    **{'ap': ['rejectx', 'coinflip']},
}

EXP_FLASHIELD = {
    **EXP_BASIC,
    **policy_fns(['hits']),
    **{'ap': ['flashield']},
}

EXP_FLASHIELDPROB = {
    **EXP_BASIC,
    **policy_fns(['hits']),
    **{'ap': ['flashieldprob']},
}

EXP_ANALYSIS = {
    **EXP_BASIC,
    **csize_wrs(wrs=WRs_ALL, extra_csizes=True, add_max=True),
    **policy_fns(['servicetimesize']),
    **prefetchs(PF_NO_AND_OPT),
    **{'ap': ['analysis']},
}


EXP_FRACTIONAL = {
    **EXP_BASIC,
    **{
        'ap': ['fractional'],
        'sim_ap': ['opt']
    },
    **prefetchs(PF_CHUNK),
}

EXP_ANALYSIS_EARLYEVICTION = {
    **EXP_ANALYSIS,
    **{'eviction_policy': ['LRU+EarlyEviction']},
}


def check_params(params_):
    for trace_group, region, sample_ratio, start in params_['region_srs']:
        filename = local_cluster.tracefilename(sample_ratio, region, start=start, trace_group=trace_group)
        assert os.path.exists(filename), f"{filename} does not exist!"


class ExpFactoryBase(object):
    def __init__(self, exp_date, name, desc='', group="spring23", region_max_wrs=None):
        self.p_name = name
        self.date = exp_date
        self.name = f'{exp_date}_{name}'
        self.desc = desc
        self.group = group
        self.params_ = []
        self.region_max_wrs = region_max_wrs


class ExpFactory(ExpFactoryBase):
    """ExpFactory.

    Usage:
        eff = ExpFactory(
            date='20220414',
            name='{date}_baseline_mlpf',
            desc='')
        eff.update(all_exps, {
            **ef.policy_fns(['hits']),
            **ef.prefetchs([
                ('predict', 'acctime-episode-predict'),
                ('at_start', 'acctime-episode'),
            ]),
            **{'ap': ['rejectx', 'coinflip']},
        })
    """
    def params(self, cfg):
        return {
            **EXP_BASIC,
            **{'_exp_name': self.name, "_group": self.group},
            **cfg}

    def add_params(self, params):
        check_params(params)
        self.params_.append(self.params(params))
        return self

    def exps(self, params, df_guess=None, exp_iter_fn=exp_iter, make_exp_fn=make_exp):
        exps = []
        if df_guess is not None:
            # Drop irrelevant columns for speed.
            keep_cols = set(FILTERS_DEFAULT)
            keep_cols |= set(['Target Write Rate', 'Write Rate (MB/s)', 'Cache Size (GB)', 'ExperimentPrefix', 'ExperimentName',
                              'AP Threshold', 'AP Probability', 'Avg Eviction Age (s)', 'Assumed Eviction Age (s)', 'IOPS Saved Ratio', 'Service Time Saved Ratio'])
            keep_cols = keep_cols & set(df_guess.columns)
            df_guess = df_guess[list(keep_cols)]
        for ek in tqdm(exp_iter_fn(params, df_guess=df_guess, region_max_wrs=self.region_max_wrs)):
            exps.append(make_exp_fn(ek))
        return exps

    def all(self, df_guess=None):
        exps = []
        for params in self.params_:
            exps += self.exps(params=params, df_guess=df_guess)
        return exps

    def update(self, all_exps, exp_params=None, **kwargs):
        if exp_params is not None:
            self.add_params(exp_params)
        all_exps[self.name] = self.all(**kwargs)
        return self
