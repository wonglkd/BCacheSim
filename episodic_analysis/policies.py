import os
import sys

import numpy as np
import pandas as pd

from . import ep_utils
from . import experiments
from . import local_cluster
from .ep_utils import compress_ext
from .ep_utils import np_safe_div
from .episodes import ResidencyListPrefetchAware
from .episodes import ResidencyListPeakAware
from .episodes import ResidencyListSizeAware
from .episodes import generate_residencies
from .episodes import process_obj_chunk_n_noprefetch_w_accs
from .episodes import process_obj_chunkheuristic
from .episodes import run_heuristic


class Policy(object):
    def __init__(self,
                 name,
                 exp='UNDEFINED',
                 output_base_dir='./',
                 target_wrs=[20, 34, 50, 75, 100, 150],
                 target_cache_sizes=[200, 400, 600, 800],
                 target_wr=None,  # For analysis only
                 target_cache_size=None,  # For analysis only
                 supplied_ea='physical',
                 train_target_wr=None,
                 train_models=None,
                 trace_kwargs=None,
                 train_ts_start=0,
                 train_ts_end=3600*24,
                 res_fn_kwargs=None,
                 rl_init_kwargs=None,
                 train_prefetch_wr=None,  # For legacy
                 extra_args='',
                 suffix='',
                 source_eps_from_policy=None):
        """
        train_split_secs: Train on first N secs of trace.
        """
        self.name = name
        self.result_dir = None
        self.trace_kwargs = trace_kwargs or {}
        self.target_wrs = target_wrs
        self.target_cache_sizes = target_cache_sizes
        self.exp = exp
        self.residency_lists_ = None
        self.output_base_dir = output_base_dir
        self.supplied_ea = supplied_ea
        if train_prefetch_wr and train_target_wr is None:
            train_target_wr = train_prefetch_wr
        self.train_target_wr = train_target_wr
        self.res_fn_kwargs = res_fn_kwargs or {}
        self.rl_init_kwargs = rl_init_kwargs or {}
        self.src_policy = source_eps_from_policy
        self.train_models = train_models or set()
        self.train_models = set(self.train_models)
        self.train_ts_start = train_ts_start
        self.train_ts_end = train_ts_end
        self.train_prefetch_path = None
        self.extra_args = extra_args
        self.suffix = suffix
        self.e_age_s = None

    def __repr__(self):
        return f'{self.__class__.__name__}({self.result_dir})'

    def _init(self, e_age_s=None):
        self.residency_lists_ = None
        self.result_dir = f'{self.exp}/{local_cluster.fmt_trace_id(**self.trace_kwargs)}/'
        self.working_dirname = local_cluster.proj_path(self.exp, self.trace_kwargs)
        if self.suffix != '':
            self.result_dir += self.suffix.strip('/') + '/'
            self.working_dirname = os.path.join(self.working_dirname, self.suffix.strip('/'))
        self.result_dir = os.path.join(self.output_base_dir, self.result_dir)
        # ext = '.shelve.db' if e_age_s < 2000 else f'.pkl{compress_ext}'
        ext = f'.pkl{compress_ext}'
        if e_age_s is None:
            return
        self.e_age_s = e_age_s
        self.filenames = {
            'analysis': os.path.join(self.result_dir, f'offline_analysis_ea_{e_age_s:g}.csv'),
            'thresholds': f'{self.working_dirname}/decisions_{self.name}_ea_{e_age_s:g}{ext}',
            # 'earlyevict': f'{dirname}/earlyevict_ea_{e_age_s:g}.pkl{compress_ext}',
            # 'earlyadmit': f'{dirname}/earlyadmit_ea_{e_age_s:g}.pkl{compress_ext}',
        }
        if self.train_target_wr is not None:
            self.train_prefetch_path = os.path.join(self.working_dirname, f"ea_{e_age_s:g}_wr_{self.train_target_wr:g}")

            if 'prefetch' in self.train_models:
                for label_name in ["offset_start", "size", "offset_end", "pred_net_pf_st_binary"]:
                    self.filenames["model_prefetch_" + label_name] = f"{self.train_prefetch_path}_prefetch_{label_name}.model"
            if 'admit' in self.train_models:
                for label_name in [f"threshold_binary"]:
                    self.filenames["model_admit_" + label_name] = f"{self.train_prefetch_path}_admit_{label_name}.model"
        self.fail_file = self.get_out_prefix() + ".fail"
        return self

    def to_cmd(self, *, e_age_s=None, e_ages=None, ram_ea_s=None, episodes_only=False, extra_wrs=[]):
        # TODO: Have it read from config file.
        args = f' --exp {self.exp} --policy {self.__class__.__name__}'
        args += f' --region {self.trace_kwargs["region"]}'
        args += f' --sample-ratio {self.trace_kwargs["sample_ratio"]}'
        args += f' --sample-start {self.trace_kwargs["start"]}'
        if "trace_group" in self.trace_kwargs:
            args += f' --trace-group {self.trace_kwargs["trace_group"]}'
        args += f' --supplied-ea {self.supplied_ea}'
        target_wrs_ = list(self.target_wrs) + extra_wrs
        args += ' --target-wrs ' + ' '.join(map('{:g}'.format, target_wrs_))
        args += ' --target-csizes ' + ' '.join(map('{:g}'.format, self.target_cache_sizes))
        args += f' --output-base-dir {self.output_base_dir}'
        # assert e_age_s or e_ages, (e_age_s, e_ages)
        if self.suffix:
            args += f' --suffix {self.suffix} '
        if e_age_s is not None:
            args += f' --eviction-age {e_age_s:g}'
        if ram_ea_s is not None:
            args += f' --ram-eviction-age {ram_ea_s:g}'
        if e_ages:
            args += ' --analysis-eviction-age ' + ' '.join(map('{:g}'.format, e_ages))
        if self.trace_kwargs.get("only_gets", False):
            args += " --only-gets"
        if 'residency_fn' in self.res_fn_kwargs:
            args += ' --residency-fn ' + self.res_fn_kwargs['residency_fn'].__name__
        if self.rl_init_kwargs:
            args += ' --rl-init-kwargs ' + ep_utils.dict_to_arg(self.rl_init_kwargs)
        if not episodes_only:
            if self.train_target_wr is not None:
                args += f" --train-target-wr {self.train_target_wr}"
            if self.train_models:
                args += ' --train-models ' + ' '.join(self.train_models)
            args += ' --no-episodes '
            # args += f' --train-split-secs {self.train_split_secs} '
            args += f' --train-split-secs-start {self.train_ts_start} '
            args += f' --train-split-secs-end {self.train_ts_end} '
        args += f' {self.extra_args} '
        return args

    def get_filenames(self, e_age_s):
        self._init(e_age_s)
        return self.filenames

    def get_out_prefix(self):
        assert self.e_age_s is not None
        return self.train_prefetch_path or (self.result_dir + f"policies_{self.e_age_s:g}")

    def FAIL(self, err):
        """Record unrecoverable failure so that we do not get retried and it gets logged in Experiment."""
        print(err)
        import traceback
        traceback.print_stack()
        with open(self.fail_file, "w") as f:
            f.write(err+"\n")
            traceback.print_stack(file=f)
        sys.exit(65)

    def get_all(self, e_age_s, ram_ea_s=None, reset=False):
        self._init(e_age_s)
        if not reset and all(os.path.exists(f)
                             for f in self.filenames.values()):
            return self.filenames
        self._prep_residencies([e_age_s], ram_ea=ram_ea_s)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.working_dirname, exist_ok=True)
        # Simulation
        assert len(self.residency_lists_) == 1
        self._save_sim_scores(self.rl, self.filenames)
        # Analysis
        self.df_analysis = self.get_analysis([e_age_s])
        self.df_analysis.to_csv(self.filenames['analysis'])
        return self.filenames

    @property
    def rl(self):
        if self.residency_lists_ is None:
            return None
        if len(self.residency_lists_) > 1:
            print("Warning: more than 1 rl")
        return list(self.residency_lists_.values())[0]

    def get_analysis(self, e_ages=None, ram_ea_s=None):
        if e_ages is not None:
            self._prep_residencies(e_ages, ram_ea=ram_ea_s)
        return self.analysis_(self.target_wrs, self.target_cache_sizes)

    def converge(self, guess=None, csize_gb=None, target_wr=None):
        # df = self.get_analysis(start_e_age)
        self.exp = experiments.ExpAnalysisInline('inline', init_guess=guess, policy=self, csize_gb=csize_gb, wr=target_wr, trace_kwargs=self.trace_kwargs)
        self.exp.run()

    def _prep_residencies(self, e_ages, ram_ea=None, res_fn=generate_residencies):
        if self.residency_lists_ is None:
            kwargs = dict(self.res_fn_kwargs)
            # if self.train_target_wr is not None:
            kwargs['residency_fn'] = process_obj_chunk_n_noprefetch_w_accs
            if ram_ea is not None:
                kwargs['e_age_ram'] = ram_ea
            if 'residencylist_class' not in kwargs:
                if 'peak_ts1_start' in self.rl_init_kwargs:
                    kwargs['residencylist_class'] = ResidencyListPeakAware
                else:
                    kwargs['residencylist_class'] = ResidencyListPrefetchAware
            if self.src_policy and self.src_policy.residency_lists_ is not None:
                print(f"Extracting policies from {self.src_policy}")
                self.residency_lists_ = {
                    k: kwargs['residencylist_class'](v.residencies, v.th, e_age=(v.eviction_age_logical, v.eviction_age_physical))
                    for k, v in self.src_policy.residency_lists_.items()}
            else:
                print(f"res_fn_kwargs: {kwargs}")
                self.residency_lists_ = res_fn(
                    e_ages, exp=self.exp,
                    trace_kwargs=self.trace_kwargs,
                    supplied_ea=self.supplied_ea, **kwargs)
        assert all(e_age in self.residency_lists_ for e_age in e_ages)
        self.sort_residencies(self.residency_lists_)

    def _save_sim_scores(self, rl, filenames):
        scores, thresholds = rl.scores, rl.write_rates
        decisions = {}
        for episode, score, th in zip(rl.residencies, scores, thresholds):
            episode.score = score
            episode.threshold = th
            episode.s[f'pol_score__{self.name}'] = score
            episode.s[f'pol_threshold__{self.name}'] = th
            if episode.key not in decisions:
                decisions[episode.key] = []
            # [block_id]
            decisions[episode.key].append(episode.export())
        decisions = {k: tuple(v) for k, v in decisions.items()}
        ep_utils.dump_pkl(decisions, filenames['thresholds'])

        # # Early Evict: {ts: [block_ids]}
        # early_evict = {}
        # for res in rl.residencies:
        #     ts_end = res.ts_logical[1]
        #     early_evict[ts_end] = early_evict.get(ts_end, ()) + (res.key, )
        # ep_utils.dump_pkl(early_evict, filenames['earlyevict'])

        # # Early Admit: {ts: [(block_id, access), ...]}
        # early_admit = {}
        # for res in rl.residencies:
        #     ts_start = res.ts_physical[0]
        #     key_acc = (res.key, (res.offset[0], res.size, ts_start))
        #     early_admit[ts_start] = early_admit.get(ts_start, ()) + (key_acc,)
        # ep_utils.dump_pkl(early_admit, filenames['earlyadmit'])
        return {'thresholds': decisions}

    def analysis_(self, target_write_rates, target_cache_sizes):
        dfs = []
        for rl in self.residency_lists_.values():
            if len(rl.write_rates[rl.timespan_logical > 0]) > 0:
                max_wr_nonzero = rl.write_rates[rl.timespan_logical > 0][-1]
                dfs.append(pd.DataFrame(rl.target_wrs(
                    [max_wr_nonzero], label='Max Write Rate (Not empty)')))
            else:
                print("Warning: No writes for this EA, EA is likely too small")
            if len(rl.write_rates[rl.scores > 0]) > 0:
                max_wr_nonzero = rl.write_rates[rl.scores > 0][-1]
                dfs.append(pd.DataFrame(rl.target_wrs(
                    [max_wr_nonzero], label='Max Write Rate (No waste)')))
            else:
                print("Warning: No writes for this EA, EA is likely too small")
            dfs.append(pd.DataFrame(rl.target_wrs(target_write_rates)))
            dfs.append(pd.DataFrame(rl.target_csizes(target_cache_sizes)))
            dfs.append(pd.DataFrame(rl.target_wrs(
                [rl.write_rates[-1]], label='Max Write Rate')))
        df = pd.concat(dfs)
        df['Policy'] = self.name
        df = df.sort_values('Target')
        return df

    def sort_residencies(self, residency_lists):
        raise NotImplementedError


class PolicyNumAcc(Policy):
    """Sorts residencies by num_accesses, then tie break on size."""

    def __init__(self, **kwargs):
        super().__init__("numaccess", **kwargs)

    def sort_residencies(self, residency_lists):
        for rl in residency_lists.values():
            rl.init()
            # Last index is first key.
            arr = np.stack([rl.sizes, -rl.num_accesses])
            order = np.lexsort(arr)
            rl.apply_policy(order, scores=rl.num_accesses, policy=self.name)
        return residency_lists


class PolicyUtilityFn(Policy):
    def __init__(self, fn=None, name=None, **kwargs):
        super().__init__("utility", **kwargs)
        self._set_utility_fn(fn, name=name)

    def sort_residencies(self, residency_lists):
        for rl in residency_lists.values():
            rl.init(**self.rl_init_kwargs)
            rl.recompute()
            score = self.utility_fn(rl)
            order = score.argsort()[::-1]
            rl.apply_policy(order, scores=score, policy=self.name)
        return residency_lists

    def _set_utility_fn(self, fn, name=None):
        self.utility_fn = fn
        if name is None and fn is not None:
            name = "utility_" + fn.__name__.replace("score_", "")
        if name:
            self.name = name

    def get_analysis_fn(self, fn, e_ages, name=None):
        self._set_utility_fn(fn, name)
        self.residency_lists_ = None
        df = self.get_analysis(e_ages)
        return df


class PolicyUtilitySize(PolicyUtilityFn):
    def __init__(self, **kwargs):
        super().__init__(fn=score_size, **kwargs)


class PolicyUtilitySize2(PolicyUtilityFn):
    def __init__(self, **kwargs):
        super().__init__(fn=score_size_fixed, **kwargs)


class PolicyUtilityServiceTime(PolicyUtilityFn):
    def __init__(self, **kwargs):
        super().__init__(fn=score_service_time, **kwargs)


class PolicyUtilityServiceTimeSize(PolicyUtilityFn):
    def __init__(self, **kwargs):
        super().__init__(fn=score_service_time_size, **kwargs)


class PolicyUtilityServiceTimeSize2(PolicyUtilityFn):
    def __init__(self, **kwargs):
        super().__init__(fn=score_service_time_size_fixed, **kwargs)


class PolicyUtilityServiceTimeDensity(PolicyUtilityFn):
    def __init__(self, **kwargs):
        super().__init__(fn=score_service_time_density, **kwargs)


class PolicyUtilityNormal(PolicyUtilityFn):
    def __init__(self, **kwargs):
        super().__init__(fn=score_normal, **kwargs)


class PolicyUtilityHits(PolicyUtilityFn):
    def __init__(self, **kwargs):
        super().__init__(fn=score_hits_fixed, **kwargs)


class PolicyUtilityHitDensity(PolicyUtilityFn):
    def __init__(self, **kwargs):
        super().__init__(fn=score_hit_density, **kwargs)


class PolicyUtilityOppCost(PolicyUtilityFn):
    def __init__(self, **kwargs):
        super().__init__(fn=score_opp_cost, **kwargs)


class PolicyUtilityPeakServiceTimeSize(PolicyUtilityFn):
    def __init__(self, **kwargs):
        super().__init__(fn=score_peak_service_time_size, **kwargs)


class PolicyUtilityPeakServiceTimeWeightedSize(PolicyUtilityFn):
    def __init__(self, weight=.9, **kwargs):
        def score_peak_service_time_weighted_size(rl):
            return np_safe_div(rl.peak_service_time_saved * weight + rl.service_time_saved * (1 - weight), rl.chunks_written)
        super().__init__(fn=score_peak_service_time_weighted_size, name=f'peak_service_time_w{weight:g}_size', **kwargs)


def score_normal(rl):
    return rl.num_accesses - 1


def score_hits_fixed(rl):
    return rl.iops_saved


def score_size(rl):
    return np_safe_div(rl.num_accesses - 1, rl.sizes)


def score_size_fixed(rl):
    return np_safe_div(rl.iops_saved, rl.chunks_written)


def score_service_time(rl):
    return rl.service_time_saved


def score_service_time_size(rl):
    return np_safe_div(rl.service_time_saved, rl.sizes)


def score_service_time_size_fixed(rl):
    return np_safe_div(rl.service_time_saved, rl.chunks_written)


def score_service_time_density(rl):
    return np_safe_div(rl.service_time_saved, (rl.timespan_logical + rl.eviction_age_logical) * rl.chunks_written)


def score_opp_cost(rl):
    return np_safe_div(rl.num_accesses - 1, rl.timespan_logical + rl.eviction_age_logical)


def score_hit_density(rl):
    return np_safe_div(rl.num_accesses - 1, (rl.timespan_logical + rl.eviction_age_logical) * rl.sizes)


def score_opp_cost_no_ea(rl):
    return np_safe_div(rl.num_accesses - 1, rl.timespan_logical + 1)


def score_hit_density_no_ea(rl):
    return np_safe_div(rl.num_accesses - 1, (rl.timespan_logical + 1) * rl.sizes)


def score_peak_service_time_size(rl):
    return np_safe_div(rl.peak_service_time_saved, rl.chunks_written)


class PolicyHeuristic(Policy):
    def __init__(self, **kwargs):
        super().__init__("heuristic", **kwargs)

    def _init(self, e_age_s):
        super()._init(e_age_s)
        # del self.filenames['earlyevict']
        # del self.filenames['earlyadmit']
        self.filenames['thresholds'] = self.filenames['thresholds'].replace('.shelve', f'.pkl{compress_ext}')
        return self

    def _prep_residencies(self, e_ages, ram_ea=None):
        if ram_ea is not None:
            raise NotImplementedError
        if self.residency_lists_ is None:
            self.residency_lists_ = generate_residencies(
                e_ages,
                residency_fn=process_obj_chunkheuristic,
                residencylist_class=ResidencyListSizeAware,
                trace_kwargs=self.trace_kwargs,
                supplied_ea=self.supplied_ea)
            assert len(self.residency_lists_) == 1
            rl = list(self.residency_lists_.values())[0]
            self.stats, self.episodes = run_heuristic(
                rl.residencies,
                trace_kwargs=self.trace_kwargs)

    def _save_sim_scores(self, rl, filenames):
        decisions = {}
        for episode in self.episodes:
            if episode.key not in decisions:
                decisions[episode.key] = []
            decisions[episode.key].append(episode.export())
        decisions = {k: tuple(v) for k, v in decisions.items()}
        ep_utils.dump_pkl(decisions, filenames['thresholds'])

        return {'scores': decisions}

    def analysis_(self, target_write_rates, target_cache_sizes):
        df = pd.DataFrame(self.stats)
        df['epsiodes_admitted_fraction'] = df['episodes_admitted'] / df['episodes_admitted'].max()
        val = df['write_rates_mb']
        thresholds = np.searchsorted(val, target_write_rates)
        return df.iloc[thresholds]


def score_sts_v(rl, v):
    return rl.service_time_saved[v] / rl.chunks_written[v]


class PolicySTSV(PolicyUtilityServiceTimeSize2):
    def sort_residencies(self, residency_lists):
        for rl in residency_lists.values():
            rl.init(**self.rl_init_kwargs)
            rl.recompute()

            scores_stsv = {x: score_sts_v(rl, x) for x in ['prefetch', 'noprefetch']}
            scores_stsv['bestprefetch'] = np.maximum(scores_stsv['prefetch'], scores_stsv['noprefetch'])
            prefetch_decision = ~(scores_stsv['noprefetch'] > scores_stsv['prefetch'])
            order = scores_stsv['bestprefetch'].argsort()[::-1]
            rl.apply_policy(order, scores=scores_stsv['bestprefetch'], prefetch_decisions=prefetch_decision, policy=self.name)
        return residency_lists

    def analysis_(self, target_write_rates, target_cache_sizes):
        dfs = []
        for rl in self.residency_lists_.values():
            dfs.append(pd.DataFrame(rl.target_wrs(
                [rl.write_rates['bestprefetch'][-1]], label='Max Write Rate')))
            max_wr_nonzero = rl.write_rates['bestprefetch'][rl.timespan_logical > 0][-1]
            dfs.append(pd.DataFrame(rl.target_wrs(
                [max_wr_nonzero], label='Max Write Rate (Not empty)')))
            dfs.append(pd.DataFrame(rl.target_wrs(target_write_rates)))
            dfs.append(pd.DataFrame(rl.target_csizes(target_cache_sizes)))
        df = pd.concat(dfs)
        df['Policy'] = self.name
        return df


class PolicyUtilityVariants(PolicyUtilityFn):
    def __init__(self, filter_sort='flash_prefetch', filter_view='prefetch', **kwargs):
        self.filter_sort = filter_sort
        self.filter_view = filter_view
        super().__init__(**kwargs)

    def sort_residencies(self, residency_lists):
        for rl in residency_lists.values():
            rl.init(**self.rl_init_kwargs)
            rl.recompute()
            rl.filter = self.filter_sort
            rl.recompute()
            score = self.utility_fn(rl)
            order = score.argsort()[::-1]
            rl.apply_policy(order, scores=score, policy=self.name)
            rl.filter = self.filter_view
            rl.recompute()
        return residency_lists


class PolicyUtilityServiceTimeSizeV(PolicyUtilityVariants):
    def __init__(self, **kwargs):
        super().__init__(fn=score_service_time_size_fixed, **kwargs)


if __name__ == '__main__':
    print("Use train.py instead")
    sys.exit(1)
