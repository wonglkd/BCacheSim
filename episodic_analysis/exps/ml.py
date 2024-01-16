import os

import pandas as pd
import numpy as np

from .. import local_cluster
from .base import ExpSizeWR
from .base import queue_gen


class ExpML(ExpSizeWR):
    """Static ML model"""
    def __init__(self, *args, model_exp=None,
                 filtercount=6,
                 granularity=None,
                 learned_size=True,
                 train_ts_start=0,
                 train_ts_end=86400,
                 queue='par4',
                 valid_kwargs=None,
                 **kwargs):
        valid_kwargs = valid_kwargs or []
        valid_kwargs += ['granularity', 'filtercount']
        valid_kwargs += ['learned_size', 'size_opt', 'chunk_sample_ratio', 'label_rejectx']
        valid_kwargs += ['train_ts_start', 'train_ts_end']
        if granularity is None:
            if kwargs.get('prefetch_when', 'never') != 'never':
                granularity = 'block'
            else:
                granularity = 'chunk'
        super().__init__(
            *args,
            filtercount=filtercount, granularity=granularity, learned_size=learned_size,
            train_ts_start=train_ts_start, train_ts_end=train_ts_end,
            queue=queue, valid_kwargs=valid_kwargs, **kwargs)
        model_proj_path = f"{local_cluster.TMP_LOCATION}/{model_exp}"
        eviction_age = 3600*2
        filename = f"ml-ea-{eviction_age:g}.model"
        self.model_path = f"{model_proj_path}/{filename}"
        self.flipped_thresholds = True
        self.target_threshold = .5
        self.min_search = 0.000001
        self.policy.suffix += f'/traints_{train_ts_start}_{train_ts_end}'

    def _get_thresholds(self, fitted_threshold):
        if fitted_threshold is not None:
            thresholds = [fitted_threshold * .5, fitted_threshold, fitted_threshold * 2, .2]
        else:
            thresholds = [.1, .2, .4, .8]
        return thresholds

    @property
    def run_args(self):
        return ""

    def _common_ml_args(self):
        run_args_ = f" --learn-ap-filtercount {self.config['filtercount']}"
        run_args_ += f" --learn-ap-granularity {self.config['granularity']}"
        if 'model_prefetch_offset_start' in self.filenames:
            run_args_ += ' --prefetcher-model-path ' + self.filenames['model_prefetch_offset_start'].replace("_offset_start.model", "_{k}.model")
        run_args_ += ' --offline-ap-decisions ' + self.filenames['thresholds']
        return run_args_

    def _ml_args(self):
        run_args_ = self._common_ml_args()
        # Old ML
        run_args_ += f" --learned-ap-model-path {self.filenames['model']}"
        if 'size_opt' in self.config:
            run_args_ += f" --size-opt {self.config['size_opt']}"
        if self.config['learned_size']:
            run_args_ += " --learned-size"
        return run_args_

    def _launch_run(self, run_args=''):
        print('\nLaunching simulations...')
        run_args_ = run_args + ' ' + self._sim_args()
        run_args_ += self._ml_args()
        return self._exp_run(
            policy_dir='ml-ap-',
            policy_dir_suffix=f'_{self.config["filtercount"]}',
            policy_args='--learned-ap --ap ml',
            venv='cachelib-py38',
            run_args=run_args_)

    def target_ready(self, i=None):
        if i is None:
            i = self.curr_iter
        dx = self.best_result(i=i)
        threshold = 0.5 if self.state.get('final_iter', False) else 2 * max(min(2, self.target_value * 0.25), 0.025 * self.target_value)
        threshold *= 2
        return np.abs(dx['Write Rate (MB/s)'] - dx['Target Write Rate']) < threshold


class ExpNewML(ExpML):
    def __init__(self, *args, valid_kwargs=None, **kwargs):
        valid_kwargs = valid_kwargs or []
        valid_kwargs += ['ap_acc_cutoff', 'ap_feat_subset']
        assert 'learned_size' not in kwargs
        super().__init__(*args, valid_kwargs=valid_kwargs, **kwargs)
        self.policy.train_models.add('admit')
        self.policy.train_target_wr = kwargs['wr']
        ap_acc_cutoff = kwargs['ap_acc_cutoff']
        ap_feat_subset = kwargs['ap_feat_subset']
        self.policy.extra_args += f' --ap-acc-cutoff {ap_acc_cutoff} '
        self.policy.extra_args += f' --ap-feat-subset {ap_feat_subset} '
        self.policy.suffix += f'/fs_{ap_feat_subset}'
        self.policy.suffix += f'/accs_{ap_acc_cutoff}'

        self.model_id = "threshold_binary"
        self.min_search = 0.000001
        self.max_search = 1
        del self.model_path

    def add_helper_thresholds(self):
        orig_len = len(self.curr_thresholds)
        self.add_threshold(.99)
        self.add_threshold(.1)
        return len(self.curr_thresholds) > orig_len

    def _get_thresholds(self, fitted_threshold):
        if fitted_threshold is not None:
            thresholds = [fitted_threshold]  # , .5]
            if self.curr_iter == 0:
                thresholds += [.2, .8]  # .1,
            else:
                thresholds += [fitted_threshold * .8, fitted_threshold * 1.25]
        else:
            # thresholds = [.1, .2, .5, .8]
            thresholds = [.2, .5, .8]
        return thresholds

    def _ml_args(self):
        run_args_ = self._common_ml_args()
        run_args_ += f" --learned-ap-model-path {self.filenames[f'model_admit_{self.model_id}']}"
        if 'prefetch' in self.policy.train_models:
            run_args_ += ' --prefetcher-model-path ' + self.filenames['model_prefetch_offset_start'].replace("_offset_start.model", "_{k}.model")
        # New
        if self.config.get('ap_feat_subset', None) is not None:
            run_args_ += f' --ap-feat-subset {self.config["ap_feat_subset"]} '
        return run_args_

    def _launch_run(self, run_args=''):
        print('\nLaunching simulations...')
        run_args_ = run_args + ' ' + self._sim_args()
        run_args_ += self._ml_args()
        # TODO: add more args like feat subset, number of past history
        return self._exp_run(
            policy_dir='ml-ap-',
            policy_dir_suffix=f'_{self.config["filtercount"]}',
            policy_args='--learned-ap --ap mlnew',
            venv='cachelib-py38',
            run_args=run_args_)


class ExpMLFixedEA(ExpNewML):
    def __init__(self, *args,
                 fixed_ea=3600*24*7,
                 init_guess=dict(ea_guess=3600*24*7, hr_guess=0.4),
                 **kwargs):
        if callable(init_guess):
            init_guess = init_guess()
        init_guess['ea_guess'] = fixed_ea
        self.fixed_ea = fixed_ea
        super().__init__(*args, init_guess=init_guess, **kwargs)

    def converge_ready(self, curr, prev):
        return True

    def process(self):
        super().process()
        if self.curr_iter + 1 in self.state['configs']:
            self.state['configs'][self.curr_iter + 1]['eviction_age'] = self.fixed_ea


class ExpMultipleAPNewML(ExpNewML):
    def __init__(self, *args, **kwargs):
        # Note: ap was renamed to sim_ap
        assert kwargs['sim_ap'] in ['and_mlnewopt']
        super().__init__(*args, **kwargs)

    def _launch_run(self, run_args=''):
        print('\nLaunching simulations...')
        run_args_ = run_args + ' ' + self._sim_args()
        run_args_ += self._ml_args()
        ap_name = self.config['sim_ap']
        dirn = f"{ap_name}-ap_"
        if 'opt' in ap_name:
            opt_ap_threshold = self.config['opt_ap_threshold']
            run_args_ += f' --opt-ap-threshold {opt_ap_threshold:g}'
            dirn += f"opt-{opt_ap_threshold:g}_"
        if 'rejectx' in ap_name:
            rejectx_threshold = self.config['rejectx_ap_threshold']
            rejectx_factor = self.config['rejectx_ap_factor']
            run_args_ += f' --rejectx-ap-threshold {rejectx_threshold:g}'
            run_args_ += f' --rejectx-ap-factor {rejectx_factor:g}'
            dirn += f"rejectx-{rejectx_threshold:g}_{rejectx_factor:g}_"
        if 'ml' in ap_name:
            dirn += "ml-"
        return self._exp_run(
            policy_dir=dirn,
            policy_dir_suffix=f'_{self.config["filtercount"]}',
            policy_args=f'--ap {ap_name} --learned-ap --offline-ap',
            venv='cachelib-py38',
            run_args=run_args_)
