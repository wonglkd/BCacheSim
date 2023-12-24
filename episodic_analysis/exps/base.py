import contextlib
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback

from pathlib import Path

import compress_json
import lzma
import numpy as np
import pandas as pd
import redis

from .. import adaptors
from .. import local_cluster
from ..trace_utils import wr_to_dwpd
from .constants import EXP_CONFIG
from .helpers import fit_and_get_new
from .helpers import not_in
from .helpers import unique_pts


partial_schedule = [.2, .4, .8]
sample_schedule = [0.01, 0.1]
queue_gen = 'gen-par4'


def config(csize_gb=400,
           target_wr=34,
           ram_csize_gb=10,
           trace_kwargs=dict(region='UNDEFINED', sample_ratio=1),
           valid_kwargs=[],
           **kwargs
           ):
    """
    target_wr: Target write rate in MB/s
    """
    params = ['eviction_age', 'eviction_age_ram', 'hit_rate', 'service_time_saved_ratio']
    EPS = [300, 3, 0.05, 0.05]
    EPS_FRAC = {'eviction_age': 0.05}
    # EPS = dict(zip(params, EPS))
    # Dampening factor after each iteration (0.5 = bisection method)
    DAMP_F = 0.5
    MAX_ITER = 20
    for k in kwargs:
        assert k in valid_kwargs, f"{k} not valid kwarg, valid are: {valid_kwargs}"
    ret = {**locals(), **kwargs}
    ret.pop('kwargs', None)
    ret.pop('k', None)
    ret.pop('valid_kwargs', None)
    return ret


def init(*, ea_guess=7400, hr_guess=0.4,
         ea_ram_guess=None, stsr_guess=0,
         threshold_guess=None,
         prev_exp_init=None,):
    curr_iter = 0
    configs = {
        -1: [float('inf')] * 4,
        0: [ea_guess, ea_ram_guess, hr_guess, stsr_guess],
    }
    if threshold_guess is not None:
        assert not np.isnan(threshold_guess)
    if ea_guess is not None:
        assert not np.isnan(ea_guess) and ea_guess != 0
    fitted_threshold = {0: threshold_guess}
    stage = 'init'
    results = []
    cmds = {}
    prepare_cmds = {}
    prepare_job_ids = {}
    result_files = {}
    completion_time = {}
    job_ids = {}
    thresholds = {}
    final_iter = False
    return locals()


class ExpBase(object):
    @property
    def stage(self):
        return self.state['stage']

    @stage.setter
    def stage(self, stage):
        self.state['stage'] = stage

    @property
    def done(self):
        return self.stage == 'complete'


class Experiment(ExpBase):
    """Converge on eviction age (till assumed and simulated EA match)tr
    Config: a single cache size and target write rate.
    Contains a state machine.
    """

    def __init__(self, *, name=None, prefix=None, output_base_dir='./', static_run_args='',
                 run_kwargs=None, do_not_load=False):
        assert name is not None and prefix is not None
        self.loaded = False
        self.name = name
        os.makedirs(os.path.join(output_base_dir, name), exist_ok=True)
        self.prefix = prefix
        self.prefix_dir = os.path.join(output_base_dir, prefix)
        self.output_base_dir = output_base_dir
        self.configfile = f'{self.prefix_dir}_config.json'
        if os.path.exists(self.configfile + '.lzma'):
            self.configfile += '.lzma'
        self.logfile = f'{self.prefix_dir}.log'
        self.logger = None
        if not do_not_load:
            self.load()
        self.pending = 0
        self.queueing = 0
        self.complete = 0
        self.live = 0
        self.stale = 0
        self.missing = 0
        # if True, will call exp_run(reset=True) on next run
        self.launch_reset = False
        self.static_run_args = static_run_args
        self.err = None
        if not run_kwargs:
            run_kwargs = {}
        self.run_kwargs = run_kwargs
        self._best_result_cached = {}
        self.suppress_exists = False
        self.last_save = None  # Hash
        # Timestamp
        self.last_x = {'prepare': time.time(), 'launch': time.time()}

    def store_err(self, e):
        self.err = {'stage': self.stage, 'exception': e}
        self.traceback = sys.exc_info()
        self.stage = 'failure'
        self.logger.critical(f"Exp failed at {self.err['stage']}", exc_info=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Avoid serializing traceback
        state.pop("traceback", None)
        return state

    def print_err(self):
        if self.err:
            print("Error on stage:", self.err['stage'])
            print("Exception:", self.err['exception'])
            traceback.print_tb(self.traceback[-1])

    def _init_logger(self):
        if self.logger is None:
            self.logger = logging.getLogger(self.prefix.replace(".", ","))
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                os.makedirs(self.prefix_dir, exist_ok=True)
                handler = logging.FileHandler(self.logfile)
                handler.setFormatter(local_cluster.GlogFormatter())
                self.logger.addHandler(handler)

            def log_msg(msg):
                if msg != '\n':
                    msg = msg.strip("\n")
                    msg = msg.replace("\n", "\n               ")
                    return self.logger.info(msg)

            self.logger.write = log_msg
            self.logger.flush = lambda: None

        if not os.path.islink(f'{self.prefix_dir}/log'):
            os.symlink(self.logfile, f'{self.prefix_dir}/log')

    def load(self):
        self._best_result_cached = {}
        dct_vars = ['result_files', 'prepare_cmds', 'prepare_job_ids', 'job_ids', 'completion_time']
        int_vars = ['launch_resets', 'prepare_resets']
        if os.path.exists(self.configfile + '.lzma'):
            self.configfile += '.lzma'
        if os.path.exists(self.configfile):
            try:
                cfg = compress_json.load(self.configfile)
                for x in ['configs', 'fitted_threshold', 'cmds', 'thresholds'] + dct_vars:
                    if x not in cfg['state']:
                        continue
                    if type(cfg['state'][x]) == dict:
                        cfg['state'][x] = {int(k): v for k, v in cfg['state'][x].items()}
                for x in int_vars:
                    if x not in cfg['state']:
                        continue
                    cfg['state'][x] = int(cfg['state'][x])
                self.state = cfg['state']
                # assert cfg['config']['trace_kwargs'] == self.config['trace_kwargs'], (cfg['config']['trace_kwargs'], self.config['trace_kwargs'])
                if 'start' not in cfg['config']['trace_kwargs']:
                    cfg['config']['trace_kwargs']['start'] = 0
                if cfg['config']['trace_kwargs'] != self.config['trace_kwargs']:
                    print(f"Warning: trace_kwargs not equal to original: saved {cfg['config']['trace_kwargs']}, new {self.config['trace_kwargs']}")
                for k in dct_vars:
                    if k not in self.state:
                        self.state[k] = {}
                for k, v in self.state['configs'].items():
                    if type(v) != dict:
                        self.state['configs'][k] = dict(zip(self.config['params'], v))
                self.loaded = True
                return True
            except (json.JSONDecodeError, EOFError, lzma.LZMAError) as e:
                print(self.prefix)
                print(self.configfile)
                # print(f.read())
                logging.error(f"Failed to load config file {self.configfile}")
                # self.store_err(e)
        return False

    def save(self):
        export = {'config': self.config, 'state': self.state}
        exported = json.dumps(export, indent=2)
        if self.last_save and self.last_save == hash(exported):
            return
        self.last_save = hash(exported)
        # if len(exported) > 2*1024*1024:
        #     logging.warning(f"large file {self.configfile}")
        if not self.configfile.endswith('.lzma'):
            self.configfile += '.lzma'
        tmp_file = self.configfile.replace('.lzma', '.part.lzma')
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_file)
        compress_json.dump(export, tmp_file, json_kwargs=dict(indent=2))
        shutil.move(tmp_file, self.configfile)

    def prepare_stale(self):
        # TODO: Make lock
        return self.stage == 'prepare-wait' and 'prepare' in self.last_x and time.time() - self.last_x['prepare'] > 10*60

    def run_stale(self):
        return self.stage == 'wait' and (self.missing > 0 or self.stale > 0) and time.time() - self.last_x['launch'] > 60*10

    def run(self, run_kwargs=None, relaunch_stale=True, do_not_save=False):
        if self.stage in ('complete', 'failure'):
            return
        if not run_kwargs:
            run_kwargs = {}
        self.run_kwargs = run_kwargs
        self._init_logger()
        with contextlib.redirect_stdout(self.logger):
            if self.stage == 'init':
                self.stage = 'prepare'
                if self.curr_iter - 1 not in self.state['completion_time']:
                    self.state['completion_time'][self.curr_iter - 1] = str(pd.Timestamp.now())
                return  # Do not save
            elif self.stage == 'prepare':
                try:
                    self.prepare()
                    self.stage = 'prepare-wait'
                    self.last_x['prepare'] = time.time()
                    if self.prepare_ready():
                        self.stage = 'get_thresholds'
                except (redis.exceptions.ConnectionError, ConnectionError):
                    traceback.print_exc()
                    self.logger.warning("Connection failed: try preparing again")
            elif self.stage == 'prepare-wait':
                # if all exists
                if self.prepare_ready():
                    self.stage = 'get_thresholds'
                elif relaunch_stale and self.prepare_stale():
                    self.state['prepare_resets'] = self.state.get('prepare_resets', 0) + 1
                    if self.state['prepare_resets'] > 20:
                        raise Exception("Relaunched prepare more than 20 times")
                    print(f"Relaunching Prepare-Stale: n={self.state['prepare_resets']}")
                    print(self.prepare_status())
                    print(self.state['prepare_cmds'][self.curr_iter])

                    self.stage = 'prepare'
            elif self.stage == 'get_thresholds':
                self.set_thresholds()
                self.stage = 'launch'
            elif self.stage.startswith('launch'):
                if self.stage == 'launch_reset':
                    self.state['launch_resets'] = self.state.get('launch_resets', 0) + 1
                    if self.state['launch_resets'] > 20:
                        raise Exception("Relaunched more than 20 times")
                    print(f"Relaunching Run-Stale: n={self.state['launch_resets']}")
                    self.launch_reset = True
                    os.system(f'find {self.prefix_dir} -type d -empty -delete')
                if not self.curr_thresholds:
                    self.stage = 'get_thresholds'
                    return
                if not self.prepare_ready():
                    self.stage = 'prepare'
                    return
                self.pending = 0
                try:
                    self.launch()
                    self.stage = 'wait'
                    self.pending = len(self.curr_outputs)
                    self.last_x['launch'] = time.time()
                except (redis.exceptions.ConnectionError, ConnectionError):
                    traceback.print_exc()
                    self.logger.warning("Connection failed: try launching again")
                self.launch_reset = False
                self.suppress_exists = False
                self.queueing = 0
                self.complete = 0
                self.live = 0
            elif self.stage == 'wait':
                if not self.prepare_ready():
                    self.stage = 'prepare'
                    return
                exists = self.ready()
                lock_files = [fn.replace("_cache_perf.txt", ".lock")
                              for fn in self.curr_outputs
                              if not os.path.exists(fn)]
                live = [os.path.exists(fn) and time.time() - os.path.getmtime(fn) < 600
                        for fn in lock_files]
                stale = [os.path.exists(fn) and time.time() - os.path.getmtime(fn) > 60*20
                         for fn in lock_files]
                # queueing = [not os.path.exists(fn.replace("_cache_perf.txt", ".lock")) and not os.path.exists(fn+'*') for fn in self.curr_outputs]
                queueing = [False] * len(self.curr_outputs)
                failed = [False] * len(self.curr_outputs)

                if self.curr_iter in self.state['job_ids']:
                    queueing = [qq or local_cluster.jobstatus.get(job_id, None) == 'pending'
                                for qq, job_id in zip(queueing, self.state['job_ids'][self.curr_iter])]
                    failed = [local_cluster.jobstatus.get(job_id, None) == 'failed'
                              for qq, job_id in zip(queueing, self.state['job_ids'][self.curr_iter])]
                if all(exists):
                    self.stage = 'process'
                    self.pending = 0
                    self.queueing = 0
                    self.live = 0
                    self.stale = 0
                    self.missing = 0
                    self.failed = 0
                    self.complete = len(self.curr_outputs)
                    print('Done Waiting -- Moving to next Stage: Process')
                else:
                    self.pending = sum(1 for v in exists if not v)
                    self.queueing = sum(1 for v in queueing if v)
                    self.live = sum(1 for v in live if v)
                    self.stale = sum(1 for v in stale if v)
                    self.failed = sum(1 for v in failed if v)
                    self.missing = self.pending - self.queueing - self.live - self.stale
                    # TODO: Log missing since
                    self.complete = len(self.curr_outputs) - self.pending
                    if relaunch_stale and self.run_stale() and self.queueing == 0:
                        if time.time() - self.last_x['launch'] > 60*10:
                            print("Relaunching as some experiments are stale")
                            self.stage = 'launch_reset'
            elif self.stage == 'process':
                self.process()
                # Check to see that we didn't already add thresholds and shift to a different stage
                if self.stage != 'process':
                    return
                if not self.post_process_thresholds():
                    if self.curr_config['eviction_age'] < 1:
                        raise Exception('Failure: diverged. Eviction Age too low.')
                    elif self.curr_iter >= self.config['MAX_ITER']:
                        print('Complete: Max Iterations Reached')
                        self.stage = 'complete'
                    # TODO: Check for % difference
                    elif not (self.converge_ready(self.curr_config, self.prev_config) and self.target_ready()):
                        self.stage = 'prepare'
                        if self.state['final_iter']:
                            check = self.converge_ready(self.curr_config, self.prev_config), self.target_ready()
                            self.logger.warning(f"Resetting final_iter as change exceeded EPS: converge_ready: {check[0]}, target_ready: {check[1]}")
                        self.state['final_iter'] = False
                    elif self.state['final_iter']:
                        print('Complete: Converged, Last Iteration')
                        self.stage = 'complete'
                    else:
                        print(f'Almost Complete: Converged on {self.next_config}. Running final iteration.')
                        self.state['final_iter'] = True
                        self.stage = 'prepare'
                    self.state['completion_time'][self.curr_iter] = str(pd.Timestamp.now())
                    self.curr_iter += 1
            else:
                raise NotImplementedError(self.stage)
            if not do_not_save:
                self.save()

    def converge_ready(self, curr, prev):
        epses = dict(zip(self.config['params'], self.config['EPS']))
        # Allow for % for EA. With low WR and high EA, 300 is too tight when EA
        # is 25000.
        for k, v in self.config['EPS_FRAC'].items():
            epses[k] = max(epses[k], self.config['EPS_FRAC'][k] * curr[k])
        assert len(epses) == len(self.config['params'])
        return all(abs(curr[k] - prev[k]) < epses[k]
                   for k in self.config['params']
                   if curr.get(k, None) is not None and prev.get(k, None) is not None)

    def target_ready(self, i=None):
        if i is None:
            i = self.curr_iter
        dx = self.best_result(i=i)
        if 'DWPD' in dx and 'Target DWPD' in dx:
            return np.abs(dx['DWPD'] - dx['Target DWPD']) < 0.5
        threshold = 0.5 if self.state.get('final_iter', False) else max(min(2, self.target_value * 0.25), 0.025 * self.target_value)
        return np.abs(dx['Write Rate (MB/s)'] - dx['Target Write Rate']) < threshold

    def relaunch(self, reset=True, reset_failures=False):
        if self.stage == 'wait' or (self.stage == 'failure' and reset_failures):
            if self.stage == 'failure':
                self.err = None
            self.stage = 'launch_reset' if reset else 'launch'
        if self.stage == 'prepare-wait' and reset:
            self.stage = 'prepare'

    def reset_failure(self):
        self.relaunch(reset_failures=True)

    def add_helper_thresholds(self):
        return False

    def post_process_thresholds(self):
        return False

    @property
    def curr_outputs(self):
        return self.state['result_files'][self.curr_iter]

    def results(self, i=None):
        if i is None:
            i = self.curr_iter
        return pd.DataFrame(self.state['results'][i])

    @property
    def curr_cmds(self):
        return self.state['cmds'][self.curr_iter]

    @property
    def curr_iter(self):
        return self.state['curr_iter']

    @curr_iter.setter
    def curr_iter(self, curr_iter):
        self.state['curr_iter'] = curr_iter

    @property
    def prev_config(self):
        return self.get_config_i(self.curr_iter - 1)

    @property
    def curr_config(self):
        return self.get_config_i(self.curr_iter)

    @property
    def next_config(self):
        return self.get_config_i(self.curr_iter + 1)

    def get_config_i(self, i):
        config = self.state['configs'][i]
        if type(config) != dict:
            config = dict(zip(self.config['params'], config))
        return config

    @property
    def curr_config_raw(self):
        return self.state['configs'][self.curr_iter]

    @property
    def curr_fitted_threshold(self):
        return self.state['fitted_threshold'].get(self.curr_iter, None)

    @property
    def curr_thresholds(self):
        return self.state['thresholds'].get(self.curr_iter, None)

    def prepare(self):
        pass

    def prepare_ready(self):
        return True

    def launch(self):
        raise NotImplementedError

    def ready(self):
        return [os.path.exists(fn) or os.path.exists(fn + ".limit.lzma") or os.path.exists(fn + ".lzma") for fn in self.curr_outputs]

    def process(self):
        raise NotImplementedError

    def best_result(self):
        raise NotImplementedError

    def best_json(self, *args, **kwargs):
        return compress_json.load(self.best_result(*args, **kwargs)['Filename'])

    def _get_thresholds(self, fitted_threshold):
        raise NotImplementedError

    def get_thresholds(self, fitted_threshold):
        thresholds = self._get_thresholds(fitted_threshold)
        if self.max_search is not None and max(thresholds) > self.max_search:
            thresholds = [t for t in thresholds if t <= self.max_search]
            thresholds.append((self.max_search + max(thresholds)) / 2)
        if self.min_search is not None and min(thresholds) < self.min_search:
            thresholds = [t for t in thresholds if t >= self.min_search]
            thresholds.append((self.min_search + min(thresholds)) / 2)

        self.state['thresholds'][self.curr_iter] = np.sort(np.unique(thresholds)).tolist()
        assert len(self.curr_thresholds) > 1, f"Need more points: {self.curr_thresholds}"
        return self.curr_thresholds

    def add_threshold(self, new_threshold):
        if any(np.isclose(new_threshold, th, atol=0.0001) for th in self.curr_thresholds):
            return False
        print(f"Adding threshold: {new_threshold}")
        self.state['thresholds'][self.curr_iter].append(new_threshold)
        self.stage = 'launch'
        self.suppress_exists = True
        return True


class ExpSizeWR(Experiment):
    def __init__(self, *, name=None,
                 policy=None,
                 csize_gb=400, wr=34,
                 ram_csize_gb=None,
                 init_guess=dict(ea_guess=7400, hr_guess=0.4),
                 output_base_dir='./',
                 static_run_args='',
                 queue='par6',
                 suffix='',
                 do_not_load=False,
                 run_kwargs=None,
                 ignore_prefix_mismatch=False,
                 valid_kwargs=None,
                 **kwargs):
        valid_kwargs = valid_kwargs or []
        valid_kwargs += [
            #  any sim
            "batch_size",
            'exp_ap', 'sim_ap',
            'eviction_policy',
            'prefetch_when', 'prefetch_range',
            # obsolete
            'admit_chunk_threshold',
            # episodic_analysis
            'EPS', 'DAMP_F', 'MAX_ITER',
            'no_partial_schedule',
            'log_interval',
            # # any ML policy making use of dynamic features
            # 'granularity',
            # 'filtercount',
            # # new ML
            # 'ap_acc_cutoff', 'ap_feat_subset',
            # # old ML
            # 'size_opt', 'learned_size',
            # # oldML - TODO: remove
            # 'chunk_sample_ratio', 'label_rejectx',
            # # hybrid / rejectx
            # 'hybrid_ap_threshold', 'opt_ap_threshold',
            # 'rejectx_ap_threshold', 'rejectx_ap_factor',
        ]
        if kwargs.get('prefetch_when', None) == 'predict':
            valid_kwargs.append('prefetch_when_threshold')
            # TODO: Find another way of setting the default value for pf_when_threshold.
            if 'prefetch_when_threshold' not in kwargs:
                kwargs['prefetch_when_threshold'] = 0.5
        valid_kwargs = list(set(valid_kwargs))
        self.config = config(csize_gb=csize_gb, ram_csize_gb=ram_csize_gb,
                             target_wr=wr, valid_kwargs=valid_kwargs, **kwargs)
        self.config['ExperimentObject'] = self.__class__.__name__
        # TODO: Remove.

        prefix = local_cluster.exp_prefix(name, self.trace_kwargs, csize_gb, wr, ram_csize_gb=ram_csize_gb, suffix=suffix)
        self.queue = queue
        self.policy = policy
        policy.exp = name
        if not policy.trace_kwargs:
            policy.trace_kwargs = self.trace_kwargs
        policy.output_base_dir = output_base_dir
        self.flipped_thresholds = False
        self.search_col = 'AP Threshold'
        self.min_search = 1  # TODO: Break this out
        self.max_search = None
        self.target_col = 'Write Rate (MB/s)'
        self.target_value = wr
        self.target_threshold = wr  # For target_result
        self._cached_desc = None

        # For prefetcher
        if 'predict' in kwargs.get('prefetch_range', '') or 'predict' in kwargs.get('prefetch_when', ''):
            policy.train_target_wr = wr
            policy.train_models.add('prefetch')

        super().__init__(name=name, prefix=prefix, output_base_dir=output_base_dir, static_run_args=static_run_args,
                         run_kwargs=run_kwargs, do_not_load=do_not_load)

        if not self.loaded:
            if callable(init_guess):
                init_guess = init_guess()
            self.state = init(**init_guess)
            for k, v in self.state['configs'].items():
                if type(v) != dict:
                    self.state['configs'][k] = dict(zip(self.config['params'], v))

        # # For backwards compatibility with running experiments.
        # if 'exp_prefix' in self.state and self.state['exp_prefix'] != prefix:
        #     if ignore_prefix_mismatch:
        #         print("Loading saved prefix")
        #     else:
        #         raise Exception(f"Saved prefix does not match: {self.state['exp_prefix']} {prefix}")
        # else:
        #     self.state['exp_prefix'] = prefix

    def __repr__(self):
        # TODO: Speed up this function / cache the output if not dirty.
        # Need to deal with changes in self.complete
        # if self._cached_desc is not None:
        #     return self._cached_desc
        add_desc = ''
        try:
            if self.stage == 'wait':
                add_desc += f'({self.complete}/{len(self.curr_outputs)})'
            if self.curr_thresholds:
                add_desc += f", #th={len(self.curr_thresholds)}"
            if self.curr_fitted_threshold is not None:
                add_desc += ', threshold={:g}'.format(self.curr_fitted_threshold)
            add_desc += ', ea={:g}'.format(self.curr_config['eviction_age'])
            if self.curr_iter > 0:
                dx = self.best_result()
                if type(dx) == dict or len(dx) == 1:
                    add_desc += ', stsr={:.3f}, hr={:.3f}, wr={:.1f}'.format(dx['Service Time Saved Ratio'], dx['IOPSSavedRatio'], dx['Write Rate (MB/s)'])
        except Exception as e:
            add_desc += str(e)
        if self.stage == 'failure':
            add_desc += str(self.err)
        self._cached_desc = f'Exp({self.prefix}, i={self.curr_iter}, stage={self.stage}{add_desc})'
        return self._cached_desc

    def prepare(self):
        e_age = self.curr_config['eviction_age']
        if math.isnan(e_age) or e_age <= 0:
            raise ValueError(f"e_age: {e_age}")
        self.state['filenames'] = self.policy.get_filenames(e_age)
        self.state['policy_failure_marker'] = self.policy.fail_file
        self.state['prepare_cmds'][self.curr_iter] = {}
        self.state['prepare_job_ids'][self.curr_iter] = {}
        to_run = []

        # -B do not generate pyc
        script = '-B -m episodic_analysis.train'
        job_id = f'gen-episodes_{self.name}_{self.policy.name}_{self.trace_id}_eage-{e_age:g}'
        if self.policy.suffix:
            job_id += local_cluster.prep_jobname(self.policy.suffix)
        locks = [job_id]

        lc_kwargs = dict(
            brooce_kwargs=dict(id=job_id, locks=locks, killondelay=False),
            venv='cachelib-py38', queue=queue_gen, timeout=3600*2)
        extra_wrs = []
        # if self.best_result
        if 'analysis_admitted_wr' in self.curr_config:
            extra_wrs.append(self.curr_config['analysis_admitted_wr'])
        lc_args = [script, self.policy.to_cmd(e_age_s=e_age, ram_ea_s=self.curr_config.get('eviction_age_ram', None), episodes_only=True, extra_wrs=extra_wrs)]
        self.state['prepare_cmds'][self.curr_iter]['opt'] = local_cluster.run(*lc_args, generate_cmd=True, **lc_kwargs)
        self.state['prepare_job_ids'][self.curr_iter]['opt'] = job_id
        to_run.append((lc_args, lc_kwargs))

        if self.policy.train_models:
            locks = []
            job_id = f'train_{self.name}_{self.policy.name}_{self.trace_id}_eage-{e_age}'
            job_id += f'_wr-{self.config["target_wr"]}'
            if self.policy.suffix:
                job_id += local_cluster.prep_jobname(self.policy.suffix)
            if 'admit' in self.policy.train_models:
                job_id += '_admit'
                locks.append(job_id)
            if 'prefetch' in self.policy.train_models:
                job_id += '_prefetch'
                locks.append(job_id)
            lc_kwargs = dict(
                brooce_kwargs=dict(id=job_id, locks=locks, killondelay=False),
                venv='cachelib-py38', queue=queue_gen, timeout=3600*2)
            lc_args = [script, self.policy.to_cmd(e_age_s=e_age, ram_ea_s=self.curr_config.get('eviction_age_ram', None))]
            self.state['prepare_cmds'][self.curr_iter]['ml_model_new'] = local_cluster.run(*lc_args, generate_cmd=True, **lc_kwargs)
            self.state['prepare_job_ids'][self.curr_iter]['opt'] = job_id
            to_run.append((lc_args, lc_kwargs))

        if self.prepare_ready():
            return
        for lc_args, lc_kwargs in to_run:
            local_cluster.run(*lc_args, **lc_kwargs)

    def prepare_ready(self):
        if 'policy_failure_marker' in self.state and os.path.exists(self.state['policy_failure_marker']):
            with open(self.state['policy_failure_marker']) as f:
                msg = f.read()
            self.logger.critical("policies.py: " + msg)
            raise Exception(f"Episode generation (policies.py) failed: see {self.state['policy_failure_marker']}")
        return all(self.prepare_status().values())

    def prepare_status(self):
        return {k: os.path.exists(f) for k, f in self.filenames.items()}

    def set_thresholds(self):
        self.get_thresholds(self.curr_fitted_threshold)

    def add_helper_thresholds(self):
        raise NotImplementedError
        # If you do not want to add any, return False

    def launch(self):
        # SETUP
        ea_dir = f'i_{self.curr_iter}' + '_ea_{eviction_age:g}'.format(**self.curr_config)
        self.state['output_dir'] = f'{self.prefix_dir}/{ea_dir}'
        os.makedirs(self.state['output_dir'], exist_ok=True)
        dest = self.state['output_dir'] + '/df_analysis.csv'
        if not os.path.islink(dest):
            os.symlink(os.path.join(os.path.abspath(self.output_base_dir), self.filenames['analysis']), dest)
        ea_dir_shortcut = f'{self.prefix_dir}/latest_ea'
        if os.path.exists(ea_dir_shortcut):
            os.unlink(ea_dir_shortcut)
        os.symlink(ea_dir, ea_dir_shortcut)
        header_str = f'\n== i={self.curr_iter}: '+'EA={eviction_age:.1f}'.format(**self.curr_config)
        if self.curr_config.get('service_time_saved_ratio', None):
            header_str += f', STSR={self.curr_config["service_time_saved_ratio"]:.5f}'
        header_str += f', IOPSR={self.curr_config["hit_rate"]:.5f}'
        if self.curr_fitted_threshold is not None:
            header_str += f', AP={self.curr_fitted_threshold:.5f}'
        header_str += ' =='
        print(header_str)
        # END SETUP
        run_args_ = self.run_args + ' ' + self.static_run_args
        if not self.state['final_iter']:
            run_args_ += ' --fast '
            if not self.config.get('no_partial_schedule', True) and self.curr_iter < len(partial_schedule):
                run_args_ += f' --limit {partial_schedule[self.curr_iter]} '
        output_files, cmds, job_ids = self._launch_run(run_args_)
        self.state['cmds'][self.curr_iter] = cmds
        self.state['result_files'][self.curr_iter] = list(output_files)
        self.state['job_ids'][self.curr_iter] = job_ids

    def process(self):
        df = adaptors.CacheSimAdaptor.read_batch(self.curr_outputs, cmds=self.curr_cmds)
        while len(self.state['results']) <= self.curr_iter:
            self.state['results'].append(None)
        self.state['results'][self.curr_iter] = df.to_dict(orient='records')
        self._best_result_cached.pop(self.curr_iter, None)
        os.system(f"find {self.state['output_dir']} -name '*.err' -print0 | parallel -0 lzma -9 "+"{}")
        dest = self.state['output_dir'] + '/closest-ap-threshold'
        dx = self.best_result(i=self.curr_iter, regenerate=True)
        pathlink = Path(dest)
        if not pathlink.exists() or pathlink.resolve() != os.path.dirname(dx['Filename']):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(dest + ".err.lzma")
            with contextlib.suppress(FileNotFoundError):
                os.unlink(dest + ".json.lzma")
            with contextlib.suppress(FileNotFoundError):
                os.unlink(dest + ".out.lzma")
            with contextlib.suppress(FileNotFoundError):
                pathlink.unlink()
            os.symlink(os.path.dirname(dx['Filename']), dest)
            os.symlink(dx['Filename'], dest + ".json.lzma")
            filename_prefix = dx['Filename'].split("_cache_perf")[0]
            os.symlink(f"{filename_prefix}.err.lzma", dest+".err.lzma")
            os.symlink(f"{filename_prefix}.out.lzma", dest+".out.lzma")

        if df['Avg Eviction Age (s)'].max() == 0 and self.add_helper_thresholds():
            self.logger.warning("No non-zero eviction ages - trying to add more thresholds")
            return
        if unique_pts(df, search_col=self.search_col) <= 1 and self.add_helper_thresholds():
            self.logger.warning("Not enough thresholds! Trying to add more")
            return

        new_config, new_threshold = fit_and_get_new(
            df,
            exp=self,
            target_val=self.config['target_wr'], prev_config=self.curr_config,
            threshold_now=self.curr_fitted_threshold,
            max_search=self.max_search,
            damp_f=self.config['DAMP_F'], flip=self.flipped_thresholds, search_col=self.search_col)
        self.state['fitted_threshold'][self.curr_iter + 1] = new_threshold
        self.state['configs'][self.curr_iter + 1] = new_config

    def post_process_thresholds(self):
        new_threshold = self.state['fitted_threshold'][self.curr_iter + 1]
        can_add_threshold = not self.target_ready()
        can_add_threshold = can_add_threshold and len(self.curr_thresholds) < 10
        can_add_threshold = can_add_threshold and not (len(self.curr_thresholds) >= 8 and any(np.isclose(new_threshold, th, atol=0.0001) for th in self.curr_thresholds))
        return can_add_threshold and self.add_threshold(new_threshold)

    @property
    def run_args(self):
        return ''

    def _get_thresholds(self, fitted_threshold):
        raise NotImplementedError

    def _sim_args(self):
        """Unlikely to be overriden."""
        run_args_ = ''
        if self.config.get('ram_csize_gb', None) is not None:
            run_args_ += f' --ram-cache --ram-cache-size_gb {self.config["ram_csize_gb"]} '
        if self.config.get('prefetch_range', None) is not None:
            run_args_ += f'--prefetch-when {self.config["prefetch_when"]} --prefetch-range {self.config["prefetch_range"]} '
        for k in ['batch_size', 'prefetch_when_threshold', 'log_interval', 'eviction_policy']:
            if self.config.get(k, None) is not None:
                run_args_ += f' --{k.replace("_", "-")} {self.config[k]} '
        if 'analysis' in self.filenames:
            run_args_ += f' --ep-analysis {self.filenames["analysis"]}'
        return run_args_

    def _launch_run(self, run_args=''):
        print('Launching simulations...')
        print('Thresholds: ', self.curr_thresholds)
        run_args_ = run_args + ' ' + self._sim_args()
        run_args_ += ' --offline-ap-decisions ' + self.filenames['thresholds']
        if 'model_prefetch_offset_start' in self.filenames:
            run_args_ += ' --prefetcher-model-path ' + self.filenames['model_prefetch_offset_start'].replace("_offset_start.model", "_{k}.model")
            self.run_kwargs['venv'] = 'cachelib-py38'
        return self._exp_run(run_args=run_args_)

    def _exp_run(self, **kwargs):
        if 'venv' in kwargs and 'venv' in self.run_kwargs:
            logging.warning(f"Duplicate values found: kwargs={kwargs} run_kwargs={self.run_kwargs}")
        return adaptors.CacheSimAdaptor.run_batch(
            self.state['output_dir'], self.curr_thresholds,
            csize=self.csize,
            queue=self.queue,
            curr_iter=self.curr_iter,
            reset=self.launch_reset,
            suppress_exists=self.suppress_exists,
            **{**kwargs,
               **self.trace_kwargs, **self.run_kwargs})

    @property
    def trace_kwargs(self):
        return self.config['trace_kwargs']

    @property
    def filenames(self):
        """From policies.py"""
        return self.state['filenames']

    @property
    def region(self):
        return self.config['trace_kwargs']['region']

    @property
    def trace_id(self):
        return local_cluster.fmt_trace_id(**self.config['trace_kwargs'])

    @property
    def csize(self):
        return self.config['csize_gb']

    def best_result(self, *, i=None, config=None, regenerate=False, regenerate_and_save=False):
        if not config:
            config = {}
        if i is None:
            i = self.curr_iter - 1
            if self.stage == 'process' or i+1 in self._best_result_cached or len(self.state['results']) > i+1 and self.state['results'][i+1] is not None:
                i += 1
        if i in self._best_result_cached and not regenerate and not config:
            return self._best_result_cached[i]
        elif not regenerate and not regenerate_and_save:
            # Takes ~100ms total
            df = pd.DataFrame(self.state['results'][i])
        else:
            # Takes ~200ms total
            df = adaptors.CacheSimAdaptor.read_batch(self.state['result_files'][i], cmds=self.state['cmds'][i])
            df = df.rename(columns={'Write Speed (MB/s)': 'Write Rate (MB/s)'})
            self.state['results'][i] = df.to_dict(orient='records')
            if regenerate_and_save:
                self.save()
        self._cached_desc = None
        if not hasattr(self, "target_col"):
            self.target_col = "Write Rate (MB/s)"
        if not hasattr(self, "target_value"):
            self.target_value = self.config['target_wr']
        df['Assumed Eviction Age (s)'] = self.state['configs'][i]['eviction_age']
        df['Assumed Hit Rate'] = self.state['configs'][i]['hit_rate']
        df['Iteration'] = i
        df = self.summary(df, config=config)
        self._best_result_cached[i] = df
        return df

    def progress(self, cols=['Iteration', 'AP Threshold', 'Assumed Eviction Age (s)', 'Avg Eviction Age (s)', 'Write Rate (MB/s)', 'Service Time Saved Ratio', 'IOPS Saved Ratio', 'NumThresholds', 'Thresholds', 'DWPD', 'Target DWPD']):
        rows = []
        for i in range(self.curr_iter):
            if i < len(self.state['results']):
                row = self.best_result(i=i)
                row['NumThresholds'] = len(self.state['thresholds'][i])
                row['Thresholds'] = self.state['thresholds'][i]
                rows.append(row)
        df = pd.DataFrame(rows)
        if cols is not None:
            df = df[cols]
        return df

    def summary(self, df, config=None):
        if not config:
            config = {}
        # Find closest WR
        df_ = df
        # Find closest WR that is LOWER than target unless it does not exist
        if self.target_col == "Write Rate (MB/s)":
            df_ = df_[df_[self.target_col] < self.target_value * 1.05]
            if len(df_) == 0:
                print("Warning: only have best_result where Write Rate is higher than Target")
                df_ = df
        closest_row = df_.iloc[(df_[self.target_col] - self.target_value).abs().argsort()[:1]]
        # Get actual HR, WR
        summary = closest_row.to_dict(orient='records')[0]
        config['csize_gb'] = 'Cache Size (GB)'
        for k, v in config.items():
            summary[v] = self.config[k]
        return self.summary_fill(summary)

    def summary_fill(self, summary):
        summary['Converged'] = self.stage == 'complete'
        if not_in(summary, 'Target Write Rate'):
            summary['Target Write Rate'] = self.config['target_wr']
        if not_in(summary, 'Target DWPD') and type(self.config['target_wr']) != str:
            summary['Target DWPD'] = wr_to_dwpd(self.config['target_wr'], self.config['csize_gb'])
        if not_in(summary, 'DWPD'):
            summary['DWPD'] = wr_to_dwpd(summary['Write Rate (MB/s)'], self.config['csize_gb'])
        if not_in(summary, 'Target Cache Size'):
            summary['Target Cache Size'] = self.config['csize_gb']
        if not_in(summary, 'SampleRatio'):
            summary['SampleRatio'] = self.config['trace_kwargs']['sample_ratio']
        if not_in(summary, 'SampleStart'):
            summary['SampleStart'] = self.config['trace_kwargs']['start']
        fillin = {
            # 'Avg Eviction Age (s)': 'Assumed Eviction Age (s)',  # Temp patch for bug in analysis.py
            'IOPS Saved Ratio': 'IOPSSavedRatio',
            'ServiceTimeSavedRatio': 'Service Time Saved Ratio',
        }
        for check_na, replace_by in fillin.items():
            if not_in(summary, check_na, replace=replace_by):
                summary[check_na] = summary[replace_by]
        summary['Policy'] = self.policy.name
        summary['Region'] = self.region
        if 'TraceGroup' in self.config:
            summary['TraceGroup'] = self.config['TraceGroup']
        elif 'trace_group' in self.trace_kwargs:
            summary['TraceGroup'] = self.trace_kwargs['trace_group']
        else:
            summary['TraceGroup'] = local_cluster.infer_trace_group(self.region)
        summary['TraceId'] = summary['TraceGroup'] + '/' + summary['Region']
        summary['ExperimentName'] = self.name
        summary['ExperimentPrefix'] = self.prefix

        for k, v in EXP_CONFIG.items():
            if k in self.config:
                summary[v] = self.config[k]
        if self.config.get('prefetch_range', '').startswith('threshold-'):
            summary['Prefetch-Threshold'] = int(summary['Prefetch-Range'].replace('threshold-', ''))

        if 'policy_size' in self.prefix:
            summary['SizeOn'] = True
        elif 'policy_normal' in self.prefix:
            summary['SizeOn'] = False
        elif 'size_on' in self.prefix or 'size_off' in self.prefix:
            summary['SizeOn'] = 'size_on' in self.prefix

        return summary

    @property
    def target_result(self):
        search_col = 'AP Threshold'
        max_iter = len(self.state['results'])
        df = pd.DataFrame(self.state['results'][-1])
        df = df.rename(columns={'Write Speed (MB/s)': 'Write Rate (MB/s)'})
        # Find closest WR
        closest_row = df.iloc[(df[search_col] - self.target_threshold).abs().argsort()[:1]]
        # Get actual HR, WR
        summary = closest_row.to_dict(orient='records')[0]
        summary['Cache Size (GB)'] = self.config['csize_gb']
        summary['Assumed Eviction Age (s)'] = self.state['configs'][max_iter]['eviction_age']
        summary['Assumed Hit Rate'] = self.state['configs'][max_iter]['hit_rate']
        summary['Converged'] = self.stage == 'complete'
        summary['Target Write Rate'] = self.config['target_wr']
        return summary

    @property
    def analysis_result(self):
        target_wr = self.config['target_wr']
        try:
            df = pd.read_csv(self.filenames['analysis'])
        except FileNotFoundError:
            logging.warning(f"{self.filenames['analysis']} does not exist")
            return {}
        summary = df[(np.isclose(df['Target Write Rate'], target_wr)
                      | np.isclose(df['Target Cache Size'], self.csize))].copy()
        summary['Assumed Eviction Age (s)'] = self.curr_config['eviction_age']
        summary['Assumed Hit Rate'] = self.curr_config['hit_rate']
        summary['Converged'] = self.stage == 'complete'
        summary['Region'] = self.region

        summary = summary.to_dict(orient='records')
        return summary


class ExpNoIterate(ExpSizeWR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state['filenames'] = {}

    def prepare(self):
        return

    def prepare_ready(self):
        return True
