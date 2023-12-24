import traceback

from multiprocessing import Pool

import pandas as pd

from . import maps
from .. import monitor_exps as monitor
from ..ep_utils import flatten
from .. import local_cluster


def extract_exp(exp):
    try:
        d_ = exp.best_result(regenerate=True)
    except:
        try:
            d_ = exp.best_result(regenerate=True, i=exp.curr_iter - 1)
        except Exception as e:
            print(exp)
            traceback.print_exc()
            print(e)
            return None
    if isinstance(d_, dict):
        return pd.DataFrame.from_dict([d_])
    return d_


class ExpLoader(object):
    def __init__(self, output_base_dir=local_cluster.OUTPUT_LOCATION, default_prefix="spring23"):
        self.exp_groups = {}
        self.all_exps = {}
        self.df_sim = {'bestsofar': {}, 'closest': {}}
        self.df_sim_bestsofar = {}
        self.df_sim_closest = {}
        self.dfc_raw = {}
        self.dfc_ = {}
        self.output_base_dir = output_base_dir
        self.default_prefix = default_prefix

    def load_and_read(self, name, *args, **kwargs):
        print(f"Loading {name}")
        self._load_group(name, *args, **kwargs)
        with Pool(processes=16) as pool:
            for v in self.exp_groups.values():
                self._read_exps(v, pool)
        self.all_exps_ = list(flatten(self.all_exps.values()))

    def _load_group(self, name, patterns, prefix=None):
        prefix = prefix or self.default_prefix
        if prefix:
            patterns = [f"{prefix}/{p}" for p in patterns]
        selected_exps = monitor.load(patterns, self.output_base_dir)
        self.exp_groups[name] = selected_exps
        self.all_exps.update(selected_exps)

        for k, v in selected_exps.items():
            print(k, len(v), sum(1 for exp in v if exp.curr_iter > 0), sum(1 for exp in v if exp.stage == 'complete'))

    def _read_exps(self, exp_group, pool):
        try:
            for k, v in exp_group.items():
                es = [exp for exp in v if exp.curr_iter != 0 and exp.stage != 'failure']
                if es:
                    self.df_sim['bestsofar'][k] = pd.concat(pool.imap_unordered(extract_exp, es))
                    self.df_sim['closest'][k] = self.df_sim['bestsofar'][k].loc[lambda x: x['Converged']]
        except Exception as e:
            print(k)
            traceback.print_exc()
            print(e)

    def process(self, which='bestsofar'):
        self.dfc_raw[which], self.dfc_[which] = maps.proc_dfs(self.df_sim[which], self.all_exps)
        self.dfc_raw[which], _ = maps.postproc(self.dfc_raw[which])
        return self.dfc_raw[which]
