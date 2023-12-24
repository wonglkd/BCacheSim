import logging
import os
import time

import numpy as np
import pandas as pd

from .. import local_cluster
from .. import adaptors
from .base import ExpNoIterate
from .helpers import fit_and_get_new, unique_pts


class ExpAnalysis(ExpNoIterate):
    def __init__(self, *args, queue='par6', EPS=[25, 25, 0.01, 0.01], **kwargs):
        super().__init__(*args, EPS=EPS, DAMP_F=.9, queue=queue, **kwargs)
        self.search_col = 'Assumed Eviction Age (s)'
        self.min_search = 1
        self.max_search = 7*24*3600
        self.search_key = 'eviction_age'
        self.target_col = "Write Rate (MB/s)"
        self.target_value = self.config['target_wr']

    def run_stale(self):
        return self.stage == 'wait' and (self.stale > 0 or ('launch' in self.last_x and time.time() - self.last_x['launch'] > 10*60))

    def launch(self):
        # SETUP
        ea_dir = f'i_{self.curr_iter}' + '_ea_{eviction_age}'.format(**self.curr_config)
        self.state['output_dir'] = f'{self.prefix_dir}/{ea_dir}'
        os.makedirs(self.state['output_dir'], exist_ok=True)
        print(f'\n== i={self.curr_iter}: '+'EA={eviction_age}, HR={hit_rate} =='.format(**self.curr_config))
        # END SETUP
        output_files, cmds, job_ids = self._launch_run(self.run_args + ' ' + self.static_run_args)
        self.state['cmds'][self.curr_iter] = cmds
        self.state['result_files'][self.curr_iter] = list(output_files)
        self.state['job_ids'][self.curr_iter] = job_ids

    @property
    def curr_fitted_threshold(self):
        return self.state['fitted_threshold'].get(self.curr_iter, self.curr_config.get('threshold', None))

    def _get_thresholds(self, fitted_ea):
        if fitted_ea is None:
            fitted_ea = 7400
        eas = [3600, 7200, fitted_ea * 0.9, fitted_ea, fitted_ea * 1.1, fitted_ea * 1.5, fitted_ea * 2, fitted_ea * 4]
        if self.config['csize_gb'] < 20:
            eas.append(10)
        if self.config['target_wr'] in ('MAX', 'MAX-OPT'):
            eas.append(200)
        return eas

    def add_helper_thresholds(self):
        orig_len = len(self.curr_thresholds)
        self.add_threshold(max(self.curr_thresholds)*2)
        self.add_threshold(200)
        ret = len(self.curr_thresholds) > orig_len
        if ret:
            os.unlink(self.curr_outputs[0])
        return ret

    def _launch_run(self, run_args=''):
        script = '-B -m episodic_analysis.train'
        output_filename = os.path.join(self.state['output_dir'], "analysis.csv")
        ram_ea_s = self.curr_config.get('eviction_age_ram', None)
        if self.config['ram_csize_gb'] is None:
            ram_ea_s = None
        cmd = self.policy.to_cmd(e_ages=self.curr_thresholds, ram_ea_s=ram_ea_s)
        cmd += ' --output ' + output_filename
        job_id = 'analysis__' + self.state['output_dir'].replace('/', '_') + f'_i={self.curr_iter}'
        if os.path.exists(output_filename):
            print("File already exists: " + output_filename)
            return [output_filename], [cmd], [job_id]
        print('Launching analysis...')
        print(f'Expected: {output_filename}')
        status, cmd_full = local_cluster.run(
            script, cmd,
            brooce_kwargs=dict(id=job_id, locks=[job_id], killondelay=True),
            venv='cachelib-py38', queue=self.queue)
        return [output_filename], [cmd_full], [job_id]

    def _subselect(self, df):
        return df[np.isclose(df['Target Cache Size'], self.config['csize_gb'])]

    def _subselect2(self, df):
        return df[np.isclose(df['Target Write Rate'], self.config['target_wr'])]

    def process(self):
        df = adaptors.CacheAnalysisAdaptor.read(self.curr_outputs[0])
        df['Region'] = self.region
        df_selected = self._subselect(df)

        while len(self.state['results']) <= self.curr_iter:
            self.state['results'].append(None)

        if unique_pts(df_selected, search_col=self.search_col) <= 1:
            if self.add_helper_thresholds():
                self.logger.warning("Not enough thresholds! Trying to add more")
            else:
                self.logger.warning("Not enough points, switching to Target Write Rate")
                df_selected = self._subselect2(df)
                if len(df_selected) <= 1:
                    self.logging.error("Still not enough points after switching to Target WR")
                self.post_process(df_selected, fit_kwargs=dict(target_col="Cache Size (GB)", target_value=self.config['csize_gb']))
        else:
            self.post_process(df_selected)

        self.state['results'][self.curr_iter] = df_selected.to_dict(orient='records')

    def post_process(self, df_selected, fit_kwargs=None):
        kwargs = dict(
            exp=self,
            target_val=self.target_value, prev_config=self.curr_config,
            threshold_now=self.curr_fitted_threshold,
            damp_f=self.config['DAMP_F'], flip=self.flipped_thresholds,
            search_key=self.search_key,
            search_col=self.search_col, target_col=self.target_col)
        if fit_kwargs:
            kwargs.update(fit_kwargs)
        new_config, new_threshold = fit_and_get_new(df_selected, **kwargs)

        self.state['fitted_threshold'][self.curr_iter + 1] = new_threshold
        self.state['configs'][self.curr_iter + 1] = new_config

    def post_process_thresholds(self):
        return len(self.curr_thresholds) > 12

    def target_ready(self):
        dx = self.best_result(i=self.curr_iter)
        return np.abs(dx['Write Rate (MB/s)'] - dx['Target Write Rate']) < .2 and np.abs(dx['Cache Size (GB)'] - dx['Target Cache Size']) < 1

    def best_result(self, *, i=None, config=None, regenerate=False, regenerate_and_save=False):
        if i is None:
            i = self.curr_iter - 1
            if self.stage == 'process' or i+1 in self._best_result_cached or len(self.state['results']) > i+1 and self.state['results'][i+1] is not None:
                i += 1
        if i in self._best_result_cached and not regenerate:
            df = self._best_result_cached[i]
        elif not regenerate and not regenerate_and_save:
            df = pd.DataFrame(self.state['results'][i])
        else:
            df = adaptors.CacheAnalysisAdaptor.read(self.state['result_files'][i][0])
            df = self._subselect(df)
            self._best_result_cached[i] = df
            self.state['results'][i] = df.to_dict(orient='records')
            if regenerate_and_save:
                self.save()
        df['Assumed Eviction Age (s)'] = self.state['configs'][i]['eviction_age']
        df['Assumed Hit Rate'] = self.state['configs'][i]['hit_rate']
        df['Iteration'] = i
        summary = self.summary(df, config=config)
        summary['AdmissionPolicy'] = 'OfflineAnalysis'
        return summary


class ExpAnalysisInline(ExpAnalysis):
    def launch(self):
        self.policy.residency_lists_ = None
        df = self.policy.get_analysis([self.curr_fitted_threshold])
        df['Region'] = self.region
        df['Avg Eviction Age (s)'] = df['Assumed Eviction Age (s)']
        df['Avg Eviction Age (Logical)'] = df['Assumed Eviction Age (Logical)']
        df_selected = self._subselect(df)
        if len(df_selected) == 0:
            logging.error("No data points: possibly target WR is too high?")
        while len(self.state['results']) <= self.curr_iter:
            self.state['results'].append(None)
        self.state['results'][self.curr_iter] = df_selected.to_dict(orient='records')

    def ready(self):
        return True

    def process(self):
        df_selected = pd.DataFrame.from_records(self.state['results'][self.curr_iter])
        self.post_process(df_selected)


class ExpAnalysisMaxWR(ExpAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_col = 'Cache Size (GB)'
        self.target_value = self.config['csize_gb']

    def _subselect(self, df):
        return df[df['Target'] == 'Max Write Rate']

    def target_ready(self):
        dx = self.best_result(i=self.curr_iter)
        return np.abs(dx['Cache Size (GB)'] - self.target_value) < 0.001

    def best_result(self, **kwargs):
        summary = super().best_result(**kwargs)
        summary['AdmissionPolicy'] = 'OfflineAnalysis-MaxWR'
        return summary


class ExpAnalysisMaxWROptimal(ExpAnalysisMaxWR):
    def _subselect(self, df):
        return df[df['Target'] == 'Max Write Rate (No waste)']

    def best_result(self, **kwargs):
        summary = super().best_result(**kwargs)
        summary['AdmissionPolicy'] = 'OfflineAnalysis-MaxWROptimal'
        return summary


class ExpAnalysisFixedEA(ExpAnalysis):
    """Meant for Max-CacheSize, if you fix EA to Max EA"""
    def __init__(self, *args,
                 fixed_ea=3600*24*7,
                 init_guess=dict(ea_guess=3600*24*7, hr_guess=0.4),
                 **kwargs):
        super().__init__(*args, init_guess=init_guess, **kwargs)
        self.target_col = None
        self.target_value = None
        self.state['final_iter'] = True
        self.state['thresholds'][self.curr_iter] = [fixed_ea]

    def converge_ready(self, curr, prev):
        return True

    def target_ready(self):
        return True

    def get_thresholds(self, fitted_threshold):
        return [3600*24*7]

    def process(self):
        df = adaptors.CacheAnalysisAdaptor.read(self.curr_outputs[0])
        df['Region'] = self.region
        df['Avg Eviction Age (s)'] = df['Assumed Eviction Age (s)']
        df_selected = df[df['Target'] == 'Write Rate']
        while len(self.state['results']) <= self.curr_iter:
            self.state['results'].append(None)
        self.state['results'][self.curr_iter] = df_selected.to_dict(orient='records')
        self.state['configs'][self.curr_iter + 1] = self.state['configs'][self.curr_iter]

    def best_result(self, *, i=None, config=None, regenerate=False):
        if i is None:
            i = self.curr_iter - 1  # max_iter - 1
        if i in self._best_result_cached:
            df = self._best_result_cached[i]
        elif not regenerate:
            df = pd.DataFrame(self.state['results'][i])
        else:
            df = adaptors.CacheAnalysisAdaptor.read(self.state['result_files'][i][0])
            self._best_result_cached[i] = df
            self.state['results'][i] = df.to_dict(orient='records')
        df['Assumed Hit Rate'] = self.state['configs'][i]['hit_rate']
        df['Iteration'] = i
        df = self.summary_fill(df)
        df['AdmissionPolicy'] = 'OfflineAnalysis-MaxCacheSize'
        return df
