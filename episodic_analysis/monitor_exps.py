"""Utilities to help with monitoring of experiments."""
import datetime
import fnmatch
import glob
import logging
import os
import pickle
import sys
import time
import traceback

from pprint import pprint

from collections import Counter
from collections import defaultdict

from . import local_cluster
from .ep_utils import flatten
from .ep_utils import safe_div
from .plotting import maps

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from IPython.display import HTML
from IPython.display import clear_output
from IPython.display import display


def linkify(df, txt_col, link_col):
    df[txt_col] = df.apply(lambda x: '<a href="{}">{}</a>'.format(x[link_col], x[txt_col]) if x[link_col] else x[txt_col], axis=1)


def makerelpath(path):
    return os.path.relpath(path, os.getcwd().replace("/n/home/dlwong/cache-analysis", "/users/dlwong/projects/cache-analysis"))


def get_log_filename(exp):
    return "../../runs/" + os.path.basename(exp.output_base_dir) + "/" + exp.prefix + ".log"


def show_err(exp):
    print(exp.prefix)
    exp.print_err()
    display(HTML(f"<a href='{get_log_filename(exp)}'>See log</a>"))


def process_exp(exp):
    """For map."""
    # TODO: Make best_result less brittle
    # Have a status entry even for those with no results
    return exp.best_result()


def status(all_exps_, *, idx=None, vals=None, cols=['Prefetch-Range', 'Prefetch-When'],
           show_wrs=[10, 34, 50, 100, 150],  # For Analysis-MaxCacheSize (show a few)
           threshold_name='Exp Threshold'):
    status_label = 'Status (P99.9-Util, Avg-Util)'
    if vals is None:
        vals = [status_label, 'WR', 'WRBreakdown',
                'EA',
                'SimTime',
                'P99.9ServiceTimeSavedRatio1',
                'PeakServiceTimeSavedRatio1',
                'ServiceTimeSavedRatio1',
                'ServiceTimeSavedRatio',
                ]
    for k, exps in all_exps_.items():
        results = []
        status_errs = 0
        get_errs = 0
        last_err = None
        for exp in exps:
            try:
                try:
                    bestr = process_exp(exp)
                except:
                    get_errs += 1
                    # bestr = {
                    #     'Avg Eviction Age (s)': -1,
                    #     'Write Rate (MB/s)': 0,
                    # }
                    # TODO: Display a status entry even for those with no best_result
                    continue
                bestr['Stage'] = exp.stage
                bestr['CurrIteration'] = exp.curr_iter
                if exp.curr_thresholds:
                    bestr['Thresholds'] = len(exp.curr_thresholds)
                else:
                    bestr['Thresholds'] = ''
                bestr['LogFilename'] = get_log_filename(exp)
                bestr['Exp Threshold'] = bestr.get(exp.search_col, None)
                if isinstance(bestr, dict):
                    bestr = pd.DataFrame.from_dict([bestr])
                else:
                    bestr = bestr.loc[bestr['Target Write Rate'].isin(show_wrs)]
                    # bestr['Target Write Rate'] = bestr['Target Write Rate'].fillna(bestr['Write Rate (MB/s)'])
                    # bestr = bestr.loc[np.isclose(bestr['Write Rate (MB/s)'], 34, atol=.1)]
                results.append(bestr)
            except:
                status_errs += 1
                last_err = sys.exc_info()
        if status_errs or get_errs:
            print(f'{k} Errs: Status={status_errs}, Get={get_errs}')
            if last_err:
                traceback.print_tb(last_err[-1])
        if not results:
            continue
        print(k)
        df = pd.concat(results)
        # For analysis
        if 'ServiceTimeSavedRatio1' not in df and 'Service Time Saved Ratio' in df:
            df['ServiceTimeSavedRatio1'] = df['Service Time Saved Ratio']
        # if 'PeakServiceTimeSavedRatio1' not in df:
        #     df['PeakServiceTimeSavedRatio1'] = 0
        try:
            # df['LogFilename'] = df['ExperimentPrefix'].apply(lambda x: f"../spring22/{x}.log")
            df['StatusPart2'] = df.apply(lambda x: '({}{}/{})'.format(x['CurrIteration'], x['Stage'][0].upper(), x['Thresholds']), axis=1)
            linkify(df, 'StatusPart2', 'LogFilename')
            df[status_label] = df.apply(lambda x: '{:.2f} {:.2f} {}'.format(
                x.get('P99.9ServiceTimeUtil1', -1),
                x.get('ServiceTimeUtil1', -1),
                x['StatusPart2']), axis=1)
            df['EA'] = df.apply(lambda x: '{:.0f}/{:.0f}={:.1f}'.format(
                x.get('Assumed Eviction Age (s)', -1),
                x.get('Avg Eviction Age (s)', -1),
                safe_div(x.get('Assumed Eviction Age (s)', -1), x.get('Avg Eviction Age (s)', -1))), axis=1)
            df['WR'] = df.apply(lambda x: "{:.1f} (Th: {:.3f})".format(x['Write Rate (MB/s)'], x.get(threshold_name, 0)), axis=1)
            df['WRBreakdown'] = df.apply(lambda x: "W: {:.2f}, PF: {:.2f}".format(x.get('WastedFlashRatio', 0), x.get('Prefetch Ratio', 0)), axis=1)

            df['RunErrFilename'] = df['Filename'].replace("_cache_perf.txt.lzma", ".err.lzma").apply(makerelpath)
            if 'SimJobId' in df:
                df['BrooceUrl'] = df['SimJobId'].apply(lambda x: f"http://localhost:8080/showlog/{x}" if x else x)
                linkify(df, 'WRBreakdown', 'BrooceUrl')
            # TODO: Wait till we have err
            # linkify(df, 'WRBreakdown', 'RunErrFilename')
            df['WR+Breakdown'] = df.apply(lambda x: "{}<br>{}".format(x['WR'], x['WRBreakdown']), axis=1)
            df['SimTime'] = pd.to_timedelta(round(df['SimWallClockTime']), unit='S')
        except:
            with pd.option_context("display.max_columns", 200):
                display(df)
            raise

        vals = [col for col in vals if col in df]
        idx_ = idx
        if idx_ is None:
            idx_ = ['Region', 'SampleRatio', 'SamplingRatio', 'Cache Size (GB)',
                    'SampleStart',
                    'AdmissionPolicy',
                    'Feat-Subset', 'Granularity',
                    'Target Write Rate',
                    'PrefetchWhenThreshold',
                    'LogInterval',
                    'TrainTsStart',
                    'TrainTsEnd',
                    'Policy']
        idx_ = [col for col in idx_ if col in df.columns]
        for col in idx_:
            if col not in ('Region', 'Target Write Rate') and df[col].nunique() == 1:
                print(f"{col}: {df[col].iloc[0]}")
        idx_ = [col for col in idx_ if df[col].nunique() > 1 or col in ('Region', 'Target Write Rate')]
        try:
            df = df.pivot(index=idx_, columns=cols, values=vals)

            def styler(x):
                if type(x) != str:
                    return 'color:grey'
                if 'C/' in x:
                    return None
                if 'W/' in x or 'P/' in x:
                    return 'color:blue'
                return 'color:brown'

            df = df.style.applymap(styler, subset=[status_label])
            with pd.option_context("display.max_columns", 200):
                display(HTML(df.to_html(escape=False)))
        except ValueError:
            display(df[idx_ + ['ExperimentPrefix'] + cols + vals])


def checkup():
    # After logging is first run, a handler is added to the root handler with level NOTSET.
    for handler in logging.getLogger('').handlers:
        if handler.level == 0:
            handler.setLevel(logging.WARNING)


def check_on_exps(exps):
    exps_k = {}
    waiting_exps = [exp for exp in exps if 'wait' in exp.stage]
    stats = {}
    for k in ['pending', 'queueing', 'complete', 'live', 'stale', 'missing']:
        stats[f'total_{k}'] = sum(getattr(exp, k) for exp in waiting_exps if hasattr(exp, k))
    statuses = Counter([exp.stage for exp in exps])
    exps_k['Prep-Stale'] = [exp for exp in exps if exp.prepare_stale()]
    exps_k['Run-Stale'] = [exp for exp in exps if exp.run_stale()]
    exps_k['Remaining'] = [exp for exp in exps if exp.stage not in ('complete', 'failure')]
    exps_k['Waiting'] = waiting_exps
    stats.update({
        'exps_total': len(exps),
        'exps_failed': sum(1 for exp in exps if exp.stage == 'failure'),
        'exps_others': sum(1 for exp in exps if 'wait' not in exp.stage and exp.stage not in ('complete', 'failure')),
        'remaining_exps': len(exps_k['Remaining']),
        'exps_waiting': len(exps_k['Waiting']),
    })
    for ekey in ['Prep-Stale', 'Run-Stale']:
        if exps_k[ekey]:
            statuses[ekey] = len(exps_k[ekey])
    return exps_k, stats, statuses


def add(msg_, qty):
    return msg_.format(qty) if qty > 0 else ''


def save_exps(exps):
    for exp in tqdm(exps):
        exp.save()


def run_exps(exps, queue=None,
             relaunch_idle_wait=60 * 30, run_n=None, relaunch_stale=True,
             update_interval=30,
             limit_running_exps=None,
             cluster_refresh=20, displayer=status, status_kwargs=None,
             do_cleanup=True,
             **run_kwargs):
    """
    run_exps.

    To run one more iteration, set stage of completed experiments to 'prepare'.
    """
    status_kwargs = status_kwargs or {}
    if displayer is None:
        displayer = lambda *args, **kwargs: None
    all_exps_ = exps
    all_exps_orig = all_exps_
    if type(exps) == dict:
        exps = flatten(exps.values())
    idxes = set()
    for exp in exps:
        eidx = exp.prefix
        if eidx in idxes:
            raise Exception(f"Duplicate Experiment ID {eidx}")
        idxes.add(eidx)

    exps = sorted(exps, key=lambda x: (x.trace_kwargs['start'], maps.REGIONS.index(x.trace_kwargs['region']), x.prefix))

    if run_n is not None:
        exps = exps[:run_n]
    last_update_time = time.time()
    last_job_refresh = 0
    print("Running")
    sleep_interval = 30
    wait_time = 0
    last_reset_time = 0
    waiting_states = ('wait', 'complete', 'prepare-wait', 'failure', 'queue')
    waiting_only_states = ('wait', 'prepare-wait', 'queue')
    if queue:
        for exp in exps:
            exp.queue = queue

    logging.debug("Test")
    checkup()

    completed_exps = set()
    for k, exps_ in all_exps_.items():
        if all(exp.done for exp in exps_):
            completed_exps.add(k)
    if completed_exps:
        print("Completed exps:", completed_exps)
        print("Waiting exps:", [k for k in all_exps_orig if k not in completed_exps])

    while True:
        if all(exp.done or exp.stage == 'failure' for exp in exps):
            if do_cleanup:
                cleanup(exps)
            print("Done!")
            save_exps(exps)
            try:
                displayer(all_exps_orig, **status_kwargs)
            except:
                print("Status failed")
            break

        # Only monitor incomplete exps
        for k, exps_ in all_exps_.items():
            if all(exp.done for exp in exps_):
                completed_exps.add(k)
        all_exps_ = {k: v for k, v in all_exps_.items() if k not in completed_exps}

        exps_to_run = [exp for exp in exps if exp.stage not in ('complete', 'failure')]
        if limit_running_exps is not None:
            exps_to_run = exps_to_run[:limit_running_exps]

        with tqdm(exps_to_run) as tq:
            for exp in tq:
                if exp.stage not in ('complete', 'failure'):
                    tq.set_postfix_str(f"{exp}")
                try:
                    exp.run(relaunch_stale=relaunch_stale, run_kwargs=run_kwargs, do_not_save=True)
                except Exception as e:
                    logging.warning(f"Experiment failed: {exp}")
                    traceback.print_exc()
                    print(exp.prefix, exp.stage, str(e))
                    exp.store_err(e)

        if cluster_refresh is not None and time.time() - last_job_refresh > cluster_refresh:
            last_job_refresh = time.time()
            try:
                local_cluster.getjobstatus()
            except Exception as e:
                print(str(e))

        if wait_time > relaunch_idle_wait and time.time() - last_reset_time > relaunch_idle_wait:
            last_reset_time = time.time()
            relaunch(exps_to_run)

        if not all(exp.stage in waiting_states for exp in exps_to_run):
            wait_time = 0

        time_for_refresh = time.time() - last_update_time > update_interval
        if time_for_refresh or wait_time == 0:
            if time_for_refresh:
                last_update_time = time.time()

            exps_k, stats, statuses = check_on_exps(exps)
            for ekey in ['Prep-Stale', 'Run-Stale']:
                if exps_k[ekey]:
                    print(f'{ekey}:')
                    pprint(exps_k[ekey])

            exps_by_iter = defaultdict(list)
            for exp in exps:
                exps_by_iter[exp.curr_iter].append(exp)

            if time_for_refresh:
                clear_output(wait=True)
                try:
                    if completed_exps:
                        print("Completed exps:", completed_exps)
                        print("Waiting exps:", [k for k in all_exps_orig if k not in completed_exps])
                    displayer(all_exps_, **status_kwargs)
                except:
                    print("Status failed")
            log_str = f"Waited for {wait_time / 60:g} mins ({100-int(stats['remaining_exps']/len(exps)*100)}% done:"
            log_str += " {remaining_exps}/{exps_total} exps remaining, {exps_waiting} waiting, {exps_failed} failed, others: {exps_others})".format(**stats)
            print(f"Last update: {datetime.datetime.now()}")
            print(log_str)
            print(dict(statuses))
            if cluster_refresh is not None:
                try:
                    local_cluster.getjobstatus()
                except Exception as e:
                    print(str(e))
                # TODO: Only count those from our queue + gen-par4
                counts_by_status = {k: len(v) for k, v in local_cluster.jobs_by_status.items()}
                print(f"Cluster: {counts_by_status}")
                job_ids_run = flatten(exp.state['job_ids'][exp.curr_iter] for exp in exps_to_run if exp.stage == 'wait' and 'job_ids' in exp.state)
                job_ids_prep = flatten(exp.state['prepare_job_ids'][exp.curr_iter].values() for exp in exps_to_run if exp.stage == 'prepare-wait' and 'prepare_job_ids' in exp.state)
                print(f"Tracking #Jobs: {len(job_ids_run)}, #Prep: {len(job_ids_prep)}")
                counts_by_status_us = {k: sum(1 for vv in v if vv in job_ids_run) for k, v in local_cluster.jobs_by_status.items()}
                counts_by_status_us = {k: v for k, v in counts_by_status_us.items() if v > 0}
                counts_by_status_us_prep = {k: sum(1 for vv in v if vv in job_ids_prep) for k, v in local_cluster.jobs_by_status.items()}
                counts_by_status_us_prep = {k: v for k, v in counts_by_status_us_prep.items() if v > 0}
                if counts_by_status_us:
                    print(f"Cluster(Us-Run): {counts_by_status_us}")
                if counts_by_status_us_prep:
                    print(f"Cluster(Us-Prep): {counts_by_status_us_prep}")
                if stats["total_missing"] > 0:
                    for exp in exps_to_run:
                        if hasattr(exp, "missing") and exp.missing > 0:
                            try:
                                exp.run()
                            except Exception as e:
                                print(str(e))
                            if exp.run_stale():
                                logging.warning(f"Relaunching Exp due to missing: {exp}, last_launch: {datetime.datetime.fromtimestamp(exp.last_x['launch'])}")
                                exp.relaunch()
                last_job_refresh = time.time()
            print("Total jobs: Pending={total_pending} (Q={total_queueing}, Live={total_live}, Stale={total_stale}, Unknown={total_missing}), Done={total_complete}".format(**stats))
            for n_iter, exps_ in sorted(exps_by_iter.items()):
                exps_k_, stats_, exps_by_stage = check_on_exps(exps_)
                msg = f"  i={n_iter}"

                msg += add(" Prep={}", exps_by_stage['prepare-wait'])
                msg += add("(Stale={})", len(exps_k_['Prep-Stale']))
                msg += add(" Wait={}", exps_by_stage['wait'])
                msg += add("(Stale={})", len(exps_k_['Run-Stale']))
                msg += add(" Complete={}", exps_by_stage['complete'])
                msg += add(" Failed={}", exps_by_stage['failure'])
                msg += add(" Others={}", stats_['exps_others'])
                if stats_['total_queueing']+stats_['total_pending']+stats_['total_complete'] > 0:
                    msg += " (jobs: "
                    msg += add("Q={}", stats_['total_queueing'])
                    msg += add(" Running={}", stats_['total_pending'])
                    msg += add(" Live={}", stats_['total_live'])
                    msg += add(" Stale={}", stats_['total_stale'])
                    msg += add(" Unknown={}", stats_['total_missing'])
                    msg += add(" Done={}", stats_['total_complete'])
                    msg += ")"
                print(msg)

        if all(exp.stage in waiting_states for exp in exps_to_run) and any(exp.stage in waiting_only_states for exp in exps_to_run):
            if displayer:
                try:
                    displayer(all_exps_, **status_kwargs)
                except:
                    print("Status failed")
            if wait_time >= 30:
                print("Saving exps")
                if wait_time % 600 == 0:
                    save_exps(exps)
                else:
                    save_exps(exps_to_run)
            if do_cleanup and wait_time % 60 == 0:
                cleanup(exps)
            time.sleep(sleep_interval)
            wait_time += sleep_interval


def relaunch(exps, reset=True, reset_failures=False):
    local_cluster.getjobstatus()
    if type(exps) == dict:
        exps = flatten(exps.values())
    for exp in exps:
        exp.relaunch(reset=reset, reset_failures=reset_failures)


def cleanup(exps, remove=True, verbose=False, clean_timeout=60*60):
    prep_dirs = set()
    used_names = set()
    if type(exps) == dict:
        exps = flatten(exps.values())
    for exp in exps:
        if 'filenames' in exp.state:
            for k, fn in exp.filenames.items():
                used_names.add(fn)
                if k != 'analysis':
                    dirn = os.path.dirname(fn)
                    assert '/runs/' not in dirn, f"Reults path detected: {dirn}"
                    prep_dirs.add(dirn)
    unused = 0
    used = 0
    others = 0
    if verbose:
        print(prep_dirs)
    for dirn in prep_dirs:
        if not os.path.exists(dirn):
            continue
        with os.scandir(dirn) as it:
            for fn in it:
                if not any(fn.path.endswith(x) for x in ['.model', '.pkl.bz']):  # '.csv'
                    others += 1
                    continue
                if fn.path not in used_names:
                    unused += 1
                    if remove:
                        if time.time() - os.path.getmtime(fn.path) > clean_timeout:
                            os.unlink(fn.path)
                else:
                    used += 1
    if verbose:
        print(used, unused, others)


def save(all_exps, suffix='', common_args=None):
    for exp_name, exps in all_exps.items():
        output_base_dir = common_args["output_base_dir"] if common_args else exps[0].output_base_dir
        with open(f'{output_base_dir}/{exp_name}/exps{suffix}.pkl', 'wb') as f:
            pickle.dump(exps, f)


def load(patterns, output_base_dir, verbose=False):
    patterns = sorted(set(patterns))
    all_exps = {}
    for pattern in patterns:
        for file in glob.glob(f"{output_base_dir}/{pattern}/exps*.pkl"):
            if verbose:
                print(f"Loading {file}")
            exp_name = file.split("/")[-2]
            try:
                exps = pickle.load(open(file, 'rb'))
            except:
                print(f"Failed to load {file}")
                raise

            for exp in exps:
                exp.load()
            if exp_name not in all_exps:
                all_exps[exp_name] = []
            all_exps[exp_name] += exps
    return all_exps


def find_closest(df_, target_wr=34, csize=400):
    df_ = df_[df_['Assumed Eviction Age (s)'] != 0]
    if target_wr is None:
        df_['distance'] = df_.apply(lambda x: ((x['Cache Size (GB)'] - csize)/10) ** 2, axis=1)
    else:
        df_['distance'] = df_.apply(lambda x: ((x['Cache Size (GB)'] - csize)/10) ** 2 + (x['Write Rate (MB/s)'] - target_wr) ** 2, axis=1)
    closest_row = df_.iloc[df_['distance'].argsort()[:1]]
    return closest_row


def filter_df_dct(df_, filters, fast=False, use_glob=True):
    """use_close is slow."""
    for k, v in filters.items():
        if k not in df_.columns:
            print(f"Warning: {k} not in columns")
            continue
        if fast:
            df_ = df_[df_[k] == v]
            continue
        if type(v) == float:
            # Orig: df_ = df_[np.isclose(df_[k].astype(np.float64, errors='ignore'), v)]
            df_ = df_[np.isclose(df_[k].astype('float64', errors='ignore'), v)]
            # df_ = df_[np.isclose(df_[k], v)]
        else:
            if not isinstance(v, list) and not isinstance(v, tuple):
                v = [v]
            try:
                vals = df_[k].unique()
            except Exception:
                print(k)
                display(df_[k])
                raise
            selected_vals = []
            for vv in v:
                if vv in ('Baleen*', 'RejectX*', 'CoinFlip*') and use_glob:
                    print("You probably meant to set use_glob=False")
                selected_vals += fnmatch.filter(vals, vv) if type(vv) == str and use_glob and '*' in vv else [vv]
            df_ = df_[df_[k].isin(selected_vals)]
    return df_


filter_dct = filter_df_dct


def guess(wr, csize, df_analysis):
    row = find_closest(df_analysis, target_wr=wr, csize=csize)
    return dict(ea_guess=int(row['Assumed Eviction Age (s)'].values[0]),
                hr_guess=float(row['IOPS Saved Ratio'].values[0]))


def guess_v2(df, *, filters=None,
             wr=None, csize=None,
             allow_mean=False, allow_closest=True,
             verbose=True,
             wr_t='Target Write Rate',
             try_drop=None):
    filters = filters or {}
    filters_orig = dict(filters)
    df_f = filter_df_dct(df, filters, fast=True)
    dropped = []
    if len(df_f) == 0:
        filters = dict(filters)
        if try_drop:
            for k in try_drop:
                if k not in filters:
                    continue
                del filters[k]
                dropped.append(k)
                df_f = filter_df_dct(df, filters, fast=True)
                if len(df_f) > 0:
                    # Success
                    if verbose:
                        print(f"Success after dropping {dropped}")
                    break
        if len(df_f) == 0:
            raise ValueError(f"Invalid filters: {filters}")
            return dict(ea_guess=7400, hr_guess=0.4)
    df = df_f

    f1 = lambda x: abs(x['Cache Size (GB)'] - csize) < 0.01
    f2 = lambda x: x[wr_t] == wr

    df_ = df
    if wr is not None:
        df_ = df_.loc[f2]
    if csize is not None:
        df_ = df_.loc[f1]
    if len(df_) == 1:
        row = df_.iloc[0].dropna().to_dict()
    elif len(df_) > 1:
        if verbose:
            with pd.option_context("display.max_columns", 200):
                cols = ['ExperimentPrefix', 'ExperimentName']
                display(df_[cols])
        if not allow_mean:
            raise ValueError(f"Duplicate rows: {len(df_)}")
        row = df_.mean(numeric_only=True).dropna().to_dict()
        if set(dropped) - set(['SamplingRatio', 'SampleRatio', 'LogInterval']):
            if verbose:
                print(f"Taking mean of {len(df_)} rows, filters: {filters}, dropped: {dropped}")
            # row.pop('AP Threshold', None)
            # row.pop('AP Probability', None)
    else:
        assert len(df_) == 0
        row = find_closest(df, wr, csize)
        if not allow_closest:
            if verbose:
                with pd.option_context("display.max_columns", 200):
                    display(row)
            raise ValueError(f"Cannot find exact for {wr} {csize} {filters}")
        if verbose:
            print(f"Finding closest for {wr} {csize}")
        row = row.to_dict(orient='records')[0]
        # row.pop('AP Threshold', None)
        # row.pop('AP Probability', None)

    if 'SimAp' in dropped or ('SimAp' in row and row['SimAp'] != filters.get('SimAp', None)):
        print("Dropping threshold due to different AP")
        row.pop('AP Threshold', None)
        row.pop('AP Probability', None)

    if 'AP Threshold' in row:
        if row['AP Threshold'] is None or np.isnan(row['AP Threshold']):
            del row['AP Threshold']
    if row.get('AP Probability', None) is not None and not np.isnan(row.get('AP Probability', None)):
        row['AP Threshold'] = row['AP Probability']

    fillin = {
        'Avg Eviction Age (s)': 'Assumed Eviction Age (s)',  # Temp patch for bug in analysis.py
        'IOPS Saved Ratio': 'IOPSSavedRatio',
    }
    for check_na, replace_by in fillin.items():
        if (check_na not in row or np.isnan(row[check_na])) and replace_by in row and not np.isnan(row[replace_by]):
            row[check_na] = row[replace_by]

    result = dict(ea_guess=row['Avg Eviction Age (s)'],
                  ea_ram_guess=row.get('RAM Cache Avg Eviction Age (s)', None),
                  stsr_guess=row.get('Service Time Saved Ratio', None),
                  hr_guess=row.get('IOPS Saved Ratio', None),
                  threshold_guess=row.get('AP Threshold', None),
                  prev_exp_init=row.get('ExperimentPrefix', None))
    for k, v in list(result.items()):
        assert type(v) != float or not np.isnan(v), (k, v, filters, filters_orig, wr, csize, row, df_)
        if k in ['ea_guess']:
            assert v is not None, (k, result, filters, filters_orig, wr, csize, row, df_)
            if v == 0:
                print("EA cannot be 0! Unsetting")
                del result[k]
    return result


def plot_progress(exp):
    import matplotlib.pyplot as plt
    with pd.option_context('display.float_format', '{:.3f}'.format):
        prog = exp.progress()
        plt.figure()
        prog.plot(x='Iteration', y='Assumed Eviction Age (s)')
        ax = plt.gca()
        prog.plot(x='Iteration', y='Avg Eviction Age (s)', ax=ax)
        plt.ylim(0, None)
        display(prog)
