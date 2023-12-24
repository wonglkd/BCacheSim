"""
Adapters to BCacheSim and CacheLib/CacheBench.

Adapters: static classes, provide functions to read/write. Internal API that should not be used by users.
ExpResults: use adapters, account for differences in available params, normalize.

ReproExp.
"""
import copy
import datetime
import gzip
import json
import os
import pathlib
import random
import sys
import time
# import logging


import compress_json
import compress_pickle
import lzma
import numpy as np
import pandas as pd

from . import ep_utils
from . import local_cluster
from .ep_utils import safe_div


def CachePkl(func):
    def cached_func(filename, *args, use_cache=False, **kwargs):
        if not use_cache or not os.path.exists(filename):
            # Doesn't fit our format, don't cache
            # Maybe print a warning here about using this
            return func(filename, *args, **kwargs)

        cached_filename = filename+".cached.tmp.pkl.gz"
        if os.path.exists(cached_filename) and os.path.getmtime(cached_filename) > os.path.getmtime(__file__):
            # logging.debug(f"Loading {cached_filename}")
            try:
                return compress_pickle.load(cached_filename)
            except Exception as e:
                print(str(e))
        result = func(filename, *args, **kwargs)
        ep_utils.dump_pkl(result, cached_filename, overwrite=True)
        return result
    return cached_func


class CacheAdaptor(object):
    """Static class, static methods."""

    def done(self):
        """Return true if it is done running."""
        raise NotImplementedError

    def launch(self):
        raise NotImplementedError

    def launch_batch(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def read_batch(self):
        raise NotImplementedError


class CacheAnalysisAdaptor(CacheAdaptor):
    """
    Bring in read_analysis_csv
    """
    @staticmethod
    def read(filename):
        df = pd.read_csv(filename)
        df['Filename'] = filename
        df['Avg Eviction Age (s)'] = df['Assumed Eviction Age (s)']
        df['IOPS Saved Ratio'] = df['IOPSSavedRatio']
        df['ServiceTimeSavedRatio'] = df['Service Time Saved Ratio']
        return df


class CacheSimAdaptor(CacheAdaptor):
    @staticmethod
    def _args_to_cfg(args, output_dir, overwrite=False):
        # TODO: Get rid of this hack
        try:
            from .. import simulate_ap as simulate_ap
        except (ValueError, ImportError):
            dirname = os.path.dirname(os.path.abspath(__file__))
            sys.path.append(os.path.join(dirname, '..'))
            import cachesim.simulate_ap as simulate_ap

        parser = simulate_ap.get_parser()
        args = parser.parse_args(args.split())
        filename = os.path.join(output_dir, 'config.json')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # TODO: Also save time generated?
        parser.save(args, filename, format='json_indented', overwrite=overwrite)
        return f' --config {filename} '

    @staticmethod
    def run_batch(output_dir, thresholds,
                  csize=400,
                  region=None, sample_ratio=1, start=0, trace_group=None,
                  queue='par4', venv='cachelib-pypy3-7.3.1',
                  threshold_name='--ap-threshold',
                  policy_dir='offline-ap-',
                  policy_args='--offline-ap --ap opt',
                  policy_dir_suffix='',
                  script='simulate_ap.py',
                  curr_iter=None,
                  return_only_ran_cmds=False,
                  suppress_exists=False,
                  reset=False, run_args='', **kwargs):
        trace = local_cluster.tracefilename(sample_ratio=sample_ratio, region=region, start=start, trace_group=trace_group)
        cnt = 0
        random.seed(1337)
        torun = []
        expected_outputs = []
        os.makedirs(output_dir, exist_ok=True)
        cmds = []
        all_cmds = []
        job_ids = []
        for threshold in thresholds:
            # TODO: Deprecate. Main thing not supplied is threshold.
            expected_dir = output_dir + f'/{policy_dir}{threshold:g}{policy_dir_suffix}_lru_{csize:g}GB'
            file_prefix = expected_dir + '/' + local_cluster.fmt_subtrace("full", start, sample_ratio)
            expected_outputs.append(file_prefix + '_cache_perf.txt')
            args = f'--trace {trace} {policy_args}'
            assert not np.isnan(threshold)
            args += f' {threshold_name} {threshold:g}'
            args += f' --size_gb {csize:g}'
            args += f' -o {output_dir} {run_args}'

            job_id = 'sim__' + file_prefix.replace('/', '__')
            if curr_iter is not None:
                job_id += f'_i={curr_iter}'
            locks = [job_id]

            args += f' --job-id {job_id}'

            failed_run = False
            if os.path.exists(file_prefix + '.lock'):
                if time.time() - os.path.getmtime(file_prefix + '.lock') > 60*20:
                    failed_run = True

            cmd = local_cluster.run(script, args, generate_cmd=True, venv=venv, **kwargs)
            all_cmds.append(cmd)
            job_ids.append(job_id)
            if os.path.exists(expected_outputs[-1]) or os.path.exists(expected_outputs[-1] + ".lzma"):
                if not suppress_exists:
                    print(f"Already exists results: {expected_outputs[-1]}")
            elif local_cluster.jobstatus.get(job_id, '') == 'queueing':
                print("Job already in queue")
            elif os.path.exists(expected_dir) and not reset and not failed_run:
                print(f"Already exists: {expected_dir}")
            else:
                cnt += 1
                args = __class__._args_to_cfg(args, expected_dir, overwrite=reset)
                torun.append((job_id, locks, args))
                cmds.append(cmd)
        random.shuffle(torun)
        for job_id, locks, args in torun:
            local_cluster.run(
                script, args,
                venv=venv, queue=queue,
                brooce_kwargs=dict(id=job_id, locks=locks, killondelay=True),
                **kwargs)
        print(f"{cnt} new jobs launched")
        return expected_outputs, cmds if return_only_ran_cmds else all_cmds, job_ids

    @staticmethod
    def read_batch(output_files, cmds=None):
        results = []
        for i, file in enumerate(output_files):
            if os.path.isfile(file+".lzma"):
                file += ".lzma"
            elif os.path.isfile(file+".limit.lzma"):
                file += ".limit.lzma"
            assert os.path.isfile(file), f"{file} does not exist"
            assert os.path.getsize(file) < 128*1024, f"{file} is big: {os.path.getsize(file)/1024} kB"
            try:
                if file.endswith('.lzma'):
                    try:
                        of_params = compress_json.load(file)
                    except (EOFError, lzma.LZMAError):
                        print(f"EOFError or corrupt input data, try deleting file: {file}", file=sys.stderr)
                        os.unlink(file)
                        raise
                else:
                    with open(file) as f:
                        of_params = json.load(f)
                try:
                    options = of_params['options']
                    options['ap_threshold']
                except TypeError:
                    try:
                        options = json.loads(of_params['options'])
                    except json.JSONDecodeError:
                        options = eval(options)
            except:
                print(file, file=sys.stderr)
                raise
            res = {
                'Write Rate (MB/s)': of_params['results']['FlashWriteRate'],
                'IOPS Saved Ratio': of_params['results']['IOPSSavedRatio'],
                # 'IOPS Saved Ratio': of_params['results']['TotalIOPSSaved'] / of_params['totalIOPS'],
                'Service Time Saved Ratio': of_params['results']['ServiceTimeSavedRatio'],
                'Hit Rate (Hz)': of_params['results']['TotalIOPSSaved'] / of_params['traceSeconds'],
                'AP Threshold': options['ap_threshold'],
                'AP Probability': options['ap_probability'],
                'Cache Size (GB)': options['size_gb'],
                'Wasted': of_params['results']['NumNoHitEvictions'],
                'Evictions': of_params['results']['NumCacheEviction'],
                'Filename': file,
            }

            if cmds:
                res['Command'] = cmds[i]
            caches = {
                '': '',
                'RamCache': 'RAM Cache ',
            }
            for c_, label in caches.items():
                if c_ + 'AvgEvictionAge' not in of_params['results']:
                    continue
                if type(of_params['results'][c_ + 'AvgEvictionAge']) == list:
                    res[label + 'Avg Eviction Age (Logical)'], res[label + 'Avg Eviction Age (s)'] = of_params['results'][c_ + 'AvgEvictionAge']
                    res[label + 'Max Max Interarrival Time (Logical)'], res[label + 'Max Max Interarrival Time (s)'] = of_params['results'][c_ + 'MaxMaxInterarrivalTime']
                    res[label + 'Avg Max Interarrival Time (Logical)'], res[label + 'Avg Max Interarrival Time (s)'] = of_params['results'][c_ + 'AvgMaxInterarrivalTime']
                else:
                    res[label + 'Avg Eviction Age (s)'] = of_params['results'][c_ + 'AvgEvictionAge']
                    if c_ + 'MaxMaxInterarrivalTime' in of_params['results']:
                        res[label + 'Max Max Interarrival Time (s)'] = of_params['results'][c_ + 'MaxMaxInterarrivalTime']
                        res[label + 'Avg Max Interarrival Time (s)'] = of_params['results'][c_ + 'AvgMaxInterarrivalTime']

            rawj = of_params
            dct = res
            pull = {
                'AdmissionPolicy': 'AdmissionPolicy',
                'AdmitBufferSize': 'options/batch_size',
                'SamplingRatio': 'samplingRatio',  # TODO: Deprecate
                'SampleRatio': 'sampleRatio',
                'SampleStart': 'sampleStart',
                'Total IOPS': 'totalIOPS',
                'PrefetchWhenThreshold': 'options/prefetch_when_threshold',
                'ApOpt': 'options/ap',
                'SimApOpt': 'options/ap',
                'SimJobId': 'options/job_id',
                'LogInterval': 'options/log_interval',
            }
            for k, v in pull.items():
                tres = rawj
                found = True
                for vvv in v.split('/'):
                    if vvv not in tres:
                        found = False
                        break
                    tres = tres[vvv]
                if found:
                    dct[k] = tres

            for col in rawj['results']:
                if not col.endswith('_stats') or type(rawj['results'][col]) == dict or type(rawj['results'][col]) == list:
                    dct[col] = rawj['results'][col]
            if 'IOPSSavedAdmitBufferRatio' not in dct and 'TotalIOPSSavedAdmitBuffer' in dct:
                dct['IOPSSavedAdmitBufferRatio'] = dct['TotalIOPSSavedAdmitBuffer'] / dct['Total IOPS']

            dct['Prefetch Ratio'] = safe_div(dct['NumPrefetches'], dct['TotalChunkWritten'])
            # if 'WastedPrefetchRatio' not in dct:
            #     dct['WastedPrefetchRatio'] = safe_div(dct['TotalWastedPrefetches'], dct['TotalWastedPrefetches']+dct['TotalUsefulPrefetches'])

            # pullstats = {
            #     'flashcache/episodes_admitted2': 'TotalEpisodesAdmitted',
            #     'flashcache/episodes_admitted': 'TotalEpisodesAdmitted',
            # }
            # for k, v in pullstats.items():
            #     if v not in dct and k in rawj['stats']:
            #         dct[v] = rawj['stats'][k]

            # if 'flashcache/total_time_in_system' in rawj['stats']:
            #     dct['Mean Time In System (s)'] = safe_div(rawj['stats']['flashcache/total_time_in_system'][1], dct['Evictions'])

            # dct['Wasted % of WR'] = safe_div(dct['NumNoHitEvictions'], dct['TotalChunkWritten'])
            # if 'flashcache/admitted_chunknotinepisode' in rawj['stats']:
            #     dct['Not in Episode (% of WR)'] = safe_div(rawj['stats']['flashcache/admitted_chunknotinepisode'], dct['TotalChunkWritten'])
            # if 'flashcache/warning_admits_partial_episodes' in rawj['stats'] and 'flashcache/episodes_admitted2' in rawj['stats']:
            #     dct['% of Episodes with Partial Admits'] = safe_div(rawj['stats']['flashcache/warning_admits_partial_episodes'], rawj['stats']['flashcache/episodes_admitted2'])
            # if 'flashcache/warning_admits_partial' in rawj['stats'] and 'flashcache/episodes_admitted2' in rawj['stats']:
            #     dct['% of Accesses with Partial Admits'] = safe_div(rawj['stats']['flashcache/warning_admits_partial'], rawj['totalIOPS'])
            # dct['Prefetch WR (MB/s)'] = dct['Prefetch Ratio'] * dct['Write Rate (MB/s)']
            # dct['IOPSSaved / UsefulChunks'] = safe_div(dct['TotalIOPSSaved'], dct['TotalChunkWritten'] - dct['NumNoHitEvictions'])
            # if 'episodes_admitted' in rawj['stats']:
            #     for col in ['episodes_admitted']:
            #         dct[col] = rawj['stats'][col]
            #     dct['IOPSSaved / #Episodes'] = safe_div(dct['TotalIOPSSaved'], dct['episodes_admitted'])

            # dct['Avg NoHit Eviction Age (s)'] = rawj['results']['AvgNoHitEvictionAge'][1]

            results.append(res)

        df = pd.DataFrame(results)
        return df


class CacheBenchAdaptor(CacheAdaptor):
    """For launching CacheLib"""
    base_template = {
        "cache_config": {
            "cacheSizeMB": 256,
            "lruUpdateOnWrite": False,
            "dipperFilePath": "/mnt/ssd/cachelib",
            #     "navyHitsReinsertionThreshold": 1,
            "dipperNavyRegionSize": 144896,
            # "dipperNavyRegionSize": 135168,
            "numPools": 1,
            "poolRebalanceIntervalSec": 0,
            "allocFactor": 1.1,
            "useTraceTimeStamp": True,
            "tickerSynchingSeconds": 60,
            "printNvmCounters": True,
            "dipperNavySizeClasses": [],
            "dipperNavyBigHashSizePct": 0,
            # // "mlNvmAdmissionPolicyLocation": """,
            "dipperSizeMB": 4096+768,
        },
        "test_config":
        {
            # Lookaslide is definitely not what we want
            #       // "enableLookaside": true,
            "generator": "warmstorage-chunk-replay",
            #     "wsFilePath": "/mnt/hdd/baleen/run/"
            #     "wsDevicePath": "/dev/sdc"
            "replayGeneratorConfig": {
                "numExtraFields": 3,
                "realtime": False,
                "mlAdmissionConfig": {
                    "numericFeatures": {
                        "op": 4,
                        "namespace": 6,
                        "user": 7
                    }
                },
            },
            "cachePieceSize": 131072,
            "numOps": 0,
            "numThreads": 1,
            # "traceFileName": "file.trace"
        }
    }

    @staticmethod
    def _row2cfg(row_):
        cfg_ = copy.deepcopy(__class__.base_template)
        row_["TraceGroup"] = row_["TraceGroup"].replace("ws/", "")
        cfg_["meta"] = {k: row_[k] for k in ["Region", "SampleRatio", "SampleStart", "TraceGroup", "Cache Size (GB)",
                                             "Target Write Rate",
                                             "AdmissionPolicy", "Prefetch-Range", "Prefetch-When"]}
        # , "AP Threshold", "AP Probability"

        cfg_["cache_config"]["cacheSizeMB"] = max(384, int((10/366.47461*410)*0.01*row_['SampleRatio'] * 1024))
        cfg_["cache_config"]["dipperSizeMB"] = max(1024, int((row_["Cache Size (GB)"]/366.47461*410)*0.01*row_['SampleRatio'] * 1024)) - cfg_["cache_config"]["cacheSizeMB"]
        cfg_["test_config"]["traceFileName"] = local_cluster.tracefilename(region=row_['Region'], sample_ratio=row_['SampleRatio'], trace_group=row_['TraceGroup'], start=row_['SampleStart'])
        cfg_["test_config"]["admissionPolicy"] = row_["AdmissionPolicy"]
        # cfg_["test_config"]["numThreads"] = 1
        cfg_["test_config"]["numThreads"] = 2
        cfg_["cache_config"]["nvmAdmissionPolicy"] = "AcceptAll"
        if row_["TraceGroup"] == "201910":
            # block_id io_offset io_size op_time op_name pipeline user_namespace user_name
            # 5 + 3
            cfg_["test_config"]["replayGeneratorConfig"]["numExtraFields"] = 3
            cfg_["test_config"]["replayGeneratorConfig"]["mlAdmissionConfig"]["numericFeatures"] = {
                "op": 4,
                "namespace": 6,
                "user": 7
            }
        else:
            # block_id io_offset io_size op_time op_name user_namespace user_name host_name op_count
            # block_id io_offset io_size op_time op_name user_namespace user_name rs_shard_id op_count host_name
            # 5 + 4
            if row_["TraceGroup"].startswith("2023"):
                cfg_["test_config"]["replayGeneratorConfig"]["numExtraFields"] = 5
            else:
                cfg_["test_config"]["replayGeneratorConfig"]["numExtraFields"] = 4
            cfg_["test_config"]["replayGeneratorConfig"]["mlAdmissionConfig"]["numericFeatures"] = {
                "op": 4,
                "namespace": 5,
                "user": 6
            }
        if row_["AdmissionPolicy"] == "CoinFlipDet-P":
            #         cfg_["cache_config"]["coinFlipProbability"] = row_["AP Probability"]
            cfg_["test_config"]["coinFlipProbability"] = row_["AP Probability"]
        elif row_["AdmissionPolicy"] == "RejectX":
            num_items = (cfg_["cache_config"]["dipperSizeMB"]+cfg_["cache_config"]["cacheSizeMB"]) * 8
            cfg_["test_config"]["rejectFirstSplits"] = 12
            cfg_["test_config"]["rejectFirstEntries"] = int(row_["AP Probability"] * num_items)
    #     navyAdmissionRejectFirstEntries
    #     navyAdmissionRejectFirstSplits = 6
        elif row_["AdmissionPolicy"] == "NewMLAP":
            cfg_["cache_config"]["apThreshold"] = row_["AP Threshold"]
            cfg_["cache_config"]["nvmAdmissionPolicy"] = "AcceptAll"
            cjs_ = compress_json.load(row_['Filename'])
            cfg_["test_config"]["replayGeneratorConfig"]["mlAdmissionConfig"]["modelPath"] = cjs_['options']['learned_ap_model_path']
            cfg_["test_config"]["replayGeneratorConfig"]["mlAdmissionConfig"]["admissionThreshold"] = row_["AP Threshold"]
        elif row_["AdmissionPolicy"] not in ("AcceptAll", "RejectAll"):
            raise NotImplementedError(row_["AdmissionPolicy"])

        if 'predict' in row_['Prefetch-Range']:
            cfg_["test_config"]["replayGeneratorConfig"]["prefetchingConfig"] = {
                "when": row_['Prefetch-When'],
                "range": 'predict' if row_['Prefetch-Range'].endswith('predict') else row_['Prefetch-Range'],
                "modelPath": cjs_['options']['prefetcher_model_path'].replace("_{k}.model", ""),
            }
        elif row_['Prefetch-Range'].endswith('all'):
            cfg_["test_config"]["replayGeneratorConfig"]["prefetchingConfig"] = {
                "when": row_['Prefetch-When'],
                "range": 'all',
            }
        elif row_['Prefetch-When'] != 'never':
            raise NotImplementedError(row_['Prefetching'])

        return cfg_

    @staticmethod
    def _write_cfg(filename, row_, cfg_):
        with open(filename, "w") as f:
            f.write("// @nolint\n")
            f.write("// Original Experiment: {}\n".format(row_['ExperimentName']))
            f.write("// Generated: {}\n".format(datetime.datetime.utcnow().isoformat()))
            f.write(json.dumps(cfg_, indent=4))

    @staticmethod
    def _get_cmds(row, expname_,
                  shortpath=None,
                  testbed_cluster="orca",
                  progress_freq=60*5,
                  output_base_dir=local_cluster.OUTPUT_LOCATION,
                  verbose=False):
        # if verbose:
        #     display(pd.DataFrame(row).T[LATEST_COLS+['AP Probability']])
        cfg = __class__._row2cfg(row)
        cfg["test_config"]["traceFileName"] = local_cluster.tracefilename(region=row['Region'], sample_ratio=row['SampleRatio'], start=row['SampleStart'], trace_group=row['TraceGroup'])
        cfg["test_config"]["samplingRatio"] = row['SampleRatio']
        if shortpath is None:
            print("DeprecationWarning: specify shortpath")
            trace_kwargs = dict(trace_group=row['TraceGroup'], region=row['Region'], sample_ratio=row['SampleRatio'], start=row['SampleStart'])
            shortpath = local_cluster.exp_prefix(expname_, trace_kwargs, row['Cache Size (GB)'], row['Target Write Rate'])
            # shortpath = expname_ + "/{TraceGroup}_{Region}_{SampleRatio}_{SampleStart}_{Cache Size (GB):g}GB_WR{Target Write Rate:g}MBS".format(**row)
            shortpath += f"/ap_{row['AdmissionPolicy']}/prefetch_{row['Prefetch-Range']}_{row['Prefetch-When']}"
        path = f"{output_base_dir}/{shortpath}"
        cfg["test_config"]["windowLogFile"] = f"{path}/windows.log"
        if testbed_cluster in ("orca", "narwhal"):
            cfg["test_config"]["mockStoreMounts"] = ["/mnt/hdd", "/mnt/hdd2"]
        if testbed_cluster == "orca":
            cfg["test_config"]["mockStoreDevices"] = ["/dev/sda", "/dev/sdb"]
            cfg["cache_config"]["writeAmpDeviceList"] = ["nvme0n1"]
        elif testbed_cluster == "narwhal":
            # Sus nodes
            cfg["test_config"]["mockStoreDevices"] = ["/dev/sdc", "/dev/sdd"]
        else:
            raise NotImplementedError

        testbed_cmd = f"bash {local_cluster.SIM_LOCATION}testbed/run_cachebench.sh {path} {progress_freq}"
        return {'path': path, 'shortpath': shortpath,
                # 'sim': newcmd,
                'testbed': testbed_cmd,
                'job_id': 'cachebench__' + local_cluster.prep_jobname(shortpath),
                'cfg': cfg,
                'row': row,
                'ramcachesize': cfg["cache_config"]["cacheSizeMB"], 'flashcachesize': cfg["cache_config"]["dipperSizeMB"]}

    @staticmethod
    def _run_cmds(cmds, run_testbed=True, reset=False, **kwargs):
        os.makedirs(cmds['path'], exist_ok=True)
        __class__._write_cfg(f"{cmds['path']}/cachebench.json", cmds['row'], cmds['cfg'])
        # if run_sim:
        #     print("Ran Sim")
        #     local_cluster.run_cmd(cmds['sim'], queue='par6', timeout=3600*24*4, brooce_kwargs={'id': 'cachesim__'+cmds['shortpath'], 'noredislogonsuccess': True})

        windowfile = pathlib.Path(cmds['cfg']['test_config']['windowLogFile'])
        if os.path.exists(cmds['cfg']['test_config']['windowLogFile']+".gz"):
            windowfile = pathlib.Path(cmds['cfg']['test_config']['windowLogFile']+".gz")
        if windowfile.exists():
            print("Already exists: {} {:.1f}M".format(cmds['path'], windowfile.stat().st_size/1048576))
            filesize = windowfile.stat().st_size
            if filesize == 0:
                print(f"Removing file: {filesize}")
                windowfile.unlink()
            elif os.path.exists(f"{cmds['path']}/cachebench.done"):
                print("A done file exists")
            elif windowfile.stat().st_mtime < time.time() - 3600 * 2 and filesize < 200*1024*1024:
                print("File is stale, removing")
                # windowfile.unlink()
            else:
                return False
        if os.path.exists(f"{cmds['path']}/cachebench.done"):
            print("A done file exists")

        print("Need to run: ", cmds['path'])
        if run_testbed:
            print("Ran Testbed")
            local_cluster.launch_cmd(cmds['testbed'], queue='orca', timeout=3600*24*8, job_id=cmds['job_id'], **kwargs)
            return True
        return False

    @staticmethod
    def read_iostat(filename):
        with open(filename) as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                # Incomplete files
                f.seek(0)
    #             print(f.read()[-10:-2])
                log = json.loads(f.read()[:-2] + "]}]}}")
        rows = []
        for entry in log['sysstat']['hosts'][0]['statistics']:
            row = {'Time': entry['timestamp']}
            for k, v in entry['avg-cpu'].items():
                row[f'avg-cpu|{k}'] = v
            for dsk in entry['disk']:
                if dsk['disk_device'] in ('sdb', 'sdd'):
                    continue
                for k, v in dsk.items():
                    if k != 'disk_device':
                        row[dsk['disk_device']+'|'+k] = v
            rows.append(row)
        df = pd.DataFrame(rows)
        df['Time'] = pd.to_datetime(df['Time'])
        return df

    @CachePkl
    @staticmethod
    def read(filename):
        opener = gzip.open if filename.endswith('.gz') else open
        with opener(filename, 'rt') as f:
            contents = f.read()
        entries = contents.split("== Allocator Stats ==")
        rows = []
        bad_cols = {}
        bad_lines = []
        for entry in entries[1:]:
            row = {}
            lines = entry.split("\n")
            for line in lines:
                if "ops completed" in line or line.startswith("="):
                    continue
                if line.startswith("Finished a full run"):
                    continue
                if "e+9Elapsed Real Time" in line:
                    line = line.replace("e+9Elapsed Real Time", "e+9, Elapsed Real Time")
                line_ = line.strip()
                if not line_:
                    continue
                line_ = line_.replace(" Success", ", Success")
                line_ = line_.split(", ")
                for kl in line_:
                    if ":" not in kl and "objectGet" in kl:
                        kl = kl.replace("objectGet ", "objectGet:")
                    kl = kl.split(":")
                    if len(kl) < 2:
                        # print(kl, line_)
                        # Usually jumbled output, file that has not ended or multiple CacheBenches writing to it
                        bad_lines.append([kl, line_])
                        continue
                    k = kl[0].strip()
                    v = kl[1].strip()
                    while k in row:
                        k += "_"
                    v = v.replace(",", "")
                    if v.endswith('%'):
                        v = float(v[:-1])*0.01
                    elif v.endswith(" us"):
                        v = v.replace(" us", "")
                        v = float(v) / 1e6
                    elif v.endswith(" million"):
                        v = v.replace(" million", "")
                        v = float(v) * 1e6
                    else:
                        for suffix in ["GB", "GB/s", "MB/s", "/s"]:
                            if v.endswith(suffix):
                                v = v.replace(suffix, "")
                                k += f" ({suffix})"
                    try:
                        v = float(v)
                        row[k] = v
                    except ValueError:
                        bad_cols[k] = v
            if not row:
                continue
            rows.append(row)
        if bad_cols:
            # logging.debug(bad_cols)
            print(ValueError(f"Problem reading keys as float: {bad_cols}"))
            # raise
        if bad_lines:
            # logging.debug(bad_lines[:10])
            print(ValueError("Lines do not have a colon (and not whitelisted)"))
            print(bad_lines[:10])
            # raise
        return pd.DataFrame(rows)


row2cfg = CacheBenchAdaptor._row2cfg
get_cmds = CacheBenchAdaptor._get_cmds
run_cmds = CacheBenchAdaptor._run_cmds
