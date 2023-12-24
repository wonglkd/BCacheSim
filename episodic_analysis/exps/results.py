import functools
import glob
import os
import sys

from multiprocessing import Pool
from pprint import pprint

import commentjson
import compress_json
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from .. import adaptors
from .. import ep_utils
from .. import local_cluster
from ..plotting import maps
from ..plotting import processors
from .base import Experiment


def infer_exp(orig, keep_orig=False, output_base_dir=f"{local_cluster.OUTPUT_LOCATION}/spring23"):
    """
    Input: Accepts a DataFrame row, or an Experiment object, or a string with the directory name.

    To be used by ReproResult, SimResult (needs Filename), CacheBenchResult (needs Config).
    """
    orig_ = orig
    if isinstance(orig, str):
        if '_cache_perf.txt' in orig:
            orig = {'Filename': orig}
        elif orig.endswith('cachebench.json'):
            orig = {'CacheBenchConfig': orig}
        elif os.path.exists(os.path.join(output_base_dir, orig)):
            potential_cb = os.path.join(output_base_dir, orig, "cachebench.json")
            if os.path.exists(potential_cb):
                orig = {'CacheBenchConfig': potential_cb}
            else:
                # Potentially detect a Sim Experiment here if needed
                raise NotImplementedError(orig)
        else:
            raise NotImplementedError(orig)
    elif issubclass(orig.__class__, Experiment):
        orig = orig.best_result()
    elif isinstance(orig, pd.Series):
        pass
    elif not isinstance(orig, dict):
        raise NotImplementedError(type(orig), orig)
    if keep_orig:
        orig['Original'] = orig_
    return orig


def get(orig):
    orig = infer_exp(orig)
    if 'Filename' in orig:
        return SimResult(orig)
    elif 'CacheBenchConfig' in orig:
        return CacheBenchResult(orig)
    raise NotImplementedError


class ExpResult(object):
    """
    # TODO: percentile stats, daily peaks
    # TODO: Show progress
    """

    def __init__(self):
        self.err = None
        self.summary = None
        self.raw_summary = None
        self.raw_progress = None
        self._progress = None
        self._pstats = {}
        self.config = {
            'Completed': False,
        }

    @property
    def done(self):
        return self.config['Completed']

    @property
    def progress_loaded(self):
        return self._progress is not None

    def daily_peaks(self, df_=None, xperiod=None,
                    days_col='Days__', time_col=None, **kwargs):
        assert xperiod is not None
        if df_ is None:
            df_ = self.get_progress(xperiod=xperiod, **kwargs)
        if time_col is None:
            time_col = 'Time' if time_col in df_ else "Trace Time"
        if days_col not in df_ and time_col in df_:
            t_col = df_[time_col]
            if not hasattr(t_col, 'dt'):
                t_col = pd.to_datetime(t_col, unit='s')
            d_col = t_col.dt.day
        elif days_col in df_:
            d_col = df_[days_col]
        df_['Day'] = d_col
        cols = [c for c in df_.columns if c not in ['Days', 'secs', 'Day', 'index', days_col]]
        return df_.groupby('Day', sort=False)[cols].max()

    def summary_fill(self, df):
        # Be wary of overwriting
        for k, v in self.config.items():
            df[k] = v
        df["Target Cache Size"] = df["Cache Size (GB)"]
        try:
            if isinstance(df, pd.Series):
                df = pd.DataFrame([df])
            maps.fill_metadata(df)
        except Exception as e:
            print(f"Error: {e}")
            print(type(df))
            print(df.keys())
            # TODO: Temp
            # raise
        # Missing
        return df

    def generate_progress(self,
                          progress,
                          col='Consumed Service Time',
                          idx='Elapsed Trace Time',
                          label=None,
                          nocache=None,
                          sample_ratio=None,
                          xperiods=[60, 60*5, 60*10, 60*15, 60*30, 60*60],
                          pqs=[.5, .95, .99, .995, .999, .9999, .99999, 1],
                          skip_first_hrs=24):
        assert sample_ratio is not None
        stats = {}
        resampled_progress = {}

        for xperiod in xperiods:
            resampled_progress[xperiod] = processors.rejigger_df(progress, col=col, xnum=1, idx=idx, xperiod=xperiod, sample_ratio=sample_ratio)

        mid = ''
        mid2 = ''
        if label == 'GET+PUT':
            mid = 'WithPut'
        elif label == 'PUT':
            mid = 'Put'
        elif label == 'NoCacheGET':
            mid2 = 'NoCache'
        elif label == 'NoCacheGET+PUT':
            mid2 = 'NoCacheWithPut'

        for xperiod in xperiods:
            testr = resampled_progress[xperiod].iloc[int(3600*skip_first_hrs/xperiod):]['Util']
            xperiod_t = f'{xperiod}s' if xperiod < 60 else f'{xperiod/60:g}m'
            stats[f'MeanServiceTimeUtil@{xperiod_t}'] = testr.mean()
            for pq in pqs:
                labelr = f'P{pq*100:g}ServiceTime{mid}{mid2}Util@{xperiod_t}'
                nocache_label = f'P{pq*100:g}ServiceTimeNoCache{mid}Util@{xperiod_t}'
                if nocache is not None and nocache_label not in nocache and 'NoCache' not in label:
                    print(label)
                    print(nocache_label)
                    print(nocache.keys())

                stats[labelr] = testr.quantile(q=pq)
                if nocache is not None and nocache_label in nocache:
                    stats[f'P{pq*100:g}ServiceTime{mid}{mid2}Percent@{xperiod_t}'] = stats[labelr] / nocache[nocache_label] * 100
        return resampled_progress, stats

    def _infer_args(self, op_type, xperiod):
        if op_type is None:
            op_type = 'GET' if self.progress and 'GET' in self.progress else 'GET+PUT'
        xperiod = xperiod if xperiod is not None else self.xperiod
        return op_type, xperiod

    def get_progress(self, op_type=None, xperiod=None):
        op_type, xperiod = self._infer_args(op_type, xperiod)
        return self.progress[op_type][xperiod]

    def get_pstats(self, op_type=None, xperiod=None):
        op_type, xperiod = self._infer_args(op_type, xperiod)
        return self.pstats[op_type]

    @property
    def progress(self):
        if self._progress is None:
            self.load_progress()
        return self._progress

    @property
    def pstats(self):
        if self._pstats is None:
            self.load_progress()
        return self._pstats

    def __repr__(self):
        add_desc = ''
        if self.config['Completed']:
            add_desc += ', Done'
        return f'{__class__.__name__}({self.suffix}{add_desc})'

    def infer_config(self):
        if 'SampleRatio' not in self.config or self.config.get('Region', '') == '':
            suffix = self.config['ExperimentSuffix']
            sr = suffix.split('/')[1].split('_')
            if 'SampleRatio' not in self.config:
                self.config['SampleRatio'] = int(sr[1])
            if self.config.get('Region', '') == '':
                self.config['Region'] = sr[0]


class SimResult(ExpResult):
    def __init__(self, orig, fileprefix=f"{local_cluster.OUTPUT_LOCATION}/spring23"):
        super().__init__()
        orig = infer_exp(orig)
        self.file_summary = orig['Filename']
        for x in ['Region', 'Prefetch-Range', 'Prefetch-When',
                  'Target Write Rate', 'TraceGroup',
                  'ExperimentName', 'Target Write Rate', 'SampleStart', 'SampleRatio', 'Filename',
                  'Cache Size (GB)', 'Assumed Eviction Age (s)']:
            if x in orig:
                self.config[x] = orig[x]
        self.exp_dir = os.path.dirname(self.file_summary)
        self.suffix = self.exp_dir.replace(fileprefix+'/', '')
        self.config['ExperimentSuffix'] = self.suffix
        self.configfile = os.path.join(self.exp_dir, "config.json")
        self.config['ConfigFile'] = self.configfile
        self.file_progress = self.file_summary.replace(".lzma", ".stats.lzma")
        try:
            self.load_summary()
            self.xperiod = self.summary['LogInterval'].iloc[0]
        except:
            pass
        try:
            self.load_config()
        except:
            pass

    def load_config(self):
        self.config_parsed = compress_json.load(self.configfile)
        self.config['MLAPModel'] = self.config_parsed['learned_ap_model_path']
        self.config['AP Threshold'] = self.config_parsed['ap_threshold']

    def get_train_cmd(self):
        polsuffix = '/'.join(self.config['MLAPModel'].split("/")[7:-1])
        self.config['output_base_dir'] = "/".join(self.config['Filename'].split("/")[:7])
        models_ = "prefetch" if 'predict' in self.config["Prefetch-Range"] else ''
        cmd = f'{local_cluster.RUNPY_LOCATION} py -B -m episodic_analysis.train '
        cmd += f'--output-base-dir {self.config["output_base_dir"]} '
        cmd += '--supplied-ea physical '
        cmd += '--rl-init-kwargs filter_=prefetch '
        cmd += f'--train-models admit {models_} --no-episodes '
        cmd += '--ap-acc-cutoff 15 --ap-feat-subset meta+block+chunk '
        cmd += '--policy PolicyUtilityServiceTimeSize2 '
        cmd += '--train-split-secs-start 0 --train-split-secs-end 86400 '
        cmd += f'--trace-group {self.config["TraceGroup"]} '
        cmd += f'--region {self.config["Region"]} '
        cmd += f'--sample-ratio {self.config["SampleRatio"]} --sample-start {self.config["SampleStart"]} '
        cmd += f'--target-wrs 34 --target-csizes {self.config["Cache Size (GB)"]:g} '
        cmd += f'--train-target-wr {self.config["Target Write Rate"]}  '
        cmd += f'--exp {self.config["ExperimentName"]} '
        cmd += f'--suffix /{polsuffix} '
        cmd += f'--eviction-age {self.config["Assumed Eviction Age (s)"]} '
        cmd += '--evaluate '
        cmd += f'--evaluate-ap-threshold {self.config["AP Threshold"]} '
        self.config['accuracy_file'] = self.config['MLAPModel'].replace('_admit_threshold_binary.model', f'_ap_{self.config["AP Threshold"]:g}.models.accuracy')
        return cmd

    def get_ml_accuracy(self, subset='Test', threshold=None):
        df_acc = pd.read_csv(self.config['accuracy_file'])
        if threshold is None:
            threshold = self.config['AP Threshold']
        df_acc = df_acc.set_index(['Label', 'Subset', 'Threshold'])
        return df_acc.loc[('threshold_binary', subset, threshold)]['Accuracy'], df_acc

    def load_summary(self):
        if os.path.exists(self.file_summary):
            self.config['Completed'] = True
            self.raw_summary = adaptors.CacheSimAdaptor.read_batch([self.file_summary]).iloc[0]
            self.config['SampleRatio'] = self.raw_summary['SampleRatio']
            # self.infer_config()
            # TODO: Port over from code that calls adaptors.CacheSimAdaptor.read_batch directly
            # Standardize naming
            # ExpSizeWR.best_result, summary*, summary_fill
            # postprocessing in sim_cache.py / StatsDumper
            # maps.proc_dfs, postproc**
            self.summary = self.postprocess(self.raw_summary)
            self.summary = self.summary_fill(self.summary)

    def postprocess(self, summary):
        # TODO: Postprocess raw summary so that columns match with testbed
        return summary

    def load_progress(self):
        if os.path.exists(self.file_progress):
            self.stats_json = compress_json.load(self.file_progress)
            out_json = compress_json.load(self.file_summary)
            bstats = self.stats_json['batches']
            ys = np.array(bstats['service_time_used_stats'])
            ys_nocache = np.array(bstats['service_time_nocache_stats'])
            xs_nooffset = np.array(bstats['time_phy'])
            xs = xs_nooffset - out_json['stats']['start_ts_phy']
            ys_flashwrites = np.array(bstats['flashcache/keys_written_stats']) * 131072
            newsim_flashwrites = pd.DataFrame({'Elapsed Trace Time': xs, 'Trace Time': xs_nooffset, 'Flash Bytes Written': ys_flashwrites})
            newsim = pd.DataFrame({'Elapsed Trace Time': xs, 'Consumed Service Time': ys, 'Trace Time': xs_nooffset, 'Flash Writes': ys_flashwrites})
            newsim_nocache = pd.DataFrame({'Elapsed Trace Time': xs, 'Consumed Service Time': ys_nocache, 'Trace Time': xs_nooffset})
            assert 'puts_ios_stats' in bstats
            ys_writes = np.array(bstats['service_time_writes_stats'])
            newsim_nocache_put = pd.DataFrame({'Elapsed Trace Time': xs, 'Consumed Service Time': np.add(ys_nocache, ys_writes), 'Trace Time': xs_nooffset})
            newsim_writes = pd.DataFrame({'Elapsed Trace Time': xs, 'Consumed Service Time': ys_writes, 'Trace Time': xs_nooffset})
            newsim_comb = pd.DataFrame({'Elapsed Trace Time': xs, 'Consumed Service Time': np.add(ys, ys_writes), 'Trace Time': xs_nooffset, 'Flash Writes': ys_flashwrites})
            self.raw_progress = {
                'NoCacheGET+PUT': newsim_nocache_put,
                'NoCacheGET': newsim_nocache,
                'GET+PUT': newsim_comb,
                'GET': newsim,
                'PUT': newsim_writes,
                'FlashWriteRate': newsim_flashwrites,
            }

            resp = {}
            stats = {}
            for label, df_ in self.raw_progress.items():
                if label == 'FlashWriteRate':
                    resp[label] = {}
                    for xperiod in [60, 60*5, 60*10, 60*15, 60*30, 60*60]:
                        d1 = processors.resamplet_df(df_, xnum=1, xperiod=xperiod, idx='Elapsed Trace Time')
                        resp[label][xperiod] = d1['Flash Bytes Written'].diff() / 1048576 * 100 / self.config['SampleRatio'] / xperiod
                else:
                    resp[label], stats[label] = self.generate_progress(df_, label=label, sample_ratio=self.config['SampleRatio'], idx='Trace Time')
            self._progress, self._pstats = resp, stats

# TODO: Aggregate multiple ExpResults (different samples, same params).


class CacheBenchResult(ExpResult):
    """For reading CacheLib experiments."""

    def __init__(self, orig, fileprefix=f"{local_cluster.OUTPUT_LOCATION}/spring23"):
        super().__init__()
        orig = infer_exp(orig)
        configfile = orig['CacheBenchConfig']
        self.configfile = configfile
        self.exp_dir = configfile.replace('/cachebench.json', '')
        self.suffix = self.exp_dir.replace(fileprefix+'/', '')
        self.config = {
            'ExperimentName': self.suffix.split('/')[0],
            'ExperimentSuffix': self.suffix,
            'Completed': False,
            'CacheBenchConfig': configfile,
        }
        self.config['Prefetch-Range'] = None
        self.config['Prefetch-When'] = 'never'
        self.xperiod = 60
        try:
            with open(configfile) as f:
                cfg = commentjson.load(f)
            self.config['AdmissionPolicy'] = cfg['test_config']['admissionPolicy']
            if "prefetchingConfig" in cfg["test_config"]["replayGeneratorConfig"]:
                pfcfg = cfg["test_config"]["replayGeneratorConfig"]["prefetchingConfig"]
                self.config["Prefetch-Range"] = pfcfg["range"]
                self.config["Prefetch-When"] = pfcfg["when"]
            if 'meta' in cfg:
                self.config.update(cfg['meta'])
            desc = self.suffix.split("/")[1].split("_")
            if "Target Write Rate" not in self.config:
                if desc[-1].startswith('WR') and desc[-1].endswith('MBS'):
                    self.config["Target Write Rate"] = float(desc[-1][2:-3])
            if "TraceGroup" not in self.config:
                if (desc[1]+"_"+desc[2]).startswith(self.config["Region"]):
                    self.config["TraceGroup"] = desc[0]
            if 'SampleStart' not in self.config:
                if desc[1] == self.config["Region"]:
                    self.config["SampleStart"] = float(desc[3])
                elif desc[1]+"_"+desc[2] == self.config["Region"]:
                    self.config["SampleStart"] = float(desc[4])
                else:
                    print(f"Unknown format: {desc}")
            self.infer_config()
        except Exception as e:
            print(f"Trouble loading {configfile}")
            print(f"Exception: {e}")
        self.windows_log = configfile.replace("cachebench.json", "windows.log")
        if os.path.exists(self.windows_log+".gz"):
            self.windows_log += ".gz"
        self.file_summary = configfile.replace("cachebench.json", "progress.out")
        self.load_summary()

    def progress_cache_uptodate(self):
        cached_filename = self.windows_log+".cached.tmp.pkl.gz"
        return os.path.exists(cached_filename) and os.path.getmtime(cached_filename) > os.path.getmtime(adaptors.__file__)

    def load_progress(self):
        """Called on demand. Can be slower."""
        if os.path.exists(self.windows_log+".gz"):
            self.windows_log += ".gz"
        if os.path.exists(self.windows_log) and self.raw_summary is not None:
            self.raw_progress = adaptors.CacheBenchAdaptor.read(self.windows_log, use_cache=True)
            # TODO: Cache post_progress instead of raw_progress (saves space, time).
            self.post_progress = self.postprocess(self.raw_progress)
            self._progress = {}
            self._pstats = {}
            self.post_progress['NoCacheGET ST'] = processors.service_time_orca_c(self.post_progress['totalIOPS'], self.post_progress['ClientBytesGB'] * 1024 * 8)
            self._progress['NoCacheGET'], self._pstats['NoCacheGET'] = self.generate_progress(self.post_progress, label='NoCacheGET', col='NoCacheGET ST', sample_ratio=self.config['SampleRatio'])
            self._progress['GET'], self._pstats['GET'] = self.generate_progress(self.post_progress, label='GET', sample_ratio=self.config['SampleRatio'], nocache=self._pstats['NoCacheGET'])
            # if 'Consumed ST (Read)' in self.raw_summary:
            #     self._progress['GET'], self._pstats['GET'] = self.generate_progress(self.post_progress, col='Consumed ST (Read)', sample_ratio=self.config['SampleRatio'])
            if 'Consumed ST (Write)' in self.raw_summary:
                self.post_progress['NoCacheGET+PUT ST'] = self.post_progress['NoCacheGET ST'] + self.post_progress['Consumed ST (Write)']
                self._progress['NoCacheGET+PUT'], self._pstats['NoCacheGET+PUT'] = self.generate_progress(self.post_progress, label='NoCacheGET+PUT', col='NoCacheGET+PUT ST', sample_ratio=self.config['SampleRatio'])
                self._progress['PUT'], self._pstats['PUT'] = self.generate_progress(self.post_progress, label='PUT', col='Consumed ST (Write)', sample_ratio=self.config['SampleRatio'])
                self.post_progress['ST (Read+Write)'] = self.post_progress['Consumed ST (Write)'] + self.post_progress['Consumed Service Time']
                self._progress['GET+PUT'], self._pstats['GET+PUT'] = self.generate_progress(self.post_progress, label='GET+PUT', col='ST (Read+Write)', sample_ratio=self.config['SampleRatio'], nocache=self._pstats['NoCacheGET+PUT'])

    def load_summary(self):
        """Will be called at init. Should be fast."""
        if os.path.exists(self.file_summary):
            self.config['Completed'] = "== Test Results ==" in open(self.file_summary).read()
            df = adaptors.CacheBenchAdaptor.read(self.file_summary)
            # Needs to be done here, as we use it for windows_log too.
            if len(df) > 1:
                print("Warning! More than 1 row in output file", self.file_summary, len(df))
            if len(df) > 0:
                self.raw_summary = df.iloc[0]
            # TODO: Process raw summary so that columns match with sim
            if self.raw_summary is not None:
                try:
                    self.summary = self.postprocess(self.raw_summary)
                    self.summary = self.summary_fill(self.summary)
                except Exception as e:
                    print(f"Failure processing summary of {self}: {e}")
                    import traceback
                    traceback.print_exc()

    def postprocess(self, summary):
        if summary is None:
            return None
        # TODO: Write Rate

        def calc_wr(written_gb):
            return written_gb * 1024 / summary['Elapsed Trace Time'] * 100 / self.config['SampleRatio']

        for subtype in ['nand', 'logical', 'physical']:
            if f'NVM bytes written ({subtype}) (GB)' in summary:
                summary[maps.l_wr + f" ({subtype})"] = calc_wr(summary[f'NVM bytes written ({subtype}) (GB)'])
        summary[maps.l_wr] = summary.get(f"{maps.l_wr} (nand)", summary[f"{maps.l_wr} (physical)"])

        # TODO: Do a whitelist of columns.
        mapping = {
            'full success': 'IOPSSavedRatio',
            "partialOrFullSuccesses": "IOPSPartialHitsRatio",
            'objectGet': 'totalIOPS',
            'ST saved ratio': 'ServiceTimeSavedRatio',
            "NVM Puts": "FlashChunkWritten",
            "egressBytes (GB)": "ClientBytesGB",
            # "egressBytes": "ClientBytesGB",
            "ingressBytes (GB)": "BackendBytesGetGB",
            # "getBytes (GB)": 
            # "NVM Misses": "FlashMisses"
        }
        blacklist = [
            "navy",
            "latency",
            "Latency",
            "Success",
            " (/s)",
            " (GB/s)",
            "ap.",
            "Items in",
            "RAM eviction rejects",
            "Coalesced",
            "Clean",
            "AbortsFrom",
            "Unclean",
            "Double",
            "Total ",
            "success",
        ]
        if isinstance(summary, pd.Series):
            summary = pd.DataFrame([summary])
        summary = summary.rename(columns=mapping)
        cols = [col for col in summary.columns if not any(word in col for word in blacklist)]
        summary = summary[cols]
        summary["ClientBandwidth"] = summary["ClientBytesGB"] * 1024 / summary['Elapsed Trace Time'] * 100 / self.config['SampleRatio']
        summary["BackendBandwidthGet"] = summary["BackendBytesGetGB"] * 1024 / summary['Elapsed Trace Time'] * 100 / self.config['SampleRatio']
        summary["NoCacheServiceTime"] = processors.service_time_orca_c(summary['totalIOPS'], summary['ClientBytesGB'] * 1024 * 8)
        return summary


def process_exp_(file=None, fileprefix=None, with_progress=False, done_only=True, require_cached=True):
    result = CacheBenchResult(file, fileprefix)
    try:
        if with_progress:
            if result.done or not done_only:
                if not require_cached or result.progress_cache_uptodate():
                    result.progress
                elif require_cached:
                    print(f"Could not find {result.windows_log}.cached.tmp.pkl.gz")
    except Exception as e:
        print(f"Error while processing {file}")
        result.err = e
    return result


def load_tb(patterns,
            output_base_dir=local_cluster.OUTPUT_LOCATION,
            prefix="spring23",
            with_progress=False):
    """Similar to results-*.ipynb, but for testbed"""
    exps = []
    fileprefix = f"{output_base_dir}/{prefix}"
    files = ep_utils.flatten([glob.glob(f"{fileprefix}/{pat}/**/cachebench.json",  # Python 3.10 , root_dir=fileprefix,
                                        recursive=True) for pat in patterns])

    # 10x speedup: 1-min -> 6s
    with Pool(processes=4) as pool:
        for exp in tqdm(pool.imap_unordered(functools.partial(process_exp_, fileprefix=fileprefix, with_progress=with_progress), files), total=len(files)):
            if exp is not None:
                exps.append(exp)
    return exps


def cache_testbed_progress(filename):
    assert os.path.exists(filename) and filename.endswith('cachebench.json')
    tbr = CacheBenchResult(filename)
    tbr.load_progress()
    if tbr.pstats.keys():
        pprint(tbr.get_pstats())
    else:
        print("No results available? " + tbr.windows_log)


if __name__ == '__main__':
    """Takes in config json"""
    cache_testbed_progress(sys.argv[1])
