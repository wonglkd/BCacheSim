from collections import defaultdict
from collections import Counter
import numpy as np
from multiprocessing import Pool
from tqdm.auto import tqdm
# import pqdict

from . import ep_utils
from . import trace_utils


try:
    from .constants_meta import service_time
except (ImportError, ModuleNotFoundError):
    from .constants_public import service_time

def st_to_util(service_time, sample_ratio=None, disks=36, duration_s=None):
    return service_time / disks * 100 / sample_ratio / duration_s


tqdm_kwargs = dict(mininterval=10, maxinterval=30, leave=True)


class Episode(object):
    """
    ts_logical: (first, last)
    ts_physical: (first, last)
    """
    __slots__ = ["key", "num_accesses",
                 "ts_logical", "ts_physical",
                 "timespan_logical", "timespan_phys",
                 "offset", "size",
                 "score", "threshold",
                 "max_interarrival",
                 "accesses", "chunk_counts",

                 # "first_iop",
                 # "bytes_saved", "bytes_queried",
                 # "service_time", "service_time_saved",
                 # "hits_ram",

                 "s",
                 "s_export",
                 "chunk_last_seen", "chunk_ranges",
                 "num_chunks", "chunk_level", "chunk_range"]

    def __init__(self, key,
                 ts_logical, ts_physical, byte_offset,
                 *,
                 s=None,
                 s_export=None,
                 num_accesses=None,
                 score=None,
                 threshold=None,
                 max_interarrival=None,

                 # bytes_saved=None,
                 # bytes_queried=None,
                 first_iop=1,

                 chunk_counts=None,
                 chunk_last_seen=None,
                 chunk_ranges=None,
                 chunk_level=False,
                 # hits_ram=None,
                 accesses=None,
                 **kwargs):
        self.s = s or {}  # Extra stats
        self.s_export = s_export or {}
        self.key = key
        self.ts_logical = ts_logical  # (first, last)
        self.ts_physical = ts_physical  # (first, last)
        self.timespan_logical = ts_logical[1] - ts_logical[0]
        self.timespan_phys = ts_physical[1] - ts_physical[0]
        self.offset = byte_offset  # (start, end)
        self.size = byte_offset[1] - byte_offset[0] + 1
        self.chunk_range = offset_to_chunks(self.offset[0], self.offset[1])
        self.num_chunks = self.chunk_range[1] - self.chunk_range[0]
        # Potentially a float, if considered in terms of fractional IOPS
        self.num_accesses = num_accesses
        # self.first_iop = first_iop
        self.accesses = accesses
        self.chunk_counts = chunk_counts
        self.max_interarrival = max_interarrival
        self.score = score
        # self.bytes_saved = bytes_saved
        # self.bytes_queried = bytes_queried
        self.threshold = threshold
        self.chunk_last_seen = chunk_last_seen
        self.chunk_ranges = chunk_ranges
        self.chunk_level = chunk_level
        # self.hits_ram = hits_ram
        self.compute()

    def compute(self):
        if 'chunks_queried' in self.s:
            self.s['service_time_orig'] = service_time(self.num_accesses, self.s['chunks_queried'])

            # Fetched from backing store.
            if 'chunks_fetched__prefetch' not in self.s:
                if self.chunk_counts is not None:
                    self.s['chunks_written__prefetch'] = len(self.chunk_counts)
                self.s['chunks_fetched__prefetch'] = self.num_chunks
                # self.s['chunks_written__prefetch'] = self.s['chunks_fetched__prefetch']
                self.s['chunks_fetched__prefetch_nowaste'] = self.s['chunks_fetched__prefetch']
                # Chunks only in first access and never again
                # TODO: Debug this.
                if self.chunk_last_seen is not None:
                    one_hit_wonders_prefetch = sum(1 for v in self.chunk_last_seen.values() if v[1] == self.ts_logical[0])
                    self.s['chunks_written__prefetch_nowaste'] = self.s['chunks_fetched__prefetch_nowaste'] - one_hit_wonders_prefetch
                
            self.s['hits__prefetch'] = self.num_accesses - 1
            self.s['hits__prefetch_nowaste'] = self.num_accesses - 1
            for x in ['__prefetch', '__noprefetch', '__ram_prefetch', '__ram_noprefetch', '__prefetch_nowaste', '__noprefetch_nowaste']:
                if 'chunks_fetched' + x in self.s and 'chunks_written' + x not in self.s:
                    self.s['chunks_written' + x] = self.s['chunks_fetched' + x]
                if 'chunks_fetched' + x in self.s and 'chunks_saved' + x not in self.s:
                    self.s['chunks_saved' + x] = self.s['chunks_queried'] - self.s['chunks_fetched' + x]
                if 'chunks_fetched' + x not in self.s and 'chunks_saved' + x in self.s:
                    self.s['chunks_fetched' + x] = self.s['chunks_queried'] - self.s['chunks_saved' + x]
                # if 'chunks_written' + x not in self.s and 'chunks_saved' + x in self.s:
                #     self.s['chunks_written' + x] = self.s['chunks_queried'] - self.s['chunks_saved' + x]
                if 'chunks_saved' + x in self.s:
                    self.s['service_time_saved' + x] = service_time(self.s['hits' + x], self.s['chunks_saved' + x])
            self.s['chunks_timespan__prefetch'] = self.s['chunks_written__prefetch'] * self.timespan_logical
            self.s['chunks_timespan__prefetch_nowaste'] = self.s['chunks_written__prefetch_nowaste'] * self.timespan_logical

            if 'service_time_saved__ram_noprefetch' in self.s:
                for y in ['prefetch', 'noprefetch']:
                    self.s[f'chunks_written__flash_{y}'] = self.s[f'chunks_written__{y}']
                    # TODO: Fix this to reflect RAM
                    self.s[f'chunks_written__ram_{y}'] = self.s[f'chunks_written__{y}']
                    for x in ['chunks_saved', 'hits', 'service_time_saved']:
                        self.s[f'{x}__flash_{y}'] = self.s[f'{x}__{y}'] - self.s[f'{x}__ram_{y}']
                    for x in ['ram', 'flash']:
                        self.s[f'chunks_timespan__{x}_{y}'] = self.s[f'chunks_timespan__{y}']
        if 'service_time_saved__prefetch' in self.s and 'service_time_saved__noprefetch' in self.s:
            self.s_export['service_time_saved__prefetch'] = self.s['service_time_saved__prefetch']
            self.s_export['service_time_saved__noprefetch'] = self.s['service_time_saved__noprefetch']
            self.s_export['prefetch_st_benefit'] = self.s_export['service_time_saved__prefetch'] - self.s_export['service_time_saved__noprefetch']

    def export(self):
        kwargs = {}
        for k in ["num_accesses", "score", "threshold", "chunk_counts", "chunk_last_seen", "chunk_level", "s_export", "max_interarrival"]: # , "bytes_saved", "bytes_queried"]:
            v = getattr(self, k)
            if v is not None:
                kwargs[k] = v
        return ((self.key, self.ts_logical, self.ts_physical, self.offset), kwargs)

    def to_access(self):
        return trace_utils.cs_utils.BlkAccess(self.offset[0], self.size, self.ts_physical[0])

    def contains(self, chunk_id):
        return self.chunk_range[0] <= chunk_id < self.chunk_range[1]

    def __repr__(self):
        try:
            return f'Ep({self.key}, ts={self.ts_physical}, chunks={self.chunk_range} ({self.num_chunks}), threshold={self.threshold:g}, offset={self.offset}, size={self.size}, num_acc={self.num_accesses})'
        except TypeError:
            return f'Ep({self.key}, ts={self.ts_physical}, chunks={self.chunk_range} ({self.num_chunks}), threshold={self.threshold}, offset={self.offset}, size={self.size}, num_acc={self.num_accesses})'
        # return f'Ep({self.key}, ts={self.ts_physical}, chunks={self.chunk_range}, threshold={self.threshold}, offset={self.offset}, size={self.size}, num_acc={self.num_accesses}, first_iop={self.first_iop})'


Residency = Episode

class ResidencyList(object):
    def __init__(self, residencies,
                 trace_helper=None,
                 e_age=None):
        if residencies and 'episode_id' not in residencies[0].s:
            for i, ep in enumerate(residencies):
                ep.s['episode_id'] = i
        self.residencies = residencies
        self.num_residencies = len(residencies)
        self.eviction_age_logical = e_age[0]
        self.eviction_age_physical = e_age[1]
        self.th = self.trace_helper = trace_helper
        self.policy = ''
        self.scores = None

    @property
    def episodes(self):
        return self.residencies

    def __len__(self):
        return len(self.episodes)

    def init(self, early_eviction=False):
        # Static
        self.num_admitted = np.arange(self.num_residencies) + 1
        self.num_chunks_admitted = 64 * self.num_admitted
        self.early_eviction = early_eviction
        # self.recompute()

    def apply_policy(self, new_order, scores=None, policy=''):
        """Reorder residencies."""
        self.residencies = [self.residencies[i] for i in new_order]
        if scores is not None:
            scores = np.array([scores[i] for i in new_order])
        self.scores = scores
        self.recompute()
        for episode, score, threshold in zip(self.residencies, self.scores, self.write_rates):
            episode.score = score
            episode.threshold = threshold
            episode.s[f'pol_score__{policy}'] = score
            episode.s[f'pol_threshold__{policy}'] = threshold
        self.policy = policy

    def recompute(self):
        # Static
        self.arrival_rates = self.num_admitted / self.th.logical_time
        # Dynamic
        self.num_accesses = np.array(
            [r.num_accesses for r in self.residencies])
        self.timespan_logical = np.array(
            [r.timespan_logical for r in self.residencies])
        self.num_accesses_csum = self.num_accesses.cumsum()
        self.timespan_logical_csum = self.timespan_logical.cumsum()
        self.mean_timespan = self.timespan_logical_csum / self.num_admitted
        if self.early_eviction:
            self.time_in_system = self.mean_timespan
        else:
            self.time_in_system = self.mean_timespan + self.eviction_age_logical
        self.dead_time_ratio = 1. - self.mean_timespan / self.time_in_system
        self.cache_sizes_GB = self.th.items_to_mb(
            self.arrival_rates) / 1024 * self.time_in_system
        self.iops_saved = self.num_accesses_csum - self.num_admitted
        self.iops_saved_ratio = self.iops_saved / self.th.logical_time
        self.write_rates = self.th.arrivals_to_wr_mbps(self.num_admitted)

    def get_thresholds(self, targets, val, label, atol=0.5):
        targets = np.asarray(targets)
        thresholds = np.searchsorted(val, targets)
        thresholds = np.clip(thresholds, 0, self.num_residencies - 1)
        legit = np.isclose(targets, val[thresholds], atol=atol)
        if not legit.all():
            print(f'Filtered out impossible {label} thresholds: ',
                  np.count_nonzero(~legit),
                  val[thresholds[~legit]],
                  targets[~legit])
        return thresholds[legit], targets[legit]

    def get_labels(self):
        ret = {
            'Assumed Eviction Age (s)': self.eviction_age_physical,
            'Assumed Eviction Age (hrs)': self.eviction_age_physical / 3600,
            'Assumed Eviction Age (days)': self.eviction_age_physical / (3600*24),
            'Assumed Eviction Age (Logical)': self.eviction_age_logical,
            # 'TotalIOPSSaved': 
            # 'TotalChunkSaved':
            'Total IOPS': self.num_accesses.sum(),
            # 'TotalChunkQueries':
        }
        labels = {
            'Write Rate (MB/s)': 'write_rates',  # Upsampled
            'Cache Size (GB)': 'cache_sizes_GB',  # Upsampled
            # 'Arrival Rate': 'arrival_rates',
            'Arrival Rate (MB/IO)': 'arrival_rates_mb',
            'Mean Residency': 'mean_timespan',
            'Mean Time In System': 'time_in_system',
            'Mean Residency (s)': 'mean_timespan_secs',
            'Mean Time In System (s)': 'time_in_system_secs',
            'TotalIOPSSaved': 'iops_saved_csum',  # Downsampled
            'IOPSSavedRatio': 'iops_saved_ratio',
            'Episodes admitted': 'num_admitted',
            'TotalChunkWritten': 'chunks_written_csum',
            'B written (based on CW)': 'sizes_cb_csum',
            'B written': 'sizes_csum',
            # 'Episode-chunks admitted': 'num_chunks_admitted',
            'Cutoff score': 'scores',
            'TotalChunkSaved': 'chunks_saved_csum',
            'TotalChunkQueries': 'chunks_queried_csum',
            # 'Chunks saved': 'chunks_saved_csum',
            # 'Chunks queried': 'chunks_queried_csum',
            'Dead Time Ratio': 'dead_time_ratio',
            # 'Orig Service Time': 'service_time_orig_csum',
            'Service Time Saved': 'service_time_saved_csum',
            'Service Time Saved Ratio': 'service_time_saved_ratio',
        }
        return ret, labels

    def to_dict(self, mask=None):
        ret, labels = self.get_labels()
        for k, v in labels.items():
            if hasattr(self, v):
                ret[k] = getattr(self, v)
            elif hasattr(self, v[:-5]) and v.endswith('_csum'):
                ret[k] = getattr(self, v[:-5]).cumsum()
            elif hasattr(self, v[:-5]) and v.endswith('_secs'):
                ret[k] = self.th.logical_dur_to_phy(getattr(self, v[:-5]))
            else:
                continue
            if mask is not None:
                ret[k] = ret[k][mask]
        return ret

    def target_wrs(self, target_write_rates, label='Write Rate'):
        wr_thresholds, target_wrs = self.get_thresholds(
            target_write_rates, self.write_rates,
            label='WR', atol=0.5)
        extra = {
            'Target': label,
            'Target Write Rate': target_wrs,
        }
        return {**self.to_dict(wr_thresholds), **extra}

    def target_csizes(self, target_cache_sizes):
        csizes_thresholds, target_csizes = self.get_thresholds(
            target_cache_sizes, self.cache_sizes_GB,
            label='Cache Size', atol=100.0)
        extra = {
            'Target': 'Cache Size',
            'Target Cache Size': target_csizes,
        }
        return {**self.to_dict(csizes_thresholds), **extra}


class ResidencyListSizeAware(ResidencyList):
    def init(self, early_eviction=False):
        self.early_eviction = early_eviction
        self.num_admitted = np.arange(self.num_residencies) + 1
        self.recompute()

    def recompute(self):
        self.sizes = np.array([r.size for r in self.residencies])
        self.num_accesses = np.array(
            [r.num_accesses for r in self.residencies])
        self.timespan_logical = np.array(
            [r.timespan_logical for r in self.residencies])
        self.num_chunks_admitted = self.sizes.cumsum() / 1024 / 128
        # self.chunks_saved = np.array([r.bytes_saved / 1024/ 128 for r in self.residencies])
        # self.chunks_saved_csum = self.chunks_saved.cumsum()
        # self.chunks_queried = np.array([r.bytes_queried / 1024/ 128 for r in self.residencies])
        # self.chunks_queried_csum = self.chunks_queried.cumsum()
        # self.service_time = np.array([r.service_time for r in self.residencies])
        # self.service_time_csum = self.service_time.cumsum()
        # self.service_time_saved = np.array([r.service_time_saved for r in self.residencies])
        # self.service_time_saved_csum = self.service_time_saved.cumsum()
        # self.service_time_saved_ratio = self.service_time_saved_csum / self.service_time_csum[-1]
        self.writes_needed_mb = self.sizes.cumsum() / 1024 / 1024
        self.arrival_rates_mb = self.writes_needed_mb / self.th.logical_time
        self.arrival_rates = self.arrival_rates_mb
        self.write_rates = self.th.upsample(
            self.writes_needed_mb) / self.th.duration
        self.num_accesses_csum = self.num_accesses.cumsum()
        self.timespan_logical_csum = self.timespan_logical.cumsum()
        self.mean_timespan = self.timespan_logical_csum / self.num_admitted
        if self.early_eviction:
            self.time_in_system = self.mean_timespan
        else:
            self.time_in_system = self.mean_timespan + self.eviction_age_logical
        self.dead_time_ratio = 1. - self.mean_timespan / self.time_in_system
        self.cache_sizes_GB = self.th.upsample(
            self.arrival_rates_mb) / 1024 * self.time_in_system
        self.iops_saved_csum = self.num_accesses_csum - self.num_admitted
        self.iops_saved_ratio = self.iops_saved_csum / self.th.logical_time


class ResidencyListPrefetchAware(ResidencyListSizeAware):
    def init(self, filter_='prefetch', early_eviction=False):
        self.early_eviction = early_eviction
        self.filter = filter_
        self.num_admitted = np.arange(self.num_residencies) + 1
        self.recompute()

    def get_res(self, k, not_exists_ok=False):
        try:
            if not_exists_ok:
                return np.array([r.s.get(k, 0) for r in self.residencies])
            return np.array([r.s[k] for r in self.residencies])
        except KeyError:
            print(self.residencies[0].s)
            raise

    def recompute(self, filter_override=None):
        self.sizes = np.array([r.size for r in self.residencies])
        self.num_accesses = np.array(
            [r.num_accesses for r in self.residencies])
        self.timespan_logical = np.array(
            [r.timespan_logical for r in self.residencies])
        self.max_interarrival = np.array(
            [r.max_interarrival for r in self.residencies])

        self.chunks_queried = self.get_res('chunks_queried')
        self.service_time_orig = self.get_res('service_time_orig')

        self.sizes_chunks = self.get_res('chunks_written__prefetch')

        x = '__' + (filter_override or self.filter)
        # ['__prefetch', '__noprefetch', '__ram_prefetch', '__ram_noprefetch', '__flash_prefetch', '__flash_noprefetch']
        self.chunks_written = self.get_res('chunks_written' + x)
        # self.sizes = self.chunks_written * 128 * 1024
        self.sizes_cb = self.sizes_chunks * 128 * 1024
        self.chunks_saved = self.get_res('chunks_saved' + x)
        self.service_time_saved = self.get_res('service_time_saved' + x)
        self.service_time_saved_ratio = self.service_time_saved.cumsum() / self.service_time_orig.sum()

        self.mean_timespan = self.get_res('chunks_timespan' + x).cumsum() / self.chunks_written.cumsum()
        # self.mean_timespan = (self.timespan_logical * self.sizes_chunks).cumsum() / self.chunks_written.cumsum()
        if self.early_eviction:
            self.time_in_system = self.mean_timespan
        else:
            self.time_in_system = self.mean_timespan + self.eviction_age_logical
        self.postupdate(x)

    def postupdate(self, x):
        self.dead_time_ratio = 1. - self.mean_timespan / self.time_in_system

        self.writes_needed_mb = self.chunks_written.cumsum() / 8
        self.arrival_rates_mb = self.writes_needed_mb / self.th.logical_time
        # self.arrival_rates = self.arrival_rates_mb
        self.write_rates = self.th.upsample(
            self.writes_needed_mb) / self.th.duration
        # self.num_accesses_csum = self.num_accesses.cumsum()
        # self.timespan_logical_csum = self.timespan_logical.cumsum()
        self.cache_sizes_GB = self.th.upsample(self.arrival_rates_mb) / 1024 * self.time_in_system
        # Hits
        self.iops_saved = self.get_res('hits' + x)
        self.iops_saved_ratio = self.iops_saved.cumsum() / self.th.logical_time


def add_peak_to_eps(ep, peak_tses):
    last_before_peak = None
    accs_in_peak = []
    for acc in ep.accesses:
        if any(start_ts <= acc.ts <= end_ts for start_ts, end_ts in peak_tses):
            accs_in_peak.append(acc)
        # if acc.ts < start_ts:
        #     last_before_peak = acc
        # elif acc.ts > end_ts:
        #     break
        # else:
        #     # We are in peak

    ret = {}
    ret.update({
        # 'StartedBefore': ret['TimeStart'] < start_ts,
        # 'EndsAfter': ret['TimeEnd'] > end_ts,
        'AccsInPeak': len(accs_in_peak),
        # 'MaxHitsInPeak': len(accs_in_peak) if last_before_peak else len(accs_in_peak)-1,
        # 'TimeFromPrevAcc': None if last_before_peak is None or len(accs_in_peak) == 0 else accs_in_peak[0].ts - last_before_peak.ts,
        # 'TimeFromPrevAccToStart': None if last_before_peak is None else start_ts - last_before_peak.ts,
        'StartInPeak': any(start_ts <= ep.accesses[0].ts <= end_ts for start_ts, end_ts in peak_tses),
        'PeakSTNoCache': sum(service_time(1, acc.num_chunks()) for acc in accs_in_peak),
    })
    ret.update({
        'PeakSTAlwaysCache': 0 if not ret['StartInPeak'] or len(accs_in_peak) == 0 else service_time(1, ep.num_chunks),
    })
    ret.update({
        'PeakSTSavedIfAdmitted': ret['PeakSTAlwaysCache'] - ret['PeakSTNoCache'],
    })
    # ret.update({
    #     'PeakSTOptCache': ret['PeakSTNoCache'] if not ret['Admitted'] else ret['PeakSTAlwaysCache'],
    # })
    # ret.update({
    #     'PeakSTPeakOptCacheAdmitted': ret['PeakSTAlwaysCache'] < ret['PeakSTOptCache'] and not ret['PeakSTNoCache'] < ret['PeakSTAlwaysCache'],
    #     'PeakSTPeakOptCache': min(ret['PeakSTAlwaysCache'], ret['PeakSTOptCache'], ret['PeakSTNoCache']),
    # })
    ep.s.update(ret)
    return ret


def overlap(start1, end1, start2, end2):
    return start1 <= end2 and end1 >= start2


class ResidencyListPeakAware(ResidencyListPrefetchAware):
    def init(self, peak_ts1_start=None, peak_ts1_end=None, **kwargs):  # peak_tses=None,
        assert peak_ts1_start is not None and peak_ts1_end is not None
        self.peak_tses = [[peak_ts1_start, peak_ts1_end]]
        self.peak_tses = [list(map(float, xs)) for xs in self.peak_tses]
        super().init(**kwargs)

    def postupdate(self, x):
        super().postupdate(x)
        for ep in self.residencies:
            if any(overlap(ep.ts_physical[0], ep.ts_physical[1], peakts[0], peakts[1])
                   for peakts in self.peak_tses):
                add_peak_to_eps(ep, self.peak_tses)
        self.peak_service_time_saved = self.get_res('PeakSTSavedIfAdmitted', not_exists_ok=True)


class ResidencyListPrefetchVariants(ResidencyListPrefetchAware):
    def apply_policy(self, new_order, prefetch_decisions=None, scores=None, policy=''):
        """Reorder residencies."""
        self.residencies = [self.residencies[i] for i in new_order]
        if scores is not None:
            scores = np.array([scores[i] for i in new_order])
        if prefetch_decisions is not None:
            prefetch_decisions = np.array([prefetch_decisions[i] for i in new_order])
        self.scores = scores
        self.prefetch_decisions = prefetch_decisions
        self.recompute()
        for episode, score, threshold in zip(self.residencies, self.scores, self.write_rates):
            episode.score = score
            episode.threshold = threshold
        self.policy = policy

    def recompute(self):
        self.sizes = np.array([r.size for r in self.residencies])
        self.num_accesses = np.array(
            [r.num_accesses for r in self.residencies])
        self.timespan_logical = np.array(
            [r.timespan_logical for r in self.residencies])

        self.chunks_queried = self.get_res('chunks_queried')
        self.service_time_orig = self.get_res('service_time_orig')

        self.sizes_chunks = self.get_res('chunks_written__prefetch')
        self.sizes_cb = self.sizes_chunks * 128 * 1024

        xs = ['__prefetch', '__noprefetch']
        if 'hits__ram_prefetch' in self.residencies[0].s:
            xs += ['__ram_prefetch', '__ram_noprefetch', '__flash_prefetch', '__flash_noprefetch']
        for k in ['chunks_written', 'chunks_saved', 'service_time_saved', 'service_time_saved_ratio', 'mean_timespan', 'time_in_system', 'dead_time_ratio',
                  'writes_needed_mb','arrival_rates_mb', 'write_rates', 'cache_sizes_GB', 'iops_saved', 'iops_saved_ratio']:
            setattr(self, k, {})
        for x in xs:
            y = x[2:]
            self.chunks_written[y] = self.get_res('chunks_written' + x)
            # self.sizes = self.chunks_written * 128 * 1024
            self.chunks_saved[y] = self.get_res('chunks_saved' + x)
            self.service_time_saved[y] = self.get_res('service_time_saved' + x)
            # Hits
            self.iops_saved[y] = self.get_res('hits' + x)
            self.mean_timespan[y] = self.get_res('chunks_timespan' + x).cumsum() / self.chunks_written[y].cumsum()

        if hasattr(self, "prefetch_decisions") and self.prefetch_decisions is not None:
            for dd in ["", "ram_", "flash_"]:
                if dd+"noprefetch" in self.chunks_written:
                    self.chunks_written[dd+'bestprefetch'] = np.where(self.prefetch_decisions, self.chunks_written[dd+'noprefetch'], self.chunks_written[dd+'prefetch'])
                    self.chunks_saved[dd+'bestprefetch'] = np.where(self.prefetch_decisions, self.chunks_saved[dd+'noprefetch'], self.chunks_saved[dd+'prefetch'])
                    self.service_time_saved[dd+'bestprefetch'] = np.where(self.prefetch_decisions, self.service_time_saved[dd+'noprefetch'], self.service_time_saved[dd+'prefetch'])
                    self.iops_saved[dd+'bestprefetch'] = np.where(self.prefetch_decisions, self.iops_saved[dd+'noprefetch'], self.iops_saved[dd+'prefetch'])
                    self.mean_timespan[dd+'bestprefetch'] = np.where(self.prefetch_decisions, self.mean_timespan[dd+'noprefetch'], self.mean_timespan[dd+'prefetch'])
                    xs.append("__"+dd+'bestprefetch')

        for x in xs:
            y = x[2:]
            self.service_time_saved_ratio[y] = self.service_time_saved[y].cumsum() / self.service_time_orig.sum()

            if self.early_eviction:
                self.time_in_system[y] = self.mean_timespan[y]
            else:
                self.time_in_system[y] = self.mean_timespan[y] + self.eviction_age_logical
            self.dead_time_ratio[y] = 1. - self.mean_timespan[y] / self.time_in_system[y]

            self.writes_needed_mb[y] = self.chunks_written[y].cumsum() / 8
            self.arrival_rates_mb[y] = self.writes_needed_mb[y] / self.th.logical_time
            # self.arrival_rates = self.arrival_rates_mb
            self.write_rates[y] = self.th.upsample(
                self.writes_needed_mb[y]) / self.th.duration
            # self.num_accesses_csum = self.num_accesses.cumsum()
            # self.timespan_logical_csum = self.timespan_logical.cumsum()
            self.cache_sizes_GB[y] = self.th.upsample(self.arrival_rates_mb[y]) / 1024 * self.time_in_system[y]
            self.iops_saved_ratio[y] = self.iops_saved[y].cumsum() / self.th.logical_time

    def target_wrs(self, target_write_rates, label='Write Rate'):
        wr_thresholds, target_wrs = self.get_thresholds(
            target_write_rates, self.write_rates['bestprefetch'],
            label='WR', atol=0.01)
        extra = {
            'Target': label,
            'Target Write Rate': target_wrs,
        }
        return {**self.to_dict(wr_thresholds), **extra}

    def target_csizes(self, target_cache_sizes):
        csizes_thresholds, target_csizes = self.get_thresholds(
            target_cache_sizes, self.cache_sizes_GB['bestprefetch'],
            label='Cache Size', atol=1.0)
        extra = {
            'Target': 'Cache Size',
            'Target Cache Size': target_csizes,
        }
        return {**self.to_dict(csizes_thresholds), **extra}

    def to_dict(self, mask=None):
        ret, labels = self.get_labels()
        for k, v in labels.items():
            v2 = v.replace("_csum", "")
            v2 = v.replace("_secs", "")
            if hasattr(self, v2):
                ret[k] = getattr(self, v2)
            else:
                continue
            if type(ret[k]) == dict:
                for bb, vv in ret[k].items():
                    if v.endswith('_csum'):
                        vv = vv.cumsum()
                    elif v.endswith('_secs'):
                        vv = self.th.logical_dur_to_phy(vv)
                    if mask is not None:
                        vv = vv[mask]
                    ret[f'{k} ({bb})'] = vv
                del ret[k]
            else:
                if v.endswith('_csum'):
                    ret[k] = ret[k].cumsum()
                elif v.endswith('_secs'):
                    ret[k] = self.th.logical_dur_to_phy(ret[k])

                if mask is not None:
                    ret[k] = ret[k][mask]
        return ret


class ResidencyListFractionalIOPS(ResidencyListSizeAware):
    def init(self, early_eviction=False):
        self.early_eviction = early_eviction
        # TODO: change this
        self.num_admitted = np.arange(self.num_residencies) + 1
        self.recompute()

    def recompute(self):
        self.sizes = np.array([r.size for r in self.residencies])

        self.num_accesses = np.array(
            [r.num_accesses for r in self.residencies])
        self.num_accesses_csum = self.num_accesses.cumsum()
        # TODO: Change this
        self.first_acc_miss = np.array([r.first_iop for r in self.residencies])
        self.first_acc_miss_csum = self.first_acc_miss.cumsum()

        self.timespan_logical = np.array(
            [r.timespan_logical for r in self.residencies])
        self.num_chunks_admitted = self.sizes.cumsum() / 1024 / 128
        self.writes_needed_mb = self.sizes.cumsum() / 1024 / 1024
        self.arrival_rates_mb = self.writes_needed_mb / self.th.logical_time
        self.arrival_rates = self.arrival_rates_mb
        self.write_rates = self.th.upsample(
            self.writes_needed_mb) / self.th.duration
        self.timespan_logical_csum = self.timespan_logical.cumsum()
        self.mean_timespan = self.timespan_logical_csum / self.num_admitted
        if self.early_eviction:
            self.time_in_system = self.mean_timespan
        else:
            self.time_in_system = self.mean_timespan + self.eviction_age_logical
        self.dead_time_ratio = 1. - self.mean_timespan / self.time_in_system
        self.cache_sizes_GB = self.th.upsample(
            self.arrival_rates_mb) / 1024 * self.time_in_system
        self.iops_saved = self.num_accesses_csum - self.first_acc_miss_csum
        self.iops_saved_ratio = self.iops_saved / self.th.logical_time


def residences_from_interarrivals(interarrivals, eviction_age):
    splits = np.argwhere(interarrivals > eviction_age) + 1
    splits = splits.flatten().tolist() + [len(interarrivals)+1]
    prev = 0
    for stop in splits:
        assert prev < stop, splits
        yield (prev, stop-1, stop-prev)
        prev = stop


def interarrivals_from_accesses(obj):
    kacx = obj['obj'][1]
    split_by = obj['split_by']
    e_age_ram = obj.get('e_age_ram', None)
    accs = kacx.accesses
    block_id = kacx.key
    tses_phy = np.array([ac.ts for ac in accs])
    tses_logical = np.array([ac.ts_logical for ac in accs])
    tses = tses_phy if split_by == 'physical' else tses_logical
    interarrivals = np.diff(tses)
    interarrivals_phy = np.diff(tses_phy)
    interarrivals_logical = np.diff(tses_logical)
    byte_starts = np.array([ac.start() for ac in accs])
    byte_ends = np.array([ac.end() for ac in accs])
    chunk_starts, chunk_ends = offset_to_chunks(byte_starts, byte_ends)
    num_chunks = chunk_ends - chunk_starts
    acc_sizes = np.array([ac.size() for ac in accs])
    split_idx = 0 if split_by == 'logical' else 1
    return {k: v for k, v in locals().items()
            if k in ['interarrivals', 'interarrivals_phy', 'interarrivals_logical',
            'tses_phy', 'tses_logical', 'tses',
            'byte_starts', 'byte_ends',
            'chunk_starts', 'chunk_ends', 'num_chunks',
            'acc_sizes', 'split_idx', 'block_id', 'e_age_ram'] and v is not None}


def get_episode(data, first, last, no_of_accesses, **kwargs):
    d_ = data
    left = d_['byte_starts'][first:last+1].min()
    right = d_['byte_ends'][first:last+1].max()
    bytes_queried = d_['acc_sizes'][first:last+1].sum()
    chunks_queried = d_['num_chunks'][first:last+1].sum()
    # bytes_saved__prefetch = bytes_queried - d_['acc_sizes'][first]
    # chunks_saved__prefetch = chunks_queried - d_['num_chunks'][first]
    stats = dict(bytes_queried=bytes_queried, chunks_queried=chunks_queried)
    #bytes_saved=bytes_saved,
     #             chunks_saved=chunks_saved)
    if d_.get('e_age_ram', None) is not None:
        stats['hits__ram_prefetch'] = (d_['interarrivals'][first:last] <= d_['e_age_ram']).sum()
        stats['chunks_saved__ram_prefetch'] = d_['num_chunks'][first+1:last+1][d_['interarrivals'][first:last] <= d_['e_age_ram']].sum()
    max_ia = (d_['interarrivals_phy'][first:last].max(initial=0),
              d_['interarrivals_logical'][first:last].max(initial=0))
    return Episode(d_['block_id'],
                   (d_['tses_logical'][first], d_['tses_logical'][last]),
                   (d_['tses_phy'][first], d_['tses_phy'][last]),
                   (left, right),
                   num_accesses=no_of_accesses, s=stats,
                   max_interarrival=max_ia, **kwargs)


def process_obj(obj):
    d_ = interarrivals_from_accesses(obj)
    residences_by_e_age = {}
    for e_age_log_phy in obj['e_ages']:
        e_age_split = e_age_log_phy[d_['split_idx']]
        residencies = []
        for first, last, no_of_accesses in residences_from_interarrivals(d_['interarrivals'], e_age_split):
            episode_ = get_episode(d_, first, last, no_of_accesses)
            residencies.append(episode_)
        residences_by_e_age[e_age_log_phy] = residencies
    return residences_by_e_age


def process_obj_w_accs(obj):
    d_ = interarrivals_from_accesses(obj)
    residences_by_e_age = {}
    for e_age_log_phy in obj['e_ages']:
        e_age_split = e_age_log_phy[d_['split_idx']]
        residencies = []
        for first, last, no_of_accesses in residences_from_interarrivals(d_['interarrivals'], e_age_split):
            episode_ = get_episode(d_, first, last, no_of_accesses)
            episode_.accesses = obj['obj'][1].accesses[first:last+1]
            residencies.append(episode_)
        residences_by_e_age[e_age_log_phy] = residencies
    return residences_by_e_age


def group_acc_by_chunks(accesses):
    chunks = {}
    for i, ac in enumerate(accesses):
        for c in ac.chunks():
            if c not in chunks:
                chunks[c] = []
            chunks[c].append(i)

    acc2chunks = defaultdict(list)
    for c, acc in chunks.items():
        assert acc == sorted(acc)
        acc2chunks[tuple(acc)].append(c)

    for acc_ids, curr_chunks in acc2chunks.items():
        curr_acc = [accesses[i] for i in acc_ids]
        chunk_grps = trace_utils.run_length_encode(sorted(curr_chunks))
        chunk_grp_offsets = [chunks_to_offset(s, e) for s, e in chunk_grps]
        yield curr_acc, chunk_grps, chunk_grp_offsets


# def process_obj_chunk(obj):
#     d_ = interarrivals_from_accesses(obj)
#     residences_by_e_age = {e_age: [] for e_age in e_ages}
#     for curr_acc, chunk_grps, chunk_grp_offsets in group_acc_by_chunks(kacx.accesses):
#         interarrivals, tses_phy, tses_logical = interarrivals_from_accesses(
#             curr_acc, split_by=split_by)
#         denom = np.array([1. / len(acc.chunks()) for acc in curr_acc])
#         denom_cumsum = denom.cumsum()
#         for e_age_log_phy in obj['e_ages']:
#             e_age_split = e_age_log_phy[0] if split_by == 'logical' else e_age_log_phy[1]
#             for first, last, no_of_accesses in residences_from_interarrivals(interarrivals, e_age_split):
#                 assert first <= last, (first, last)
#                 assert no_of_accesses > 0
# #                 denom_c = denom_cumsum[last] - denom_cumsum[first]
#                 denom_c = denom[first:last+1].sum()
#                 max_ia = interarrivals[first:last].max(initial=0)
#                 for (chunk_s, chunk_e), (left, right) in zip(chunk_grps, chunk_grp_offsets):
#                     num_chunks = (chunk_e - chunk_s) + 1
#                     num_accesses_weighted = num_chunks * denom_c
#                     assert num_chunks > 0
#                     assert 0 <= num_accesses_weighted <= no_of_accesses, denom_c
#                     residences_by_e_age[e_age_log_phy].append(
#                         Episode(block_id,
#                                   (tses_logical[first], tses_logical[last]),
#                                   (tses_phy[first], tses_phy[last]),
#                                   (left, right),
#                                   num_accesses=num_accesses_weighted,
#                                   max_interarrival=max_ia,
#                                   first_iop=num_chunks / len(curr_acc[0].chunks())))
#     return residences_by_e_age


def offset_to_chunks(s, e):
    # e is the last byte of the block.
    # returns exclusive range
    return s // trace_utils.ALIGNMENT + 1, (e+1) // trace_utils.ALIGNMENT + 1


def chunks_to_offset(start_chunk, end_chunk):
    # chunks are numbered from 1, not 0.
    # exclusive range
    return (start_chunk - 1) * trace_utils.ALIGNMENT, (end_chunk - 1) * trace_utils.ALIGNMENT - 1


def get_chunk_stats(first, last, d_):
    chunks = Counter()
    chunk_last_seen = {}
    for i in range(first, last+1):
        chks = list(range(*offset_to_chunks(d_['byte_starts'][i], d_['byte_ends'][i])))
        assert len(chks) > 0
        assert len(chks) == int(round((d_['byte_ends'][i] - d_['byte_starts'][i])/trace_utils.ALIGNMENT))
        for c in chks:
            chunks[c] += 1
            chunk_last_seen[c] = (d_['tses_phy'][i], d_['tses_logical'][i])
    # TODO: What about timestamp?
    assert len(chunks) > 0
    assert all(v > 0 for v in chunks.values())
    return dict(chunks), chunk_last_seen


def process_obj_chunk_n(obj):
    d_ = interarrivals_from_accesses(obj)
    residences_by_e_age = {}
    for e_age_log_phy in obj['e_ages']:
        e_age_split = e_age_log_phy[d_['split_idx']]
        residencies = []
        for first, last, no_of_accesses in residences_from_interarrivals(d_['interarrivals'], e_age_split):
            chunk_counts, chunk_last_seen = get_chunk_stats(first, last, d_)
            episode_ = get_episode(d_, first, last, no_of_accesses, chunk_counts=chunk_counts, chunk_last_seen=chunk_last_seen)
            residencies.append(episode_)
        residences_by_e_age[e_age_log_phy] = residencies
    return residences_by_e_age


def update_noprefetch_stats(first, last, d_, episode_, no_of_accesses, e_age_split):
    refs = {0: 'tses_logical', 1: 'tses_phy'}
    chunk_last_seen_by_split = {}
    chunk_hit_after_admit = {}
    noprefetch_iops = 0
    noprefetch_chunks_fetched = 0
    excess_noprefetch_chunks_saved_ram = 0
    excess_noprefetch_iops_ram = 0
    chunks_timespan_noprefetch = 0
    useful_chunks_to_write = 0
    for i in range(first, last+1):
        new_time_ = d_[refs[d_['split_idx']]][i]
        extra_chunks = 0
        for c in range(d_['chunk_starts'][i], d_['chunk_ends'][i]):
            if c not in chunk_last_seen_by_split or new_time_ - chunk_last_seen_by_split[c] > e_age_split:
                extra_chunks += 1
                chunk_hit_after_admit[c] = False
            else:
                if not chunk_hit_after_admit[c]:
                    useful_chunks_to_write += 1
                    chunk_hit_after_admit[c] = True
                chunks_timespan_noprefetch += new_time_ - chunk_last_seen_by_split[c]
            chunk_last_seen_by_split[c] = new_time_
        if extra_chunks > 0:
            noprefetch_chunks_fetched += extra_chunks
            noprefetch_iops += 1
            if first < i and 'e_age_ram' in d_ and d_['interarrivals'][i-1] <= d_['e_age_ram']:
                # Chunks which are actually misses in no prefetch, but were counted as
                # saved for RAM Prefetch. These need to be fetched (higher ST)
                # and potentially (re)admitted into RAM cache.
                excess_noprefetch_chunks_saved_ram += extra_chunks
                excess_noprefetch_iops_ram += 1
    episode_.s['hits__noprefetch'] = no_of_accesses - noprefetch_iops
    # For WR and ST calculation
    episode_.s['chunks_fetched__noprefetch'] = noprefetch_chunks_fetched
    assert useful_chunks_to_write <= noprefetch_chunks_fetched
    episode_.s['chunks_timespan__noprefetch'] = chunks_timespan_noprefetch
    for x in ['hits', 'chunks_fetched', 'chunks_timespan']:
        episode_.s[x + '__noprefetch_nowaste'] = episode_.s[x + '__noprefetch']
    episode_.s['chunks_written__noprefetch_nowaste'] = useful_chunks_to_write
    # assert chunks_timespan_noprefetch <= episode_.s['chunks_timespan__prefetch'], (chunks_timespan_noprefetch, episode_, episode_.s)
    assert noprefetch_chunks_fetched >= episode_.s['chunks_written__prefetch']
    if 'e_age_ram' in d_:
        episode_.s['hits__ram_noprefetch'] = episode_.s['hits__ram_prefetch'] - excess_noprefetch_iops_ram
        episode_.s['chunks_fetched__ram_noprefetch'] = episode_.s['chunks_fetched__ram_prefetch'] + excess_noprefetch_chunks_saved_ram
    episode_.compute()


def process_obj_chunk_n_noprefetch(obj):
    d_ = interarrivals_from_accesses(obj)
    residences_by_e_age = {}
    for e_age_log_phy in obj['e_ages']:
        e_age_split = e_age_log_phy[d_['split_idx']]
        residencies = []
        for first, last, no_of_accesses in residences_from_interarrivals(d_['interarrivals'], e_age_split):
            chunk_counts, chunk_last_seen = get_chunk_stats(first, last, d_)
            episode_ = get_episode(d_, first, last, no_of_accesses, chunk_counts=chunk_counts, chunk_last_seen=chunk_last_seen)
            update_noprefetch_stats(first, last, d_, episode_, no_of_accesses, e_age_split)
            residencies.append(episode_)
        residences_by_e_age[e_age_log_phy] = residencies
    return residences_by_e_age


def process_obj_chunk_n_noprefetch_w_accs(obj):
    d_ = interarrivals_from_accesses(obj)
    residences_by_e_age = {}
    for e_age_log_phy in obj['e_ages']:
        e_age_split = e_age_log_phy[d_['split_idx']]
        residencies = []
        for first, last, no_of_accesses in residences_from_interarrivals(d_['interarrivals'], e_age_split):
            chunk_counts, chunk_last_seen = get_chunk_stats(first, last, d_)
            episode_ = get_episode(d_, first, last, no_of_accesses, chunk_counts=chunk_counts, chunk_last_seen=chunk_last_seen)
            update_noprefetch_stats(first, last, d_, episode_, no_of_accesses, e_age_split)
            episode_.accesses = obj['obj'][1].accesses[first:last+1]
            residencies.append(episode_)
        residences_by_e_age[e_age_log_phy] = residencies
    return residences_by_e_age


class SubEpisode(object):
    def __init__(self,
                 block_id,
                 chunk_range,
                 ts_logical,
                 ts_physical,
                 *,
                 num_accesses=None,
                 hits_iops=None,
                 hits_chunks=None,
                 time_from_prefetch=0,
                 first_iop=None):
        self.block_id = block_id
        self.chunk_range = chunk_range
        self.num_chunks = chunk_range[1] - chunk_range[0]
        # ts_logical -= time_from_prefetch[0]
        # ts_physical -= time_from_prefetch[1]
        self.ts_logical = ts_logical  # (first, last)
        self.ts_physical = ts_physical  # (first, last)
        self.timespan_logical = ts_logical[1] - ts_logical[0]
        self.timespan_phys = ts_physical[1] - ts_physical[0]
        self.first_iop = first_iop
        self.score = None
        self.threshold = None
        self.size = self.num_chunks * 131072
        self.num_accesses = num_accesses
        self.time_from_prefetch = time_from_prefetch
        self.s = {
            'chunks_written__prefetch': self.num_chunks,
            'chunks_written__noprefetch': self.num_chunks,
            'chunks_queried': self.num_chunks * num_accesses,
            'hits__prefetch': hits_iops,
            'hits__noprefetch': hits_iops - first_iop,
            'chunks_saved__prefetch': hits_chunks,
            'chunks_saved__noprefetch': hits_chunks,
        }
        assert first_iop > 0
        if self.time_from_prefetch[0] == 0:
            self.s['hits__prefetch'] = self.s['hits__noprefetch']
        self.s.update({
            'service_time_orig': service_time(hits_iops, self.s['chunks_queried']),
             # TODO: timespan prefetch should include time from prefetch (start of episode / last miss)
            'chunks_timespan__prefetch': (self.timespan_logical + self.time_from_prefetch[0]) * self.num_chunks,
            'chunks_timespan__noprefetch': self.timespan_logical * self.num_chunks,
            'service_time_saved__prefetch': service_time(self.s['hits__prefetch'], hits_chunks),
            'service_time_saved__noprefetch': service_time(self.s['hits__noprefetch'], hits_chunks),
        })

    @property
    def key(self):
        return self.block_id

    def export(self):
        kwargs = {
            'chunk_level': True,
            's_export': {
                'time_from_prefetch': self.time_from_prefetch,
                'service_time_saved__prefetch': self.s['service_time_saved__prefetch'],
                'service_time_saved__noprefetch': self.s['service_time_saved__noprefetch'],
                'prefetch_st_benefit': self.s['service_time_saved__prefetch'] - self.s['service_time_saved__noprefetch'],
            },
        }
        for k in ["num_accesses", "score", "threshold", "chunk_range"]:
            v = getattr(self, k)
            if v is not None:
                kwargs[k] = v
        offset = chunks_to_offset(*self.chunk_range)
        return ((self.key, self.ts_logical, self.ts_physical, offset), kwargs)

    def to_access(self):
        return trace_utils.cs_utils.BlkAccess(self.offset[0], self.size, self.ts_physical[0])

    def contains(self, chunk_id):
        return self.chunk_range[0] <= chunk_id < self.chunk_range[1]

    def __repr__(self):
        return f'SEp({self.block_id}, chunks={self.chunk_range}, ts={self.ts_physical}, tfp={self.time_from_prefetch}, threshold={self.threshold}, size={self.size}, num_acc={self.num_accesses}, hits=P{self.s["hits__prefetch"]},NP{self.s["hits__noprefetch"]}, chunks_saved=P{self.s["chunks_saved__prefetch"]},NP{self.s["chunks_saved__noprefetch"]})'
        # return f'Ep({self.key}, ts={self.ts_physical}, chunks={self.chunk_range}, threshold={self.threshold}, offset={self.offset}, size={self.size}, num_acc={self.num_accesses}, first_iop={self.first_iop})'


def get_fractional(first, last, d_, e_age_split):
    # for each chunk
        # get access times
        # optimization: if same as prev, merge
        # split access times into episodes

    chunks_acc_idx = defaultdict(list)

    for i in range(first, last+1):
        # new_time_ = d_[refs[d_['split_idx']]][i]
        # extra_chunks = 0
        for c in range(d_['chunk_starts'][i], d_['chunk_ends'][i]):
            chunks_acc_idx[c].append(i)

    fractional_iops_per_chunk_acc = 1/d_['num_chunks']

    subepisodes = []
    first_ts = (d_['tses_logical'][first],  d_['tses_phy'][first])
    chunk_groups_acc_idx = {}
    # prev = None
    # chunk_groups_acc_idx = []
    # for chunk_id, access_idxes in sorted(chunks_acc_idx.items()):
    #     if prev == access_idxes and chunk_id == chunk_groups_acc_idx[-1][0][1]:
    #         chunk_groups_acc_idx[-1][0][1] += 1
    #     else:
    #         chunk_groups_acc_idx.append([[chunk_id, chunk_id+1], access_idxes])
    #         prev = access_idxes

    # for (chunk_s, chunk_e), access_idxes in chunk_groups_acc_idx:
    for chunk_id, access_idxes in sorted(chunks_acc_idx.items()):
        tses_for_chunk = d_['tses'][access_idxes]
        chunk_interarrivals = np.diff(tses_for_chunk)
        for c_first_, c_last_, c_num_accesses_ in residences_from_interarrivals(chunk_interarrivals, e_age_split):
            subeps_idxes = tuple(access_idxes[c_first_:c_last_+1])
            if subeps_idxes not in chunk_groups_acc_idx:
                chunk_groups_acc_idx[subeps_idxes] = [[chunk_id, chunk_id+1]]
            elif chunk_groups_acc_idx[subeps_idxes][-1][1] == chunk_id:
                chunk_groups_acc_idx[subeps_idxes][-1][1] += 1
            else:
                chunk_groups_acc_idx[subeps_idxes].append([chunk_id, chunk_id+1])
    for access_idxes, chk_ranges in chunk_groups_acc_idx.items():
        access_idxes = np.asarray(access_idxes)
        iops = fractional_iops_per_chunk_acc[access_idxes].sum()
        c_num_accesses = len(access_idxes)
        # iops = fractional_iops[c_first:c_last+1].sum()
        # For no prefetch:
        # iops = fractional_iops[c_first+1:c_last+1].sum()
        # Time from start
        curr_ts = (d_['tses_logical'][access_idxes[0]], d_['tses_phy'][access_idxes[0]])
        time_from_start = (curr_ts[0] - first_ts[0], curr_ts[1] - first_ts[1])
        # Time from previous access
        # time_from_prefetch = d_['tses_logical'][access_idxes[c_first]] - d_['tses_logical'][max(access_idxes[c_first]-1, 0)]
        # TODO: Fix. Assumes e_age_split is logical.
        if time_from_start[0] > e_age_split:
            time_from_prefetch = (0, 0)
        else:
            time_from_prefetch = time_from_start
        for chunk_s, chunk_e in chk_ranges:
            num_chunks = chunk_e - chunk_s
            seps = SubEpisode(d_['block_id'],
                              (chunk_s, chunk_e),
                              (d_['tses_logical'][access_idxes[0]], d_['tses_logical'][access_idxes[-1]]),
                              (d_['tses_phy'][access_idxes[0]], d_['tses_phy'][access_idxes[-1]]),
                              num_accesses=c_num_accesses,
                              hits_iops=iops * num_chunks,
                              hits_chunks=(c_num_accesses - 1) * num_chunks,
                              time_from_prefetch=time_from_prefetch,
                              first_iop=fractional_iops_per_chunk_acc[access_idxes[0]] * num_chunks)
            seps.s['episode_id'] = (d_['block_id'], (first, last))
            seps.s['acc_idxes'] = d_['tses'][access_idxes]
            seps.s['time_from_start'] = time_from_start
            subepisodes.append(seps)
    return subepisodes


def process_obj_fractional(obj):
    d_ = interarrivals_from_accesses(obj)
    residences_by_e_age = {}
    for e_age_log_phy in obj['e_ages']:
        e_age_split = e_age_log_phy[d_['split_idx']]
        residencies = []
        for first, last, no_of_accesses in residences_from_interarrivals(d_['interarrivals'], e_age_split):
            residencies += get_fractional(first, last, d_, e_age_split)
        residences_by_e_age[e_age_log_phy] = residencies
    return residences_by_e_age


def process_obj_chunkheuristic(obj):
    d_ = interarrivals_from_accesses(obj)
    residences_by_e_age = {}
    for e_age_log_phy in obj['e_ages']:
        e_age_split = e_age_log_phy[d_['split_idx']]
        residencies = []
        for first, last, no_of_accesses in residences_from_interarrivals(d_['interarrivals'], e_age_split):
            chunk_ranges = Counter()
            for i in range(first, last+1):
                chunk_r = offset_to_chunks(d_['byte_starts'][i], d_['byte_ends'][i])
                assert chunk_r[0] < chunk_r[1], (d_['byte_starts'][i], d_['byte_ends'][i])
                chunk_ranges[chunk_r] += 1
            left = d_['byte_starts'][first:last+1].min()
            right = d_['byte_ends'][first:last+1].max()
            residencies.append((d_['block_id'],
                                (d_['tses_phy'][first], d_['tses_phy'][last]),
                                (d_['tses_logical'][first], d_['tses_logical'][last]),
                                chunk_ranges,
                                offset_to_chunks(left, right)))
        residences_by_e_age[e_age_log_phy] = residencies
    return residences_by_e_age


def run_heuristic(res,
                  thelper=None,
                  trace_kwargs=None):
    if thelper is None:
        thelper = trace_utils.TraceHelper(element_size=8,  # 8MB
                                          trace_kwargs=trace_kwargs)
    pq = {}
    tses = {}
    for episode_id, (block_id, tee, tee_l, chunk_counts, _) in enumerate(res):
        tses[episode_id] = (tee, tee_l)
        for (s, e), hits in chunk_counts.items():
            cost = e-s
            assert cost > 0
            more = len(chunk_counts) > 1
            # (s,e) are exclusive here
            pq[(block_id, (s,e), episode_id, more)] = (-hits/(.001+cost), cost, -hits)
    pq = pqdict.pqdict(pq)

    chunks_written = {}
    ranges_written = {}
    decisions = {}
    episodes = []

    total_writes = 0
    total_iops_saved = 0
    episodes_admitted = Counter()
    chunks_if_full_episode = 0

    stats = {'writes': [], 'iops': [], 'episodes_admitted': [], 'writes_fullepisode': []}
    for item, val in tqdm(pq.popitems(), total=len(pq),
                          desc='heuristic', **tqdm_kwargs):
        block_id, (s,e), episode_id, more = item
        score, cost, hits = val
        hits = -hits
        total_writes += cost
        total_iops_saved += hits
        if episode_id not in episodes_admitted:
            ep_chunkrange = res[episode_id][4]
            chunks_if_full_episode += ep_chunkrange[1] - ep_chunkrange[0]
        episodes_admitted[episode_id] += 1
        stats['writes'].append(total_writes)
        stats['iops'].append(total_iops_saved)
        stats['episodes_admitted'].append(len(episodes_admitted))
        stats['writes_fullepisode'].append(chunks_if_full_episode)

        key = (block_id, episode_id)

        # (self, key, ts_logical, ts_physical, byte_offset

        episodes.append(
            Episode(block_id,
                    tses[episode_id][1],  # TODO: Change to actual length
                    tses[episode_id][0],  # Prefetching
                    chunks_to_offset(s, e),
                    num_accesses=hits,
                    chunk_level=True,
                    threshold=thelper.chunks_to_wr_mbps(total_writes)))
        assert episodes[-1].chunk_range == (s, e), (episodes[-1].chunk_range, (s, e))

        if more:
            if key not in chunks_written:
                chunks_written[key] = set()
                ranges_written[key] = set()
            count = 0
            for c in range(s, e):
                dkey = (block_id, tses[episode_id][0], c, episode_id)
                if dkey not in decisions:
                    count += 1
                    decisions[dkey] = thelper.chunks_to_wr_mbps(total_writes)
                chunks_written[key].add(c)
            assert count == cost, (count, cost, (s, e))
        # Note: hits here is potentially inaccurate (does not exclude first hit)

            ranges_written[key].add((s, e))
            chunk_counts = res[episode_id][3]
            for (s_, e_), hits in chunk_counts.items():
                if (s_, e_) not in ranges_written[key] and s <= e_-1 and s_ <= e - 1:
                    new_cost = sum(1 for c in range(s_, e_)
                                   if c not in chunks_written[key])
                    pqkey = (block_id, (s_, e_), episode_id, True)
                    pq[pqkey] = (-hits/(.001+new_cost), new_cost, -hits)

    stats = {k: np.array(v) for k, v in stats.items()}
    stats['iops_saved'] = stats['iops'] - stats['episodes_admitted']
    stats['iops_saved_ratio'] = stats['iops_saved'] / thelper.logical_time
    stats['write_rates_mb'] = thelper.chunks_to_wr_mbps(stats['writes'])
    stats['write_rates_mb_fullepisode'] = thelper.chunks_to_wr_mbps(stats['writes_fullepisode'])

    return stats, episodes


def generate_residencies(e_ages,
                         supplied_ea='physical',
                         split_by='logical',
                         residency_fn=process_obj_chunk_n_noprefetch,
                         workers=32,
                         batchsize=32,  # No of accesses to give each worker
                         thelper=None,
                         residencylist_class=ResidencyListPrefetchAware,
                         trace_kwargs=None, **kwargs):
    accesses_by_obj = trace_utils.get_accesses_kv(**trace_kwargs)['acc']
    if thelper is None:
        thelper = trace_utils.TraceHelper(element_size=8,  # 8MB
                                          trace_kwargs=trace_kwargs)

    if supplied_ea == 'physical':
        e_ages_phys = e_ages
        e_ages_logical = thelper.phy_dur_to_logical(
            np.asarray(e_ages)).tolist()
    elif supplied_ea == 'logical':
        e_ages_logical = e_ages
        e_ages_phys = thelper.logical_dur_to_phy(np.asarray(e_ages)).tolist()

    e_ages_log_phy = list(zip(e_ages_logical, e_ages_phys))
    with Pool(processes=workers) as p:
        args = (dict(obj=obj, e_ages=e_ages_log_phy, split_by=split_by, **kwargs)
                for obj in accesses_by_obj.items())
        r_stream = tqdm(p.imap_unordered(residency_fn,
                                         args,
                                         chunksize=batchsize),
                        total=len(accesses_by_obj),
                        desc='gen_episodes', **tqdm_kwargs)
        combined = {e_age: [] for e_age in e_ages_log_phy}
        for obj_result in r_stream:
            for e_age in e_ages_log_phy:
                combined[e_age] += obj_result[e_age]
    return {k[1] if supplied_ea == 'physical' else k[0]:
            residencylist_class(v, e_age=k, trace_helper=thelper)
            for k, v in combined.items()}


get_residencies = ep_utils.cache_residencies(generate_residencies)
