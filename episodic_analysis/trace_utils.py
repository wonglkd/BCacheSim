try:
    from ..cachesim import utils as cs_utils
except (ValueError, ImportError):
    import os
    import sys
    dirname = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, '..'))
    import cachesim.utils as cs_utils
from . import local_cluster


_cached = {}


def get_accesses(sample_ratio=1, region=None, trace_group=None,
                 start=0, subtrace='full',
                 only_gets=True,
                 override_cache=False,
                 **kwargs):
    global _cached
    key = (sample_ratio, trace_group, region, start, subtrace, only_gets)
    if key not in _cached or override_cache:
        # accesses, start_ts, end_ts, physical_to_logical
        _cached[key] = cs_utils.read_processed_file_with_logical_ts(
            local_cluster.tracefilename(sample_ratio=sample_ratio, region=region, start=start, subtrace=subtrace, trace_group=trace_group),
            only_gets=only_gets, **kwargs)
    return key, _cached[key][3], _cached[key][0]


ALIGNMENT = cs_utils.BlkAccess.ALIGNMENT
MAX_BLOCK_SIZE = cs_utils.BlkAccess.MAX_BLOCK_SIZE

run_length_encode = cs_utils.run_length_encode


def get_accesses_kv(sample_ratio=1, region=None, **kwargs):
    global _cached
    key, _, _ = get_accesses(sample_ratio=sample_ratio, region=region, **kwargs)
    # 0: k_accesses, 1: start_ts, 2: end_ts, 3: physical_to_logical
    return {
        'acc': _cached[key][0],
        # Logical time also equal to no of accesses
        'logical_time': len(_cached[key][3]),
        'duration': _cached[key][2] - _cached[key][1],
        'start_ts': _cached[key][1],
        'end_ts': _cached[key][2],
        'num_blocks': len(_cached[key][0]),
    }


class TraceHelper(object):
    def __init__(self, *,
                 element_size=8,  # 8 MB
                 trace_kwargs=None):
        self.trace_kwargs = trace_kwargs or {}
        acc_kv = get_accesses_kv(**self.trace_kwargs)
        for k in ['duration', 'start_ts', 'end_ts', 'logical_time']:
            setattr(self, k, acc_kv[k])
        self.element_size = element_size

    def phy_dur_to_logical(self, phy_duration):
        return phy_duration / self.duration * self.logical_time

    def phy_ts_to_logical(self, phy_ts):
        return self.phy_dur_to_logical(phy_ts - self.start_ts)

    def logical_dur_to_phy(self, logical_duration):
        return logical_duration / self.logical_time * self.duration

    def logical_ts_to_phy(self, logical_ts):
        return self.logical_dur_to_phy(logical_ts) + self.start_ts

    def downsample(self, qty):
        return qty / 100 * self.trace_kwargs['sample_ratio']

    def upsample(self, qty):
        return qty * 100 / self.trace_kwargs['sample_ratio']

    def wr_mbps_to_arrivals_allowed(self, wr):
        return self.downsample(wr / self.element_size * self.duration)

    def bytes_to_wr_mbps(self, size):
        return self.upsample(size / 1024 / 1024 / self.duration)

    def chunks_to_wr_mbps(self, chunks):
        return self.arrivals_to_wr_mbps(chunks / (MAX_BLOCK_SIZE / ALIGNMENT))

    def arrivals_to_wr_mbps(self, arrivals):
        return self.items_to_mb(arrivals / self.duration)

    def items_to_mb(self, arrivals):
        return self.upsample(arrivals * self.element_size)


def wr_to_dwpd(wr, csize, rescale_for_cachelib_overhead=True):
    csize_gb = 357.44751
    dwpd = wr / (csize * 1000) * 24 * 3600
    if rescale_for_cachelib_overhead:
        dwpd /= 400 / csize_gb
    return round(dwpd, 3)


def dwpd_to_wr(dwpd, csize, rescale_for_cachelib_overhead=True):
    csize_gb = 357.44751
    wr_ = dwpd * csize * 1000 / (24*3600)
    if rescale_for_cachelib_overhead:
        wr_ *= 400 / csize_gb
    return round(wr_, 3)
