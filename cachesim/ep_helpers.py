from collections import defaultdict
from collections import namedtuple
import functools
import shelve

from .utils import ods
from .legacy_utils import GET_OPS, PUT_OPS
from ..episodic_analysis.episodes import Episode
from ..episodic_analysis.episodes import service_time


class AccessPlus(object):
    """
    Wrapper for BlkAccess.
    Goals: easier to pass around, optional bits, accommodate PUTs.
    Contains: Access + Block ID, TimeStamp, Prefetch predictions.
    """
    def __init__(self, block_id, access, block_level=False):
        self.acc = access
        self.block_id = block_id
        self.ts = Timestamp(physical=access.ts, logical=access.ts_logical)
        self.chunks_ = access.chunks()
        # Exclusive range
        self.chunk_range = self.chunks[0], self.chunks[-1] + 1
        self.num_chunks = self.chunk_range[1] - self.chunk_range[0]
        assert self.chunks == list(range(*self.chunk_range))
        self.pred_prefetch = None

        assert access.features
        if self.features.op in GET_OPS:
            self.op = 'GET'
        elif self.features.op in PUT_OPS:
            self.op = 'PUT'
        else:
            raise NotImplementedError(f"Unknown op: {self.features.op}")

    @property
    def features(self):
        return self.acc.features

    @property
    def chunks(self):
        """In case we want to implement something more dynamic? TODO: Reconsider."""
        return self.chunks_

    @property
    def is_get(self):
        return self.op == 'GET'

    @property
    def is_put(self):
        return self.op == 'PUT'


def _init_sim_eps_s(eps_s):
    eps_s["sim_chunk_written"] = defaultdict(int)
    eps_s["sim_admitted"] = False
    eps_s["sim_admitted_ts"] = set()


def _lookup_episode(decisions, block_id, ts, *, chunk_id=None, prune_old=False):
    if decisions is None:
        return None
    if isinstance(decisions, shelve.Shelf) or str(block_id) in decisions:
        block_id = str(block_id)
    decs = decisions[block_id]
    if isinstance(decs[0], tuple):
        def filter_dct(eps_kwargs_):
            # if "--fast" in sys.argv and "optplus" not in sys.argv:
            #     for k in ["chunk_last_seen", "chunk_counts"]:
            #         if k in eps_kwargs_:
            #             del eps_kwargs_[k]
            return eps_kwargs_
        decs = tuple(Episode(*eps_args, **filter_dct(eps_kwargs))
                     for eps_args, eps_kwargs in decs)
    result = None
    for episode in decs:
        eps_ts = episode.ts_physical
        if chunk_id is not None and episode.chunk_level and not episode.contains(chunk_id):
            continue
        if eps_ts[0] <= ts.physical <= eps_ts[1]:
            result = episode
            break
        # Pick the episode with the closest start time
        if eps_ts[0] <= ts.physical and (result is None or eps_ts[0] > result.ts_physical[0]):
            result = episode
    if result is None:
        raise Exception(f'Error: Episode not found: (block_id, ts, chunk_id) ({block_id}, Timestamp{ts}, chunk_id={chunk_id})')
    if prune_old:
        # Note: assumes that timestamps are strictly increasing in calls to this function.
        decs = tuple(eps for eps in decs if eps.ts_physical[1] > ts.physical)
    if not isinstance(decisions, shelve.Shelf):
        decisions[block_id] = decs
    if prune_old and len(decs) == 0:
        del decisions[block_id]
    if ts.physical > result.ts_physical[1]:
        ods.bump("warning_lookup_after_episode_end")
    if "sim_chunk_written" not in episode.s:
        _init_sim_eps_s(episode.s)
    return result


def _get_chunks_for_episode(decisions, block_id, ts, threshold=None):
    if isinstance(decisions[block_id][0], tuple):
        decisions[block_id] = tuple(
            Episode(*eps_args, **eps_kwargs)
            for eps_args, eps_kwargs in decisions[block_id])
    chunks = set()
    for episode in decisions[block_id]:
        if threshold is not None and episode.threshold > threshold:
            continue
        eps_ts = episode.ts_physical
        if eps_ts[0] <= ts.physical <= eps_ts[1]:
            for chunk_id in range(*episode.chunk_range):
                chunks.add(chunk_id)
    return chunks


def _prefetchable_chunks(decisions, block_id, ts_prefetch, assumed_ea=None, threshold=None):
    if isinstance(decisions[block_id][0], tuple):
        decisions[block_id] = tuple(
            Episode(*eps_args, **eps_kwargs)
            for eps_args, eps_kwargs in decisions[block_id])
    chunks = set()
    eps_chunk = {}
    # TODO: use assumed EA so we don't just need to prefetch at episode start
    # assert assumed_ea is not None
    for episode in decisions[block_id]:
        if threshold is not None and episode.threshold > threshold:
            continue
        if episode.s_export['time_from_prefetch'][0] == 0:
            continue
        eps_ts = episode.ts_physical
        # print(eps_ts, episode.s_export['time_from_prefetch'][1], ts_prefetch.physical)
        to_prefetch = assumed_ea is not None and assumed_ea.physical != 0 and ts_prefetch.physical < eps_ts[0] and ts_prefetch.physical + assumed_ea.physical > eps_ts[0]
        to_prefetch = to_prefetch or (eps_ts[0] - episode.s_export['time_from_prefetch'][1] <= ts_prefetch.physical <= eps_ts[0])
        if to_prefetch:
            for chunk_id in range(*episode.chunk_range):
                chunks.add(chunk_id)
                if chunk_id not in eps_chunk:
                    eps_chunk[chunk_id] = episode
                    if "sim_chunk_written" not in episode.s:
                        _init_sim_eps_s(episode.s)
    return chunks, eps_chunk


@functools.total_ordering
class Timestamp(namedtuple('Timestamp', ['logical', 'physical'])):
    def __hash__(self):
        return hash((self.logical, self.physical))

    def __infer(self, other):
        if type(other) == Timestamp:
            return other
        if type(other) == int and other == 0:
            return Timestamp(0, 0)
        raise ValueError(f"{other}: explicitly use .physical or .logical")

    def __sub__(self, other):
        other = self.__infer(other)
        return Timestamp(self.logical - other.logical,
                         self.physical - other.physical)

    def __add__(self, other):
        other = self.__infer(other)
        return Timestamp(self.logical + other.logical,
                         self.physical + other.physical)

    def __lt__(self, other):
        other = self.__infer(other)
        return super().__lt__(other)

    def __gt__(self, other):
        other = self.__infer(other)
        return super().__gt__(other)

    def __eq__(self, other):
        other = self.__infer(other)
        return super().__eq__(other)

    def __truediv__(self, other):
        return Timestamp(self.logical / other, self.physical / other)

    def __round__(self, n=0):
        return Timestamp(round(self.logical, n), round(self.physical, n))

    def __format__(self, format_spec):
        phy, ext = self.physical, 's'
        if phy > 3600:
            phy, ext = phy / 3600, 'h'
        elif phy > 60:
            phy, ext = phy / 60, 'm'
        return f'({format(self.logical, format_spec)},{format(phy, format_spec)}{ext})'


def record_service_time_get(need_fetch, need_prefetch, acc):
    tags = [f"ns_{acc.features.namespace}", f"user_{acc.features.user}"]

    ods.bump("fetches_ios")
    range_fetch = min(need_fetch), max(need_fetch)
    num_fetch = range_fetch[1] - range_fetch[0] + 1
    range_prefetch = range_fetch
    if need_prefetch:
        range_prefetch = min(range_prefetch[0], min(need_prefetch)), max(max(need_prefetch), range_prefetch[1])
    num_with_prefetch = range_prefetch[1] - range_prefetch[0] + 1
    ods.bump("fetches_chunks", v=num_with_prefetch)
    # TODO: Replace _ with namespacing.
    ods.bump("fetches_chunks_demandmiss", v=num_fetch)
    ods.bump("fetches_chunks_prefetch", v=num_with_prefetch-num_fetch)
    ods.bump("service_time", v=service_time(1, num_with_prefetch))
    ods.bump("service_time_used", v=service_time(1, num_with_prefetch))
    ods.bump("service_time_used_demand", v=service_time(1, num_fetch))
    ods.bump("service_time_used_prefetch", v=service_time(0, num_with_prefetch-num_fetch))

    for tag in tags:
        ods.bump(["fetches", "ios", tag])
        ods.bump(["fetches", "chunks", tag], v=num_with_prefetch)
        ods.bump(["fetches", "chunks", "demandmiss", tag], v=num_fetch)
        ods.bump(["fetches", "chunks", "prefetch", tag], v=num_with_prefetch-num_fetch)
        ods.bump(["service_time", tag], v=service_time(1, num_with_prefetch))
        ods.bump(["service_time", "get", tag], v=service_time(1, num_with_prefetch))
        ods.bump(["service_time", "demand", tag], v=service_time(1, num_fetch))
        ods.bump(["service_time", "prefetch", tag], v=service_time(0, num_with_prefetch-num_fetch))

    # If we measure from first miss till end
    range2_fetch = min(need_fetch), max(acc.chunks)
    assert range2_fetch[1] >= range_fetch[1]
    range2_prefetch = range_prefetch[0], max(range2_fetch[1], range_prefetch[1])
    assert range2_prefetch[1] >= range2_fetch[1]
    assert range2_prefetch[0] <= range2_fetch[0]
    num_fetch2 = range2_fetch[1] - range2_fetch[0] + 1
    num_with_prefetch2 = range2_prefetch[1] - range2_prefetch[0] + 1
    ods.bump("fetches2_chunks", v=num_with_prefetch2)
    ods.bump("fetches2_chunks_demandmiss", v=num_fetch2)
    ods.bump("fetches2_chunks_prefetch", v=num_with_prefetch2-num_fetch2)
    ods.bump("service_time_used2", v=service_time(1, num_with_prefetch2))
    ods.bump("service_time_used2_demand", v=service_time(1, num_fetch2))
    ods.bump("service_time_used2_prefetch", v=service_time(0, num_with_prefetch2-num_fetch2))

    # If we measure from first chunk in access to end
    range3_fetch = min(acc.chunks), max(acc.chunks)
    range3_prefetch = min(range2_prefetch[0], range3_fetch[0]), range2_prefetch[1]
    num_fetch3 = range3_fetch[1] - range3_fetch[0] + 1
    num_with_prefetch3 = range3_prefetch[1] - range3_prefetch[0] + 1
    ods.bump("fetches3_chunks", v=num_with_prefetch3)
    ods.bump("fetches3_chunks_demandmiss", v=num_fetch3)
    ods.bump("fetches3_chunks_prefetch", v=num_with_prefetch3-num_fetch3)
    ods.bump("service_time_used3", v=service_time(1, num_with_prefetch3))
    ods.bump("service_time_used3_demand", v=service_time(1, num_fetch3))
    ods.bump("service_time_used3_prefetch", v=service_time(0, num_with_prefetch3-num_fetch3))


def record_service_time_put(acc):
    ods.bump("puts_ios")
    ods.bump("puts_chunks", v=acc.num_chunks)
    ods.bump("service_time_writes", v=service_time(1, acc.num_chunks))
    ods.bump("service_time", v=service_time(1, acc.num_chunks))
    ods.bump(["puts_ios", "op", acc.features.op.name])
    ods.bump(["puts_chunks", "op", acc.features.op.name], v=acc.num_chunks)
    ods.bump(["service_time_writes", "op", acc.features.op.name], v=service_time(1, acc.num_chunks))

    tags = [f"ns/{acc.features.namespace}", f"user/{acc.features.user}"]
    for tag in tags:
        ods.bump(["puts_ios", tag])
        ods.bump(["puts_chunks", tag], v=acc.num_chunks)
        ods.bump(["service_time_writes", tag], v=service_time(1, acc.num_chunks))
