"""Meta trace format specific utilities."""
from enum import Enum, unique
import sys

try:
    from ..episodic_analysis.constants_meta import trace_has_pipeline
except ImportError:
    from ..episodic_analysis.constants_public import trace_has_pipeline


@unique
class OpType(Enum):
    GET_TEMP = 1
    GET_PERM = 2
    PUT_TEMP = 3
    PUT_PERM = 4
    GET_NOT_INIT = 5
    PUT_NOT_INIT = 6
    UNKNOWN = 100


PUT_OPS = [OpType.PUT_PERM, OpType.PUT_TEMP, OpType.PUT_NOT_INIT]
GET_OPS = [OpType.GET_PERM, OpType.GET_TEMP, OpType.GET_NOT_INIT]


# MODEL CONFIG
ACCESS_HISTORY_COUNT = 6
FEATURES = [f"bf_{i}" for i in range(0, ACCESS_HISTORY_COUNT)] + [
    "op",
    "namespace",
    "user",
]


class KeyFeatures(object):
    # helps reduce memory footprint
    __slots__ = ["op", "pipeline", "namespace", "user", "offset", "size", "repeat"]

    def __init__(self, *, op=None, pipeline=None, namespace=None, user=None, offset=None, size=None, repeat=1):
        self.op = OpType(op)
        self.pipeline = pipeline
        self.namespace = namespace
        self.user = user
        self.offset = offset
        self.size = size
        self.repeat = repeat

    def __str__(self):
        return "op={}, pipeline={}, namespace={}, user={}".format(
            self.op, self.pipeline, self.namespace, self.user
        )

    def __repr__(self):
        # for easy debugging purpose
        return self.__str__()

    def toList(self, with_size=False):
        """Used in training and simulation for ML admission policies
        `pipeline` feature is not used in ML model due to it has too many categories to
        be encoded for production inference, as well as its collinearity with
        namespace and user, therefore omitted here.
        """
        feat = [self.op.value, self.namespace, self.user]
        if with_size:
            return feat + [self.offset, self.offset+self.size, self.size]
        return feat


class BlkAccess(object):
    # chunk alignment for warm storage
    ALIGNMENT = 128 * 1024
    MAX_BLOCK_SIZE = 8 * 1024 * 1024

    # helps reduce memory footprint
    __slots__ = ["ts", "ts_logical", "offset", "endoffset", "c", "features",
                 "orig_offset", "orig_endoffset", "block", "episode"]

    @staticmethod
    def roundDownToBlockBegin(off):
        return (off // BlkAccess.ALIGNMENT) * BlkAccess.ALIGNMENT

    @staticmethod
    def roundUpToBlockEnd(off):
        return (off // BlkAccess.ALIGNMENT + 1) * BlkAccess.ALIGNMENT - 1

    # offsets can be in the middle of a block. round them to the alignment to
    # emulate caching at a chunk level
    def __init__(self, offset, size, time, *,
                 features=None, ts_logical=None, block=None, episode=None):
        self.block = block
        self.ts = time
        self.ts_logical = ts_logical
        self.orig_offset = offset
        self.orig_endoffset = offset + size - 1  # inclusive
        self.offset = BlkAccess.roundDownToBlockBegin(offset)
        self.endoffset = BlkAccess.roundUpToBlockEnd(offset + size - 1)
        # list of chunks
        self.c = []
        self.features = features
        self.episode = episode

    def __str__(self):
        return "Acc(block={}, offset={}, size={}, ts={}, features={})".format(
            self.block, self.offset, self.size(), self.ts, self.features
        )

    def __repr__(self):
        # for easy debugging purpose
        return self.__str__()

    def size(self):
        return self.end() - self.start() + 1

    def start(self):
        return int(self.offset)

    def end(self):
        return int(self.endoffset)

    def origsize(self):
        return self.orig_endoffset - self.offset + 1

    def chunks(self):
        if DEBUG_FLAG_ONECHUNK():
            return list(range(1))
        if len(self.c) > 0:
            return self.c
        i = self.start()
        while i < self.end():
            # TODO: Figure out if this should be <= self.end()
            i += BlkAccess.ALIGNMENT
            self.c.append(i // BlkAccess.ALIGNMENT)
        return self.c

    def num_chunks(self):
        return len(self.chunks())


class KeyAndAccesses(object):
    # helps reduce memory footprint
    __slots__ = ["key", "accesses"]

    def __init__(self, key):
        self.key = key
        self.accesses = []

    def addAccess(self, access):
        self.accesses.append(access)

    def sortAccesses(self):
        self.accesses.sort(key=lambda a: a.ts, reverse=False)


# read the processed file and return a dictionary of key to all its BlkAccess
# sorted by access time.
def read_processed_file(f, get_features=True, only_gets=True, only_puts=False,
                        with_pipeline=None, assert_monotonic=True,
                        min_ts_from_start=None,
                        max_ts_from_start=None):
    if with_pipeline is None:
        with_pipeline = trace_has_pipeline(f)
    print(f"Reading from file {f}")
    accesses = {}
    start_ts = None
    end_ts = None
    last_ts = None
    i = 0
    with open(f, "r") as of:
        # key_map = {}
        for line in of:
            try:
                if line.startswith('#'):
                    continue
                parts = line.split(" ")
                parts = [p.strip("\n") for p in parts]
                # k = lookup_or_add_uuid(key_map, parts[0])
                k = parts[0]
                off = int(parts[1])
                size = int(parts[2])
                ts = float(parts[3])
                pipeline = None
                op = None
                repeat = 1
                if size == 0:
                    print(f"ERROR! 0-sized IO: {line}")
                    continue

                # compute the time window of the trace
                # TODO: TS start vs real start
                start_ts = ts if start_ts is None else min(start_ts, ts)
                end_ts = ts if end_ts is None else max(end_ts, ts)
                if last_ts is not None:
                    assert last_ts <= ts, f"last_ts > ts: {last_ts} > {ts}"
                last_ts = ts
                if min_ts_from_start and ts - start_ts < min_ts_from_start:
                    continue
                if max_ts_from_start and ts - start_ts > max_ts_from_start:
                    break

                f = None
                if len(parts) >= 8:
                    if with_pipeline:
                        op = int(parts[4])
                        pipeline = int(parts[5])
                        namespace = int(parts[6])
                        user = int(parts[7])
                        if len(parts) >= 9:
                            repeat = int(parts[8])
                    else:
                        op = int(parts[4])
                        namespace = int(parts[5])
                        user = int(parts[6])
                        hostname = int(parts[7])
                        if len(parts) >= 9:
                            repeat = int(parts[8])
                        k = (k, hostname)
                elif len(parts) == 7:
                    op = int(parts[4])
                    namespace = int(parts[5])
                    user = int(parts[6])

                if op is not None:
                    f = KeyFeatures(op=op, pipeline=pipeline, namespace=namespace, user=user, offset=off, size=size, repeat=1)

                if only_gets and (f is None or f.op not in GET_OPS):
                    continue
                if only_puts and (f is None or f.op not in PUT_OPS):
                    continue

                if not get_features:
                    f = None

                if k not in accesses:
                    accesses[k] = KeyAndAccesses(k)

                interval = 0 if repeat == 1 else 1. / (repeat - 1)
                for repeat_i in range(repeat):
                    acc = BlkAccess(off, size, ts + repeat_i * interval, features=f, block=k, ts_logical=i)
                    accesses[k].addAccess(acc)
                    i += 1
                    # This is used for IOPS saved ratio, so going by chunks doesn't work.
                    # But going by chunks might be more accurate for determining eviction age.
                    # TODO: += repeat * chunks
                    # i += acc.num_chunks()
            except (ValueError, IndexError):
                print("Error in parsing line ", line, parts)

    for k in accesses:
        accesses[k].sortAccesses()

    return accesses, start_ts, end_ts


def add_logical_timestamps(k_accesses):
    physical_timestamps = set()
    for k in k_accesses:
        for a in k_accesses[k].accesses:
            physical_timestamps.add((a.ts, a.ts_logical))
    physical_to_logical = dict((b, a) for a, b in enumerate(sorted(physical_timestamps)))
    for k in k_accesses:
        for a in k_accesses[k].accesses:
            a.ts_logical = physical_to_logical[(a.ts, a.ts_logical)]
    return physical_to_logical


def read_processed_file_with_logical_ts(f, get_features=True, **kwargs):
    k_accesses, start_ts, end_ts = read_processed_file(f, get_features=get_features, **kwargs)
    physical_to_logical = add_logical_timestamps(k_accesses)
    return k_accesses, start_ts, end_ts, physical_to_logical


# read the processed file and return  list of (k, BlkAccess) sorted by access time.
def read_processed_file_list_accesses(f, **kwargs):
    k_accesses, start_ts, end_ts = read_processed_file(f, **kwargs)
    _ = add_logical_timestamps(k_accesses)
    accesses = []

    for k in k_accesses:
        for a in k_accesses[k].accesses:
            accesses.append((k, a))
    accesses.sort(key=lambda a: a[1].ts_logical, reverse=False)

    return accesses, start_ts, end_ts


def DEBUG_FLAG_ONECHUNK():
    return "--one-chunk" in sys.argv


def get_output_suffix(options):
    out = "/"
    # admission policy notes
    if options.rejectx_ap:
        out += f"rejectx-ap-{options.ap_threshold:g}_{options.ap_probability:g}"
    elif options.ap == "hybrid" or options.ap.startswith("either") or options.ap.startswith("and"):
        if options.ap == "hybrid":
            out += f"hybrid-ap-{options.hybrid_ap_threshold:g}_"
        else:
            out += f"{options.ap}-ap_"
        if options.opt_ap_threshold:
            out += f"opt-{options.opt_ap_threshold:g}_"
        if options.rejectx_ap_threshold:
            out += f"rejectx-{options.rejectx_ap_factor:g}_{options.rejectx_ap_threshold:g}_"
        out += f"ml-{options.ap_threshold:g}_{options.learned_ap_filter_count}"
    elif options.learned_ap:
        out += f"ml-ap-{options.ap_threshold:g}_{options.learned_ap_filter_count}"
    elif options.coinflip_ap:
        out += f"coinflip-ap-{options.ap_probability:g}"
    elif options.offline_ap:
        out += f"offline-ap-{options.ap_threshold:g}"
    elif options.ap == "flashieldprob":
        out += f"{options.ap}-{options.flashieldprob_ap_min_hits}-{options.ap_threshold:g}"
    else:
        out += f"{options.ap}-{options.ap_threshold:g}"

    # cache type notes
    if options.lirs:
        out += "_lirs"
    elif options.fifo:
        out += "_fifo"
    else:
        out += "_lru"

    if options.write_mbps != 0:
        out += f"-{options.write_mbps:g}"

    out += f"_{options.size_gb:g}GB"
    return out
