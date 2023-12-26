import contextlib
import datetime
import fnmatch
import gzip
import pickle
import os
import shelve
import subprocess
import hashlib
import itertools
import json
# import jsonpickle
import sys
import time
from pathlib import Path
from .legacy_utils import BlkAccess
from .legacy_utils import read_processed_file_list_accesses
from .legacy_utils import read_processed_file_with_logical_ts  # noqa: F401
from .legacy_utils import GET_OPS, PUT_OPS, get_output_suffix  # noqa: F401


def stream_processed_accesses(f, *, region=None, input_file_name=None, sample_ratio=None, start=None, **kwargs):
    # memoize
    assert os.path.exists(f), f"{f} does not exist"
    filehash = subprocess.check_output(f"md5sum {f}", shell=True).split()[0]
    # import platform
    # pywhich = platform.python_implementation()
    kwargs_hash = hashlib.md5(json.dumps(kwargs, sort_keys=True).encode('utf-8')).hexdigest()[-6:]
    cached_filename = f'/tmp/cache-sim-accesses-{region}_{input_file_name}_{filehash[-6:].decode()}_{kwargs_hash}_batched.pkl'
    if not os.path.exists(cached_filename) or os.path.getmtime(cached_filename) <= os.path.getmtime(__file__) or os.path.getmtime(cached_filename) <= os.path.getmtime(__file__.replace("utils", "legacy_utils")):
        if os.path.exists(cached_filename):
            mtime = os.path.getmtime(cached_filename)
        else:
            mtime = 0
        accesses, start_ts, end_ts = read_processed_file_list_accesses(f, **kwargs)
        max_key = max(accesses, key=lambda x: x[0])
        total_iops = len(accesses)
        total_iops_get = sum(1 for _, acc in accesses if acc.features.op in GET_OPS)
        total_iops_put = sum(1 for _, acc in accesses if acc.features.op in PUT_OPS)
        assert total_iops == total_iops_get + total_iops_put
        trace_duration_secs = round(end_ts - start_ts, 2)
        stats = {
            'total_iops': total_iops,
            'total_iops_get': total_iops_get,
            'total_iops_put': total_iops_put,
            'max_key': max_key,
            'trace_duration_secs': trace_duration_secs,
            'start_ts': start_ts,
            'end_ts': end_ts,
            'filename': f,
            'trace_hash': filehash,
            'kwargs_hash': kwargs_hash,
            'kwargs': kwargs,
        }
        if not os.path.exists(cached_filename) or os.path.getmtime(cached_filename) == mtime:
            if os.path.exists(cached_filename):
                os.remove(cached_filename)
            try:
                with open(cached_filename, 'wb') as f:
                    pickle.dump(stats, f, protocol=4)
                    batch = []
                    for acc in accesses:
                        batch.append(acc)
                        if len(batch) >= 512:
                            pickle.dump(batch, f, protocol=4)
                            batch = []
                    if batch:
                        pickle.dump(batch, f, protocol=4)
                        batch = []
            except:
                if os.path.exists(cached_filename):
                    os.remove(cached_filename)
        return stats, accesses
    else:
        print(f"Streaming {f} ({cached_filename})")
        gen = stream_pickle(cached_filename)
        stats = next(gen)
        return stats, gen


def stream_pickle(filename):
    with open(filename, 'rb') as f:
        while f.peek(1):
            try:
                # Note: cannot use same unpickler, as some info is left over.
                batch = pickle.load(f)
                if isinstance(batch, dict):
                    yield batch
                else:
                    for acc in batch:
                        yield acc
            except Exception as e:
                if os.path.exists(filename) and not isinstance(e, GeneratorExit):
                    os.unlink(filename)
                    print(f"Deleting cached {filename}")
                import traceback
                traceback.print_exc()
                print(f)
                raise
                # import sys
                # Disable for now: seems to not be working
                # sys.exit(75) # Temp failure: retry


def make_format_string(fields):
    print_fmt_hdr = "{0[0]:<12}"
    print_fmt_line = "{:<12}"

    for i in range(len(fields) - 1):
        print_fmt_hdr += " {0[" + str(i + 1) + "]:<20}"
        print_fmt_line += " {:<20}"
    return print_fmt_hdr, print_fmt_line


def pct(x, y):
    return round(x * 100.0 / y, 2)


def mb_per_sec(chunks, time_secs, sample_ratio):
    return safe_div(chunks * BlkAccess.ALIGNMENT * 100.0,
                    time_secs * 1024 * 1024 * sample_ratio)


def compress_load(filename):
    if '.shelve' in filename:
        raise NotImplementedError
        return shelve.open(filename.replace(".shelve.db", ".shelve"), flag='r')
    try:
        import compress_pickle
        from retry.api import retry_call
        return retry_call(compress_pickle.load, fargs=[filename],
                          exceptions=(PermissionError, EOFError),
                          delay=1, jitter=1, tries=3)
    except ImportError:
        if filename.endswith('.pkl.gz'):
            with gzip.GzipFile(filename, 'rb') as f:
                return pickle.load(f)
        elif filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError


def run_length_encode(ids):
    if len(ids) == max(ids)-min(ids)+1:
        return [[min(ids), max(ids)]]
    ret = [[ids[0], ids[0]]]
    for idx in ids[1:]:
        if ret[-1][1] == idx - 1:
            ret[-1][1] += 1
        else:
            ret.append([idx, idx])
    return ret


def DEBUG_FLAG():
    return "--debug" in sys.argv


# TODO: Reimplement.
def stringify_keys(d):
    """Convert a dict's keys to strings if they are not."""
    for key in list(d):

        # check inner dict
        if isinstance(d[key], dict):
            value = stringify_keys(d[key])
        else:
            value = d[key]

        # convert nonstring to string if needed
        if not isinstance(key, str):
            try:
                if str(key) in d:
                    d[str(key)] += value
                else:
                    d[str(key)] = value
            except Exception:
                try:
                    if repr(key) in d:
                        d[repr(key)] += value
                    else:
                        d[repr(key)] = value
                except Exception:
                    raise

            # delete old key
            del d[key]
    return d


def memory_usage():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 3
    except ImportError:
        return 0


# class FileLogger(object):
#     """ TODO: Deprecate."""
#     def init(self,
#              output_dir,
#              input_file_name,
#              ids=[]):
#         self.filenames = {}
#         self.handles = {}
#         for idx in ids:
#             self.filenames[idx] = f"{output_dir}/{input_file_name}_{idx}.txt"
#             self.handles[idx] = open(self.filenames[idx], "w+")
#         if self.handles:
#             os.makedirs(output_dir, 0o755, exist_ok=True)

#     def get(self, idx):
#         return self.handles.get(idx, None)

#     def write(self, idx, msg):
#         if idx in self.handles:
#             self.handles[idx].write(msg)

#     def dump(self, idx, obj):
#         if idx in self.handles:
#             self.handles[idx].write(jsonpickle.dumps(obj)+"\n")

#     def close(self):
#         for idx, hd in self.handles.items():
#             hd.close()
#             os.system(f"bzip2 -f {self.filenames['idx']}")


class Stats(object):
    def __init__(self):
        self.counters = {}  # Single values
        self.freq = {}  # dict-of-dicts, like Counter
        self.batches = {}  # dict-of-lists, like stats_batch
        # Clock based on time interval
        self.idx = 0

    def _key(self, k_):
        if isinstance(k_, list) or isinstance(k_, tuple):
            k_ = '/'.join(map(str, k_))
        return k_

    def bump(self, key, v=1, init=0):
        key = self._key(key)
        if key not in self.counters:
            self.counters[key] = init
        self.counters[key] += v

    def bump_counter(self, key, v, inc=1, init=0):
        if "--fast" in sys.argv:
            return
        key = self._key(key)
        if key not in self.freq:
            self.freq[key] = {}
        if v not in self.freq[key]:
            self.freq[key][v] = init
        self.freq[key][v] += inc

    def get(self, key, *, init=None):
        key = self._key(key)
        if key in self.counters:
            return self.counters[key]
        elif key in self.freq:
            return self.freq[key]
        elif key in self.batches:
            return self.batches[key]
        else:
            if init is None:
                init = [0] * (self.idx + 1) if key.endswith('_stats') else 0
            return init

    def append(self, key, v, init=0):
        key = self._key(key)
        if key not in self.batches:
            self.batches[key] = [init] * self.idx
        while len(self.batches[key]) < self.idx:
            self.batches[key].append(init)
        assert len(self.batches[key]) == self.idx
        self.batches[key].append(v)

    def checkpoint(self, key):
        key = self._key(key)
        # TODO: Fix so it sets at fixed index rather than appends
        self.append(key + "_stats", self.get(key))

    def checkpoint_many(self, keys):
        keys = [self._key(k) for k in keys]
        for k_ in set(flatten(fnmatch.filter(self.counters.keys(), k) for k in keys)):
            self.checkpoint(k_)

    def last_span(self, key, **kwargs):
        key = self._key(key)
        return self.span(key, i=-1 if key+"_stats" in self.batches else -2, **kwargs)

    def span(self, key, fmt=None, init=0, i=None):
        """Used before checkpoint - takes the current value"""
        key = self._key(key)
        v = self.get(key, init=init)
        if type(v) == list:
            v = v[-1]
            last_ = self.get_at(key, i=i, init=init)
        else:
            key_ = key + "_stats"
            last_ = self.get_at(key_, i=i, init=init)
        v = v - last_
        if fmt is not None:
            return key, fmt.format(v)
        return v

    def diff(self, key, fmt=None, init=0, i=None):
        """Used when you append, not checkpoint."""
        key = self._key(key)
        last_ = self.get_at(key, i=i, init=init)
        v = self.get_at(key, i=-1, init=init) - last_
        if fmt is not None:
            return key, fmt.format(v)
        return v

    def get_at(self, key, i=None, init=0):
        key = self._key(key)
        if i is None or key not in self.batches:
            return init
        try:
            return self.batches[key][i]
        except IndexError:
            return init

    def last(self, key, fmt=None, **kwargs):
        key = self._key(key)
        v = self.get_at(key, -1, **kwargs)
        if fmt is not None:
            return key, fmt.format(v)
        return key, v

    def get_all_with_prefix(self, prefix):
        return [(k, v) for k, v in self.counters.items() if k.startswith(prefix)]


ods = Stats()


def key_refmt(key):
    block_id, chunk_id = key
    return f"{block_id}|#|body-0-{chunk_id-1}"


def LOG_REQ(namespace, key, key_ts, op, result=None):
    if "--log-req" not in sys.argv:
        return
    key2 = key_refmt(key)
    log_str = f"{namespace} T= {key_ts.logical+1} {op} {key2}"
    if result:
        log_str += f" {result}"
    print(log_str)


def LOG_IOPS(ts, block_id, is_hit, chunk_hit):
    if "--log-req" not in sys.argv:
        return
    is_hit = int(is_hit)
    chunk_hit = int(chunk_hit)
    print(f"IOPS T= {ts.logical+1} GET {block_id} {chunk_hit} {is_hit}")


def LOG_DEBUG(*args, **kwargs):
    if "--debug" in sys.argv:
        print(*args, file=sys.stderr, **kwargs)


def capitalize_and_leave_existing(txt):
    return txt[0].upper() + txt[1:]


def to_camelcase(txt):
    return ''.join(capitalize_and_leave_existing(w) for w in txt.split('_'))


def rm_missing_ok(filename):
    f = Path(filename)
    if f.exists():
        f.unlink()  # missing_ok=True
        print(f"Removed file {filename}")


def touch_lockfile(lockfile_name):
    with open(lockfile_name, "w") as f:
        f.write(f"{os.getpid()} {time.time()}")


# COMMON FUNCTIONS, ALSO IN EP_UTILS
def flatten(l):
    return [item for sublist in l for item in sublist]


def safe_div(num, div, *, default=0):
    if div == 0:
        return default
    return num / div


class CopyStream(object):
    def __init__(self, stream, filename):
        self.stream = stream
        self.log = open(filename, "a")

    def write(self, msg):
        msg = str(msg)
        self.stream.write(msg)
        if "\n" in msg:
            self.stream.flush()
            token = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "> "
            msg = msg.replace("\n", "\n" + token)
        try:
            self.log.write(msg)
            self.log.flush()
        except Exception as e:
            print(f"Log write failed: {e}")

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def arg_to_dict(arg):
    """
    Command-line friendly serialization of dict-based args.

    In: arg1=val,arg2=true
    Out: {arg1: val, arg2: True}}
    """
    args = {}
    if arg:
        args = arg.split(",")
        for kv in args:
            if "=" in kv:
                k, v = kv.split("=")
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
                args[k] = v
            else:
                args[kv] = True
    return args


def closest_row(df_, col, val):
    """
    Returns row of dataframe with closest value in col.
    """

    # Don't use pd argsort as it drops NAs
    return df_.iloc[(df_[col] - val).abs().idxmin()]


def fmt_dur(dur_secs,
            smallest=None,
            small_fmt='{:.2g}',
            fmt='{:g}',
            sep=' ', sep2=' ', short=False, top_only=False, verbose=None, v=5):
    """
    verbose = 0: 1d
    verbose = 1: 1d1h1m1s
    verbose = 2: 1d 1h 1m 1s
    default:     1 days 1 hrs 1 mins 1 secs
    """
    if dur_secs < 0:
        return dur_secs
    if verbose is None:
        verbose = v
    if verbose is not None:
        if verbose == 0:
            top_only = True
        if verbose <= 1:
            sep = ''
        if verbose <= 2:
            sep2 = ''
            short = True
    label = ['secs', 'mins', 'hrs', 'days']
    divr = [60, 60, 24, dur_secs+1]
    start = 0
    if smallest:
        for i in range(len(divr)):
            if label[i].startswith(smallest):
                start = i
                break
    ret = []
    for i in range(0, start):
        dur_secs /= divr[i]
    for i in range(start, len(divr)):
        dur_secs, small = divmod(dur_secs, divr[i])
        lx = label[i]
        if small == 1:
            lx = lx[:-1]
        if short:
            lx = lx[0]
        if small > 0:
            if start == i:
                small = small_fmt.format(small)
            else:
                small = fmt.format(small)
            ret.append(f'{small}{sep2}{lx}')
        if dur_secs == 0:
            break
    ret = list(reversed(ret))
    if top_only:
        return ret[0]
    return sep.join(ret)


class LockFile(object):
    def __init__(self, filename, timeout=60*10):
        self.filename = filename
        self.f = None
        self.timeout = timeout

    def exists(self):
        return os.path.exists(self.filename)

    def stale(self):
        return self.exists() and time.time() - os.path.getmtime(self.filename) > self.timeout

    def check(self, strict=False):
        if self.exists():
            if self.stale():
                if strict:
                    print("Lockfile is stale, probably can delete")
                else:
                    print("Lockfile is stale, deleting and continuing")
                    self.delete()
                    return False
            return True
        return False

    def touch(self):
        with open(self.filename, "w") as f:
            f.write(f"{os.getpid()} {time.time()}")

    def delete(self):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.filename)

    def __del__(self):
        self.delete()
