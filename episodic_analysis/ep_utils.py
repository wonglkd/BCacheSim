import datetime
import os
import traceback

import compress_pickle
import numpy as np
import shelve

from . import local_cluster

# compress_ext = ".lz4"
compress_ext = ".bz"
PICKLE_PROTOCOL = 4


def dump_pkl(obj, filename, overwrite=False):
    if '.shelve' in filename:
        raise NotImplementedError
        return dump_shelve(obj, filename)
    try:
        if not os.path.exists(filename) or overwrite:
            compress_pickle.dump(obj, filename, pickler_kwargs=dict(protocol=PICKLE_PROTOCOL))
        else:
            print(filename + " already exists")

        # Test load
        compress_pickle.load(filename)
    except Exception:
        os.remove(filename)
        raise


def dump_shelve(obj, filename):
    try:
        with shelve.open(filename, flag='n', protocol=PICKLE_PROTOCOL) as s:
            for k, v in obj.items():
                if not isinstance(k, str):
                    k = str(k)
                s[k] = v
    except Exception:
        os.remove(filename)
        raise


def cache_residencies(fn):
    """Decorator for caching residencies."""
    def wrapper(e_ages, **kwargs):
        dirname = local_cluster.proj_path(
            kwargs.pop('exp'), kwargs['trace_kwargs'])
        ea_filenames = [(ea, f'{dirname}/residencies_{ea:g}.pkl{compress_ext}')
                        for ea in e_ages]
        if all(os.path.exists(filename) for _, filename in ea_filenames):
            ret = {}
            try:
                for ea, filename in ea_filenames:
                    ret[ea] = compress_pickle.load(filename)
                return ret
            except Exception:
                traceback.print_exc()
                print("Skipping load")
        ret = fn(e_ages, **kwargs)
        try:
            for ea, filename in ea_filenames:
                dump_pkl(ret[ea], filename)
        except KeyboardInterrupt:
            print("Interrupted caching")
        return ret
    return wrapper


def np_safe_div(num, div):
    div = np.asarray(div, dtype=float)
    return np.divide(num, div,
                     out=np.zeros_like(div),
                     where=div != 0.)


# COMMON FUNCTIONS, ALSO IN UTILS
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
            token = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "> "
            msg = msg.replace("\n", "\n" + token)
        self.log.write(msg)
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def arg_to_dict(arg):
    """
    Command-line friendly deserialization of dict-based args.

    In: arg1=val,arg2=true
    Out: {arg1: val, arg2: True}}
    """
    args = {}
    if arg:
        line = arg.split(",")
        for kv in line:
            if "=" in kv:
                k, v = kv.split("=")
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
                args[k] = v
            else:
                args[kv] = True
    return args


def dict_to_arg(dct):
    """
    Command-line friendly serialization of dict-based args.

    In: {arg1: val, arg2: True}}
    Out: arg1=val,arg2=true
    """
    return ",".join(f"{k}={v}" for k, v in dct.items())


def make_unique(lst):
    """Removes duplicates preserving order"""
    return list(dict.fromkeys(lst))
