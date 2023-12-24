"""
Take a trace and pre-allocate files on disk.
"""
import argparse
import shutil
import os
from enum import Enum, unique
import sys
try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm import tqdm_notebook as tqdm
    if 'terminal' in ipy_str:
        from tqdm import tqdm
except:
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        def tqdm(iterable, **kwargs):
            return iterable

# perl -e '$count=8*1024*1024; while ($count>0) { print "The quick brown fox jumps over the lazy dog. "; $count-=45; }'

@unique
class OpType(Enum):
    GET_TEMP = 1
    GET_PERM = 2
    PUT_TEMP = 3
    PUT_PERM = 4
    GET_NOT_INIT = 5
    PUT_NOT_INIT = 6
    UNKNOWN = 100


GET_OPS = [OpType.GET_PERM.value, OpType.GET_TEMP.value, OpType.GET_NOT_INIT.value]


def parse_trace(location, filename):
    i = 0
    num_bytes = os.path.getsize(filename)
    with open(filename) as f, tqdm(total=num_bytes, unit_scale=True, unit='B') as pbar:
        for line in f:
            pbar.update(len(line))
            if line.startswith('#'):
                continue
            line = line.split()
            if int(line[4]) not in GET_OPS:
                continue
            newfilename = location+"run/"+line[0]
            if not os.path.exists(newfilename):
                shutil.copyfile(location+"stdfile", newfilename)
                i += 1
    print(f"{i} blocks created")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracefile")
    args = parser.parse_args()
    parse_trace("/mnt/hdd/baleen/", args.tracefile)


if __name__ == '__main__':
    main()
