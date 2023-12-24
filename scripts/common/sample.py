import argparse
import glob
import multiprocessing
import os
import random
import sys

import numpy as np

from tqdm import tqdm

import utils


tqdm_kwargs = dict(unit_scale=True, unit='B', mininterval=10, maxinterval=60)


def get_trace_settings(trace_id, trace_group, args):
    result = {}
    result['old_format'] = trace_group.endswith('201910')

    if result['old_format']:
        new_sampling = False
        key_getter = get_key_oct2019
    else:
        new_sampling = True
        key_getter = infer_header(f"{args.trace_prefix}.header")
        if trace_id == "Region4" and trace_group.endswith('202110'):
            result['sampling_rate'] = 2000
        else:
            result['sampling_rate'] = 4000

    result['num_hosts'] = None

    if new_sampling:
        if key_getter == get_key_feb2023:
            print("No of shards = No of hosts seen")
        # TODO(Release): Anonymize
        if trace_id == "Region6":
            num_hosts = 12114
        elif trace_id in ("Region4"):
            num_hosts = 1526
        elif trace_id == "Region5":
            num_hosts = 10527
        elif trace_id == "Region7":
            num_hosts = 8287
        else:
            raise NotImplementedError(f"{trace_id}")
        max_sample_end = num_hosts / result['sampling_rate']
        print(f"Max sampling: {max_sample_end:g}, No of Hosts: {num_hosts}")
        if max_sample_end < 1:
            print("Warning: Max Sample < 1")
        assert args.end < max_sample_end

        print(f"Will downsample to get one host worth: {result['sampling_rate']} / {num_hosts} = {result['sampling_rate'] / num_hosts:3f}")
        result['num_hosts'] = num_hosts

    result['key_getter'] = key_getter
    result['new_sampling'] = new_sampling
    return result


def get_key_oct2019(line_):
    # block_id
    return line_[0]


def get_key_oct2021(line_):
    # block_id,host_id
    return line_[0]+','+line_[7]


def get_key_feb2023(line_):
    """
    block_id,rs_shard_id

    While this is the same as get_key_oct2021 for now, we compare the function
    to determine other things like counting hosts/shards
    """
    return line_[0]+','+line_[7]


class SamplerSetup(object):
    def __init__(self, trace_id, trace_group, args):
        self.trace = trace_id
        self.group = trace_group
        self.args = args
        for k, v in get_trace_settings(trace_id, trace_group, args).items():
            setattr(self, k, v)

    def load_keys(self, key_file):
        blocks = []
        num_accesses = []
        num_accesses_get = []
        hosts = set()

        num_bytes = os.path.getsize(key_file)
        with open(key_file) as f, tqdm(total=num_bytes, **tqdm_kwargs) as pbar:
            for line in f:
                items = line.split()
                # key, #Accs, #GETs, #PUTs, #Bytes
                key, num_accs, num_accs_get = items[:3]
                if not self.old_format:
                    # For vll1, host_id is actually rs_shard_id
                    key_id, host_id = key.split(",")
                    hosts.add(host_id)
                blocks.append(key)
                num_accesses.append(int(num_accs))
                num_accesses_get.append(int(num_accs_get))
                pbar.update(len(line))

        print(f"2/Memory usage: {utils.memory_usage():.1f} GB")
        if len(hosts) < 100:
            print(f"Hosts: {sorted(map(int, hosts))}")
        self.num_hosts_seen = len(hosts)
        if self.num_hosts is None:
            self.num_hosts = len(hosts)
        self.blocks = blocks
        self.num_accesses = num_accesses
        self.num_accesses_get = num_accesses_get

    def get_allowed_blocks(self, out_keys_filename, legacy=True):
        args = self.args
        # TODO: There is probably a more time and memory efficient way involving a hash
        # function and just calculating the total. And having a shuffle happen
        # on disk, once, according to hash value, rather than everytime we
        # sample. What we would be doing is to cache results of first half of
        # this function on disk.
        if legacy:
            idx = list(range(len(self.blocks)))
            random.seed(args.seed)
            random.shuffle(idx)
            acc_x_gets = np.array([self.num_accesses_get[x] for x in idx])
            acc_x_gets = acc_x_gets.cumsum()
            acc_x_total = np.array([self.num_accesses[x] for x in idx]).cumsum()
        else:
            self.num_accesses = np.array(self.num_accesses)
            self.num_accesses_get = np.array(self.num_accesses_get)
            idx = np.arange(len(self.blocks))
            np.random.seed(args.seed)
            np.random.shuffle(idx)
            acc_x_gets = self.num_accesses_get[idx].cumsum()
            acc_x_total = self.num_accesses[idx].cumsum()
        one_host_gets = acc_x_gets[-1]
        one_host_accs = acc_x_total[-1]
        if self.new_sampling:
            factor = self.sampling_rate / self.num_hosts
            one_host_gets *= factor
            one_host_accs *= factor
        if args.orig_rate and args.orig_rate != 1:
            print("Downsampling by", args.orig_rate)
            one_host_gets /= args.orig_rate
            one_host_accs /= args.orig_rate

        if args.weight == 'gets':
            found = np.searchsorted(acc_x_gets / one_host_gets, [args.start, args.end])
        elif args.weight == 'accs':
            found = np.searchsorted(acc_x_total / one_host_accs, [args.start, args.end])
        elif args.weight == 'equal':
            raise NotImplementedError
            # found = np.searchsorted(acc_x_total / one_host_accs, [args.start, args.end])
        else:
            raise NotImplementedError

        found[1] += 1
        print("Indicies:", found)
        num_items = found[1] - found[0]
        print("Items selected: {} ({:g})".format(num_items, num_items / len(self.blocks)))
        print("Total items:", len(self.blocks))
        acc_sampled = acc_x_gets[found][1] - acc_x_gets[found][0]
        acc_sampled_all = acc_x_total[found][1] - acc_x_total[found][0]
        print("GET Accesses of items selected: {} ({:g} {:g})".format(acc_sampled, acc_sampled / one_host_gets, acc_sampled / acc_x_gets[-1]))
        print("Total GET accesses: {} (1 host: {:.1f})".format(acc_x_gets[-1], one_host_gets))
        print("Accesses of items selected: {} ({:g} {:g})".format(acc_sampled_all, acc_sampled_all / one_host_accs, acc_sampled_all / acc_x_total[-1]))
        print("Total accesses: {} (1 host: {:.1f})".format(acc_x_total[-1], one_host_accs))

        # frozenset used for quick membership test.
        allowed_blocks = frozenset(self.blocks[idx[i]] for i in range(found[0], found[1]))

        assert len(allowed_blocks) == num_items, (len(allowed_blocks), num_items)

        print("Saving keys")
        with open(out_keys_filename, 'w') as f:
            for i in range(found[0], found[1]):
                f.write(f"{self.blocks[idx[i]]} {self.num_accesses[idx[i]]} {self.num_accesses_get[idx[i]]}\n")

        return allowed_blocks


def process_file(args):
    i, file_in, file_out, allowed_blocks, key_getter = args
    num_bytes = os.path.getsize(file_in)
    print(f"Memory usage: {utils.memory_usage():.1f} GB")
    # i if sys.stdout.isatty() else None
    with open(file_in) as f, open(file_out, 'w') as fw, tqdm(total=num_bytes, position=None, desc=os.path.basename(file_in), **tqdm_kwargs) as pbar:
        for line in f:
            if line.startswith('#'):
                continue
            line_ = line.split()
            key = key_getter(line_)

            if key in allowed_blocks:
                fw.write(line)
            pbar.update(len(line))


def infer_header(filename):
    # Comment followed by space separated list
    with open(filename) as f:
        return get_key_feb2023 if 'rs_shard_id' in f.readline() else get_key_oct2021


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_prefix")
    parser.add_argument("-s", "--start", type=float, default=0)
    parser.add_argument("-l", "--size", type=float)
    parser.add_argument("-e", "--end", type=float)
    parser.add_argument("--old-format", action='store_true')
    parser.add_argument("--seed", default=1337)
    parser.add_argument("--orig-rate", default=1, type=float, help='Set to no of hosts (in volume) in original trace')
    parser.add_argument("-t", "--trace-id", required=True)
    parser.add_argument("-g", "--trace-group", required=True)
    parser.add_argument("--weight", choices=["gets", "accs"], default="accs")
    parser.add_argument("-j", "--num-workers", default=None, type=int,
                        help='Be careful on large traces to avoid OOM')
    args = parser.parse_args()

    trace_id = args.trace_id
    if "/" in trace_id:
        trace_id = trace_id.split("/")[0]
    print(f"Trace ID: {trace_id}")

    key_file = f"{args.trace_prefix}.keys"

    if args.num_workers is None:
        args.num_workers = 8
        # if trace_id == 'vll1':
        #     args.num_workers = 6
        if trace_id == 'vll3' and '/raw/' in args.trace_prefix:  # No raw-split
            args.num_workers = 4

    assert args.size is not None or args.end is not None
    if args.end:
        args.size = args.end - args.starts
    else:
        args.end = args.start + args.size

    out_prefix = f"{args.trace_prefix}_{args.start*100:g}_{args.size*100:g}"
    output_file_prefix = out_prefix + ".trace"
    sampled_keys = out_prefix + ".keys"

    if os.path.exists(output_file_prefix):
        print("File already exists:", output_file_prefix)
        return

    lock = utils.LockFile(output_file_prefix + ".lock", timeout=60*5)
    if lock.check():
        raise Exception(f"Lock file exists - remove it first: {output_file_prefix}.lock")
        return
    lock.touch()

    print("Logging stdout to:", out_prefix + ".log")
    sys.stdout = utils.CopyStream(sys.stdout, out_prefix + ".log")

    print(f"1/Memory usage: {utils.memory_usage():.1f} GB")
    ssetup = SamplerSetup(trace_id, args.trace_group, args)

    print("OUT Key file:", sampled_keys)
    print("IN Key file:", key_file)
    print("OUT File:", output_file_prefix)
    print("IN Trace files:")
    if ssetup.old_format:
        trace_files = [f"{args.trace_prefix}.trace"]
        print(trace_files)
    else:
        trace_files = sorted(glob.glob(f"{args.trace_prefix}.trace.*"), key=lambda x: int(x.split(".")[-1]))
        for i, filename in enumerate(trace_files):
            print(i, filename)
            assert i == int(filename.split(".")[-1]), (i, filename)

    ssetup.load_keys(key_file)
    print(f"3/Memory usage: {utils.memory_usage():.1f} GB")
    allowed_blocks = ssetup.get_allowed_blocks(sampled_keys, legacy=False)
    print(f"4/Memory usage: {utils.memory_usage():.1f} GB")
    key_getter = ssetup.key_getter
    del ssetup
    import gc
    gc.collect()
    print(f"5/Memory usage: {utils.memory_usage():.1f} GB")

    # allowed_blocks = frozenset(allowed_blocks_)
    # if len(frozenset(allowed_blocks)) != num_items:
    #     present = set()
    #     for k in allowed_blocks:
    #         if k in present:
    #             assert False, k
    #         present.add(k)
    # allowed_blocks = frozenset(allowed_blocks)
    # TODO: Debug this. allowed_blocks = num_items - 1.
    # This check is now also done in reformat, by making sure it in sorted order.
    tmpdir = output_file_prefix + "_tmpdir"
    cmd = f"sort -k3,3 -r -s -n {sampled_keys} -o {sampled_keys}"
    try:
        utils.check_cmd(cmd)
    except Exception as e:
        print(e)
        cmd += f' -T {tmpdir}'
        utils.cmd_with_tmpdir(cmd, tmpdir)

    pargs = [(i, fn, output_file_prefix+f".{i}", allowed_blocks, key_getter)
             for i, fn in enumerate(trace_files)]
    if args.num_workers == 1:
        # TODO: Deprecate?
        for zargs in pargs:
            process_file(zargs)
    else:
        try:
            # TODO: Sometimes it gets stuck here, but works on a retry
            # , maxtasksperchild=1
            with multiprocessing.Pool(processes=args.num_workers) as pool:
                pool.map(process_file, pargs)
        except (MemoryError, OSError, KeyboardInterrupt):
            lock.delete()
            raise
    out_files = [x[2] for x in pargs]

    print("Merging files")
    # Sort by timestamp. -m assumes files are sorted.
    utils.cmd_with_tmpdir(f"sort -m -s -n -k 4,4 -T {tmpdir} -o {output_file_prefix}.tmp " + ' '.join(out_files), tmpdir)
    assert os.path.getsize(output_file_prefix+".tmp") > 0
    utils.check_cmd(f"cat {args.trace_prefix}.header {output_file_prefix}.tmp > {output_file_prefix}.tmp2")
    os.rename(f"{output_file_prefix}.tmp2", output_file_prefix)
    os.unlink(f'{output_file_prefix}.tmp')
    print("DONE:", output_file_prefix)
    lock.delete()


if __name__ == '__main__':
    main()
