#!/usr/bin/env python3
# from argparse import ArgumentParser
from jsonargparse import ArgumentParser, ActionConfigFile, ActionYesNo


from . import sim_cache


def get_parser():
    parser = ArgumentParser(description="CacheLib simulator")
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_argument("tracefile", nargs='?')  # TODO(230316): Deprecate after current runs.
    parser.add_argument("-t", "--trace")
    parser.add_argument("--ram-cache", action="store_true",
                        help="Simulate RAM Cache")
    parser.add_argument("--ram-cache-elems", type=int,
                        help="RAM Cache size as no of elements")
    parser.add_argument("--ram-cache-size_gb", type=float, default=10,
                        help="RAM Cache size in GB")
    parser.add_argument("--eviction-policy", help="Select eviction policy")
    parser.add_argument("--ttl-model-path")
    # TODO: Deprecate --fifo, --lirs
    parser.add_argument("--fifo", action="store_true",
                        help="Simulate fifo")
    parser.add_argument("--lirs", action="store_true",
                        help="Enable lirs instead of LRU/FIFO")
    parser.add_argument("--ap", help="Select admission policy")
    parser.add_argument("--optplus-args", help="OfflinePlus args")
    parser.add_argument("--rejectx-ap", action="store_true",
                        help="Simulate reject x admission policy")
    parser.add_argument("--learned-ap", action="store_true",
                        help="Simulate with a learned admission policy")
    parser.add_argument("--ram-ap-clone", action="store_true",
                        help="Have same RAM AP as flash")
    parser.add_argument("--batch-size", default=512, type=int,
                        help="Batchsize for GBM")
    parser.add_argument("--offline-ap",
                        help="Simulate with a offline admission policy",
                        action="store_true")
    parser.add_argument("--coinflip-ap",
                        help="Simulate with a coin flip admission policy",
                        action="store_true")
    parser.add_argument("--ap-threshold",
                        help="Set the admission policy's threshold",
                        type=float)
    parser.add_argument("--ap-probability",
                        help="Set the admission probability",
                        type=float)
    parser.add_argument("--ap-feat-subset",
                        help="Subset of features for admission plicy")
    parser.add_argument("--ap-chunk-threshold",
                        help="Set the admission policy's threshold",
                        type=int)
    parser.add_argument("--learned-ap-model-path",
                        help="Set the file that stores the learned admission policy's model")
    parser.add_argument("--learn-ap-filtercount", dest='learned_ap_filter_count',
                        help="Set the number of bloom filters for history features",
                        type=int,
                        default=6)
    parser.add_argument("--learn-ap-granularity", dest='learned_ap_granularity',
                        help="Block (for prefetching) or chunk level")
    parser.add_argument("--learned-size",
                        help="Size aware",
                        action="store_true")
    parser.add_argument("--size-opt",
                        help="Size aware",
                        default="access")
    parser.add_argument("--peak-strategy")
    parser.add_argument("--block-level",
                        help="simulate at block level",
                        action="store_true")
    parser.add_argument("--hybrid-ap-threshold",
                        help="Threshold to decide between OPT and ML",
                        type=float)
    parser.add_argument("--opt-ap-threshold",
                        help="OPT threshold for hybrid",
                        type=float)
    parser.add_argument("--rejectx-ap-threshold",
                        help="Set RejectX threshold",
                        type=float)
    parser.add_argument("--rejectx-ap-factor",
                        help="Set RejectX factor (of cache size history to keep)",
                        type=float)
    parser.add_argument("--flashieldprob-ap-min-hits",
                        help="Flashield: Min No of DRAM hits",
                        type=int)
    parser.add_argument("--retrain-interval-hrs",
                        help="ML online",
                        type=float)
    parser.add_argument("--train-history-hrs",
                        help="ML online",
                        type=float)
    parser.add_argument("--offline-ap-decisions",
                        help="Set the file that stores the offline admission policy's decisions")
    # jsonargparse extension: for argparse in Py3.9, use BooleanOptionalAction.
    parser.add_argument("--flip-threshold", action=ActionYesNo(no_prefix='no-'),
                        default=True,
                        help="Do not flip threshold")
    parser.add_argument("--evict-by-episode",
                        help="Evict by episode",
                        action="store_true")
    parser.add_argument("--prefetch-when",
                        help="never or always or rejectfirst",
                        default="never")
    parser.add_argument("--prefetch-when-threshold",
                        help="Prefetch When Threshold (for ML)",
                        type=float)
    parser.add_argument("--prefetch-range",
                        default="episode",
                        help="episode or all")
    parser.add_argument("--prefetcher-model-path",
                        help="Set the file that stores the prefetch policy's model")
    parser.add_argument("--early-evict",
                        help="Early eviction decisions")
    parser.add_argument("--prefetch",
                        help="Prefetch (early admission)")
    parser.add_argument("--admit-only-prefetches", action="store_true",
                        help="Admit only prefetches")
    parser.add_argument("-o",
                        "--output-dir",
                        help="Destination directory for results")
    parser.add_argument("--override",
                        help="Overwrite existing solution if it exists, otherwise exit by default",
                        action="store_true")
    parser.add_argument("-w", "--write_mbps", default=0,
                        help="Expected write rate MB/sec",)
    parser.add_argument("-s", "--size_gb", type=float, default=400,
                        help="Cache size in GB")
    parser.add_argument("--cache-elems", type=int,
                        help="Cache size as no of elements")
    parser.add_argument("--debug", action="store_true",
                        help="Debugging")
    parser.add_argument("--profile", action="store_true",
                        help="Profiling")
    # TODO: Refactor so that functions don't check sys.argv directly.
    parser.add_argument("--one-chunk", action="store_true",
                        help="Debugging: one chunk")
    parser.add_argument("--log-req", action="store_true",
                        help="Log requests")
    parser.add_argument("--log-prefetch", action="store_true",
                        help="Log prefetchs")
    parser.add_argument("--fast", action="store_true",
                        help="Fast (skips things)")

    parser.add_argument("--limit", type=float,
                        help="Process at most this fraction of total IOPS")
    parser.add_argument("--ignore-existing",
                        help="Ignore existing results",
                        action="store_true")
    parser.add_argument("--job-id",
                        help="Job ID for tracking")
    parser.add_argument("--cachelib-trace")
    parser.add_argument("--ep-analysis", help="Episodic Analysis")
    parser.add_argument("--log-interval", default=3600.0, type=float,
                        help="Used for determining Peak ST, etc (in trace seconds)")
    parser.add_argument("--stats-start", default=24*3600, type=float,
                        help="When to start stats")
    # TODO: Deprecate.
    parser.add_argument("--log-decisions",
                        help="Log each admission decision. Warning: large file.",
                        action="store_true")
    parser.add_argument("--log-evictions",
                        help="Log each eviction. Warning: VERY large file.",
                        action="store_true")
    parser.add_argument("--log-episodes",
                        help="Log stats for each episode. Warning: VERY VERY large file.",
                        action="store_true")
    return parser


def get_parsed_args():
    args = get_parser().parse_args()
    assert args.trace or args.tracefile
    if args.trace:
        args.tracefile = args.trace
    if args.config:
        args.config = [str(x) for x in args.config]
    assert not (args.fifo and args.lirs), "Cannot run with both fifo and lirs"
    return args


if __name__ == "__main__":
    sim_cache.simulate_cache_driver(get_parsed_args())
