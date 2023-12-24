import os
import sys
import argparse
import math
import time

import pandas as pd

from . import ep_utils
from . import train_ap
from . import policies
from . import episodes
from . import train_prefetcher as tpf
from . import trace_utils
from . import local_cluster


def display_dfa(policy, cols, df_analysis):
    with pd.option_context('display.max_columns', 50,
                           'display.expand_frame_repr', False,
                           'display.width', 1000,
                           'display.max_colwidth', 30):
        fx = lambda x: (x['Target Write Rate'] == policy.target_wrs[0]) | (x['Target Cache Size'] == policy.target_cache_sizes[0])
        print("Primary targets only")
        print(df_analysis.loc[fx, cols])
        print("Others")
        print(df_analysis[cols])


def main():
    all_policies = [k for k in policies.__dict__.keys() if k.startswith('Policy') and k != 'Policy']
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--policy", default='PolicyNumAcc',
                        choices=all_policies)
    parser.add_argument("-e", "--exp", required=True)
    parser.add_argument("--region")
    parser.add_argument("--trace-group")
    parser.add_argument("--sample-ratio", default=1, type=float)
    parser.add_argument("--sample-start", default=0, type=float)
    parser.add_argument("-a", "--eviction-age", type=float)
    parser.add_argument("--analysis-eviction-ages", type=float, nargs='*')
    parser.add_argument("--ram-eviction-age", type=float)
    parser.add_argument("--residency-fn")
    parser.add_argument("--rl-filter")
    parser.add_argument("--rl-init-kwargs")
    parser.add_argument("--supplied-ea", default='physical', choices=['logical', 'physical'])
    parser.add_argument("--target-wrs", required=True, type=float, nargs='+')
    parser.add_argument("--output-base-dir", default='./')
    parser.add_argument("--output", help='For analysis only')
    parser.add_argument("--target-csizes", required=True, type=float, nargs='+')
    parser.add_argument("--not-only-gets", action='store_true')
    parser.add_argument("--ignore-existing", action='store_true')
    parser.add_argument("--no-episodes", action='store_true')
    parser.add_argument("--suffix", default='')
    parser.add_argument("--train-models", nargs='*', default=[])
    parser.add_argument("--train-split-secs-start", type=float, default=0,
                        help='Starting part of trace to use for training (seconds from start)')
    parser.add_argument("--train-split-secs-end", type=float,
                        help='Ending part of trace to use for training (seconds from start)')
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--evaluate-ap-threshold", type=float)

    tpf.add_args(parser)
    train_ap.add_args(parser)
    args = parser.parse_args()

    if args.ram_eviction_age is not None and (args.ram_eviction_age == "nan" or math.isnan(args.ram_eviction_age)):
        args.ram_eviction_age = None
    trace_kwargs = dict(region=args.region, sample_ratio=args.sample_ratio,
                        start=args.sample_start,
                        trace_group=args.trace_group,
                        only_gets=not args.not_only_gets,
                        min_ts_from_start=args.train_split_secs_start,
                        max_ts_from_start=args.train_split_secs_end)
    if args.train_target_wr:
        # Note: also fetches features. Must be before first get_accesses/policy.
        # TODO: Check to see if this really needs to be here.
        # Just for setting trace_kwargs['get_features'] - can abstract that out.
        assert args.supplied_ea == 'physical'
        pt = tpf.PrefetcherConfTrainer(
            trace_kwargs=trace_kwargs,
            e_age_s=args.eviction_age,
            wr_threshold=args.train_target_wr)
        assert trace_kwargs['get_features']
    res_fn_kwargs = {}
    if args.residency_fn is not None and args.residency_fn in episodes.__dict__ and args.residency_fn.startswith('process_'):
        res_fn_kwargs['residency_fn'] = episodes.__dict__[args.residency_fn]

    if args.sample_ratio >= 1:
        res_fn_kwargs['workers'] = 1
    else:
        res_fn_kwargs['workers'] = 8

    rl_init_kwargs = ep_utils.arg_to_dict(args.rl_init_kwargs)
    if args.rl_filter is not None:
        rl_init_kwargs['filter_'] = args.rl_filter
    print(f"rl_init_kwargs: {rl_init_kwargs}")
    policy = policies.__dict__[args.policy](
        exp=args.exp, trace_kwargs=trace_kwargs,
        target_wrs=args.target_wrs, target_cache_sizes=args.target_csizes,
        supplied_ea=args.supplied_ea,
        output_base_dir=args.output_base_dir,
        train_target_wr=args.train_target_wr,
        train_models=args.train_models,
        res_fn_kwargs=res_fn_kwargs,
        rl_init_kwargs=rl_init_kwargs,
        suffix=args.suffix)

    print(f"Sample cmd for debug: {policy.to_cmd(e_age_s=args.eviction_age, ram_ea_s=args.ram_eviction_age, e_ages=args.analysis_eviction_ages)}")

    cols = ['Assumed Eviction Age (s)', 'Target Write Rate', 'Target Cache Size',
            'Service Time Saved Ratio', 'IOPSSavedRatio',
            'Write Rate (MB/s)', 'Cache Size (GB)',
            'Mean Time In System (s)', 'Episodes admitted', 'Cutoff score', 'Target']

    if args.analysis_eviction_ages and len(args.analysis_eviction_ages) > 0:
        # To control memory consumption.
        # TODO: Look at filesize or something instead of hardcoding region
        policy.res_fn_kwargs['workers'] = 1 if args.region == "Region1" or args.sample_ratio >= 1 else 8
        print("Analysis only")
        print(f"Output to {args.output}")
        lockfile_name = args.output + ".lock"
        with open(lockfile_name, "w") as f:
            f.write(str(time.time()))

        try:
            print(f"Workers: {policy.res_fn_kwargs['workers']}")
            if os.path.exists(args.output):
                display_dfa(policy, cols, pd.read_csv(args.output).sort_values('Target'))
                print("Returning as file already exists")
                return
            df_analysis = policy.get_analysis(
                args.analysis_eviction_ages,
                ram_ea_s=args.ram_eviction_age)
            df_analysis.to_csv(args.output)
            display_dfa(policy, cols, df_analysis.sort_values('Target'))
        finally:
            os.unlink(lockfile_name)
        return
    else:
        assert args.eviction_age is not None and args.eviction_age != 0, "eviction_age required"
    filenames = policy.get_filenames(args.eviction_age)
    if os.path.exists(policy.fail_file):
        print(f"Previous failure - please delete file if incorrect: {policy.fail_file}")
        sys.exit(65)
        return

    out_prefix = policy.get_out_prefix()
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    sys.stdout = ep_utils.CopyStream(sys.stdout, out_prefix + ".out")
    sys.stderr = ep_utils.CopyStream(sys.stderr, out_prefix + ".err")
    print(f"Logging to {out_prefix}.out")
    print(f"Cmd for rerun: {local_cluster.RUNPY_LOCATION} py -B -m episodic_analysis.train {' '.join(sys.argv[1:])}")

    print('Files to generate')
    for k, v in filenames.items():
        print('{} {} {}'.format(k, v, 'Exists' if os.path.exists(v) else ''))

    if all(os.path.exists(v) for v in filenames.values()) and not args.ignore_existing:
        display_dfa(policy, cols, pd.read_csv(filenames['analysis']))
        if not args.evaluate:
            print("Returning as files already exist")
            return
        else:
            policy.df_analysis = pd.read_csv(filenames['analysis'])

    print(f"trace_kwargs: {trace_kwargs}")
    print("Trace details: Duration={duration}, Start={start_ts}, End={end_ts}, NumAccesses={logical_time}".format(**trace_utils.get_accesses_kv(**trace_kwargs)))

    if not args.train_models or not args.no_episodes:
        policy.get_all(args.eviction_age,
                       ram_ea_s=args.ram_eviction_age,
                       reset=args.ignore_existing)
        if not args.evaluate:
            display_dfa(policy, cols, policy.df_analysis)

    model_stats = []

    if args.train_target_wr:
        policy._prep_residencies([args.eviction_age])
        print(f"Episode filter: {policy.rl.filter}")
        print(f"#Episodes: {len(policy.rl.episodes)}, #Accesses: {policy.rl.num_accesses.sum()}")
        if len(policy.rl.episodes) < 100:
            policy.FAIL(f"Not enough episodes: {len(policy.rl.episodes)}")
        pt.set_policy(policy)
        if 'prefetch' in args.train_models:
            if not args.evaluate:
                pt.train_and_save_all()
            else:
                pt.load_all()
                dfe = pt.evaluate_all()
                print(dfe)
                model_stats.append(dfe)

    if 'admit' in args.train_models:
        ap_trainer = train_ap.AdmissionTrainer(
            policy=policy, prefetch_trainer=pt,
            labels=['threshold_binary'],
            **train_ap.opts_to_args(args))
        if not args.evaluate:
            ap_trainer.train_all()
            ap_trainer.save_all()
        else:
            ap_trainer.load_all()
        dfe = ap_trainer.evaluate_all()
        print(dfe)
        model_stats.append(dfe)
        if args.evaluate_ap_threshold is not None:
            dfe = ap_trainer.evaluate_all(args.evaluate_ap_threshold)
            model_stats.append(dfe)

    if args.evaluate:
        model_stats = pd.concat(model_stats)
        print(model_stats)
        model_stats_out = filenames['model_admit_threshold_binary'].replace('_admit_threshold_binary.model', f'_ap_{args.evaluate_ap_threshold:g}.models.accuracy')
        print(f"Saved to: {model_stats_out}")
        model_stats.to_csv(model_stats_out)

    print('Filenames generated:')
    for k, v in filenames.items():
        filesize = "NoExists"
        if os.path.exists(v):
            filesize = os.path.getsize(v) / 1048576
            filesize = f'{filesize:g}M'
        # elif k == 'thresholds':
        #     raise Exception("File does not exist")
        print(f'{k} {v} {filesize}')


if __name__ == '__main__':
    main()
