#!/usr/bin/env python
# Note: this shebang conflicts with Debian package policy. Try changing it to the Python of your virtualenv if this causes you issues.
import argparse
import local_cluster


def main():
    parser = argparse.ArgumentParser(description="Helper to run Brooce cluster commands.\nExample usage: episodic_analysis/local_runcmd.py 'scripts/mar2023/split.sh 202110 <trace>' --queue sus --timeout 7200 --maxtries 1")
    parser.add_argument("command")
    parser.add_argument("-q", "--queue", default="par6")
    parser.add_argument("-t", "--timeout", default=3600*24, type=int)
    parser.add_argument("-m", "--maxtries", default=3, type=int)
    parser.add_argument("-n", "--no-log", action='store_true')
    parser.add_argument("--job-id")
    parser.add_argument("--dry-run", action='store_true')
    args = parser.parse_args()

    if args.job_id is None:
        args.job_id = local_cluster.generate_job_id(args.command)

    msg = f"Queueing to {args.queue} ({args.command}), Id: {args.job_id}"
    print(msg)
    kwargs = {}
    if args.no_log:
        kwargs['noredislogonsuccess'] = True
    kwargs['maxtries'] = args.maxtries
    kwargs['id'] = local_cluster.prep_jobname(args.job_id)
    kwargs['locks'] = [args.job_id]
    kwargs['killondelay'] = True
    if not args.dry_run:
        local_cluster.run_cmd(args.command, queue=args.queue, timeout=args.timeout, brooce_kwargs=kwargs)


if __name__ == '__main__':
    main()
