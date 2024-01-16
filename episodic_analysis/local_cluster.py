import json
import logging
import os
import hashlib
import pathlib
import time
import warnings

from collections import defaultdict
from collections import Counter

import redis

try:
    from .constants_meta import infer_trace_group
except (ImportError, ModuleNotFoundError):
    from .constants_public import infer_trace_group


def load_dotenv():
    filename = os.path.dirname(__file__) + "/.env"
    if not os.path.exists(filename):
        return {}
    with open(filename) as f:
        return dict([kv.split("=") for kv in f.read().strip().split("\n")])


ENV_CFG = load_dotenv()
SIM_LOCATION = ENV_CFG.get('SIM_LOCATION', 'bcachesim')
TMP_LOCATION = ENV_CFG.get('TMP_LOCATION', 'tmp')
TRACE_LOCATION = ENV_CFG.get('TRACE_LOCATION', 'data/tectonic')
OUTPUT_LOCATION = ENV_CFG.get('OUTPUT_LOCATION', 'runs')
RUNPY_LOCATION = f'{SIM_LOCATION}run_py.sh'


def get_redis_client(REDIS_HOST=None,
                     REDIS_PASSWORD=None,
                     REDIS_PORT=6379,
                     db=0,
                     **kwargs):
    dotenv = load_dotenv()
    for k, v in locals().items():
        if k.startswith('REDIS_'):
            newk = k.replace('REDIS_', '').lower()
            kwargs[newk] = v
            if v is None and k in dotenv:
                kwargs[newk] = dotenv[k]
    return redis.StrictRedis(db=db, **kwargs)


DEFAULT_CLIENT = None


def get_default_client():
    global DEFAULT_CLIENT
    if DEFAULT_CLIENT is None:
        DEFAULT_CLIENT = get_redis_client()
    return DEFAULT_CLIENT


def run(script, args, timeout=3600 * 24 * 2, top=False, queue='common',
        venv='cachelib-py38',
        generate_cmd=False,
        brooce_kwargs=dict(),
        path=SIM_LOCATION,
        noredislogonsuccess=False,
        dry_run=False):
    if 'id' in brooce_kwargs:
        brooce_kwargs['id'] = prep_jobname(brooce_kwargs['id'])
    if noredislogonsuccess:
        brooce_kwargs['noredislogonsuccess'] = True

    if venv == 'cachelib-py38':
        pyenv = 'py'
    elif venv == 'cachelib-pypy3-7.3.1':
        pyenv = 'pypy'
    else:
        raise NotImplementedError(venv)

    cmd = f'{RUNPY_LOCATION} {pyenv} {script} {args}'
    if generate_cmd:
        return cmd
    elif dry_run:
        print(cmd)
        return
    ret = run_cmd(cmd, queue, timeout, brooce_kwargs, top=top)
    return ret, cmd


def truncate_middle(s, maxlen=60):
    if len(s) < maxlen:
        return s
    else:
        return s[:30] + '...' + s[-25:]


def generate_job_id(cmd):
    return "job_" + truncate_middle(prep_jobname(cmd).strip('.')) + "_" + hashlib.md5(cmd.encode("utf-8")).hexdigest()[-6:]


def run_cmd(cmd, queue='par6', timeout=3600*24*7, brooce_kwargs=None,
            top=False, dry_run=False,
            client=None):
    """Most functions should use launch_cmd instead."""
    client = client or get_default_client()
    brooce_kwargs = brooce_kwargs or {}
    if 'id' in brooce_kwargs:
        brooce_kwargs['id'] = prep_jobname(brooce_kwargs['id'])
    payload = {**brooce_kwargs, **{'command': cmd, 'timeout': timeout, 'queue_time': int(time.time())}}
    payload = json.dumps(payload)
    # from retry.api import retry_call
    fn_call = client.rpush if top else client.lpush
    # ret = retry_call(fn_call,
    #                  fargs=[f'brooce:queue:{queue}:pending', payload],
    #                  exceptions=redis.exceptions.ConnectionError,
    #                  delay=2, backoff=2, jitter=1, tries=5)
    if dry_run:
        return cmd, payload
    return fn_call(f'brooce:queue:{queue}:pending', payload)


def launch_cmd(cmd, timeout=3600*24, job_id=None, maxtries=1, **kwargs):
    if job_id is None:
        job_id = generate_job_id(cmd)
    brooce_kwargs = {
        'id': job_id,
        'noredislogonsuccess': False,
        'killondelay': True,
        'locks': [job_id],
        'maxtries': maxtries,
    }
    return run_cmd(cmd, timeout=timeout, brooce_kwargs=brooce_kwargs, **kwargs)


jobstatus = {}
# TODO: Break down by queue
jobs_by_status = defaultdict(list)
job_objs_by_status = defaultdict(list)


def get_job_status(client=None):
    client = client or get_default_client()
    jobstatus_ = {}
    jobs_by_status_ = defaultdict(list)
    job_objs_by_status_ = defaultdict(list)
    for qkey in client.scan_iter("brooce:queue:*"):
        q = qkey.split(b":")
        qname, status = q[2:4]
        status = status.decode('ascii')
        if status == 'working':
            machine_q = q[4]
        if status == 'done':
            continue
        qlen = client.llen(qkey)
        if qlen == 0:
            continue
        # print(qname, status)
        for job in client.lrange(qkey, 0, -1):
            if job and job.startswith(b'{'):
                try:
                    job = json.loads(job)
                except json.JSONDecodeError:
                    print(job)
                    raise
                job_id = job['id']
            else:
                job_id = job
            jobstatus_[job_id] = status
            jobs_by_status_[status].append(job_id)
            job_objs_by_status_[status].append(job)
    return jobstatus_, jobs_by_status_, job_objs_by_status_


def getjobstatus(**kwargs):
    global jobstatus, jobs_by_status, job_objs_by_status
    jobstatus_, jobs_by_status_, job_objs_by_status_ = get_job_status(**kwargs)
    jobs_by_status.clear()
    jobs_by_status.update(jobs_by_status_)
    job_objs_by_status.clear()
    job_objs_by_status.update(job_objs_by_status_)
    jobstatus.clear()
    jobstatus.update(jobstatus_)
    deduplicate()


def deduplicate(queues=None, client=None, dry_run=False):
    """TODO: Port this into Go and make it part of prune.go"""
    client = client or get_default_client()
    if queues:
        if type(queues) == str:
            queues = [queues]
        queues = ['brooce:queue:{}:pending'.format(q) for q in queues]
    else:
        queues = client.scan_iter("brooce:queue:*:pending")
    for qkey in queues:
        jobs = client.lrange(qkey, 0, -1)
        # done = set(client.lrange(qkey.replace("pending", "done"), 0, -1))
        # working = local_cluster.DEFAULT_CLIENT.lrange(qkey.replace("pending", "working"), 0, -1)

        jobs_by_id = defaultdict(list)
        for job in jobs:
            if job and job.startswith(b'{'):
                try:
                    jobo = json.loads(job)
                except json.JSONDecodeError:
                    print(job)
                    raise
                jobs_by_id[jobo['id']].append(job)

        dup_removed = 0
        for job_id, jobs_ in jobs_by_id.items():
            if dry_run and len(jobs_) > 1:
                print(job_id, len(jobs_)-1)
            # TODO: Sort by queue time
            for extra_job in jobs_[:-1]:
                print(extra_job, len(jobs_), len(jobs_[:-1]))
                dup_removed += 1
                # Negative so that things at head (newest) are removed first.
                if not dry_run:
                    client.lrem(qkey, 1, extra_job)
        if dup_removed:
            print(f"Removed {dup_removed} duplicates from {qkey}")


def fmt_sample_ratio(sample_ratio):
    return f"{sample_ratio:g}"


def fmt_trace_id(*, sample_ratio=None, region=None, start=0, trace_group=None, **kwargs):
    trace_id = f"{region}_{start:g}_{fmt_sample_ratio(sample_ratio)}"
    if trace_group is None:
        # Temporary, to allow existing experiments to continue
        warnings.warn("Using old trace_id when trace_group=None")
        return trace_id
    trace_group = infer_trace_group(region) if trace_group is None else trace_group
    trace_group_ = trace_group.replace("/", ".")
    return f"{trace_group_}_" + trace_id


def fmt_f(val):
    if type(val) == float:
        return f'{val:g}'
    return val


def exp_prefix(name, trace_kwargs, csize_gb, target_wr, ram_csize_gb=None, suffix=""):
    csize_gb_f = fmt_f(csize_gb)

    if name > "20230325":
        prefix = f'{name}/{fmt_trace_id(**trace_kwargs)}_{csize_gb_f}GB'
    else:
        print("Warning: using old format")
        prefix = f'{name}/{fmt_trace_id(**{**trace_kwargs, **dict(trace_group=None)})}_{csize_gb_f}GB'
    if ram_csize_gb is not None:
        prefix += f"_RAM-{ram_csize_gb:g}GB"
    prefix += f'_WR{fmt_f(target_wr)}MBS'
    prefix += suffix
    return prefix


def fmt_subtrace(subtrace="full", start=0, sample_ratio=None):
    return f"{subtrace}_{start:g}_{fmt_sample_ratio(sample_ratio)}"


def tracefilename(sample_ratio=None, region=None, start=0, trace_group=None, *, subtrace="full", not_exists_ok=False, trace_location=None):
    if trace_group is None:
        trace_group = infer_trace_group(region)
    if trace_location is None:
        trace_location = TRACE_LOCATION
    filename1 = f'{trace_location}/ws/{trace_group}/processed/{region}/{fmt_subtrace(subtrace, start, sample_ratio)}.trace'
    filename2 = f'{trace_location}/ws/{trace_group}/{region}/processed/{fmt_subtrace(subtrace, start, sample_ratio)}.trace'
    filename3 = f'{trace_location}/{trace_group}/{region}/{fmt_subtrace(subtrace, start, sample_ratio)}.trace'
    filename4 = '../' + filename3
    if os.path.exists(filename1):
        return filename1
    if os.path.exists(filename2):
        return filename2
    if os.path.exists(filename4):
        return filename4
    if not os.path.exists(filename3) and not not_exists_ok:
        print(f"Warning: {filename3} does not exist")
    return filename3


def prep_jobname(name):
    return name.replace("/", ".").replace(" ", "_")


def proj_path(exp, trace_kwargs):
    proj_path = TMP_LOCATION
    dirname = f'{proj_path}/{exp}/{fmt_trace_id(**trace_kwargs)}'
    os.makedirs(dirname, exist_ok=True)
    return dirname


def format_message(record):
    try:
        record_message = '%s' % (record.msg % record.args)
    except TypeError:
        record_message = record.msg
    if record_message.count("\n") > 1 and not record_message.startswith("\n"):
        record_message = "\n" + record_message
    return record_message


class GlogFormatter(logging.Formatter):
    LEVEL_MAP = {
        logging.FATAL: 'F',  # FATAL is alias of CRITICAL
        logging.ERROR: 'E',
        logging.WARN: 'W',
        logging.INFO: 'I',
        logging.DEBUG: 'D'
    }

    def __init__(self):
        logging.Formatter.__init__(self)

    def format(self, record):
        try:
            level = GlogFormatter.LEVEL_MAP[record.levelno]
        except KeyError:
            level = '?'
        date = time.localtime(record.created)
        record_message = '%c%02d%02d %02d:%02d:%02d] %s' % (
            level, date.tm_mon, date.tm_mday, date.tm_hour, date.tm_min,
            date.tm_sec,
            format_message(record))
        record.getMessage = lambda: record_message
        return logging.Formatter.format(self, record)


def generate_sample(region, sample_ratio, subtrace="full", trace_group=None, start=0):
    trace_group = infer_trace_group(region) if trace_group is None else trace_group

    args = ' --old-format ' if trace_group.endswith('201910') else ''
    if region == 'combined_cl_sc':
        args += ' --orig-rate 2 '
    cmd = f"{SIM_LOCATION}/scripts/mar2023/sample_trace.sh {trace_group} {region} {subtrace} -s {start / 100:g} -l {sample_ratio / 100} {args} # rate={sample_ratio:g}, start={start:g}"
    job_id = f"sample__{trace_group}_{region}_{subtrace}_{start:g}_{sample_ratio}"
    filename = tracefilename(sample_ratio, region, start=start, trace_group=trace_group, subtrace=subtrace, not_exists_ok=True)
    path_trace = pathlib.Path(filename).parent.joinpath(f'{subtrace}.keys')
    if not path_trace.exists():
        print(f"Error! Trace not processed yet: {path_trace}")
    return cmd, job_id, filename


def wait_for(files, sleep_interval=60, print_interval=15):
    start_time = time.time()
    waited = False
    since_last_print = 0
    while True:
        num_left = sum(1 for file in files if not os.path.exists(file))
        if num_left == 0:
            if waited:
                print("Done")
            return
        waited = True
        if since_last_print % print_interval == 0:
            print(f"{num_left} of {len(files)} left: {(time.time() - start_time)//60} mins elapsed")
        time.sleep(sleep_interval)
        since_last_print += 1


def ensure_samples(params,
                   only_wait=False,
                   queue='par2', timeout=3600*2, dry_run=False, **kwargs):
    cmds = []
    filenames = []
    for trace_group, region, sample_ratio, start in params['region_srs']:
        cmd, job_id, filename = generate_sample(region, sample_ratio, start=start, trace_group=trace_group)
        if not os.path.exists(filename):
            cmds.append([job_id, cmd])
            filenames.append(filename)

    if dry_run:
        print(cmds)
        return

    if not only_wait:
        for job_id, cmd in cmds:
            launch_cmd(cmd, queue=queue, timeout=timeout, job_id=job_id)

    wait_for(filenames, **kwargs)
