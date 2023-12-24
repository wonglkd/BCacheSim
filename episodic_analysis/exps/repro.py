import os


from . import factory
from . import results
from .. import adaptors
from .. import local_cluster
from .base import ExpBase


class ReproExpFactory(factory.ExpFactoryBase):
    """
    - Feed in:
         - Thresholds (ML, RejectX, CoinFlip), Assumed EA
         - Trained ML model
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_base_dir = factory.OUTPUT_BASE_DIR + self.group
        self.rows = []

    def add_row(self, row, **kwargs):
        row = row.copy()
        for k, v in kwargs.items():
            row['Orig'+k] = row[k]
            row[k] = v
        self.rows.append(row)

    def all(self, on):
        assert on in ['cachebench', 'sim']
        exps = []
        for row in self.rows:
            if on == 'cachebench':
                exps.append(ExpReproCacheBench(row, name=self.name, output_base_dir=self.output_base_dir))
            elif on == 'sim':
                exps.append(ExpReproSim(row, name=self.name, output_base_dir=self.output_base_dir))
        return exps

    def update(self, all_exps, **kwargs):
        all_exps[self.name] = self.all(**kwargs)
        return self


class OneShotExp(ExpBase):
    """
    Unlike Exp, does not search for any thresholds or EA.
    Like Exp, it launches, waits and checks for result, postprocesses result.

    Can be used to train on one thing and test on another.

    Possible use cases:
    - Run Testbed to reproduce a simulation run
    - Run Simulation on a different SampleStart, or even SampleRatio
    - Run Baleen*
    """

    def __init__(self, orig, *, name='UNDEFINED', output_base_dir=None, **kwargs):
        self.orig = orig
        self.output_base_dir = output_base_dir
        self.trace_kwargs = dict(trace_group=orig['TraceGroup'], region=orig['Region'], sample_ratio=orig['SampleRatio'], start=orig['SampleStart'])
        self.suffix = factory.make_suffix(dict(prefetch_combo=(orig["Prefetch-When"], orig["Prefetch-Range"]), ap=orig["AdmissionPolicy"]))
        self.prefix = local_cluster.exp_prefix(name, self.trace_kwargs, orig['Cache Size (GB)'], orig['Target Write Rate'], suffix=self.suffix)
        self.dir = os.path.join(output_base_dir, self.prefix)
        self.name = name
        self.state = {'stage': 'init', 'cmds': [], 'outputs': []}
        self.prereqs = []
        self.curr_outputs = []

    def working(self):
        return any(self.ready_status())  # TODO: or queueing, in Brooce

    def prepare_ready(self):
        return all(self.prepare_status())

    def prepare_stale(self):
        return False

    def run_stale(self):
        return False

    @property
    def curr_iter(self):
        return 0

    def prepare_status(self):
        return [os.path.exists(f) for f in self.prereqs]

    def run(self, relaunch_stale=False, **kwargs):
        if self.stage == 'init':
            self.stage = 'prepare'
            return
        elif self.stage == 'prepare':
            self.prepare()
            self.stage = 'prepare-wait'
            if self.prepare_ready():
                self.stage = 'launch'
        elif self.stage == 'prepare-wait':
            if self.prepare_ready():
                self.stage = 'launch'
        elif self.stage == 'launch':
            if not self.ready() and not self.working():
                self.launch()
            self.stage = 'queue'
        elif self.stage == 'queue':
            if self.working():
                self.stage = 'wait'
        elif self.stage == 'wait':
            if self.ready():
                self.stage = 'process'
        elif self.stage == 'process':
            self.process()
            self.stage = 'complete'
        elif self.stage in ('complete', 'failure'):
            pass
        else:
            raise NotImplementedError(self.stage)

    def __repr__(self):
        #         add_desc = ''
        # try:
        #     if self.stage == 'wait':
        #         add_desc += f'({self.complete}/{len(self.curr_outputs)})'
        #     if self.curr_thresholds:
        #         add_desc += f", #th={len(self.curr_thresholds)}"
        #     if self.curr_fitted_threshold is not None:
        #         add_desc += ', threshold={:g}'.format(self.curr_fitted_threshold)
        #     add_desc += ', ea={:g}'.format(self.curr_config['eviction_age'])
        #     if self.curr_iter > 0:
        #         dx = self.best_result()
        #         if type(dx) == dict or len(dx) == 1:
        #             add_desc += ', stsr={:.3f}, hr={:.3f}, wr={:.1f}'.format(dx['Service Time Saved Ratio'], dx['IOPSSavedRatio'], dx['Write Rate (MB/s)'])
        # except Exception as e:
        #     add_desc += str(e)
        # if self.stage == 'failure':
        #     add_desc += str(self.err)
        return f'{type(self).__name__}({self.prefix}, stage={self.stage})'

    def ready(self):
        return all(self.ready_status())

    def ready_status(self):
        return [os.path.exists(fn) for fn in self.curr_outputs]


class ExpReproSim(OneShotExp):
    def prepare(self):
        # Read old config file
        # Prepare to write new config file, with replaced args
        self.curr_outputs = [f'{self.dir}/..._cache_perf.txt']
        # newcmd = rework(row['Command'], row, cfg, path) + f' --log-interval {sim_interval}'

    def launch(self):
        pass

    def process(self):
        pass

    def result(self):
        return results.SimResult(self.curr_outputs[0])
    pass

    # def rework(cmd, row, cfg, path):
    #     dirn = os.path.dirname(os.path.dirname(row['Filename']))
    #     cmd = cmd.replace(dirn, path)
    #     cmd = cmd.split(' ')
    #     assert cmd[10].endswith('.trace')
    #     cmd[10] = cfg["test_config"]["traceFileName"]
    #     cmd = ' '.join(cmd).split(' -')
    #     cmd = [c for c in cmd if not (c.startswith('-ep-analysis') or c.startswith('-offline-ap'))]
    #     cmd = ' -'.join(cmd)
    #     return cmd


class ExpReproCacheBench(OneShotExp):
    def prepare(self):
        self.cmds = adaptors.CacheBenchAdaptor._get_cmds(self.orig, self.name, shortpath=self.prefix, output_base_dir=self.output_base_dir)
        path = self.cmds['path']
        modelpaths = []
        replayCfg = self.cmds['cfg']["test_config"]["replayGeneratorConfig"]
        if "modelPath" in replayCfg["mlAdmissionConfig"]:
            modelpaths.append(replayCfg["mlAdmissionConfig"]["modelPath"])
        if "prefetchingConfig" in replayCfg and "modelPath" in replayCfg["prefetchingConfig"]:
            for k in ["offset_start", "offset_end", "size"]:
                modelpaths.append(replayCfg["prefetchingConfig"]["modelPath"] + f"_{k}.model")
            if "predict" in replayCfg["prefetchingConfig"]["when"]:
                modelpaths.append(replayCfg["prefetchingConfig"]["modelPath"] + f"_pred_net_pf_st_binary.model")
        for f in modelpaths:
            if not os.path.exists(f):
                cmd = f'{local_cluster.RUNPY_LOCATION} py -B -m episodic_analysis.train --exp {self.orig["ExperimentName"]} --policy PolicyUtilityServiceTimeSize2 --region {self.orig["Region"]} --sample-ratio {self.orig["OrigSampleRatio"]} --sample-start {self.orig["OrigSampleStart"]} --trace-group {self.orig["TraceGroup"]} --supplied-ea physical --target-wrs 34 50 100 75 20 10 60 90 30 --target-csizes 366.475 --output-base-dir {local_cluster.OUTPUT_LOCATION}/spring23 --suffix /fs_meta+block+chunk/accs_15 --eviction-age {self.orig["Assumed Eviction Age (s)"]} --rl-init-kwargs filter_=prefetch --train-target-wr {self.orig["Target Write Rate"]} --train-models prefetch admit --no-episodes --train-split-secs 86400 --ap-acc-cutoff 15 --ap-feat-subset meta+block+chunk'
                # print(cmd)
                local_cluster.launch_cmd(cmd, queue='gen-par4')
                # raise Exception(f"Prereq does not exist: {f}: possible cmd: {cmd}")
        self.prereqs = modelpaths
        self.configfile = f"{path}/cachebench.json"
        self.curr_outputs = [f"{path}/windows.log", f"{path}/progress.out", f"{path}/cachebench.done"]

    def launch(self):
        # Do some tests to prevent repeats
        success = adaptors.CacheBenchAdaptor._run_cmds(self.cmds, run_testbed=True)
        if not success:
            raise Exception("Failure launching")

    def process(self):
        # check if complete
        # init post processing
        cmd = f'{local_cluster.RUNPY_LOCATION} py -B -m episodic_analysis.exps.results {self.configfile}'
        local_cluster.launch_cmd(cmd, queue='par2')

    def result(self):
        return results.CacheBenchResult(self.configfile)
