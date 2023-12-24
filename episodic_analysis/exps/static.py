from .base import ExpSizeWR, ExpNoIterate


class ExpCoinFlip(ExpSizeWR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_col = 'AP Probability'
        self.min_search = 0.000001
        self.max_search = 1

    def _launch_run(self, run_args=''):
        print('\nLaunching simulations...')
        run_args_ = run_args + ' ' + self._sim_args()
        run_args_ += ' --offline-ap-decisions ' + self.filenames['thresholds']
        if 'model_prefetch_offset_start' in self.filenames:
            run_args_ += ' --prefetcher-model-path ' + self.filenames['model_prefetch_offset_start'].replace("_offset_start.model", "_{k}.model")
        return self._exp_run(
            threshold_name='--ap-probability',
            policy_dir='coinflip-ap-',
            policy_args='--coinflip-ap --ap coinflip',
            run_args=run_args_)

    def _get_thresholds(self, fitted_threshold):
        if fitted_threshold is not None:
            thresholds = [.2, fitted_threshold * 0.9, fitted_threshold, fitted_threshold * 1.1]
        else:
            thresholds = [.1, .2, .3]
        return thresholds

    def add_helper_thresholds(self):
        orig_len = len(self.curr_thresholds)
        self.add_threshold(.2)
        self.add_threshold(.4)
        return len(self.curr_thresholds) > orig_len


class ExpRejectX(ExpSizeWR):
    """
    RejectX.
    - Factor: how much history to keep (multiple of cache size). Used to be set as ap-
    - Threshold: the X. How many past accesses we need to see before accepting.

    Generally, we can set the threshold (since it is an integer and of limited interesting values and converge on factor).
    """
    def __init__(self, *args, rejectx_ap_threshold=1, valid_kwargs=None, **kwargs):
        valid_kwargs = valid_kwargs or []
        valid_kwargs += ['rejectx_ap_threshold']
        super().__init__(*args, rejectx_ap_threshold=rejectx_ap_threshold, valid_kwargs=valid_kwargs, **kwargs)
        self.search_col = 'AP Probability'
        self.min_search = 0.01
        self.max_search = None

    def _launch_run(self, run_args=''):
        print('\nLaunching simulations...')
        # threshold = [1,2,3]
        # factor = 0.1, 0.5, ... (how long to keep history for)
        run_args_ = run_args + ' ' + self._sim_args()
        run_args_ += f' --ap-threshold {self.config["rejectx_ap_threshold"]}'
        run_args_ += ' --offline-ap-decisions ' + self.filenames['thresholds']
        if 'model_prefetch_offset_start' in self.filenames:
            run_args_ += ' --prefetcher-model-path ' + self.filenames['model_prefetch_offset_start'].replace("_offset_start.model", "_{k}.model")
        return self._exp_run(
            threshold_name='--ap-probability',
            policy_dir=f'rejectx-ap-{self.config["rejectx_ap_threshold"]}_',
            policy_args='--rejectx-ap --ap rejectx',
            run_args=run_args_)

    def _get_thresholds(self, fitted_threshold):
        if fitted_threshold is not None:
            thresholds = [.25, fitted_threshold - 0.1, fitted_threshold, fitted_threshold + 0.1, .5]
        else:
            thresholds = [.1, .25, .5]
        return thresholds

    def add_helper_thresholds(self):
        orig_len = len(self.curr_thresholds)
        self.add_threshold(.1)
        self.add_threshold(.7)
        return len(self.curr_thresholds) > orig_len


class ExpFlashield(ExpSizeWR):
    def _get_thresholds(self, fitted_threshold):
        # thresholds = [1, 2, 3, 4, 5]
        thresholds = [1, 2]
        return thresholds

    def converge_ready(self, curr, prev):
        return True

    def add_threshold(self, new_threshold):
        return super().add_threshold(int(new_threshold))

    def _launch_run(self, run_args=''):
        print('\nLaunching simulations...')
        run_args_ = run_args + ' ' + self._sim_args()
        ap_name = self.config['sim_ap']
        dirn = f"{ap_name}-"
        return self._exp_run(
            policy_dir=dirn,
            policy_args=f'--ap {ap_name}',
            venv='cachelib-py38',
            run_args=run_args_)

    def add_helper_thresholds(self):
        return False


class ExpFlashieldProb(ExpSizeWR):
    def __init__(self, *args, flashieldprob_ap_min_hits=1, valid_kwargs=None, **kwargs):
        valid_kwargs = valid_kwargs or []
        valid_kwargs += ['flashieldprob_ap_min_hits']
        super().__init__(*args, flashieldprob_ap_min_hits=flashieldprob_ap_min_hits, valid_kwargs=valid_kwargs, **kwargs)
        self.min_search = 0.000001
        self.max_search = 1

    def _get_thresholds(self, fitted_threshold):
        if fitted_threshold is not None:
            thresholds = [fitted_threshold * .5, fitted_threshold, fitted_threshold * 2, .2]
        else:
            thresholds = [.1, .2, .4, .8]
        return thresholds

    def converge_ready(self, curr, prev):
        return True

    def _launch_run(self, run_args=''):
        print('\nLaunching simulations...')
        run_args_ = run_args + ' ' + self._sim_args()
        run_args_ += f' --flashieldprob-ap-min-hits {self.config["flashieldprob_ap_min_hits"]}'
        ap_name = self.config['sim_ap']
        dirn = f'{ap_name}-{self.config["flashieldprob_ap_min_hits"]}-'

        return self._exp_run(
            policy_dir=dirn,
            policy_args=f'--ap {ap_name}',
            venv='cachelib-py38',
            run_args=run_args_)

    def add_helper_thresholds(self):
        orig_len = len(self.curr_thresholds)
        self.add_threshold(.99)
        self.add_threshold(.1)
        return len(self.curr_thresholds) > orig_len
