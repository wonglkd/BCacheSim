from .base import ExpSizeWR


class ExpOPT(ExpSizeWR):
    def add_helper_thresholds(self):
        orig_len = len(self.curr_thresholds)
        self.add_threshold(5)
        self.add_threshold(75)
        return len(self.curr_thresholds) > orig_len

    def _get_thresholds(self, fitted_threshold):
        wr = self.config['target_wr']
        thresholds = [wr]
        if fitted_threshold is not None:
            thresholds += [fitted_threshold - 2.5, fitted_threshold, fitted_threshold + 2.5]
        else:
            thresholds += [wr - 10, wr + 10]
        return thresholds


class ExpOPTFixedEA(ExpOPT):
    def __init__(self, *args,
                 fixed_ea=3600*24*7,
                 init_guess=dict(ea_guess=3600*24*7, hr_guess=0.4),
                 **kwargs):
        init_guess['ea_guess'] = fixed_ea
        self.fixed_ea = fixed_ea
        super().__init__(*args, init_guess=init_guess, **kwargs)

    def converge_ready(self, curr, prev):
        return True

    def process(self):
        super().process()
        if self.curr_iter + 1 in self.state['configs']:
            self.state['configs'][self.curr_iter + 1]['eviction_age'] = self.fixed_ea


# class ExpSizeOPTEarlyAdmit(ExpOPT):
#     @property
#     def run_args(self):
#         return f" --prefetch {self.filenames['earlyadmit']}"


# class ExpSizeOPTEarlyAll(ExpOPTEarlyAdmit):
#     @property
#     def run_args(self):
#         args = f" --prefetch {self.filenames['earlyadmit']}"
#         args += f" --early-evict {self.filenames['earlyevict']}"
#         return args
