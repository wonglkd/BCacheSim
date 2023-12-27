import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from .episodes import service_time
from . import train_utils


def add_args(parser):
    parser.add_argument("--train-target-wr", type=float)


def proc_episode(episode):
    block_id = episode.key
    # num_hits = episode.num_accesses - episode.first_iop
    # res_score = num_hits / episode.size
    res_score = episode.score
    size = episode.size
    accs = episode.accesses
    acc = accs[0]
    feat = acc.features
    features = [feat.op.value, feat.namespace, feat.user,
                acc.orig_offset, acc.orig_endoffset, acc.origsize()]
    labels = [episode.num_accesses, size,
              episode.offset[0], episode.offset[1],
              episode.timespan_logical, episode.timespan_phys,
              np.log(episode.timespan_logical +
                     1), np.log(episode.timespan_phys+0.0001),
              episode.timespan_logical / episode.num_accesses,
              episode.max_interarrival[0],
              episode.max_interarrival[1],
              res_score, np.log(res_score+1)]
    metadata = [block_id, episode.ts_logical[0]]
    return features, labels, metadata


class PrefetcherTrainer(object):
    def __init__(self, *,
                 targets=["offset_start", "size", "offset_end"],
                 trace_kwargs=None, e_age_s=None, wr_threshold=None):
        self.rl = None
        self.policy = None
        self.clfs = {}
        self.trace_kwargs = trace_kwargs
        trace_kwargs['get_features'] = True
        self.e_age_s = e_age_s
        self.targets = targets
        self.wr_threshold = wr_threshold
        self.eps_processor = proc_episode
        self.init_config()

    def init_config(self):
        self.iterations = 1000
        learnrate = 0.005
        featurefraction, threads = 0.9, 20
        self.params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "mse",
            "num_leaves": 63,
            "learning_rate": learnrate,
            "max_bin": 255,
            "feature_fraction": featurefraction,
            "bagging_fraction": featurefraction,
            "bagging_freq": 5,
            "min_data_in_leaf": 50,
            "min_sum_hessian_in_leaf": 5.0,
            "num_threads": threads,
            "verbosity": -1,
        }

        # Create training and validation datasets
        self.feat_names = ["op", "namespace", "user",
                           "acc_start", "acc_end", "acc_size"]
        self.label_names = ["admitted_opt", "rank", "accesses", "size", "offset_start", "offset_end",
                            "timespan_logical", "timespan_physical",
                            "log_timespan_logical", "log_timespan_physical",
                            "avg_interarrival",
                            "max_interarrival_physical",
                            "max_interarrival_logical",
                            "score", "log_score"]

    def set_policy(self, policy):
        self.policy = policy

    def prep_data(self, threshold):
        X, Y, meta = [], [], []
        wr = 0
        # TODO: Nicer way of printing this?
        # ex = self.rl.residencies[0]
        # print(self.eps_processor(ex), ex, ex.s)
        for i, episode in enumerate(self.rl.residencies):
            f_, l_, m_ = self.eps_processor(episode)
            wr += self.rl.th.bytes_to_wr_mbps(episode.size)
            l_ = [int(wr <= threshold), i] + l_
            if wr > threshold:
                break
            X.append(f_)
            Y.append(l_)
            meta.append(m_)
        # print(len(Y))
        # print(set([len(Y[i]) for i in range(len(Y))]))
        # print(Y[0][0])
        # print(Y)

        self.X = np.array(X)
        self.Y = np.array(Y)

    def split(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.3, random_state=42)

    def train(self, label_name):
        print(f"Training model for {label_name}")
        i = self.label_names.index(label_name)
        lgb_train = lgb.Dataset(self.X_train, self.Y_train[:, i],
                                feature_name=self.feat_names)
        lgb_vl = lgb.Dataset(
            self.X_test, self.Y_test[:, i],
            feature_name=self.feat_names, reference=lgb_train)
        params_ = dict(self.params)
        if label_name == "admitted_opt" or label_name.endswith('_binary'):
            params_["objective"] = "binary"
            params_["metric"] = "binary_logloss"
        else:
            params_["objective"] = "regression"
            params_["metric"] = "mse"
        clf = lgb.train(params_,
                        lgb_train,
                        num_boost_round=self.iterations,
                        valid_sets=lgb_vl,
                        verbose_eval=False,
                        early_stopping_rounds=25)
        return clf

    # def filenames(self):
    #     self._filenames = {}
    #     for label_name in self.targets:
    #         self._filenames["model_prefetch_" + label_name] = f"{self.policy.results_dir}/{self.trace_kwargs['region']}_ea={self.e_age_s:g}_{label_name}.model"
    #         return self._filenames

    def prep(self):
        self.rl = self.policy.residency_lists_[self.e_age_s]
        self.prep_data(self.wr_threshold)
        self.split()

    def train_and_save_all(self):
        self.prep()
        for label_name in self.targets:
            self.clfs[label_name] = self.train(label_name)
            self.clfs[label_name].save_model(self.policy.filenames["model_prefetch_" + label_name])
        print(self.evaluate_all())

    def load_all(self):
        self.prep()
        for label_name in self.targets:
            self.clfs[label_name] = lgb.Booster(model_file=self.policy.filenames["model_prefetch_" + label_name])

    def _predicter(self, label_name):
        return self.clfs[label_name].predict

    def evaluate(self, label_name, threshold=0.5):
        i = self.label_names.index(label_name)
        # preds_train = self.clfs[label_name].predict(self.X_train)
        # preds_test = self.clfs[label_name].predict(self.X_test)
        # return {
        #     'label': label_name,
        #     'MSE (Train)': mean_squared_error(self.Y_train[:, i], preds_train),
        #     'MSE (Test)': mean_squared_error(self.Y_test[:, i], preds_test),
        #     'R2 (Train)': r2_score(self.Y_train[:, i], preds_train),
        #     'R2 (Test)': r2_score(self.Y_test[:, i], preds_test),
        # }
        Y_train = self.Y_train[:, i]
        Y_test = self.Y_test[:, i]
        r1 = train_utils.evaluate_clf(self._predicter(label_name), label_name, self.X_train, Y_train, threshold)
        r1['Label'] = label_name
        r1['Subset'] = 'Train'
        r2 = train_utils.evaluate_clf(self._predicter(label_name), label_name, self.X_test, Y_test, threshold)
        r2['Label'] = label_name
        r2['Subset'] = 'Test'
        return [r1, r2]

    def evaluate_all(self):
        rows = []
        for label in self.targets:
            if label in self.clfs:
                rows += self.evaluate(label)
        return pd.DataFrame(rows)


def proc_episode_st(episode):
    block_id = episode.key
    res_score = episode.score
    size = episode.size
    accs = episode.accesses
    acc = accs[0]
    feat = acc.features
    pf_benefit = episode.s['service_time_saved__prefetch'] - episode.s['service_time_saved__noprefetch']
    sts = episode.s['service_time_saved__prefetch']
    sts_nopf = episode.s['service_time_saved__noprefetch']
    max_pf_cost = max(64, episode.chunk_range[1]) - episode.s['chunks_written__prefetch_nowaste']
    sts_ratio = 0 if sts == 0 else pf_benefit/sts

    features = [feat.op.value, feat.namespace, feat.user,
                acc.orig_offset, acc.orig_endoffset, acc.origsize()]
    labels = [episode.num_accesses, size,
              episode.offset[0], episode.offset[1],
              episode.timespan_logical, episode.timespan_phys,
              np.log(episode.timespan_logical +
                     1), np.log(episode.timespan_phys+0.0001),
              episode.timespan_logical / episode.num_accesses,
              episode.max_interarrival[0],
              episode.max_interarrival[1],
              res_score, np.log(res_score+1),
              pf_benefit, sts, sts_nopf, max_pf_cost,
              sts_ratio,
              episode.chunk_range[0], episode.chunk_range[1]]
    metadata = [block_id, episode.ts_logical[0]]
    return features, labels, metadata


def preds_to_range(start_, size_, end_):
    start_ = max(0, start_)
    size_ = max(0, size_)
    end_ = max(end_, start_+size_)
    start_ //= 131072
    end_ //= 131072
    return start_+1, end_+1, end_-start_


class PrefetcherConfTrainer(PrefetcherTrainer):
    """With 'confidence'."""
    def __init__(self, *, targets=["pred_net_pf_st_binary", "pf_benefit", "pred_pf_benefit", "pred_net_pf_st"], **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.init_config()
        self.eps_processor = proc_episode_st
        self.trainer = PrefetcherTrainer(**kwargs)

    def init_config(self):
        super().init_config()
        self.feat_names += ["pred_eps_start", "pre_eps_end"]
        self.label_names += ["pf_benefit", "sts", "sts_nopf", "max_pf_cost", "sts_ratio", "eps_chunk_start", "eps_chunk_end"]
        self.label_names += ["pred_net_pf_st", "pred_pf_benefit", "pred_pf_cost", "pred_net_pf_st_binary"]

    def set_policy(self, policy):
        self.policy = policy
        self.trainer.policy = policy

    def splice(self):
        preds = {}
        for label_name in self.trainer.targets:
            preds[label_name] = self.trainer.clfs[label_name].predict(self.X)
        xl = len(preds["offset_start"])
        y_ = np.zeros((xl, 3))
        for i in range(xl):
            y_[i] = preds_to_range(preds["offset_start"][i], preds["size"][i], preds["offset_end"][i])
        df_tt = pd.DataFrame()
        # df_tt['access_start'] = self.X[:, 3] / 131072 + 1
        # df_tt['access_end'] = (self.X[:, 4]+1) / 131072 + 1
        df_tt['eps_start'] = self.Y[:, self.label_names.index("eps_chunk_start")]
        df_tt['eps_end'] = self.Y[:, self.label_names.index("eps_chunk_end")]
        df_tt['pf_benefit'] = self.Y[:, self.label_names.index("pf_benefit")]
        df_tt['pred_eps_start'] = y_[:, 0]
        df_tt['pred_eps_end'] = y_[:, 1]
        # df_tt['wrongstart'] = (df_tt["eps_start"] < df_tt["pred_eps_start"])
        # df_tt['wrongend'] = (df_tt["eps_end"] > df_tt["pred_eps_end"])
        df_tt['notenough'] = (df_tt["eps_start"] < df_tt["pred_eps_start"]) |  (df_tt["eps_end"] > df_tt["pred_eps_end"])
        # approximation
        df_tt['overfetch'] = df_tt["pred_eps_end"] - df_tt["pred_eps_start"] - (df_tt["eps_end"] - df_tt["eps_start"])
        df_tt.loc[df_tt['overfetch'] < 0, 'overfetch'] = 0
        # df_tt['max_overfetch'] = self.Y[:, self.label_names.index("max_pf_cost")]
        df_tt['pred_pf_benefit'] = df_tt['pf_benefit'] * ~df_tt['notenough']
        df_tt['pred_pf_cost'] = service_time(0, df_tt['overfetch'])
        df_tt['pred_net_pf_st'] = df_tt['pred_pf_benefit']-df_tt['pred_pf_cost']
        df_tt['pred_net_pf_st_binary'] = df_tt['pred_net_pf_st'] > 0.005  # ~0.5 IO

        self.X = np.hstack([self.X, df_tt[['pred_eps_start', 'pred_eps_end']]])
        self.Y = np.append(self.Y.T, [df_tt['pred_net_pf_st']], axis=0).T
        self.Y = np.append(self.Y.T, [df_tt['pred_pf_benefit']], axis=0).T
        self.Y = np.append(self.Y.T, [df_tt['pred_pf_cost']], axis=0).T
        self.Y = np.append(self.Y.T, [df_tt['pred_net_pf_st_binary']], axis=0).T
        # print(self.X.shape, self.Y.shape)
        assert self.X.shape[1] == len(self.feat_names)
        assert self.Y.shape[1] == len(self.label_names)

    def prep(self):
        self.rl = self.policy.residency_lists_[self.e_age_s]
        self.prep_data(self.wr_threshold)
        self.trainer.train_and_save_all()
        self.splice()
        self.split()

    def load_all(self):
        self.rl = self.policy.residency_lists_[self.e_age_s]
        self.prep_data(self.wr_threshold)
        self.trainer.load_all()
        self.splice()
        self.split()
        for label_name in self.targets:
            if "model_prefetch_" + label_name in self.policy.filenames:
                self.clfs[label_name] = lgb.Booster(model_file=self.policy.filenames["model_prefetch_" + label_name])

    def train_and_save_all(self):
        self.prep()
        for label_name in self.targets:
            self.clfs[label_name] = self.train(label_name)
            if "model_prefetch_" + label_name in self.policy.filenames:
                self.clfs[label_name].save_model(self.policy.filenames["model_prefetch_" + label_name])
        print(self.evaluate_all())

    def evaluate_all(self):
        return pd.concat([self.trainer.evaluate_all(), super().evaluate_all()])
