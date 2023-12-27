from .episodes import service_time
from ..cachesim import dynamic_features

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from . import train_utils


def add_args(parser):
    parser.add_argument("--ap-feat-subset", default="meta+block+chunk")
    parser.add_argument("--ap-acc-cutoff", default=6, type=int)


def opts_to_args(args):
    return dict(feat_subset=args.ap_feat_subset,
                target_wr=args.train_target_wr,
                acc_cutoff=args.ap_acc_cutoff)


class BaseAdmissionTrainer(object):
    def __init__(self, *, policy=None, prefetch_trainer=None,
                 target_wr=None, feat_subset=None,
                 acc_cutoff=6, hrs=6, hr_mult=1,
                 split_seed=42, test_size=0.3,
                 nonstd_pol=False,
                 labels=['threshold_binary', 'remaining_binary']):
        self.ts_use_logical = False
        # Backwards compatibility
        if acc_cutoff == -1:
            acc_cutoff = None
        self.acc_cutoff = acc_cutoff  # No of accesses to use from each episode
        self.hrs = hrs
        self.HR = 3600 * hr_mult
        self.split_seed = split_seed
        self.test_size = test_size
        self.init_config()
        self.policy = policy
        self.rl = policy.rl
        self.HR_L = self.rl.th.phy_dur_to_logical(self.HR)
        self.prefetch_trainer = prefetch_trainer
        self.clfs = {}
        self.df_X = None
        self.df_Y = None
        self.target_wr = target_wr
        if 'remaining_binary' in labels:
            # Because calculation of remaining_binary in prep is hardcoded to ST / size
            assert self.policy.__class__.__name__ == 'PolicyUtilityServiceTimeSize2' or nonstd_pol
            self.cutoff_score = self.policy.get_analysis().loc[lambda x: x['Target Write Rate'] == target_wr].iloc[0]['Cutoff score'].tolist()
        self.feat_subset = feat_subset
        self.labels = labels

    def init_config(self):
        pass

    def prep(self, force=False):
        if self.df_X is not None:
            return
        self.preprocess()
        self.rows, self.label_rows = self.generate_data()
        print(f"Source Episodes: {len(self.rl.episodes)}, Used Episodes: {pd.DataFrame(self.rows)['feat_ep_id'].nunique()}")
        self.split_idxes, self.split_block_ids = self.split(self.rows)
        print(f"Rows: {len(self.rows)}, Train: {len(self.split_idxes['train'])}, Test: {len(self.split_idxes['test'])}")
        self.df_X = pd.DataFrame(flatten_feat(self.rows))
        self.df_Y = pd.DataFrame(self.label_rows)
        self.df_Y['threshold_binary'] = self.df_Y['threshold'] < self.target_wr
        print(self.df_Y['threshold_binary'].describe())
        print(self.df_Y['threshold_binary'].sum())
        if 'remaining_binary' in self.labels:
            self.df_Y['remaining_binary'] = self.df_Y[f'remaining_sts__{self.rl.filter}'] / self.df_X['feat_eps|opt-num_chunks'] > self.cutoff_score

    def copy_from(self, other):
        for x in ['rows', 'label_rows', 'split_idxes', 'split_block_ids', 'df_X', 'df_Y']:
            setattr(self, x, getattr(other, x))

    def train_all(self):
        if self.df_X is None:
            self.prep()

        subsets_cols = subsets(self.df_X)
        X_ = self.df_X
        if self.feat_subset:
            self.feat_cols = subsets_cols[self.feat_subset]
            X_ = X_[self.feat_cols]
        else:
            self.feat_cols = list(X_.columns)
        Y_ = self.df_Y
        assert len(X_) > 0, "Empty df_X"
        for label_name in self.labels:
            self.train(label_name, X_, Y_, self.split_idxes)

    def load_all(self):
        self.prep()
        for label_name in self.labels:
            if "model_admit_" + label_name in self.policy.filenames:
                self.clfs[label_name] = lgb.Booster(model_file=self.policy.filenames["model_admit_" + label_name])
                print(self.clfs[label_name].feature_name())
            else:
                print(f"No filename for model_admit_{label_name}, not saving")

    def save_all(self):
        for label_name in self.labels:
            if "model_admit_" + label_name in self.policy.filenames:
                self.clfs[label_name].save_model(self.policy.filenames["model_admit_" + label_name])
                print(self.clfs[label_name].feature_name())
            else:
                print(f"No filename for model_admit_{label_name}, not saving")
        # Test that it loads
        for label_name in self.labels:
            if "model_admit_" + label_name in self.policy.filenames:
                _ = lgb.Booster(model_file=self.policy.filenames["model_admit_" + label_name])

    def preprocess(self):
        self.eps_by_block = {}
        for eps in self.rl.episodes:
            if eps.key not in self.eps_by_block:
                self.eps_by_block[eps.key] = []
            self.eps_by_block[eps.key].append(eps)

    def generate_data(self):
        filter_ = self.rl.filter
        rows = []
        label_rows = []
        # pol.rl.th.start_ts
        hr_unit = self.HR_L if self.ts_use_logical else self.HR
        for i, (block_id, epses) in enumerate(self.eps_by_block.items()):
            dynfeat_c = dynamic_features.DynamicFeatures(
                self.hrs, granularity='chunk', hr_unit=hr_unit)
            dynfeat_b = dynamic_features.DynamicFeatures(
                self.hrs, granularity='block', hr_unit=hr_unit)
            last_ts = 0
            epses = sorted(epses, key=lambda x: x.ts_logical[0])
            prev_ep_stats = None
            for j, eps in enumerate(epses):
                st_remaining = eps.s[f'service_time_saved__{filter_}']
                cts = eps.accesses[0].ts_logical if self.ts_use_logical else eps.accesses[0].ts
                tts = last_ts+1 + hr_unit
                while tts < min(cts, tts+hr_unit*(self.hrs+1)) and last_ts != 0:
                    dynfeat_c.updateFeatures(-1, tts)
                    dynfeat_b.updateFeatures(-1, tts)
                    tts += hr_unit
                dynfeat_c.updateFeatures(-1, cts)
                dynfeat_b.updateFeatures(-1, cts)
                eps_feat = eps2feat(eps)
                eps_labels = eps2labels(eps)
                for k, ac in enumerate(eps.accesses):
                    cts = ac.ts_logical if self.ts_use_logical else ac.ts
                    assert cts >= last_ts, (cts, last_ts)
                    if self.acc_cutoff is not None and k >= self.acc_cutoff:
                        continue
                    # TODO: Reenable but only for training
                    # if ac.ts <= self.rl.th.start_ts + hr_unit*(self.hrs+1):
                    #     continue
                    next_ac_st = service_time(
                        1, eps.accesses[k+1].num_chunks()) if k+1 < len(eps.accesses) else 0
                    rowd = {'id': (i, j, k), 'block_id': block_id}
                    meta = ac.features.toList(with_size=True)

                    rowd['feat_metadata'] = dict(zip(['op', 'ns', 'user'], meta[:3]))
                    rowd['feat_metadata_size'] = dict(zip(['start', 'end', 'size'], meta[3:]))
                    rowd['feat_dynamic_b'] = dynfeat_b.getFeature(block_id)
                    rowd['feat_eps'] = eps_feat
                    rowd['feat_ep_i'] = j
                    if type(block_id) == tuple:
                        rowd['feat_shard'] = block_id[1]
                    if 'episode_id' in eps.s:
                        rowd['feat_ep_id'] = eps.s['episode_id']
                    rowd['feat_ep_acc_i'] = k
                    rowd['feat_time_since_last_acc_inep'] = 100000000.0 if k == 0 else cts - last_ts
                    if prev_ep_stats:
                        rowd['feat_prev_ep'] = prev_ep_stats
                    comb = np.zeros(self.hrs, dtype=int)
                    for chunk_id in ac.chunks():
                        key = (block_id, chunk_id)
                        cfeat = dynfeat_c.getFeature(key)
                        rowd[f'feat_dynamic_c|{chunk_id}'] = cfeat
                        comb += np.asarray(cfeat)
                    rowd['feat_dynamic_c_combined'] = comb.tolist()

                    rows.append(rowd)
                    labelr = dict(eps_labels)
                    next_ac_st = min(next_ac_st, st_remaining)
                    if self.acc_cutoff is None or k + 1 < self.acc_cutoff:
                        labelr[f'marginal_sts__{filter_}'] = next_ac_st
                    else:
                        labelr[f'marginal_sts__{filter_}'] = st_remaining

                    labelr[f'remaining_sts__{filter_}'] = st_remaining
                    if k + 1 < len(eps.accesses) or j + 1 < len(epses):
                        if k + 1 < len(eps.accesses):
                            next_acc = eps.accesses[k+1]
                        else:
                            next_acc = epses[j+1].accesses[0]
                        next_ts = next_acc.ts_logical if self.ts_use_logical else next_acc.ts
                        labelr['time_to_next_acc'] = next_ts - cts
                    else:
                        # Proxy value should probably be a very high value.
                        labelr['time_to_next_acc'] = None

                    st_remaining -= next_ac_st
                    label_rows.append(labelr)
                    for chunk_id in ac.chunks():
                        dynfeat_c.updateFeatures(key, cts)
                    dynfeat_b.updateFeatures(block_id, cts)
                    last_ts = cts
                prev_ep_stats = {
                    'opt-num_chunks': eps_feat['opt-num_chunks'],
                    'opt-num_accesses': eps_labels['num_accesses'],
                    'opt-timespan_physical': eps_labels['timespan_phys'],
                    'opt-timespan_logical': eps_labels['timespan_logical'],
                    'opt-max_ia': eps_labels['max_interarrival'],
                    'opt-score': eps_labels['score'],
                    'opt-threshold': eps_labels['threshold'],
                    f'opt-sts__{filter_}': eps_labels[f'service_time_saved__{filter_}'],
                }
            del dynfeat_c
            del dynfeat_b
        return rows, label_rows

    def split(self, rows):
        train_idx, test_idx = train_test_split(
            list(self.eps_by_block.keys()), random_state=self.split_seed, test_size=self.test_size)
        split_idxes = {'train': [], 'test': []}
        for i, row in enumerate(rows):
            split = 'train' if row['block_id'] in train_idx else 'test'
            split_idxes[split].append(i)
        return split_idxes, {'train': train_idx, 'test': test_idx}

    def train(self, label_name, X_, Y_, split_idxes):
        # def train(label_name, X_, Y_):
        # OPT-Range, ML-Range, ML-When
        print(f"Training {label_name}")
        Y__ = Y_[label_name]
        X_train, X_test = X_.iloc[self.split_idxes['train']], X_.iloc[self.split_idxes['test']]
        Y_train, Y_test = Y__.iloc[self.split_idxes['train']], Y__.iloc[self.split_idxes['test']]
        self.clfs[label_name] = self._train(label_name, X_train, X_test, Y_train, Y_test)
        return self.clfs[label_name]

    def _train(self, label_name, X_train, X_test, Y_train, Y_test):
        """Children to implement actual training"""
        raise NotImplementedError

    def _predicter(self, label_name):
        return self.clfs[label_name].predict

    def evaluate(self, label_name, threshold=0.5):
        Y__ = self.df_Y[label_name]
        subsets_cols = subsets(self.df_X)
        X_ = self.df_X
        if self.feat_subset:
            self.feat_cols = subsets_cols[self.feat_subset]
            X_ = X_[self.feat_cols]

        X_train, X_test = X_.iloc[self.split_idxes['train']], X_.iloc[self.split_idxes['test']]
        Y_train, Y_test = Y__.iloc[self.split_idxes['train']], Y__.iloc[self.split_idxes['test']]
        self._used = {
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test,
        }

        r1 = train_utils.evaluate_clf(self._predicter(label_name), label_name, X_train, Y_train, threshold)
        # r1['Feat'] = self.feat_subset
        r1['Label'] = label_name
        r1['Subset'] = 'Train'
        r2 = train_utils.evaluate_clf(self._predicter(label_name), label_name, X_test, Y_test, threshold)
        # r2['Feat'] = self.feat_subset
        r2['Label'] = label_name
        r2['Subset'] = 'Test'
        return [r1, r2]

    def evaluate_all(self, threshold=0.5):
        print('Labels:', self.labels)
        rows = [pd.DataFrame(self.evaluate(label, threshold=threshold)) for label in self.labels]
        for label in self.labels:
            print(label)
            booster = self.clfs[label]
            importance = booster.feature_importance()
            feature_name = booster.feature_name()
            tuples = sorted(zip(feature_name, importance), key=lambda x: x[1])
            feats, values = zip(*tuples)
            for k, v in tuples:
                print(f"\t{k}: {v}")
        return pd.concat(rows)


class GBAdmissionTrainer(BaseAdmissionTrainer):
    def init_config(self):
        self.iterations = 2000
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

    def _train(self, label_name, X_train, X_test, Y_train, Y_test):
        dsTrain = lgb.Dataset(X_train, Y_train)
        dsTest = lgb.Dataset(X_test, Y_test, reference=dsTrain)
        params_ = dict(self.params)
        if label_name.endswith("_binary"):
            params_["objective"] = "binary"
            params_["metric"] = "binary_logloss"
        else:
            params_["objective"] = "regression"
            params_["metric"] = "mse"
        clf = lgb.train(params_,
                        dsTrain,
                        num_boost_round=self.iterations,
                        valid_sets=dsTest,
                        verbose_eval=False,
                        early_stopping_rounds=25
                        )
        return clf


# For backwards compatibility
AdmissionTrainer = GBAdmissionTrainer


class BayesOptAdmissionTrainer(BaseAdmissionTrainer):
    """
    Bayes Optimal Policy.

    Assign score equal to all those that share features.
    """
    def init_config(self):
        # For unknowns: ideally should be average acceptance rate.
        self.default_value = 0
        self.train_split = 'train'

    def train(self, label_name, X_, Y_, split_idxes):
        print(f"Training {label_name}")
        assert label_name == 'score'
        den = 'feat_eps|opt-num_chunks'
        if label_name == 'score':
            num = f'service_time_saved__{self.rl.filter}'
        elif label_name == 'remaining':
            num = f'remaining_sts__{self.rl.filter}'
        else:
            raise NotImplementedError(label_name)

        dfc = pd.concat([self.df_X, Y_], axis=1)
        if self.train_split != 'all':
            dfc = dfc.iloc[self.split_idxes[self.train_split]]
        self.clfs[label_name] = self._train(dfc, num, den)
        return self.clfs[label_name]

    def _train(self, dfc, num, den):
        model = {}
        i, j = 0, 0
        for featx, dshared in dfc.groupby(self.feat_cols):
            i += 1
            j += len(dshared)
            score = dshared[num].sum() / dshared[den].sum()
            model[featx] = score
        assert i < j
        assert j == len(dfc)
        return model

    def _predicter(self, label_name):
        def predicter(X_):
            model = self.clfs[label_name]

            def row_pred(row):
                idx = tuple(row)
                return model.get(idx, self.default_value)
            return X_.apply(row_pred, axis=1)
        return predicter


def eps2feat(self):
    return {
        'opt-range_s': self.chunk_range[0],
        'opt-range_e': self.chunk_range[1],
        'opt-num_chunks': self.num_chunks,
    }


def eps2labels(self):
    attrs = ['num_accesses', 'score', 'threshold', 'key', 'timespan_phys', 'timespan_logical', 'max_interarrival']
    ret = {k: getattr(self, k) for k in attrs}
    for k in ['service_time_saved__prefetch', 'service_time_saved__noprefetch', 'service_time_orig']:
        ret[k] = self.s[k]
    return ret


def flatten_feat(rows):
    ret = []
    for row in rows:
        dd = {}
        for k, v in row.items():
            if not k.startswith('feat_'):
                continue
            if isinstance(v, list) or isinstance(v, tuple):
                for i, vv in enumerate(v):
                    dd[f'{k}|{i}'] = vv
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    dd[f'{k}|{kk}'] = vv
            else:
                dd[k] = v
        ret.append(dd)
    return ret


def subsets(df_X):
    opt_feat = ['feat_eps', 'feat_ep_i', 'feat_ep_id', 'feat_ep_acc_i', 'feat_prev_ep', 'feat_time_since_last_acc_inep']
    subsets = {
        'meta': ['feat_metadata', 'feat_metadata_size'],
        'meta_nosize': ['feat_metadata'],
    }
    orig = list(subsets.keys())
    for k in orig:
        subsets[f'{k}+block'] = subsets[k] + ['feat_dynamic_b']
        subsets[f'{k}+block+chunk'] = subsets[k] + ['feat_dynamic_b', 'feat_dynamic_c_combined']
        subsets[f'{k}+block+chunk_ind'] = subsets[k] + ['feat_dynamic_b', 'feat_dynamic_c_combined', 'feat_dynamic_c']
    orig = list(subsets.keys())

    for k in orig:
        subsets[f'{k}+shard'] = subsets[k] + ['feat_shard']
        subsets[f'{k}++opt'] = subsets[k] + opt_feat
        subsets[f'{k}++opt_prev'] = subsets[k] + ['feat_prev_ep']
        subsets[f'{k}++pred'] = subsets[k] + ['feat_pred']
        subsets[f'{k}++opt_acc_i'] = subsets[k] + ['feat_ep_acc_i']
        subsets[f'{k}++opt_acc_i,pred'] = subsets[k] + ['feat_ep_acc_i', 'feat_pred']
        subsets[f'{k}++opt_ep_i'] = subsets[k] + ['feat_ep_i']
        subsets[f'{k}++opt_ep_i,pred'] = subsets[k] + ['feat_ep_i', 'feat_pred']
        subsets[f'{k}++opt_ep_i,acc_i'] = subsets[k] + ['feat_ep_i', 'feat_ep_acc_i']
        subsets[f'{k}++opt_ep_i,acc_i,pred'] = subsets[k] + ['feat_ep_i', 'feat_ep_acc_i', 'feat_pred']

    subsets['opt'] = opt_feat

    subsets_cols = {}
    for k, v in subsets.items():
        subsets_cols[k] = [col for col in df_X if col.split("|")[0] in v]
    return subsets_cols
