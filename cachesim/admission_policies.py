from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from collections import deque
# import methodtools
import random

try:
    import lightgbm as lgb
except ModuleNotFoundError:
    print("Unable to import lightgbm")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

import numpy as np
import spookyhash

from .sim_features import count_feat
from .utils import LOG_REQ
from .utils import ods
from . import utils
from .ep_helpers import AccessPlus
from ..episodic_analysis.episodes import service_time


class AP(object):
    def __init__(self):
        self.features = ''

    def accept(self, key, ts, metadata=None):
        raise NotImplementedError

    def batchAccept(self, batch, ts, metadata=None, check_only=False):
        decisions = {k: self.accept(k, ts, metadata=metadata) for k in batch}
        if not check_only:
            self.count_decisions(decisions)
        return decisions

    def count_decisions(self, decisions):
        accepts = sum(1 for v in decisions.values() if v)
        ods.bump(f"ap_{self.name}_accepts", v=accepts)
        ods.bump(f"ap_{self.name}_rejects", v=len(decisions)-accepts)
        return decisions

    @property
    def name(self):
        return type(self).__name__

    def __repr__(self):
        return f"{self.name}()"


class AcceptAll(AP):
    def accept(self, k, ts, metadata=None):
        return True


class RejectXAP(AP):
    def __init__(self, threshold, window_count, factor=None):
        """
        threshold: X rejects that happen before accept
        window_count: no of cache items to keep in history (aka size of shadow cache)
        factor: ideally size of shadow cache / real cache
        """
        super().__init__()
        self.window_count = window_count
        self.history = OrderedDict()
        self.threshold = threshold
        self.accepts = 0
        self.factor = factor

    def accept(self, key, ts, metadata=None, check_only=False):
        if check_only:
            return self.history.get(key, 0) + 1 > self.threshold

        # keep history only for window count
        if len(self.history) >= self.window_count:
            self.history.popitem(last=False)

        # if never seen before: reject
        if key not in self.history:
            self.history[key] = 1
            result = False
        else:
            self.history[key] += 1
            result = self.history[key] > self.threshold
        if metadata is None:
            ts_inserted = ts
        else:
            ts_inserted = metadata['ts'][key]
        if result:
            self.accepts += 1
            LOG_REQ("flashcache", key, ts, "SET", result="Accept {}".format(self.accepts))
        return result

    @property
    def name(self):
        return "RejectX"

    def __repr__(self):
        desc = f", factor={self.factor}" if self.factor else ""
        return f"{self.name}(threshold={self.threshold}, window={self.window_count:g}{desc})"


"""
Adapted from Juncheng Yang's Flashield implementation (SOSP23, S3-FIFO)
https://github.com/Thesys-lab/sosp23-s3fifo/blob/79fde46c03180b95b1091249a6f140282aeae333/scripts/flashield/flashield.py
"""


class FlashieldModel(object):
    def __init__(self, threshold, probability=False):
        self.features = {}
        self.labels = defaultdict(int)
        # Flashliness threshold (n) in paper: an integer, denoting how many
        # past accesses are necessary
        self.threshold = threshold
        self.probability = probability
        self.positive_examples = False

    def add_training_sample_features(self, obj_id, n_access):
        if random.randint(1, n_access) == 1:
            self.features[obj_id] = n_access

    def add_training_sample_labels(self, obj_id):
        if obj_id in self.features:
            self.labels[obj_id] += 1
            self.positive_examples = True

    def train(self):
        if not self.positive_examples:
            self.features = {}
            self.labels = defaultdict(int)
            return None
        print(f"Training Flashield with {len(self.features)} examples")
        feature_list, label_list = [], []
        for obj_id, n_access in self.features.items():
            feature_list.append([n_access])
            label_list.append(self.labels[obj_id] > self.threshold)
        # TODO: Replace this with uniform sampling
        if len(feature_list) > 1000000:
            feature_list = feature_list[:1000000]
            label_list = label_list[:1000000]
        clf = make_pipeline(StandardScaler(), SVC(probability=self.probability))
        clf.fit(np.reshape(feature_list, (-1, 1)), label_list)
        print("Flashield training completed")
        return clf


class FlashieldAP(AP):
    def __init__(self, *args, threshold=None, **kwargs):
        assert threshold is not None
        super().__init__(*args, **kwargs)
        self.start_ts = None
        self.train_ts_start = 0
        self.train_ts_end = 3600
        self.train_ts_label_end = 7200
        self.threshold = threshold

        self.trainer = FlashieldModel(threshold=threshold)
        self.classifier = None
        self.hooks = {"every_chunk_before_insert": [self.on_every_chunk_before_insert]}

    def on_every_chunk_before_insert(self, key, ts, *, ram_cache=None, **kwargs):
        """This is to be called after hits are counted"""
        if self.classifier is not None:
            return
        if self.start_ts is None:
            self.start_ts = ts
        ts_since_start = ts - self.start_ts
        if self.train_ts_start <= ts_since_start.physical <= self.train_ts_end:
            # Collecting features and training samples
            num_reads = 1
            if key in ram_cache.cache:
                num_reads = ram_cache.cache[key].hits
            self.trainer.add_training_sample_features(key, num_reads)
        elif self.train_ts_end < ts_since_start.physical <= self.train_ts_label_end:
            # Collecting labels
            self.trainer.add_training_sample_labels(key)
        elif ts_since_start.physical > self.train_ts_label_end:
            self.classifier = self.trainer.train()
            if self.classifier is None:
                print("No positive examples; reset and start collecting again")
                # Try to train again
                self.start_ts = ts

    def batchAccept(self, batch, ts, metadata=None, check_only=False):
        feats = [metadata['ramcache_hits'][key] for key in batch]
        if self.classifier is None:
            decs = [True] * len(feats)
        else:
            ods.bump("ml_batches")
            ods.bump("ml_predictions", v=len(feats))
            feats = np.reshape(feats, (-1, 1))
            decs = self.classifier.predict(feats)
        decisions = dict(zip([k for k in batch], decs))
        if not check_only:
            self.count_decisions(decisions)
        return decisions

    def accept(self, key, ts, metadata=None):
        if self.classifier is None:
            return True
        ramhits = metadata['ramcache_hits'][key]
        ods.bump("ml_predictions")
        return self.classifier.predict([ramhits])

    def __repr__(self):
        return f"{self.name}({self.threshold})"


class FlashieldProbAP(FlashieldAP):
    """Vanilla Flashield becomes too selective because of lack of DRAM hits.
    We try and predict a flashiness score. Use predict_proba"""
    def __init__(self, *args, n=None, **kwargs):
        assert n is not None
        super().__init__(*args, **kwargs)
        self.n = n
        self.trainer = FlashieldModel(threshold=n, probability=True)

    def batchAccept(self, batch, ts, metadata=None, check_only=False):
        feats = [metadata['ramcache_hits'][key] for key in batch]
        if self.classifier is None:
            decs = [True] * len(feats)
        else:
            ods.bump("ml_batches")
            ods.bump("ml_predictions", v=len(feats))
            feats = np.reshape(feats, (-1, 1))
            decs = self.classifier.predict_proba(feats)[:, 1]
            decs = [v >= self.threshold for v in decs]
        decisions = dict(zip([k for k in batch], decs))
        # if self.classifier:
        #     print(decisions)
        if not check_only:
            self.count_decisions(decisions)
        return decisions

    def accept(self, key, ts, metadata=None):
        if self.classifier is None:
            return True
        ramhits = metadata['ramcache_hits'][key]
        ods.bump("ml_predictions")
        return self.classifier.predict_proba([ramhits])[:, 1] >= self.threshold

    def __repr__(self):
        return f"{self.name}(n={self.n}, threshold={self.threshold})"


class CoinFlipAP(AP):
    def __init__(self, probability):
        super().__init__()
        self.prob = probability

    def accept(self, key, ts):
        return random.random() < self.prob


class CoinFlipDetAP(AP):
    """Deterministic Coin Flip AP"""
    def __init__(self, probability):
        super().__init__()
        self.prob = probability
        self.seed = 1

    def accept(self, key, ts, metadata=None, check_only=False, **kwargs):
        if metadata is None:
            ts_inserted = ts
        else:
            ts_inserted = metadata['ts'][key]
        h = spookyhash.hash64(bytes(f"{key[0]}|{ts_inserted.logical+1}", "ascii"),seed=self.seed)
        hf = h / ((1 << 64) - 1)
        result = hf < self.prob
        if not check_only:
            LOG_REQ("flashcache", key, ts, "SET", result=f"{ts_inserted.logical+1} {h} {hf} {'Accept' if result else 'Reject'}")
        return result

    @property
    def name(self):
        return "CoinFlipDet-P"

    def __repr__(self):
        return f"{self.name}({self.prob})"


# learned admission policy
class LearnedAP(AP):
    def __init__(self, threshold, model_path=None):
        assert model_path
        self.threshold = threshold
        self.gbm = lgb.Booster(model_file=model_path)
        self.seen_before = Counter()
        self.features = 'dfeat+meta'

    def _predict(self, batch, ts):
        Xs = list(batch.values())
        # remove size and offset for now
        Xs = [x[:-3] if len(x) == 12 else x for x in Xs]
        features = np.array(Xs)
        # for x in Xs:
        #     self.seen_before[tuple(x)] += 1    
        # TODO: Log and check for seen-before features
        # result:
        # dynF0 .. dynF11 lAD kf1 kf2 kf3 size
        ods.bump("ml_batches")
        ods.bump("ml_predictions", v=len(features))
        try:
            return self.gbm.predict(features)
        except:
            print(Xs)
            print(features)
            raise

    def batchAccept(self, batch, ts, *, metadata=None, check_only=False):
        predictions = self._predict(batch, ts)
        # for a, b in zip(batch.keys(), predictions):
        #     print(f"ML_PRED {a} {b > self.threshold} {b}")
        decisions = dict(
            zip(batch.keys(), [pred > self.threshold for pred in predictions])
        )
        if not check_only:
            self.count_decisions(decisions)
        return decisions

    def __repr__(self):
        return f"{self.name}({self.threshold})"


class LearnedSizeAP(LearnedAP):
    def __init__(self, *args, size_opt=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.size_opt = size_opt

    def batchAccept(self, batch, ts, metadata=None, check_only=False):
        predictions = self._predict(batch, ts)
        sizes = metadata['size']
        decisions = {k: pred / sizes[k] > self.threshold
                     for k, pred in zip(batch.keys(), predictions)}
        if not check_only:
            self.count_decisions(decisions)
        return decisions


class NewMLAP(LearnedAP):
    def __init__(self, *args, feat_subset='meta+block+chunk', **kwargs):
        super().__init__(*args, **kwargs)
        self.features = feat_subset
        self.num_features = count_feat(feat_subset)
        assert len(self.gbm.feature_name()) == self.num_features, (self.gbm.feature_name(), self.num_features)

    def __predict(self, batch, ts):
        Xs = list(batch.values())
        assert len(Xs[0]) == self.num_features
        features = np.array(Xs)
        ods.bump("ml_batches")
        ods.bump("ml_predictions", v=len(features))
        try:
            return self.gbm.predict(features)
        except:
            print(Xs)
            print(features)
            raise

    def __repr__(self):
        return f"{self.name}(th={self.threshold}, fs={self.features})"


class TrainingEpisode(object):
    def __init__(self, acc):
        self.key = acc.block_id
        self.chunk_range = acc.chunk_range
        self.num_chunks = self.chunk_range[1] - self.chunk_range[0]
        self.ts_range = [acc.ts, acc.ts]
        self.num_accesses = 1
        self.features = OrderedDict()
        self.s = {
            "service_time_orig": service_time(1, acc.num_chunks),
            "service_time_saved__prefetch": 0,
        }
        self.score = None

    def add_access(self, acc):
        new_chunk_range = [min(self.chunk_range[0], acc.chunk_range[0]), max(self.chunk_range[1], acc.chunk_range[1])]
        added_chunks = new_chunk_range[1] - new_chunk_range[0] - self.num_chunks 
        self.num_chunks = new_chunk_range[1] - new_chunk_range[0]
        self.chunk_range = new_chunk_range
        self.s["service_time_orig"] += service_time(1, acc.num_chunks)
        self.s["service_time_saved__prefetch"] += service_time(1, acc.num_chunks) - added_chunks
        assert acc.ts > self.ts_range[1]
        self.ts_range[1] = acc.ts
        self.num_accesses += 1

    def add_features(self, ts, feat):
        if len(self.features) < 6:
            self.features[ts] = feat

    def compute(self):
        # Call when finalizing episode
        self.score = score_dt_size(self)

    def get_examples(self):
        return list(self.features.values()), len(self.features) * [self.score]


def score_dt_size(eps: TrainingEpisode) -> float:
    return utils.safe_div(eps.s["service_time_saved__prefetch"], eps.num_chunks)


class GBTrainer(object):
    def __init__(self):
        # Ordered by LRU
        self.eps_in_progress = OrderedDict()
        self.sample_in_progress_eps = 0.1
        self.min_eps_for_training = 100

        # Selected episodes for training
        # Currently only those that are finalized/complete
        self.eps_for_training = []
        self.X = []
        self.Y = []

        self.init_config()

    def add_examples(self, batch, ts):
        # Called for every miss
        for key, feats in batch.items():
            # print(key, self.eps_in_progress.keys())
            block = key[0]
            # TODO: Sample 1/n of the chunks, and take it off the AP path
            if block in self.eps_in_progress:
                # print("Adding features")
                self.eps_in_progress[block].add_features(ts, feats)

    def update_labels(self, key, acc: AccessPlus, *, cache=None, **kwargs):
        # Update stats of episode for current key (hits, DT saved, segment range)
        if key not in self.eps_in_progress:
            # Start recording an episode
            self.eps_in_progress[key] = TrainingEpisode(acc)
            # Potentially set is as None as if we don't want to collect everything
        else:
            eps = self.eps_in_progress[key]
            if eps is not None:
                eps.add_access(acc)
            # TODO: Consider deduplicating metadata -- #hits already recorded for cached items.
            self.eps_in_progress.move_to_end(key)

        # TODO: Add features here
        # featvec = cache.collect_features(k, acc)

        # 'Calculated' eviction based on: if time to last access > current eviction age
        # TODO: Get a more recent sample, rather than the long term average
        assumed_eviction_age = cache.computeEvictionAge()
        if assumed_eviction_age is None or assumed_eviction_age == 0:
            assumed_eviction_age = 3600 * 2

        # As a start: do all based on episodes analysis
        # Process order and use last access time
        while self.eps_in_progress:
            least_recent = next(iter(self.eps_in_progress.values()))
            # TODO: If admitted, wait for actual eviction? Otherwise (below)
            if (acc.ts - least_recent.ts_range[1]).physical > assumed_eviction_age:
                self.eps_in_progress.popitem(last=False)
                # Finalize episode.
                least_recent.compute()
                # Add to training episodes
                self.eps_for_training.append(least_recent)
            else:
                break

    def on_evict(self, key, ts, _, **kwargs):
        # Real eviction
        if key in self.eps_in_progress:
            eps = self.eps_in_progress[key]
            eps.s["eviction_age_actual"] = ts - self.ts_range[1]
            eps.compute()
            self.eps_for_training.append(eps)
            del self.eps_in_progress[key]

    def init_config(self):
        self.iterations = 2000
        learnrate = 0.005
        featurefraction, threads = 0.9, 20
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
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

    def X_from_eps(self, eps_for_training_, threshold):
        X, scores = [], []
        eps_scores = []
        eps_sizes = []
        for eps in eps_for_training_:
            x_, scores_ = eps.get_examples()
            eps_scores.append(eps.score)
            eps_sizes.append(eps.num_chunks * 131072)
            assert len(x_) == len(scores_)
            X += x_
            scores += scores_
        X = np.array(X)
        # TODO: Weigh episodes by size/flash writes
        cutoff = np.percentile(eps_scores, threshold / 100)
        scores = np.array(self.scores)
        return X, scores, cutoff

    def Y_from_scores(self, scores, cutoff):
        Y = scores >= cutoff
        return Y

    def compute_data(self, threshold):
        # Sample some eps from in_progress -- to get those long-lived ones
        in_progress = []
        for eps in self.eps_in_progress.values():
            if random.random() < self.sample_in_progress_eps:
                eps.compute()
                in_progress.append(eps)

        eps_for_training_ = self.eps_for_training + in_progress

        # TODO: Which training examples to keep over time?

        if len(eps_for_training_) < self.min_eps_for_training:
            return False
        self.eps_train, self.eps_test = train_test_split(eps_for_training_, test_size=0.3, random_state=42)
        self.X_train, self.Y_scores_train, self.cutoff = self.X_from_eps(self.eps_train, threshold)
        self.X_test, self.Y_scores_test, self.cutoff_test = self.X_from_eps(self.eps_test, threshold)
        self.Y_train = self.Y_from_scores(self.Y_scores_train, self.cutoff)
        self.Y_test = self.Y_from_scores(self.Y_scores_test, self.cutoff)
        self.dsTrain = lgb.Dataset(self.X_train, self.Y_train)
        self.dsTest = lgb.Dataset(self.X_test, self.Y_test, reference=self.dsTrain)

        if len(self.X_train) < 100:
            return False

        # self.X, self.scores = [], []
        # eps_scores = []
        # eps_sizes = []
        # for eps in eps_for_training_:
        #     x_, scores_ = eps.get_examples()
        #     eps_scores.append(eps.score)
        #     assert len(x_) == len(scores_)
        #     self.X += x_
        #     self.scores += scores_
        # if len(self.X) < 100:
        #     return False
        # self.X = np.array(self.X)
        # # TODO: Have cutoff be write rate based.
        # self.cutoff = np.percentile(eps_scores, 50)
        # self.scores = np.array(self.scores)
        # print(self.X.shape)
        # self.Y = self.scores >= self.cutoff
        # self.dsTrain = lgb.Dataset(self.X, self.Y)
        # print(self.X[0])
        print(f"Training Baleen model with {len(eps_for_training_)} episodes ({len(self.eps_train)} done) - {len(self.X_train)} examples")
        print(f"Cutoff: {self.cutoff}")
        return True

    def reset_data(self, min_end_ts=None):
        self.eps_for_training = [ep for ep in self.eps_for_training
                                 if ep.ts_range[1].physical >= min_end_ts]

    def train(self, threshold):
        if not self.compute_data(threshold):
            return
        return lgb.train(self.params,
                         self.dsTrain,
                         num_boost_round=self.iterations,
                         valid_sets=self.dsTest,
                         verbose_eval=False,
                         early_stopping_rounds=25,
                         )


class LocalMLAP(NewMLAP):
    """
    Collects training samples locally and trains a local model.

    Need to determine right threshold to achieve write rate.
    Need to know what labels are by only looking back.
    Labels: episode DT saved/size > score_cutoff [based on WR]
    """
    def __init__(self, *,
                 threshold=None,
                 retrain_interval_hrs=6,
                 train_history_hrs=24,
                 **kwargs):
        self.threshold = threshold
        assert threshold is not None
        self.retrain_interval_hrs = retrain_interval_hrs
        self.train_history_hrs = train_history_hrs
        self.gbm = None
        self.trainer = GBTrainer()
        self.hooks = {
            # "every_chunk_before_insert": [self.on_every_chunk_before_insert],
            "every_acc_before_insert": [self.on_every_acc_before_insert],
            "evict": [self.trainer.on_evict],
        }
        # self.features = 'dfeat+meta'
        self.features = 'meta+block+chunk'
        # TODO: Make fallback RejectX
        self.fallback = AcceptAll()
        self.ts_last_trained = None

    # def on_every_chunk_before_insert(self, key, ts, *, ram_cache=None, **kwargs):
    #     self.trainer.update_labels(key, ts)

    def retrain(self, acc):
        gbm = self.trainer.train(self.threshold)
        if gbm is not None:
            print("Retrained model")
            self.gbm = gbm
            self.ts_last_trained = acc.ts
            self.trainer.reset_data(acc.ts.physical - 3600 * self.train_history_hrs)
        return gbm is not None

    def on_every_acc_before_insert(self, acc: AccessPlus, **kwargs) -> None:
        self.trainer.update_labels(acc.block_id, acc, **kwargs)
        if self.gbm is None and len(self.trainer.eps_for_training) > 100:
            self.retrain(acc)
            if self.gbm is not None:
                print("Trained initial model")
        elif self.gbm is not None:
            # Check criteria for retraining
            if (acc.ts - self.ts_last_trained).physical > self.retrain_interval_hrs * 3600:
                self.retrain(acc)

    def batchAccept(self, batch, ts, metadata=None, check_only=False):
        # print("AP call: ", check_only)
        if not check_only:
            self.trainer.add_examples(batch, ts)
        if self.gbm is None:
            decisions = self.fallback.batchAccept(batch, ts, metadata=metadata)
            if not check_only:
                self.count_decisions(decisions)
            return decisions
        return super().batchAccept(batch, ts, metadata=metadata)


class HybridAP(AP):
    def __init__(self, aps, threshold, seed=1, which_ts='episode'):
        self.aps = aps
        self.threshold = threshold
        self.seed = seed
        self.ts = which_ts
        self.features = 'dfeat+meta'
        assert len(aps) == 2

    def split(self, key, ts, metadata=None, **kwargs):
        if self.ts == 'episode':
            episode = metadata['episode'][key]
            ts_hash = episode.ts_logical[0]
        else:
            if metadata is None:
                ts_inserted = ts
            else:
                ts_inserted = metadata['ts'][key]
            ts_hash = ts_inserted
        h = spookyhash.hash64(bytes(f"{key[0]}|{ts_hash+1}", "ascii"), seed=self.seed)
        hf = h / ((1 << 64) - 1)
        result = hf < self.threshold
        return 0 if result else 1

    def batchAccept(self, batch, ts, *, metadata=None, check_only=False):
        locations = {k: self.split(k, ts, metadata=metadata) for k in batch}
        batches = [{}, {}]
        results = {0: {}, 1: {}}
        decisions = {}
        for k, v in locations.items():
            batches[v][k] = batch[k]
        for i, ap in enumerate(self.aps):
            ods.bump(f"ap_hybrid_to_{i}_{ap.name}", v=len(batches[i]))
        for i in [0, 1]:
            if batches[i]:
                results[i] = self.aps[i].batchAccept(batches[i], ts, metadata=metadata)
                decisions.update(results[i])
        assert len(decisions) == len(batch)
        if not check_only:
            self.count_decisions(decisions)
        return decisions


class EitherAP(AP):
    def __init__(self, aps):
        self.aps = aps
        self.features = 'dfeat+meta'

    def batchAccept(self, batch, ts, *, metadata=None, check_only=False):
        results = {}
        decisions = {}
        for i, ap in enumerate(self.aps):
            results[i] = ap.batchAccept(batch, ts, metadata=metadata)
        for k in batch:
            decisions[k] = False
            for i, ap in enumerate(self.aps):
                if results[i][k]:
                    ods.bump(f"ap_either_acceptby_{i}_{ap.name}")
                    decisions[k] = True
                    break
        assert len(decisions) == len(batch)
        if not check_only:
            self.count_decisions(decisions)
        return decisions


class AndAP(EitherAP):
    def batchAccept(self, batch, ts, *, metadata=None, check_only=False):
        results = {}
        decisions = {}
        for i, ap in enumerate(self.aps):
            results[i] = ap.batchAccept(batch, ts, metadata=metadata)
        for k in batch:
            decisions[k] = True
            for i, ap in enumerate(self.aps):
                if not results[i][k]:
                    ods.bump(f"ap_all_rejectby_{i}_{ap.name}")
                    decisions[k] = False
                    break
        assert len(decisions) == len(batch)
        if not check_only:
            self.count_decisions(decisions)
        return decisions


# class CachedLearnedSizeAP(LearnedSizeAP):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.cache = {}

#     @methodtools.lru_cache(maxsize=1024*1024)
#     def _predict_cached(self, feat):
#         ods.bump("ml_batches")
#         ods.bump("ml_predictions", v=len(feat))
#         return self.gbm.predict(feat)

#     def _predict(self, batch, ts):
#         assert len(batch) == 1
#         Xs = list(batch.values())
#         features = np.array(Xs)
#         for x in Xs:
#             feat = tuple(x)
#             self.seen_before[feat] += 1
#             if feat in self.cache:
#                 return self.cache[feat]
#         # return self._predict_cached(features)
#         result = self.gbm.predict(features)
#         ods.bump("ml_batches")
#         ods.bump("ml_predictions", v=len(features))
#         self.cache[feat] = result
#         return result


# class CacheLearnedAP(AP):
#     def __init__(self, threshold, model_path):
#         assert model_path
#         self.threshold = threshold
#         self.gbm = lgb.Booster(model_file=model_path)
#         self.seen_before = Counter()

#     @methodtools.lru_cache(maxsize=1024*1024)
#     def _predict(self, features):
#         return self.gbm.predict(features)

#     def accept(self, key, ts):


class OfflineAP(AP):
    def __init__(self, decisions, threshold, flip_threshold=True):
        super().__init__()
        self.threshold = threshold
        self.flip_threshold = flip_threshold
        self.decisions = decisions
        assert self.flip_threshold

    def accept(self, key, ts, metadata=None):
        block_id, chunk_id = key
        if metadata is None:
            ts_inserted = ts
        else:
            ts_inserted = metadata['ts'][key]
        episode = metadata['episode'][key]
        if episode is None:
            print(f'Error: Episode not found: Block: {block_id}, Chunk: {chunk_id}, TS: {ts_inserted}')
            return False
        return episode.threshold <= self.threshold

    @property
    def name(self):
        return "Offline-AP"

    def __repr__(self):
        return f"{self.name}({self.threshold})"


class OfflinePlus(AP):
    def __init__(self, threshold, *, only_used_chunks=True, check_future_use=True):
        super().__init__()
        self.threshold = threshold
        self.only_used_chunks = only_used_chunks
        self.check_future_use = check_future_use

    def accept(self, key, ts, metadata=None):
        block_id, chunk_id = key
        ts_inserted = metadata['ts'][key]
        episode = metadata['episode'][key]
        if episode is None:
            print(f'Error: Episode not found: Block: {block_id}, Chunk: {chunk_id}, TS: {ts_inserted}')
            return False
        if not (episode.threshold <= self.threshold):
            return False
        assert episode.chunk_last_seen
        if chunk_id not in episode.chunk_last_seen:
            if self.only_used_chunks:
                return False
        elif self.check_future_use and ts.physical >= episode.chunk_last_seen[chunk_id][0]:
            return False
        # if episode.s["sim_chunk_written"][chunk_id] > 0:
        #     ods.bump("attempted_readmission")
        #     return False
        return True

    @property
    def name(self):
        return "OfflinePlus-AP"


# class OfflineChunkAP(OfflineAP):
#     def accept(self, key, ts, metadata=None):
#         block_id, chunk_id = key
#         block_id, chunk_id = key
#         assert self.flip_threshold
#         if metadata is None:
#             ts_inserted = ts
#         else:
#             ts_inserted = metadata['ts'][key]
#         episode = _lookup_episode(self.decisions, block_id, ts_inserted, chunk_id=chunk_id)
#         if episode is None:
#             print(f'Error: Episode not found: Block: {block_id}, Chunk: {chunk_id}, TS: {ts_inserted}')
#             return False
#         return episode.threshold <= self.threshold


# Rejects writes based on the amount of writes admitted so far. Computes the
# current write rate since begin and rejects based on the expected write rate.
# Assumes that the callee uses this as the leaf admission policy
class WriteRateRejectAP(AP):
    def __init__(self, write_mbps, val_size):
        super().__init__()
        self.expected_rate = write_mbps
        self.bytes_written = 0
        self.start_ts = 0
        self.val_size = val_size

    def accept(self, k, ts, check_only=False):
        if self.expected_rate == 0:
            return True

        if self.start_ts == 0:
            self.start_ts = ts
        delta = float(ts - self.start_ts)
        assert delta >= 0
        if delta == 0:
            return False

        write_rate = (self.bytes_written + self.val_size) / delta
        if write_rate > self.expected_rate * 1024.0 * 1024.0:
            return False

        if not check_only:
            self.bytes_written += self.val_size
        return True

    def batchAccept(self, batch, ts, *, check_only=False):
        decisions = {}
        for key in batch:
            decisions[key] = self.accept(key, None, None, ts, check_only=check_only)
        return decisions

    @property
    def name(self):
        return "WriteRateReject"


class RejectFirstWriteRateAP(AP):
    def __init__(self, window_count, write_mbps, val_size):
        super().__init__()
        self.reject_first_ap = RejectXAP(1, window_count)
        self.write_rate_ap = WriteRateRejectAP(write_mbps, val_size)

    def accept(self, k, ts, check_only=False):
        return self.reject_first_ap.accept(k, ts, check_only=check_only) and self.write_rate_ap.accept(k, ts, check_only=check_only)

    def batchAccept(self, batch, ts, check_only=False):
        decisions = {}
        for key in batch:
            decisions[key] = self.accept(key, ts, check_only=check_only)
        return decisions

    @property
    def name(self):
        return "RejectFirstWriteRate"


def construct(ap_id, options, threshold=None, **kwargs):
    ap = None
    if threshold is None:
        threshold = options.ap_threshold
    scaled_write_mbps = float(options.write_mbps) * float(kwargs['sample_ratio']) / 100.0
    if ap_id == "rejectx":
        if threshold is None:
            threshold = 1
        factor = 2
        if options.ap_probability:
            factor = options.ap_probability
        if scaled_write_mbps == 0:
            ap = RejectXAP(threshold, factor * kwargs['num_cache_elems'], factor=factor)
        else:
            ap = RejectFirstWriteRateAP(
                2 * kwargs['num_cache_elems'], scaled_write_mbps, utils.BlkAccess.ALIGNMENT
            )
    elif ap_id == "optplus":
        assert threshold
        ap_kwargs = utils.arg_to_dict(options.optplus_args)
        ap = OfflinePlus(threshold, **ap_kwargs)
        print(f"{ap.name} AP (threshold: {threshold}, {ap_kwargs})")
    elif ap_id == "hybrid":
        assert options.hybrid_ap_threshold is not None
        aps = [
            construct("ml", options, threshold=options.ap_threshold, **kwargs),
            construct("opt", options, threshold=options.opt_ap_threshold, **kwargs),
        ]
        ap = HybridAP(aps, options.hybrid_ap_threshold)
    elif ap_id == "either_mlrejectx":
        aps = [
            construct("ml", options, threshold=options.ap_threshold, **kwargs),
            construct("rejectx", options, threshold=options.rejectx_ap_threshold, **kwargs),
        ]
        ap = EitherAP(aps)
    elif ap_id == "either_mlopt":
        aps = [
            construct("ml", options, threshold=options.ap_threshold, **kwargs),
            construct("opt", options, threshold=options.opt_ap_threshold, **kwargs),
        ]
        ap = EitherAP(aps)
    elif ap_id == "either_optml":
        aps = [
            construct("opt", options, threshold=options.opt_ap_threshold, **kwargs),
            construct("ml", options, threshold=options.ap_threshold, **kwargs),
        ]
        ap = EitherAP(aps)
    elif ap_id == "and_mlnewopt":
        aps = [
            construct("mlnew", options, threshold=options.ap_threshold, **kwargs),
            construct("opt", options, threshold=options.opt_ap_threshold, **kwargs),
        ]
        ap = AndAP(aps)
    elif ap_id == "mlnew":
        assert options.learned_ap_model_path and threshold, (options, threshold)
        kwargs_ = {}
        if options.ap_feat_subset:
            kwargs_['feat_subset'] = options.ap_feat_subset
        ap = NewMLAP(threshold, model_path=options.learned_ap_model_path, **kwargs_)
        print(f"{ap.name} with model: {options.learned_ap_model_path} threshold: {threshold}")
    elif ap_id == "mlonline":
        # assert threshold, (options, threshold)
        kwargs_ = {}
        if options.ap_feat_subset:
            kwargs_['feat_subset'] = options.ap_feat_subset
        if options.retrain_interval_hrs:
            kwargs_['retrain_interval_hrs'] = options.retrain_interval_hrs
        if options.train_history_hrs:
            kwargs_['train_history_hrs'] = options.train_history_hrs
        ap = LocalMLAP(threshold=threshold, **kwargs_)
        print(f"{ap.name} with threshold: {threshold}, retrain every {ap.retrain_interval_hrs} hrs on last {ap.train_history_hrs} hrs")
    elif ap_id == "ml":
        assert options.learned_ap_model_path and threshold, (options, threshold)
        if options.learned_size:
            # ap = CachedLearnedSizeAP(options.ap_threshold, model_path=options.learned_ap_model_path)
            ap = LearnedSizeAP(threshold,
                               model_path=options.learned_ap_model_path,
                               size_opt=options.size_opt)
        else:
            ap = LearnedAP(threshold, model_path=options.learned_ap_model_path)
        print(f"{ap.name} with model: {options.learned_ap_model_path} threshold: {threshold}")
    elif ap_id == "flashield":
        assert threshold, (threshold, options)
        ap = FlashieldAP(threshold=threshold)
        print(f"Flashield AP (threshold: {threshold})")
    elif ap_id == "flashieldprob":
        assert options.flashieldprob_ap_min_hits and threshold, (threshold, options)
        ap = FlashieldProbAP(threshold=threshold, n=options.flashieldprob_ap_min_hits)
        print(f"FlashieldProb AP (threshold: {threshold}, n: {options.flashieldprob_ap_min_hits})")
    elif ap_id == "tinylfu":
        # window_frac = 0.01
        assert threshold, (threshold, options)
        window_frac = threshold
        ap = TinyLFUAP(window_frac=window_frac, cache_size=kwargs['num_cache_elems'])
        print(f"TinyLFU AP (w_frac: {window_frac}, w_size: {int(window_frac * kwargs['num_cache_elems'])})")
    elif ap_id == "opt":
        assert options.offline_ap_decisions and threshold
        ap = OfflineAP(kwargs['episodes'], threshold,
                       flip_threshold=options.flip_threshold)
        print(f"Offline AP (threshold: {threshold})")
    elif ap_id == "coinflip":
        assert options.ap_probability
        ap = CoinFlipDetAP(options.ap_probability)
        print("CoinFlipDet AP with probability:", options.ap_probability)
    elif ap_id == 'wrreject':
        assert scaled_write_mbps != 0
        ap = WriteRateRejectAP(scaled_write_mbps, utils.BlkAccess.ALIGNMENT)
    elif ap_id == 'acceptall':
        ap = AcceptAll()
    else:
        raise NotImplementedError(ap_id)
    return ap
