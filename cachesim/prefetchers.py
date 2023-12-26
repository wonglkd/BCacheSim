import sys

import numpy as np
try:
    import lightgbm as lgb
except ModuleNotFoundError:
    print("Unable to import lightgbm")

from . import utils
from .utils import ods
from ..episodic_analysis.episodes import offset_to_chunks
from .ep_helpers import _get_chunks_for_episode
from .ep_helpers import _prefetchable_chunks
from .ep_helpers import AccessPlus


class Prefetcher(object):
    def __init__(self, *, options=None):
        self.pf_when = options.prefetch_when
        self.pf_range = options.prefetch_range
        self.model = None
        self.cache = None
        self.ram_cache = None
        self.assumed_ea = None
        if self.enabled:
            print(f"Prefetcher(when={self.pf_when}, range={self.pf_range})")
        if 'episode-predict' in self.pf_range or 'predict' in self.pf_when:
            if self.pf_when == 'predict':
                assert options.prefetch_when_threshold
                self.model = LearnedRangeConfPrefetcherModel(
                    options.prefetcher_model_path,
                    threshold=options.prefetch_when_threshold)
                print(f"LearnedRangeConfPrefetcherModel(threshold={options.prefetch_when_threshold}, path={options.prefetcher_model_path})")
            else:
                self.model = LearnedRangePrefetcherModel(
                    options.prefetcher_model_path)
                print(f"LearnedRangePrefetcherModel({options.prefetcher_model_path})")

    def __repr__(self):
        desc = ""
        if 'predict' in self.pf_when:
            desc += f", th={self.model.threshold}"
        return f"Prefetcher(when={self.pf_when}, range={self.pf_range}{desc})"

    @property
    def enabled(self):
        return self.pf_when != 'never'

    def set_cache(self, *, cache=None, ram_cache=None, insert_cache=None, ap=None):
        self.cache = cache
        self.ram_cache = ram_cache
        self.insert_cache = insert_cache if insert_cache else (ram_cache if ram_cache else cache)
        self.ap = ap

    def decide(self, block_id, is_hit, *args, **kwargs):
        if self.pf_when == 'never' or is_hit:
            return False
        result = self.decide_(block_id, is_hit, *args, **kwargs)
        if result:
            ods.bump("prefetch_when_accepts")
        else:
            ods.bump("prefetch_when_rejects")
        return result

    def decide_(self, block_id, is_hit, chunk_hit, episode, pred_prefetch, acc_ts):
        if self.pf_when.startswith('rejectfirst'):
            in_cache = block_id in self.cache.cached_episodes
            iops_misses = 0
            if in_cache:
                iops_misses += self.cache.cached_episodes[block_id]["iops_misses"]
            if self.pf_when == 'rejectfirst-either':
                in_cache = in_cache or block_id in self.ram_cache.cached_episodes
                if block_id in self.ram_cache.cached_episodes:
                    iops_misses += self.ram_cache.cached_episodes[block_id]["iops_misses"]
            if not in_cache:
                ods.bump("prefetch_rejectfirst_ep_notfound")
                return False
                ods.bump("prefetch_rejectfirst_misses_zero")
                return False
        elif self.pf_when == 'partial':
            # Fetch when we have Partial IOPS.
            if not (not is_hit and chunk_hit):
                return False
        elif self.pf_when == 'benefit':
            return episode.s_export['prefetch_st_benefit'] > 0
        elif self.pf_when == 'at_start':
            return episode.ts_physical[0] == acc_ts.physical
        elif self.pf_when == 'predict':
            chks, predict_stats = pred_prefetch
            return predict_stats['prob'] >= self.model.threshold
        else:
            assert self.pf_when in ('always')
        return True

    def get_chunks(self, block_id, acc_ts, episode, pred_prefetch, size, chunk_range):
        cache = self.cache
        chks = []
        metadata_chks = None
        if self.pf_range.startswith('chunk') or self.pf_range.startswith('acctime-'):
            # TODO: does size really makes sense here?
            metadata_init = {'size': size, 'prefetch': True, 'ts': acc_ts,
                             'acc_chunk_range': chunk_range, 'episode': episode}
            if self.pf_range == 'chunk':
                chks = _get_chunks_for_episode(cache.episodes, block_id, acc_ts)
            elif self.pf_range == 'chunk2':
                chks, chk_episodes = _prefetchable_chunks(cache.episodes, block_id, acc_ts, assumed_ea=self.assumed_ea)
                metadata_chks = {}
                for chk in chks:
                    metadata_chks[chk] = dict(metadata_init)
                    metadata_chks[chk]['episode'] = chk_episodes[chk]
            elif self.pf_range == 'acctime-episode-predict':
                chks, predict_stats = pred_prefetch
                if "--fast" not in sys.argv:
                    # WANTS: episode
                    if episode:
                        cache.bump_counter("loss_prefetch_start", predict_stats["chunk_r"][0] - episode.chunk_range[0])
                        cache.bump_counter("loss_prefetch_end", predict_stats["chunk_r"][1] - episode.chunk_range[1])
            else:
                # REQUIRES: episode
                prefetch_size = utils.BlkAccess.MAX_BLOCK_SIZE
                if episode:
                    metadata_init['at_ep_start'] = episode.ts_physical[0] == acc_ts.physical
                    prefetch_size = max(prefetch_size, episode.size)
                if self.pf_range == 'acctime-all':
                    acc_ = utils.BlkAccess(0, prefetch_size, acc_ts.physical, block=block_id)
                    chks = acc_.chunks()
                else:
                    chks = range(*episode.chunk_range)
        if metadata_chks is None:
            metadata_chks = {chk: dict(metadata_init) for chk in chks}
        return chks, metadata_chks

    def filter_existing(self, chks, misses, block_id):
        filtered_chks = []
        for chunk_id in chks:
            if chunk_id in misses:
                self.insert_cache.bump("prefetches_failed_inmiss")
                self.insert_cache.prefetches_failed_exists += 1
                continue
            k = (block_id, chunk_id)

            ram_cache_found = self.ram_cache and k in self.ram_cache.cache
            flash_cache_found = k in self.cache.cache
            if not ram_cache_found and not flash_cache_found:
                filtered_chks.append(chunk_id)
            else:
                self.insert_cache.bump("prefetches_failed_exists_incache")
                self.insert_cache.prefetches_failed_exists += 1
        return filtered_chks

    def filter_ap(self, chks, acc, metadata):
        admit_buffer_ = {}
        metadata_ = {kk: {} for kk in metadata}
        for chk in chks:
            k = (acc.block_id, chk)
            admit_buffer_[k] = self.cache.collect_features(k, acc)
            for kk in metadata[chk]:
                if kk not in metadata_:
                    metadata_[kk] = {}
                metadata_[kk][k] = metadata[chk][kk]
        need_prefetch = []
        decisions = self.ap.batchAccept(admit_buffer_, acc.ts, metadata=metadata_, check_only=True)
        for nkey, dec in decisions.items():
            if dec:
                need_prefetch.append(nkey[1])
        ods.bump("prefetch_ap_rejects", v=len(chks) - len(need_prefetch))
        return need_prefetch

    def run(self,
            acc: AccessPlus,
            is_hit,
            chunk_hit,
            episode,
            misses,
            size):
        need_prefetch = []
        if self.decide(acc.block_id, is_hit, chunk_hit, episode, acc.pred_prefetch, acc.ts):
            chks, metadata_chks = self.get_chunks(acc.block_id, acc.ts, episode, acc.pred_prefetch, size, acc.chunk_range)
            chks = self.filter_existing(chks, misses, acc.block_id)
            if chks:
                need_prefetch = self.filter_ap(chks, acc, metadata_chks)
                for chunk_id in need_prefetch:
                    k = (acc.block_id, chunk_id)
                    featvec = self.cache.collect_features(k, acc)
                    self.insert_cache.insert(k, acc.ts, featvec, metadata=metadata_chks[chunk_id])
        return need_prefetch


class PrefetcherModel(object):
    def predict(self, features):
        raise NotImplementedError


class LearnedRangePrefetcherModel(PrefetcherModel):
    def __init__(self, path, keys=["offset_start", "offset_end", "size"]):
        self.keys = keys
        # print("Prefetcher Model Path:  " + path)
        self.models = {k: lgb.Booster(model_file=path.format(k=k))
                       for k in self.keys}

    def predict(self, features, metadata=None):
        metadata = metadata or {}
        preds = {k: self.models[k].predict(features)[0].astype(int) for k in self.keys}
        ods.bump("ml_batches", v=len(self.keys))
        ods.bump("ml_predictions", v=len(self.keys))
        start = max(0, preds["offset_start"])
        end = max(0, preds["offset_end"], max(0, preds["size"])+preds["offset_start"])
        start = max(0, utils.BlkAccess.roundDownToBlockBegin(start))
        end = utils.BlkAccess.roundUpToBlockEnd(end)
        if not (0 <= start <= end):
            ods.bump("prefetch_predict_out_of_range")
            return [], {}
        assert 0 <= start <= end, (start, end, preds)

        if "--log-prefetch" in sys.argv:
            print(f"PREFETCH_KEY {metadata['block_id']} {metadata['ts'].logical+1}")
            print(f"PREFETCH_FEAT {features.flatten().tolist()}")
            print(f"PREFETCH {start} {end} ({preds['offset_start']} {preds['offset_end']} {preds['size']})")

        chunk_r = offset_to_chunks(start, end)
        return list(range(*chunk_r)), {"chunk_r": chunk_r}

    def predict_batch(self, features):
        preds = {k: self.models[k].predict(features) for k in self.keys}
        ods.bump("ml_batches", v=len(self.keys))
        ods.bump("ml_predictions", v=len(features) * len(self.keys))
        starts = np.maximum(0, preds["offset_start"])
        ends = np.maximum(0, preds["offset_end"])
        sizes = np.maximum(0, preds["size"])
        ends = np.maximum(ends, sizes+starts)
        starts = [utils.BlkAccess.roundDownToBlockBegin(int(s)) for s in starts]
        ends = [utils.BlkAccess.roundUpToBlockEnd(int(e)) for e in ends]
        ranges = list(zip(starts, ends))
        for s, e in ranges:
            assert 0 <= s <= e
        chunks_rs = [offset_to_chunks(start, end)
                     for start, end in ranges]
        return [(list(range(*cr)), {"chunk_r": cr}) for cr in chunks_rs]


class LearnedRangeConfPrefetcherModel(LearnedRangePrefetcherModel):
    def __init__(self, path, *, e_keys=["pred_net_pf_st_binary"], threshold=None, **kwargs):
        self.e_keys = e_keys
        self.threshold = threshold
        super().__init__(path, **kwargs)
        for k in self.e_keys:
            self.models[k] = lgb.Booster(model_file=path.format(k=k))

    def predict(self, features, metadata=None):
        metadata = metadata or {}
        chunks, stats = super().predict(features, metdata=metadata)
        # TODO: Check.
        features_with_preds = np.append(features, [stats["chunk_r"]], 1)
        # Add to X
        preds = {k: self.models[k].predict(features_with_preds)
                 for k in self.e_keys}
        ods.bump("ml_batches", v=len(self.e_keys))
        ods.bump("ml_predictions", v=len(self.e_keys))
        stats["prob"] = preds["pred_net_pf_st_binary"][0]
        return chunks, stats

    def predict_batch(self, features):
        results = super().predict_batch(features)
        xs = []
        for _, stats in results:
            xs.append(stats["chunk_r"])
        features_with_preds = np.append(features, np.array(xs), 1)
        preds_ = {k: self.models[k].predict(features_with_preds)
                  for k in self.e_keys}
        ods.bump("ml_batches", v=len(self.e_keys))
        ods.bump("ml_predictions", v=len(features) * len(self.e_keys))
        for i, (_, stats) in enumerate(results):
            stats["prob"] = preds_["pred_net_pf_st_binary"][i]

        return results
