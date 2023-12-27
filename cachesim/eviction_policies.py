from collections import Counter
from collections import defaultdict
from collections import OrderedDict
import pqdict
import itertools
import sys

import numpy as np
try:
    import lightgbm as lgb
except ModuleNotFoundError:
    print("Unable to import lightgbm")

from . import admission_policies as aps
from . import sim_features
from . import utils
from .utils import LOG_DEBUG, LOG_REQ, ods

from .ep_helpers import _get_chunks_for_episode
from .ep_helpers import _lookup_episode
from .ep_helpers import Timestamp


class QueueItem(object):
    def __init__(self, ts, key):
        self.last_access_time = ts
        self.admission_time = ts
        self.hits = 0
        self.key = key

    def touch(self, ts):
        self.last_access_time = ts

    def markAccessed(self, ts):
        self.touch(ts)
        self.hits += 1


class QueueItemWithStats(QueueItem):
    def __init__(self, ts, key, ts_access=None, episode=None, **kwargs):
        super().__init__(ts, key)
        self.ts_access = ts_access
        self.max_interarrival_time = Timestamp(0, 0)
        self.group = None
        self.stats = kwargs
        self.episode = episode

    @property
    def all_hits(self):
        return self.hits + self.stats.get('ramcache_hits', 0)

    def markAccessed(self, ts):
        reuse_dist = ts - self.last_access_time
        super().markAccessed(ts)
        self.max_interarrival_time = max(self.max_interarrival_time, reuse_dist)


class LIRSItem(QueueItemWithStats):
    def __init__(self, ts, key):
        super().__init__(ts, key)
        self.is_bait = True

    def upgradeFromBait(self, ts):
        assert self.is_bait
        if self.is_bait:
            self.is_bait = False
            self.hits = 0
            self.last_access_time = ts

    def isBait(self):
        return self.is_bait

    def isNotBait(self):
        return not self.is_bait


class TTLItem(QueueItemWithStats):
    def __init__(self, ts, *args, ttl=None, **kwargs):
        self.ttl = ttl
        if ttl is not None:
            self.ts_expire = ts.physical + ttl
        super().__init__(ts, *args, **kwargs)

    def markAccessed(self, ts):
        super().markAccessed(ts)
        if self.ttl is not None:
            self.ts_expire = ts.physical + self.ttl


class EvictionImpl(object):
    def keys(self):
        return self.items.keys()

    def values(self):
        return self.items.values()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key):
        return self.items[key]

    def __contains__(self, key):
        return key in self.items

    def victim(self):
        """The next item that will be evicted."""
        raise NotImplementedError


class TTLPolicy(EvictionImpl):
    """Evicts item with lowest TTL"""
    def __init__(self):
        self.pqueue = pqdict.pqdict()
        self.items = {}

    def admit(self, key, item):
        self.items[key] = item
        self.pqueue[key] = item.ts_expire

    def touch(self, key):
        """Update priority in pq"""
        if self.pqueue[key] != self.items[key].ts_expire:
            self.pqueue.updateitem(key, self.items[key].ts_expire)

    def evict(self, key):
        if key is not None:
            del self.pqueue[key]
        else:
            key, _ = self.pqueue.popitem()
        return key, self.items.pop(key)

    def victim(self):
        if len(self.pqueue) > 0:
            return self.pqueue.top()
        return None


class LRUPolicy(EvictionImpl):
    def __init__(self):
        self.items = OrderedDict()

    def admit(self, key, item):
        self.items[key] = item

    def touch(self, key):
        # Moves to end. We evict from start.
        self.items.move_to_end(key)

    def evict(self, key=None):
        # pop in FIFO order, from front
        if key:
            self.items.move_to_end(key, last=False)
        return self.items.popitem(last=False)

    def victim(self):
        if len(self.items) == 0:
            return None
        return next(iter(self.items))


class TTLModel(object):
    def __init__(self, options):
        model_path = options.ttl_model_path
        self.models = {'ttl': lgb.Booster(model_file=model_path)}
        self.keys = ['ttl']

    def predict(self, features):
        # print(features[0].shape)
        features = [features]
        preds = {k: self.models[k].predict(features)[0].astype(int) for k in self.keys}
        ods.bump("ml_batches", v=len(self.keys))
        ods.bump("ml_predictions", v=len(self.keys))
        return preds["ttl"]


class TTLOpt(object):
    def predict(self, features, metadata=None):
        # use episodes.max_interarrival
        return metadata['episode'].max_interarrival[0]


class EvictionPolicy(object):
    def __init__(self, evictions_log, cache_size, *,
                 evict_by='chunk',
                 namespace='flashcache',
                 dynamic_features=None,
                 prefetch_range='episode',
                 prefetch_when='never',
                 options=None):
        self.cache_size = cache_size
        self.keys_written = 0
        self.rejections = 0
        self.evictions = 0
        self.eviction_age_cum = Timestamp(0, 0)
        self.un_accessed_evictions = 0
        self.un_accessed_eviction_age_cum = Timestamp(0, 0)
        self.max_interarrival_time_cum = Timestamp(0, 0)
        self.max_max_interarrival_time = Timestamp(0, 0)
        # dynamic features, which have to be update on each request
        self.dynamic_features = dynamic_features
        self.evictions_log = evictions_log
        header = ["block_id", "admission_time", "ts", "last_access_time",
                  "hits", "useful_time", "time_in_system",
                  "time_since_last_access", "max_interarrival_time",
                  "num_chunks", "chunks"]
        if evictions_log:
            evictions_log.write(",".join(header)+"\n")
        self.eviction_buffer = {}
        self.prefetches = 0
        self.prefetches_failed_firstaccess = 0
        self.prefetches_failed_exists = 0
        # self.prefetches_caught = 0
        # self.episodes_admitted = 0
        self.early_evictions = 0
        # self.episode_touches = 0
        self.evict_by = evict_by
        self.chunk_grp = {}

        self.admitted_buffer = defaultdict(list)
        self.prefetch_range = prefetch_range
        self.prefetch_when = prefetch_when
        self.admit_history_debug = {}
        self.namespace = namespace
        self.warmup_finished = None

    def bump(self, k, tags=[], **kwargs):
        ods.bump([self.namespace, k], **kwargs)
        tags = sorted(tags)
        for r in range(len(tags)):
            for comb in itertools.combinations(tags, r+1):
                k_all = k + "_" + "_".join(comb)
                ods.bump([self.namespace, k_all], **kwargs)

    def bump_counter(self, k, v, **kwargs):
        if "--fast" not in sys.argv:
            ods.bump_counter([self.namespace, k], v, **kwargs)

    def log_eviction(self, ts, evicted):
        # evicted: (key, item)
        if self.evictions == 0:
            ods.counters[f"{self.namespace}/warmup_finished"] = ts
            self.warmup_finished = ts
        self.evictions += 1
        self.bump("evictions")
        item = evicted[1]
        LOG_REQ(self.namespace, item.key, ts, "EVICT")
        # Time in system
        time_in_system = ts - item.admission_time
        useful_time = item.last_access_time - item.admission_time
        # Deadtime
        time_since_last_access = ts - item.last_access_time
        self.eviction_age_cum += time_since_last_access
        self.bump("eviction_age_cum", v=time_since_last_access, init=Timestamp(0, 0))
        self.bump_counter("eviction_age_dist", int(round(time_since_last_access.physical / 60)))
        self.bump_counter("eviction_age_dist_logical", int(round(time_since_last_access.logical / 500))*500)
        self.bump_counter("time_in_system_dist", int(round(time_in_system.physical / 60)))
        self.bump_counter("time_in_system_dist_logical", int(round(time_in_system.logical / 1000))*1000)
        self.bump_counter("hits_on_eviction", item.hits)
        self.bump("total_time_in_system", v=time_in_system, init=Timestamp(0, 0))
        block_id, chunk_id = item.key
        if item.hits == 0:
            self.un_accessed_evictions += 1
            self.bump("unaccessed_evictions")
            self.un_accessed_eviction_age_cum += time_since_last_access
            self.bump("unaccessed_eviction_age_cum", v=time_since_last_access, init=Timestamp(0, 0))

            if item.stats.get('prefetch', False) and item.stats.get('ramcache_hits', 0) == 0:
                self.bump("evicted_nohits_prefetch")
            # TODO: log

        if True or "--fast" not in sys.argv:
            extra_tags = []
            if item.all_hits == 0:
                extra_tags.append("nohitsatall")
            elif item.hits == 0:
                extra_tags.append("nohits")
            if item.stats.get('prefetch', False):
                extra_tags.append("prefetch")
            if item.stats.get('doomed', False):
                self.bump("evicted_doomed", tags=extra_tags)
                if item.hits != 0:
                    self.bump("evicted_doomed_withhits")
                extra_tags.append("doomed")

            ts_inserted = item.ts_access or item.admission_time
            episode = item.episode

            if episode and episode.chunk_last_seen:
                if chunk_id not in episode.chunk_last_seen:
                    self.bump("evicted_chunknotinepisode", tags=extra_tags)
                else:
                    if item.last_access_time.physical > episode.chunk_last_seen[chunk_id][0]:
                        if item.hits > 0:
                            extra_tags.append("hitsAfterEp")
                        else:
                            extra_tags.append("admittedAfterEp")
                    if ts.physical < episode.chunk_last_seen[chunk_id][0]:
                        self.bump("evicted_with_hits_remaining", tags=extra_tags)
                    else:
                        self.bump("evicted_without_hits_remaining", tags=extra_tags)
            elif episode and episode.chunk_level:
                if item.last_access_time.physical > episode.ts_physical[1]:
                    if item.hits > 0:
                        extra_tags.append("hitsAfterEp")
                    else:
                        extra_tags.append("admittedAfterEp")
                if ts.physical < episode.ts_physical[1]:
                    self.bump("evicted_with_hits_remaining", tags=extra_tags)
                else:
                    self.bump("evicted_without_hits_remaining", tags=extra_tags)
            else:
                self.bump("warning_evicted_episodenotfound")
                if item.ts_access is None:
                    self.bump("warning_evicted_episodenotfound_notsaccess")

            self.bump_counter("max_interarrival_time_dist_mins", int(round(item.max_interarrival_time.physical / 60)))
            self.bump_counter("max_interarrival_time_dist_logical", int(round(item.max_interarrival_time.logical / 500))*500)

        self.max_interarrival_time_cum += item.max_interarrival_time
        self.bump("max_interarrival_time_cum", v=item.max_interarrival_time, init=Timestamp(0, 0))
        self.max_max_interarrival_time = max(self.max_max_interarrival_time, item.max_interarrival_time)

    def on_before_access(self, block_id, access, ts):
        pass

    def on_access_end(self, ts, groups=[], access=None):
        # Early Admission.
        ts_key = ts.physical

        self.prefetch_admit_buffer(ts, access=access)

        if self.early_evict and ts_key in self.early_evict:
            for block_id in self.early_evict[ts_key]:
                for chunk_id in range(128):
                    key_to_evict = (block_id, chunk_id)
                    if key_to_evict in self.cache:
                        self.do_eviction(ts, key=key_to_evict)
                        self.early_evictions += 1
                        self.bump("early_evictions")
        # TODO: If scheduled for early eviction, do not admit.

        # Avoid readmissions during episodes
        if self.evict_by == 'episode':
            for block_id, acc in groups:
                for chunk_id in acc.chunks():
                    k = (block_id, chunk_id)
                    if k in self.cache and self.cache[k].last_access_time != ts:
                        self.find(k, ts, count_as_hit=False)
                        self.episode_touches += 1

    def prefetch_admit_buffer(self, ts, access=None):
        # Prefetch
        if self.episodes and self.prefetch_when != 'never' and not self.prefetch_range.startswith('chunk') and not self.prefetch_range.startswith('acctime'):
            # Obsolete.
            for block_id, items in self.admitted_buffer.items():
                if self.prefetch_when == 'rejectfirst':
                    eps_stats = self.cached_episodes[block_id]
                    if eps_stats["iops_misses"] == 0:
                        # TODO: assert not in block IDs
                        continue

                unique_ts = set(v[1] for v in items)
                episodes = []
                ts_to_episode = {}
                for ts_inserted in unique_ts:
                    episode = _lookup_episode(self.episodes, block_id, ts_inserted)
                    ts_to_episode[ts_inserted] = episode
                    if episode not in episodes:
                        episodes.append(episode)
                # self.episodes_admitted += len(episodes)
                self.bump("episodes_prefetched", v=len(episodes))
                if not episodes:
                    # print(items)
                    LOG_DEBUG(access)
                    LOG_DEBUG(ts)
                    LOG_DEBUG(self.admitted_buffer)
                    LOG_DEBUG(block_id, items)
                    print(len(items))
                    ts_inserted = items[0][1]
                    raise Exception(f'Error: Episode not found: Block: {block_id}, TS: {ts_inserted}')
                if self.prefetch_when == 'always_debug':
                    t1, t2 = [], []
                    # self.admit_history_debug += episodes
                    for episode in episodes:
                        t1.append([block_id, episode.offset[0], episode.size, ts.physical])
                        # print('>', block_id, episode.offset[0], episode.size, ts.physical)
                    if ts.physical not in self.prefetch:
                        # print("Readmission:", block_id, ts)
                        # assert len(episodes) == 1
                        # continue
                        pass
                    else:
                        for block_id, access_args in self.prefetch[ts.physical]:
                            t2.append([block_id] + list(access_args))
                        if sorted(t1) != sorted(t2):
                            print(ts, unique_ts)
                            print('<<')
                            print(t1)
                            print('>>')
                            print(t2)

                if len(episodes) > 1:
                    self.bump_counter("multiple_episodes_in_admit_buffer", len(episodes))

                if self.prefetch_range == 'accend-chunk':
                    for acc_ts in unique_ts:
                        episode_chunks = _get_chunks_for_episode(self.episodes, block_id, acc_ts)
                        chks = []
                        for chunk_id in episode_chunks:
                            k = (block_id, chunk_id)
                            if k not in self.cache and self.ap.accept(k, acc_ts):
                                chks.append(chunk_id)
                        self._admit_prefetch(chks, block_id, ts, access)
                elif self.prefetch_range == 'episode-predict':
                    fx = set()
                    for chunk_id, ts_inserted, features in items:
                        if len(features) == 6+5:
                            features = features[6:]
                        fx.add(tuple(features))
                    fx = list(fx)
                    if len(fx) > 1:
                        self.bump_counter("warning_multiple_features_in_admit_buffer", len(fx))
                    elif len(fx) == 0:
                        continue
                    try:
                        fx = np.array([fx[0]])
                        chks, stats = self.prefetcher.predict(fx)
                    except:
                        print(fx)
                        print(items)
                        raise
                    if chks:
                        self.bump_counter("loss_prefetch_start", stats["chunk_r"][0] - episodes[0].chunk_range[0])
                        self.bump_counter("loss_prefetch_end", stats["chunk_r"][1] - episodes[0].chunk_range[1])
                        self._admit_prefetch(chks, block_id, ts, access)
                    # collect stats on how much it differs from actual episode
                else:
                    for episode in episodes:
                        if self.prefetch_range == 'episode':
                            access = utils.BlkAccess(episode.offset[0], episode.size, ts.physical, block=episode.key)
                            assert episode.offset[0] >= 0
                            chks = access.chunks()
                            # assert episode.size <= utils.BlkAccess.MAX_BLOCK_SIZE
                        elif self.prefetch_range == 'all':
                            access = utils.BlkAccess(0, max(episode.size, utils.BlkAccess.MAX_BLOCK_SIZE), ts.physical, block=episode.key)
                            chks = access.chunks()
                        elif self.prefetch_range.startswith('threshold-'):
                            prefetch_chk_threshold = int(self.prefetch_range.replace('threshold-', ''))
                            chks = [chk for chk, cnt in episode.chunk_counts.items()
                                    if cnt >= prefetch_chk_threshold]

                        else:
                            raise NotImplementedError(f"Unknown prefetch_range option: {self.prefetch_range}")

                        self._admit_prefetch(chks, block_id, ts, access)
        self.admit_history_debug = dict(self.admitted_buffer)
        self.admitted_buffer.clear()

    def _admit_prefetch(self, chks, block_id, ts, access):
        # OBSOLETE.
        for chunk_id in chks:
            k = (block_id, chunk_id)
            if k not in self.cache:
                self.incr_episode(k, ts)
                self.admit(k, ts, prefetch=True)
                self.prefetches += 1
                self.bump("prefetches")
            elif self.cache[k].admission_time == ts:
                self.bump("prefetches_failed_firstaccess")
                self.prefetches_failed_firstaccess += 1
            else:
                self.bump("prefetches_failed_exists")
                self.prefetches_failed_exists += 1
            self.cache[k].group = access

    def computeEvictionAge(self):
        return utils.safe_div(self.eviction_age_cum, self.evictions)

    def computeNoHitEvictionAge(self):
        return utils.safe_div(self.un_accessed_eviction_age_cum, self.un_accessed_evictions)

    def computeMaxMaxInterarrivalTime(self):
        return max(max((item.max_interarrival_time for item in self.cache.values()),
                       default=Timestamp(0, 0)), self.max_max_interarrival_time)

    def computeAvgMaxInterarrivalTime(self):
        num = self.max_interarrival_time_cum + sum((item.max_interarrival_time for item in self.cache.values()),
                                                   start=Timestamp(0, 0))
        den = len(self.cache) + self.evictions
        return utils.safe_div(num, den)

    def computeAvgObjectSize(self):
        sizes = {}
        for block_id, chunk_id in self.cache:
            if block_id not in sizes:
                sizes[block_id] = 0
            sizes[block_id] += 1
        if not sizes:
            return 0
        return sum(sizes.values()) / len(sizes)

    def computeAvgMaxInterarrivalTimeEvicted(self):
        return utils.safe_div(self.max_interarrival_time_cum, self.evictions)


class LIRSCache(EvictionPolicy):
    def __init__(self, evictions_log, num_elems, bait_factor, ap):
        super().__init__(evictions_log, num_elems)
        self.cache = OrderedDict()

        # number of baits  in cache
        self.num_baits = 0

        # number of values in cache
        self.num_vals = 0

        # required ratio of baits to values
        self.bait_factor = bait_factor

        # to optimize bait eviction, we let more baits stay in cache and prune
        # once it reaches past this threshold.
        self.prune_ratio = 1.25

        self.ap = ap

    def str(self):
        return "size={} vals={} baits={}".format(
            len(self.cache), self.num_vals, self.num_baits
        )

    def find(self, key, key_ts):
        found = key in self.cache
        reuse_dist = 0
        is_bait = False
        if found:
            reuse_dist = key_ts - self.cache[key].last_access_time
            assert reuse_dist.logical >= 0
            is_bait = self.cache[key].isBait()
            self.cache[key].markAccessed(key_ts)
            self.cache.move_to_end(key)

        return found and not is_bait

    def insert(self, key, ts, keyfeaturelist=None):
        # ML ap is not supported by LIRS Cache at this point,
        # "keyfeaturelist" is just a placeholder here for api consistency
        # first time insertion introduces just the bait to track hotness
        if key not in self.cache:
            self.insert_bait(key, ts)
            return

        if not self.ap.accept(key, ts):
            self.rejections += 1
            return
        self.admit(key, ts)

    def admit(self, key, ts):
        self.cache[key].upgradeFromBait(ts)
        self.cache.move_to_end(key)
        self.num_baits -= 1
        self.num_vals += 1
        self.keys_written += 1

        if self.num_vals > self.cache_size:
            self.do_lirs_eviction(ts)

    def too_many_baits(self):
        return (
            self.num_baits > 100 and self.num_baits > self.bait_factor * self.num_vals
        )

    def should_prune_baits(self):
        return self.num_baits > 100 and (
            self.num_baits > self.bait_factor * self.prune_ratio * self.num_vals
        )

    # scan from the bottom of lru and evict baits if we are above the
    # threshold. This is done lazily to ammortize the cost of baits to O(1)
    def do_lirs_bait_eviction(self):
        baits_to_remove = []
        for k in self.cache.keys():
            if self.cache[k].isBait():
                baits_to_remove.append(k)
                self.num_baits -= 1

            if not self.too_many_baits():
                break

        for k in baits_to_remove:
            self.cache.pop(k)

    def insert_bait(self, key, key_ts):
        # insert the element as bait
        self.cache[key] = LIRSItem(key_ts, key)
        self.num_baits += 1

        if self.should_prune_baits():
            self.do_lirs_bait_eviction()

    def remove_baits_lru_end(self):
        # eliminate all baits in the end so that inserting values is O(1)
        baits_to_remove = []
        for k in self.cache.keys():
            if self.cache[k].isNotBait():
                break
            baits_to_remove.append(k)

        for k in baits_to_remove:
            self.num_baits -= 1
            self.cache.pop(k)

    def do_lirs_eviction(self, ts):
        self.remove_baits_lru_end()
        evicted = self.cache.popitem(last=False)

        # the bottom of the stack can not be a bait.
        assert evicted[1].isNotBait(), "{}->{}".format(evicted[0], evicted[1])

        self.evictions += 1
        self.num_vals -= 1
        time_since_last_access = ts - evicted[1].last_access_time
        self.eviction_age_cum += time_since_last_access
        if evicted[1].hits == 0:
            self.un_accessed_evictions += 1
            self.un_accessed_eviction_age_cum += time_since_last_access


class QueueCache(EvictionPolicy):
    def __init__(self, evictions_log, num_elems, ap, *,
                 lru=True,
                 offline_feat_df=None,
                 episodes=None,
                 batch_size=1,
                 prefetcher=None,
                 on_evict=None,
                 keep_metadata=False,
                 options=None,
                 **kwargs):
        super().__init__(evictions_log, num_elems, options=options, **kwargs)
        early_evict = options.early_evict
        prefetch = options.prefetch

        self.lru = lru
        if options.eviction_policy and options.eviction_policy.startswith('ttl'):
            self.cache = TTLPolicy()
        else:
            self.cache = LRUPolicy()
        self.block_counts = Counter()
        self.ap = ap
        if isinstance(ap, aps.OfflineAP):
            self.dynamic_features = None
        # queue for batch admissions
        self.admit_buffer = {}
        # Metadata for admission buffer
        self.admit_buffer_blocks = set()
        self.admit_buffer_metadata = defaultdict(dict)
        # self.admit_buffer_inserted = {}
        # self.sizes = {}

        self.offline_feat_df = offline_feat_df
        self.early_evict = None

        if early_evict:
            self.early_evict = utils.compress_load(early_evict)
        self.prefetch = None
        if prefetch:
            self.prefetch = utils.compress_load(prefetch)
        self.episodes = episodes
        self.batch_size = batch_size

        self.cached_episodes = {}

        self.prefetcher = prefetcher

        self.ttl_predicter = None

        if options.eviction_policy == 'ttl-ml':
            self.ttl_predicter = TTLModel(options)
        elif options.eviction_policy == 'ttl-opt':
            self.ttl_predicter = TTLOpt()
        elif not options.eviction_policy.startswith('LRU'):
            raise NotImplementedError(options.eviction_policy)

        self.on_evict = on_evict
        self.insert_metadata = {}
        self.keep_metadata = keep_metadata

    def incr_episode(self, key, ts, *, admit_buffer=False):
        if "--fast" in sys.argv:
            return
        block_id, chunk_id = key
        if block_id not in self.cached_episodes:
            self.cached_episodes[block_id] = {
                'block_id': block_id,
                'first_access_ts': ts,
                'last_access_ts': ts,
                'admitted_ts': Counter(),
                'rejected_ts': Counter(),
                'evicted': None,
                'chunks': set(),
                'active_chunks': set(),
                'admitbuffer_chunks': set(),
                'num_accesses': 0,
                'iops_hits': 0,
                'iops_partial_hits': 0,
                'iops_misses': 0,
                'ts_hits': [],
                'ts_misses': [],
                'admits_by_chunk': Counter(),
            }
        eps_stats = self.cached_episodes[block_id]
        eps_stats['last_access_ts'] = ts
        eps_stats['chunks'].add(chunk_id)
        if admit_buffer:
            eps_stats['admitbuffer_chunks'].add(chunk_id)
        else:
            if len(eps_stats['active_chunks']) == 0:
                self.bump("episodes_admitted")
            eps_stats['active_chunks'].add(chunk_id)
            eps_stats['admits_by_chunk'][chunk_id] += 1

    def admit_episode(self, key, ts):
        if "--fast" in sys.argv:
            return
        block_id, chunk_id = key
        eps_stats = self.cached_episodes[block_id]
        eps_stats['admitted_ts'][ts] += 1
        eps_stats['admits_by_chunk'][chunk_id] += 1
        try:
            eps_stats['admitbuffer_chunks'].remove(chunk_id)
        except KeyError:
            # print(key, ts)
            # print(eps_stats)
            # raise
            pass
        eps_stats['active_chunks'].add(chunk_id)

    def dec_episode(self, key, ts, *, admit_buffer=False):
        if "--fast" in sys.argv:
            return
        block_id, chunk_id = key
        eps_stats = self.cached_episodes[block_id]
        if chunk_id not in eps_stats['chunks']:
            print(key)
            # print(eps_stats)
        try:
            if admit_buffer:
                eps_stats['rejected_ts'][ts] += 1
                eps_stats['admitbuffer_chunks'].remove(chunk_id)
            else:
                eps_stats['active_chunks'].remove(chunk_id)
        except KeyError:
            pass
            # print(f"Error: couldn't find chunk {chunk_id}")
            # raise
        if len(eps_stats['active_chunks']) == 0:
            eps_stats['evicted'] = ts
            # TODO: log eps_stats
            if len(eps_stats['admitbuffer_chunks']) == 0:
                a_ts = set(eps_stats['admitted_ts'].keys())
                r_ts = set(eps_stats['rejected_ts'].keys())
                eps_stats["partial_admits"] = len(a_ts & r_ts)
                if eps_stats["partial_admits"] > 0:
                    self.bump("warning_admits_partial", v=eps_stats["partial_admits"])
                    self.bump("warning_admits_partial_episodes")
                for chunk_id, times_admitted in eps_stats['admits_by_chunk'].items():
                    self.bump_counter("chunk_admits_in_epsiode_dist", times_admitted)
                # logger.dump("episodes", eps_stats)
                del self.cached_episodes[block_id]

    def rec_episode(self, block_id, is_hit, chunk_hit, ts):
        if "--fast" in sys.argv:
            return
        if block_id not in self.cached_episodes:
            return
        eps_stats = self.cached_episodes[block_id]
        eps_stats["num_accesses"] += 1
        if is_hit:
            eps_stats["iops_hits"] += 1
            eps_stats["ts_hits"].append(ts)
        else:
            eps_stats["iops_misses"] += 1
            if chunk_hit:
                eps_stats["iops_partial_hits"] += 1
            eps_stats["ts_misses"].append(ts)

    def str(self):
        return "size={}".format(len(self.cache))

    def find(self, key, key_ts, count_as_hit=True, touch=True, check_only=False):
        """Checks if key in cache, and updates access statistics (touch)"""
        found = key in self.cache
        f2 = found or key in self.admit_buffer
        LOG_REQ(self.namespace, key, key_ts, "GET", result=f2)
        if found and not check_only:
            reuse_dist = key_ts - self.cache[key].last_access_time
            assert reuse_dist.logical >= 0, "{} {}".format(key_ts, self.cache[key])
            if count_as_hit:
                self.cache[key].markAccessed(key_ts)
                self.bump("hits")
            elif touch:
                self.cache[key].touch(key_ts)

            # promote by removing and reinserting at the head.
            if self.lru:
                self.cache.touch(key)
        if not check_only:
            self.bump("queries")

        # check if object in admission buffer --> hit
        return found or key in self.admit_buffer

    def handle_miss(self, key, ts, *args, **kwargs):
        if not self.find(key, ts, count_as_hit=False):
            self.insert(key, ts, *args, **kwargs)
            return True
        else:
            if key in self.cache:
                metadata = self.cache[key].stats
                for k, v in kwargs['metadata'].items():
                    if k.startswith('ramcache_'):
                        if k.endswith('_time'):
                            metadata[k] = v
                        else:
                            metadata[k] = metadata.get(k, 0) + v
            else:
                assert key in self.admit_buffer
                for k, v in kwargs['metadata'].items():
                    if k.startswith('ramcache_'):
                        self.admit_buffer_metadata[k][key] = v
                # Corner case: if two RAM evictions happen during admit buffer time, some hits will not be recorded in Flash Stats.
            ods.bump("ram_eviction_already_in_flash")
            return False

    def collect_features(self, key, acc):
        return sim_features.collect_features(self, key, acc)

    def insert(self, key, ts, keyfeaturelist, *, metadata=None):
        metadata = metadata or {}
        LOG_REQ(self.namespace, key, ts, "SET")
        assert key not in self.cache
        self.incr_episode(key, ts, admit_buffer=True)
        # record features
        self.admit_buffer[key] = keyfeaturelist

        if 'ts' not in metadata:
            metadata['ts'] = ts
        for k in metadata:
            self.admit_buffer_metadata[k][key] = metadata[k]

        if self.keep_metadata:
            self.insert_metadata[key] = (keyfeaturelist, metadata)

        block_id, _ = key
        self.admit_buffer_blocks.add(block_id)
        if len(self.admit_buffer) < self.batch_size:
            # still space
            return
        self.process_admit_buffer(ts)

    def process_admit_buffer(self, ts):
        # process batch admission
        if not self.admit_buffer:
            return
        decisions = self.ap.batchAccept(self.admit_buffer, ts, metadata={**{'victim': self.cache.victim()}, **self.admit_buffer_metadata})
        for nkey, dec in decisions.items():
            self.bump("ap.called")
            if self.admit_buffer_metadata['ramcache_hits'].get(nkey, 0) > 0:
                self.bump(["ap.called", "ram_hits"])
            if not dec:
                self.rejections += 1
                self.bump("rejections")
                # Log episode rejections
                self.dec_episode(nkey, ts, admit_buffer=True)
                if nkey in self.admit_buffer_metadata['ramcache_hits']:
                    if self.admit_buffer_metadata['ramcache_hits'][nkey] == 0:
                        self.bump("rejections_no_hit_in_ram")
                        if self.admit_buffer_metadata['prefetch'].get(nkey, False):
                            self.bump("rejections_no_hit_in_ram_prefetches")

                if self.keep_metadata:
                    del self.insert_metadata[nkey]
            else:
                # ts_access = self.admit_buffer_inserted[nkey]
                ts_access = self.admit_buffer_metadata['ts'][nkey]
                item_kwargs = {}
                for k, dct in self.admit_buffer_metadata.items():
                    if nkey in dct and k != 'ts':
                        item_kwargs[k] = dct[nkey]
                if self.admit_buffer_metadata['prefetch'].get(nkey, False):
                    self.bump("prefetches")
                    self.prefetches += 1
                    if item_kwargs.get('at_ep_start', False):
                        self.bump("prefetches_at_ep_start")
                    else:
                        self.bump("prefetches_after_ep_start")
                # admit into cache
                self.admit_episode(nkey, ts)
                # TTL
                ttl = None
                if isinstance(self.cache, TTLPolicy):
                    # TODO: OPT-TTL
                    # print(len(self.admit_buffer_metadata))
                    # print(self.admit_buffer_metadata)
                    metadata_ = {k: v[nkey] for k, v in self.admit_buffer_metadata.items() if nkey in v}
                    ttl = self.ttl_predicter.predict(self.admit_buffer[nkey], metadata=metadata_)
                    # print(ttl)
                    self.bump("total_ttl", v=int(ttl))
                    # assert ttl is not None
                    # TODO: Log average TTL
                # ttl = ttl * 1.25
                # ttl = min(ttl, 3600*2)
                self.admit(nkey, ts, ts_access=ts_access, ttl=ttl, **item_kwargs)
                # Queue for prefetching
                block_id, chunk_id = nkey
                self.admitted_buffer[block_id].append((chunk_id, ts_access, self.admit_buffer[nkey]))
        self.admit_buffer.clear()
        self.admit_buffer_blocks.clear()
        self.admit_buffer_metadata.clear()

    def admit(self, key, ts, *, ttl=None, ts_access=None, episode=None, **item_kwargs):
        block_id, chunk_id = key
        if self.block_counts.get(block_id, 0) == 0:
            self.bump("episodes_admitted2")
        ts_inserted = ts_access or ts

        readmission_from_ep = False
        if episode:
            episode.s["sim_chunk_written"][chunk_id] += 1
            if episode.s["sim_chunk_written"][chunk_id] > 1:
                self.bump("readmission_from_ep")
                readmission_from_ep = True
            # Record only once per IO.
            if ts_inserted not in episode.s["sim_admitted_ts"]:
                if len(episode.s["sim_admitted_ts"]) == 1:
                    # Only count readmissions on the first time.
                    self.bump("readmission_from_ep_io")
                episode.s["sim_admitted_ts"].add(ts_inserted)
                if item_kwargs.get('at_ep_start', False):
                    self.bump("admits_at_ep_start_ios")
                else:
                    # Could have been prefetches/admitted earlier, i.e., Late/Readmits.
                    self.bump("admits_after_ep_start_ios")

        if True or "--fast" not in sys.argv:
            if episode is not None:
                # TODO: Figure out why with fractional prefetching, this is not set.
                if not episode.s.get("sim_admitted", False):
                    episode.s["sim_admitted"] = True
                    self.bump("service_time_saved__prefetch_from_episode", v=episode.s_export["service_time_saved__prefetch"])
                    self.bump("service_time_saved__noprefetch_from_episode", v=episode.s_export["service_time_saved__noprefetch"])
                    # Ideally, hits__prefetch
                    self.bump("hits__prefetch_from_episode", v=episode.num_accesses - 1)
                    self.bump("service_time_saved_pf_from_episode", v=episode.s_export["prefetch_st_benefit"])
                    if episode.chunk_counts:
                        self.bump("admitted_chunks_from_analysis", v=len(episode.chunk_counts))
                    else:
                        self.bump("admitted_chunks_from_analysis", v=episode.num_chunks)

                if episode.num_accesses:
                    self.bump_counter("admitted_eps_hits", episode.num_accesses)
                self.bump_counter("admitted_eps_score", round(episode.score, 7))
                self.bump_counter("admitted_eps_size", episode.num_chunks)
                self.bump_counter("admitted_eps_threshold", round(episode.threshold))
                self.bump_counter("admitted_after_eps_start", round(ts_inserted.physical - episode.ts_physical[0]))
                if episode.chunk_last_seen:
                    tags = []
                    if item_kwargs.get('prefetch', False):
                        tags.append("prefetch")
                    if "--fast" not in sys.argv:
                        eps_stats = self.cached_episodes[block_id]
                        if eps_stats['admits_by_chunk'][chunk_id] > 1:
                            tags.append("readmission")
                    if readmission_from_ep:
                        tags.append("readmissionEp")
                    if item_kwargs.get("ramcache_hits", 0) > 0:
                        tags.append("hitsinram")
                    if item_kwargs.get("ramcache_doomed", False):
                        tags.append("ramdoomed")
                    if episode.num_accesses == 1:
                        tags.append("oneacc")
                    if chunk_id not in episode.chunk_last_seen:
                        self.bump("admitted_chunknotinepisode", tags=tags)
                    elif ts.physical >= episode.chunk_last_seen[chunk_id][0]:
                        if episode.chunk_counts[chunk_id] in [1, 2, 3]:
                            tags.append(f"{episode.chunk_counts[chunk_id]}chunkacc")
                        if ts_inserted.physical == episode.chunk_last_seen[chunk_id][0]:
                            tags.append("onlastseen")
                        self.bump("admitted_doomed", tags=tags)
                        item_kwargs['doomed'] = True
                        self.bump_counter('admitted_without_hits_remaining_hits_dist', episode.chunk_counts[chunk_id])
                else:
                    item_kwargs['doomed'] = False

                rounded = ts_inserted.logical - episode.ts_logical[0]
                if rounded > 100:
                    if rounded < 10000:
                        rounded = 100 * round(rounded / 100)
                    else:
                        rounded = 10 ** round(np.log10(rounded))
                self.bump_counter("admitted_after_eps_start_logical", rounded)
                # log how many accesses after episode start
            else:
                self.bump("warning_admitted_eps_unknown")

            tags = []
            if item_kwargs.get('promotion', 0) > 0:
                tags.append("prefetch")
            if item_kwargs.get('prefetch', False):
                tags.append("prefetch")

            if "--fast" not in sys.argv:
                eps_stats = self.cached_episodes[block_id]
                if eps_stats['admits_by_chunk'][chunk_id] > 1:
                    tags.append("readmission")
            if readmission_from_ep:
                tags.append("readmissionEp")
            self.bump("admitted", tags=tags)

        if item_kwargs.get('at_ep_start', False):
            self.bump("admits_at_ep_start")
        else:
            self.bump("admits_after_ep_start")

        # insertion
        # TODO: Predict TTL
        self.cache.admit(key, TTLItem(ts, key, ts_access=ts_access, ttl=ttl, episode=episode, **item_kwargs))

        self.block_counts[block_id] += 1
        self.keys_written += 1
        self.bump("keys_written")

        if "|" in block_id:
            trace_id = block_id.split("|")[0]
            self.bump(f"keys_written/{trace_id}")

        if len(self.cache) >= self.cache_size:
            self.do_eviction(ts)
        assert len(self.cache) < self.cache_size
        # logger.dump("admissions", [key, (ts.physical, ts.logical)])

    def do_eviction(self, ts, key=None):
        evicted = self.cache.evict(key)
        if key:
            assert key == evicted[1].key
        self.block_counts[evicted[1].key[0]] -= 1
        self.dec_episode(evicted[1].key, ts)
        self.log_eviction(ts, evicted)
        if self.on_evict:
            keyfeaturelist, metadata = self.insert_metadata[evicted[1].key]
            for k in ['last_access_time', 'admission_time', 'hits']:
                metadata['ramcache_' + k] = getattr(evicted[1], k)
            for k in ['doomed', 'promotion']:
                if k in evicted[1].stats:
                    metadata['ramcache_' + k] = evicted[1].stats[k]

            metadata['ramcache_admissions'] = 1
            insert_success = self.on_evict(evicted[1].key, ts, keyfeaturelist, metadata=metadata)
            if not insert_success:
                self.bump("evicted_already_in_lower_cache")

        if self.keep_metadata:
            del self.insert_metadata[evicted[1].key]
