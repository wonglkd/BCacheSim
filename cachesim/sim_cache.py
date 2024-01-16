import copy
import gc
import json
import os
import random
from collections import defaultdict
import traceback
import sys
import time
import pickle
import pprint

import compress_json
import numpy as np
import pandas as pd

from . import admission_policies as aps
from . import eviction_policies as evictp
from . import dynamic_features as dyn_features
from . import prefetchers
from . import utils
from .utils import LOG_IOPS, ods, fmt_dur

from ..episodic_analysis.episodes import service_time
from ..episodic_analysis.episodes import st_to_util
from .ep_helpers import _lookup_episode
from .ep_helpers import Timestamp
from .ep_helpers import record_service_time_get
from .ep_helpers import record_service_time_put
from .ep_helpers import AccessPlus


CACHE_LOCATIONS = {
    "": "",
    "_admitbuffer": "AdmitBuffer",
    "_flash": "Flash",
    "_ram_prefetch": "RamPrefetch",
    "_ram_prefetch_firsthit": "RamPrefetchFirstHit",
    "_flash_noram": "FlashNotInRam",
    "_flash_prefetch": "FlashPrefetch",
    "_flash_prefetch_firsthit": "FlashPrefetchFirstHit",
    "_ram": "Ram",
}


class StatsDumper(object):
    def __init__(self, cache, logjson, output_dir, filename, *,
                 trace_stats=None, ram_cache=None, prefetcher=None,
                 admission_policy=None,
                 start_time=None, skip_first_secs=None):
        self.cache = cache
        self.ram_cache = ram_cache
        self.prefetcher = prefetcher
        self.logjson = logjson
        self.output_dir = output_dir
        self.filename = filename
        self.trace_stats = trace_stats
        self.start_time = start_time
        self.skip_first_secs = skip_first_secs
        self.df_analysis = None
        self.admission_policy = admission_policy
        analysis_filename = os.path.join(output_dir, "df_analysis.csv")
        if os.path.exists(analysis_filename):
            self.df_analysis = pd.read_csv(analysis_filename)

    def dump(self, stats_, *, suffix="", verbose=False, dump_stats=False):
        logjson = self.logjson
        trace_duration_secs = logjson["traceSeconds"]
        sample_ratio = logjson["sampleRatio"]
        total_iops_get = logjson["totalIOPSGet"]
        chunk_queries = ods.get("chunk_queries")
        prefetcher_enabled = self.prefetcher.enabled

        # ods.span("puts_ios", i=bi)

        # This should be the only use of ods.last()
        secs_so_far = ods.last("time_phy")[1] - ods.get("start_ts_phy")
        iops_requests_sofar = ods.get("iops_requests")
        logjson["results"]["ServiceTimeTotalOrig"] = service_time(iops_requests_sofar, chunk_queries)

        # TODO: if we skip the firstm, we also need to decrease duration_s here.
        util_kwargs = dict(sample_ratio=logjson["sampleRatio"], duration_s=secs_so_far)
        util_peak_kwargs = dict(sample_ratio=logjson["sampleRatio"], duration_s=logjson['options']['log_interval'])

        # TODO: Deprecate.
        for k, v in CACHE_LOCATIONS.items():
            k_ = "/"+k[1:] if k.startswith("_") else k
            logjson["results"][f"TotalIOPSSaved{v}"] = ods.get(f"iops_saved{k_}")
            logjson["results"][f"TotalIOPSSaved{v}Only"] = ods.get(f"iops_saved{k_}_only")
            logjson["results"][f"TotalIOPSPartialHits{v}"] = ods.get(f"iops_partial_hits{k_}")
            logjson["results"][f"IOPSSaved{v}Ratio"] = utils.safe_div(logjson["results"][f"TotalIOPSSaved{v}"], iops_requests_sofar)
            logjson["results"][f"IOPSSaved{v}OnlyRatio"] = utils.safe_div(logjson["results"][f"TotalIOPSSaved{v}Only"], iops_requests_sofar)
            logjson["results"][f"IOPSPartialHits{v}Ratio"] = utils.safe_div(logjson["results"][f"TotalIOPSPartialHits{v}"], iops_requests_sofar)
            logjson["results"][f"TotalChunkHits{v}"] = ods.get(f"chunk_hits{k_}")
            logjson["results"][f"TotalChunkSaved{v}"] = ods.get(f"chunk_saved{k_}")
            logjson["results"][f"ChunkHit{v}Ratio"] = utils.safe_div(logjson["results"][f"TotalChunkHits{v}"], chunk_queries)

        flash_queries = chunk_queries
        if "TotalChunkHitsRam" in logjson["results"]:
            flash_queries -= logjson["results"]["TotalChunkHitsRam"]
        logjson["results"]["TotalFlashQueries"] = flash_queries

        logjson["results"]["FlashCacheHitRate"] = utils.safe_div(logjson["results"]["TotalChunkHitsFlashNotInRam"], flash_queries)

        logjson["results"]["ClientBandwidth"] = utils.mb_per_sec(chunk_queries, secs_so_far, sample_ratio)

        chunks_written = ods.get("flashcache/keys_written")
        # logjson["results"]["ChunkWritten"] = stats["chunks_written"]
        # writes_per_reqs = chunks_written / total_iops * sample_ratio
        logjson["results"]["FlashWriteRate"] = utils.mb_per_sec(
            chunks_written, secs_so_far, sample_ratio
        )
        logjson["results"]["FlashPrefetchWriteRate"] = utils.mb_per_sec(
            ods.get("flashcache/prefetches"), secs_so_far, sample_ratio
        )
        logjson["results"]["TotalChunkWritten"] = chunks_written
        logjson["results"]["FlashChunkWritten"] = chunks_written

        caches = {'': [self.cache, "flashcache"]}
        if self.ram_cache:
            caches['RamCache'] = [self.ram_cache, "ramcache"]
            logjson["results"]["RamWriteRate"] = utils.mb_per_sec(
                ods.get("ramcache/keys_written"), secs_so_far, sample_ratio
            )
            logjson['results']['RamChunkWritten'] = ods.get("ramcache/keys_written")
            logjson["results"]["RamPrefetchWriteRate"] = utils.mb_per_sec(
                ods.get("ramcache/prefetches"), secs_so_far, sample_ratio
            )
            logjson["results"]["RamCacheHitRate"] = utils.safe_div(logjson["results"]["TotalChunkHitsRam"], ods.get("ramcache/queries"))
        else:
            logjson['results']['RamWriteRate'] = 0
            logjson['results']['RamChunkWritten'] = 0
            logjson['results']['RamPrefetchWriteRate'] = 0
        for label, (cache, cache_ns) in caches.items():
            # TODO: Replace with ODS stats.
            logjson["results"][label + "AvgMaxInterarrivalTime"] = cache.computeAvgMaxInterarrivalTime()

            logjson["results"][label + "MaxMaxInterarrivalTime"] = max(logjson["results"][label + "AvgMaxInterarrivalTime"], max(ods.get("max_ia_max", init=[Timestamp(0, 0)])))
            logjson["results"][label + "WarmupFinished"] = ods.get(f"{cache_ns}/warmup_finished")
            logjson["results"][label + "AvgMaxInterarrivalTimeEvicted"] = utils.safe_div(ods.get(f"{cache_ns}/max_interarrival_time_cum"), ods.get(f"{cache_ns}/evictions"))
            logjson["results"][label + "AvgEvictionAge"] = utils.safe_div(ods.get(f"{cache_ns}/eviction_age_cum"), ods.get(f"{cache_ns}/evictions"))
            logjson["results"][label + "AvgNoHitEvictionAge"] = utils.safe_div(ods.get(f"{cache_ns}/unaccessed_eviction_age_cum"), ods.get(f"{cache_ns}/unaccessed_evictions"))
            logjson["results"][label + "KeysWritten"] = ods.get(f"{cache_ns}/keys_written")
            logjson["results"][label + "NumCacheEviction"] = ods.get(f"{cache_ns}/evictions")
            logjson["results"][label + "MeanTimeInSystem"] = utils.safe_div(ods.get(f"{cache_ns}/total_time_in_system"), ods.get(f"{cache_ns}/keys_written"))
            logjson["results"][label + "NumNoHitEvictions"] = ods.get(f"{cache_ns}/unaccessed_evictions")
            logjson["results"][label + "NumEarlyEvictions"] = ods.get(f"{cache_ns}/early_evictions")
            logjson["results"][label + "NumPrefetches"] = ods.get(f"{cache_ns}/prefetches")
            logjson["results"][label + "NumPrefetchesFirstAcc"] = ods.get(f"{cache_ns}/prefetches_failed_firstaccess")
            # logjson["results"][label + "NumPrefetchesCaught"] = cache.prefetches_caught
            logjson["results"][label + "NumFailedPrefetchesExists"] = ods.get(f"{cache_ns}/prefetches_failed_exists")
            logjson["results"][label + "NumFailedPrefetches"] = logjson["results"][label + "NumPrefetchesFirstAcc"] + logjson["results"][label + "NumFailedPrefetchesExists"]
            # logjson["results"][label + "NumEpisodeTouches"] = cache.episode_touches
            logjson["results"][label + "NumCacheRejection"] = ods.get(f"{cache_ns}/rejections")
            logjson["results"][label + "AcceptanceRatio"] = utils.safe_div(logjson["results"][label + "KeysWritten"], logjson["results"][label + "KeysWritten"] + logjson["results"][label + "NumCacheRejection"])
            logjson["results"][label + "PrefetchSuccessRate"] = utils.safe_div(logjson["results"][label + "NumPrefetches"], logjson["results"][label + "NumPrefetches"] + logjson["results"][label + "NumPrefetchesFirstAcc"])

        # == Pinning down mistakes ==
        # = Comparing to analysis =
        # Eviction Age
        # Extra writes
        logjson["results"]["AssumedFlashEaTooLong"] = ods.get('flashcache/evicted_with_hits_remaining')
        # Bonus hits
        logjson["results"]["AssumedFlashEaTooShort"] = ods.get('flashcache/evicted_without_hits_remaining_hitsAfterEp')
        # -- This shouldn't exist? + ods.get('flashcache/evicted_with_hits_remaining_hitsAfterEp')
        logjson["results"]["TooShortEaBonusServiceTimeSavedRatio"] = utils.safe_div(service_time(0, logjson["results"]["AssumedFlashEaTooShort"]), logjson["results"]["ServiceTimeTotalOrig"])
        # TODO: Bonus IOPS Saved Ratio. Assume it saves one miss per episode that you join.

        logjson["results"]["AnalysisServiceTimeSaved"] = ods.get("flashcache/service_time_saved__{}prefetch_from_episode".format("" if prefetcher_enabled else "no"))
        logjson["results"]["AnalysisServiceTimeSavedRatio"] = utils.safe_div(logjson["results"]["AnalysisServiceTimeSaved"], logjson["results"]["ServiceTimeTotalOrig"])
        logjson["results"]["AnalysisServiceTimeSavedFromPrefetchRatio"] = utils.safe_div(ods.get("flashcache/service_time_saved_pf_from_episode"), logjson["results"]["ServiceTimeTotalOrig"])
        logjson["results"]["AnalysisIOPSSavedRatio"] = utils.safe_div(ods.get("flashcache/hits__prefetch_from_episode"), iops_requests_sofar)

        logjson["results"]["AnalysisAdmittedChunkRatio"] = utils.safe_div(ods.get("flashcache/admitted_chunks_from_analysis"), chunks_written)
        logjson["results"]["AnalysisAdmittedWriteRate"] = utils.mb_per_sec(
            ods.get("flashcache/admitted_chunks_from_analysis"), secs_so_far, sample_ratio
        )

        logjson["results"]["AnalysisOfflineWR"] = -1
        logjson["results"]["AnalysisOfflineServiceTimeSavedRatio"] = -1
        logjson["results"]["AnalysisOfflineMeanTimeInSystem"] = -1
        logjson["results"]["AnalysisOfflineIOPSSavedRatio"] = -1
        logjson["results"]["AnalysisOfflineEpisodesAdmitted"] = -1
        logjson["results"]["AnalysisOfflineCacheSize"] = -1
        logjson["results"]["AnalysisAdjWR"] = -1
        logjson["results"]["AnalysisAdjServiceTimeSavedRatio"] = -1
        logjson["results"]["AnalysisAdjIOPSSavedRatio"] = -1
        logjson["results"]["AnalysisAdjMeanTimeInSystem"] = -1
        logjson["results"]["AnalysisAdjEpisodesAdmitted"] = -1
        logjson["results"]["AnalysisAdjCacheSize"] = -1
        if self.df_analysis is not None:
            row_analysis, row_analysis_adjusted = None, None
            try:
                row_analysis = utils.closest_row(self.df_analysis, 'Target Write Rate', logjson["results"]["FlashWriteRate"]).to_dict()
                logjson["results"]["AnalysisOfflineWR"] = float(row_analysis["Write Rate (MB/s)"])
                logjson["results"]["AnalysisOfflineServiceTimeSavedRatio"] = float(row_analysis["Service Time Saved Ratio"])
                logjson["results"]["AnalysisOfflineMeanTimeInSystem"] = float(row_analysis["Mean Time In System (s)"])
                logjson["results"]["AnalysisOfflineIOPSSavedRatio"] = float(row_analysis["IOPSSavedRatio"])
                logjson["results"]["AnalysisOfflineEpisodesAdmitted"] = int(row_analysis["Episodes admitted"])
                logjson["results"]["AnalysisOfflineCacheSize"] = float(row_analysis["Cache Size (GB)"])
                # Test serializability
                json.dumps(logjson["results"])

                row_analysis_adjusted = utils.closest_row(self.df_analysis, 'Target Write Rate', logjson["results"]["AnalysisAdmittedWriteRate"]).to_dict()
                logjson["results"]["AnalysisAdjWR"] = float(row_analysis_adjusted["Write Rate (MB/s)"])
                logjson["results"]["AnalysisAdjServiceTimeSavedRatio"] = float(row_analysis_adjusted["Service Time Saved Ratio"])
                logjson["results"]["AnalysisAdjIOPSSavedRatio"] = float(row_analysis_adjusted["IOPSSavedRatio"])
                logjson["results"]["AnalysisAdjMeanTimeInSystem"] = float(row_analysis_adjusted["Mean Time In System (s)"])
                logjson["results"]["AnalysisAdjEpisodesAdmitted"] = int(row_analysis_adjusted["Episodes admitted"])
                logjson["results"]["AnalysisAdjCacheSize"] = float(row_analysis_adjusted["Cache Size (GB)"])
                json.dumps(logjson["results"])

            except Exception as e:
                print(str(e))
                print(row_analysis)
                print(row_analysis_adjusted)
                traceback.print_exc()
                self.df_analysis = None

        logjson["results"]["ServiceFetchIOs"] = ods.get("fetches_ios")
        logjson["results"]["ServiceFetchChunks"] = ods.get("fetches_chunks")
        logjson["results"]["ServiceFetchChunksDemandmiss"] = ods.get("fetches_chunks_demandmiss")
        logjson["results"]["ServiceFetchChunksPrefetch"] = ods.get("fetches_chunks_prefetch")
        logjson["results"]["ServicePutIOs"] = ods.get("puts_ios")
        logjson["results"]["ServicePutChunks"] = ods.get("puts_chunks")
        logjson["results"]["ServiceGetPutIOs"] = ods.get("fetches_ios") + ods.get("puts_ios")
        logjson["results"]["ServiceGetPutChunks"] = ods.get("fetches_chunks") + ods.get("puts_chunks")
        logjson["results"]["BackendBandwidthGet"] = utils.mb_per_sec(logjson["results"]["ServiceFetchChunks"], secs_so_far, sample_ratio)
        logjson["results"]["BackendBandwidthPut"] = utils.mb_per_sec(logjson["results"]["ServicePutChunks"], secs_so_far, sample_ratio)
        logjson["results"]["BackendBandwidth"] = utils.mb_per_sec(logjson["results"]["ServiceGetPutChunks"], secs_so_far, sample_ratio)

        # Lost hits
        # admits_at_ep_start
        logjson["results"]["WastedHitsAfterEpStart"] = ods.get("flashcache/admits_after_ep_start")
        logjson["results"]["WastedHitsAfterEpStartIOs"] = ods.get("flashcache/admits_after_ep_start_ios")
        logjson["results"]["WastedHitsAfterEpStartIOsRatio"] = utils.safe_div(
            logjson["results"]["WastedHitsAfterEpStartIOs"], iops_requests_sofar)
        logjson["results"]["WastedHitsAfterEpStartServiceTimeRatio"] = utils.safe_div(
            service_time(ods.get("flashcache/admits_after_ep_start_ios"), ods.get("flashcache/admits_after_ep_start")),
            logjson["results"]["ServiceTimeTotalOrig"])
        logjson["results"]["WastedHitsAfterEpStartServiceTimeRatioIOs"] = utils.safe_div(
            service_time(ods.get("flashcache/admits_after_ep_start_ios"), 0),
            logjson["results"]["ServiceTimeTotalOrig"])
        logjson["results"]["WastedHitsAfterEpStartServiceTimeRatioChunks"] = utils.safe_div(
            service_time(0, ods.get("flashcache/admits_after_ep_start")),
            logjson["results"]["ServiceTimeTotalOrig"])

        # log_items += ods.get_all_with_prefix("flashcache/admitted_without_hits_remaining")

        # Real wasted:
        # - Readmission (with no more hits, on last seen): should not be admitted
        # - Readmission (with more hits - not registered in wasted): EA/etc should be adjusted to consider the cost of extra write. Might be considered in chunkaware.
        # - Not In Episode / Prefetch mistakes. Should not be admitted.

        # Wasted writes
        logjson["results"]["WastedFlashRatio"] = utils.safe_div(logjson["results"]["NumNoHitEvictions"], chunks_written)
        logjson["results"]["FlashRatioEAHitsLeft"] = utils.safe_div(logjson["results"]["AssumedFlashEaTooLong"], chunks_written)
        # Evicted with future hits: need to be readmitted. EA should match ReadmissionEp.
        # ReadmissionEp can overlap with Doomed, esp DoomedOnLastSeen, but not DoomedOneChunkAcc(?).
        # Readmissions can be split into:
        # - no future hits = (DoomedOnLastSeen - DoomedOneChunkAcc) = DoomedOnLastSeenReadmit
        # - with more hits = ReadmissionEp - (DoomedOnLastSeen - DoomedOneChunkAcc)
        logjson["results"]["WastedFlashRatioEvictEANoHits"] = utils.safe_div(ods.get("flashcache/evicted_with_hits_remaining_nohits"), chunks_written)
        logjson["results"]["WastedFlashRatioEvictEA"] = utils.safe_div(ods.get("flashcache/evicted_with_hits_remaining"), chunks_written)
        logjson["results"]["WastedFlashRatioAdmitReadmissionEp"] = utils.safe_div(ods.get("flashcache/readmission_from_ep"), chunks_written)
        logjson["results"]["WastedFlashRatioAdmitDoomedOnLastSeenReadmit"] = utils.safe_div(ods.get("flashcache/admitted_doomed_onlastseen_readmissionEp"), chunks_written)
        logjson["results"]["WastedFlashRatioAdmitReadmissionEpHidden"] = logjson["results"]["WastedFlashRatioAdmitReadmissionEp"] - logjson["results"]["WastedFlashRatioAdmitDoomedOnLastSeenReadmit"]
        # TODO: Fix or remove. Incomplete.
        # logjson["results"]["WastedFlashRatioAdmitReadmissionSim"] = utils.safe_div(ods.get("flashcache/admitted_readmission"), chunks_written)
        # Doomed - No future hits
        logjson["results"]["WastedFlashRatioAdmitDoomed"] = utils.safe_div(ods.get("flashcache/admitted_doomed"), chunks_written)
        logjson["results"]["WastedFlashRatioAdmitDoomedOnLastSeen"] = utils.safe_div(ods.get("flashcache/admitted_doomed_onlastseen"), chunks_written)
        logjson["results"]["WastedFlashRatioAdmitDoomed1ChunkAcc"] = utils.safe_div(ods.get("flashcache/admitted_doomed_1chunkacc"), chunks_written)
        logjson["results"]["WastedFlashRatioAdmitDoomed2ChunkAcc"] = utils.safe_div(ods.get("flashcache/admitted_doomed_2chunkacc"), chunks_written)
        logjson["results"]["WastedFlashRatioAdmitDoomed3ChunkAcc"] = utils.safe_div(ods.get("flashcache/admitted_doomed_3chunkacc"), chunks_written)
        logjson["results"]["WastedFlashRatioAdmitDoomedOneAcc"] = utils.safe_div(ods.get("flashcache/admitted_doomed_oneacc"), chunks_written)
        # Also doomed, but not classified as such
        logjson["results"]["WastedFlashRatioAdmitNotInEpisode"] = utils.safe_div(ods.get("flashcache/admitted_chunknotinepisode"), chunks_written)
        # Orthogonal category
        logjson["results"]["WastedFlashRatioEvictPrefetch"] = utils.safe_div(ods.get("flashcache/evicted_nohits_prefetch"), chunks_written)
        flash_wasted_breakdown_admit = ''
        flash_wasted_breakdown_evict = []
        for k, v in logjson["results"].items():
            if k.startswith("WastedFlashRatioAdmit") and k != "WastedFlashRatioAdmit" and v > 0:
                flash_wasted_breakdown_admit += f"\n    {k.replace('WastedFlashRatioAdmit','')} - {v:.3f}"
            elif k.startswith("WastedFlashRatioEvict") and k != "WastedFlashRatioEvict" and v > 0:
                flash_wasted_breakdown_evict.append(f"{k.replace('WastedFlashRatioEvict','')}: {v:.3f}")
        flash_wasted_breakdown_evict = ', '.join(flash_wasted_breakdown_evict)

        logjson["results"]["TotalChunkQueries"] = chunk_queries
        logjson["results"]["TotalChunkSaved"] = ods.get("chunks_saved")

        # Total Prefetches
        prefetches_exists = ods.get("flashcache/prefetches_failed_exists_incache")
        prefetches_ = logjson["results"][("RamCache" if self.ram_cache else "") + "NumPrefetches"]
        # + prefetches_exists
        wasted_pf_ram = ods.get("flashcache/rejections_no_hit_in_ram_prefetches")
        wasted_pf_flash = ods.get("flashcache/evicted_nohits_prefetch")
        wasted_prefetches = wasted_pf_ram + wasted_pf_flash
        useful_pf_ram = ods.get(["chunks_saved", "ram_prefetch_firsthit"])
        useful_pf_flash = ods.get(["chunks_saved", "flash_prefetch_firsthit"])
        useful_prefetches = useful_pf_ram + useful_pf_flash
        logjson["results"]["TotalUsefulPrefetches"] = useful_prefetches
        logjson["results"]["TotalWastedPrefetches"] = wasted_prefetches
        logjson["results"]["WastedPrefetchRatio"] = utils.safe_div(wasted_prefetches, wasted_prefetches + useful_prefetches)

        for k, v in CACHE_LOCATIONS.items():
            logjson["results"][f"ServiceTimeSavedFrom{v}"] = service_time(
                logjson["results"][f"TotalIOPSSaved{v}"],
                logjson["results"][f"TotalChunkSaved{v}"])
            logjson["results"][f"ServiceTimeSaved{v}Ratio"] = utils.safe_div(
                logjson["results"][f"ServiceTimeSavedFrom{v}"],
                logjson["results"]["ServiceTimeTotalOrig"])

        logjson["results"]["ServiceTimeSaved"] = service_time(
            logjson["results"]["TotalIOPSSaved"],
            logjson["results"]["TotalChunkSaved"] - prefetches_)
        logjson["results"]["ServiceTimeSavedRatio"] = utils.safe_div(logjson["results"]["ServiceTimeSaved"], logjson["results"]["ServiceTimeTotalOrig"])

        st_keys = {1: '', 2: '2', 3: '3'}
        st_stats_nocache = np.diff(ods.get('service_time_nocache_stats'), prepend=0)
        st_stats_puts = np.diff(ods.get("service_time_writes_stats"), prepend=0)
        if self.skip_first_secs:
            intervals_skip = int(self.skip_first_secs // logjson['options']['log_interval'])
            if len(st_stats_nocache) > intervals_skip + 1:
                st_stats_nocache = st_stats_nocache[intervals_skip:]
                st_stats_puts = st_stats_puts[intervals_skip:]
        st_nocache_with_put = np.add(st_stats_nocache, st_stats_puts)
        logjson["results"]["ServiceTimePutUtil"] = st_to_util(ods.get("service_time_writes"), **util_kwargs) * 100
        logjson["results"]["PeakServiceTimePutUtil"] = st_to_util(max(st_stats_puts, default=0), **util_peak_kwargs) * 100
        for percentile in [.5, .9, .95, .99, .995, .999, .9999, .99999]:
            if len(st_stats_nocache) > 0:
                logjson['results'][f'P{percentile*100:g}ServiceTimeUsedNoCache'] = np.percentile(st_stats_nocache, percentile*100)
                logjson['results'][f'P{percentile*100:g}ServiceTimeNoCacheUtil'] = st_to_util(logjson['results'][f'P{percentile*100:g}ServiceTimeUsedNoCache'], **util_peak_kwargs) * 100
                logjson['results'][f'P{percentile*100:g}ServiceTimeUsedWithPutNoCache'] = np.percentile(st_nocache_with_put, percentile*100)
            if len(st_stats_puts) > 0:
                logjson["results"][f"P{percentile*100:g}ServiceTimePut"] = np.percentile(st_stats_puts, percentile*100)
                logjson["results"][f"P{percentile*100:g}ServiceTimePutUtil"] = st_to_util(
                    logjson["results"][f"P{percentile*100:g}ServiceTimePut"], **util_peak_kwargs) * 100
        for k, v in st_keys.items():
            for kk, vv in {"": "used", "OnPut": "writes"}.items():
                st_ = ods.get(f"service_time_used{v}")
                logjson["results"][f"ServiceTimeUsed{kk}{k}"] = st_
                logjson["results"][f"ServiceTime{kk}Util{k}"] = st_to_util(st_, **util_kwargs) * 100
            if k == 1:
                logjson["results"][f"ServiceTimeUsedWithPut{k}"] = ods.get("service_time")
                logjson["results"][f"ServiceTimeWithPutUtil{k}"] = st_to_util(ods.get("service_time"), **util_kwargs) * 100
            logjson["results"][f"ServiceTimeOnPrefetchRatio{k}"] = utils.safe_div(ods.get(f"service_time_used_prefetch{v}"), ods.get(f"service_time_used{v}"))
            logjson["results"][f"ServiceTimeSavedRatio{k}"] = 1. - utils.safe_div(ods.get(f"service_time_used{v}"), logjson["results"]["ServiceTimeTotalOrig"])
            st_stats = np.diff(ods.get(f"service_time_used{v}_stats"), prepend=0)
            if self.skip_first_secs and len(st_stats) > intervals_skip + 1:
                st_stats = st_stats[intervals_skip:]
            logjson["results"][f"PeakServiceTimeUsed{k}"] = max(st_stats, default=0)
            logjson["results"][f'PeakServiceTimeSavedRatio{k}'] = 1 - utils.safe_div(logjson["results"][f"PeakServiceTimeUsed{k}"], max(st_stats_nocache, default=0))
            logjson["results"][f"PeakServiceTimeUtil{k}"] = st_to_util(max(st_stats), **util_peak_kwargs) * 100
            assert len(st_stats) == len(st_stats_puts)
            st_with_put = np.add(st_stats, st_stats_puts)
            logjson["results"][f"PeakServiceTimeUsedWithPut{k}"] = max(st_with_put, default=0)
            logjson["results"][f"PeakServiceTimeUsedWithPutUtil{k}"] = st_to_util(max(st_with_put, default=0), **util_peak_kwargs) * 100
            logjson["results"][f'PeakServiceTimeSavedWithPutRatio{k}'] = 1 - utils.safe_div(logjson["results"][f"PeakServiceTimeUsedWithPut{k}"], max(st_nocache_with_put, default=0))
            for percentile in [.5, .9, .95, .99, .995, .999, .9999, .99999]:
                logjson['results'][f'P{percentile*100:g}ServiceTimeUsed{k}'] = np.percentile(st_stats, percentile*100)
                logjson['results'][f'P{percentile*100:g}ServiceTimeUtil{k}'] = st_to_util(logjson['results'][f'P{percentile*100:g}ServiceTimeUsed{k}'], **util_peak_kwargs) * 100
                logjson['results'][f'P{percentile*100:g}ServiceTimeSavedRatio{k}'] = 1 - utils.safe_div(logjson['results'][f'P{percentile*100:g}ServiceTimeUsed{k}'], logjson['results'][f'P{percentile*100:g}ServiceTimeUsedNoCache'])
                logjson['results'][f'P{percentile*100:g}ServiceTimeUsedWithPut{k}'] = np.percentile(st_with_put, percentile*100)
                logjson['results'][f'P{percentile*100:g}ServiceTimeWithPutUtil{k}'] = st_to_util(logjson['results'][f'P{percentile*100:g}ServiceTimeUsedWithPut{k}'], **util_peak_kwargs) * 100
                logjson['results'][f'P{percentile*100:g}ServiceTimeSavedWithPutRatio{k}'] = 1 - utils.safe_div(logjson['results'][f'P{percentile*100:g}ServiceTimeUsedWithPut{k}'], logjson['results'][f'P{percentile*100:g}ServiceTimeUsedWithPutNoCache'])

        logjson["results"]["ServiceTimeTotalNew"] = service_time(
            iops_requests_sofar - logjson["results"]["TotalIOPSSaved"],
            chunk_queries - logjson["results"]["TotalChunkSaved"] + prefetches_)
        logjson["results"]["ServiceTimeOnPrefetch"] = service_time(0, prefetches_)
        logjson["results"]["ServiceTimeOnPrefetchRatio"] = utils.safe_div(logjson["results"]["ServiceTimeOnPrefetch"], logjson["results"]["ServiceTimeTotalOrig"])
        logjson["results"]["ServiceTimeOnWastedPrefetch"] = service_time(0, wasted_prefetches)
        logjson["results"]["ServiceTimeOnWastedPrefetchRatio"] = utils.safe_div(logjson["results"]["ServiceTimeOnWastedPrefetch"], logjson["results"]["ServiceTimeTotalOrig"])
        iops_saved_by_prefetch = logjson['results']['TotalIOPSSavedRamPrefetchFirstHit'] + logjson['results']['TotalIOPSSavedFlashPrefetchFirstHit']
        logjson["results"]["ServiceTimeSavedByPrefetch"] = service_time(iops_saved_by_prefetch, useful_prefetches)
        logjson["results"]["ServiceTimeSavedByPrefetchRatio"] = utils.safe_div(logjson["results"]["ServiceTimeSavedByPrefetch"], logjson["results"]["ServiceTimeTotalOrig"])
        logjson["results"]["NetServiceTimeForPrefetch"] = logjson['results']['ServiceTimeSavedByPrefetchRatio'] - logjson['results']['ServiceTimeOnPrefetchRatio']
        logjson["results"]["NetServiceTimeForGoodPrefetch"] = logjson["results"]["NetServiceTimeForPrefetch"] + logjson['results']['ServiceTimeOnWastedPrefetchRatio']

        logjson["results"]["TotalEpisodesAdmitted"] = ods.get("flashcache/episodes_admitted2")

        logjson["results"]["EvictionAvgTTL"] = utils.safe_div(ods.get("flashcache/total_ttl"), chunks_written)

        logjson["stats"] = ods.counters

        statsjson = {}
        statsjson["freq"] = ods.freq
        statsjson["batches"] = ods.batches

        warmup_time_txt = "unfinished"
        if logjson['results']['WarmupFinished']:
            warmup_time = logjson['results']['WarmupFinished'] - Timestamp(0, self.trace_stats['start_ts'])
            logjson['results']['WarmupTime'] = warmup_time
            warmup_time_txt = f"{warmup_time.logical}, {fmt_dur(warmup_time.physical)}"

        wall_time = time.time() - self.start_time if self.start_time else 0
        logjson["results"]["SimWallClockTime"] = wall_time
        logjson["results"]["SimRAMUsage"] = utils.memory_usage()

        msg = (
            "Results preview: \n "
            f"Trace: {logjson['trace_kwargs']} \n "
            f"AP: {logjson['options']['ap']}, {self.admission_policy} \n "
            f"Prefetching: {self.prefetcher} \n "
            f"Eviction Policy: {logjson['options']['eviction_policy']}, {self.cache} \n "
            f"Average TTL: {logjson['results']['EvictionAvgTTL']:.2f} s \n "
            f"Duration so far: {fmt_dur(secs_so_far)} \n "
            f"Duration: {fmt_dur(trace_duration_secs)} ({fmt_dur(logjson['options']['log_interval'])} intervals) \n "
            f"Service Time Utilization (%)                  - {logjson['results']['ServiceTimeWithPutUtil1']:.5f} \n "
            f"Service Time Utilization (%) [GET]            - {logjson['results']['ServiceTimeUtil1']:.5f} \n "
            f"Service Time Utilization (%) [PUT]            - {logjson['results']['ServiceTimePutUtil']:.5f} \n "
            f"Peak Service Time Utilization (%)  - {logjson['results']['PeakServiceTimeUsedWithPutUtil1']:.5f} \n "
            f"Peak Service Time Utilization (%) [PUT] - {logjson['results']['PeakServiceTimePutUtil']:.5f} \n "
            f"Peak Service Time Utilization (%) [GET] - {logjson['results']['PeakServiceTimeUtil1']:.5f} \n "
            f"P99 Service Time Utilization (%)               - {logjson['results']['P99ServiceTimeWithPutUtil1']:.5f} \n "
            f"P99.9 Service Time Utilization (%)             - {logjson['results']['P99.9ServiceTimeWithPutUtil1']:.5f} \n "
            f"P99.99 Service Time Utilization (%)            - {logjson['results']['P99.99ServiceTimeWithPutUtil1']:.5f} \n "
            f"P99 Service Time Utilization (%) [GET]               - {logjson['results']['P99ServiceTimeUtil1']:.5f} \n "
            f"P99.9 Service Time Utilization (%) [GET]  - {logjson['results']['P99.9ServiceTimeUtil1']:.5f} \n "
            f"P99.99 Service Time Utilization (%) [GET] - {logjson['results']['P99.99ServiceTimeUtil1']:.5f} \n "
            f"P99 Service Time Utilization (%) [GET-NoCache]  - {logjson['results']['P99ServiceTimeNoCacheUtil']:.5f} \n "
            f"P99.9 Service Time Utilization (%) [GET-NoCache] - {logjson['results']['P99.9ServiceTimeNoCacheUtil']:.5f} \n "
            f"P99.99 Service Time Utilization (%) [GET-NoCache] - {logjson['results']['P99.99ServiceTimeNoCacheUtil']:.5f} \n "
            f"P99 Service Time Utilization (%) [PUT]  - {logjson['results']['P99ServiceTimePutUtil']:.5f} \n "
            f"P99.9 Service Time Utilization (%) [PUT] - {logjson['results']['P99.9ServiceTimePutUtil']:.5f} \n "
            f"P99.99 Service Time Utilization (%) [PUT] - {logjson['results']['P99.99ServiceTimePutUtil']:.5f} \n "
            f"P99 Service Time Saved ratio 1                 - {logjson['results']['P99ServiceTimeSavedRatio1']:.5f} \n "
            f"P99.9 Service Time Saved ratio 1               - {logjson['results']['P99.9ServiceTimeSavedRatio1']:.5f} \n "
            f"P99.99 Service Time Saved ratio 1              - {logjson['results']['P99.99ServiceTimeSavedRatio1']:.5f} \n "
            f"Peak Service Time Saved ratio 1 (range)        - {logjson['results']['PeakServiceTimeSavedRatio1']:.5f} \n "
            f"Peak Service Time Saved ratio 2 (1st M to end) - {logjson['results']['PeakServiceTimeSavedRatio2']:.5f} \n "
            f"Service Time Saved ratio 1 (range)             - {logjson['results']['ServiceTimeSavedRatio1']:.5f} \n "
            f"Service Time Saved ratio 2 (first miss to end) - {logjson['results']['ServiceTimeSavedRatio2']:.5f} \n "
            f"Service Time Saved ratio 3 (start to end)      - {logjson['results']['ServiceTimeSavedRatio3']:.5f} \n "
            f"Service Time Saved ratio                       - {logjson['results']['ServiceTimeSavedRatio']:.5f} \n "
            f"Bonus STS beyond STS(A) (Assumed EA too short) - {logjson['results']['TooShortEaBonusServiceTimeSavedRatio']:.5f} \n "
            f"Est Service Time Lost from late/readmits       - {logjson['results']['WastedHitsAfterEpStartServiceTimeRatio']:.5f} \n "
            f"ST + late/readmits potential                   - {(logjson['results']['ServiceTimeSavedRatio1']+logjson['results']['WastedHitsAfterEpStartServiceTimeRatio']):.5f} \n "
            f"Potential STS(Analysis) from Admitted Episodes - {logjson['results']['AnalysisServiceTimeSavedRatio']:.5f} \n "
            f"Service Time Saved (Analysis @ {logjson['results']['AnalysisOfflineWR']:.1f}MB/s, {logjson['results']['AnalysisOfflineCacheSize']:.1f}GB)  - {logjson['results']['AnalysisOfflineServiceTimeSavedRatio']:.5f} \n "
            f"Service Time Saved (Analysis @ {logjson['results']['AnalysisAdjWR']:.1f}MB/s)         - {logjson['results']['AnalysisAdjServiceTimeSavedRatio']:.5f} \n "
            f"Service Time Saved By Prefetch ratio                 - {logjson['results']['ServiceTimeSavedByPrefetchRatio']:.5f} \n "
            f"Service Time Saved by DRAM (No Prefetch cost) ratio   - {logjson['results']['ServiceTimeSavedRamRatio']:.5f} \n "
            f"Service Time Saved by Flash (No Prefetch cost) ratio - {logjson['results']['ServiceTimeSavedFlashNotInRamRatio']:.5f} \n "
            f"Service Time Saved (No Prefetch Cost) ratio          - {logjson['results']['ServiceTimeSavedRatio'] + logjson['results']['ServiceTimeOnPrefetchRatio']:.5f} \n "
            f"Service Time Spent on All Prefetches ratio   - {logjson['results']['ServiceTimeOnPrefetchRatio']:.5f} \n "
            f"Service Time Spent on Wasted Prefetch ratio  - {logjson['results']['ServiceTimeOnWastedPrefetchRatio']:.5f} \n "
            f"Net Service Time From Good Prefetching ratio - {logjson['results']['NetServiceTimeForGoodPrefetch']:.5f} \n "
            f"Net Service Time From Prefetch ratio         - {logjson['results']['NetServiceTimeForPrefetch']:.5f} \n "
            f"Potential ST Saved (PF) (Analysis) from Admitted Eps - {logjson['results']['AnalysisServiceTimeSavedFromPrefetchRatio']:.2f} \n "
            f"Wasted hits from late/readmits: {logjson['results']['WastedHitsAfterEpStartIOs']} IOs, {logjson['results']['WastedHitsAfterEpStart']} chunks \n "
            f"IOPS saved ratio                         - {logjson['results']['IOPSSavedRatio']:.5f} \n "
            f"IOPS saved ratio (DRAM Only)              - {logjson['results']['IOPSSavedRamOnlyRatio']:.4f} \n "
            f"IOPS saved ratio (Flash Not in DRAM)      - {logjson['results']['IOPSSavedFlashNotInRamRatio']:.4f} \n "
            f"IOPS saved ratio (DRAMPrefetchFirstHit)   - {logjson['results']['IOPSSavedRamPrefetchFirstHitRatio']:.4f} \n "
            f"IOPS saved ratio (FlashPrefetchFirstHit) - {logjson['results']['IOPSSavedFlashPrefetchFirstHitRatio']:.4f} \n "
            f"IOPS saved ratio (DRAM)                   - {logjson['results']['IOPSSavedRamRatio']:.4f} \n "
            f"IOPS saved ratio (Flash)                 - {logjson['results']['IOPSSavedFlashRatio']:.4f} \n "
            f"IOPS saved ratio (AdmitBuffer)           - {logjson['results']['IOPSSavedAdmitBufferRatio']:.4f} \n "
            f"IOPS saved ratio (DRAMPrefetch)           - {logjson['results']['IOPSSavedRamPrefetchRatio']:.4f} \n "
            f"IOPS saved ratio (FlashPrefetch)         - {logjson['results']['IOPSSavedFlashPrefetchRatio']:.4f} \n "
            f"IOPS saved ratio (Partial)               - {logjson['results']['IOPSPartialHitsRatio']:.5f} \n "
            f"IOPS saved ratio lost from late/readmits - {logjson['results']['WastedHitsAfterEpStartIOsRatio']:.5f} \n "
            f"IOPS SR (Analysis) from Admitted Episodes - {logjson['results']['AnalysisIOPSSavedRatio']:.5f} \n "
            f"IOPS SR (Analysis, ConstantThisEA)       - {logjson['results']['AnalysisOfflineIOPSSavedRatio']:.5f} \n "
            f"GETs - {iops_requests_sofar} / {total_iops_get} \n "
            f"    Saved - {logjson['results']['TotalIOPSSaved']} \n "
            f"    Misses - {logjson['results']['ServiceFetchIOs']} \n ")
        for k, v in ods.get_all_with_prefix("iops_requests/op/"):
            msg += f"    {k} - {v} \n "
        msg += f"GET (No Cache) ST - {st_to_util(ods.get('service_time_nocache'), **util_kwargs)*100:.1f}% \n "
        for k, v in ods.get_all_with_prefix("service_time_nocache/op/"):
            msg += f"    {k} - {st_to_util(v, **util_kwargs)*100:.1f}% \n "
        msg += f"PUTs       - {ods.get('puts_ios')} \n "
        for k, v in ods.get_all_with_prefix("puts_ios/op/"):
            msg += f"    {k} - {v} \n "
        msg += f"PUT ST       - {st_to_util(ods.get('service_time_writes'), **util_kwargs)*100:.1f}% \n "
        for k, v in ods.get_all_with_prefix("service_time_writes/op/"):
            msg += f"    {k} - {st_to_util(v, **util_kwargs)*100:.1f}% \n "

        # Too verbose for release. TODO: Make this an option.
        # for kk in ["ns", "user"]:
        #     msg += f"IOs by {kk} \n "
        #     for k, v in ods.get_all_with_prefix(f"iops_requests/{kk}/"):
        #         msg += f"    {k} - {v} \n "
        #     msg += f"GET (No Cache) ST - {st_to_util(ods.get('service_time_nocache'), **util_kwargs)*100:.1f}% \n "
        #     for k, v in ods.get_all_with_prefix(f"service_time_nocache/{kk}/"):
        #         msg += f"    {k} - {st_to_util(v, **util_kwargs)*100:.1f}% \n "
        #     msg += f"PUTs       - {ods.get('puts_ios')} \n "
        #     for k, v in ods.get_all_with_prefix(f"puts_ios/{kk}/"):
        #         msg += f"    {k} - {v} \n "
        #     msg += f"PUT ST       - {st_to_util(ods.get('service_time_writes'), **util_kwargs)*100:.1f}% \n "
        #     for k, v in ods.get_all_with_prefix(f"service_time_writes/{kk}/"):
        #         msg += f"    {k} - {st_to_util(v, **util_kwargs)*100:.1f}% \n "

        msg += (
            f"Episodes admitted - {logjson['results']['TotalEpisodesAdmitted']} (Analysis: {logjson['results']['AnalysisOfflineEpisodesAdmitted']}) \n "
            f"Chunk hit ratio - {logjson['results']['ChunkHitRatio']:.5f} \n "
            f"Chunk hit ratio (DRAM)   - {logjson['results']['ChunkHitRamRatio']:.5f} \n "
            f"Chunk hit ratio (Flash) - {logjson['results']['ChunkHitFlashRatio']:.5f} \n "
            f"Chunk hit ratio (Flash Not In DRAM) - {logjson['results']['ChunkHitFlashNotInRamRatio']:.5f} \n "
            f"Flash Cache Hit Rate - {logjson['results']['FlashCacheHitRate']:.5f} \n "
            f"Client Bandwidth [GET] - {logjson['results']['ClientBandwidth']:.2f} MB/s \n "
            f"Backend Bandwidth [GET+PUT] - {logjson['results']['BackendBandwidth']:.2f} MB/s \n "
            f"Backend Bandwidth [GET] - {logjson['results']['BackendBandwidthGet']:.2f} MB/s \n "
            f"Backend Bandwidth [PUT] - {logjson['results']['BackendBandwidthPut']:.2f} MB/s \n "
            f"Chunks Queried - {chunk_queries} \n "
            f"Chunks Saved - {logjson['results']['TotalChunkSaved']} \n "
            f"Chunks Fetched from Backend - {logjson['results']['ServiceFetchChunks']} \n "
            f"   Demand [Not Prefetch] - {logjson['results']['ServiceFetchChunksDemandmiss']} \n"
            f"   Prefetch - {logjson['results']['ServiceFetchChunksPrefetch']} \n ")
        if prefetcher_enabled:
            msg += (
                f"Prefetches - {prefetches_} \n "
                f"   Good - {useful_prefetches} \n "
                f"   Wasted - {wasted_prefetches} ({logjson['results']['WastedPrefetchRatio']:.4f}) \n "
                f"       Breakdown - DRAM: {wasted_pf_ram}, Flash: {wasted_pf_flash} \n "
                f"   Failed (Exists) - {prefetches_exists} \n ")
        msg += (
            f"Flash writes - {chunks_written} \n "
            f"Acceptance Ratio - {logjson['results']['AcceptanceRatio']:.5f} \n "
            f"Flash write rate - {logjson['results']['FlashWriteRate']:.2f} MB/s \n "
            f"Writing Chunks from Admitted Episodes - {logjson['results']['AnalysisAdmittedWriteRate']:.1f} MB/s (WR Ratio: {logjson['results']['AnalysisAdmittedChunkRatio']:.2f}) \n "
            f"Analysis Closest Write Rate - {logjson['results']['AnalysisOfflineWR']:.2f} \n "
            f"Flash Wasted % of WR - {logjson['results']['WastedFlashRatio']:.2f} \n "
            f"Flash Wasted (by admit) - {flash_wasted_breakdown_admit} \n "
            f"Flash Wasted (by evict) - {flash_wasted_breakdown_evict} \n "
            f"Flash prefetch ratio - {utils.safe_div(logjson['results']['FlashPrefetchWriteRate'], logjson['results']['FlashWriteRate']):.2f} \n "
            f"Flash prefetch write rate - {logjson['results']['FlashPrefetchWriteRate']:.2f} MB/s \n "
            f"Cache Size - {logjson['results']['AnalysisOfflineCacheSize']:.2f} GB (Analysis) vs {logjson['options']['size_gb']:.2f} (Sim, {logjson['results']['NumCacheElems']} items) \n "
            f"Flash Avg Eviction Age - {logjson['results']['AvgEvictionAge']:.1f}\n "
            f"Assumed Flash EA \n "
            f"    Too Short/Bonus Hits - {logjson['results']['AssumedFlashEaTooShort']} (STSR: {logjson['results']['TooShortEaBonusServiceTimeSavedRatio']:.2f}) \n "
            f"    Too Long/Extra Writes - {logjson['results']['AssumedFlashEaTooLong']} (WRR: {logjson['results']['FlashRatioEAHitsLeft']:.2f}) \n "
            f"Flash Mean Time in System - {logjson['results']['MeanTimeInSystem']:.1f} \n "
            f"Analysis Mean Time in System - {fmt_dur(logjson['results']['AnalysisOfflineMeanTimeInSystem'], v=2)} \n ")

        if self.ram_cache:
            msg += (
                f"DRAM Cache Hit Rate - {logjson['results']['RamCacheHitRate']:.5f} \n "
                f"DRAM writes - {logjson['results']['RamChunkWritten']} \n "
                f"DRAM write rate - {logjson['results']['RamWriteRate']:.2f} MB/s \n "
                f"DRAM Prefetch write rate - {logjson['results']['RamPrefetchWriteRate']:.2f} MB/s \n "
                f"DRAM prefetches - {logjson['results']['RamCacheNumPrefetches']} \n "
                f"DRAM failed (exists) prefetches - {logjson['results']['RamCacheNumFailedPrefetchesExists']} \n "
                f"DRAM prefetch success rate - {logjson['results']['RamCachePrefetchSuccessRate']} \n "
                f"DRAM Cache Avg Eviction Age - {logjson['results']['RamCacheAvgEvictionAge']}\n ")

        msg += (
            f"Time to warmup - {warmup_time_txt} \n "
            f"Simulator RAM usage - {logjson['results']['SimRAMUsage']:.1f} GB \n "
            f"Simulator Time - {fmt_dur(wall_time)} \n ")

        word_filters = []
        if not self.ram_cache:
            word_filters.append("dram")
        if not prefetcher_enabled:
            word_filters.append("prefetch")
        msg = msg.split("\n")
        for word in word_filters:
            msg = [line for line in msg if word.lower() not in line.lower()]
        col_width = max(len(line.split(" - ")[0]) for line in msg if " - " in line)

        def alignleft(line):
            if ' - ' not in line:
                return line
            line = line.split(' - ')
            line[0] = f'{line[0]:<{col_width}}'
            return ' - '.join(line)

        msg = "\n".join(alignleft(line) for line in msg)

        if verbose:
            print(msg, file=sys.stderr)

        logjson_ = utils.stringify_keys(copy.deepcopy(logjson))
        os.makedirs(self.output_dir, 0o755, exist_ok=True)
        if dump_stats:
            # TODO: Check how long this call is taking
            statsjson_ = utils.stringify_keys(copy.deepcopy(statsjson))
            dump_logjson(statsjson_, self.filename+'.stats'+suffix, verbose=verbose)

        # Dump this last because manager will think it is complete once it sees this
        dump_logjson(logjson_, self.filename+suffix, verbose=verbose)

        if verbose:
            print("Command:")
            print(logjson['command'])

        return self.filename+suffix


def dump_logjson(json_, filename, verbose=False):
    if filename.endswith('.lzma'):
        compress_json.dump(json_, filename, json_kwargs=dict(indent=2))
    else:
        with open(filename, "w+") as out:
            json.dump(json_, out, indent=2)
    if verbose:
        print(f"Results written to {filename}", file=sys.stderr)


def simulate_cachelib(cache, accesses):
    ts = 0
    stats = {"chunk_hits": 0, "chunk_queries": 0, "rejects_clean": 0}
    for op, key in accesses:
        block_id, chunk_id = key.split("|#|body-0-")
        block_id = int(block_id)
        chunk_id = int(chunk_id)
        k = (block_id, chunk_id+1)
        acc_ts = Timestamp(physical=ts, logical=ts)
        if op == "GET":
            found = cache.find(k, acc_ts)
            if found:
                stats["chunk_hits"] += 1
            stats["chunk_queries"] += 1
        elif op == "SET":
            if k not in cache.cache:
                cache.insert(k, acc_ts, [])
            else:
                stats["rejects_clean"] += 1
        else:
            raise NotImplementedError
        ts += 1
    stats["chunk_hit_ratio"] = utils.safe_div(stats["chunk_hits"], stats["chunk_queries"])
    print(stats)
    return stats


class CacheSimulator(object):
    def __init__(self,
                 cache,
                 ram_cache=None,
                 sample_ratio=None,
                 prefetcher=None,
                 sdumper=None,
                 # admit_chunk_threshold=None,
                 # block_level=False,
                 # log_interval=None,
                 # limit=None,
                 options=None,
                 **kwargs):
        self.cache = cache
        self.insert_cache = ram_cache if ram_cache else cache
        self.ram_cache = ram_cache
        self.prefetcher = prefetcher
        assert sdumper
        self.sdumper = sdumper
        self.sample_ratio = sample_ratio
        self.options = options
        self.config = kwargs
        assert not self.config.get('block_level', False)
        self._init_logs()
        self.hooks = defaultdict(list)
        if hasattr(cache.ap, "hooks"):
            for k, v in cache.ap.hooks.items():
                self.hooks[k] += v

    def _init_logs(self):
        # TODO: Make this configurable.
        self.print_every_n_iops = 50000
        self.print_every_n_mins = 1

        if utils.DEBUG_FLAG() and '--profile' in sys.argv:
            from pympler import tracker
            self.tr = tracker.SummaryTracker()
            self.print_every_n_iops = 10000

        self.checkpoints_since_last_increase = 0

        self.header_prev = ""
        self.col_maxwidth = defaultdict(int)

        # For logging granularity, in trace time
        self.last_log_tracetime = None
        # In wallclock time, for triggering gc, touching lockfile
        self.last_syscheck = 0
        self.last_print = {'i': None, 'time': 0, 'time_frac': 0, 'io_frac': 0, 'tracetime_elapsed': 0}

        self.start_ts = None

        # TODO: Refactor
        self.last_util_peak = 0
        # self.last_util_peak_withput = 0

        # self.dumped_for_limit = False

    def _syscheck(self):
        self.last_syscheck = time.time()
        # gc.collect()
        self._touch_lockfile()

    def _log_prof_memory(self):
        if utils.DEBUG_FLAG() and '--profile' in sys.argv:
            from pympler import muppy
            self.tr.print_diff()
            all_objs = muppy.get_objects()
            leaked_objects1 = muppy.filter(all_objs, Type=list)
            leaked_objects2 = muppy.filter(all_objs, Type=tuple)
            for _ in range(20):
                o1 = random.choice(leaked_objects1)
                print("Object:", o1)
                referents = muppy.get_referents(o1)
                print("Referents: ", len(referents))
                if referents:
                    print("Choice:", random.choice(referents))
            for _ in range(20):
                o2 = random.choice(leaked_objects2)
                print("Object (tuple):", o2)
                referents = muppy.get_referents(o2)
                print("Referents (tuple): ", len(referents))
                if referents:
                    print("Choice:", random.choice(referents))

    def _checkpoint(self, acc_ts, print_log=True, save=True):
        """
        _checkpoint serves 2 purposes: period logging, and progress update.
        """
        cache = self.cache
        prefetcher = self.prefetcher
        col_maxwidth = self.col_maxwidth
        dur = (acc_ts - self.last_log_tracetime).physical

        assert cache.keys_written == cache.evictions + len(cache.cache), f"{cache.keys_written} {cache.evictions} {len(cache.cache)}"

        # TODO: Peak mitigation
        last_span = ods.last_span("time_phy", init=self.start_ts.physical)

        # First check if we have at least a few batches
        if "service_time_used_stats" in ods.batches and len(ods.batches["service_time_used_stats"]) > 10:
            # 1. Record last peak (also, print it)
            # log_items.append(("STGet%$", "{:.2f}", st_to_util(ods.span("service_time_used", i=bi), sample_ratio=self.sample_ratio, duration_s=log_dur) * 100))
            prev_peaks = [ods.get_at("service_time_used_stats", x) for x in [-4, -3, -2, -1]]
            prev_peaks_writes = [ods.get_at("service_time_writes", x) for x in [-4, -3, -2, -1]]
            prev_times = np.diff([ods.get_at("time_phy", x) for x in [-4, -3, -2, -1]])
            # print(prev_peaks)
            # print(prev_times)
            prev_peaks = [st_to_util(v, sample_ratio=self.sample_ratio, duration_s=prev_times[i]) for i, v in enumerate(np.diff(prev_peaks))] 
            prev_peaks_writes = [st_to_util(v, sample_ratio=self.sample_ratio, duration_s=prev_times[i]) for i, v in enumerate(np.diff(prev_peaks_writes))]
            prev_peaks_total = [x+y for x, y in zip(prev_peaks, prev_peaks_writes)]
            # print("Recent ST Util (%) peaks, taking max:", prev_peaks)
            # print("Prev max: ", self.last_util_peak)
            # print(f"ST Util is {stutil_get*100:.1f}%; changing AP threshold to {self.cache.ap.threshold}")
            recent_max = max(prev_peaks_total)
            self.last_util_peak = max(self.last_util_peak, recent_max)

            if self.options.peak_strategy is None:
                pass
            elif self.options.peak_strategy.startswith("zero_nonpeak"):
                if self.options.peak_strategy == "zero_nonpeak6":
                    if recent_max < 0.5 * self.last_util_peak:
                        self.cache.ap.threshold = self.options.ap_threshold * 0.1 # Don't admit at all
                    elif recent_max >= 0.8 * self.last_util_peak:
                        self.cache.ap.threshold = self.options.ap_threshold * 1.25
                    else:
                        self.cache.ap.threshold = self.options.ap_threshold
                elif self.options.peak_strategy == "zero_nonpeak5":
                    if recent_max < 0.3 * self.last_util_peak:
                        self.cache.ap.threshold = 0 # Don't admit at all
                    elif recent_max >= 0.8 * self.last_util_peak:
                        self.cache.ap.threshold = self.options.ap_threshold * 1
                    else:
                        self.cache.ap.threshold = self.options.ap_threshold
                elif self.options.peak_strategy == "zero_nonpeak4":
                    if recent_max < 0.5 * self.last_util_peak:
                        self.cache.ap.threshold = 0 # Don't admit at all
                    elif recent_max >= 0.8 * self.last_util_peak:
                        self.cache.ap.threshold = self.options.ap_threshold * 2
                    else:
                        self.cache.ap.threshold = self.options.ap_threshold
            # print(f"Changing threshold to {self.cache.ap.threshold}")
        # if last_span > 0:
        #     stutil_get = st_to_util(ods.last_span("service_time_used"), sample_ratio=self.sample_ratio, duration_s=last_span)
        #     if stutil_get < 0.03:
        #         self.cache.ap.threshold = self.options.ap_threshold / 2
        #     elif stutil_get > 0.06:
        #         self.cache.ap.threshold = self.options.ap_threshold * 2
        #     else:
        #         self.cache.ap.threshold = self.options.ap_threshold
            # print(f"ST Util is {stutil_get*100:.1f}%; changing AP threshold to {self.cache.ap.threshold}")

        ods.append("time_phy", acc_ts.physical)
        ods.append("time_elapsed_phy", (acc_ts - self.start_ts).physical)
        ods.append("time_log", acc_ts.logical)
        ods.append("duration", dur / 3600)
        ods.append("realtime_elapsed", time.time() - self.realtime_start)

        # TODO: Deprecate servicetime_orig, no longer used in this file (just in python notebooks). Replaced by service_time_nocache
        ods.append("servicetime_orig", service_time(ods.last_span("iops_requests"), ods.last_span("chunk_queries")))

        if prefetcher and prefetcher.pf_range == 'chunk2' and cache.evictions > 0:
            prefetcher.assumed_ea = cache.computeEvictionAge()

        tracetime_elapsed = acc_ts.physical - self.start_ts.physical
        time_frac = tracetime_elapsed / self.total_secs
        io_frac = ods.get('iops_requests') / self.total_iops_get
        print_log_frac = (io_frac - self.last_print['io_frac']) > .2
        print_log_frac = print_log_frac or (tracetime_elapsed - self.last_print['tracetime_elapsed']) > 3600*24

        if print_log or print_log_frac:
            bi = self.last_print['i']
            log_items = []
            est_time_left = utils.safe_div(self.total_iops_get - ods.get("iops_requests"),
                                           ods.span("iops_requests", i=bi)) * ods.span("realtime_elapsed", i=bi)
            log_items.append(("TimeLeft", fmt_dur(est_time_left, verbose=1, smallest='m')))
            log_items.append(("i  ", len(ods.get('time_phy', init=[]))))
            # log_items.append(("%TimeP", f"{time_frac*100:.1f}"))
            log_items.append(("TraceTime", fmt_dur(tracetime_elapsed, v=1, smallest='h')))
            log_dur = ods.span("time_phy", i=bi, init=self.start_ts.physical)
            log_items.append(("Hrs$", f"{log_dur / 3600:.1f}"))

            log_items.append(("%GETs", f"{io_frac*100:.1f}"))
            # $ means it was calculated on this log window (Diff-based)
            log_items.append(("GETs$", ods.span("iops_requests", i=bi)))
            log_items.append(("PUTs$", ods.span("puts_ios", i=bi)))
            # TODO: Show PeakST here

            log_items.append(("STGet%$", "{:.2f}", st_to_util(ods.span("service_time_used", i=bi), sample_ratio=self.sample_ratio, duration_s=log_dur) * 100))
            log_items.append(("PeakST%$", "{:.2f}", self.last_util_peak * 100))
            log_items.append(("STGetNoCa%$", "{:.2f}", st_to_util(ods.span("service_time_nocache", i=bi), sample_ratio=self.sample_ratio, duration_s=log_dur) * 100))
            log_items.append(("STPut%$", "{:.2f}", st_to_util(ods.span("service_time_writes", i=bi), sample_ratio=self.sample_ratio, duration_s=log_dur) * 100))
            # log_items.append(("STGet$", ods.span("service_time_used", fmt="{:.1f}", i=bi)[1]))
            # log_items.append(("STPut$", ods.span("service_time_writes", fmt="{:.1f}", i=bi)[1]))
            wr_k = {
                'ReqMBs$': 'chunk_queries',
                'GetMBs$': 'fetches_chunks',
                'PutMBs$': 'puts_chunks',
                'FlaMBs$': 'flashcache/keys_written',
                # 'EvictMBps$': 'flashcache/evictions',
                'PreMBs$': 'flashcache/prefetches',
            }
            for log_key, stat_key in wr_k.items():
                if log_key != 'PrefetchMBps$' or (prefetcher and prefetcher.enabled):
                    v = utils.mb_per_sec(ods.span(stat_key, i=bi), log_dur, self.sample_ratio)
                    log_items.append([log_key, "{:.2f}", v])

            # log_items.append(ods.last("servicetime_saved_ratio", "{:.4f}"))

            sts_ratio = 1. - utils.safe_div(ods.span("service_time_used", i=bi), ods.span("service_time_nocache", i=bi))
            log_items.append(("STSaved%$", "{:.1f}%", sts_ratio * 100))
            if prefetcher and prefetcher.enabled:
                sts_pf_ratio = utils.safe_div(ods.span("service_time_used_prefetch", i=bi), ods.span("service_time_nocache", i=bi))
                log_items.append(("STPref%$", "{:.1f}%", sts_pf_ratio * 100))
            # st_new = ods.span("service_time_nocache", i=bi) - ods.span("servicetime_saved", i=bi)
            # st_new = ods.span("service_time_used", i=bi)
            # log_items.append(("STNew$", "{:.1f}".format(st_new)))
            log_items.append(("Accept%$", "{:.2f}%", 100*utils.safe_div(
                ods.span("flashcache/keys_written", i=bi),
                ods.span("flashcache/rejections", i=bi) + ods.span("flashcache/keys_written", i=bi),
            )))
            l_ios_late = "flashcache/admits_after_ep_start_ios"
            l_chunks_late = "flashcache/admits_after_ep_start"
            st_late = service_time(ods.span(l_ios_late, i=bi), ods.span(l_chunks_late, i=bi))
            # log_items.append(("STLate$", f"{st_late:.1f}"))
            st_late_ratio = utils.safe_div(st_late, ods.span("service_time_nocache", i=bi))
            log_items.append(("STLate%$", "{:.1f}%", st_late_ratio * 100))
            # log_items.append(("IOsLate$", ods.span(l_ios_late, i=bi)))
            # log_items.append(("ChunksLate$", ods.span(l_chunks_late, i=bi)))

            # TODO: Restore.
            # for location in CACHE_LOCATIONS:
            #     ods.append(f"iops_saved{location}_ratio", utils.safe_div(
            #         stats["iops_saved" + location][stats_idx],
            #         stats["iops_requests"][stats_idx]))
            #     ods.append(f"iops_partial{location}_ratio", utils.safe_div(
            #         stats["iops_partial_hits" + location][stats_idx],
            #         stats["iops_requests"][stats_idx]))
            # for k in CACHE_LOCATIONS:
            #     if "ram" in k and self.ram_cache is None:
            #         continue
            #     if "prefetch" in k and not (prefetcher and prefetcher.enabled):
            #         continue
            #     log_items.append(ods.last(f"iops_saved{k}_ratio", "{:.5f}" if k == "" else "{:.4f}"))
            # log_items.append(ods.last("iops_partial_ratio", "{:.4f}"))

            wasted_rate_t = utils.safe_div(
                ods.span("flashcache/unaccessed_evictions", i=bi),
                ods.span("flashcache/evictions", i=bi))
            log_items.append(("Wasted%$", "{:.1f}%", 100 * wasted_rate_t))
            # wasted_rate_all = utils.safe_div(cache.un_accessed_evictions, cache.evictions)
            # log_items.append(("WastedRate", f"{wasted_rate_all:.4f}"))

            # ods.append("max_ia_max", cache.computeMaxMaxInterarrivalTime())
            # cache.max_max_interarrival_time = 0  # Hack to make it hour-specific.

            # log_items.append(("EA", f"{cache.computeEvictionAge():.1f}"))
            ea_t = utils.safe_div(ods.span("flashcache/eviction_age_cum", i=bi, init=Timestamp(0, 0)),
                                  ods.span("flashcache/evictions", i=bi))
            log_items.append(("EA$", f"{ea_t:.0f}"))

            # log_items.append(ods.last("max_ia_max", "{:.2f}"))
            # AvgIAMax is close to AvgIAMaxEvicted
            log_items.append(("AvgIAMax", f"{cache.computeAvgMaxInterarrivalTime():.0f}"))
            # log_items.append(("AvgIAMaxEvicted", f"{cache.computeAvgMaxInterarrivalTimeEvicted():.1f}"))
            # ods.append("eage_nohit", cache.computeNoHitEvictionAge())
            # ods.append("cache_avg_object_size", cache.computeAvgObjectSize())

            # log_items.append(ods.last("duration", "{:.1f}"))
            # for k in ["", "_ram", "_flash_noram", "_flash"]:
            #     if "ram" not in k or self.ram_cache is not None:
            #         log_items.append(ods.last(f"chunk_hits{k}_ratio", "{:.2f}"))
            # log_items += ods.get_all_with_prefix("flashcache/evicted_")
            log_items.append(("RAMGb", f"{utils.memory_usage():.1f}"))
            log_items.append(("EstTotalTime", fmt_dur(est_time_left + ods.span("realtime_elapsed", i=0), verbose=1, smallest='m')))
            log_items.append(("Speedup$", "{:.2f}", utils.safe_div(log_dur * self.sample_ratio / 100, ods.span("realtime_elapsed", i=bi))))
            log_items.append(("GET/Ts$", "{:.1f}", utils.safe_div(ods.span("iops_requests", i=bi), log_dur)))
            log_items.append(("GET/s$", "{:.1f}", utils.safe_div(ods.span("iops_requests", i=bi), ods.span("realtime_elapsed", i=bi))))
            if ods.get("ml_batches") > 0:
                log_items.append(("MLbatch/s$", "{:.1f}", utils.safe_div(ods.span("ml_batches", i=bi), ods.span("realtime_elapsed", i=bi))))
                log_items.append(("MLpred/s$", "{:.1f}", utils.safe_div(ods.span("ml_predictions", i=bi), ods.span("realtime_elapsed", i=bi))))

            log_items += ods.get_all_with_prefix("flashcache/admitted_after_eps_start")
            # log_items += ods.get_all_with_prefix("flashcache/admitted_chunknotinepisode")
            log_items += ods.get_all_with_prefix("flashcache/admitted_without_hits_remaining")
            log_items += ods.get_all_with_prefix("warning_")
            log_items += ods.get_all_with_prefix("flashcache/warning_")
            log_items += ods.get_all_with_prefix("ramcache/warning_")

            # print(f" | {cache_avg_object_size:.2f}")
            # Dump to file
            hdr = [row[0] for row in log_items]
            log_items_ = [row[1].format(*row[2:]) if len(row) > 2 else row[1] for row in log_items]

            def shorten(v):
                v = v.replace("_this", "$")
                v = utils.to_camelcase(v)
                return v

            hdr = [f'[{i}] {shorten(v)}' for i, v in enumerate(hdr)]
            log_items_ = [f'[{i}] {v}' for i, v in enumerate(log_items_)]
            # Padding
            for i, hd in enumerate(hdr):
                col_maxwidth[hd] = max(col_maxwidth[hd], len(log_items_[i]))
                if len(hd) < 20:
                    col_maxwidth[hd] = max(col_maxwidth[hd], len(hd))
                hdr[i] = hd.ljust(col_maxwidth[hd])
                log_items_[i] = log_items_[i].ljust(col_maxwidth[hd])
            header = " | ".join(hdr)
            log_str = " | ".join(log_items_)
        # cntr = cache.ap.seen_before
        # print(Counter(cntr.values()))

        # Problem with * is that if it's not registered, it won't appear.
        ods.checkpoint_many([
            "service_time",
            "service_time_used*",
            "service_time_nocache*",
            "service_time_writes*",
            "fetches_*",
            "puts_*",
            "iops_requests", "chunk_queries",
            "time_phy", "time_log",
            "realtime_elapsed", "ml_batches", "ml_predictions",
            "flashcache/keys_written", "flashcache/prefetches",
            "flashcache/eviction_age_cum", "flashcache/evictions",
            "flashcache/unaccessed_evictions", "flashcache/unaccessed_eviction_age_cum",
            "flashcache/admits_after_ep_start", "flashcache/admits_after_ep_start_ios"])

        if print_log or print_log_frac:
            self.last_print['i'] = len(ods.get("time_phy", init=[])) - 1
            self.last_print['io_frac'] = io_frac
            self.last_print['time_frac'] = time_frac
            self.last_print['tracetime_elapsed'] = tracetime_elapsed

            self._syscheck()

            if header != self.header_prev:
                print(header, file=sys.stderr)
                self.header_prev = header
            print(log_str, file=sys.stderr)
            if print_log:
                self.checkpoints_since_last_increase += 1
                if self.checkpoints_since_last_increase >= 5 and self.print_every_n_mins < 10:
                    self.print_every_n_mins = min(10, self.print_every_n_mins * 2)
                    print(f"(Increasing print interval to {self.print_every_n_mins} mins)")
                    self.checkpoints_since_last_increase = 0

            if self.sdumper and save:
                # Only dump stats on the first one (to check it works)
                self.sdumper.dump(None,
                                  suffix=".part.lzma",
                                  dump_stats=self.last_print['time'] == 0)

            # Put this after dump because dumping is slow
            self.last_print['time'] = time.time()

            self._syscheck()
            self._log_prof_memory()

        # if limit is not None and iops_sofar/total_iops > limit and not dumped_for_limit:
        #     dumped_for_limit = True
        #     sdumper.dump(stats, verbose=True, suffix=".limit.lzma")
        #     if False:
        #         return stats

        self.last_log_tracetime = acc_ts
        assert len(ods.get("service_time_writes_stats")) == len(ods.get("service_time_nocache_stats"))
        ods.idx = int((acc_ts - self.start_ts).physical // self.config['log_interval'])

    def _touch_lockfile(self):
        self.config['lock'].touch()

    def _stats(self, acc_ts):
        # Start time
        # if self.last_val["trace_time"] == 0:
        #     self.last_val["trace_time"] = acc_ts

        if self.last_log_tracetime is None:
            self.last_log_tracetime = acc_ts

        if self.start_ts is None:
            self.start_ts = acc_ts
            ods.counters["start_ts_phy"] = acc_ts.physical

        if time.time() - self.last_syscheck >= 60*2:
            self._syscheck()

        # stats management
        # TODO: Make boundaries more exact, to account for empty intervals.
        # Buckets should be x[timestamp / interval]++
        curr_i = int((acc_ts - self.start_ts).physical // self.config['log_interval'])
        # dur = (acc_ts - self.last_log_tracetime).physical
        # if dur > self.config['log_interval']:
        if curr_i != ods.idx:
            self._checkpoint(acc_ts, print_log=self.last_print['i'] is None or time.time() - self.last_print['time'] > self.print_every_n_mins * 60)

    def _prefetch_batch(self, batch_pf, batch):
        features = np.array([acc_.features.toList(with_size=True)
                             for acc_ in batch_pf])
        if len(features) > 0:
            predictions = self.cache.prefetcher.predict_batch(features)
            for acc_, pred in zip(batch_pf, predictions):
                acc_.pred_prefetch = pred
        return batch

    def _add_prefetch_predictions(self, acc_iterable):
        batch = []
        batch_pf = []
        for acc_ in acc_iterable:
            batch.append(acc_)
            if acc_.is_get:
                batch_pf.append(acc_)
            if len(batch_pf) >= 128:
                yield from self._prefetch_batch(batch_pf, batch)
                batch, batch_pf = [], []
        if len(batch) > 0:
            yield from self._prefetch_batch(batch_pf, batch)

    def _get_chunk_range(self, access):
        """
        TODO: Deprecate/Fix.
        Block level doesn't really work well, as it is not applied consistently enough. And what does it mean?
        """
        # iterate over each chunk of this request
        if self.config.get('block_level', False):
            # num_blocks = utils.BlkAccess.MAX_BLOCK_SIZE / utils.BlkAccess.ALIGNMENT
            num_blocks = 64
            acc_chunks = list(range(1, num_blocks+1))
        else:
            acc_chunks = access.chunks()

        # Exclusive
        chunk_range = acc_chunks[0], acc_chunks[-1]+1
        assert acc_chunks == list(range(*chunk_range))
        return acc_chunks, chunk_range

    def _log_chunk_hit(self, key, block_id, chunk_id, found, found_ramcache, hits_location, groups):
        found_locations = set()
        chunk_hit = found or found_ramcache
        if chunk_hit:
            # TODO: Deprecate.
            # self.stats["chunk_hits"][self.stats_idx] += 1
            ods.bump("chunk_hits")
            if found_ramcache:
                found_locations.add('ram')
                item = self.ram_cache.cache[key]
                if item.stats.get('prefetch', False):
                    found_locations.add('ram_prefetch')
                    if item.hits == 1:
                        found_locations.add('ram_prefetch_firsthit')
            if key in self.cache.cache:
                found_locations.add('flash')
                item = self.cache.cache[key]
                if not found_ramcache:
                    hits_location['flash_noram'] += 1
                if item.stats.get('prefetch', False):
                    found_locations.add('flash_prefetch')
                    if item.all_hits == 1 and not found_ramcache:
                        found_locations.add('flash_prefetch_firsthit')
            if key not in self.cache.cache and key in self.cache.admit_buffer:
                found_locations.add('admitbuffer')
            if key in self.cache.cache and self.cache.cache[key].group is not None:
                groups.add((block_id, self.cache.cache[key].group))
        if "--fast" not in sys.argv:
            ods.bump_counter("chunk_hits_location_dist", tuple(sorted(found_locations)))
        for k in found_locations:
            hits_location[k] += 1
        return chunk_hit

    def _log_req_hit(self, any_chunk_hit, all_chunks_hit,
                     hits_location, acc_ts, acc_chunks, block_id):
        for k, v in hits_location.items():
            # TODO: Deprecate.
            # self.stats["chunk_hits_"+k][self.stats_idx] += v
            ods.bump(["chunk_hits", k], v=v)

        self.cache.rec_episode(block_id, all_chunks_hit, any_chunk_hit, acc_ts)

        LOG_IOPS(acc_ts, block_id, all_chunks_hit, any_chunk_hit)

        if all_chunks_hit:
            # Hit.
            ods.bump("iops_saved")
            ods.bump("chunks_saved", len(acc_chunks))
            for location, v in hits_location.items():
                if v > 0:
                    ods.bump(["iops_saved", location])
                    ods.bump(["chunks_saved", location], v)

            # Ram only if there are no chunks from flash_noram
            if hits_location.get("flash_noram", 0) == 0 and hits_location.get("ram", 0) > 0:
                ods.bump(["iops_saved", "ram_only"])
            if hits_location.get("flash_noram", 0) > 0 and hits_location.get("ram", 0) == 0:
                ods.bump(["iops_saved", "flash_only"])

            if "--fast" not in sys.argv:
                ods.bump_counter("hits_location_dist", tuple(sorted(hits_location.items())))
                ods.bump_counter("hits_location", tuple(sorted(k for k, v in hits_location.items() if v > 0)))
        else:
            # Miss.
            if any_chunk_hit:
                # request-level miss, but chunk_hit
                ods.bump("iops_partial_hits")
                for location, v in hits_location.items():
                    if v > 0:
                        ods.bump(["iops_partial_hits", location])
                if "--fast" not in sys.argv:
                    filtered = sorted((k, v) for k, v in hits_location.items() if v > 0)
                    ods.bump_counter("partial_hits_location_dist", tuple(filtered))
                    ods.bump_counter("partial_hits_location", tuple(k for k, _ in filtered))

    def _get_size(self, access, misses, promotions, episode, block_id):
        cache = self.cache
        if not hasattr(cache.ap, "size_opt") or cache.ap.size_opt == 'access':
            size = access.size() / (4*1024*1024)
        elif cache.ap.size_opt == 'access_marginal':
            size = (len(misses) - len(promotions)) * utils.BlkAccess.ALIGNMENT / (4*1024*1024)
        elif cache.ap.size_opt == 'episode':
            # REQUIRES: episode
            size = episode.size / (4*1024*1024)
        elif cache.ap.size_opt == 'episode_marginal':
            # REQUIRES: episode
            size = episode.size / utils.BlkAccess.ALIGNMENT
            if block_id in cache.cached_episodes:
                size -= len(cache.cached_episodes[block_id]["active_chunks"])
            size *= utils.BlkAccess.ALIGNMENT
            size /= 4*1024*1024
        return size

    def _touch_whole_block(self, acc):
        prefetch_size = utils.BlkAccess.MAX_BLOCK_SIZE
        acc_ = utils.BlkAccess(0, prefetch_size, acc.ts.physical, block=acc.block_id)
        chks = acc_.chunks()
        for chunk_id in chks:
            k = (acc.block_id, chunk_id)
            _ = self.ram_cache and self.ram_cache.find(k, acc.ts, count_as_hit=False)
            _ = self.cache.find(k, acc.ts, count_as_hit=False)

    def _update_dynamic_features(self, acc):
        cache = self.cache
        # update dynamic features (independent of in the cache)
        if cache.dynamic_features:
            granularity = cache.dynamic_features.granularity
            if granularity.startswith('block'):
                weight = 1
                if granularity == 'block-st':
                    weight = service_time(1, len(acc.chunks))
                cache.dynamic_features.updateFeatures(acc.block_id, acc.ts.physical, weight=weight)
            elif granularity == 'chunk':
                for chunk_id in acc.chunks:
                    k = (acc.block_id, chunk_id)
                    cache.dynamic_features.updateFeatures(k, acc.ts.physical)
            elif granularity == 'both':
                cache.dynamic_features.updateFeatures(acc.block_id, acc.ts.physical)
                for chunk_id in acc.chunks:
                    k = (acc.block_id, chunk_id)
                    cache.dynamic_features.updateFeatures(k, acc.ts.physical)
            else:
                raise Exception(f"Unknown granularity: {granularity}")

    def _log_st(self, need_fetch, need_prefetch, all_chunks_hit, acc):
        acc_chunks = acc.chunks
        ods.bump("iops_requests")
        ods.bump("chunk_queries", len(acc_chunks))
        ods.bump("service_time_nocache", service_time(1, len(acc_chunks)))
        ods.bump(["iops_requests", "op", acc.features.op.name])
        ods.bump(["chunk_queries", "op", acc.features.op.name], len(acc_chunks))
        ods.bump(["service_time_nocache", "op", acc.features.op.name], service_time(1, len(acc_chunks)))

        tags = [f"ns/{acc.features.namespace}", f"user/{acc.features.user}"]
        for tag in tags:
            ods.bump(["iops_requests", tag])
            ods.bump(["chunk_queries", tag], len(acc_chunks))
            ods.bump(["service_time_nocache", tag], service_time(1, len(acc_chunks)))

        if len(need_prefetch) > 0:
            assert len(need_fetch) > 0
        if len(need_fetch) == 0:
            assert len(need_prefetch) == 0 and all_chunks_hit
        else:
            assert not all_chunks_hit
            record_service_time_get(need_fetch, need_prefetch, acc)

    def run_get(self, acc):
        cache = self.cache
        ram_cache = self.ram_cache
        insert_cache = self.insert_cache

        acc_chunks, chunk_range = acc.chunks, acc.chunk_range

        misses = []
        need_fetch = []
        promotions = []
        any_chunk_hit = False
        all_chunks_hit = True
        hits_location = defaultdict(int)
        groups = set()

        for chunk_id in acc_chunks:
            k = (acc.block_id, chunk_id)

            found_ramcache = False
            if ram_cache:
                found_ramcache = ram_cache.find(k, acc.ts)
            found = cache.find(k, acc.ts, check_only=found_ramcache)

            hit_ = self._log_chunk_hit(k, acc.block_id, chunk_id, found, found_ramcache, hits_location, groups)
            any_chunk_hit = any_chunk_hit or hit_
            all_chunks_hit = all_chunks_hit and hit_

            if not found_ramcache and ram_cache and found:
                # Promotion.
                misses.append(chunk_id)
                promotions.append(chunk_id)
                ods.bump("promotion_chunk_marked")
            elif not found and not found_ramcache:
                # Full miss
                misses.append(chunk_id)
                need_fetch.append(chunk_id)

            for hook in self.hooks['every_chunk_before_insert']:
                hook(k, acc.ts, ram_cache=ram_cache, cache=cache)

        self._log_req_hit(any_chunk_hit, all_chunks_hit, hits_location, acc.ts, acc_chunks, acc.block_id)

        episode = _lookup_episode(cache.episodes, acc.block_id, acc.ts, prune_old="--debug" in sys.argv)
        if episode is None:
            ods.bump("warning_root_ep_notfound")

        size = self._get_size(acc.acc, misses, promotions, episode, acc.block_id)

        if self.config.get('admit_chunk_threshold', None):
            # REQUIRES: episode
            misses = [chunk_id for chunk_id in misses
                      if chunk_id in episode.chunk_counts
                      and episode.chunk_counts[chunk_id] >= self.config['admit_chunk_threshold']]

        metadata_init = {'size': size, 'acc_chunk_range': chunk_range, 'episode': episode}
        if episode:
            metadata_init['at_ep_start'] = episode.ts_physical[0] == acc.ts.physical

        for hook in self.hooks['every_acc_before_insert']:
            hook(acc, ram_cache=ram_cache, cache=cache)

        for chunk_id in misses:
            k = (acc.block_id, chunk_id)
            metadata = dict(metadata_init)
            if chunk_id in promotions:
                metadata['promotion'] = 1
            # TODO: Fix this hack.
            if episode and episode.chunk_level:
                episode_chk = _lookup_episode(cache.episodes, acc.block_id, acc.ts, chunk_id=chunk_id, prune_old="--debug" in sys.argv)
                metadata['episode'] = episode_chk
            featvec = insert_cache.collect_features(k, acc)
            insert_cache.insert(k, acc.ts, featvec, metadata=metadata)

        # TODO: Make this a flag.
        insert_cache.process_admit_buffer(acc.ts)

        # TODO: Make this a flag.
        # To touch chunks from the same block and hopefully avoid readmissions
        self._touch_whole_block(acc)

        self._update_dynamic_features(acc)

        # START PREFETCHING
        # We only let prefetching happen when there is a miss.
        need_prefetch = self.prefetcher.run(
            acc, all_chunks_hit, any_chunk_hit, episode, misses, size)
        insert_cache.process_admit_buffer(acc.ts)  # TODO: Make this a flag.
        # END PREFETCHING

        self._log_st(need_fetch, need_prefetch, all_chunks_hit, acc)

        # Trigger event handler for prefetching.
        # TODO: Review and possibly deprecate.
        cache.on_access_end(acc.ts, groups=groups, access=acc.acc)

    def run_put(self, acc):
        if acc.chunk_range[0] != 0:
            ods.bump("warning_put_starts_after_zero")
        if acc.block_id in self.cache.block_counts:
            ods.bump("warning_put_notfirst")
            if self.cache.block_counts[acc.block_id] > 0:
                ods.bump("warning_put_already_in_cache")
        record_service_time_put(acc)

    def run(self, accesses, total_iops_get, total_iops, total_secs):
        self.realtime_start = time.time()
        self.total_iops = total_iops
        self.total_iops_get = total_iops_get
        self.total_secs = total_secs

        accesses = (AccessPlus(*args) for args in accesses)

        if self.cache.prefetch_range == 'acctime-episode-predict' or self.cache.prefetch_when == 'predict':
            accesses = self._add_prefetch_predictions(accesses)

        for acc in accesses:
            self._stats(acc.ts)
            try:
                if acc.is_get:
                    self.run_get(acc)
                elif acc.is_put:
                    self.run_put(acc)
                else:
                    raise NotImplementedError
            except Exception:
                traceback.print_exc()
                print(f"Access to block {acc.block_id} at TS={acc.ts}, {acc.acc}")
                raise

        self._checkpoint(acc.ts, print_log=True, save=False)


def simulate_cache(cache, accesses, sample_ratio, total_iops_get, total_iops, total_secs, *,
                   ram_cache=None,
                   **kwargs):
    csim = CacheSimulator(
        cache,
        ram_cache=ram_cache,
        sample_ratio=sample_ratio,
        **kwargs)
    csim.run(accesses, total_iops_get, total_iops, total_secs)
    return None
    # return csim.stats


def simulate_cache_driver(options):
    start_time = time.time()
    print(pprint.pformat(options.as_dict()), flush=True)
    use_lru = not (options.fifo or options.lirs)
    assert use_lru or options.lirs or options.fifo

    tracefile = options.tracefile
    if not options.output_dir:
        output_dir = "/".join(tracefile.split("/")[:-1])
    else:
        output_dir = options.output_dir

    # TODO: Remove output_suffix.
    output_dir += utils.get_output_suffix(options)
    # TODO: Output this in run_sim.sh instead
    # For convenience: may not be accurate
    pywhich = 'pypy' if 'pypy' in sys.executable else 'py'
    try:
        this_dir = os.path.dirname(os.path.realpath(__file__))
    except:
        raise ValueError
    command = f'{this_dir}/../run_py.sh {pywhich} ' + ' '.join(sys.argv)

    print(f'Command: {command}', flush=True)
    print("Output dir: {}".format(output_dir), flush=True)

    # create the output directory
    os.makedirs(output_dir, 0o755, exist_ok=True)

    input_file_name = tracefile[: -len(".trace")].split("/")[-1]
    out_prefix = f"{output_dir}/{input_file_name}"
    # TODO: Make this be an argument
    results_file = out_prefix + "_cache_perf.txt"
    lock = utils.LockFile(out_prefix + ".lock", timeout=600)

    if os.path.exists(results_file) and os.stat(results_file).st_size > 0 or (os.path.exists(results_file + ".lzma") and os.stat(results_file + ".lzma").st_size > 0):
        print(f"Results file already exists: {results_file}", flush=True)
        utils.rm_missing_ok(results_file + ".part")
        utils.rm_missing_ok(results_file + ".part.lzma")
        utils.rm_missing_ok(results_file + ".stats.part.lzma")
        utils.rm_missing_ok(lock.filename)
        if not options.ignore_existing:
            print("To rerun, add --ignore-existing")
            return
    # if os.path.exists(results_file + ".part") and os.stat(results_file + ".part").st_size > 0:
    #     print(f"Results file already exists: {results_file}.part")
    #     if not options.ignore_existing:
    #         return

    if lock.check():
        print("Lockfile has been touched recently; retry in a while", flush=True)
        print(f"Lockfile: {out_prefix}.lock", flush=True)
        time.sleep(600)
        print("Try again and see if we can take a lock")
        if not lock.check():
            print("Still can't get a lock")
            sys.exit(1)
        else:
            lock.touch()
            print("Got a lock!")

    # TODO: Use run_sim.sh with tee to log this instead
    sys.stdout = utils.CopyStream(sys.stdout, out_prefix + ".out")
    sys.stderr = utils.CopyStream(sys.stderr, out_prefix + ".err")
    print(f"Logging to {out_prefix}.out")

    # TODO: These should be read from config
    region = tracefile[: -len(".trace")].split("/")[-3]
    sample_ratio = float(input_file_name.split("_")[-1])
    sample_start = float(input_file_name.split("_")[-2])

    # output is formatted as json from the following dict
    logjson = {}
    logjson["options"] = options.as_dict()
    logjson["chunkSize"] = utils.BlkAccess.ALIGNMENT
    logjson["command"] = command

    if options.lirs:
        logjson["EvictionPolicy"] = "LIRS"
    elif options.fifo:
        logjson["EvictionPolicy"] = "FIFO"
    else:
        logjson["EvictionPolicy"] = "LRU"

    trace_kwargs = dict(region=region, sample_ratio=sample_ratio, start=sample_start, only_gets=False)

    logjson["sampleRatio"] = sample_ratio
    # TODO: Phase out sampling ratio.
    logjson["samplingRatio"] = sample_ratio
    logjson["sampleStart"] = sample_start
    logjson["trace_kwargs"] = trace_kwargs

    logjson["results"] = {}

    if options.cache_elems:
        num_cache_elems = options.cache_elems
    else:
        num_cache_elems = (
            options.size_gb * 1024 * 1024 * 1024 * sample_ratio / 100
        ) // utils.BlkAccess.ALIGNMENT
    num_cache_elems = int(num_cache_elems)

    episodes = None
    if options.offline_ap_decisions:
        print(f"Loading offline decisions {options.offline_ap_decisions}")
        if not os.path.exists(options.offline_ap_decisions):
            print("Failed to load - does not exist")
        else:
            try:
                episodes = utils.compress_load(options.offline_ap_decisions)
            except (OSError, TypeError, pickle.UnpicklingError, EOFError):
                if os.path.exists(options.offline_ap_decisions):
                    filesize = os.stat(options.offline_ap_decisions).st_size / 1048576
                    filesize = f'{filesize:g}M'
                    print("Bad file: " + filesize)
                else:
                    print("File no longer exists")
                if os.path.exists(options.offline_ap_decisions) and time.time() - os.path.getmtime(options.offline_ap_decisions) > 60*5:
                    utils.rm_missing_ok(options.offline_ap_decisions)
                raise

    ap = aps.construct(
        options.ap, options,
        sample_ratio=sample_ratio, num_cache_elems=num_cache_elems, episodes=episodes)

    logjson["AdmissionPolicy"] = ap.name

    prefetcher = prefetchers.Prefetcher(options=options)

    if options.learned_ap_granularity is None:
        options.learned_ap_granularity = 'block' if options.prefetch_when != 'never' else 'chunk'
    dfeat = dyn_features.DynamicFeatures(
        options.learned_ap_filter_count,
        granularity=options.learned_ap_granularity)

    if options.lirs:
        cache = evictp.LIRSCache(None, num_cache_elems, 1.0, ap)
    else:
        cache = evictp.QueueCache(
            None, num_cache_elems, ap,
            lru=use_lru,
            dynamic_features=dfeat,
            options=options,
            batch_size=options.batch_size,
            episodes=episodes,
            evict_by='episode' if options.evict_by_episode else 'chunk',
            prefetch_when=options.prefetch_when,
            prefetch_range=options.prefetch_range,
            prefetcher=prefetcher.model,
        )

    ram_cache = None
    if options.ram_cache:
        if options.ram_cache_elems:
            ram_cache_elems = options.ram_cache_elems
        else:
            ram_cache_elems = (
                options.ram_cache_size_gb * 1024 * 1024 * 1024 * sample_ratio / 100
            ) // utils.BlkAccess.ALIGNMENT
        ram_ap = ap if options.ram_ap_clone else aps.AcceptAll()
        ram_cache = evictp.QueueCache(
            None, ram_cache_elems, ram_ap,
            lru=True,
            dynamic_features=dfeat,
            options=options,  # TODO: Check for side-effecfts
            episodes=episodes,
            keep_metadata=True,
            on_evict=cache.handle_miss,
            namespace="ramcache")

    prefetcher.set_cache(cache=cache, ram_cache=ram_cache, insert_cache=ram_cache, ap=ap)

    if options.cachelib_trace:

        def stream_cachelib_trace(filename):
            with open(filename) as f:
                for line in f:
                    yield line.split()
        accesses = stream_cachelib_trace(options.cachelib_trace)
        simulate_cachelib(cache, accesses)
        return

    trace_stats, accesses = utils.stream_processed_accesses(tracefile, input_file_name=input_file_name, **trace_kwargs)
    print(trace_stats)

    logjson["blkCount"] = trace_stats["max_key"][0]
    logjson["totalIOPSGet"] = trace_stats["total_iops_get"]
    logjson["totalIOPS"] = logjson["totalIOPSGet"]
    logjson["totalIOPSPut"] = trace_stats["total_iops_put"]
    logjson["traceSeconds"] = trace_stats["trace_duration_secs"]
    logjson["results"]["NumCacheElems"] = num_cache_elems
    if options.ram_cache:
        logjson["results"]["NumRamCacheElems"] = ram_cache_elems

    sdumper = StatsDumper(cache, logjson, options.output_dir, results_file,
                          prefetcher=prefetcher, admission_policy=ap,
                          ram_cache=ram_cache, trace_stats=trace_stats, start_time=start_time,
                          skip_first_secs=options.stats_start)

    stats = simulate_cache(cache, accesses, sample_ratio,
                           trace_stats["total_iops_get"],
                           trace_stats["total_iops"],
                           trace_stats["trace_duration_secs"],
                           options=options,
                           ram_cache=ram_cache,
                           limit=options.limit,
                           log_interval=options.log_interval,
                           prefetcher=prefetcher,
                           sdumper=sdumper,
                           admit_chunk_threshold=options.ap_chunk_threshold,
                           block_level=options.block_level,
                           lock=lock)

    dump_stats = "--fast" not in sys.argv
    dump_stats = dump_stats or options.log_interval >= 600
    dump_stats = dump_stats or time.time() - sdumper.start_time > 3600
    sdumper.dump(stats, verbose=True, suffix=".lzma", dump_stats=dump_stats)
    utils.rm_missing_ok(results_file + ".part")
    utils.rm_missing_ok(results_file + ".part.lzma")
    utils.rm_missing_ok(results_file + ".stats.part.lzma")
    lock.delete()
    utils.rm_missing_ok(lock.filename)
    print("Complete")
