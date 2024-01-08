import os
from tqdm.auto import tqdm

import compress_json
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from .. import ep_utils
from .. import episodes
from . import maps
from .maps import service_time
from .styles import COLORLISTS


def service_time_orca(ios, getbytes):
    # return ios * 11.5e-3 + getbytes / 1048576 / 143
    raise NotImplementedError


def service_time_orca_c(ios, chunks):
    return ios * 11.5e-3 + chunks * 131072 / 1048576 / 143


stonios_label = 'STOnSeeks'
stonbw_label = 'STOnDataTransfer'
stonpf_label = 'STOnDataTransfer(Prefetch)'

breakdown_plotlabels = {
    stonios_label: 'Seeks',
    stonbw_label: 'Bandwidth (Misses)',
    stonpf_label: 'Bandwidth (Prefetch)',
}

breakdown_colormap = {
    stonios_label: COLORLISTS[5][1],
    stonbw_label: COLORLISTS[5][2],
    stonpf_label: COLORLISTS[5][3],
}
# stonios_label = 'STOnIOs'
# stonbw_label = 'STOnBandwidth'
# stonpf_label = 'STOnPrefetch'


def maxstats(row, idx=None, idx_label=None):
    if os.path.exists(row['Filename'].replace(".lzma", ".stats.lzma")):
        jxs = compress_json.load(row['Filename'].replace(".lzma", ".stats.lzma"))
        sb = jxs['batches']
    else:
        jx = compress_json.load(row['Filename'])
        sb = jx['stats_batch']
    dst = np.diff(sb['service_time_used_stats'], prepend=0)
    # assert row['PeakServiceTime
    ret = {'PeakSTInterval': np.argmax(dst)}
    if idx is None:
        idx = np.argmax(dst)
        idx_label = 'Peak'
    elif idx_label is None:
        idx_label = str(idx)

    tst = dst[idx]
    if 'fetches_ios_stats' in sb:
        ios_fetched = np.diff(sb['fetches_ios_stats'], prepend=0)[idx]
        chunks_fetched = np.diff(sb['fetches_chunks_demandmiss_stats'], prepend=0)[idx]
        chunks_prefetched = np.diff(sb['fetches_chunks_prefetch_stats'], prepend=0)[idx]
        total_ios_fetched = sb['fetches_ios_stats'][-1]
        total_chunks_fetched = sb['fetches_chunks_stats'][-1]
        total_chunks_prefetched = sb['fetches_chunks_prefetch_stats'][-1]
    else:
        ios_fetched = sb['iops_requests'][idx] - sb['iops_saved'][idx]
        chunks_fetched = sb['chunk_queries'][idx] - sb['chunks_saved'][idx] + sb['chunks_prefetched'][idx]
        chunks_prefetched = sb['chunks_prefetched'][idx]
        total_ios_fetched = np.sum(sb['iops_requests']) - np.sum(sb['iops_saved'])
        total_chunks_fetched = np.sum(sb['chunk_queries']) - np.sum(sb['chunks_saved']) + np.sum(sb['chunks_prefetched'])
        total_chunks_prefetched = np.sum(sb['chunks_prefetched'])
    # st_1 = service_time(ios_fetched, chunks_fetched)
    st_pf = service_time(0, chunks_prefetched)
    st_io = service_time(ios_fetched, 0)
    # st_bw = service_time(0, chunks_fetched)
    st_bw_nopf = service_time(0, chunks_fetched - chunks_prefetched)

    # total_st_1 = service_time(total_ios_fetched, total_chunks_fetched)
    total_st_pf = service_time(0, total_chunks_prefetched)
    total_st_io = service_time(total_ios_fetched, 0)
    # total_st_bw = service_time(0, total_chunks_fetched)
    total_st_bw_nopf = service_time(0, total_chunks_fetched - total_chunks_prefetched)

    ret.update({
        f'ST@{idx_label}': tst,
        f'{stonios_label}@{idx_label}': st_io,
        f'{stonbw_label}@{idx_label}': st_bw_nopf,
        f'{stonpf_label}@{idx_label}': st_pf,
        f'IOs@{idx_label}': ios_fetched,
        f'ChunksFetched@{idx_label}': chunks_fetched,
        f'ChunksPrefetched@{idx_label}': chunks_prefetched,
    })
    ret.update({
        f'Total{stonios_label}': total_st_io,
        f'Total{stonbw_label}': total_st_bw_nopf,
        f'Total{stonpf_label}': total_st_pf,
        'IOsFetchedTotal': total_ios_fetched,
        'ChunksFetchedTotal': total_chunks_fetched,
        'PrefetchedTotal': total_chunks_prefetched,
    })
    return ret


order_ = ['No prefetch', 'ML-Range\non\nEvery Miss', 'ML-Range\non\nML-When']


def plot_breakdown(dtx, i, ax=None, x='Prefetch-When-Label', stack_order=[stonios_label, stonbw_label, stonpf_label], data_labels=False, sample_ratio=.05, xperiod=3600):
    new_cols_ = dtx.apply(maxstats, axis=1, result_type='expand', idx=i)
    dtx_ = pd.concat([dtx[[x, 'ST@Peak']], new_cols_], axis='columns').set_index(x)
    # display(dtx_)
    dtx_scaled = dtx_[[f'{stonios_label}@{i}', f'{stonbw_label}@{i}', f'{stonpf_label}@{i}', 'ST@Peak']]
    # display(dtx_scaled)
    dtx_scaled = episodes.st_to_util(dtx_scaled, sample_ratio=sample_ratio, duration_s=xperiod) * 100
    dtx_scaled[f'{stonpf_label}@{i}'] = dtx_scaled[f'{stonpf_label}@{i}'].clip(lower=0.0001)
    colors = [breakdown_colormap[l] for l in stack_order]
    dtx_scaled[[f'{l}@{i}' for l in stack_order]].plot.bar(stacked=True, ax=ax, color=colors)
    if data_labels:
        for cont in ax.containers:
            ax.bar_label(cont, fmt='%.1f', label_type='center')
        ax.bar_label(cont, fmt='%.1f')
    ax.grid(True, axis='y')
    ax.grid(False, axis='x')


def plot_breakdowns(dtx_, x='Prefetch-When-Label', order=order_,
                    subplots_order=None,
                    figsize=(6.4*1.5, 4.8),
                    sample_ratio=0.05,
                    xperiod=3600,
                    leg_kwargs={},
                    max_y=None,
                    **kwargs):
    import matplotlib.pyplot as plt
    numcols = len(dtx_)
    if order is not None:
        #         print("Sorting by", order)
        #         dtx_ = dtx_.sort_values(x, key=lambda xr: [order.index(xx) for xx in xr])
        dtx_ = dtx_.set_index(x).reindex(order).reset_index()
        numcols = len(dtx_)
    if subplots_order is not None:
        if dtx_[x].nunique() != len(dtx_):
            print("Warning: dropping {} duplicates".format(len(dtx_)-dtx_[x].nunique()))
            dtx_ = dtx_.drop_duplicates(subset=x)
        peakrows = dtx_.set_index(x).reindex(subplots_order)
        numcols = len(subplots_order)
    peaktitles = peakrows.index.values

    fig, axes = plt.subplots(ncols=numcols, figsize=figsize, sharey=True)
    if max_y is None:
        max_y = episodes.st_to_util(dtx_['ST@Peak'].max(), sample_ratio=sample_ratio, duration_s=xperiod) * 1.1 * 100

    for ii, i in enumerate(peakrows['PeakSTInterval'].values):
        i = int(i)
        ax = axes[ii]
        plot_breakdown(dtx_, i, ax=ax, x=x, sample_ratio=sample_ratio, xperiod=xperiod, **kwargs)
        if ii == 0:
            h, l = ax.get_legend_handles_labels()
            l = [ll.split('@')[0]+'@' for ll in l]
            l= [breakdown_plotlabels[ll[:-1]] for ll in l ]
            h, l = reversed(h), reversed(l)
            fig.legend(handles=h, labels=l, **leg_kwargs)  # bbox_to_anchor=bbox_to_anchor)#, loc='lower center')
            ax.get_legend().remove()
        else:
            ax.get_legend().remove()

        # ax.set_xlabel('Baleen: Prefetching')
        ax.set_ylim(0, max_y)
        # 20
        if ii > 0:
            ax.set_ylabel(None)
        else:
            ax.set_ylabel('Backend Load (%)')
        ax.set_xlabel('')
        maps.add_fig_label(f"Window={i}")
        ax.set_title(f'Peak for\n{peaktitles[ii]}')
        plt.sca(ax)
        # add_fig_label('Chlorine, 34MB/s')


def resamplet(v, xnum=3600, xperiod=600,
              t_min=None, t_max=None,
              idx='Elapsed Trace Time', col='Consumed Service Time'):
    if t_min is None:
        t_min = v[idx].min()
        t_min = t_min - t_min % xperiod
    if t_max is None:
        t_max = v[idx].max()
        t_max = t_max - t_max % xperiod
    newxs = np.arange(t_min, t_max, xperiod)
    fx = interp1d(v[idx], v[col], fill_value="extrapolate")
    newys = fx(newxs)
    # newys = np.interp(newxs, v[idx], v[col])
    return pd.Series(index=newxs/xnum, data=newys)


def resamplet_df(v, xnum=3600, xperiod=600,
                 t_min=None, t_max=None,
                 idx='Elapsed Trace Time'):
    if t_min is None:
        t_min = v[idx].min()
        t_min = t_min - t_min % xperiod
    if t_max is None:
        t_max = v[idx].max()
        t_max = t_max - t_max % xperiod
    newxs = np.arange(t_min, t_max, xperiod)
    cols = {}
    for col in v.columns:
        fx = interp1d(v[idx], v[col], fill_value="extrapolate")
        cols[col] = fx(newxs)
    # newys = np.interp(newxs, v[idx], v[col])
    return pd.DataFrame(index=newxs/xnum, data=cols)


def rejigger(data, sampling_ratio=None, xperiod=None, xnum=3600*24, **kwargs):
    news_ = resamplet(data, xnum=xnum, xperiod=xperiod, **kwargs).diff()
    news_ = episodes.st_to_util(news_, duration_s=xperiod, sample_ratio=sampling_ratio)
    #  *100 for %
    return news_ * 100


def rejigger_df(data, col='Consumed Service Time', sample_ratio=None, sampling_ratio=None, xperiod=None, xnum=3600*24, **kwargs):
    """Unlike rejigger, gives resampled ST as well in addition to Util"""
    news_ = resamplet_df(data, xnum=xnum, xperiod=xperiod, **kwargs)
    news_[f'{col} Diff'] = news_[col].diff()
    news_['Util'] = episodes.st_to_util(news_[f'{col} Diff'], duration_s=xperiod, sample_ratio=sample_ratio) * 100
    return news_


def add_percentile_stats(row,
                         pqs=[.5, .95, .99, .995, .999, .9999, .99999, 1],
                         xperiods=[60, 60*5, 60*10, 60*15, 60*30, 60*60],
                         st_fn=service_time_orca_c,
                         skip_first_secs=3600*24):
    if 'Analysis' in row['AdmissionPolicy']:
        return {}
    filename = row['Filename'].replace(".lzma", ".stats.lzma")
    jz = compress_json.load(row['Filename'])
    if os.path.exists(filename):
        try:
            jz_stats = compress_json.load(filename)
        except Exception as e:
            print(str(e))
            return {}
        if 'fetches_ios_stats' not in jz_stats['batches'] or 'fetches_chunks_stats' not in jz_stats['batches']:
            return {}
        ys = st_fn(np.array(jz_stats['batches']['fetches_ios_stats']), np.array(jz_stats['batches']['fetches_chunks_stats']))
        ys_nocache = st_fn(np.array(jz_stats['batches']['iops_requests_stats']), np.array(jz_stats['batches']['chunk_queries_stats']))
    else:
        if 'fetches_ios_stats' not in jz['stats'] or 'fetches_chunks_stats' not in jz['stats']:
            return {}
        ys = st_fn(np.array(jz['stats']['fetches_ios_stats']), np.array(jz['stats']['fetches_chunks_stats']))
        ys_nocache = st_fn(np.array(jz['stats_batch']['iops_requests']), np.array(jz['stats_batch']['chunk_queries']))
    xs = np.arange(len(ys)) * jz['options']['log_interval']
    newsim = pd.DataFrame({'Elapsed Trace Time': xs, 'Consumed Service Time': ys})

    if not os.path.exists(filename):
        ys_nocache = np.cumsum(ys_nocache)[1:]
    newsim_nocache = pd.DataFrame({'Elapsed Trace Time': xs, 'Consumed Service Time': ys_nocache})

    stats = {}
    # label_maps = {}
    for xperiod in xperiods:
        news_ = rejigger(newsim, sampling_ratio=row['SamplingRatio'], xperiod=xperiod)
        newsim_nocache_ = rejigger(newsim_nocache, sampling_ratio=row['SamplingRatio'], xperiod=xperiod)
        news_ = news_.iloc[int(skip_first_secs/xperiod):]
        newsim_nocache_ = newsim_nocache_.iloc[int(skip_first_secs/xperiod):]
        stats['MeanServiceTimeUtil'] = news_.mean()
        stats['MeanServiceTimeNoCacheUtil'] = newsim_nocache_.mean()
        stats['MeanServiceTimeUsedPercent'] = 100 * ep_utils.safe_div(stats['MeanServiceTimeUtil'], stats['MeanServiceTimeNoCacheUtil'])
        stats['MeanServiceTimeSavedPercent'] = 100. - stats['MeanServiceTimeUsedPercent']
        for pq in pqs:
            xperiod_t = f'{xperiod}s' if xperiod < 60 else f'{xperiod/60:g}m'
            # short_label = f'{pq*100:g}%@{xperiod_t}'
            long_label = f'P{pq*100:g}ServiceTime' + '{l}' + f'@{xperiod_t}'
            # label_maps[short_label] = long_label
            stats[long_label.format(l='Util')] = news_.quantile(q=pq)
            stats[long_label.format(l='NoCacheUtil')] = newsim_nocache_.quantile(q=pq)
            # Percent is same as previous Peak ST ratio?
            stats[long_label.format(l='Percent')] = 100 * ep_utils.safe_div(stats[long_label.format(l='Util')], stats[long_label.format(l='NoCacheUtil')])
            stats[long_label.format(l='SavedPercent')] = 100. - stats[long_label.format(l='Percent')]
    return stats


tqdm.pandas()


def apply_to_columns(orig_df, fn):
    # TODO: Parallelize
    # progress_apply instead of apply is part of tqdm
    new_cols_ = orig_df.progress_apply(fn, axis=1, result_type='expand')
    if set(new_cols_.keys()) & set(orig_df.columns):
        return orig_df.assign(**new_cols_)
    else:
        return pd.concat([orig_df, new_cols_], axis='columns')
