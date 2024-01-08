import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from ..trace_utils import wr_to_dwpd
from .styles import COLORLISTS
from .styles import COLORS
from .styles import LINESTYLES
from .styles import MARKERS

try:
    from ..constants_meta import REGIONS, REGIONS_CANON, region_labels_short, region_labels_orig, region_labels, service_time
except (ImportError, ModuleNotFoundError):
    from ..constants_public import REGIONS, REGIONS_CANON, region_labels_short, region_labels_orig, region_labels, service_time


lbm_new = {
    ('episode', 'never'): ('No prefetching', dict(color='green')),
}
lbm_range = {
    'acctime-episode': 'OPT-Range',
    'acctime-episode-predict': 'ML-Range',
    'acctime-all': 'All',
    'chunk2': 'OPT(Frac)-Chunk',
}
lbm_when = {
    'at_start': 'OPT-Ep-Start',
    'predict': 'ML-When',
    'partial': 'Partial Hit',
    'always': 'Every Miss',
    'benefit': 'OPT-Benefit',
    'rejectfirst': '2nd-Miss',
}
lbs = dict(zip(lbm_when.keys(), COLORLISTS[6]))
lb1 = dict(zip(lbm_range.keys(), LINESTYLES))
lb2 = dict(zip(lbm_range.keys(), MARKERS))
for pp, ppv in lbm_range.items():
    for qq, qqv in lbm_when.items():
        lbm_new[(pp, qq)] = (ppv + ' on ' + qqv, dict(color=lbs[qq], ls=lb1[pp], marker=lb2[pp]))

        lbm = {
            ('acctime-episode', 'benefit'): ('OPT-Range on Benefit', 'purple'),
            ('acctime-episode', 'at_start'): ('OPT-Range on At Start', 'turquoise'),
            ('acctime-episode', 'predict'): ('OPT-Range on Predict', 'green'),
            #     ('acctime-episode', 'partial'): ('OPT-Episode on Partial Hit', 'red'),
            ('acctime-episode', 'always'): ('OPT-Range on Every Miss', 'pink'),
            ('acctime-episode', 'rejectfirst'): ('OPT-Range on 2nd flash miss', 'red'),
            ('acctime-episode', 'rejectfirst-either'): ('OPT-Range on 2nd RAM/flash miss', 'blue'),

            ('acctime-all', 'partial'): ('Whole-Object on Partial Hit', 'red'),
            ('acctime-all', 'rejectfirst-either'): ('All on 2nd RAM/flash miss', 'purple'),
            ('acctime-all', 'rejectfirst'): ('All on 2nd flash miss', 'turquoise'),

            ('acctime-episode-predict', 'at_start'): ('ML-Range at Start', 'blue'),
            ('acctime-episode-predict', 'predict'): ('ML-Range on ML-PF-Predicted', 'pink'),
            ('acctime-episode-predict', 'partial'): ('ML-Range on Partial', 'purple'),
            ('acctime-episode-predict', 'always'): ('ML-Range on Every Miss', 'orange'),

            ('episode', 'never'): ('No prefetching', 'green'),
            #     ('all', 'rejectfirst'): ('All on 2nd miss (after admit)', 'blue'),
            #     ('episode', 'always'): ('Episode on 1st miss (after admit)', 'orange'),


            #     ('acctime-episode-predict', 'always'): ('Episode predicted on 1st RAM/flash miss', 'orange'),
            #     ('acctime-episode', 'always'): ('Episode on 1st RAM/flash miss', 'yellow'),
            #     ('accend-chunk', 'always'): ('Greedy chunkaware (after admit)', 'red'),
            ('chunk', 'always'): ('Greedy chunkaware on 1st RAM/flash miss', 'pink'),

        }

lbs = {k: v[0] for k, v in lbm.items()}
colormap = {v[0]: v[1] for v in lbm.values()}

sizeon = {True: 'SizeOn', False: 'SizeOff'}

lbs = {k: v[0] for k, v in lbm_new.items()}
colormap = {v[0]: v[1]['color'] for v in lbm_new.values()}
kwargsmap = {v[0]: v[1] for v in lbm_new.values()}


lbs2 = {('episode', 'never'): 'No prefetching',
        ('acctime-all', 'rejectfirst'): 'All on 2nd flash miss',
        #  ('acctime-episode', 'rejectfirst'): 'Episode on 2nd flash miss',
        #  ('acctime-episode', 'rejectfirst-either'): 'Epiosde on 2nd RAM/flash miss',
        #  ('acctime-all', 'rejectfirst-either'): 'All on 2nd RAM/flash miss',
        ('acctime-episode-predict',
         'always'): 'Episode predicted on 1st RAM/flash miss',
        ('acctime-episode', 'always'): 'Episode on 1st RAM/flash miss',
        ('chunk', 'always'): 'Greedy chunkaware on 1st RAM/flash miss'}


l_st = 'Service Time Saved Ratio'
l_ea_hr = 'Avg Eviction Age (hrs)'
# target_wr = 34
# l_pf_r, l_pf_w
# l_s

l_wr = 'Write Rate (MB/s)'
l_hr = 'IOPS Saved Ratio'
l_ea = 'Assumed Eviction Age (s)'
l_cs = 'Cache Size (GB)'
l_pf_r = 'Prefetch-Range'
l_pf_w = 'Prefetch-When'
l_s = 'SizeOn'
cols = [l_wr, l_hr, l_st] + ['AdmissionPolicy',
                             #                              'AdmitBufferSize',
                             'Prefetching', 'Region', 'AP Threshold',
                             #                        'IOPS Saved Ratio (AdmitBuffer)',
                             #                         'IOPSPartialHitsRatio',
                             #                         'IOPS Saved Ratio (Flash)',
                             #                       'IOPS Saved Ratio (RAM)',
                             'Converged', 'Policy']
l_hr_a = 'Analysis IOPS Saved Ratio'
l_stp = 'Service Time Saved (%)'
l_pstp = 'Peak Service Time Saved (%)'
# l_iop = 'IO Saved (%)'
l_iop = 'IO hit ratio (%)'
# l_stp_f = "Service Time required\n(% of no caching)"
# l_pstp_f = 'Peak Service Time\n(% of no caching)'

l_stp_f = "Service Time required (%)\n(weighted miss rate)"
l_stp_f = "Service Time required (%)"
l_pstp_f = 'Peak Service Time (%)\n(max hourly backend load)'

# l_pstp_f_short = 'Peak Service Time (%)'
l_pstp_f_short = 'Peak ST (%)'

# l_iop_f = 'IOs required\n(% of no caching)'
l_iop_f = 'IO miss rate (%)'
l_bw_f = 'Byte miss rate (%)'
l_bload = 'Backend load (%)'
l_pbload = 'Peak backend load (%)'
# servicetimeused
# used = np.diff(jx['stats']['service_time_used_stats'], prepend=0) /(jx['sampleRatio']/100)/3600/36*100
#         orig = np.array(jx['stats']['servicetime_orig']) /(jx['sampleRatio']/100)/3600/36*100

region_labels_long = region_labels
REGION_LABELS_CANON = [region_labels[r] for r in REGIONS_CANON]

policy_label_ = {
    'utility_service_time_size_fixed': ('STS', 'dashed'),
    'utility_hits_fixed': ('Hits', 'dashed'),
    'utility_service_time_density': ('STDensity', 'dashed'),
}

prefetch_hatches_ = {
    'No prefetching': None,
    'ML-Range on ML-When': '///',
    'All on Partial-Hit': '*',
    'ML-Range on Every Miss': 'o',
}

ap_label_ = {
    'OfflineAnalysis-MaxCacheSize': ('OfflineAnalysis-MaxCacheSize', 'dotted'),
    'OfflineAnalysis-MaxWR': ('OfflineAnalysis-MaxWR', 'dotted'),
    'OfflineAnalysis-MaxWROptimal': ('OfflineAnalysis-MaxWROpt', 'dotted'),
    'OfflineAnalysis': ('OfflineAnalysis', 'dotted'),
    'AndAP': ('AndAP', 'dashdot'),
    'OfflinePlus-AP': ('OPT+', 'dashed'),
    'Offline-AP': ('OPT AP', 'dashed'),
    'NewMLAP': ('Baleen', 'solid'),
    'LearnedAP': ('CompanyA-ML', 'dashdot'),
    # 'LearnedAP': ('FB-ML', 'dashdot'),
    'RejectX': ('RejectX', 'dashed'),
    'CoinFlipDet-P': ('CoinFlip', 'solid'),
    'FlashieldAP': ('Flashield', 'solid'),
    'FlashieldProbAP': ('FlashieldPr', 'solid'),
}


ap_long2short = {k: v[0] for k, v in ap_label_.items()}
ap_short2long = {v: k for k, v in ap_long2short.items()}
ap_ls = {k: v[1] for k, v in ap_label_.items()}


def add_fig_label(label, loc='top', x=.99, y_top=.98, y_bottom=.02, **kwargs):
    ax = plt.gca()
    if label in region_labels:
        label = region_labels[label]
    kwargs_ = dict(transform=ax.transAxes, va='top', ha='right', weight="normal", color="grey")
    if loc == 'bottom':
        kwargs_['va'] = 'bottom'
        y = y_bottom
    else:
        y = y_top
    ax.text(x, y, label, {**kwargs_, **kwargs})


def add_target_label(loc='top', twr=34, xoffset=8, fmt='{xlabel}: {twr}', xlabel='Target WR', top_offset=.95, bottom_offset=.05):
    import matplotlib.transforms as transforms
    # Data for X, Axis for Y
    ax = plt.gca()
    ypos = top_offset if loc == 'top' else bottom_offset
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    plt.annotate(fmt.format(xlabel=xlabel, twr=twr),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 xy=(twr, ypos),
                 xytext=(twr+xoffset, ypos),
                 va='center', xycoords=trans, ha='left')


def service_time_final(x):
    iops = x['Total IOPS'] - x['TotalIOPSSaved']
    chunks_used = x['TotalChunkQueries'] - x['TotalChunkSaved'] + x['RamCacheNumPrefetches']
    return service_time(iops, chunks_used)


def service_time_saved_ratio(x):
    service_time_saved = service_time(x['TotalIOPSSaved'], x['TotalChunkSaved'] - x['RamCacheNumPrefetches'])
    service_time_orig = service_time(x['Total IOPS'], x['TotalChunkQueries'])
    return service_time_saved / service_time_orig


def policy_f(p):
    return lambda df: df['Policy'] == p


def tcsize(t):
    return lambda df: df['Target Cache Size'] == t


def twr(t):
    return lambda df: df['Target Write Rate'] == t


def fregion(t):
    return lambda df: df['Region'] == t


def fdf(ll, t):
    return lambda df: df[ll] == t


def proc_dfs(dfs_all, all_exps):
    dfs = []
    for v in dfs_all.values():
        for region in v['Region'].unique():
            assert region in REGIONS, region
    for region in REGIONS:
        for idx in all_exps:
            if idx not in dfs_all:
                continue
            if dfs_all[idx].empty:
                continue
    #         dfx = dfx[twr(target_wr)]
            if dfs_all[idx][fregion(region)].empty:
                continue
            dfx = dfs_all[idx].loc[fregion(region)]

            if dfx['Prefetch-Range'].str.contains('chunk').all():
                dfs_all[idx].loc[fregion(region), 'AdmissionPolicy'] = 'GreedyOffline'
                if 'SizeOn' not in dfx:
                    dfs_all[idx].loc[fregion(region), 'SizeOn'] = True

            # if idx in df_sim_analysis:
            #     dfa = df_sim_analysis[idx][twr(target_wr)][fregion(region)]
            #     def lookup(row):
            #         dfa_ = dfa[lambda x: x['Assumed Eviction Age (s)'] == row['Assumed Eviction Age (s)']]
            #         target_policy = 'utility_size' if row['SizeOn'] else 'utility_normal'
            #         dfa_ = dfa_[lambda x: x['Policy'] == target_policy]
            #         if not dfa_.empty:
            #             return dfa_['IOPS Saved Ratio']
            #         return None
            #     dfx['Analysis IOPS Saved Ratio'] = dfx.apply(lookup, axis=1)

            if 'IOPSSavedAdmitBufferRatio' in dfx.columns:
                dfs_all[idx].loc[fregion(region), 'IOPS Saved Ratio (AdmitBuffer)'] = dfx['IOPSSavedAdmitBufferRatio']
            if 'IOPSSavedFlashRatio' in dfx.columns:
                dfs_all[idx].loc[fregion(region), 'IOPS Saved Ratio (Flash)'] = dfx['IOPSSavedFlashRatio']
            if 'IOPSSavedRamRatio' in dfx.columns:
                dfs_all[idx].loc[fregion(region), 'IOPS Saved Ratio (RAM)'] = dfx['IOPSSavedRamRatio']
            if 'IOPSSavedFlashPrefetchRatio' in dfx.columns:
                dfs_all[idx].loc[fregion(region), 'IOPS Saved Ratio (Flash Prefetch)'] = dfx['IOPSSavedFlashPrefetchRatio']
                dfs_all[idx].loc[fregion(region), 'IOPS Saved Ratio (RAM Prefetch)'] = dfx['IOPSSavedRamPrefetchRatio']
            # dfx['IOPS Saved Ratio'] - dfx['IOPSSavedAdmitBufferRatio']
            dfs.append(dfx)
    dfc_raw = pd.concat(dfs)
    dfc_raw.loc[:, 'Prefetching'] = dfc_raw.apply(lambda x: lbs[(x[l_pf_r], x[l_pf_w])], axis=1)
    dfc_raw.loc[:, 'AdmissionPolicy'] = dfc_raw['AdmissionPolicy'].str.replace("LearnedSizeAP", "LearnedAP")

    if 'TotalIOPSSaved' in dfc_raw.columns and 'Service Time Saved Ratio' not in dfc_raw.columns:
        dfc_raw.loc[:, 'Service Time Saved Ratio'] = dfc_raw.apply(service_time_saved_ratio, axis=1)
#     if 'Total IOPS' in dfc_raw.columns and 'Service Time' not in dfc_raw.columns:
#         dfc_raw.loc[:, 'Service Time'] = dfc_raw.apply(service_time_final, axis=1)

    dfc_raw[l_ea_hr] = dfc_raw[l_ea] / 3600

    cols_filtered = [c for c in cols if c in dfc_raw.columns]
    dfc = dfc_raw[cols_filtered]
#     cols_index = ['Region', 'AdmissionPolicy', 'SizeOn', 'Policy', 'Prefetching', 'AdmitBufferSize']
    cols_index = ['Region', 'AdmissionPolicy', 'Policy', 'Prefetching', 'AdmitBufferSize']

    dfc_ = dfc.set_index([c for c in cols_index if c in cols_filtered])
    return dfc_raw, dfc_


colors_ = list(reversed(COLORLISTS[4])) + COLORLISTS[5]
DEFAULT_COLORMAP = {
    'CoinFlip No prefetching': colors_[3],
    'CoinFlip All on Partial Hit': colors_[8],  # TODO: Deconflict
    'RejectX No prefetching': colors_[2],
    'RejectX All on Partial Hit': '#ffdf0f',  # TODO: Deconflict
    'Baleen No prefetching': colors_[1],
    'Baleen ML-Range on Every Miss': COLORS['brown'],
    'Baleen ML-Range on ML-When': colors_[0],
    'Baleen All on Partial Hit': colors_[4],  #colors_[4],  # TODO: Deconflict
    'Baleen ML-Range on Partial Hit': COLORS['purple'],  #colors_[5],  # TODO: Deconflict
    'OPT AP No prefetching': colors_[7],
    'OPT AP ML-Range on Every Miss': colors_[5],
    'OPT AP OPT-Range on OPT-Ep-Start': colors_[8],
    'CompanyA-ML No prefetching': 'pink', #colors_[5], #'dark green',
    'Flashield No prefetching': colors_[8],
    'FlashieldPr No prefetching': colors_[8],
    'Baleen-TCO': 'green',
}
DEFAULT_MARKERMAP = {
    'Baleen ML-Range on ML-When': MARKERS[0],
    'Baleen No prefetching': MARKERS[1],
    'RejectX No prefetching': MARKERS[2],
    'CoinFlip No prefetching': MARKERS[3],
    'Baleen All on Partial-Hit': MARKERS[4],
}

short_to_long = {
    'CoinFlip': 'CoinFlip No prefetching',
    'CoinFlip (All on Partial Hit)': 'CoinFlip All on Partial Hit',
    'RejectX': 'RejectX No prefetching',
    'Baleen (No Prefetch)': 'Baleen No prefetching',
    'RejectX (All on Partial Hit)': 'RejectX All on Partial Hit',
    'Baleen (All on Partial Hit)': 'Baleen All on Partial Hit',
    'Baleen (ML-Range on Partial Hit)': 'Baleen ML-Range on Partial Hit',
    'Baleen (ML-Range on Partial-Hit)': 'Baleen ML-Range on Partial Hit',
    'Baleen': 'Baleen ML-Range on ML-When',
    'Baleen (ML Prefetch)': 'Baleen ML-Range on ML-When',
    'OPT AP (No Prefetch)': 'OPT AP No prefetching',
    'OPT AP': 'OPT AP OPT-Range on OPT-Ep-Start',
    'OPT AP (OPT Prefetch)': 'OPT AP OPT-Range on OPT-Ep-Start',
    'CompanyA-ML': 'CompanyA-ML No prefetching',
    'Flashield': 'Flashield No prefetching',
    'FlashieldPr': 'FlashieldPr No prefetching',
    'Baleen-TCO': 'Baleen-TCO',
}

SHORT_COLORMAP = {k: DEFAULT_COLORMAP[v] for k, v in short_to_long.items()}
SHORT_MARKERMAP = {k: DEFAULT_MARKERMAP[v] for k, v in short_to_long.items() if v in DEFAULT_MARKERMAP}


def get_short_label(x):
    label = ap_label_[x['AdmissionPolicy']][0]
    short_pf = f" ({x['Prefetching']})"
    if x['AdmissionPolicy'] in ('NewMLAP', 'Offline-AP') and x['Prefetch-When'] == 'never':
        short_pf = ' (No Prefetch)'
    elif x['Prefetching'] == 'ML-Range on ML-When':
        short_pf = ' (ML Prefetch)'
    elif x['Prefetching'] == 'OPT-Range on OPT-Ep-Start':
        short_pf = ' (OPT Prefetch)'
    elif x['Prefetching'] == 'All on Partial Hit':
        short_pf = ' (All on Partial Hit)'
    elif x['Prefetch-When'] == 'never' and (x['AdmissionPolicy'] in ('CoinFlipDet-P', 'RejectX', 'LearnedAP') or 'Flashield' in x['AdmissionPolicy']):
        short_pf = ''
    return label + short_pf


def fill_metadata(dfc_raw_):
    dfc_raw_['RegionLabel'] = dfc_raw_['Region'].apply(lambda x: region_labels.get(x, x))
    dfc_raw_['RegionLabelOrig'] = dfc_raw_['Region'].apply(lambda x: region_labels_orig.get(x, x))
    dfc_raw_['AdmissionPolicyLabel'] = dfc_raw_['AdmissionPolicy'].apply(lambda x: ap_label_[x][0] if x in ap_label_ else x)
    dfc_raw_['Prefetching'] = dfc_raw_.apply(lambda x: lbs[(x[l_pf_r], x[l_pf_w])], axis=1)
    dfc_raw_['PlotLabel'] = dfc_raw_.apply(lambda x: ap_label_[x['AdmissionPolicy']][0] + ' ' + x['Prefetching'], axis=1)
    dfc_raw_['ShortLabel'] = dfc_raw_.apply(get_short_label, axis=1)

    dfc_raw_['Scaled Write Rate (MB/s)'] = (dfc_raw_['Write Rate (MB/s)'] / (dfc_raw_['Cache Size (GB)']/366.47461)).astype(np.float64).round(2)
    dfc_raw_['DWPD'] = wr_to_dwpd(dfc_raw_[l_wr], dfc_raw_[l_cs])
    dfc_raw_['Target DWPD'] = dfc_raw_.apply(lambda x: x['Target Write Rate'] if type(x['Target Write Rate']) == str else wr_to_dwpd(x['Target Write Rate'], x[l_cs]), axis=1)
    dfc_raw_['WRClose'] = np.isclose(dfc_raw_[l_wr], dfc_raw_['Target Write Rate'], rtol=0.01, atol=0.5)
    dfc_raw_['WRLess'] = (dfc_raw_[l_wr] < dfc_raw_['Target Write Rate']) | dfc_raw_['WRClose']
    dfc_raw_['DWPDClose'] = np.isclose(dfc_raw_['DWPD'], dfc_raw_['Target DWPD'], rtol=0.05, atol=0.5)
    dfc_raw_['DWPDLess'] = (dfc_raw_['DWPD'] < dfc_raw_['Target DWPD']) | dfc_raw_['DWPDClose']
    dfc_raw_['DWPDNotFar'] = np.isclose(dfc_raw_['DWPD'], dfc_raw_['Target DWPD'], rtol=0.2, atol=1)

    dfc_raw_['TraceGroup'] = dfc_raw_['TraceGroup'].str.replace("ws/", "")
    dfc_raw_['TraceGroupLabel'] = dfc_raw_['TraceGroup'].str.replace("ws/", "")
    dfc_raw_['Trace'] = dfc_raw_['TraceGroupLabel'] + '/' + dfc_raw_['Region']
    return dfc_raw_


def postproc(dfc_raw_):
    dfc_raw_['Target Write Rate'] = pd.to_numeric(dfc_raw_['Target Write Rate'], errors='coerce')
    dfc_raw_[l_stp] = dfc_raw_['ServiceTimeSavedRatio1'] * 100
    dfc_raw_[l_pstp] = dfc_raw_['PeakServiceTimeSavedRatio1'] * 100
    dfc_raw_[l_iop] = dfc_raw_['IOPSSavedRatio'] * 100
    dfc_raw_['IOPSMissRatio'] = 1 - dfc_raw_['IOPSSavedRatio']
    if 'BackendBandwidthGet' in dfc_raw_:
        dfc_raw_['BandwidthMissRatio'] = dfc_raw_['BackendBandwidthGet'] / dfc_raw_['ClientBandwidth']
        dfc_raw_[l_bw_f] = dfc_raw_['BandwidthMissRatio'] * 100
    # For Analysis
    dfc_raw_[l_stp] = dfc_raw_[l_stp].fillna(dfc_raw_['ServiceTimeSavedRatio'] * 100)
    dfc_raw_[l_stp] = dfc_raw_[l_stp].fillna(dfc_raw_['Service Time Saved Ratio'] * 100)
    dfc_raw_[l_stp_f] = 100 - dfc_raw_[l_stp]
    dfc_raw_[l_pstp_f] = 100 - dfc_raw_[l_pstp]
    dfc_raw_[l_pstp_f_short] = dfc_raw_[l_pstp_f]
    dfc_raw_[l_iop_f] = 100 - dfc_raw_[l_iop]
    fill_metadata(dfc_raw_)
    plot2short_ = dfc_raw_[['PlotLabel', 'ShortLabel']].drop_duplicates()
    plot2short_ = plot2short_.set_index('PlotLabel').to_dict()['ShortLabel']
    plot2short_['No caching'] = 'No caching'

    return dfc_raw_, plot2short_


def savefig(dirn='', figname=None, base_dir='figs', exts=['.pdf'], **kwargs):
    assert figname is not None
    dirn = os.path.join(base_dir, dirn)
    os.makedirs(dirn, exist_ok=True)
    for ext in exts:
        plt.savefig(os.path.join(dirn, figname+ext), bbox_inches='tight', **kwargs)


def flip_items(items, ncol=1):
    return list(itertools.chain(*[items[i::ncol] for i in range(ncol)]))


def export_legend(legend=None, h=None, l=None, flip=False, figsize=None, show=True, **kwargs):
    if legend is None:
        legend = plt.gca().get_legend()
    ax = plt.gca()
    if not h or not l:
        h, l = ax.get_legend_handles_labels()
    if legend:
        legend.remove()
    if flip:
        ncol = kwargs['ncol']
        h, l = flip_items(h, ncol), flip_items(l, ncol)
    fig2 = plt.figure(figsize=figsize if figsize else (3, .25*len(l)))
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(h, l, frameon=False, loc='upper left', **kwargs)
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if show:
        plt.show()
    return legend


def relabel_leg(mapping=None, ax=None, **kwargs):
    ax = ax or plt.gca()
    h, labels = ax.get_legend_handles_labels()
    labels = [mapping[x] for x in labels]
    return plt.legend(h, labels, **kwargs)
