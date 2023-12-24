import logging
import math


import numpy as np
import pandas as pd

from scipy.interpolate import interp1d


def fit_and_plot_results(writespeeds, avg_eas, target_ws,
                         annot=[],
                         plot=True, plot_title='',
                         plot_yaxis='Average eviction age (s)', plot_out=''):
    f1 = interp1d(writespeeds, avg_eas, fill_value='extrapolate')
    # f2 = interp1d(writespeeds, avg_eas, kind='cubic', fill_value='extrapolate')

    if plot:
        import matplotlib.pyplot as plt
        xfit = np.linspace(min(writespeeds) - 10,
                           max(writespeeds) + 10, num=50, endpoint=True)
        plt.figure(figsize=(8, 8))
        plt.xlim((min(writespeeds)-10, max(writespeeds)+10))
        plt.plot(writespeeds, avg_eas, 'o', label='samples', ms=10)

        plt.plot(xfit, f1(xfit), '-', label='linear', lw=2)
#         plt.plot(xfit, f2(xfit), '-', label='cubic', lw=2)
        plt.axvline(target_ws, ls='--', color='red', lw=2)
        plt.title(plot_title, fontsize=20)
        if annot:
            for x, y, z in zip(writespeeds, avg_eas, annot):
                plt.annotate(z, (x, y))
        plt.tick_params(axis="both", labelsize=15)
        plt.xlabel('Write Rate (MB/s)', fontsize=15)
        plt.ylabel(plot_yaxis, fontsize=15)
        plt.xlim(0, None)
        plt.ylim(0, .6)
        plt.legend(fontsize=15)
        plt.grid(True)

        if plot_out != '':
            plt.savefig(plot_out)

    return f1([target_ws]).item()


def fit_results(xs, ys, target_x):
    f1 = interp1d(xs, ys, fill_value='extrapolate')
    result = f1([target_x]).item()
    if math.isnan(result):
        f1 = interp1d(xs, ys, kind='quadratic', fill_value='extrapolate')
        result = f1([target_x]).item()
    return result


def unique_pts(df, *, search_col='AP Threshold', target_col='Write Rate (MB/s)'):
    df = df.sort_values(search_col)
    num_unique_pts = (~df.duplicated(subset=[target_col])).sum()
    return num_unique_pts


def fit_and_get_new(df, *,
                    exp=None,
                    target_val=None, prev_config=None, threshold_now=None, damp_f=0.5,
                    split_by='avgea_s', flip=False,
                    search_col='AP Threshold',
                    search_key='threshold',
                    max_search=None,
                    target_col='Write Rate (MB/s)',
                    target_key='write_rate'):
    """
    Takes a DataFrame, sorts it by search_col (e.g., AP Threshold).
    Interpolates/extrapolates target_col (WR) at target value.
    Takes previous values (EA, STS, threshold) and moves to new fitted values with dampening.
    """
    logger = exp.logger if exp and exp.logger else logging
    assert target_val and prev_config
    df = df.sort_values(search_col)
    # EA, RAM EA, IOPS, STS
    ea_col = {
        'avgea_s': 'Avg Eviction Age (s)',
        'avgea_dist': 'Avg Eviction Age (Logical)',
        'maxmaxia_dist': 'Max Max Interarrival Time (Logical)',
    }
    other_cols = {
        'service_time_saved_ratio': 'ServiceTimeSavedRatio1',
        'analysis_admitted_wr': 'AnalysisAdmittedWriteRate',
        'hit_rate': 'IOPSSavedRatio',
        'sim_time': 'SimWallClockTime',
        'avgea_dist': ea_col['avgea_dist'],
        'maxmaxia_dist': ea_col['maxmaxia_dist'],
        'eviction_age_ram': 'RAM Cache ' + ea_col[split_by],
        'service_time_saved_ratio_old': 'ServiceTimeSavedRatio',
    }
    view_cols = [search_col, target_col, ea_col[split_by], ea_col['avgea_s']]
    view_keys = [search_key, target_key, 'eviction_age', 'eviction_age_s']
    for k, v in other_cols.items():
        if v in df and v not in view_cols:
            view_cols.append(v)
            view_keys.append(k)
    views_dct = dict(zip(view_keys, view_cols))
    view_cols_ = list(dict.fromkeys(view_cols))

    with pd.option_context("display.width", 2000):
        print("Results:\n"+str(df[view_cols_]))

    df = df.dropna(subset=[ea_col[split_by]])
    for k in ea_col.values():
        if k in df:
            df[k] = df[k].fillna(0)
    # Drop zeros
    # df = df[df[ea_col[split_by]] != 0]
    unique_pts = (~df.duplicated(subset=[target_col])).sum()
    if unique_pts < 2:
        with pd.option_context("display.width", 4000):
            print(df[view_cols])
        # if len(df) == 1 and df.iloc[0]
        raise Exception(f"Need more unique points: only {unique_pts}")
    # Drop duplicates except for first and last
    df = df[(~df.duplicated(subset=[target_col], keep='first')) | (~df.duplicated(subset=[target_col], keep='last'))]
    # df = df.drop_duplicates(subset=[target_col], keep='first' if flip else 'last')
    if unique_pts != len(df):
        # Add a little epsilon to second duplicate to make it monotonic and thus fitting easier
        df.loc[df.duplicated(subset=[target_col], keep='first'), target_col] += -1e-5 if flip else 1e-5
        with pd.option_context("display.width", 2000):
            print("After dropping:\n"+str(df[view_cols]))

    # Set to twice of max EA found
    max_ea_here = 1.5 * df[ea_col[split_by]].max()
    if max_ea_here == 0:
        logger.warning("Max EA is 0, setting to 7 days")
        max_ea_here = 7*24*3600

    df.loc[lambda x: x[ea_col[split_by]] == 0, ea_col[split_by]] = max_ea_here

    fitted = {k: fit_results(df[target_col].values, df[col].values, target_val)
              for k, col in zip(view_keys, view_cols)}

    # Closest row
    closest_row = df.iloc[(df[target_col] - target_val).abs().argsort()[:1]]
    print(f'Target Col: {target_col} = {target_val}')

    with pd.option_context("display.width", 2000):
        print('Closest Row:')
        print(closest_row[view_cols])
    display_rows = {}
    display_rows['Fitted'] = fitted.copy()
    #     print('Fitted:')
    #     print(pd.DataFrame([fitted]))
    closest_row_d = closest_row.iloc[0].to_dict()
    closest_row_dr = {k: closest_row_d[col] for k, col in views_dct.items()}
    extrapolate = False
    if fitted[search_key] <= 0:
        print(f"Extrapolating: correcting {search_key} to half between 0 and closest")
        fitted[search_key] = closest_row_dr[search_key] / 2
        extrapolate = True

    if fitted[search_key] > df[search_col].max():
        print("Extrapolating (upper range)")
        if max_search is None:
            print("Try doubling")
            fitted[search_key] = closest_row_dr[search_key] * 2
            fitted['extrapolate'] = True
        else:
            print(f"Try setting {search_key} to halfway between last and max ({max_search})")
            if closest_row_dr[search_key] < max_search:
                fitted[search_key] = (closest_row_dr[search_key] + max_search) / 2
                extrapolate = True
    # if closest is zero -- then set threshold to between left and right
    # if max_search is not None and fitted['threshold'] >= max_search:
    #     print(f"Try setting threshold to halfway between last and max ({max_search})")
    #     if closest_row_d['AP Threshold'] < max_search:
    #         fitted['threshold'] = (closest_row_d['AP Threshold'] + max_search) / 2
    #         extrapolate = True

    # with pd.option_context("display.width", 2000):
    #     if extrapolate:
    #         for idx in fitted:
    #             if idx not in (search_key, target_key):
    #                 fitted[idx] = closest_row_d[views_dct[idx]]
    #         print('Fitted (corrected):')
    #         print(pd.DataFrame([fitted]))

    #     print('Previous config:')
    #     print(pd.DataFrame([prev_config]))
    display_rows['Fitted (corrected)'] = fitted
    display_rows['Prev config'] = prev_config

    fitted_final = {}
    for k, v in fitted.items():
        fitted_final[k] = v
        prev_val = prev_config.get(k, None)
        if prev_val is not None and not math.isnan(prev_val) and not math.isinf(prev_val) and k != 'threshold':
            dampening_diff = (v - prev_config[k]) * (1 - damp_f)
            fitted_final[k] = v - dampening_diff
        if math.isnan(fitted_final[k]):
            # print(f"Dampening diff: {dampening_diff} (({v} - {prev_config[k]}) * (1 - {damp_f}))")
            print(pd.DataFrame(display_rows).T)
            raise ValueError(f"bad fitted values: {k} {v} {fitted_final[k]} {prev_val}")
        fitted_final[k] = round(fitted_final[k], 6)
        if 'eviction_age' in k:
            fitted_final[k] = max(round(fitted_final[k], 3), 1)
    if fitted_final['eviction_age_s'] < 5:
        logger.warning(f"Fitted EA ({fitted_final['eviction_age_s']}) too low, likely due to extrapolation")
        fitted_final['eviction_age'] = (600 / fitted_final['eviction_age_s']) * fitted_final['eviction_age']
        fitted_final['eviction_age_s'] = 600
        fitted_final['eviction_age'] = min(fitted_final['eviction_age'], closest_row_dr['eviction_age'])
        fitted_final['eviction_age_s'] = min(fitted_final['eviction_age_s'], closest_row_dr['eviction_age_s'])
    elif search_key != 'eviction_age' and (fitted['eviction_age'] > 1.5 * closest_row_dr['eviction_age'] or extrapolate):
        # Only do this for non-Analysis.
        logger.warning(f"Fitted EA too high\n(Prev {prev_config.get('eviction_age', None)} -(smoothing)-> {fitted_final['eviction_age']:g}. Fitted: {fitted['eviction_age']:g}) or Extrapolation: {extrapolate}. Use closest row ({closest_row_dr['eviction_age']:g}) instead.")
        fitted_final['eviction_age'] = closest_row_dr['eviction_age']
        fitted_final['eviction_age_s'] = closest_row_dr['eviction_age_s']
    # with pd.option_context("display.width", 2000):
    #     print('Fitted (Final):')
    #     print(pd.DataFrame([fitted_final]))

    display_rows['Fitted (Final)'] = fitted_final
    with pd.option_context("display.width", 2000):
        print("Display:\n"+str(pd.DataFrame(display_rows).T))

    return fitted_final, fitted_final[search_key]


def not_in(summary, find, replace=None):
    # summary could be a dict or a pd.Series
    if replace is None:
        got_replace = True
    else:
        got_replace = np.isnan(summary[replace]).tolist()
        if type(got_replace) != bool:
            got_replace = all(got_replace)
    if find not in summary:
        not_found = True
    else:
        not_found = np.isnan(summary[find]).tolist()
        if type(not_found) != bool:
            not_found = any(not_found)
    return not_found and got_replace
