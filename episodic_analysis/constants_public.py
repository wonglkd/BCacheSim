

def service_time(ios, chunks):
    """From academic testbed (Orca)"""
    return ios * 11.5e-3 + chunks * 131072 / 1048576 / 143
    # return ios * 11.5e-3 + chunks * 128 * 0.006829108392e-3


def infer_trace_group(region, warn=True):
    if region in ('Region1', 'Region2', 'Region3', 'combined_Region1_Region3', 'combined_Region1_Region3_Region2'):
        return '201910'
    if region in ('Region4'):
        return '202110'
    elif region in ('Region5'):
        return '20230325'
    elif region in ('Region6'):
        return '20230325'
    elif region in ('Region7'):
        return '20230325'
    raise NotImplementedError(region)


def trace_has_pipeline(f):
    return "Region1" in f or "Region2" in f or "Region3" in f


REGIONS = [f'Region{i}' for i in range(1, 8)]
REGIONS_CANON = REGIONS
region_labels = {k: k for k in REGIONS}
region_labels_short = region_labels
region_labels_orig = region_labels


REGIONS_DEFAULT = ['Region1', 'Region5']
REGIONS_ALL = ['Region1', 'Region5', 'Region4', 'Region2', 'Region3', 'Region6']
