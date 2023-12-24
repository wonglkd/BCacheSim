import numpy as np


def count_feat(feat_subset):
    cnt = 0
    feat_subset = feat_subset.split('+')
    for fx in feat_subset:
        if fx in ('meta', 'block', 'chunk'):
            cnt += 6
        elif fx in ('shard'):
            cnt += 1
        elif fx == 'meta_nosize':
            cnt += 3
        elif fx == '':
            continue
        else:
            raise NotImplementedError(f"Unimplemented feature: {fx} {feat_subset}")
    return cnt


def collect_features(cache, key, acc):
    block_id, chunk_id = key
    features = cache.ap.features.split('+')
    featvec = []
    for feat_idx in features:
        if feat_idx == 'meta':
            featvec.extend(acc.features.toList(with_size=True))
        elif feat_idx == 'meta_nosize':
            featvec.extend(acc.features.toList(with_size=False))
        elif feat_idx == 'dfeat':
            featvec.extend(cache.dynamic_features.getFeature(key))
        elif feat_idx == 'block':
            assert cache.dynamic_features.granularity.startswith('block') or cache.dynamic_features.granularity == 'both'
            featvec.extend(cache.dynamic_features.getFeature(block_id))
        elif feat_idx == 'chunk':
            assert cache.dynamic_features.granularity == 'both'
            cfeat = np.zeros(cache.dynamic_features.hours, dtype=int)
            # range(1, 65):
            # TODO: Consider if this should be based on granularity.
            # For chunk, do we iterate over all chunks in block or just current access?
            for chunk_id_ in acc.chunks:
                cfeat += cache.dynamic_features.getFeature((block_id, chunk_id_))
            featvec.extend(cfeat.tolist())
        elif feat_idx == 'shard':
            featvec.append(key[0][1])
        elif feat_idx == 'chunk_ind':
            raise NotImplementedError('chunk_ind')
            # for chunk_id_ in range(1, 65):
            #     featvec += self.dynamic_features.getFeature((block_id, chunk_id_))
        elif feat_idx == '':
            pass
        else:
            raise NotImplementedError(feat_idx)
    return featvec
    # if self.dynamic_features:
    #     self.admit_buffer[key] = self.dynamic_features.getFeature(key)
    #     # self.admit_buffer[key].append(
    #     #     self.dynamic_features.getLastAccessDistance(key, ts.physical)
    #     # )
    # self.admit_buffer[key].extend(keyfeaturelist)  # .toList()
    # # insert offline features
    # if self.offline_feat_df is not None:
    #     self.admit_buffer[key].extend(self.offline_feat_df.loc[key])
