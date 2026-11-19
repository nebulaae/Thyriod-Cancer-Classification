"""
Near-duplicate image detection for leakage-safe splitting.

Exact and near-duplicate images have been reported in the public US + cytology
archive, so patients that share (near-)identical images must land in the same
fold. Each image gets a difference hash (dHash); two images are near-duplicates
when the Hamming distance of their hashes is <= max_dist. Patients connected by
any near-duplicate pair form one group; a group that mixes benign and PTC labels
is a label conflict.
"""

import numpy as np
import pandas as pd
from PIL import Image


def dhash(path, size=16):
    """Difference hash of an image as a Python int with size*size bits."""
    img = Image.open(path).convert('L').resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(''.join('1' if b else '0' for b in bits), 2)


def hamming(a, b):
    return (a ^ b).bit_count()


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


def find_duplicate_groups(df, max_dist=8, size=16):
    """Group patients that share near-duplicate images.

    df needs columns patient_id, label, us_path, patch_paths (';'-separated).
    US images are only compared with US images and patches with patches.
    Returns (groups, pairs):
      groups: np.ndarray[int] group id per row of df (rows without duplicates get their own id)
      pairs:  DataFrame of cross-patient near-duplicate pairs with their distance
    """
    df = df.reset_index(drop=True)
    records = []   # (row, modality, path, hash)
    for i, row in df.iterrows():
        records.append((i, 'us', row['us_path'], dhash(row['us_path'], size)))
        for p in row['patch_paths'].split(';'):
            records.append((i, 'cyto', p, dhash(p, size)))

    uf = _UnionFind(len(df))
    pairs = []
    for modality in ('us', 'cyto'):
        recs = [r for r in records if r[1] == modality]
        hashes = [r[3] for r in recs]
        for a in range(len(recs)):
            ra, ha = recs[a][0], hashes[a]
            for b in range(a + 1, len(recs)):
                rb = recs[b][0]
                if ra == rb:
                    continue
                d = hamming(ha, hashes[b])
                if d <= max_dist:
                    uf.union(ra, rb)
                    pairs.append({
                        'modality': modality, 'distance': d,
                        'patient_a': df.at[ra, 'patient_id'], 'label_a': int(df.at[ra, 'label']), 'path_a': recs[a][2],
                        'patient_b': df.at[rb, 'patient_id'], 'label_b': int(df.at[rb, 'label']), 'path_b': recs[b][2],
                    })

    roots = [uf.find(i) for i in range(len(df))]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    groups = np.array([remap[r] for r in roots])
    return groups, pd.DataFrame(pairs)


def label_conflict_groups(df, groups):
    """Group ids whose patients carry both labels."""
    g = pd.DataFrame({'group': groups, 'label': df['label'].values})
    n_labels = g.groupby('group')['label'].nunique()
    return set(n_labels[n_labels > 1].index)
