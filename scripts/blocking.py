import numpy as np
from pathlib import Path
import pandas as pd
import scipy.sparse as ss
embed_dir = Path('./2')

def weighted_relation_embedding(rels, t1, t2):
    r1 = t1['r'].unique()
    r2 = t2['r'].unique()
    r_w = np.matmul(rels, rels.T)
    r_w[np.ix_(r1, r1)] = 0
    r_w[np.ix_(r1, r1)] = 0
    r_w[np.ix_(r2, r2)] = 0
    return r_w

def signature(t, embeds, rels):
    hr = ss.csr_matrix((np.ones(len(t)), (t['h'], t['r'])), (embeds.shape[0], rels.shape[0])).toarray()
    tr = ss.csr_matrix((np.ones(len(t)), (t['t'], t['r'])), (embeds.shape[0], rels.shape[0])).toarray()
    return np.concatenate([hr, tr], 1)


try:
    from daakg.eval import ranking_l2r
except ImportError:
    def ranking_l2r(left, right, idx):
        dist = left @ right.T
        return (dist >= dist.diagonal().reshape(-1, 1)).sum(1)

# for method in ['_GCN_Align_', '_MTransE_', '_RotatE_']:
for method in ['__MTransE_']:
    # __001_MTransE_D_Y_15K_V1_20221011-2129_ins
    for task in ['D_W_15K_V1']:
        t1 = pd.read_csv(f'data/dwyv2/{task}/triples_1', sep='\t', names=['h', 'r', 't'])
        t2 = pd.read_csv(f'data/dwyv2/{task}/triples_2', sep='\t', names=['h', 'r', 't'])
        ill_ent_ids = pd.read_csv(f'data/dwyv2/{task}/ill_ent_ids', sep='\t', names=['e1', 'e2']).values
        fname = list(embed_dir.glob(method + task + '*_mapping.npy'))[-1]
        fname = str(fname)
        fname = fname.replace('mapping', 'ins')
        # if fname.find('_enh') > 0:
        #     rels = np.load(fname.replace('enh_ins', 'ins'))
        embeds = np.load(fname)
        if method == '__MTransE_':
            mapping = np.load(fname.replace('ins', 'mapping'))
            left, right = embeds[ill_ent_ids[:, 0]], embeds[ill_ent_ids[:, 1]]
            left = left @ mapping
            embeds[ill_ent_ids[:, 0]] = left
        else:
            left = embeds[ill_ent_ids[:, 0]]
            right = embeds[ill_ent_ids[:, 1]]
        left = left / np.maximum(np.linalg.norm(left, 2, 1, keepdims=True), 1e-6)
        right = right / np.maximum(np.linalg.norm(right, 2, 1, keepdims=True), 1e-6)
        rank3 = ranking_l2r(left, right, np.arange(right.shape[0]))
        rank4 = ranking_l2r(right, left, np.arange(right.shape[0]))
        fname = str(fname)
        rels = np.zeros((max(t1['r'].max(), t2['r'].max()) + 1, 100))
        rel = (signature(t1, embeds, rels).T + signature(t2, embeds, rels).T) @ embeds
        rel = rel / np.maximum(np.linalg.norm(rel, 2, 1, keepdims=True), 1e-8)
        r_w_raw = weighted_relation_embedding(rel, t1, t2)
        z = r_w_raw * (r_w_raw >= 0)
        z = r_w_raw.max(1)
        z = z - z.min()
        z = z / z.max()
        tau = 0.7
        z = z * (z >= tau) + z * 0.1 * (z < tau)
        r_w = z.reshape((-1, 1)) * rel
        left = signature(t1, embeds, rels)[ill_ent_ids[:, 0]] @ r_w
        left = left / np.maximum(np.linalg.norm(left, 2, 1, keepdims=True), 1e-6)
        right = signature(t2, embeds, rels)[ill_ent_ids[:, 1]] @ r_w
        right = right / np.maximum(np.linalg.norm(right, 2, 1, keepdims=True), 1e-6)
        rank1 = ranking_l2r(left, right, np.arange(right.shape[0]))
        rank2 = ranking_l2r(right, left, np.arange(right.shape[0]))
        for k in range(100, 1001, 100):
            print(method, task, (((rank1 <= k) | (rank2 <= k))).sum() / right.shape[0], (((rank1 <= k) | (rank2 <= k)) & ((rank3 <= 1) | (rank4 <= 1))).sum() / right.shape[0], (((rank3 <= 1) | (rank4 <= 1))).sum() / right.shape[0])
            # print(rank1.mean(), rank2.mean())
        # break