import random
import numpy as np
import pandas as pd
import random
import time


def cal_neg(pos, task, triples, r_hs_dict, r_ts_dict, ids, k):
    triples = pd.DataFrame(triples, columns=['h', 'r', 't'])
    pos = pd.DataFrame(np.array(pos)[task], columns=['h', 'r', 't'])
    neg = pd.concat([pos] * k, ignore_index=True)

    idx = neg.index
    for i in range(0, 10):
        for r, g in neg.loc[idx].groupby('r'):
            choice = np.random.binomial(1, 0.5, size=len(g.index)).astype(bool)
            neg.loc[g.index[choice], 'h'] = random.choices(list(r_hs_dict[r]), k=choice.sum())
            neg.loc[g.index[~choice], 't'] = random.choices(list(r_ts_dict[r]), k=(~choice).sum())

        neg = pd.merge(neg, triples.assign(pos=True), how='left').fillna(False)
        idx = neg[neg['pos']].index
        neg = neg[['h', 'r', 't']]
    return neg.values.reshape((k, -1, 3))


def multi_cal_neg(pos, task, triples, r_hs_dict, r_ts_dict, ids, k):
    pos = np.array(pos)[task]
    neg = pos.reshape((1, -1, 3)).repeat(k, axis=0)
    choice = np.random.binomial(1, 0.5, size=(k, len(task))).astype(bool)
    for idx, tas in enumerate(task):
        (h, r, t) = pos[idx]
        for dup in range(k):
            temp_scope, num = True, 0
            while True:
                h2, r2, t2 = h, r, t
                if choice[dup, idx]:
                    if temp_scope:
                        h2 = random.sample(r_hs_dict[r], 1)[0]
                    else:
                        for id in ids:
                            if h2 in id:
                                h2 = random.sample(id, 1)[0]
                                # break
                else:
                    if temp_scope:
                        t2 = random.sample(r_ts_dict[r], 1)[0]
                    else:
                        for id in ids:
                            if t2 in id:
                                t2 = random.sample(id, 1)[0]
                                # break
                if (h2, r2, t2) not in triples:
                    break
                else:
                    num += 1
                    if num > 10:
                        temp_scope = False
            neg[dup][idx] = (h2, r2, t2)
    return neg


class TypedSampling(object):
    def __init__(self) -> None:
        self.cache = []
    
    def generate_cache_data(self, triples):
        triples = set(triples)
        r_hs_dict, r_ts_dict = {}, {}
        for (h, r, t) in triples:
            if r not in r_hs_dict:
                r_hs_dict[r] = set()
            if r not in r_ts_dict:
                r_ts_dict[r] = set()
            r_hs_dict[r].add(h)
            r_ts_dict[r].add(t)
        return triples, r_hs_dict, r_ts_dict

    def get_cached_data(self, triples):
        for old_triples, out in self.cache:
            if old_triples == triples:
                return out
        out = self.generate_cache_data(triples)
        self.cache.append((triples, out))
        return out
    
    def __call__(self, pos, triples, ills, ids, k, params):
        t_ = time.time()
        if len(pos[0]) == 2:    # triple: 1:k
            raise NotImplementedError("typed_sampling is not supported in ills sampling")
        triples, r_hs_dict, r_ts_dict = self.get_cached_data(triples)
        tasks = np.array_split(np.array(range(len(pos)), dtype=np.int32), 1)
        neg_part = multi_cal_neg(pos, tasks[0], triples, r_hs_dict, r_ts_dict, ids, k)
        return neg_part.reshape(-1, 3)


typed_sampling = TypedSampling()