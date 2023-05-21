import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import networkx as nx
import pandas as pd


def get_r_d_without_training(ent_embeds, rel_embeds, h, r, t, decoder):
    h = ent_embeds[h, :]
    r = rel_embeds[r, :]
    t = ent_embeds[t, :]
    rr = decoder.projection(h, r, t)
    d = decoder.uncertainty(h, r, t)
    return rr, d


def get_r_d_with_training(
    ent_embeds, rel_embeds, h, r, t, encoder, decoder, sample_size=10
):
    hid = h
    tid = t
    h = torch.Tensor(ent_embeds[h, :]).to(encoder.device)
    r = torch.Tensor(rel_embeds[r, :]).to(encoder.device)
    t = torch.Tensor(ent_embeds[t, :]).to(encoder.device)
    rr = torch.zeros_like(t)
    d = torch.zeros(t.shape[0])
    tail_embedding = nn.Embedding(h.shape[0], h.shape[1]).weight.to(encoder.device)

    for i in range(0, sample_size):
        nn.init.xavier_normal_(tail_embedding)
        opt = torch.optim.Adam([tail_embedding], lr=0.01)
        for _ in range(0, 50):
            opt.zero_grad()
            loss = decoder(encoder(hid, h), r, encoder(tid, tail_embedding))
            loss = loss.sum()
            loss.backward()
            opt.step()

        rr += tail_embedding
        d += torch.norm(tail_embedding - t, dim=1)

    return rr / sample_size, d / sample_size


def direct_inference_weight(
    er_graph,
    encoder,
    decoder,
    ent_embeds,
    rel_embeds,
    ent_mapping,
    sample_size=10,
    batch_size=128,
):
    Rds = np.zeros((len(er_graph), ent_embeds.shape[1]))
    Ds = np.zeros(len(er_graph))

    if sample_size == 0:
        for idx, ((h1, h2), (r1, r2), (t1, t2)) in enumerate(er_graph):
            rr1, d1 = get_r_d_without_training(
                ent_embeds, rel_embeds, h1, r1, t1, decoder
            )
            rr2, d2 = get_r_d_without_training(
                ent_embeds, rel_embeds, h2, r2, t2, decoder
            )

            Rds[idx] = np.matmul(rr1, ent_mapping) - rr2
            Ds[idx] = d1 + d2

    else:
        batch_size = 64
        loader = DataLoader(er_graph, batch_size=batch_size, shuffle=False)
        for idx, batch in enumerate(loader):
            rr1, d1 = get_r_d_with_training(
                ent_embeds,
                rel_embeds,
                batch[:, 0],
                batch[:, 1],
                batch[:, 2],
                encoder,
                decoder,
                sample_size,
            )
            rr2, d2 = get_r_d_with_training(
                ent_embeds,
                rel_embeds,
                batch[:, 3],
                batch[:, 4],
                batch[:, 5],
                encoder,
                decoder,
                sample_size,
            )

            Rds[idx] = np.matmul(rr1, ent_mapping) - rr2
            Ds[idx] = d1 + d2


def enumerate_khop_paths(er_graph, Rds, Ds, cutoff=4):
    df = pd.DataFrame(
        [
            dict(h1=h1, r1=r1, t1=t1, h2=h2, r2=r2, t2=t2, r_d=r_d, d=d)
            for (h1, h2), (r1, r2), (t1, t2), r_d, d in zip(er_graph, Rds, Ds)
        ]
    )

    g["dist"] = g["r_d"].apply(np.linalg.norm) + g["d"]
    g = g.groupby(by=["h1", "h2", "t1", "t2"])["dist"].min().reset_index()
    g["h"] = g["h1"] + "_" + g["h2"]
    g["t"] = g["t1"] + "_" + g["t2"]

    g = nx.from_pandas_edgelist(g, source="h", target="t", edge_attr=["dist"])

    D = nx.all_pairs_shortest_path_length(g, cutoff=cutoff)
    return D


def greed_sampling(
    er_graph,
    encoder,
    decoder,
    ent_embeds,
    rel_embeds,
    ent_mapping,
    probs,
    budget=10,
    sample_size=10,
    batch_size=128,
    cutoff=4,
):
    Rds, Ds = direct_inference_weight(
        er_graph,
        encoder,
        decoder,
        ent_embeds,
        rel_embeds,
        ent_mapping,
        sample_size,
        batch_size,
    )

    D = enumerate_khop_paths(er_graph, Rds, Ds, cutoff)

    best_nodes = set()
    weight = { n : 0 for n in D.keys()}

    best_weight = 0.0
    best_node = None

    for i in range(budget):
        for n in D.keys():
            if n not in best_nodes:
                new_weight = 0
                for t in D[n].keys():
                    new_weight += weight[t] * (1 - probs[t]) + 1 / (1 + D[n][t]) * probs[t]
                if new_weight > best_weight:
                    best_weight = new_weight
                    best_node = n
        if best_node is None:
            break
        best_nodes.add(best_node)
        for t in D[n].keys():
            weight[t] = weight[t] * (1 - probs[t]) + 1 / (1 + D[n][t]) * probs[t]
    return best_nodes
    
           

